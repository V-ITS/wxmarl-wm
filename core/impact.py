"""Route-impact screen: how the weather of a scenario affects its direct routes.

Traffic draws random numbers only at reset and the weather is the only thing that draws during a
step, so the weather recorded while every aircraft flies direct is exactly the weather any controller
meets on the same scenario. A scenario is ``WxEnv.reset(seed)`` at a given batch size (the draws
depend on the batch shape), so screen with the same seed and batch size as the evaluation.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from baselines import POLICIES
from config import EnvCfg
from core.weather import FAMILIES, step_weather
from env import WxEnv


def w_class(tracks_in_envelope: Tensor) -> Tensor:
    """W0-W3 from the number of direct tracks entering the avoidance envelope (thresholds set for N = 5)."""
    return tracks_in_envelope.long().clamp_max(3)


# Moves of the grid search (16 directions, exact lengths): paths stay within about 3 % of the straight line.
_MOVES = [(dx, dy) for dx in range(-2, 3) for dy in range(-2, 3) if math.gcd(abs(dx), abs(dy)) == 1]


def _shift(t: Tensor, dx: int, dy: int) -> Tensor:
    """``out[..., i, j] = t[..., i - dx, j - dy]``, inf where that falls off the grid."""
    out, G = torch.full_like(t, math.inf), t.shape[-1]
    out[..., max(dx, 0) : G + min(dx, 0), max(dy, 0) : G + min(dy, 0)] = t[
        ..., max(-dx, 0) : G + min(-dx, 0), max(-dy, 0) : G + min(-dy, 0)
    ]
    return out


def _earliest(t: Tensor, free: Tensor | None, deadline: Tensor) -> Tensor:
    """Earliest arrival, in decision steps, at every cell of [b, N, G, G] (``t``: 0 at the start, inf elsewhere).

    A grid cell is one step of flight. ``free`` [b, F, G*G] says whether a cell is outside the envelope
    during step k; an arrival at time tau falls in step ceil(tau) - 1. Label-correcting: relax all moves
    until nothing improves. No holding pattern is modelled.
    """
    b, N, G, _ = t.shape
    src = None if free is None else free.unsqueeze(1).expand(b, N, *free.shape[1:])
    while True:
        changed = False
        for dx, dy in _MOVES:
            cand = _shift(t, dx, dy) + math.hypot(dx, dy)
            better = (cand < t) & (cand <= deadline)
            if src is not None:
                k = (cand.ceil() - 1).clamp(0, src.shape[2] - 1).long().view(b, N, 1, G * G)
                better &= src.gather(2, k).view(b, N, G, G)
            if better.any():
                t, changed = torch.where(better, cand, t), True
        if not changed:
            return t


def reachability(
    cfg: EnvCfg, frames: list[tuple[Tensor, Tensor, Tensor]], p0: Tensor, e: Tensor, chunk: int = 128
) -> dict[str, Tensor]:
    """Earliest arrival at the exit that stays outside the avoidance envelope, from a grid search.

    ``frames[n]`` = (q, rho, alive) is the weather during decision step ``n``. The grid cell is one
    step of flight at nominal speed; turn limits are ignored. Returns per aircraft [B, N]: ``steps``
    (inf if not reached by the deadline), ``steps_free`` (same search without weather), ``detour`` =
    steps / steps_free, ``feasible`` = reached within ``timeout_factor`` x ``steps_free`` and the
    episode limit; per scenario [B]: ``coverage`` and ``envelope_coverage``, the mean share of the
    sector inside a cell and inside the envelope.
    """
    d, w = cfg.domain, cfg.weather
    res = cfg.aircraft.v_nom * d.dt / 3600.0
    G = math.ceil((d.L + 2 * d.oob_margin) / res)
    dev = p0.device
    xs = -d.oob_margin + (torch.arange(G, device=dev) + 0.5) * res
    pts = torch.stack(torch.meshgrid(xs, xs, indexing="ij"), dim=-1).view(-1, 2)  # cell (i, j) = (x_i, y_j)
    in_sector = ((pts >= 0) & (pts <= d.L)).all(-1)
    B, N = p0.shape[:2]
    out = {k: torch.full((B, N), math.inf, device=dev) for k in ("steps", "steps_free")}
    out["coverage"], out["envelope_coverage"] = torch.zeros(B, device=dev), torch.zeros(B, device=dev)
    for s in range(0, B, chunk):
        sl = slice(s, min(s + chunk, B))
        b = sl.stop - sl.start
        free = torch.empty(b, len(frames), G * G, dtype=torch.bool, device=dev)
        for n, (q, rho, alive) in enumerate(frames):
            edge = (
                (torch.cdist(pts.unsqueeze(0).expand(b, -1, -1), q[sl]) - rho[sl].unsqueeze(1))
                .masked_fill(~alive[sl].unsqueeze(1), math.inf)
                .amin(-1)
            )  # [b, G*G]
            free[:, n] = edge > w.envelope_nm
            out["coverage"][sl] += ((edge < 0) & in_sector).sum(-1) / in_sector.sum() / len(frames)
            out["envelope_coverage"][sl] += (~free[:, n] & in_sector).sum(-1) / in_sector.sum() / len(frames)
        t0 = torch.full((b, N, G, G), math.inf, device=dev)
        i0 = ((p0[sl] + d.oob_margin) / res).long().clamp(0, G - 1)
        t0[torch.arange(b).unsqueeze(1), torch.arange(N), i0[..., 0], i0[..., 1]] = 0.0
        goal = ((pts.view(1, 1, -1, 2) - e[sl].unsqueeze(2)).norm(dim=-1) <= cfg.traffic.r_exit).view(b, N, G, G)
        off_goal = ~goal
        t_free = _earliest(t0, None, torch.tensor(math.inf))
        out["steps_free"][sl] = t_free.masked_fill(off_goal, math.inf).flatten(2).amin(-1)
        deadline = torch.clamp(cfg.traffic.timeout_factor * out["steps_free"][sl], max=float(d.T))
        t_wx = _earliest(t0, free, deadline.view(b, N, 1, 1))
        out["steps"][sl] = t_wx.masked_fill(off_goal, math.inf).flatten(2).amin(-1)
    out["feasible"] = out["steps"].isfinite()
    out["detour"] = out["steps"] / out["steps_free"]
    return out


def screen(cfg: EnvCfg, seed: int, batch_size: int, device: str = "cpu") -> dict[str, Tensor]:
    """Fly every aircraft direct on the scenarios of ``WxEnv.reset(seed)`` and describe each one.

    Returns per scenario [B]: ``family`` (-1 for scattered), ``n_systems``, ``tracks_wx`` and
    ``tracks_envelope`` (direct tracks entering a cell or the envelope), ``w_class``, ``infeasible``
    (aircraft without an envelope-free path in time), ``wind_kt``, ``cells`` (mean alive), the
    direct run's ``los_events``, and the coverages of ``reachability``; per aircraft [B, N]:
    ``detour`` and ``feasible``.
    """
    env = WxEnv(cfg, batch_size, device, seed)
    *_, avail = env.reset()
    p0, e, fam, n_sys = env.ac.p0.clone(), env.ac.e.clone(), env.cells.fam.clone(), env.cells.n_sys.clone()
    wind = env.cells.drift.norm(dim=-1)
    frames: list[tuple[Tensor, Tensor, Tensor]] = []
    info: dict = {}
    while "episode" not in info:
        frames.append((env.cells.q.clone(), env.cells.rho.clone(), env.cells.alive.clone()))
        *_, avail, info = env.step(POLICIES["direct"](env, avail))
    while len(frames) < cfg.domain.T:  # the weather a longer episode would meet: same draws as the env's
        frames.append((env.cells.q.clone(), env.cells.rho.clone(), env.cells.alive.clone()))
        step_weather(env.cells, cfg, env.rng)
    ep = info["episode"]
    r = reachability(cfg, frames, p0, e)
    return {
        "family": fam,
        "n_systems": n_sys,
        "tracks_wx": ep["flights_wx"],
        "tracks_envelope": ep["flights_envelope"],
        "w_class": w_class(ep["flights_envelope"]),
        "infeasible": (~r["feasible"]).sum(1),
        "wind_kt": wind,
        "cells": torch.stack([f[2].sum(1).float() for f in frames]).mean(0),
        "los_events": ep["los_events"],
        "coverage": r["coverage"],
        "envelope_coverage": r["envelope_coverage"],
        "detour": r["detour"],
        "feasible": r["feasible"],
    }


def class_table(sc: dict[str, Tensor]) -> dict[str, dict[str, float]]:
    """Per W-class (and all) of ``screen`` rows: scenarios, direct tracks in the envelope and hit rate
    (their share of the N tracks), detour of feasible aircraft, infeasible share, coverage, wind, alive
    cells, family counts; p10 / p50 / p90 over scenarios for the quantities that vary between them."""

    def q(x: Tensor, p: float) -> float:
        return x.float().quantile(p).item() if x.numel() else float("nan")

    rows = {}
    for name, m in [(f"W{k}", sc["w_class"] == k) for k in range(4)] + [("all", sc["w_class"] >= 0)]:
        feas = sc["feasible"][m]
        det = sc["detour"][m][feas]
        row = {
            "scenarios": int(m.sum()),
            "tracks_envelope": sc["tracks_envelope"][m].mean().item(),
            "detour_mean": det.mean().item() if det.numel() else float("nan"),
            "detour_p90": q(det, 0.9),
            "detour_max": q(det, 1.0),
            "infeasible_share": 1.0 - feas.float().mean().item() if feas.numel() else float("nan"),
            "coverage": sc["coverage"][m].mean().item(),
            "envelope_coverage": sc["envelope_coverage"][m].mean().item(),
            "hit_rate": (sc["tracks_envelope"][m] / sc["detour"].shape[1]).mean().item(),
        }
        for key, x in [
            ("hit_rate", sc["tracks_envelope"][m] / sc["detour"].shape[1]),
            ("coverage", sc["coverage"][m]),
            ("wind_kt", sc["wind_kt"][m]),
            ("cells", sc["cells"][m]),
        ]:
            row |= {f"{key}_p{p}": q(x, p / 100) for p in (10, 50, 90)}
        rows[name] = row | {fam: int((m & (sc["family"] == f)).sum()) for f, fam in enumerate(FAMILIES)}
    return rows
