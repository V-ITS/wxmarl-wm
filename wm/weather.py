"""Learned weather forecaster: the weather half of a factored world model.

Aircraft motion is known exactly (commanded turns at constant speed); what an aircraft cannot know is
the weather behind its radar picture. This model reads what radar tracking provides, the last two
pictures of every cell (associated between pictures) and the age of the latest one, and forecasts each
cell's true centre, radius, intensity and survival ``k = 0 .. H`` decision steps from now. Cells born
after the latest picture are invisible and are not forecast. A transformer over the cell tokens lets
the forecast use the motion of the other cells (they share the episode's wind).

The oracle uses it as the information level ``learned`` (``wm/oracle.py``): aircraft roll forward on
the simulator, the weather follows the forecast. ``llearned`` is the same with each planning aircraft
feeding it only the cells within its sensing range, on a model trained on such views (``local_views``). Training data is weather-only simulation (the weather
does not depend on the aircraft) on seeds disjoint from every scenario and dataset seed.

    python -m wm.weather WX-sector-s0         # train -> <out>/02_forecasters/WX-sector-s0/{ckpt.pt,train.json}
    python -m wm.weather WX-eval              # score every predictor -> <out>/06_forecast_quality/WX-eval/eval.json

``eval.json`` compares every trained forecaster with frozen radar, extrapolation, the tracker and the true state
with sampled futures, by weather family, 1, 2.5 and 5 minutes ahead: footprint overlap with the truth, the share
of the true footprint missed, centre and radius error of tracked cells (motion and shape), survival error, and
the share of the true footprint that is cells born after the picture, which no radar-based predictor can see.
Local forecasters are scored inside a random sensing disc, against frozen local radar.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch import Tensor, nn

from config import EnvCfg
from core.rng import Rng
from core.weather import FAMILIES, Cells, init_cells, severity, snapshot, step_weather
from wm import experiments
from wm.config import DiagCfg, parse_overrides
from wm.module import mlp, transformer

F_IN, F_OUT = 11, 5  # per cell token; per forecast step: dq (2), radius, intensity, survival logit
DQ_NM = 10.0  # scale of the centre displacement target


def _same(a: Cells, b: Cells, steps: float, cfg: EnvCfg) -> Tensor:
    """[B, K] bool: slot holds the same cell in ``a`` and ``b``, ``steps`` decision steps apart (tracking)."""
    return a.alive & b.alive & ((a.age - b.age - steps * cfg.dt_min).abs() < 1e-3)


def features(radar: Cells, prev: Cells, age: int, cfg: EnvCfg) -> Tensor:
    """Token features [B, K, F_IN] from the latest and previous radar pictures; ``age`` in decision steps."""
    L, w, R = cfg.domain.L, cfg.weather, cfg.radar_steps
    same = _same(radar, prev, R, cfg).float()
    a = radar.alive.float()
    x = torch.cat(
        (
            radar.q / L,
            (radar.rho / w.rho_max)[..., None],
            radar.I[..., None],
            a[..., None],
            (radar.q - prev.q) / DQ_NM * same[..., None],
            ((radar.rho - prev.rho) / w.rho_max * same)[..., None],
            ((radar.I - prev.I) * same)[..., None],
            same[..., None],
            torch.full_like(a, age / R)[..., None],
        ),
        -1,
    )
    return x * a[..., None]


def repeat_cells(c: Cells, n: int) -> Cells:
    return Cells(**{k: v.repeat_interleave(n, 0) for k, v in vars(c).items()})


def local_view(radar: Cells, prev: Cells, p: Tensor, R: float) -> tuple[Cells, Cells]:
    """The two pictures as seen from positions ``p`` [B, 2]: a cell beyond range (to its edge) is not on that picture."""

    def seen(c: Cells) -> Cells:
        near = (c.q - p.unsqueeze(1)).norm(dim=-1) - c.rho <= R
        return Cells(**{**vars(c), "alive": c.alive & near})

    return seen(radar), seen(prev)


class WeatherForecaster(nn.Module):
    def __init__(self, m, env_cfg: EnvCfg) -> None:
        super().__init__()
        self.m, self.H = m, m.horizon
        self.L, self.rho_max = env_cfg.domain.L, env_cfg.weather.rho_max
        self.embed = mlp(F_IN, m.dim)
        self.encoder = transformer(m.dim, m.layers, m.heads)
        self.head = mlp(m.dim, (m.horizon + 1) * F_OUT)

    def forward(self, x: Tensor) -> Tensor:
        """[B, K, F_IN] -> [B, K, H + 1, F_OUT] (normalized units)."""
        B, K, _ = x.shape
        return self.head(self.encoder(self.embed(x))).view(B, K, self.H + 1, F_OUT)

    @torch.no_grad()
    def forecast(self, radar: Cells, prev: Cells, age: int, cfg: EnvCfg) -> dict[str, Tensor]:
        """Forecast cells in physical units: q [B, H+1, K, 2], rho, I, alive [B, H+1, K]."""
        y = self(features(radar, prev, age, cfg)).transpose(1, 2)  # [B, H+1, K, F_OUT]
        return {
            "q": radar.q.unsqueeze(1) + y[..., :2] * DQ_NM,
            "rho": (y[..., 2] * self.rho_max).clamp(0.0, self.rho_max),
            "I": y[..., 3].clamp(0.0, 1.0),
            "alive": radar.alive.unsqueeze(1) & (y[..., 4] > 0),
        }

    @classmethod
    def load(cls, path: str | Path, env_cfg: EnvCfg, device) -> WeatherForecaster:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(ckpt["cfg"], env_cfg)
        model.load_state_dict(ckpt["state_dict"])
        return model.to(device).eval()


# ---------- data: weather-only simulation with the env's radar schedule ----------
def simulate(cfg: EnvCfg, B: int, steps: int, seed: int) -> list[tuple[Cells, Cells, Cells]]:
    """Per decision step: (truth, latest radar picture, previous picture), as ``WxEnv.reset/step`` produce them."""
    rng, R = Rng(seed, "cpu"), cfg.radar_steps
    cells = init_cells(B, cfg, rng, steps=cfg.weather.burn_in - R)
    prev = snapshot(cells)
    for _ in range(R):
        step_weather(cells, cfg, rng)
    radar = snapshot(cells)
    hist = []
    for t in range(steps):
        hist.append((snapshot(cells), radar, prev))
        step_weather(cells, cfg, rng)
        if (t + 1) % R == 0:
            prev, radar = radar, snapshot(cells)
    return hist


def _radar_target(hist: list, radar: Cells, born: int, when: int, cfg: EnvCfg) -> tuple[Tensor, ...]:
    """Centre, radius, intensity and survival of the cells of ``radar`` (picture of step ``born``) at step ``when``,
    from later radar pictures only: linear between the two around ``when``, no simulator truth."""
    R = cfg.radar_steps
    a = when - when % R
    frac = (when % R) / R
    A, B = hist[a][1], hist[a + R][1]
    in_a, in_b = _same(A, radar, a - born, cfg), _same(B, radar, a + R - born, cfg)
    both = in_a & in_b
    lerp = lambda x, y: torch.where(both, x + frac * (y - x), x)
    q = torch.where(both.unsqueeze(-1), A.q + frac * (B.q - A.q), A.q)
    return q, lerp(A.rho, B.rho), lerp(A.I, B.I), in_b if frac >= 0.5 else in_a


def dataset(cfg: EnvCfg, B: int, H: int, seed: int, views: int = 0, radar_targets: bool = False) -> dict[str, Tensor]:
    """Samples at every decision step of ``B`` weather episodes: inputs and the truth ``0 .. H`` steps ahead.

    ``views`` > 0 makes that many samples per step, each from a random point in the sector that sees only
    the cells within sensing range; the weather has no aircraft, and aircraft fly all over the sector.
    ``radar_targets`` takes the targets from later radar pictures, as recorded data would provide them.
    """
    T, R, w = cfg.domain.T, cfg.radar_steps, cfg.weather
    hist = simulate(cfg, B, T + H + (R if radar_targets else 0), seed)
    gen = torch.Generator().manual_seed(seed)
    xs, ys, ms = [], [], []
    for t in range(T):
        _, radar_t, prev_t = hist[t]
        age = t % R
        for _ in range(max(views, 1)):
            radar, prev = radar_t, prev_t
            if views:
                p = torch.rand(B, 2, generator=gen) * cfg.domain.L
                radar, prev = local_view(radar, prev, p, cfg.obs.sense_range_nm)
            xs.append(features(radar, prev, age, cfg))
            y, m = [], []
            for k in range(H + 1):
                if radar_targets:
                    q, rho, I, same = _radar_target(hist, radar, t - age, t + k, cfg)
                else:
                    c = hist[t + k][0]
                    q, rho, I, same = c.q, c.rho, c.I, _same(c, radar, age + k, cfg)
                y.append(torch.cat(((q - radar.q) / DQ_NM, (rho / w.rho_max)[..., None], I[..., None],
                                    same.float()[..., None]), -1))  # fmt: skip
                m.append(radar.alive)
            ys.append(torch.stack(y, 2))  # [B, K, H+1, 5]
            ms.append(torch.stack(m, 2))
    return {"x": torch.cat(xs), "y": torch.cat(ys), "m": torch.cat(ms)}


def loss_fn(out: Tensor, y: Tensor, m: Tensor) -> Tensor:
    """Masked regression on tracked cells that survive, survival BCE on every cell on the radar."""
    surv = (y[..., 4] > 0.5) & m
    reg = (out[..., :4] - y[..., :4]).pow(2).sum(-1)
    l_reg = (reg * surv).sum() / surv.sum().clamp_min(1)
    bce = nn.functional.binary_cross_entropy_with_logits(out[..., 4], y[..., 4], reduction="none")
    return l_reg + (bce * m).sum() / m.sum().clamp_min(1)


def train(cfg: DiagCfg, env_cfg: EnvCfg, out: Path, log) -> WeatherForecaster:
    f = cfg.forecast
    torch.manual_seed(f.seed)
    tr = dataset(env_cfg, f.train_episodes, f.horizon, f.seed, f.local_views, f.radar_targets)
    va = dataset(env_cfg, f.val_episodes, f.horizon, f.seed + 1, f.local_views, f.radar_targets)
    log(f"data: {len(tr['x'])} train / {len(va['x'])} val samples")
    model = WeatherForecaster(f, env_cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=f.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, f.epochs)
    best, hist = math.inf, []
    for ep in range(1, f.epochs + 1):
        model.train()
        perm, tot, n = torch.randperm(len(tr["x"])), 0.0, 0
        for i in range(0, len(perm), f.batch):
            idx = perm[i : i + f.batch]
            loss = loss_fn(model(tr["x"][idx]), tr["y"][idx], tr["m"][idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot, n = tot + loss.item() * len(idx), n + len(idx)
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = sum(loss_fn(model(va["x"][i : i + 4096]), va["y"][i : i + 4096], va["m"][i : i + 4096]).item()
                     * len(va["x"][i : i + 4096]) for i in range(0, len(va["x"]), 4096)) / len(va["x"])  # fmt: skip
        hist.append({"epoch": ep, "train": tot / n, "val": vl})
        log(f"epoch {ep:3d} train {tot / n:.4f} val {vl:.4f}")
        if vl < best:
            best = vl
            torch.save({"cfg": f, "state_dict": model.state_dict(), "epoch": ep}, out / "ckpt.pt")
    (out / "train.json").write_text(json.dumps({"history": hist}, indent=1))
    return WeatherForecaster.load(out / "ckpt.pt", env_cfg, "cpu")


# ---------- evaluation against the beliefs of the oracle ladder ----------
def _footprint(q: Tensor, rho: Tensor, I: Tensor, alive: Tensor, pts: Tensor) -> Tensor:
    c = Cells(q=q, rho=rho, I=I, alive=alive, **{k: None for k in ("u", "g", "g0", "age", "tau", "I_pk", "q_prev",
                                                                   "rho_prev", "drift", "fam", "n_sys", "Q", "axis", "half_len", "sys")})  # fmt: skip
    return severity(pts.unsqueeze(0).expand(q.shape[0], -1, -1), c) > 0


def _advance(c: Cells, steps: int, cfg: EnvCfg, rng: Rng) -> Cells:
    c = snapshot(c)
    for _ in range(steps):
        step_weather(c, cfg, rng)
    return c


HS = (2, 5, 10)  # decision steps ahead: 1, 2.5 and 5 minutes


def evaluate(root: Path, cfg: DiagCfg, env_cfg: EnvCfg) -> dict:
    """Mean of each score per predictor, steps ahead and family; see the module docstring."""
    from types import SimpleNamespace

    from wm.oracle import belief_cells, noiseless, prior_cfg, tracked_cells

    f, R, L, S = cfg.forecast, env_cfg.radar_steps, env_cfg.domain.L, env_cfg.obs.sense_range_nm
    g = torch.linspace(1, L - 1, 75)
    pts = torch.stack(torch.meshgrid(g, g, indexing="xy"), -1).reshape(-1, 2)  # 2 NM probe grid
    hist = simulate(env_cfg, f.eval_episodes, 60 + f.horizon, f.eval_seed)
    rng, F, prior, det = Rng(f.eval_seed + 1, "cpu"), 4, prior_cfg(env_cfg), noiseless(env_cfg)
    gen = torch.Generator().manual_seed(f.eval_seed + 2)
    models = {p.parent.name: WeatherForecaster.load(p, env_cfg, "cpu") for p in sorted((root / experiments.PACKAGES["WX"]).glob("WX-*/ckpt.pt"))}
    acc: dict = {}

    def add(name: str, h: int, fam: Tensor, key: str, values: Tensor, mask: Tensor | None = None) -> None:
        for i, family in enumerate(FAMILIES):
            sel = fam == i
            v = values[sel][mask[sel]] if mask is not None else values[sel]
            if v.numel():
                acc.setdefault(name, {}).setdefault(f"+{h * env_cfg.domain.dt:.0f}s", {}).setdefault(family, {}).setdefault(key, []).append(v.mean().item())

    def score(name: str, h: int, c: dict, truth: Tensor, tr: Cells, tracked: Tensor, disc: Tensor | None = None) -> None:
        fam = tr.fam
        n = c["q"].shape[0] // tr.q.shape[0]
        t, foot = truth.repeat_interleave(n, 0), _footprint(c["q"], c["rho"], c["I"], c["alive"], pts)
        foot = foot if disc is None else foot & disc.repeat_interleave(n, 0)
        inter, union, tot = (foot & t).sum(-1).float(), (foot | t).sum(-1).float(), t.sum(-1).float()
        fam = fam.repeat_interleave(n, 0)
        add(name, h, fam, "iou", inter / union.clamp_min(1), union > 0)
        add(name, h, fam, "miss", (t & ~foot).sum(-1).float() / tot.clamp_min(1), tot > 0)
        if disc is None:
            tk = tracked.repeat_interleave(n, 0)
            per_row = lambda e: (e * tk).sum(-1) / tk.sum(-1).clamp_min(1)
            has = tk.any(-1)
            add(name, h, fam, "centre_nm", per_row((c["q"] - tr.q.repeat_interleave(n, 0)).norm(dim=-1)), has)
            add(name, h, fam, "radius_nm", per_row((c["rho"] - tr.rho.repeat_interleave(n, 0)).abs()), has)
            add(name, h, fam, "survival_err", per_row((c["alive"] != tr.alive.repeat_interleave(n, 0)).float()), has)

    cell = lambda c: {"q": c.q, "rho": c.rho, "I": c.I, "alive": c.alive}
    at = lambda fc, h: {k: v[:, h] for k, v in fc.items()}
    for t0 in range(10, 50, 3):
        truth_now, radar, prev = hist[t0]
        age, snap = t0 % R, SimpleNamespace(radar=radar, radar_prev=prev)
        bel = belief_cells(snap, env_cfg, cfg)
        track = _advance(tracked_cells(snap, env_cfg, cfg), age, det, rng)
        p = torch.rand(f.eval_episodes, 2, generator=gen) * L
        disc = (pts.unsqueeze(0) - p.unsqueeze(1)).norm(dim=-1) <= S
        lr, lp = local_view(radar, prev, p, S)
        fcs = {k: m.forecast(lr if "local" in k else radar, lp if "local" in k else prev, age, env_cfg) for k, m in models.items()}
        for h in HS:
            tr = hist[t0 + h][0]
            truth = _footprint(tr.q, tr.rho, tr.I, tr.alive, pts)
            tracked = _same(tr, radar, age + h, env_cfg)
            old = _footprint(tr.q, tr.rho, tr.I, tracked, pts)
            add("born after the picture", h, tr.fam, "share_of_footprint", (truth & ~old).sum(-1).float() / truth.sum(-1).clamp_min(1), truth.any(-1))
            score("frozen radar", h, cell(radar), truth, tr, tracked)
            score("extrapolation", h, cell(_advance(repeat_cells(bel, F), h, prior, rng)), truth, tr, tracked)
            score("tracker", h, cell(_advance(track, h, det, rng)), truth, tr, tracked)
            score("true state", h, cell(_advance(repeat_cells(truth_now, F), h, env_cfg, rng)), truth, tr, tracked)
            score("frozen local radar", h, cell(lr), truth & disc, tr, tracked, disc)
            for k, fc in fcs.items():
                score(k, h, at(fc, h), truth & disc if "local" in k else truth, tr, tracked, disc if "local" in k else None)
    mean = lambda v: round(sum(v) / len(v), 4)
    return {name: {h: {fam: {k: mean(v) for k, v in d.items()} for fam, d in fams.items()} for h, fams in hs.items()} for name, hs in acc.items()}


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_id", choices=[k for k, v in experiments.REGISTRY.items() if v["exp"] in ("WX", "E2")])
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("overrides", nargs="*", help="a.b=value")
    args = ap.parse_intermixed_args(argv)
    exp, cfg = experiments.resolve(args.run_id, parse_overrides(args.overrides), args.mock)
    env_cfg = cfg.env_cfg()
    out = experiments.run_dir(exp, args.run_id, cfg)
    out.mkdir(parents=True, exist_ok=True)
    log = lambda s: print(s, flush=True)
    t0 = time.time()
    if args.run_id == "WX-eval":
        res = evaluate(out.parent.parent, cfg, env_cfg)
        (out / "eval.json").write_text(json.dumps({"seconds": round(time.time() - t0), "beliefs": res}, indent=1))
    else:
        train(cfg, env_cfg, out, log)
    log(f"{args.run_id}: {round(time.time() - t0)} s")
    return out


if __name__ == "__main__":
    main()
