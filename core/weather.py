"""Weather systems of stochastic convective cells, and severity queries."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import torch
from torch import Tensor

from config import EnvCfg
from core.geometry import pairwise_dist
from core.rng import Rng

FAMILIES = ("isolated", "clustered", "banded")  # weather-system families; "scattered" (births anywhere, independent cells) is the beliefs' prior
_BANDED = FAMILIES.index("banded")


@dataclass
class Cells:
    """Per-slot cell state, [B, K] or [B, K, 2], plus episode-level fields ([B], [B, 2]) and systems ([B, J, ...])."""

    q: Tensor
    rho: Tensor
    I: Tensor
    u: Tensor  # kt
    g: Tensor  # NM/min
    g0: Tensor
    age: Tensor  # min
    tau: Tensor  # min
    I_pk: Tensor
    alive: Tensor  # bool
    q_prev: Tensor  # centre and radius one decision step earlier
    rho_prev: Tensor
    drift: Tensor  # episode-level mean velocity, kt
    fam: Tensor  # [B] long: index into FAMILIES (-1 for scattered)
    n_sys: Tensor  # [B] long: number of weather systems
    Q: Tensor  # [B, J, 2] system centres, NM; they drift with ``drift``
    axis: Tensor  # [B, J, 2] unit band axis (banded)
    half_len: Tensor  # [B, J] band half-length, NM (banded)
    sys: Tensor  # [B, K] long: system of the cell in each slot (-1 for scattered)


def init_cells(B: int, cfg: EnvCfg, rng: Rng, steps: int | None = None) -> Cells:
    """Sample the episode drift (and weather systems), start with no cells, run ``steps`` (default: burn-in) updates.

    System centres are placed so that they reach their t = 0 position after the full burn-in.
    """
    K, w = cfg.domain.n_cells_max, cfg.weather
    if w.family not in FAMILIES + ("mixed", "scattered"):
        raise ValueError(f"unknown weather family '{w.family}'")
    J = max(w.n_systems_max)
    z = lambda *s: torch.zeros(B, K, *s, device=rng.device)
    speed = rng.uniform(w.drift_lo, w.drift_hi, B)
    phi = rng.uniform(0.0, 2 * math.pi, B)
    drift = speed.unsqueeze(-1) * torch.stack((torch.cos(phi), torch.sin(phi)), dim=-1)
    cells = Cells(
        q=z(2), rho=z(), I=z(), u=z(2), g=z(), g0=z(), age=z(), tau=z() + 1.0, I_pk=z(),
        alive=torch.zeros(B, K, dtype=torch.bool, device=rng.device),
        q_prev=z(2), rho_prev=z(), drift=drift,
        fam=torch.full((B,), -1, dtype=torch.long, device=rng.device), n_sys=torch.zeros(B, dtype=torch.long, device=rng.device),
        Q=torch.zeros(B, J, 2, device=rng.device), axis=torch.zeros(B, J, 2, device=rng.device),
        half_len=torch.zeros(B, J, device=rng.device), sys=torch.full((B, K), -1, dtype=torch.long, device=rng.device),
    )  # fmt: skip
    if w.family != "scattered":  # the scattered generator draws no systems
        _init_systems(cells, cfg, rng)
    for _ in range(w.burn_in if steps is None else steps):
        step_weather(cells, cfg, rng)
    return cells


def _init_systems(c: Cells, cfg: EnvCfg, rng: Rng) -> None:
    """Draw each episode's family, systems and band geometry; walk the centres back by the burn-in."""
    w, L = cfg.weather, cfg.domain.L
    B, J = c.half_len.shape
    dev = rng.device
    draw = rng.rand(B)
    c.fam = (
        (draw * len(FAMILIES)).long().clamp_max(len(FAMILIES) - 1)
        if w.family == "mixed"
        else torch.full((B,), FAMILIES.index(w.family), dtype=torch.long, device=dev)
    )
    n_max = torch.tensor(w.n_systems_max, device=dev)[c.fam]
    c.n_sys = 1 + (rng.rand(B) * n_max).long().clamp_max(n_max - 1)
    q0 = rng.uniform(w.sys_inset, L - w.sys_inset, B, J, 2)
    wind_dir = torch.atan2(c.drift[:, 1], c.drift[:, 0]).unsqueeze(1)
    ang = wind_dir + math.pi / 2 + math.radians(w.band_jitter_deg) * rng.uniform(-1.0, 1.0, B, J)
    c.axis = torch.stack((torch.cos(ang), torch.sin(ang)), dim=-1)
    c.half_len = rng.uniform(w.band_half_len_lo, w.band_half_len_hi, B, J)
    c.Q = q0 - c.drift.unsqueeze(1) * w.burn_in * cfg.dt_min / 60.0


def _system_births(c: Cells, cfg: EnvCfg, rng: Rng) -> tuple[Tensor, Tensor]:
    """Candidate birth centres [B, K, 2] around a uniformly chosen system [B, K] of each episode."""
    w = cfg.weather
    B, K = c.rho.shape
    j = (rng.rand(B, K) * c.n_sys.unsqueeze(1)).long().clamp_max(c.n_sys.unsqueeze(1) - 1)
    j2 = j.unsqueeze(-1).expand(B, K, 2)
    centre, axis = c.Q.gather(1, j2), c.axis.gather(1, j2)
    # isolated, clustered: uniform in a disc around the centre
    radius = torch.tensor((w.r_isolated, w.r_clustered, 0.0), device=rng.device)[c.fam].unsqueeze(1)
    r, phi = radius * rng.rand(B, K).sqrt(), rng.uniform(0.0, 2 * math.pi, B, K)
    disc = r.unsqueeze(-1) * torch.stack((torch.cos(phi), torch.sin(phi)), dim=-1)
    # banded: uniform along the axis within the half-length, and across it within the half-width
    along = rng.uniform(-1.0, 1.0, B, K) * c.half_len.gather(1, j)
    across = rng.uniform(-w.band_half_width, w.band_half_width, B, K)
    normal = torch.stack((-axis[..., 1], axis[..., 0]), dim=-1)
    band = along.unsqueeze(-1) * axis + across.unsqueeze(-1) * normal
    return centre + torch.where((c.fam == _BANDED).view(B, 1, 1), band, disc), j


def step_weather(c: Cells, cfg: EnvCfg, rng: Rng) -> None:
    """One decision-step weather update: dynamics, death, then birth."""
    w, h = cfg.weather, cfg.dt_min
    lo, hi = cfg.domain.wx_inset, cfg.domain.L - cfg.domain.wx_inset
    B, K = c.rho.shape
    sqrt_h = math.sqrt(h)
    alive = c.alive
    c.q_prev, c.rho_prev = c.q, c.rho

    frac = c.age / c.tau
    u = c.u + w.theta_u * (c.drift.unsqueeze(1) - c.u) * h + w.sigma_u * sqrt_h * rng.randn(B, K, 2)
    g = c.g + w.theta_g * (c.g0 * (1 - 2 * frac) - c.g) * h + w.sigma_g * sqrt_h * rng.randn(B, K)
    rho = (c.rho + g * h).clamp(0.0, w.rho_max)
    I_mean = c.I_pk * (1 - (2 * frac - 1).abs())
    I = (c.I + w.theta_I * (I_mean - c.I) * h + w.sigma_I * sqrt_h * rng.randn(B, K)).clamp(0.0, 1.0)
    q = c.q + u * h / 60.0
    age = c.age + h

    c.u = torch.where(alive.unsqueeze(-1), u, c.u)
    c.q = torch.where(alive.unsqueeze(-1), q, c.q)
    c.g, c.rho, c.I, c.age = (
        torch.where(alive, new, old) for new, old in ((g, c.g), (rho, c.rho), (I, c.I), (age, c.age))
    )

    outside = ((c.q < lo) | (c.q > hi)).any(-1)
    c.alive = alive & ~((c.rho < w.rho_min) | (c.age > c.tau) | outside)

    c.Q = c.Q + c.drift.unsqueeze(1) * h / 60.0

    # Birth into the lowest free slots; around a weather system, a birth outside the inset square is dropped.
    free = ~c.alive
    if w.family == "scattered":
        new = free & (free.cumsum(1) <= rng.poisson(w.lambda_b * h, B).unsqueeze(1))
        q_new = rng.uniform(lo, hi, B, K, 2)
    else:
        rate = w.lambda_b * h * torch.tensor(w.birth_factor, device=rng.device)[c.fam]
        new = free & (free.cumsum(1) <= rng.poisson(rate, B).unsqueeze(1))
        q_new, sys_new = _system_births(c, cfg, rng)
        new = new & ((q_new >= lo) & (q_new <= hi)).all(-1)
        c.sys = torch.where(new, sys_new, c.sys)
    g0_new = rng.uniform(w.g0_lo, w.g0_hi, B, K)
    tau_new = rng.uniform(w.tau_lo, w.tau_hi, B, K)
    I_pk_new = rng.uniform(w.I_pk_lo, w.I_pk_hi, B, K)
    u_new = c.drift.unsqueeze(1) + w.sigma_u0 * rng.randn(B, K, 2)
    new2 = new.unsqueeze(-1)
    c.q, c.q_prev, c.u = (
        torch.where(new2, q_new, c.q),
        torch.where(new2, q_new, c.q_prev),
        torch.where(new2, u_new, c.u),
    )
    c.rho = torch.where(new, w.rho_init, c.rho)
    c.rho_prev = torch.where(new, w.rho_init, c.rho_prev)
    c.I = torch.where(new, w.I_init_frac * I_pk_new, c.I)
    c.g, c.g0 = torch.where(new, g0_new, c.g), torch.where(new, g0_new, c.g0)
    c.tau, c.I_pk = torch.where(new, tau_new, c.tau), torch.where(new, I_pk_new, c.I_pk)
    c.age = torch.where(new, 0.0, c.age)
    c.alive = c.alive | new


def snapshot(c: Cells) -> Cells:
    """Copy of the cell state, e.g. as seen by the weather radar at one update."""
    return Cells(**{f.name: getattr(c, f.name).clone() for f in fields(c)})


def edge_distance(p: Tensor, c: Cells) -> Tensor:
    """Distance to the nearest alive cell edge, negative inside a cell, inf without cells; [B, N, 2] -> [B, N]."""
    d = pairwise_dist(p, c.q) - c.rho.unsqueeze(1)
    return d.masked_fill(~c.alive.unsqueeze(1), float("inf")).amin(dim=-1)


def severity(p: Tensor, c: Cells) -> Tensor:
    """sev(p) = max_k alive_k * I_k * max(0, 1 - |p - q_k| / rho_k); [B, N, 2] -> [B, N]."""
    depth = (1 - pairwise_dist(p, c.q) / c.rho.clamp_min(1e-6).unsqueeze(1)).clamp_min(0.0)
    return (c.alive.unsqueeze(1) * c.I.unsqueeze(1) * depth).amax(dim=-1)
