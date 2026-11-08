"""Route sampling, initial placement of the N aircraft and flight termination."""

from __future__ import annotations

import torch
from torch import Tensor

from config import EnvCfg
from core.aircraft import Aircraft
from core.geometry import bearing, pairwise_dist
from core.rng import Rng
from core.weather import Cells

# Exit-fix candidates drawn per route. A draw is valid with probability > 0.6, and a
# mirrored point (always valid) is used if all of them fail.
_EXIT_DRAWS = 16


def perimeter_point(s: Tensor, L: float) -> tuple[Tensor, Tensor]:
    """Map arc length s in [0, 4L) to a perimeter point [..., 2] and its edge index."""
    edge = (s / L).floor().clamp(0, 3)
    o = s - edge * L
    zero, full = torch.zeros_like(s), torch.full_like(s, L)
    x = torch.where(edge == 0, o, torch.where(edge == 1, full, torch.where(edge == 2, L - o, zero)))
    y = torch.where(edge == 0, zero, torch.where(edge == 1, o, torch.where(edge == 2, full, L - o)))
    return torch.stack((x, y), dim=-1), edge


def sample_routes(shape: tuple[int, ...], cfg: EnvCfg, rng: Rng) -> tuple[Tensor, Tensor]:
    """Sample entry and exit points [*shape, 2]: different edges, at least min_route_frac * L apart."""
    L = cfg.domain.L
    p0, edge0 = perimeter_point(rng.uniform(0.0, 4 * L, *shape), L)
    cand, edge1 = perimeter_point(rng.uniform(0.0, 4 * L, *shape, _EXIT_DRAWS), L)
    dist = (cand - p0.unsqueeze(-2)).norm(dim=-1)
    valid = (edge1 != edge0.unsqueeze(-1)) & (dist >= cfg.traffic.min_route_frac * L)
    first = valid.int().argmax(dim=-1, keepdim=True)
    e = cand.gather(-2, first.unsqueeze(-1).expand(*shape, 1, 2)).squeeze(-2)
    e = torch.where(valid.any(-1, keepdim=True), e, L - p0)
    return p0, e


def spawn(ac: Aircraft, cells: Cells, slot: int, cfg: EnvCfg, rng: Rng) -> Tensor:
    """Place a new flight in ``slot`` of every env, at its perimeter entry point.

    Takes the first candidate that is at least ``spawn_sep`` from the aircraft already placed and
    outside every cell. If there is none, the weather check is waived; if that also fails, the
    candidate farthest from the other aircraft is used. Returns the [B] mask of such relaxed spawns.
    """
    tr = cfg.traffic
    B = ac.alive.shape[0]
    p0, e = sample_routes((B, tr.spawn_candidates), cfg, rng)
    d_min = pairwise_dist(p0, ac.p).masked_fill(~ac.alive.unsqueeze(1), float("inf")).amin(-1)
    in_cell = (cells.alive.unsqueeze(1) & (pairwise_dist(p0, cells.q) < cells.rho.unsqueeze(1))).any(-1)
    separated = d_min >= tr.spawn_sep
    valid = separated & ~in_cell
    pick = torch.where(separated.any(1), separated.int().argmax(1), d_min.argmax(1))
    pick = torch.where(valid.any(1), valid.int().argmax(1), pick)

    b = torch.arange(B, device=pick.device)
    p0, e = p0[b, pick], e[b, pick]
    ac.p[:, slot], ac.p0[:, slot], ac.e[:, slot] = p0, p0, e
    ac.psi[:, slot] = ac.psi_cmd[:, slot] = bearing(p0, e)
    ac.v[:, slot] = ac.v_cmd[:, slot] = cfg.aircraft.v_nom
    ac.t_dir[:, slot] = (e - p0).norm(dim=-1) / cfg.aircraft.v_nom * 3600.0
    ac.alive[:, slot] = True
    return ~valid.any(1)


def failures(ac: Aircraft, cfg: EnvCfg) -> Tensor:
    """Timeout mask [B, N] (not yet masked by alive); aircraft cannot leave the sector (``substep``)."""
    return ac.flight_time > cfg.traffic.timeout_factor * ac.t_dir
