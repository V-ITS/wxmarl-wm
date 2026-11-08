"""Per-heading scan observation and global state builders."""

from __future__ import annotations

import torch
from torch import Tensor

from config import EnvCfg
from core.aircraft import Aircraft
from core.geometry import bearing, pairwise_dist, wrap
from core.weather import Cells


def scan(ac: Aircraft, cells: Cells, cfg: EnvCfg) -> tuple[Tensor, Tensor, Tensor]:
    """Predicted conflict, weather and time to weather [B, N, H] for each turn of each aircraft.

    The aircraft's track on ``psi_cmd + turn`` is extrapolated in a straight line for ``obs.horizon_s``
    (or until its exit), with the other aircraft continuing on their current velocity and the cells held still.
    ``conflict`` is ``max(0, 1 - d_cpa / (d_sep + buffer))`` over the other aircraft; ``weather`` is
    the highest severity on the track with every cell radius enlarged by ``obs.buffer_nm``;
    ``entry`` is the time until the track first enters such an enlarged cell, divided by
    ``obs.horizon_s`` (0: already inside, 1: clear within the horizon).
    """
    o, N = cfg.obs, ac.alive.shape[1]
    offs = torch.deg2rad(torch.tensor(cfg.action.heading_offsets_deg, device=ac.p.device))  # [H]
    unit = lambda ang: torch.stack((torch.cos(ang), torch.sin(ang)), dim=-1)
    speed = ac.v / 3600.0  # NM/s
    vel_now = speed.unsqueeze(-1) * unit(ac.psi)  # [B, N, 2]
    vel = speed[..., None, None] * unit(ac.psi_cmd.unsqueeze(-1) + offs)  # [B, N, H, 2]
    horizon = ((ac.e - ac.p).norm(dim=-1) / speed.clamp_min(1e-6)).clamp(max=o.horizon_s)  # [B, N]

    # Closest approach to each other aircraft j, for each candidate of aircraft i: [B, i, H, j].
    r = (ac.p.unsqueeze(1) - ac.p.unsqueeze(2)).unsqueeze(2)
    dv = vel_now[:, None, None] - vel.unsqueeze(3)
    t_cpa = -(r * dv).sum(-1) / (dv * dv).sum(-1).clamp_min(1e-9)
    t_cpa = torch.minimum(t_cpa.clamp_min(0.0), horizon[..., None, None])
    d_cpa = (r + dv * t_cpa.unsqueeze(-1)).norm(dim=-1)
    other = ac.alive.unsqueeze(1) & ~torch.eye(N, dtype=torch.bool, device=ac.p.device)  # [B, i, j]
    seen = cells.alive.unsqueeze(1)  # [B, 1 or N, K]: cells each aircraft sees
    if o.sense_range_nm is not None:
        other = other & (pairwise_dist(ac.p, ac.p) <= o.sense_range_nm)
        seen = seen & (pairwise_dist(ac.p, cells.q) - cells.rho.unsqueeze(1) <= o.sense_range_nm)
    conflict = ((1 - d_cpa / (cfg.reward.d_sep + o.buffer_nm)).clamp_min(0.0) * other.unsqueeze(2)).amax(-1)

    # Highest buffered severity at points along each candidate track: [B, N, H, points, K].
    t = horizon.unsqueeze(-1) * torch.arange(1, o.track_points + 1, device=ac.p.device) / o.track_points
    pos = ac.p[:, :, None, None] + vel.unsqueeze(3) * t[:, :, None, :, None]
    dist = (pos.unsqueeze(-2) - cells.q[:, None, None, None]).norm(dim=-1)
    depth = (1 - dist / (cells.rho[:, None, None, None] + o.buffer_nm)).clamp_min(0.0)
    weather = (depth * (cells.I.unsqueeze(1) * seen)[:, :, None, None]).amax(dim=(-1, -2))

    # First time each track enters each enlarged cell: |rel + v t| = R, [B, N, H, K].
    rel = ac.p[:, :, None, None] - cells.q[:, None, None]  # [B, N, 1, K, 2]
    vv = vel.unsqueeze(3)  # [B, N, H, 1, 2]
    a = (vv * vv).sum(-1).clamp_min(1e-12)
    b = (rel * vv).sum(-1)
    c = (rel * rel).sum(-1) - (cells.rho + o.buffer_nm)[:, None, None] ** 2
    disc = b * b - a * c
    t_in = torch.where(c <= 0, 0.0, (-b - disc.clamp_min(0.0).sqrt()) / a)
    ahead = (disc >= 0) & (t_in >= 0) & (t_in <= horizon[..., None, None])
    hit = seen.unsqueeze(2) & ((c <= 0) | ahead)
    entry = torch.where(hit, t_in, o.horizon_s).amin(-1) / o.horizon_s
    return conflict, weather, entry


def build_obs(ac: Aircraft, radar: Cells, radar_prev: Cells, cfg: EnvCfg) -> Tensor:
    """Observations [B, N, 3 + 5H]; all-zero for inactive slots.

    ``d_exit/L``, ``cos β``, ``sin β`` (β = bearing to exit relative to the heading), then per
    turn (``H`` each) from ``scan``: conflict and weather on the latest radar update,
    weather on the update before it, and time to weather on the latest and on the previous update.
    """
    beta = wrap(bearing(ac.p, ac.e) - ac.psi)
    own = torch.stack(((ac.e - ac.p).norm(dim=-1) / cfg.domain.L, torch.cos(beta), torch.sin(beta)), dim=-1)
    conflict, weather, entry = scan(ac, radar, cfg)
    _, weather_prev, entry_prev = scan(ac, radar_prev, cfg)
    obs = torch.cat((own, conflict, weather, weather_prev, entry, entry_prev), dim=-1)
    return obs * ac.alive.unsqueeze(-1)


GRID_PLANES = 6  # weather latest / previous radar update, traffic, traffic velocity (right, ahead), beyond the wall


def build_grid(ac: Aircraft, radar: Cells, radar_prev: Cells, t: int, cfg: EnvCfg) -> Tensor:
    """Image observations [B, N, GRID_PLANES * G * G + 4]; all-zero for inactive slots.

    A ``G x G`` window of ``grid_res_nm`` pixels centred on the aircraft, rotated so that its
    assigned heading points up (row 0 is the farthest ahead, column 0 the leftmost). Planes: severity of the
    latest and previous radar update; other aircraft as disks of radius ``d_sep / 2`` with a
    one-pixel soft edge; on each disk, that aircraft's velocity / ``v_nom`` to the right and ahead
    in the image frame; and the area beyond the boundary wall. Weather and traffic are limited to
    ``sense_range_nm``. Then ``d_exit/L``, ``cos β``, ``sin β`` and the radar age (steps since the
    latest update / steps between updates).
    """
    o, d = cfg.obs, cfg.domain
    G, R = o.grid_n, o.sense_range_nm
    B, N = ac.alive.shape
    c = (torch.arange(G, device=ac.p.device) - (G - 1) / 2) * o.grid_res_nm
    x, y = c.view(1, G), c.flip(0).view(G, 1)  # right, ahead (NM) of each pixel
    phi = ac.psi_cmd
    ahead =torch.stack((torch.cos(phi), torch.sin(phi)), dim=-1)[:, :, None, None]  # [B, N, 1, 1, 2]
    right = torch.stack((torch.sin(phi), -torch.cos(phi)), dim=-1)[:, :, None, None]
    pts = ac.p[:, :, None, None] + y.unsqueeze(-1) * ahead + x.unsqueeze(-1) * right  # [B, N, G, G, 2]
    in_range = torch.ones(G, G, dtype=torch.bool, device=ac.p.device) if R is None else x**2 + y**2 <= R**2

    def weather(cells: Cells) -> Tensor:
        seen = cells.alive.unsqueeze(1)
        if R is not None:
            seen = seen & (pairwise_dist(ac.p, cells.q) - cells.rho.unsqueeze(1) <= R)
        dist = (pts.unsqueeze(-2) - cells.q[:, None, None, None]).norm(dim=-1)  # [B, N, G, G, K]
        depth = (1 - dist / cells.rho.clamp_min(1e-6)[:, None, None, None]).clamp_min(0.0)
        return (depth * (cells.I.unsqueeze(1) * seen)[:, :, None, None]).amax(-1)

    other = ac.alive.unsqueeze(1) & ~torch.eye(N, dtype=torch.bool, device=ac.p.device)  # [B, i, j]
    if R is not None:
        other = other & (pairwise_dist(ac.p, ac.p) <= R)

    dist = (pts.unsqueeze(-2) - ac.p[:, None, None, None]).norm(dim=-1)  # [B, i, G, G, j]
    dist = torch.where(other[:, :, None, None], dist, torch.inf)
    res = o.grid_res_nm
    disk = ((cfg.reward.d_sep / 2 + res / 2 - dist) / res).clamp(0.0, 1.0).amax(-1)  # [B, i, G, G]
    vel = (ac.v / cfg.aircraft.v_nom).unsqueeze(-1) * torch.stack((torch.cos(ac.psi), torch.sin(ac.psi)), dim=-1)
    v_right = (vel.unsqueeze(1) * right[:, :, 0]).sum(-1)  # [B, i, j]: velocity of j in i's image frame
    v_ahead = (vel.unsqueeze(1) * ahead[:, :, 0]).sum(-1)
    nearest = dist.argmin(-1, keepdim=True)  # the aircraft whose disk a pixel shows
    on_disk = disk > 0
    pick = lambda v: torch.gather(v[:, :, None, None].expand(-1, -1, G, G, -1), -1, nearest).squeeze(-1) * on_disk

    outside = ((pts < -d.oob_margin) | (pts > d.L + d.oob_margin)).any(-1).float()
    planes = torch.stack((weather(radar), weather(radar_prev), disk, pick(v_right), pick(v_ahead)), dim=2) * in_range
    image = torch.cat((planes, outside.unsqueeze(2)), dim=2).flatten(2)  # [B, N, GRID_PLANES * G * G]
    beta = wrap(bearing(ac.p, ac.e) - ac.psi)
    own = torch.stack(
        ((ac.e - ac.p).norm(dim=-1) / d.L, torch.cos(beta), torch.sin(beta),
         torch.full_like(beta, (t % cfg.radar_steps) / cfg.radar_steps)),
        dim=-1,
    )  # fmt: skip
    return torch.cat((image, own), dim=-1) * ac.alive.unsqueeze(-1)


def build_state(ac: Aircraft, cells: Cells, t: int, cfg: EnvCfg) -> Tensor:
    """Global state [B, state_dim]; inactive aircraft and dead cells are zeroed."""
    L, v_nom, w = cfg.domain.L, cfg.aircraft.v_nom, cfg.weather
    B = ac.alive.shape[0]
    alive = ac.alive.unsqueeze(-1)
    air = torch.cat(
        (ac.p / L, torch.cos(ac.psi).unsqueeze(-1), torch.sin(ac.psi).unsqueeze(-1),
         (ac.v / v_nom).unsqueeze(-1), ac.e / L, alive),
        dim=-1,
    )  # fmt: skip
    hidden = torch.cat((cells.u / v_nom, (cells.g / w.g0_hi).unsqueeze(-1)), dim=-1)
    if not cfg.state.include_hidden_weather:
        hidden = torch.zeros_like(hidden)
    c_alive = cells.alive.unsqueeze(-1)
    wx = torch.cat((cells.q / L, (cells.rho / w.rho_max).unsqueeze(-1), cells.I.unsqueeze(-1), hidden, c_alive), dim=-1)
    time = torch.full((B, 1), t / cfg.domain.T, device=ac.p.device)
    return torch.cat(((air * alive).flatten(1), (wx * c_alive).flatten(1), time), dim=-1)
