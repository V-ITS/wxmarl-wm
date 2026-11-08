"""Aircraft state and point-mass kinematics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from config import EnvCfg
from core.geometry import wrap


@dataclass
class Aircraft:
    """Per-slot aircraft state, all [B, N] or [B, N, 2]."""

    p: Tensor
    psi: Tensor
    v: Tensor
    psi_cmd: Tensor
    v_cmd: Tensor
    e: Tensor  # exit fix
    p0: Tensor  # entry point
    flight_time: Tensor  # s
    t_dir: Tensor  # direct-route time, s
    path_len: Tensor  # NM
    last_dpsi: Tensor  # rad
    wx_hit: Tensor  # bool: flight has been inside a cell
    env_hit: Tensor  # bool: flight has been within the avoidance envelope (weather.envelope_nm of a cell edge)
    alive: Tensor  # bool

    @classmethod
    def empty(cls, B: int, N: int, device: str | torch.device) -> Aircraft:
        z = lambda *s: torch.zeros(B, N, *s, device=device)
        b = lambda: torch.zeros(B, N, dtype=torch.bool, device=device)
        return cls(z(2), z(), z(), z(), z(), z(2), z(2), z(), z() + 1.0, z(), z(), b(), b(), b())


def apply_actions(ac: Aircraft, actions: Tensor, cfg: EnvCfg) -> None:
    """Set heading and speed commands from discrete actions [B, N].

    The heading action is a turn from the assigned heading ``psi_cmd``.
    """
    n_speed = len(cfg.action.speed_deltas_kt)
    dpsi = torch.deg2rad(torch.tensor(cfg.action.heading_offsets_deg, device=actions.device))[actions // n_speed]
    dv = torch.tensor(cfg.action.speed_deltas_kt, device=actions.device)[actions % n_speed]
    ac.psi_cmd = torch.where(ac.alive, wrap(ac.psi_cmd + dpsi), ac.psi_cmd)
    ac.v_cmd = torch.where(ac.alive, (ac.v_cmd + dv).clamp(cfg.aircraft.v_min, cfg.aircraft.v_max), ac.v_cmd)
    ac.last_dpsi = torch.where(ac.alive, dpsi, ac.last_dpsi)


def substep(ac: Aircraft, moving: Tensor, cfg: EnvCfg) -> None:
    """Advance the ``moving`` slots by one physics substep; other slots are frozen."""
    a, delta = cfg.aircraft, cfg.domain.substep
    max_turn = math.radians(a.turn_rate_deg) * delta
    psi = wrap(ac.psi + wrap(ac.psi_cmd - ac.psi).clamp(-max_turn, max_turn))
    v = ac.v + (ac.v_cmd - ac.v).clamp(-a.accel * delta, a.accel * delta)
    step = v * delta / 3600.0
    p = ac.p + step.unsqueeze(-1) * torch.stack((torch.cos(psi), torch.sin(psi)), dim=-1)
    # Hard boundary: the outward part of the motion is dropped, so the aircraft slides along the edge.
    p = p.clamp(-cfg.domain.oob_margin, cfg.domain.L + cfg.domain.oob_margin)
    ac.path_len = ac.path_len + moving * (p - ac.p).norm(dim=-1)
    ac.psi = torch.where(moving, psi, ac.psi)
    ac.v = torch.where(moving, v, ac.v)
    ac.p = torch.where(moving.unsqueeze(-1), p, ac.p)
    ac.flight_time = ac.flight_time + moving * delta
