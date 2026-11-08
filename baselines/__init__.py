"""Sanity baselines: not training algorithms.

Every policy has the signature ``policy(env, avail, gen) -> actions [B, N]``.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor

from core.observation import build_obs
from env import WxEnv


def direct(env: WxEnv, avail: Tensor, gen: torch.Generator | None = None) -> Tensor:
    """Always hold heading: the direct route, since flights spawn heading at their exit."""
    return torch.full(avail.shape[:-1], env.cfg.nominal_action, dtype=torch.long, device=avail.device)


def random(env: WxEnv, avail: Tensor, gen: torch.Generator | None = None) -> Tensor:
    """Uniform over the available actions (inactive agents only have the nominal one)."""
    flat = avail.reshape(-1, avail.shape[-1]).float().cpu()
    return torch.multinomial(flat, 1, generator=gen).view(avail.shape[:-1]).to(avail.device)


def _best_heading(env: WxEnv, beta: Tensor, conflict: Tensor, weather: Tensor) -> Tensor:
    """Turn with the lowest cost, scored with the env's reward weights:

    w_l * conflict + w_w * weather + w_p * (1 - cos(turn - beta)) * steps + w_s * |turn| / unit

    ``beta`` is the bearing to the exit relative to the heading. The efficiency term is charged over
    the ``steps`` of the scan horizon, since a heading error lasts until it is corrected.
    """
    cfg, ac, w = env.cfg, env.ac, env.cfg.reward
    offs = torch.deg2rad(torch.tensor(cfg.action.heading_offsets_deg, device=ac.p.device))
    steps = cfg.obs.horizon_s / cfg.domain.dt
    eff = (1 - torch.cos(offs - beta.unsqueeze(-1))) * steps
    change = offs.abs() / math.radians(w.switch_unit_deg)
    cost = w.w_l * conflict + w.w_w * weather + w.w_p * eff + w.w_s * change
    n_speed = len(cfg.action.speed_deltas_kt)
    actions = cost.argmin(-1) * n_speed + cfg.action.speed_deltas_kt.index(0.0)
    return torch.where(ac.alive, actions, cfg.nominal_action)


def avoid(env: WxEnv, avail: Tensor, gen: torch.Generator | None = None) -> Tensor:
    """Scripted avoidance on the agents' own observation: the lowest predicted cost wins.

    From the observation it takes the per-turn conflict and weather (latest radar update) and the
    bearing to the exit relative to the heading, and scores them with ``_best_heading``. It ignores
    the previous-update weather block.
    """
    H = len(env.cfg.action.heading_offsets_deg)
    obs = build_obs(env.ac, env.radar, env.radar_prev, env.cfg)
    beta = torch.atan2(obs[..., 2], obs[..., 1])
    return _best_heading(env, beta, obs[..., 3 : 3 + H], obs[..., 3 + H : 3 + 2 * H])


POLICIES: dict[str, Callable[..., Tensor]] = {
    "direct": direct,
    "random": random,
    "avoid": avoid,
}
