"""The planner's objective and the scripted direct action.

The planning cost of a rollout is the negative sum of the environment's own reward terms over the
horizon, per aircraft (so the team cost is their sum and each aircraft's share is available). No
latent-distance or goal-image cost exists anywhere in this module: the Push-T study showed the cost
readout, not the dynamics, to be the first bottleneck, so the readout here is the simulator's reward
(true-simulator oracle) or the same formula on decoded entity features (learned model, ``jepa.py``).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from config import EnvCfg
from core.aircraft import Aircraft
from core.geometry import bearing, pairwise_dist, wrap

REWARD_KEYS = ("r_eff", "r_los", "r_wx", "r_switch", "r_exit", "r_fail")


def nearest_action(turn: Tensor, cfg: EnvCfg) -> Tensor:
    """The heading-only action whose offset is nearest to ``turn`` [rad], same shape."""
    offs = torch.deg2rad(torch.tensor(cfg.action.heading_offsets_deg, device=turn.device))
    idx = (offs - wrap(turn).unsqueeze(-1)).abs().argmin(-1)
    return idx * len(cfg.action.speed_deltas_kt) + cfg.action.speed_deltas_kt.index(0.0)


def direct_action(ac: Aircraft, cfg: EnvCfg) -> Tensor:
    """Fly to the exit [B, N]: the turn nearest the bearing to it, recomputed from the rolled-out state."""
    return nearest_action(bearing(ac.p, ac.e) - ac.psi_cmd, cfg)


def step_cost(terms: dict[str, Tensor]) -> Tensor:
    """Per-aircraft cost of one decision step [B, N] from the env's reward terms."""
    return -sum(terms[k] for k in REWARD_KEYS)


def decoded_step_cost(cfg: EnvCfg, p: Tensor, p_next: Tensor, alive: Tensor, e: Tensor, q: Tensor, rho: Tensor,
                      I: Tensor, c_alive: Tensor, dpsi: Tensor) -> Tensor:  # fmt: skip
    """The reward formula on decoded entity features, one decision step, per aircraft [B, N].

    Positions in NM, headings in rad. Loss of separation and weather severity are evaluated at the
    end of the step (the simulator samples every 5 s substep; the model predicts decision steps).
    """
    w, N = cfg.reward, p.shape[1]
    nominal_step = cfg.aircraft.v_nom * cfg.domain.dt / 3600.0
    prog = ((e - p).norm(dim=-1) - (e - p_next).norm(dim=-1)) / nominal_step - 1.0
    other = alive.unsqueeze(1) & ~torch.eye(N, dtype=torch.bool, device=p.device)
    los = (other & (pairwise_dist(p_next, p_next) < w.d_sep)).any(-1).float()
    depth = (1 - pairwise_dist(p_next, q) / rho.clamp_min(1e-6).unsqueeze(1)).clamp_min(0.0)
    sev = (c_alive.unsqueeze(1) * I.unsqueeze(1) * depth).amax(-1)
    switch = dpsi.abs() / math.radians(w.switch_unit_deg)
    cost = -(w.w_p * prog - w.w_l * los - w.w_w * sev - w.w_s * switch)
    return cost * alive
