"""Individual reward components."""

from __future__ import annotations

import math

from torch import Tensor

from config import EnvCfg


def reward_terms(
    cfg: EnvCfg, active: Tensor, d_before: Tensor, d_after: Tensor, flown_frac: Tensor, los_frac: Tensor,
    wx: Tensor, dpsi_change: Tensor, exited: Tensor, failed: Tensor,
) -> dict[str, Tensor]:  # fmt: skip
    """Weighted, signed reward terms [B, N]; their sum is the reward. Inactive agents get 0.

    ``flown_frac`` is the fraction of the step the aircraft was flying (< 1 only in the step it
    exits); ``dpsi_change`` is the absolute turn commanded this step, rad.
    """
    w = cfg.reward
    nominal_step = cfg.aircraft.v_nom * cfg.domain.dt / 3600.0
    terms = {
        # 0 when flying direct at nominal speed, negative by the progress lost otherwise.
        "r_eff": w.w_p * ((d_before - d_after) / nominal_step - flown_frac),
        "r_los": -w.w_l * los_frac,
        "r_wx": -w.w_w * wx,
        "r_switch": -w.w_s * dpsi_change / math.radians(w.switch_unit_deg),
        "r_exit": w.w_e * exited,
        "r_fail": -w.w_f * failed,
    }
    return {k: v * active for k, v in terms.items()}
