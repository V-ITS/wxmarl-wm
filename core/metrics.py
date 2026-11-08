"""Episode metrics accumulator."""

from __future__ import annotations

from collections import defaultdict

import torch
from torch import Tensor


class Metrics:
    """Per-env running sums [B]; ``finalize`` derives the episode ratios."""

    def __init__(self, B: int, device: str | torch.device) -> None:
        self.sums: dict[str, Tensor] = defaultdict(lambda: torch.zeros(B, device=device))

    def add(self, name: str, value: Tensor) -> None:
        """Add a per-env value [B], or sum a per-slot value [B, ...] into it."""
        self.sums[name] = self.sums[name] + value.float().reshape(value.shape[0], -1).sum(1)

    def finalize(self, unfinished: Tensor, wx_hit: Tensor, env_hit: Tensor) -> dict[str, Tensor]:
        """Episode metrics [B]; ``unfinished``, ``wx_hit`` and ``env_hit`` are the [B, N] masks at episode end."""
        s = self.sums
        ratio = lambda a, b: s[a] / s[b].clamp_min(1.0)
        out = {
            k: s[k]
            for k in (
                "ep_length",
                "spawn_relaxed",
                "exits",
                "fail_timeout",
                "los_events",
                "los_time_s",
                "wx_incursion_time_s",
                "wx_severity_integral",
                "wx_envelope_time_s",
                "r_eff",
                "r_los",
                "r_wx",
                "r_switch",
                "r_exit",
                "r_fail",
            )
        }
        out["unfinished"] = unfinished.float().sum(1)
        out["flights"] = torch.full_like(out["exits"], unfinished.shape[1])
        out["flights_wx"] = s["_flights_wx"] + (unfinished & wx_hit).float().sum(1)
        out["flights_envelope"] = s["_flights_envelope"] + (unfinished & env_hit).float().sum(1)
        out["mean_active_aircraft"] = s["_active_steps"] / s["ep_length"].clamp_min(1.0)
        terminated = s["exits"] + s["fail_timeout"]
        out["exit_rate"] = s["exits"] / terminated.clamp_min(1.0)
        out["extra_track_nm"] = ratio("_extra_track", "exits")
        out["delay_s"] = ratio("_delay", "exits")
        out["return_mean"] = sum(
            s[k] for k in ("r_eff", "r_los", "r_wx", "r_switch", "r_exit", "r_fail")
        )
        return out


def summarize(episodes: list[dict[str, float]]) -> dict[str, tuple[float, float]]:
    """Mean and 95% confidence half-width (normal approximation) of each metric over episodes."""
    out = {}
    for k in episodes[0]:
        x = torch.tensor([ep[k] for ep in episodes], dtype=torch.float64)
        half = 1.96 * x.std().item() / len(x) ** 0.5 if len(x) > 1 else float("nan")
        out[k] = (x.mean().item(), half)
    return out
