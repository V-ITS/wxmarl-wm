"""Per-decision-step snapshots of one env, saved as .npz plus .json metadata."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from env import WxEnv


class Recorder:
    """Captures CPU copies of env ``index``. Call ``capture`` after reset and after every step."""

    def __init__(self, env: WxEnv, index: int = 0) -> None:
        self.env, self.index = env, index
        self.frames: dict[str, list[np.ndarray]] = {}

    def capture(self) -> None:
        env, i = self.env, self.index
        ac, c, sums = env.ac, env.cells, env.metrics.sums
        snap = {
            "p": ac.p[i], "psi": ac.psi[i], "alive": ac.alive[i], "e": ac.e[i], "p0": ac.p0[i],
            "cell_q": c.q[i], "cell_rho": c.rho[i], "cell_I": c.I[i], "cell_alive": c.alive[i],
            "los": env.los_step[i],
            "los_events": sums["los_events"][i], "wx_incursion_time_s": sums["wx_incursion_time_s"][i],
        }  # fmt: skip
        for k, v in snap.items():
            self.frames.setdefault(k, []).append(v.cpu().numpy().copy())

    def save(self, stem: str | Path, **meta) -> Path:
        """Write ``<stem>.npz`` and ``<stem>.json``; ``meta`` (seed, algorithm, checkpoint, ...) goes to the json."""
        cfg = self.env.cfg
        stem = Path(stem)
        stem.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            stem.with_suffix(".npz"),
            L=cfg.domain.L, dt=cfg.domain.dt, d_sep=cfg.reward.d_sep,
            **{k: np.stack(v) for k, v in self.frames.items()},
        )  # fmt: skip
        meta = {"config_hash": cfg.hash(), **meta, "config": cfg.to_dict()}
        stem.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        return stem.with_suffix(".npz")
