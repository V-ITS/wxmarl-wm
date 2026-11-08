"""Record traffic for the learned model: frames of entity tokens under one teammate policy.

    python -m wm.data DATA-avoid            # results/redesign/data/avoid-N5.pt

Episodes run in batches of ``data.batch`` with seeds ``data.seed + batch index`` (disjoint from the
scenario seeds). Every decision step stores the frame (``wm.jepa.Frame``), the executed actions,
the alive mask and the per-aircraft step cost. Padded to the episode cap with a ``running`` mask.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import Tensor

from baselines import POLICIES
from env import WxEnv
from wm import experiments
from wm.config import DiagCfg, parse_overrides
from wm.cost import step_cost
from wm.jepa import frame_from_snapshot
from wm.snapshot import take

FIELDS = ("air_in", "cell_in", "air_tgt", "cell_tgt", "actions", "alive", "cost", "running")


class Recorder:
    """Accumulates one decision step at a time into the dataset layout [B, T, ...] (padded to ``T``)."""

    def __init__(self, T: int) -> None:
        self.T = T
        self.rows: dict[str, list[Tensor]] = {k: [] for k in FIELDS}

    def add(self, frame, actions: Tensor, alive: Tensor, cost: Tensor, running: Tensor) -> None:
        for k, v in (("air_in", frame.air_in), ("cell_in", frame.cell_in), ("air_tgt", frame.air_tgt), ("cell_tgt", frame.cell_tgt),
                     ("actions", actions), ("alive", alive), ("cost", cost), ("running", running)):  # fmt: skip
            self.rows[k].append(v.detach().cpu())

    def finish(self) -> dict[str, Tensor]:
        out = {k: torch.stack(v, 1) for k, v in self.rows.items()}
        B, pad = out["running"].shape[0], self.T - out["running"].shape[1]
        if pad > 0:
            for k, v in out.items():
                out[k] = torch.cat((v, torch.zeros(B, pad, *v.shape[2:], dtype=v.dtype)), 1)
        return out


def save_dataset(data: dict[str, Tensor], path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**data, "meta": meta}, path)


def load_dataset(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def dataset_path(cfg: DiagCfg, name: str | None = None) -> Path:
    name = name or cfg.data.name or f"{cfg.data.policy}-N{cfg.n_agents}"
    return Path(cfg.out) / "data" / f"{name}.pt"


def collect_batch(cfg: DiagCfg, env_cfg, seed: int, B: int, device, log) -> dict[str, Tensor]:
    env = WxEnv(env_cfg, batch_size=B, device=str(device))
    _, _, _, avail = env.reset(seed=seed)
    gen = torch.Generator().manual_seed(seed)
    rec = Recorder(env_cfg.domain.T)
    info: dict = {}
    while "episode" not in info:
        snap = take(env)
        f = frame_from_snapshot(snap, env_cfg)
        actions = POLICIES[cfg.data.policy](env, avail, gen)
        running = ~env.finished
        *_, avail, info = env.step(actions)
        rec.add(f, actions, snap.ac.alive, step_cost(info), running)
    return rec.finish()


def collect(cfg: DiagCfg, log=print) -> Path:
    env_cfg = cfg.env_cfg()
    device = torch.device(cfg.device)
    B = cfg.data.batch
    assert cfg.data.episodes % B == 0
    parts: list[dict[str, Tensor]] = []
    t0 = time.time()
    for i in range(cfg.data.episodes // B):
        parts.append(collect_batch(cfg, env_cfg, cfg.data.seed + i, B, device, log))
        log(f"  batch {i + 1}/{cfg.data.episodes // B} {time.time() - t0:.0f}s")
    data = {k: torch.cat([p[k] for p in parts], 0) for k in FIELDS}
    meta = {"policy": cfg.data.policy, "n_agents": cfg.n_agents, "episodes": cfg.data.episodes, "seed": cfg.data.seed,
            "env_cfg_hash": env_cfg.hash(), "cfg": cfg.to_dict()}  # fmt: skip
    path = dataset_path(cfg)
    save_dataset(data, path, meta)
    steps = int(data["running"].sum())
    log(f"{path}: {cfg.data.episodes} episodes, {steps} steps, {time.time() - t0:.0f}s")
    return path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_id", choices=sorted(k for k in experiments.REGISTRY if k.startswith("DATA-")))
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_intermixed_args(argv)
    _, cfg = experiments.resolve(args.run_id, parse_overrides(args.overrides), args.mock)
    collect(cfg)


if __name__ == "__main__":
    main()
