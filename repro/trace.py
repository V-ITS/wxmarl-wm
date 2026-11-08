"""Replay a small batch under simultaneous replanning and under a fixed order, recording every aircraft's track and plan,
and under simultaneous replanning the plans of the first round of exchange as well.

    python -m repro.trace [--B 16] [--seed 200000]   # -> results/redesign/paper_cache/trace-<mode>.npz
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from env import WxEnv
from repro.jobs import results_root
from wm import experiments
from wm.eval import make_dynamics
from wm.planner import Planner
from wm.snapshot import take


def replay(run: str, B: int, seed: int, device: str) -> dict:
    _, cfg = experiments.resolve(run, {"device": device})
    env_cfg = cfg.env_cfg()
    live = WxEnv(env_cfg, batch_size=B, device=device)
    live.reset(seed=seed)
    planner = Planner(cfg, env_cfg, make_dynamics(cfg, env_cfg, device), device)
    planner.reset(B, cfg.planner_seed * 1_000_003 + seed)
    rounds: list = []
    respond = planner.best_responses

    def record(*a, **k):  # each round of plan exchange within one replanning (simultaneous mode)
        plan, cost = respond(*a, **k)
        rounds.append(plan.detach().cpu().numpy().copy())
        return plan, cost

    planner.best_responses = record
    rec: dict = {k: [] for k in ("p", "psi", "alive", "action", "plan", "round1", "cq", "crho", "calive")}
    info, t0 = {}, time.time()
    while "episode" not in info:
        rounds.clear()
        actions = planner.act(take(live))
        rec["round1"].append(rounds[0] if rounds else planner.plan.detach().cpu().numpy().copy())
        for k, v in (("p", live.ac.p), ("psi", live.ac.psi), ("alive", live.ac.alive), ("action", actions), ("plan", planner.plan),
                     ("cq", live.cells.q), ("crho", live.cells.rho), ("calive", live.cells.alive)):  # fmt: skip
            rec[k].append(v.detach().cpu().numpy().copy())
        *_, info = live.step(actions)
        print(f"{run} t={live.t} {time.time() - t0:.0f}s", flush=True)
    out = {k: np.stack(v, 1) for k, v in rec.items()}  # [B, T, ...]
    out["exit"] = live.ac.e.cpu().numpy()
    out["los"] = info["episode"]["los_events"].cpu().numpy()
    out["offsets"] = np.asarray(env_cfg.action.heading_offsets_deg)
    out["v"] = live.ac.v.cpu().numpy()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=16)
    ap.add_argument("--seed", type=int, default=200_000)
    ap.add_argument("--modes", default="shared,fixed")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    cache = results_root() / "paper_cache"
    cache.mkdir(parents=True, exist_ok=True)
    for mode in a.modes.split(","):
        np.savez_compressed(cache / f"trace-{mode}.npz", **replay(f"E3-N5-{mode}", a.B, a.seed, a.device))


if __name__ == "__main__":
    main()
