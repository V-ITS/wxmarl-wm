"""Env throughput: env steps/s for B in {1, 64} on CPU, and on a GPU if available."""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baselines import random
from config import load_config
from env import WxEnv


def bench(B: int, device: str, steps: int, preset: str) -> float:
    """Env steps per second under the random policy, with auto-reset.

    Only steps of envs whose episode is still running are counted: an env that finishes early
    idles until the rest of the batch is done.
    """
    cfg = load_config(preset)
    env = WxEnv(cfg, batch_size=B, device=device, seed=0, auto_reset=True)
    gen = torch.Generator().manual_seed(0)
    _, _, _, avail = env.reset()
    counted = 0
    for i in range(steps + 10):
        if i == 10:  # warm-up done
            start = time.perf_counter()
        if i >= 10:
            counted += B - int(env.finished.sum())
        *_, avail, _ = env.step(random(env, avail, gen))
    if device != "cpu":
        getattr(torch, device).synchronize()
    return counted / (time.perf_counter() - start)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", default="default")
    ap.add_argument("--steps", type=int, default=600, help="decision steps per measurement")
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 64])
    args = ap.parse_args()

    devices = ["cpu"] + [d for d in ("cuda", "mps") if getattr(torch.backends, d).is_built() and (
        torch.cuda.is_available() if d == "cuda" else torch.backends.mps.is_available())]  # fmt: skip
    print(f"torch {torch.__version__}, {torch.get_num_threads()} CPU threads, preset={args.preset}")
    print(f"{'device':8s} {'B':>5s} {'env steps/s':>14s}")
    for device in devices:
        for B in args.batch_sizes:
            print(f"{device:8s} {B:5d} {bench(B, device, args.steps, args.preset):14.0f}")
