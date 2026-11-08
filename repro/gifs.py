"""The README animation: one scenario flown by four controllers, side by side.

    python -m repro.gifs [--seed 200000 --row 28] [--device cuda]     # -> docs/controllers.gif

Each controller flies the whole chunk exactly as in its run (``wm.eval.run_chunk``), so the recorded row is the same
episode as in that run's metrics.json. Needs the trained forecaster and QMIX weights under results/redesign/.
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
from PIL import Image

from baselines import POLICIES
from env import WxEnv
from render.animate import _scene
from render.recorder import Recorder
from repro.jobs import ROOT
from wm import experiments
from wm.eval import make_dynamics
from wm.planner import Planner
from wm.snapshot import take

RUNS = (("flying direct", "BASE-direct"), ("reactive avoid", "BASE-avoid"), ("QMIX", "BASE-qmix-s0"), ("planner + forecaster", "E1-forecast-s0"))


def fly(run: str, seed: int, row: int, device: str) -> dict:
    """Fly the chunk ``seed`` under ``run`` and return the recorded arrays of scenario ``row``."""
    _, cfg = experiments.resolve(run, {"device": device})
    env_cfg = cfg.env_cfg()
    live = WxEnv(env_cfg, batch_size=cfg.chunk, device=device)
    *_, avail = live.reset(seed=seed)
    if cfg.policy == "marl":
        from wm.marl import MarlPolicy

        policy = MarlPolicy(cfg.checkpoint, live)
    elif cfg.policy:
        policy = POLICIES[cfg.policy]
    else:
        planner = Planner(cfg, env_cfg, make_dynamics(cfg, env_cfg, device), device)
        planner.reset(cfg.chunk, cfg.planner_seed * 1_000_003 + seed)
    gen = torch.Generator().manual_seed(seed)
    rec, info = Recorder(live, row), {}
    rec.capture()
    while "episode" not in info:
        actions = policy(live, avail, gen) if cfg.policy else planner.act(take(live))
        *_, avail, info = live.step(actions)
        rec.capture()
    cfg_ = live.cfg
    d = {"L": cfg_.domain.L, "dt": cfg_.domain.dt, "d_sep": cfg_.reward.d_sep, **{k: np.stack(v) for k, v in rec.frames.items()}}
    print(f"{run:16s} weather {info['episode']['wx_incursion_time_s'][row]:6.0f} s  LoS {int(info['episode']['los_events'][row])}", flush=True)
    return d


def frames(d: dict, title: str, length: int, dpi: int) -> list[Image.Image]:
    fig, draw = _scene(d, False, "")
    L, ax = float(d["L"]), fig.axes[0]
    fig.set_size_inches(5.0, 5.3)
    fig.set_dpi(dpi)
    ax.set(xlim=(-6, L + 6), ylim=(-6, L + 6), xlabel="", ylabel="", xticks=[], yticks=[])
    ax.title.set_fontsize(8.5)
    fig.suptitle(title, fontsize=14, weight="bold", y=0.985)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.88)
    out = []
    for f in range(length):
        draw(min(f, d["alive"].shape[0] - 1))
        fig.canvas.draw()
        out.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=200_000, help="chunk seed")
    ap.add_argument("--row", type=int, default=28, help="scenario in the chunk (28: W3, weather across most routes)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dpi", type=int, default=80)
    ap.add_argument("--fps", type=int, default=8)
    args = ap.parse_args()
    runs = [(title, fly(run, args.seed, args.row, args.device)) for title, run in RUNS]
    length = max(d["alive"].shape[0] for _, d in runs) + 6  # hold the end for a moment
    panels = [frames(d, title, length, args.dpi) for title, d in runs]
    w, h = panels[0][0].size
    out = []
    for f in range(length):
        grid = Image.new("RGB", (2 * w, 2 * h), "white")
        for i, p in enumerate(panels):
            grid.paste(p[f], ((i % 2) * w, (i // 2) * h))
        out.append(grid.quantize(colors=64, method=Image.Quantize.MEDIANCUT))
    path = ROOT / "docs" / "controllers.gif"
    out[0].save(path, save_all=True, append_images=out[1:], duration=1000 // args.fps, loop=0, optimize=True)
    print(path, f"{path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
