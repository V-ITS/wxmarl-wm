"""Evaluate an EPyMARL checkpoint or a sanity baseline on the fixed evaluation scenarios.

Rendering is on by default. Examples:
    python scripts/evaluate.py --policy direct --preset default
    python scripts/evaluate.py --checkpoint "results/models/<run>" --preset default
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from baselines import POLICIES
from config import load_config
from core.metrics import summarize
from env import WxEnv
from render.animate import animate
from render.recorder import Recorder


def evaluate_baseline(policy: str, preset: str, episodes: int, out: Path, render: bool) -> list[dict[str, float]]:
    """Run ``policy`` on the first ``episodes`` eval seeds with B=1; returns per-episode metrics."""
    cfg = load_config(preset)
    env = WxEnv(cfg, batch_size=1)
    results = []
    for ep, seed in enumerate(cfg.eval_seeds[:episodes]):
        gen = torch.Generator().manual_seed(seed)
        _, _, _, avail = env.reset(seed=seed)
        record = render and ep < cfg.render.n_episodes
        if record:
            rec = Recorder(env)
            rec.capture()
        info: dict = {}
        while "episode" not in info:
            *_, avail, info = env.step(POLICIES[policy](env, avail, gen))
            if record:
                rec.capture()
        results.append({k: v.item() for k, v in info["episode"].items()})
        if record:
            replay = rec.save(out / f"episode_{ep}_seed{seed}", seed=seed, algorithm=policy, checkpoint=None)
            print("video:", animate(replay, fps=cfg.render.fps, title=policy))
    return results


def evaluate_checkpoint(
    checkpoint: Path, algo: str, preset: str, episodes: int, out: Path, render: bool, wandb_mode: str, load_step: int
) -> list[dict[str, float]]:
    """Run EPyMARL in evaluate mode on ``checkpoint``; returns per-episode metrics."""
    cmd = [
        sys.executable, "-m", "wrappers.epymarl_main", f"--config={algo}", "--eval", "with", "seed=0",
        f"checkpoint_path={checkpoint.resolve()}", f"load_step={load_step}", f"test_nepisode={episodes}",
        f"env_args.map_name={preset}", f"env_args.render={render}", f"env_args.out_dir={out.resolve()}",
        f"env_args.checkpoint={checkpoint.name}",
    ]  # fmt: skip
    if wandb_mode != "off":
        cmd += ["use_wandb=True", f"wandb_mode={wandb_mode}"]
    env = os.environ | {
        "WANDB_TAGS": f"{algo},eval,{preset}",
        "WANDB_NAME": os.environ.get("WANDB_NAME", f"eval_{checkpoint.name}"),
    }
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)
    return json.loads((out / "metrics.json").read_text())["episodes"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    what = ap.add_mutually_exclusive_group(required=True)
    what.add_argument("--policy", choices=sorted(POLICIES))
    what.add_argument("--checkpoint", type=Path, help="EPyMARL run directory under results/models/")
    ap.add_argument("--algo", default=None, help="algorithm of the checkpoint (default: from its directory name)")
    ap.add_argument("--load-step", type=int, default=0, help="checkpoint step to load (0 = latest)")
    ap.add_argument("--preset", default="default")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--wandb", choices=("off", "offline", "online"), default="off", help="checkpoints only")
    args = ap.parse_args()

    name = args.policy or args.checkpoint.name.replace(" ", "_").replace(":", "-")
    out = args.out or ROOT / "results" / "eval" / f"{name}_{args.preset}"
    out.mkdir(parents=True, exist_ok=True)
    if args.policy:
        episodes = evaluate_baseline(args.policy, args.preset, args.episodes, out, not args.no_render)
        (out / "metrics.json").write_text(json.dumps({"episodes": episodes, "summary": summarize(episodes)}, indent=2))
    else:
        algo = args.algo or args.checkpoint.name.split("_seed")[0]
        episodes = evaluate_checkpoint(
            args.checkpoint, algo, args.preset, args.episodes, out, not args.no_render, args.wandb, args.load_step
        )
    print(f"{name} on '{args.preset}', {len(episodes)} episodes (mean ± 95% CI half-width); outputs in {out}")
    for k, (mean, half) in summarize(episodes).items():
        print(f"  {k:28s} {mean:10.3f} ± {half:.3f}")
