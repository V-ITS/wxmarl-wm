"""Launch EPyMARL with the WxMARL env registered, without editing EPyMARL.

Usage:
    python -m wrappers.epymarl_main --config=<algo> [--eval] with key=value ...

Mirrors the ``__main__`` block of ``third_party/epymarl/src/main.py``, but reads the env and
algorithm overrides from ``configs/epymarl/`` and writes results to ``results/`` in this repo.
Config merge order: EPyMARL default -> configs/epymarl/env.yaml -> EPyMARL algs/<algo>.yaml
-> configs/epymarl/<algo>.yaml -> (configs/epymarl/eval.yaml) -> command line.
"""

from __future__ import annotations

import os
import subprocess
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EPYMARL = ROOT / "third_party" / "epymarl"
CONFIGS = ROOT / "configs" / "epymarl"
for path in (str(ROOT), str(EPYMARL / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)
# EPyMARL's `envs` package imports smaclite unconditionally. It is not needed here, so an empty
# stand-in avoids installing it. This runs at import time so spawned env workers get it too.
sys.modules.setdefault("smaclite", types.ModuleType("smaclite"))
# One BLAS/OpenMP thread per process, as EPyMARL sets for the main process; the parallel runner's
# env workers would otherwise oversubscribe the CPU.
os.environ.setdefault("OMP_NUM_THREADS", "1")


def git_commit(repo: Path) -> str:
    """HEAD hash of ``repo``, with a ``-dirty`` suffix if tracked files are modified; "unknown" outside git."""
    run = lambda *a: subprocess.run(
        ("git", "-C", str(repo), *a), check=True, capture_output=True, text=True
    ).stdout.strip()
    try:
        return run("rev-parse", "HEAD") + ("-dirty" if run("status", "--porcelain", "--untracked-files=no") else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main(argv: list[str]) -> None:
    import main as epymarl_main  # third_party/epymarl/src/main.py
    import torch as th
    import yaml
    from envs import REGISTRY
    from modules.agents import REGISTRY as AGENTS
    from sacred.observers import FileStorageObserver

    from wrappers.cnn_agent import CNNAgent
    from wrappers.epymarl_env import make_env

    REGISTRY["wxmarl"] = make_env
    AGENTS["wx_cnn"] = CNNAgent
    params = list(argv)
    th.set_num_threads(1)  # as in EPyMARL's main.py

    algo = next(p.split("=", 1)[1] for p in params if p.startswith("--config="))
    evaluate = "--eval" in params
    params = [p for p in params if not p.startswith("--config=") and p != "--eval"]

    load = lambda path: yaml.safe_load(path.read_text()) or {}
    layers = [
        EPYMARL / "src" / "config" / "default.yaml",
        CONFIGS / "env.yaml",
        EPYMARL / "src" / "config" / "algs" / f"{algo}.yaml",
        CONFIGS / f"{algo}.yaml",
    ] + ([CONFIGS / "eval.yaml"] if evaluate else [])
    config: dict = {}
    for layer in layers:
        config = epymarl_main.recursive_dict_update(config, load(layer))
    config["local_results_path"] = str(ROOT / "results")
    config["env_args"] |= {
        "algorithm": algo,
        "repo_commit": git_commit(ROOT),
        "epymarl_commit": git_commit(EPYMARL),
    }
    epymarl_main.ex.add_config(config)

    preset = config["env_args"]["map_name"]
    for p in params:
        if p.startswith("env_args.map_name="):
            preset = p.split("=", 1)[1]
    sacred_dir = ROOT / "results" / "sacred" / config["name"] / preset
    epymarl_main.ex.observers.append(FileStorageObserver.create(str(sacred_dir)))
    epymarl_main.ex.run_commandline(params)


if __name__ == "__main__":
    main(sys.argv)
