"""Train QMIX with EPyMARL (CNN agent on the grid observation) and keep the last checkpoint at <out>/03_qmix/qmix-s<seed>.

    python -m repro.train_qmix 0

With t_max = 400,000 and a checkpoint every 100,000 steps, training ends just before the fourth save, so the last
checkpoint, the one the paper evaluates, is at about 0.3 M steps.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

from repro.jobs import ROOT, results_root
from wm.experiments import PACKAGES

ARGS = "t_max=400000 agent=wx_cnn buffer_size=500"
MOCK_ARGS = "t_max=2000 agent=wx_cnn use_cuda=False buffer_size=16 test_interval=1000 save_model_interval=1000"


def main(seed: str, mock: bool) -> None:
    env = os.environ | {"WANDB_MODE": "disabled", "RUN_TAG": f"s{seed}"}
    subprocess.run(["bash", "scripts/train.sh", "qmix", seed, "grid", *(MOCK_ARGS if mock else ARGS).split()], cwd=ROOT, env=env, check=True)
    runs = sorted((ROOT / "results" / "models").glob(f"qmix_seed{seed}_grid*"), key=lambda p: p.stat().st_mtime)
    last = max(runs[-1].iterdir(), key=lambda p: int(p.name))
    out = results_root(mock) / PACKAGES["QMIX"] / f"qmix-s{seed}"
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(last, out)
    print(f"{last} -> {out}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], "--mock" in sys.argv)
