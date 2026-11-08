"""Screen the scenarios of one weather setting by route impact with ``core.impact.screen``.

    python -m wm.suite SUITE-N5 device=cuda   # <out>/01_scenarios/SUITE-N5/suite.json

No controller flies here. Scenario rows are identified by (chunk seed, row) as in ``wm.eval``; ``wm.stats``
joins them to any run of the same env hash. ``w_class`` counts direct tracks entering the envelope and is set for
N = 5; at other N use ``tracks_envelope / flights``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.impact import screen
from wm import experiments
from wm.config import parse_overrides


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_id", choices=[k for k, v in experiments.REGISTRY.items() if v["exp"] == "SUITE"])
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("overrides", nargs="*", help="a.b=value")
    args = ap.parse_intermixed_args(argv)
    exp, cfg = experiments.resolve(args.run_id, parse_overrides(args.overrides), args.mock)
    env_cfg = cfg.env_cfg()
    rows = []
    for c in range(cfg.scenarios // cfg.chunk):
        seed = cfg.scenario_seed0 + c
        s = screen(env_cfg, seed, cfg.chunk, cfg.device)
        for b in range(cfg.chunk):
            rows.append({"chunk_seed": seed, "row": b, **{k: v[b].tolist() for k, v in s.items()}})
    out = experiments.run_dir(exp, args.run_id, cfg)
    out.mkdir(parents=True, exist_ok=True)
    (out / "suite.json").write_text(json.dumps({"run_id": args.run_id, "env_cfg_hash": env_cfg.hash(), "scenarios": rows}))
    print(f"{args.run_id}: {len(rows)} scenarios, W0-W3 " + str([sum(r["w_class"] == w for r in rows) for w in range(4)]), flush=True)
    return out


if __name__ == "__main__":
    main()
