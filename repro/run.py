"""Run jobs on this machine, each after the jobs it needs. Finished jobs are skipped, so a restart continues where it
stopped, and a run in progress resumes from its cached chunks.

    python -m repro.run all                            # every job behind the paper
    python -m repro.run E1-tracker BASE-avoid          # some jobs, plus whatever they need that is not done
    python -m repro.run all --gpus 0,1 --per-gpu 3 --cpu 4
    python -m repro.run all --mock                     # CPU smoke test of every job, under results/redesign/mock

Logs go to <results>/logs/<job>.log and done markers to <results>/done/<job>.json. Then: python -m repro.report.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

from repro.jobs import ROOT, job, names, results_root
from wm.eval import git_commit


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jobs", nargs="+", help='job names (python -m repro.jobs lists them), or "all"')
    ap.add_argument("--gpus", default="0", help="GPU indices, e.g. 0,1")
    ap.add_argument("--per-gpu", type=int, default=1, help="jobs at a time on each GPU")
    ap.add_argument("--cpu", type=int, default=1, help="CPU jobs at a time")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads per job")
    ap.add_argument("--device", default="cuda", help="torch device of the planner runs (cuda or cpu)")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print what would run")
    args = ap.parse_args(argv)
    order = names()
    unknown = set(args.jobs) - {*order, "all"}
    if unknown:
        ap.error(f"unknown jobs: {sorted(unknown)}")
    root = results_root(args.mock)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "done").mkdir(parents=True, exist_ok=True)
    done = {p.stem for p in (root / "done").glob("*.json")}
    jobs = {n: job(n, args.mock, args.device) for n in order}
    want, stack = set(), list(order if "all" in args.jobs else args.jobs)
    while stack:  # the requested jobs and everything they need that is not done
        n = stack.pop()
        if n not in want and n not in done:
            want.add(n)
            stack += jobs[n].needs
    todo = [n for n in order if n in want]
    say = lambda s: print(f"{time.strftime('%m-%d %H:%M')} {s}", flush=True)
    say(f"{len(todo)} jobs to run, {len(done)} already done")
    if args.dry_run:
        for n in todo:
            print(" ", n, " ".join(jobs[n].cmd[1:]))
        return
    gpus = args.gpus.split(",")
    slots = {"gpu": len(gpus) * args.per_gpu, "cpu": args.cpu}
    if min(slots.values()) < 1:
        ap.error("--per-gpu and --cpu must be at least 1")
    env = os.environ | {"PYTHONPATH": str(ROOT), "OMP_NUM_THREADS": str(args.threads), "MKL_NUM_THREADS": str(args.threads), "WANDB_MODE": "disabled"}
    running: dict[str, tuple] = {}
    failed: set[str] = set()
    while todo or running:
        for n, (p, t0, gpu) in list(running.items()):
            if p.poll() is None:
                continue
            del running[n]
            if p.returncode == 0:
                marker = {"job": n, "seconds": round(time.time() - t0), "git": git_commit(), "time": time.strftime("%Y-%m-%d %H:%M")}
                (root / "done" / f"{n}.json").write_text(json.dumps(marker))
                done.add(n)
                say(f"done {n} after {marker['seconds']} s")
            else:
                failed.add(n)
                say(f"FAILED {n} (exit {p.returncode}): see {root / 'logs' / (n + '.log')}")
        for n in [n for n in todo if any(d in failed for d in jobs[n].needs)]:
            todo.remove(n)
            failed.add(n)
            say(f"BLOCKED {n}: a job it needs failed")
        for n in list(todo):
            j = jobs[n]
            if any(d not in done for d in j.needs) or sum(jobs[r].pool == j.pool for r in running) >= slots[j.pool]:
                continue
            load = {g: sum(r[2] == g for r in running.values()) for g in gpus}
            gpu = min(gpus, key=load.get)
            log = (root / "logs" / f"{n}.log").open("a")
            p = subprocess.Popen(j.cmd, cwd=ROOT, env=env | {"CUDA_VISIBLE_DEVICES": gpu}, stdout=log, stderr=subprocess.STDOUT)
            running[n] = (p, time.time(), gpu)
            todo.remove(n)
            say(f"start {n} ({j.pool}, GPU {gpu})")
        time.sleep(2)
    say(f"finished; failed: {sorted(failed) or 'none'}")


if __name__ == "__main__":
    main()
