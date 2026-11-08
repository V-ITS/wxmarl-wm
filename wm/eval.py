"""Run one condition on the fixed scenario set and write its metrics.

    python -m wm.eval E1-tracker device=cuda   # <out>/05_rq1_forecasting/E1-tracker/metrics.json
    python -m wm.eval E1-tracker --mock        # CPU sizes, <out>/mock/...

Every condition sees the same scenarios: chunk ``c`` is ``WxEnv.reset(scenario_seed0 + c)`` with ``chunk``
parallel episodes, so the weather and routes of a scenario are the same whatever the controller does and
tests can be paired. A row cannot be flown alone (the draws depend on the batch), but chunks are independent:
each is cached in ``chunks/``, so an interrupted run resumes and a run can be split across machines.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import torch

from baselines import POLICIES
from env import WxEnv
from wm import experiments
from wm.config import DiagCfg, parse_overrides
from wm.oracle import Oracle
from wm.planner import Planner
from wm.snapshot import take


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def make_dynamics(cfg: DiagCfg, env_cfg, device):
    if cfg.planner.dynamics == "wm":
        from wm.jepa import WorldModelDynamics

        return WorldModelDynamics.load(cfg, env_cfg, device)
    return Oracle(cfg, env_cfg, device)


def run_chunk(cfg: DiagCfg, env_cfg, chunk_seed: int, B: int, device, log) -> list[dict]:
    """Simulate one chunk of scenarios: the env's episode metrics, plus each aircraft's turns and delay."""
    live = WxEnv(env_cfg, batch_size=B, device=str(device))
    _, _, _, avail = live.reset(seed=chunk_seed)
    if cfg.policy == "marl":  # a trained EPyMARL agent (wm/marl.py), loaded per chunk to reset its GRU state
        from wm.marl import MarlPolicy

        policy = MarlPolicy(cfg.checkpoint, live)
    elif cfg.policy:
        policy = POLICIES[cfg.policy]
    else:
        dyn = make_dynamics(cfg, env_cfg, device)
        planner = Planner(cfg, env_cfg, dyn, device)
        planner.reset(B, cfg.planner_seed * 1_000_003 + chunk_seed)
    gen = torch.Generator().manual_seed(chunk_seed)  # scripted policies' own randomness
    offs = torch.tensor(env_cfg.action.heading_offsets_deg, device=device).abs()
    turns, turn_deg = torch.zeros_like(live.ac.p[..., 0]), torch.zeros_like(live.ac.p[..., 0])
    info: dict = {}
    t0 = time.time()
    while "episode" not in info:
        if cfg.policy:
            actions = policy(live, avail, gen)
        else:
            snap = take(live)
            if hasattr(dyn, "observe"):
                dyn.observe(snap)
            actions = planner.act(snap)
        turning = live.ac.alive & (actions != env_cfg.nominal_action)
        turns += turning.float()
        turn_deg += turning * offs[actions // len(env_cfg.action.speed_deltas_kt)]
        *_, avail, info = live.step(actions)
        if live.t % 10 == 0:
            log(f"  t={live.t:3d} {time.time() - t0:6.1f}s")
    ac, nm_per_step = live.ac, env_cfg.aircraft.v_nom * env_cfg.domain.substep / 3600.0
    to_circle = ((ac.e - ac.p0).norm(dim=-1) - env_cfg.traffic.r_exit).clamp_min(0.0)
    delay = ac.flight_time - torch.ceil(to_circle / nm_per_step) * env_cfg.domain.substep
    ep, out = info["episode"], []
    for b in range(B):
        row = {"chunk_seed": chunk_seed, "row": b, **{k: float(v[b]) for k, v in ep.items()}}
        row["flight_hours"] = row["mean_active_aircraft"] * row["ep_length"] * env_cfg.domain.dt / 3600.0
        row |= {"turns": turns[b].tolist(), "turn_deg": turn_deg[b].tolist(), "delay_by_aircraft": delay[b].tolist()}
        out.append(row)
    return out


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_id", choices=sorted(experiments.REGISTRY))
    ap.add_argument("--mock", action="store_true", help="CPU test sizes; writes under results/redesign/mock")
    ap.add_argument("overrides", nargs="*", help="a.b=value")
    args = ap.parse_intermixed_args(argv)
    exp, cfg = experiments.resolve(args.run_id, parse_overrides(args.overrides), args.mock)
    assert cfg.scenarios % cfg.chunk == 0, "scenarios must be a multiple of chunk (fixed chunking keeps conditions paired)"
    env_cfg, device = cfg.env_cfg(), torch.device(cfg.device)
    out = experiments.run_dir(exp, args.run_id, cfg)
    (out / "chunks").mkdir(parents=True, exist_ok=True)
    log = lambda s: print(s, flush=True)
    log(f"{args.run_id}: {cfg.scenarios} scenarios x N={cfg.n_agents}, {cfg.planner.mode}/{cfg.planner.assume}, {cfg.oracle.info}, policy={cfg.policy or '-'} -> {out}")
    t0, scenarios = time.time(), []
    same = hashlib.sha256(json.dumps({**cfg.to_dict(), "scenarios": 0}, sort_keys=True).encode()).hexdigest()[:10]  # more scenarios reuse these chunks
    for c in range(cfg.scenarios // cfg.chunk):
        seed, path = cfg.scenario_seed0 + c, out / "chunks" / f"{cfg.scenario_seed0 + c}-{same}.json"
        if not path.exists():
            log(f"chunk {c} (seed {seed})")
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(run_chunk(cfg, env_cfg, seed, cfg.chunk, device, log)))
            tmp.rename(path)
        scenarios += json.loads(path.read_text())
    result = {"run_id": args.run_id, "exp": exp, "cfg": cfg.to_dict(), "cfg_hash": cfg.hash(), "env_cfg_hash": env_cfg.hash(),
              "git": git_commit(), "seconds": round(time.time() - t0, 1), "scenarios": scenarios}  # fmt: skip
    (out / "metrics.json").write_text(json.dumps(result, indent=1))
    from wm.stats import headline

    log(headline(result))
    return out


if __name__ == "__main__":
    main()
