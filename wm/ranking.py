"""Can each predictor tell good plans from bad? Plan-ranking agreement, and the end-to-end model's state error.

    python -m wm.ranking RANK               # <out>/09_attribution/RANK/ranking.json

One set of candidate plans per moment, under avoid traffic: aircraft 0 varies its plan, the others hold their
heading. Every scorer is compared with the clairvoyant oracle by Kendall tau. ``state_error_nm`` is how far the
end-to-end model's decoded aircraft already are from the truth before any prediction, which bounds what its
ranking can mean: a weak model may be weak because of that, not because learning everything is wrong.
"""

from __future__ import annotations

import argparse
import copy
import json

import numpy as np
import torch
from scipy import stats as sps

from baselines import POLICIES
from core.weather import edge_distance
from env import WxEnv
from wm import experiments
from wm.config import parse_overrides
from wm.jepa import WorldModelDynamics
from wm.oracle import Oracle
from wm.planner import Planner
from wm.snapshot import take

SEED, N_PLANS, TIMES = 600_000, 64, range(4, 64, 6)


def ranking(cfg, scenarios: int, times=TIMES) -> dict:
    env_cfg = cfg.env_cfg()
    cfg.planner.candidates, cfg.planner.iters = 64, 6

    def oracle(**kw) -> Oracle:
        c = copy.deepcopy(cfg)
        c.oracle.__dict__.update({"weather_ckpt": "", "belief_speed": True, "futures": 4, **kw})
        return Oracle(c, env_cfg, "cpu")

    wx = lambda run: f"{cfg.out}/{experiments.PACKAGES['WX']}/{run}/ckpt.pt"
    truth = oracle(info="clairvoyant")
    scorers = {"frozen": oracle(info="gstale", belief_speed=False), "extrap": oracle(info="gstale"), "tracker": oracle(info="gpool"),
               "forecast": oracle(info="learned", weather_ckpt=wx("WX-sector-s0")), "perfect": oracle(info="perfect")}  # fmt: skip
    dyn = WorldModelDynamics.load(cfg, env_cfg, "cpu") if cfg.checkpoint else None
    planner = Planner(cfg, env_cfg, truth, "cpu")
    planner.gen.manual_seed(SEED)
    env = WxEnv(env_cfg, batch_size=scenarios, seed=SEED)
    _, _, _, avail = env.reset(seed=SEED)
    H, N, rows, state_err, info = planner.H, planner.N, [], [], {}
    while "episode" not in info:
        snap = take(env)
        if dyn is not None:
            dyn.observe(snap)
        if env.t in times:
            cand = planner._sample(planner._init_probs(snap.B, 1, 1, None), N_PLANS)[:, 0, :, :, 0]
            acts = torch.full((snap.B, N_PLANS, H, N), env_cfg.nominal_action, dtype=torch.long)
            acts[..., 0] = cand
            flat = acts.reshape(-1, H, N)
            costs = {"truth": truth.evaluate(snap, flat, 1, N_PLANS).sum((-1, -2)).view(snap.B, N_PLANS)}
            if dyn is not None:
                costs["end-to-end"] = dyn.evaluate(snap, flat, 1, N_PLANS).sum((-1, -2)).view(snap.B, N_PLANS)
                air, _ = dyn.model.decode(dyn.z_hist[:, -1], N)
                err = (air[..., :2] * env_cfg.domain.L - snap.ac.p).norm(dim=-1)
                state_err.append(float(err[snap.ac.alive].mean()))
            for k, o in scorers.items():
                rep = flat.view(snap.B, 1, N_PLANS, H, N).expand(snap.B, o.futures, N_PLANS, H, N).reshape(-1, H, N)
                costs[k] = o.evaluate(snap, rep, 1, N_PLANS, seed=SEED).sum((-1, -2)).mean(1).view(snap.B, N_PLANS)
            near = edge_distance(snap.ac.p, snap.cells)[:, 0] <= cfg.wx_near_nm
            for b in (snap.ac.alive[:, 0] & ~snap.finished).nonzero().flatten().tolist():
                o = costs["truth"][b].numpy()
                if float(np.median(o) - o.min()) >= 1e-6:
                    rows.append({"t": env.t, "b": b, "near_weather": bool(near[b]),
                                 "tau": {k: float(sps.kendalltau(v[b].numpy(), o).statistic) for k, v in costs.items() if k != "truth"}})  # fmt: skip
        *_, avail, info = env.step(POLICIES["avoid"](env, avail, None))
    near = [r for r in rows if r["near_weather"]]
    tau = lambda sel: {k: float(np.nanmean([r["tau"][k] for r in sel])) for k in (rows[0]["tau"] if rows else ())}
    return {"moments": len(rows), "near_weather_moments": len(near), "tau_near": tau(near), "tau_all": tau(rows),
            "state_error_nm": float(np.mean(state_err)) if state_err else None, "rows": rows}  # fmt: skip


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_id", choices=["RANK"])
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_intermixed_args(argv)
    exp, cfg = experiments.resolve(args.run_id, parse_overrides(args.overrides), args.mock)
    res = ranking(cfg, 2, range(14, 30, 10)) if args.mock else ranking(cfg, 64)
    out = experiments.run_dir(exp, args.run_id, cfg)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ranking.json").write_text(json.dumps(res, indent=1))
    print({k: v for k, v in res.items() if k != "rows"}, flush=True)


if __name__ == "__main__":
    main()
