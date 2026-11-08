"""Train one entity-token world model and select its checkpoint by closed-loop safety.

    python -m wm.train E4-s0               # results/redesign/E4/E4-s0/{ckpt.pt, train.json}
    python -m wm.train E4-joint-s1 --mock

The dataset is ``results/redesign/data/<train.data>.pt``; its last tenth of episodes is held out (the
audits' reference for prediction error under the training policy). Every ``val_every`` epochs the
current model plans closed-loop on ``val_scenarios`` validation scenarios (seeds disjoint from
training data and from the evaluation scenarios), and the checkpoint with the fewest losses of
separation per flight-hour (ties: least time in weather) is kept. Prediction loss is logged, never
used for selection (the objective-mismatch lesson).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import Tensor

from wm import experiments
from wm.config import DiagCfg, load, parse_overrides
from wm.data import dataset_path, load_dataset
from wm.jepa import EntityJEPA

VAL_SEED0 = 300_000


def split(data: dict, holdout: float = 0.1) -> tuple[dict, dict]:
    E = data["running"].shape[0]
    n_val = max(1, round(holdout * E))
    sl = lambda a, b: {k: v[a:b] for k, v in data.items() if isinstance(v, Tensor)}
    return sl(0, E - n_val), sl(E - n_val, E)


def window_index(running: Tensor, length: int) -> Tensor:
    """[M, 2] (episode, start) of every window of ``length`` running steps."""
    T = running.shape[1]
    ok = running.clone()
    for s in range(1, length):
        ok[:, : T - s] &= running[:, s:]
    ok[:, T - length + 1 :] = False
    return ok.nonzero()


def windows(data: dict, idx: Tensor, length: int, device) -> dict[str, Tensor]:
    e, t = idx[:, 0], idx[:, 1]
    ar = torch.arange(length)
    pick = lambda k: data[k][e[:, None], t[:, None] + ar[None]]
    out = {k: pick(k).to(device) for k in ("air_in", "cell_in", "air_tgt", "cell_tgt")}
    out["actions"] = pick("actions")[:, : length - 1].to(device)
    out["alive"] = data["alive"][e, t].to(device)
    return out


def target_stats(data: dict) -> tuple[Tensor, Tensor]:
    """Per-feature mean and std of the privileged targets over running frames (std floored at 0.05)."""
    m = data["running"]
    air = data["air_tgt"][m].reshape(-1, data["air_tgt"].shape[-1])
    cell = data["cell_tgt"][m].reshape(-1, data["cell_tgt"].shape[-1])
    mean = torch.cat((air.mean(0), cell.mean(0)))
    std = torch.cat((air.std(0), cell.std(0))).clamp_min(0.05)
    return mean, std


def validate(model: EntityJEPA, cfg: DiagCfg, env_cfg, device, tmp: Path, log) -> dict[str, float]:
    """Closed-loop safety of the model as the planner's dynamics on the validation scenarios."""
    from wm.eval import run_chunk
    from wm.stats import summary

    torch.save({"model_cfg": model.m, "state_dict": model.state_dict()}, tmp)
    val = load(cfg.to_dict(), {"checkpoint": str(tmp), "policy": "", "planner": {"dynamics": "wm", "mode": "dec", "assume": "shared"}})
    B = min(cfg.train.val_scenarios, cfg.chunk)
    rows = []
    for c in range(max(1, cfg.train.val_scenarios // B)):
        rows += run_chunk(val, env_cfg, VAL_SEED0 + c, B, device, lambda s: None)
    s = summary({"run_id": "val", "scenarios": rows})
    return {"M1": s["los_per_fh"][0], "M3": s["storm_s"][0], "exit_rate": s["exit_rate"][0]}


def train(run_id: str, cfg: DiagCfg, out: Path, log=print) -> Path:
    env_cfg = cfg.env_cfg()
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.train.seed)
    gen = torch.Generator().manual_seed(cfg.train.seed)
    data = load_dataset(dataset_path(cfg, cfg.train.data))
    train_set, _ = split(data)  # the held-out tenth is the audits' reference
    model = EntityJEPA(cfg.model, env_cfg.n_actions).to(device)
    mean, std = target_stats(train_set)
    model.tgt_mean.copy_(mean.to(device))
    model.tgt_std.copy_(std.to(device))
    length = cfg.model.history + cfg.model.k
    idx = window_index(train_set["running"], length)
    log(
        f"{run_id}: {len(train_set['running'])} episodes, {len(idx)} windows of {length}, device={device}"
    )
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    steps_per_epoch = max(1, len(idx) // cfg.train.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.train.epochs * steps_per_epoch)
    history, best, best_key = [], None, None
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        perm = idx[torch.randperm(len(idx), generator=gen)]
        logs: dict[str, float] = {}
        for i in range(steps_per_epoch):
            batch = windows(train_set, perm[i * cfg.train.batch : (i + 1) * cfg.train.batch], length, device)
            loss, l = model.loss(batch, gen)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            logs = {k: logs.get(k, 0.0) + v / steps_per_epoch for k, v in l.items()}
        rec = {"epoch": epoch, **logs, "seconds": round(time.time() - t0)}
        if epoch % cfg.train.val_every == 0 or epoch == cfg.train.epochs:
            model.eval()
            rec["val"] = validate(model, cfg, env_cfg, device, out / "ckpt_last.pt", log)
            key = (rec["val"]["M1"], rec["val"]["M3"])
            if best_key is None or key < best_key:
                best_key, best = key, epoch
                torch.save({"model_cfg": model.m, "state_dict": model.state_dict(), "epoch": epoch, "val": rec["val"],
                            "run_id": run_id, "cfg": cfg.to_dict()}, out / "ckpt.pt")  # fmt: skip
        history.append(rec)
        log(json.dumps(rec))
        (out / "train.json").write_text(
            json.dumps({"run_id": run_id, "cfg": cfg.to_dict(), "best_epoch": best, "history": history}, indent=1)
        )
    log(f"best epoch {best} by closed-loop safety {best_key}; {out / 'ckpt.pt'}")
    return out / "ckpt.pt"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "run_id",
        choices=sorted(
            k for k, v in experiments.REGISTRY.items() if v["exp"] == "E4" and "checkpoint" not in v
        ),
    )
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_intermixed_args(argv)
    exp, cfg = experiments.resolve(args.run_id, parse_overrides(args.overrides), args.mock)
    train(args.run_id, cfg, experiments.run_dir(exp, args.run_id, cfg))


if __name__ == "__main__":
    main()
