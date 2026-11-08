"""The learned model end to end on CPU: frames, loss, a training step, and planning through it."""

import torch

from env import WxEnv
from wm.config import MOCK, load
from wm.data import Recorder
from wm.jepa import EntityJEPA, WorldModelDynamics, frame_from_snapshot
from wm.planner import Planner
from wm.snapshot import take
from wm.train import split, target_stats, window_index, windows


def _tiny_dataset(cfg, B=4, steps=12):
    env = WxEnv(cfg, batch_size=B)
    env.reset(seed=5)
    rec = Recorder(cfg.domain.T)
    for _ in range(steps):
        snap = take(env)
        a = torch.randint(cfg.n_actions, (B, env.N))
        running = ~env.finished
        env.step(a)
        rec.add(frame_from_snapshot(snap, cfg), a, snap.ac.alive, torch.zeros(B, env.N), running)
    return rec.finish()


def test_loss_decreases_with_a_few_steps():
    diag = load(MOCK, {"n_agents": 3})
    cfg = diag.env_cfg()
    data = _tiny_dataset(cfg)
    model = EntityJEPA(diag.model, cfg.n_actions)
    mean, std = target_stats(data)
    model.tgt_mean.copy_(mean)
    model.tgt_std.copy_(std)
    length = diag.model.history + diag.model.k
    idx = window_index(data["running"], length)
    assert len(idx) > 0
    batch = windows(data, idx, length, "cpu")
    gen = torch.Generator().manual_seed(0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    first = None
    for _ in range(15):
        loss, logs = model.loss(batch, gen)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first or logs["loss"]
    assert logs["loss"] < first


def test_window_index_only_returns_fully_running_windows():
    running = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    idx = window_index(running, 3)
    assert idx.tolist() == [[0, 0], [1, 0], [1, 1], [1, 2]]


def test_split_holds_out_the_last_tenth():
    data = {"running": torch.ones(20, 3, dtype=torch.bool), "x": torch.arange(20)}
    tr, held = split(data)
    assert len(tr["x"]) == 18 and held["x"].tolist() == [18, 19]


def test_planning_through_the_model_returns_legal_actions():
    diag = load(MOCK, {"n_agents": 3, "planner": {"mode": "dec", "assume": "shared", "dynamics": "wm",
                       "candidates": 8, "iters": 1, "horizon": 3}})  # fmt: skip
    cfg = diag.env_cfg()
    model = EntityJEPA(diag.model, cfg.n_actions)
    dyn = WorldModelDynamics(model, diag, cfg, "cpu")
    env = WxEnv(cfg, batch_size=2)
    env.reset(seed=1)
    planner = Planner(diag, cfg, dyn, "cpu")
    planner.reset(2, 0)
    for _ in range(3):
        snap = take(env)
        dyn.observe(snap)
        a = planner.act(snap)
        assert a.shape == (2, 3) and int(a.max()) < cfg.n_actions
        env.step(a)
