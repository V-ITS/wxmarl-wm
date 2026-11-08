"""The oracle's contract: snapshots round-trip bitwise and the clairvoyant stream replays the live future."""

import torch

from env import WxEnv
from wm.config import load
from wm.snapshot import ReplicatedRng, equal, expand, restore, scratch_env, take


def _env(B=4, N=3, seed=7):
    cfg = load({"n_agents": N}).env_cfg()
    env = WxEnv(cfg, batch_size=B, seed=seed)
    env.reset(seed=seed)
    for _ in range(3):
        env.step(torch.randint(cfg.n_actions, (B, N)))
    return cfg, env


def test_roundtrip_is_bitwise_and_leaves_the_live_env_untouched():
    cfg, live = _env()
    before = take(live)
    scratch = restore(scratch_env(cfg), before)
    for _ in range(5):
        scratch.step(torch.full((live.B, live.N), 2))
    assert equal(take(live), before), "restoring into a scratch env and stepping it must not touch the live env"
    acts = [torch.randint(cfg.n_actions, (live.B, live.N)) for _ in range(6)]
    for a in acts:
        live.step(a)
    after = take(live)
    assert not equal(after, before)
    restore(live, before)
    assert equal(take(live), before)
    for a in acts:
        live.step(a)
    assert equal(take(live), after), "the same actions from a restored state must reproduce the same future bitwise"


def test_expanded_scratch_env_with_replicated_rng_replays_the_live_future():
    cfg, live = _env(B=3)
    snap = take(live)
    reps = 4
    rng = ReplicatedRng(0, "cpu", reps=reps)
    scratch = restore(scratch_env(cfg), expand(snap, reps), rng)
    for _ in range(6):
        a = torch.randint(cfg.n_actions, (live.B, live.N))
        live.step(a)
        scratch.step(a.repeat_interleave(reps, 0))
    for name in ("q", "rho", "I", "alive"):
        x, y = getattr(live.cells, name), getattr(scratch.cells, name)
        assert torch.equal(x.repeat_interleave(reps, 0), y), name
    assert torch.equal(live.ac.p.repeat_interleave(reps, 0), scratch.ac.p)
    assert torch.equal(live.rng.gen.get_state(), scratch.rng.gen.get_state())
