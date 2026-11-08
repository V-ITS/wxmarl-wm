"""Exact copies of the simulator state, the one door through which the oracle touches the environment.

A ``WxEnv``'s complete state is a set of tensors plus one CPU ``torch.Generator``. ``take`` copies all of
it, ``restore`` writes a copy into any env (its batch size follows the copy), and ``expand`` repeats each
scenario ``reps`` times so one env can roll ``reps`` candidate plans forward from the same state. The
live env is never rendered into, scored on, or stepped by the planner: every rollout happens in a
scratch env restored from a snapshot (lesson from the Push-T ``_placed`` bug; see ``tests``).

Clairvoyance: ``ReplicatedRng`` draws one random stream per *scenario* and repeats it over the
candidate axis, so a scratch env restored with the live generator state consumes exactly the random
numbers the live env is about to consume, candidate by candidate. With a fresh seed it gives
``futures`` independent samples of the hidden weather process instead.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, fields

import torch
from torch import Tensor

from config import EnvCfg
from core.aircraft import Aircraft
from core.metrics import Metrics
from core.rng import Rng
from core.weather import Cells
from env import WxEnv


class ReplicatedRng(Rng):
    """``Rng`` whose draws have ``reps`` identical copies along the batch axis (layout: scenario-major)."""

    def __init__(self, seed: int, device: str | torch.device, reps: int = 1) -> None:
        super().__init__(seed, device)
        self.reps = reps

    def _rep(self, x: Tensor) -> Tensor:
        return x.repeat_interleave(self.reps, dim=0) if self.reps > 1 else x

    def rand(self, *shape: int) -> Tensor:
        assert shape[0] % self.reps == 0, (shape, self.reps)
        return self._rep(torch.rand((shape[0] // self.reps, *shape[1:]), generator=self.gen)).to(self.device)

    def randn(self, *shape: int) -> Tensor:
        assert shape[0] % self.reps == 0, (shape, self.reps)
        return self._rep(torch.randn((shape[0] // self.reps, *shape[1:]), generator=self.gen)).to(self.device)

    def poisson(self, rate: float | Tensor, *shape: int) -> Tensor:
        assert shape[0] % self.reps == 0, (shape, self.reps)
        size = (shape[0] // self.reps, *shape[1:])
        rates = rate.float().cpu().expand(shape)[:: self.reps].contiguous() if isinstance(rate, Tensor) else torch.full(size, rate)
        base = torch.poisson(rates, generator=self.gen)
        return self._rep(base).long().to(self.device)


@dataclass
class Snapshot:
    t: int
    ac: Aircraft
    cells: Cells
    radar: Cells
    radar_prev: Cells
    sums: dict[str, Tensor]
    los_pairs: Tensor
    los_step: Tensor
    finished: Tensor
    gen_state: Tensor  # CPU generator state (ByteTensor)

    @property
    def B(self) -> int:
        return self.ac.alive.shape[0]

    def map(self, f) -> Snapshot:
        """Apply ``f`` to every batched tensor (not the generator state)."""
        tc = lambda obj: type(obj)(**{x.name: f(getattr(obj, x.name)) for x in fields(obj)})
        return Snapshot(
            self.t, tc(self.ac), tc(self.cells), tc(self.radar), tc(self.radar_prev),
            {k: f(v) for k, v in self.sums.items()}, f(self.los_pairs), f(self.los_step), f(self.finished),
            self.gen_state.clone(),
        )  # fmt: skip


def take(env: WxEnv) -> Snapshot:
    """Deep copy of ``env``'s state."""
    return Snapshot(
        env.t, env.ac, env.cells, env.radar, env.radar_prev, dict(env.metrics.sums), env.los_pairs, env.los_step,
        env.finished, env.rng.gen.get_state(),
    ).map(lambda x: x.clone())  # fmt: skip


def restore(env: WxEnv, snap: Snapshot, rng: Rng | None = None) -> WxEnv:
    """Write ``snap`` into ``env`` (copying, so ``snap`` stays intact); ``env.B`` follows the snapshot."""
    s = snap.map(lambda x: x.clone().to(env.device))
    env.t, env.ac, env.cells, env.radar, env.radar_prev = s.t, s.ac, s.cells, s.radar, s.radar_prev
    env.los_pairs, env.los_step, env.finished = s.los_pairs, s.los_step, s.finished
    env.B = s.B
    env.metrics = Metrics(s.B, env.device)
    env.metrics.sums = defaultdict(lambda: torch.zeros(s.B, device=env.device), s.sums)
    if rng is not None:
        env.rng = rng
    env.rng.gen.set_state(snap.gen_state.clone())
    return env


def expand(snap: Snapshot, reps: int) -> Snapshot:
    """Repeat every scenario ``reps`` times (scenario-major: copy ``j`` of scenario ``b`` is row ``b * reps + j``)."""
    return snap.map(lambda x: x.repeat_interleave(reps, dim=0))


def select(snap: Snapshot, idx: Tensor) -> Snapshot:
    return snap.map(lambda x: x[idx])


class ScratchEnv(WxEnv):
    """A ``WxEnv`` that never builds observations: rollouts only read the reward terms in ``info``.

    The per-heading scan observation is four fifths of a decision step at planner batch sizes and
    the planner never looks at it; dynamics, weather, metrics and the random stream are untouched.
    """

    def _outputs(self):
        return None, None, self.ac.alive.clone(), None


def scratch_env(cfg: EnvCfg, device: str | torch.device = "cpu") -> WxEnv:
    """An env to restore snapshots into; its batch size is set by the first ``restore``."""
    return ScratchEnv(cfg, batch_size=1, device=str(device))


def equal(a: Snapshot, b: Snapshot) -> bool:
    """Bitwise equality of two snapshots (used by the round-trip test)."""
    same = lambda x, y: x.shape == y.shape and bool(torch.equal(x, y))
    tensors = lambda obj: [getattr(obj, x.name) for x in fields(obj)]
    pairs = list(zip(tensors(a.ac), tensors(b.ac))) + list(zip(tensors(a.cells), tensors(b.cells)))
    pairs += list(zip(tensors(a.radar), tensors(b.radar))) + list(zip(tensors(a.radar_prev), tensors(b.radar_prev)))
    pairs += [
        (a.los_pairs, b.los_pairs),
        (a.los_step, b.los_step),
        (a.finished, b.finished),
        (a.gen_state, b.gen_state),
    ]
    pairs += [(a.sums[k], b.sums[k]) for k in set(a.sums) | set(b.sums) if k in a.sums and k in b.sums]
    return a.t == b.t and set(a.sums) == set(b.sums) and all(same(x, y) for x, y in pairs)
