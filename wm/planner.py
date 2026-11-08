"""One planner for every condition: cross-entropy method over discrete heading sequences.

The population is categorical: a distribution over the action set at each horizon step for each
controlled aircraft. Candidates are sampled with persistence (a heading is kept from one step to the
next with probability ``persist``), the discrete analogue of temporally correlated proposals. The
elite fraction re-estimates the distribution. Nothing else differs between conditions:

  joint  one population over A^N (the centralized reference; a ground system with full surveillance)
  dec    one population per aircraft over its own A, the others fixed by an assumption:
           shared   the others' latest plans, exchanged for ``rounds`` best-response rounds
           direct   the others fly to their exit
           priority a rule for who turns: aircraft plan one after another in index order, each against
                    the plans just made by those ahead of it and the previous plans of those behind

``Dynamics`` is whatever scores candidates: the true-simulator ``Oracle`` or a learned world model
(``wm.jepa.Dynamics``). The planner never touches the live env; it reads a ``Snapshot``.
"""

from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor

from config import EnvCfg
from wm.config import DiagCfg
from wm.snapshot import Snapshot


class Dynamics(Protocol):
    futures: int

    def evaluate(
        self, snap: Snapshot, actions: Tensor, P: int, n: int, direct_mask: Tensor | None = None,
        slots: Tensor | None = None, seed: int | None = None,
    ) -> Tensor: ...  # fmt: skip


class Planner:
    def __init__(self, diag: DiagCfg, env_cfg: EnvCfg, dynamics: Dynamics, device: str | torch.device = "cpu") -> None:
        self.p, self.cfg, self.dyn, self.device = diag.planner, env_cfg, dynamics, torch.device(device)
        self.N, self.A, self.H = env_cfg.domain.n_agents_max, env_cfg.n_actions, self.p.horizon
        self.gen = torch.Generator().manual_seed(diag.planner_seed)
        self.plan: Tensor | None = None  # [B, H, N] the last executed plan (warm start)
        assert self.p.mode in ("joint", "dec") and self.p.assume in ("shared", "direct", "priority")

    def reset(self, B: int, seed: int) -> None:
        self.gen.manual_seed(seed)
        self.plan = torch.full((B, self.H, self.N), self.cfg.nominal_action, dtype=torch.long, device=self.device)

    # ---------- the cross-entropy method ----------
    def _init_probs(self, B: int, P: int, M: int, warm: Tensor | None) -> Tensor:
        """[B, P, H, M, A]: mass ``init_nominal`` on the nominal action (or the warm-started plan), rest uniform."""
        probs = torch.full((B, P, self.H, M, self.A), (1 - self.p.init_nominal) / self.A, device=self.device)
        if warm is None:
            probs[..., self.cfg.nominal_action] += self.p.init_nominal
        else:
            probs.scatter_add_(
                -1, warm.unsqueeze(-1), torch.full_like(warm.unsqueeze(-1), self.p.init_nominal, dtype=probs.dtype)
            )
        return probs

    def _sample(self, probs: Tensor, n: int) -> Tensor:
        """Candidates [B, P, n, H, M] with persistence."""
        B, P, H, M, A = probs.shape
        flat = probs.permute(0, 1, 3, 2, 4).reshape(-1, A)  # [(B P M H), A]
        draws = torch.multinomial(flat.cpu(), n, replacement=True, generator=self.gen).to(self.device)
        draws = draws.view(B, P, M, H, n).permute(0, 1, 4, 3, 2)  # [B, P, n, H, M]
        keep = torch.rand(B, P, n, H, M, generator=self.gen).to(self.device) < self.p.persist
        out = draws.clone()
        for h in range(1, H):
            out[:, :, :, h] = torch.where(keep[:, :, :, h], out[:, :, :, h - 1], draws[:, :, :, h])
        return out

    def _cem(
        self, snap: Snapshot, P: int, M: int, warm: Tensor | None, assemble, slots: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        """Returns the best sequence per population [B, P, H, M] and its mean cost [B, P]."""
        B, n = snap.B, self.p.candidates
        F = self.dyn.futures
        n_elite = max(1, round(self.p.elite_frac * n))
        disc = self.p.gamma ** torch.arange(self.H, device=self.device)
        probs = self._init_probs(B, P, M, warm)
        seed = int(torch.randint(1 << 30, (1,), generator=self.gen))
        best, best_cost = None, None
        for _ in range(self.p.iters):
            cand = self._sample(probs, n)  # [B, P, n, H, M]
            joint, direct_mask = assemble(cand)  # [B, F, P, n, H, N], [B*F*P*n, N] | None
            cost = self.dyn.evaluate(snap, joint.reshape(B * F * P * n, self.H, self.N), P, n, direct_mask, slots, seed)
            team = (cost.sum(-1) * disc).sum(-1).mean(1)  # [B, P, n]: team cost over the horizon, mean over futures
            idx = team.argsort(dim=-1)[..., :n_elite]  # [B, P, n_elite]
            elite = torch.gather(cand, 2, idx[..., None, None].expand(-1, -1, -1, self.H, M))
            freq = torch.zeros_like(probs).scatter_add_(
                -1, elite.permute(0, 1, 3, 4, 2), torch.ones(B, P, self.H, M, n_elite, device=self.device)
            )
            probs = (1 - self.p.alpha) * probs + self.p.alpha * freq / n_elite
            top = idx[..., 0]
            top_cost = torch.gather(team, 2, top.unsqueeze(-1)).squeeze(-1)
            top_seq = torch.gather(cand, 2, top[..., None, None, None].expand(-1, -1, 1, self.H, M)).squeeze(2)
            if best is None:
                best, best_cost = top_seq, top_cost
            else:
                better = top_cost < best_cost
                best = torch.where(better[..., None, None], top_seq, best)
                best_cost = torch.where(better, top_cost, best_cost)
        return best, best_cost

    # ---------- assembling joint actions ----------
    def _others(self, snap: Snapshot, plans: Tensor) -> tuple[Tensor, Tensor | None]:
        """Assumed actions of every aircraft [B, H, N] and the direct-policy mask [B, N] (or None).

        ``direct`` (fly to the exit) depends on the state along the rollout: a mask the dynamics resolve each step.
        """
        if self.p.assume == "shared":
            return plans, None
        return plans, torch.ones(snap.B, self.N, dtype=torch.bool, device=self.device)

    def joint_plan(self, snap: Snapshot, warm: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Joint mode: best joint sequence [B, H, N] and its cost [B]."""
        F = self.dyn.futures

        def assemble(cand: Tensor):  # [B, 1, n, H, N]
            return cand.unsqueeze(1).expand(-1, F, -1, -1, -1, -1), None

        best, cost = self._cem(snap, 1, self.N, None if warm is None else warm.unsqueeze(1), assemble, None)
        return best[:, 0], cost[:, 0]

    def best_responses(
        self, snap: Snapshot, others: Tensor, direct_mask: Tensor | None, warm: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Decentralized mode, one round: each aircraft's best own sequence given ``others`` [B, H, N].

        Returns the plan [B, H, N] with every aircraft's own slot filled by its response, and costs [B, N].
        """
        B, F, N = snap.B, self.dyn.futures, self.N
        slots = torch.arange(N, device=self.device)
        eye = torch.eye(N, dtype=torch.bool, device=self.device)

        def assemble(cand: Tensor):  # [B, N, n, H, 1]
            n = cand.shape[2]
            base = others[:, None, None].expand(-1, N, n, -1, -1)  # [B, P=N, n, H, N]
            joint = torch.where(eye[None, :, None, None, :], cand.expand(-1, -1, -1, -1, N), base)
            joint = joint.unsqueeze(1).expand(-1, F, -1, -1, -1, -1)
            mask = None
            if direct_mask is not None:
                mask = (direct_mask[:, None, :] & ~eye[None]).unsqueeze(1).expand(-1, F, -1, -1)  # [B, F, P, N]
                mask = mask[:, :, :, None, :].expand(-1, -1, -1, n, -1).reshape(B * F * N * n, N)
            return joint, mask

        w = None if warm is None else warm.permute(0, 2, 1).unsqueeze(-1)  # [B, P=N, H, 1]
        best, cost = self._cem(snap, N, 1, w, assemble, slots)  # [B, N, H, 1]
        return best[..., 0].permute(0, 2, 1), cost

    def priority_plan(self, snap: Snapshot, warm: Tensor) -> Tensor:
        """Decentralized mode with a fixed order: aircraft ``i`` plans its own sequence against the plan [B, H, N]
        as it stands, with aircraft ``0 .. i-1`` already replanned this step; nobody revises after a later one."""
        F, plan = self.dyn.futures, warm.clone()
        for i in range(self.N):

            def assemble(cand: Tensor, i: int = i):  # [B, 1, n, H, 1]
                joint = plan[:, None, None].expand(-1, 1, cand.shape[2], -1, -1).clone()
                joint[..., i] = cand[..., 0]
                return joint.unsqueeze(1).expand(-1, F, -1, -1, -1, -1), None

            slot = torch.tensor([i], device=self.device)
            best, _ = self._cem(snap, 1, 1, warm[:, None, :, i : i + 1], assemble, slot)
            plan[..., i] = best[:, 0, :, 0]
        return plan

    def act(self, snap: Snapshot) -> Tensor:
        """Joint actions [B, N] to execute now; keeps the plan for the next warm start."""
        warm = torch.cat((self.plan[:, 1:], self.plan[:, -1:]), 1)  # the last plan, shifted one step
        if self.p.mode == "joint":
            plan, _ = self.joint_plan(snap, warm)
        elif self.p.assume == "priority":
            plan = self.priority_plan(snap, warm)
        else:
            plan = warm
            rounds = self.p.rounds if self.p.assume == "shared" else 1
            for _ in range(rounds):
                others, direct_mask = self._others(snap, plan)
                plan, _ = self.best_responses(snap, others, direct_mask, warm)
        self.plan = plan
        return plan[:, 0]
