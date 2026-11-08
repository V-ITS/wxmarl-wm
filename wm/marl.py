"""A trained EPyMARL policy (QMIX with the CNN agent) as a reference controller on the batched env.

EPyMARL evaluates on one env at a time and its own eval seeds; this runs the saved agent greedily on the
diagnostics' scenario chunks instead, so its metrics pair with every other condition. Inputs are built as
EPyMARL's basic controller builds them for QMIX (observation, then the agent one-hot; no last action), from
the grid observation the agent was trained on, and unavailable actions are masked before the argmax.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from torch import Tensor

from core.observation import build_grid
from env import WxEnv
from wrappers.cnn_agent import CNNAgent


class MarlPolicy:
    """Greedy decentralized execution of a saved agent network; ``policy(env, avail, gen) -> actions [B, N]``."""

    def __init__(self, ckpt_dir: str | Path, env: WxEnv, hidden_dim: int = 64) -> None:
        cfg, (B, N) = env.cfg, env.ac.alive.shape
        args = SimpleNamespace(env_args={"map_name": "grid"}, hidden_dim=hidden_dim, use_rnn=True, n_actions=cfg.n_actions)
        n_obs = build_grid(env.ac, env.radar, env.radar_prev, env.t, cfg).shape[-1]  # call after env.reset
        self.agent = CNNAgent(n_obs + N, args).to(env.device)
        self.agent.load_state_dict(torch.load(Path(ckpt_dir) / "agent.th", map_location=env.device, weights_only=True))
        self.agent.eval()
        self.ids = torch.eye(N, device=env.device).expand(B, N, N).reshape(B * N, N)
        self.h = self.agent.init_hidden().expand(B * N, -1).contiguous()

    @torch.no_grad()
    def __call__(self, env: WxEnv, avail: Tensor, gen: torch.Generator | None = None) -> Tensor:
        B, N = avail.shape[:2]
        obs = build_grid(env.ac, env.radar, env.radar_prev, env.t, env.cfg).reshape(B * N, -1)
        q, self.h = self.agent(torch.cat((obs, self.ids), 1), self.h)
        return q.masked_fill(~avail.reshape(B * N, -1), -float("inf")).argmax(-1).view(B, N)
