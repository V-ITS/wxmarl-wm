"""EPyMARL ``MultiAgentEnv`` adapter around the core env with B=1.

The class is duck-typed against EPyMARL's interface so this module does not import EPyMARL.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from config import load_config
from core.metrics import summarize
from env import WxEnv
from render.recorder import Recorder

ROOT = Path(__file__).resolve().parents[1]


class WxEpymarlEnv:
    """One WxEnv instance behind EPyMARL's env API.

    Args:
        map_name: weather preset name (EPyMARL requires a ``map_name`` or ``key`` env arg).
        seed: env seed, set by EPyMARL from the run seed (+ worker index).
        common_reward, reward_scalarisation: EPyMARL reward options; the env scalarises itself.
        render: record the first ``render.n_episodes`` episodes and write replays and videos.
        eval_scenarios: reset episode k with ``cfg.eval_seeds[k]`` instead of the running stream.
        out_dir: where replays, videos and (on close) ``metrics.json`` go.
        overrides: nested dict of env config overrides.
        **meta: extra run metadata (algorithm, checkpoint, commit hashes) stored with replays.
    """

    def __init__(
        self, map_name: str = "default", seed: int = 0, common_reward: bool = True,
        reward_scalarisation: str = "sum", render: bool = False, eval_scenarios: bool = False,
        out_dir: str | None = None, overrides: dict | None = None, **meta,
    ) -> None:  # fmt: skip
        assert reward_scalarisation in ("sum", "mean"), reward_scalarisation
        self.cfg = load_config(map_name, overrides)
        self.env = WxEnv(self.cfg, batch_size=1, device="cpu", seed=seed)
        self.n_agents = self.cfg.domain.n_agents_max
        self.episode_limit = self.cfg.domain.T
        self.common_reward, self.reward_scalarisation = common_reward, reward_scalarisation
        self.render_enabled, self.eval_scenarios = render, eval_scenarios
        self.out_dir = Path(out_dir) if out_dir else ROOT / "results" / "renders" / f"{map_name}_seed{seed}"
        self.write_metrics = out_dir is not None
        self.meta = {"seed": seed, **meta}
        self.episode = -1
        self.episode_seed: int | None = None
        self.episodes: list[dict[str, float]] = []
        self.recorder: Recorder | None = None
        self._obs = self._state = self._avail = None

    def reset(self, seed: int | None = None, options: dict | None = None):
        """Returns initial observations and info."""
        self.episode += 1
        if seed is None and self.eval_scenarios:
            seed = self.cfg.eval_seeds[self.episode % len(self.cfg.eval_seeds)]
        self.episode_seed = seed
        self._obs, self._state, _, self._avail = self.env.reset(seed=seed)
        self.recorder = None
        if self.render_enabled and self.episode < self.cfg.render.n_episodes:
            self.recorder = Recorder(self.env)
            self.recorder.capture()
        return self.get_obs(), {}

    def step(self, actions):
        """Returns obss, reward, terminated, truncated, info."""
        actions = torch.as_tensor(np.asarray([int(a) for a in actions])).view(1, -1)
        self._obs, self._state, reward, _, term, trunc, _, self._avail, info = self.env.step(actions)
        if self.recorder is not None:
            self.recorder.capture()
        if self.common_reward:
            total = reward.sum().item()
            reward = total if self.reward_scalarisation == "sum" else total / self.n_agents
        else:
            reward = reward[0].numpy().astype(np.float32)
        terminated, truncated = bool(term[0]), bool(trunc[0])
        out_info: dict[str, float] = {}
        if terminated or truncated:
            episode = {k: v.item() for k, v in info["episode"].items()}
            self.episodes.append(episode)
            # EPyMARL already logs returns and episode length.
            out_info = {k: v for k, v in episode.items() if k not in ("return_mean", "ep_length")}
            if truncated:  # tells EPyMARL this is a time-limit truncation, not a termination
                out_info["episode_limit"] = True
            if self.recorder is not None:
                self._write_video()
        return self.get_obs(), reward, terminated, truncated, out_info

    def _write_video(self) -> None:
        from render.animate import animate

        stem = self.out_dir / f"episode_{self.episode}" if self.episode_seed is None else (
            self.out_dir / f"episode_{self.episode}_seed{self.episode_seed}"
        )  # fmt: skip
        replay = self.recorder.save(stem, **self.meta, episode_seed=self.episode_seed)
        video = animate(replay, fps=self.cfg.render.fps, title=str(self.meta.get("algorithm", "")))
        try:
            import wandb
        except ImportError:
            return
        if wandb.run is not None:  # only true when the env lives in the main process (episode runner)
            wandb.log({f"video/episode_{self.episode}": wandb.Video(str(video))}, commit=False)

    def get_obs(self):
        """Returns all agent observations in a list."""
        return list(self._obs[0].numpy())

    def get_obs_agent(self, agent_id):
        return self._obs[0, agent_id].numpy()

    def get_obs_size(self):
        return self.cfg.obs_dim

    def get_state(self):
        return self._state[0].numpy()

    def get_state_size(self):
        return self.cfg.state_dim

    def get_avail_actions(self):
        return self._avail[0].int().tolist()

    def get_avail_agent_actions(self, agent_id):
        return self._avail[0, agent_id].int().tolist()

    def get_total_actions(self):
        return self.cfg.n_actions

    def get_env_info(self):
        return {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit,
        }

    def get_stats(self):
        return {}

    def render(self):
        """No-op: rendering is offline, toggled by ``env_args.render``."""

    def save_replay(self):
        """No-op: replays are written at episode end when ``env_args.render`` is set."""

    def seed(self, seed=None):
        self.env.rng.seed(seed)

    def close(self):
        if self.write_metrics and self.episodes:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            payload = {"episodes": self.episodes, "summary": summarize(self.episodes)}
            (self.out_dir / "metrics.json").write_text(json.dumps(payload, indent=2))


def make_env(**kwargs) -> WxEpymarlEnv:
    """Factory registered as ``REGISTRY["wxmarl"]`` in EPyMARL."""
    return WxEpymarlEnv(**kwargs)
