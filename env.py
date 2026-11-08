"""WxEnv: batched core environment. Pure PyTorch, no EPyMARL dependency."""

from __future__ import annotations

import torch
from torch import Tensor

from config import EnvCfg
from core import traffic
from core.aircraft import Aircraft, apply_actions, substep
from core.geometry import pairwise_dist
from core.metrics import Metrics
from core.observation import build_grid, build_obs, build_state
from core.reward import reward_terms
from core.rng import Rng
from core.weather import Cells, edge_distance, init_cells, severity, snapshot, step_weather


class WxEnv:
    """Batched weather-aware conflict-resolution environment.

    Every episode has ``N`` aircraft, one per agent, each with its own entry and exit. An env's
    episode terminates when all its aircraft have finished and is truncated at ``t = T`` otherwise.
    The ``B`` envs share the step counter ``t``; an env that finishes early stays empty until the
    whole batch is done, which is when the episode metrics are returned.
    """

    def __init__(
        self, cfg: EnvCfg, batch_size: int = 1, device: str = "cpu", seed: int = 0, auto_reset: bool = False
    ) -> None:
        self.cfg = cfg
        self.B, self.N = batch_size, cfg.domain.n_agents_max
        self.device = torch.device(device)
        self.auto_reset = auto_reset
        self.rng = Rng(seed, self.device)
        self.t = 0
        self.ac: Aircraft
        self.cells: Cells
        self.radar: Cells  # cells as of the latest radar update, which the agents observe
        self.radar_prev: Cells  # cells as of the update before it
        self.metrics: Metrics
        self.los_pairs: Tensor  # [B, N, N] bool, pairs in LoS at the latest substep
        self.los_step: Tensor  # [B, N, N] bool, pairs in LoS at any substep of the latest step
        self.finished: Tensor  # [B] bool, episode over (terminated or truncated)

    def reset(self, seed: int | None = None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Start new episodes. Returns ``(obs, state, alive, avail)``."""
        if seed is not None:
            self.rng.seed(seed)
        cfg, B, N = self.cfg, self.B, self.N
        self.t = 0
        self.metrics = Metrics(B, self.device)
        # The burn-in ends with a radar update at t = 0; the update before it is radar_steps earlier.
        self.cells = init_cells(B, cfg, self.rng, steps=cfg.weather.burn_in - cfg.radar_steps)
        self.radar_prev = snapshot(self.cells)
        for _ in range(cfg.radar_steps):
            step_weather(self.cells, cfg, self.rng)
        self.radar = snapshot(self.cells)
        self.ac = Aircraft.empty(B, N, self.device)
        self.los_pairs = torch.zeros(B, N, N, dtype=torch.bool, device=self.device)
        self.los_step = self.los_pairs
        self.finished = torch.zeros(B, dtype=torch.bool, device=self.device)
        for slot in range(N):
            self.metrics.add("spawn_relaxed", traffic.spawn(self.ac, self.cells, slot, cfg, self.rng))
        return self._outputs()

    def step(self, actions: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, dict]:
        """Advance one decision step with ``actions`` LongTensor[B, N].

        Returns ``(obs, state, reward, done, env_term, env_trunc, alive, avail, info)``. ``done``
        [B, N] marks flights that ended this step; ``env_term`` [B] envs whose last aircraft just
        finished; ``env_trunc`` [B] envs cut off at ``t = T``.
        """
        cfg, ac, m = self.cfg, self.ac, self.metrics
        delta, n_sub = cfg.domain.substep, cfg.n_substeps
        active = ac.alive.clone()  # slots that act this step
        actions = torch.where(active, actions.to(self.device), cfg.nominal_action)
        d_before = (ac.e - ac.p).norm(dim=-1)
        apply_actions(ac, actions, cfg)
        dpsi_change = ac.last_dpsi.abs()

        moving = active.clone()
        exited = torch.zeros_like(active)
        los_steps = torch.zeros(self.B, self.N, device=self.device)
        wx_sum = torch.zeros_like(los_steps)
        flown = torch.zeros_like(los_steps)
        self.los_step = torch.zeros_like(self.los_pairs)
        for _ in range(n_sub):
            flown += moving
            substep(ac, moving, cfg)
            reached = moving & ((ac.e - ac.p).norm(dim=-1) <= cfg.traffic.r_exit)
            exited |= reached
            moving = moving & ~reached

            pair = moving.unsqueeze(1) & moving.unsqueeze(2) & (pairwise_dist(ac.p, ac.p) < cfg.reward.d_sep)
            pair = pair & ~torch.eye(self.N, dtype=torch.bool, device=self.device)
            los_steps += pair.any(-1)
            m.add("los_events", (pair & ~self.los_pairs).triu(1))
            m.add("los_time_s", pair.triu(1) * delta)
            self.los_pairs = pair
            self.los_step = self.los_step | pair

            sev = severity(ac.p, self.cells) * moving
            wx_sum += sev
            ac.wx_hit |= sev > 0
            m.add("wx_incursion_time_s", (sev > 0) * delta)
            m.add("wx_severity_integral", sev * delta)
            near = moving & (edge_distance(ac.p, self.cells) <= cfg.weather.envelope_nm)
            ac.env_hit |= near
            m.add("wx_envelope_time_s", near * delta)

        step_weather(self.cells, cfg, self.rng)

        timeout = traffic.failures(ac, cfg) & moving
        done = exited | timeout
        terms = reward_terms(
            cfg, active, d_before, (ac.e - ac.p).norm(dim=-1), flown / n_sub, los_steps / n_sub, wx_sum / n_sub,
            dpsi_change, exited, timeout,
        )  # fmt: skip
        reward = sum(terms.values())

        running = ~self.finished
        m.add("ep_length", running)
        m.add("_active_steps", active)
        m.add("exits", exited)
        m.add("fail_timeout", timeout)
        m.add("_flights_wx", done & ac.wx_hit)
        m.add("_flights_envelope", done & ac.env_hit)
        # Against flying straight to the exit circle at nominal speed, on the substep grid: 0 for fly-direct.
        to_circle = ((ac.e - ac.p0).norm(dim=-1) - cfg.traffic.r_exit).clamp_min(0.0)
        t_ref = torch.ceil(to_circle / (cfg.aircraft.v_nom * delta / 3600.0)) * delta
        m.add("_extra_track", exited * (ac.path_len - t_ref * cfg.aircraft.v_nom / 3600.0))
        m.add("_delay", exited * (ac.flight_time - t_ref))
        for k, v in terms.items():
            m.add(k, v)

        ac.alive = active & ~done
        self.los_pairs = self.los_pairs & ~done.unsqueeze(1) & ~done.unsqueeze(2)
        self.t += 1
        if self.t % cfg.radar_steps == 0:
            self.radar_prev, self.radar = self.radar, snapshot(self.cells)

        empty = ~ac.alive.any(1)
        env_term = running & empty
        env_trunc = running & ~empty & (self.t >= cfg.domain.T)
        self.finished = self.finished | env_term | env_trunc
        batch_done = bool(self.finished.all())
        info: dict = dict(terms)
        if batch_done:
            info["episode"] = m.finalize(ac.alive, ac.wx_hit, ac.env_hit)
        obs, state, alive, avail = self.reset() if batch_done and self.auto_reset else self._outputs()
        return obs, state, reward, done, env_term, env_trunc, alive, avail, info

    def _outputs(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        cfg, alive = self.cfg, self.ac.alive
        avail = alive.unsqueeze(-1).repeat(1, 1, cfg.n_actions)
        avail[..., cfg.nominal_action] = True
        if cfg.obs.mode == "grid":
            obs = build_grid(self.ac, self.radar, self.radar_prev, self.t, cfg)
        else:
            obs = build_obs(self.ac, self.radar, self.radar_prev, cfg)
        return obs, build_state(self.ac, self.cells, self.t, cfg), alive.clone(), avail
