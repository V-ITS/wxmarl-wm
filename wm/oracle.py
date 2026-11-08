"""True-simulator rollouts at one level of weather information.

``Oracle.evaluate`` rolls a batch of joint action sequences forward from a snapshot in a scratch env
and returns the per-aircraft cost of every step. The information level decides what the rollout
knows about the weather:

  clairvoyant  the true hidden cells and the live random stream: the future that will happen
  perfect      the true hidden cells, ``futures`` sampled continuations of the weather process
  seen         as perfect, but only the cells on the latest radar picture: what the forecast cannot know about births
  gstale       a belief built from the last two radar updates of every cell in the sector
  gpool        a deterministic tracker: velocity pooled over the tracked cells (they share the wind), growth held,
               advanced by the picture's age, no noise and no births; the best hand-written predictor we have
  learned      the weather follows the learned forecaster (wm/weather.py) read from the two radar
               pictures; one deterministic future, cells born after the latest picture are absent
  llearned     the same, but each planning aircraft feeds it only the cells it can sense on each picture

Aircraft motion is deterministic given the actions, so the levels differ only through the weather.
A belief is a ``Cells`` state whose observable fields (centre, radius, intensity) come from the
radar, whose velocity is estimated from the two updates (``belief_speed``) and whose hidden fields
(growth, age, lifetime) are set to the prior's centre; it is then advanced by the simulator's own
weather process. That is the belief a planner can hold without any learned model. Beliefs do not know
the hidden weather systems: they advance under ``prior_cfg``, where new cells can appear anywhere.

Batch layout of a rollout: ``[B scenarios, F futures, P populations, n candidates]`` flattened
scenario-major, so ``ReplicatedRng`` with ``reps = P * n`` gives every candidate of a scenario the
same weather future, and with ``F = 1`` and the live generator state exactly the live env's future.
"""

from __future__ import annotations

import copy

import torch
from torch import Tensor

from config import EnvCfg
from core.weather import Cells, step_weather
from wm.config import DiagCfg
from wm.cost import direct_action, step_cost
from wm.snapshot import ReplicatedRng, Snapshot, expand, restore, scratch_env


def belief_cells(snap: Snapshot, cfg: EnvCfg, diag: DiagCfg) -> Cells:
    """Weather belief from the radar: observable fields only, hidden fields from the prior."""
    r, rp = snap.radar, snap.radar_prev
    period_h = cfg.radar_steps * cfg.domain.dt / 3600.0
    both = r.alive & rp.alive
    u = torch.where(both.unsqueeze(-1), (r.q - rp.q) / period_h, torch.zeros_like(r.q))  # kt
    if not diag.oracle.belief_speed:
        u = torch.zeros_like(u)
    n_alive = r.alive.sum(1, keepdim=True).clamp_min(1)
    drift = (u * r.alive.unsqueeze(-1)).sum(1) / n_alive  # [B, 2]: sector-mean cell velocity
    tau = torch.full_like(r.rho, diag.oracle.belief_tau_min)
    return Cells(
        q=r.q.clone(), rho=r.rho.clone(), I=r.I.clone(), u=u, g=torch.zeros_like(r.g), g0=torch.zeros_like(r.g0),
        age=tau / 2, tau=tau, I_pk=r.I.clone(), alive=r.alive.clone(), q_prev=r.q.clone(), rho_prev=r.rho.clone(),
        drift=drift, fam=r.fam, n_sys=r.n_sys, Q=r.Q, axis=r.axis, half_len=r.half_len, sys=r.sys,
    )  # fmt: skip


def tracked_cells(snap, cfg: EnvCfg, diag: DiagCfg) -> Cells:
    """Belief with growth from the two radii and velocity shrunk toward the mean of the tracked cells."""
    from wm.weather import _same

    r, rp, R = snap.radar, snap.radar_prev, cfg.radar_steps
    b = belief_cells(snap, cfg, diag)
    same = _same(r, rp, R, cfg)
    g = torch.where(same, (r.rho - rp.rho) / (R * cfg.dt_min), torch.zeros_like(r.rho))
    b.g, b.g0 = g, g.clone()
    own = (r.q - rp.q) / (R * cfg.domain.dt / 3600.0)  # kt
    mean = (own * same.unsqueeze(-1)).sum(1) / same.sum(1, keepdim=True).clamp_min(1)  # [B, 2]
    w = diag.oracle.pool_weight
    b.u = torch.where(same.unsqueeze(-1), w * own + (1 - w) * mean.unsqueeze(1), mean.unsqueeze(1))
    b.drift = mean
    frac = 0.1  # early in life, where the process keeps growth and intensity near their current values
    b.age, b.g0, b.I_pk = frac * b.tau, g / (1 - 2 * frac), r.I / (2 * frac)
    return b


def prior_cfg(cfg: EnvCfg) -> EnvCfg:
    """The weather process a belief assumes: the same, with births anywhere in the sector."""
    cfg = copy.deepcopy(cfg)
    cfg.weather.family = "scattered"
    return cfg


def noiseless(cfg: EnvCfg) -> EnvCfg:
    """The belief's process with no noise and no births: the mean future a tracker predicts."""
    cfg = prior_cfg(cfg)
    w = cfg.weather
    w.lambda_b, w.sigma_u, w.sigma_g, w.sigma_I = 0.0, 0.0, 0.0, 0.0
    return cfg


class Oracle:
    """Rollouts on the true simulator at one information level."""

    def __init__(self, diag: DiagCfg, env_cfg: EnvCfg, device: str | torch.device = "cpu") -> None:
        self.diag, self.cfg, self.device = diag, env_cfg, torch.device(device)
        self.info = diag.oracle.info
        assert self.info in ("clairvoyant", "perfect", "seen", "gstale", "gpool", "learned", "llearned"), self.info
        rollout = {"clairvoyant": env_cfg, "perfect": env_cfg, "seen": env_cfg, "gpool": noiseless(env_cfg)}.get(self.info, prior_cfg(env_cfg))
        self.scratch = scratch_env(rollout, device)
        self.futures = 1 if self.info in ("clairvoyant", "gpool", "learned", "llearned") else diag.oracle.futures
        self.weather = None
        if self.info in ("learned", "llearned"):
            from wm.weather import WeatherForecaster

            self.weather = WeatherForecaster.load(diag.oracle.weather_ckpt, env_cfg, device)
        self._seed = 0

    def evaluate(
        self, snap: Snapshot, actions: Tensor, P: int, n: int, direct_mask: Tensor | None = None,
        slots: Tensor | None = None, seed: int | None = None,
    ) -> Tensor:  # fmt: skip
        """Cost [B, F, P, n, H, N] of joint action sequences ``actions`` [B * F * P * n, H, N].

        ``direct_mask`` [B * F * P * n, N] marks aircraft that follow the scripted direct policy
        (turn toward their exit, recomputed from the rolled-out state every step) instead of the
        given action. ``slots`` [P] names the planning aircraft of
        each population (needed by ``llearned``). ``seed`` fixes the sampled futures (default: a new
        seed per call; ignored by ``clairvoyant``, which replays the live stream).
        """
        B, F, H, N = snap.B, self.futures, actions.shape[1], actions.shape[2]
        reps = P * n
        assert actions.shape[0] == B * F * reps, (actions.shape, B, F, P, n)
        rng = ReplicatedRng(0, self.device, reps=reps)
        env = restore(self.scratch, expand(snap, F * reps), rng)
        if self.info == "seen":
            from wm.weather import _same

            keep = _same(snap.cells, snap.radar, snap.t % self.cfg.radar_steps, self.cfg)
            env.cells.alive = env.cells.alive & keep.repeat_interleave(F * reps, 0)
        if self.info != "clairvoyant":
            if seed is None:
                seed, self._seed = self._seed, self._seed + 1
            rng.seed(seed)
        fc = None
        if self.info == "learned":
            fc = self.weather.forecast(snap.radar, snap.radar_prev, snap.t % self.cfg.radar_steps, self.cfg)
            fc = {k: v.repeat_interleave(F * reps, dim=0) for k, v in fc.items()}
        elif self.info == "llearned":
            from wm.weather import local_view, repeat_cells

            assert slots is not None and slots.shape == (P,)
            p = snap.ac.p[:, slots].reshape(B * P, 2)  # the planning aircraft of each population
            radar = repeat_cells(snap.radar, P)
            prev = repeat_cells(snap.radar_prev, P)
            radar, prev = local_view(radar, prev, p, self.cfg.obs.sense_range_nm)
            fc = self.weather.forecast(radar, prev, snap.t % self.cfg.radar_steps, self.cfg)
            fc = {k: v.repeat_interleave(F * n, dim=0) for k, v in fc.items()}
        if self.info not in ("clairvoyant", "perfect", "seen"):
            cells = tracked_cells(snap, self.cfg, self.diag) if self.info == "gpool" else belief_cells(snap, self.cfg, self.diag)
            cells = Cells(**{k: v.repeat_interleave(F * reps, dim=0) for k, v in vars(cells).items()})
            if self.info == "gpool":
                for _ in range(snap.t % self.cfg.radar_steps):  # the picture is old: bring it up to now
                    step_weather(cells, env.cfg, rng)
            env.cells = cells
        cost = torch.zeros(B * F * reps, H, N, device=self.device)
        for h in range(H):
            if fc is not None:  # the weather of step h is the forecast, whatever the process did
                k = min(h, fc["q"].shape[1] - 1)
                c = env.cells
                c.q, c.rho, c.I, c.alive = fc["q"][:, k], fc["rho"][:, k], fc["I"][:, k], fc["alive"][:, k]
                c.q_prev, c.rho_prev = c.q, c.rho
            a = actions[:, h]
            if direct_mask is not None:
                a = torch.where(direct_mask, direct_action(env.ac, self.cfg), a)
            *_, info = env.step(a)
            cost[:, h] = step_cost(info)
        return cost.view(B, F, P, n, H, N)
