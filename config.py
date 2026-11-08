"""Config dataclasses and YAML preset loading.

Dataclass defaults are the spec defaults. Presets under ``configs/env/`` hold
only the values that differ from them.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent / "configs" / "env"


@dataclass
class DomainCfg:
    L: float = 150.0  # sector side, NM
    dt: float = 30.0  # decision interval, s
    substep: float = 5.0  # physics substep, s
    T: int = 120  # episode step cap (truncation), decision steps
    n_agents_max: int = 5  # aircraft per episode, one per agent
    n_cells_max: int = 16
    oob_margin: float = 10.0  # NM
    wx_inset: float = 10.0  # NM: cell centres stay this far inside the sector edge


@dataclass
class AircraftCfg:
    v_nom: float = 225.0  # kt
    v_min: float = 200.0
    v_max: float = 250.0
    turn_rate_deg: float = 3.0  # deg/s
    accel: float = 1.0  # kt/s


@dataclass
class TrafficCfg:
    min_route_frac: float = 0.5  # |exit - entry| >= frac * L
    spawn_sep: float = 10.0  # NM between entry points at reset
    spawn_candidates: int = 32  # entry/exit draws per aircraft at reset
    r_exit: float = 5.0  # NM
    timeout_factor: float = 2.5


@dataclass
class WeatherCfg:
    theta_u: float = 0.1  # /min
    sigma_u: float = 2.0  # kt/sqrt(min)
    sigma_u0: float = 5.0  # kt
    theta_g: float = 0.2  # /min
    sigma_g: float = 0.05  # NM/min/sqrt(min)
    g0_lo: float = 0.5  # NM/min
    g0_hi: float = 1.5
    tau_lo: float = 40.0  # min
    tau_hi: float = 90.0
    rho_init: float = 2.0  # NM
    rho_min: float = 1.0
    rho_max: float = 10.0  # <= wx_inset, so no cell reaches the sector edge
    theta_I: float = 0.2  # /min
    sigma_I: float = 0.05  # /sqrt(min)
    lambda_b: float = 0.08  # births per minute
    drift_lo: float = 10.0  # kt
    drift_hi: float = 30.0
    I_pk_lo: float = 0.4
    I_pk_hi: float = 1.0
    I_init_frac: float = 0.2
    burn_in: int = 180  # weather-only steps before t = 0
    # Weather systems
    family: str = "mixed"  # isolated | clustered | banded | mixed (one of the three per episode) | scattered (independent cells born anywhere)
    birth_factor: tuple[float, ...] = (0.75, 1.25, 2.0)  # lambda_b multiplier: isolated, clustered, banded
    n_systems_max: tuple[int, ...] = (2, 2, 1)  # systems per episode drawn from 1..max: isolated, clustered, banded
    sys_inset: float = 30.0  # NM: system centres at t = 0 lie this far inside the sector edge
    r_isolated: float = 8.0  # NM: births uniform in a disc of this radius around the system centre
    r_clustered: float = 25.0
    band_half_len_lo: float = 30.0  # NM: births uniform along the band axis within +- half-length
    band_half_len_hi: float = 60.0
    band_half_width: float = 5.0  # NM: and within +- this across it
    band_jitter_deg: float = 30.0  # band axis = wind direction + 90 deg +- this
    envelope_nm: float = 5.0  # NM: avoidance envelope around cell edges, for the reported exposure (not the reward)


@dataclass
class ObsCfg:
    horizon_s: float = 300.0  # look-ahead of the per-heading scan
    buffer_nm: float = 2.5  # added to d_sep and to cell radii in the scan
    track_points: int = 10  # points sampled along each candidate track
    radar_period_s: float = 300.0  # weather is observed as of the latest radar update, refreshed this often
    sense_range_nm: float | None = 20.0  # only cells (to their edge) and aircraft within this range are seen
    mode: str = "grid"  # "scan": per-heading scan; "grid": image window for the CNN agent (experimental)
    grid_n: int = 32  # grid mode: pixels per side
    grid_res_nm: float = 1.25  # grid mode: NM per pixel


@dataclass
class StateCfg:
    include_hidden_weather: bool = True


@dataclass
class ActionCfg:
    heading_offsets_deg: tuple[float, ...] = (-45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0)
    speed_deltas_kt: tuple[float, ...] = (0.0,)  # heading-only; e.g. (-20, 0, 20) adds speed control


@dataclass
class RewardCfg:
    w_p: float = 1.0  # efficiency: w_p * (prog - f), zero when direct at nominal speed
    w_l: float = 50.0
    w_w: float = 100.0
    w_s: float = 0.3  # switching: per switch_unit_deg of turn
    switch_unit_deg: float = 30.0
    w_e: float = 10.0
    w_f: float = 20.0
    d_sep: float = 5.0  # NM


@dataclass
class RenderCfg:
    enabled: bool = False
    n_episodes: int = 3
    fps: int = 10


@dataclass
class EnvCfg:
    domain: DomainCfg = field(default_factory=DomainCfg)
    aircraft: AircraftCfg = field(default_factory=AircraftCfg)
    traffic: TrafficCfg = field(default_factory=TrafficCfg)
    weather: WeatherCfg = field(default_factory=WeatherCfg)
    obs: ObsCfg = field(default_factory=ObsCfg)
    state: StateCfg = field(default_factory=StateCfg)
    action: ActionCfg = field(default_factory=ActionCfg)
    reward: RewardCfg = field(default_factory=RewardCfg)
    render: RenderCfg = field(default_factory=RenderCfg)
    gamma: float = 0.99
    # Fixed evaluation scenarios, disjoint from training seeds (small ints).
    eval_seeds: tuple[int, ...] = tuple(range(100_000, 100_100))

    @property
    def dt_min(self) -> float:
        return self.domain.dt / 60.0

    @property
    def n_substeps(self) -> int:
        return round(self.domain.dt / self.domain.substep)

    @property
    def n_actions(self) -> int:
        return len(self.action.heading_offsets_deg) * len(self.action.speed_deltas_kt)

    @property
    def nominal_action(self) -> int:
        a = self.action
        return a.heading_offsets_deg.index(0.0) * len(a.speed_deltas_kt) + a.speed_deltas_kt.index(0.0)

    @property
    def radar_steps(self) -> int:
        return max(1, round(self.obs.radar_period_s / self.domain.dt))

    @property
    def obs_dim(self) -> int:
        if self.obs.mode == "grid":  # 6 image planes + d_exit, cos β, sin β, radar age
            return 6 * self.obs.grid_n**2 + 4
        # own block + per heading: conflict, weather (latest, previous radar), time to weather (latest, previous)
        return 3 + 5 * len(self.action.heading_offsets_deg)

    @property
    def state_dim(self) -> int:
        return self.domain.n_agents_max * 8 + self.domain.n_cells_max * 8 + 1

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def hash(self) -> str:
        """Short stable hash of the resolved config."""
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:10]


def _update(obj, overrides: dict) -> None:
    for key, value in overrides.items():
        if not hasattr(obj, key):
            raise KeyError(f"Unknown config key '{key}' for {type(obj).__name__}")
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current):
            _update(current, value)
        else:
            setattr(obj, key, tuple(value) if isinstance(value, list) else value)


def load_config(preset: str = "default", overrides: dict | None = None) -> EnvCfg:
    """Load ``configs/env/<preset>.yaml`` (or a YAML path) on top of the defaults."""
    path = Path(preset) if preset.endswith(".yaml") else CONFIG_DIR / f"{preset}.yaml"
    cfg = EnvCfg()
    _update(cfg, yaml.safe_load(path.read_text()) or {})
    _update(cfg, overrides or {})
    return cfg
