"""Diagnostic configuration: planner, oracle, model, data and training knobs, plus YAML presets.

``DiagCfg`` is the dataclass defaults, then the run registry's overrides (``wm/experiments.py``),
then command-line overrides (``a.b=value``) and, with ``--mock``, the CPU test sizes. The environment
itself is configured through WxMARL's own presets (``configs/env``), referenced by ``env_preset``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from config import EnvCfg, _update, load_config


@dataclass
class PlannerCfg:
    mode: str = "joint"  # "joint": one population over A^N; "dec": one per aircraft, others fixed
    assume: str = "shared"  # dec: what the others do: "shared" | "direct" | "priority"
    rounds: int = 2  # dec+shared: best-response rounds (plans exchanged between rounds)
    horizon: int = 10  # decision steps (5 min)
    candidates: int = 128  # calibrated on chunk 0 against 300 x 30: identical for the joint planner (team cost
    iters: int = 8  # within 1 unit, same IGM rate); the larger budget made the decentralized planner worse
    elite_frac: float = 0.1
    persist: float = 0.8  # probability that a candidate keeps its heading from the previous step
    alpha: float = 0.7  # update rate of the categorical distribution toward the elites
    init_nominal: float = 0.5  # initial mass on the nominal (direct) action; the rest is uniform
    gamma: float = 1.0  # discount inside the horizon
    dynamics: str = "sim"  # "sim": true simulator through the oracle; "wm": learned world model


@dataclass
class OracleCfg:
    info: str = "clairvoyant"  # "clairvoyant" | "perfect" | "seen" | "gstale" | "gpool" | "learned" | "llearned"
    futures: int = 4  # sampled weather futures for perfect / stale beliefs (1 for clairvoyant)
    belief_speed: bool = True  # stale: estimate each cell's velocity from the last two radar updates
    belief_tau_min: float = 65.0  # stale: assumed cell lifetime (midpoint of the env prior), min
    weather_ckpt: str = ""  # learned: the weather forecaster (wm/weather.py), e.g. WX/WX-forecast/ckpt.pt
    pool_weight: float = 0.75  # gpool (best of 0, .25, .5, .75, 1 on held-out seeds): weight of a cell's own velocity against the velocity pooled over tracked cells


@dataclass
class ModelCfg:
    dim: int = 128
    latent: int = 64
    enc_layers: int = 2
    pred_layers: int = 3
    heads: int = 4
    history: int = 3  # frames fed to the predictor
    sigreg_weight: float = 0.09
    head_weight: float = 1.0  # state-head loss (the planner's cost readout; privileged targets)
    k: int = 1  # open-loop training steps
    rollout_chunk: int = 8192  # candidate sequences per forward pass when planning (GPU memory)
    rollout_fp16: bool = (
        True  # run the predictor under fp16 autocast when planning on a GPU (decode and cost stay fp32)
    )


@dataclass
class ForecastCfg:
    """The learned weather forecaster (wm/weather.py), trained on weather-only simulation."""

    dim: int = 64
    layers: int = 2
    heads: int = 4
    horizon: int = 10  # decision steps forecast (the planner's horizon)
    train_episodes: int = 1024
    val_episodes: int = 128
    seed: int = 500_000  # weather seeds, disjoint from scenario, dataset and env-statistics seeds
    epochs: int = 30
    batch: int = 512
    lr: float = 1e-3
    eval_episodes: int = 256
    eval_seed: int = 400_000
    local_views: int = 0  # 0: reads the whole sector's radar; else trains on this many sensing-range views per sample
    radar_targets: bool = False  # train on future radar pictures, interpolated between refreshes, not simulator truth


@dataclass
class DataCfg:
    policy: str = "avoid"  # a scripted policy of baselines.POLICIES
    episodes: int = 2048
    batch: int = 256
    seed: int = 1000  # data seeds: seed .. seed + episodes / batch; disjoint from scenario seeds
    name: str = ""  # file stem under <out>/data; default: <policy>-N<n>-s<seed>


@dataclass
class TrainCfg:
    epochs: int = 30
    batch: int = 256
    lr: float = 3e-4
    weight_decay: float = 0.05
    seed: int = 0
    val_scenarios: int = 32  # closed-loop validation scenarios (model selection by safety, never by loss)
    val_every: int = 5  # epochs
    data: str = ""  # dataset name (see DataCfg.name); set by the registry


@dataclass
class DiagCfg:
    env_preset: str = "default"
    env: dict = field(default_factory=dict)  # overrides on the WxMARL EnvCfg
    n_agents: int = 5
    scenarios: int = 1024
    scenario_seed0: int = (
        200_000  # scenario i uses seed scenario_seed0 + i (disjoint from env.eval_seeds and data seeds)
    )
    chunk: int = 64  # scenarios simulated in one batch (16 chunks; the scratch env is launch-bound below this)
    planner_seed: int = 0
    device: str = "cpu"
    out: str = "results/redesign"  # every run writes under <out>/<package>/<run id>
    planner: PlannerCfg = field(default_factory=PlannerCfg)
    oracle: OracleCfg = field(default_factory=OracleCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    forecast: ForecastCfg = field(default_factory=ForecastCfg)
    data: DataCfg = field(default_factory=DataCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    policy: str = ""  # reference runs: a scripted policy acts for every aircraft, no planner
    checkpoint: str = ""  # wm dynamics: <out>/E4/<run>/ckpt.pt; policy "marl": an EPyMARL model dir
    wx_near_nm: float = 10.0  # ranking: a moment is "near weather" within this of a cell edge

    def env_cfg(self) -> EnvCfg:
        cfg = load_config(self.env_preset, self.env)
        cfg.domain.n_agents_max = self.n_agents
        cfg.obs.mode = "scan"  # the planners never read observations; QMIX builds its own grid
        return cfg

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:10]


MOCK = {  # CPU sizes for tests and smoke runs: --mock
    "scenarios": 8, "chunk": 8,
    "planner": {"candidates": 32, "iters": 3, "horizon": 5},
    "oracle": {"futures": 2},
    "data": {"episodes": 16, "batch": 8},
    "train": {"epochs": 2, "batch": 64, "val_scenarios": 4, "val_every": 1},
    "model": {"dim": 32, "latent": 16, "enc_layers": 1, "pred_layers": 1, "heads": 2},
    "forecast": {"dim": 16, "layers": 1, "heads": 2, "train_episodes": 16, "val_episodes": 8, "epochs": 2,
                 "eval_episodes": 8},
}  # fmt: skip


def load(*overrides: dict | None) -> DiagCfg:
    """The defaults, then ``overrides`` in order (a YAML path string is read as one override)."""
    cfg = DiagCfg()
    for o in overrides:
        if isinstance(o, str):
            o = yaml.safe_load(Path(o).read_text()) or {}
        if o:
            _update(cfg, o)
    return cfg


def parse_overrides(items: list[str]) -> dict:
    """``a.b=value`` command-line overrides -> nested dict; values are parsed as YAML."""
    out: dict = {}
    for item in items:
        key, _, value = item.partition("=")
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(value)
    return out
