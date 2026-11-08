"""The run registry: every run id behind the paper, resolved to a full configuration.

The README says what each package answers; ``repro/jobs.py`` turns a run id into a command and its dependencies.
"""

from __future__ import annotations

from pathlib import Path

from wm.config import MOCK, DiagCfg, load

REGISTRY: dict[str, dict] = {}
# The named result packages: <out>/<package>/<run id>/. The report is in <out>/report.
PACKAGES = {"SUITE": "01_scenarios", "DATA": "data", "WX": "02_forecasters", "QMIX": "03_qmix", "BASE": "04_baselines", "E1": "05_rq1_forecasting", "E2": "06_forecast_quality",
            "E3": "07_rq2_coordination", "AIR": "08_airborne", "E4": "09_attribution", "E5": "10_difficulty"}  # fmt: skip
SUITE = 1344  # 21 chunks: at least 384 rows in each of W1 and W2 (first 16 chunks: W0-W3 = 136, 308, 311, 269)
HALF = 512  # 8 chunks: the airborne system and coordination at N = 5, a prefix of the suite
SMALL = 256  # 4 chunks: other traffic levels, and the difficulty factors
SEEDS = (0, 1, 2)


def reg(run_id: str, exp: str, **overrides) -> None:
    assert run_id not in REGISTRY, run_id
    REGISTRY[run_id] = {"exp": exp, **overrides}


JOINT = {"mode": "joint"}
FIXED = {"mode": "dec", "assume": "priority"}
FROZEN = {"info": "gstale", "belief_speed": False}
EXTRAP = {"info": "gstale"}
TRACKER = {"info": "gpool"}
PERFECT = {"info": "perfect"}
EXACT = {"info": "clairvoyant"}


def forecast(view: str = "sector", seed: int = 0) -> dict:
    return {"info": "llearned" if view == "local" else "learned", "weather_ckpt": f"{PACKAGES['WX']}/WX-{view}-s{seed}/ckpt.pt"}


def qmix(seed: int = 0) -> dict:
    return {"policy": "marl", "checkpoint": f"{PACKAGES['QMIX']}/qmix-s{seed}"}


SEEN = {"info": "seen"}


reg("SUITE-N5", "SUITE", scenarios=SUITE)

# Forecasters (E2), trained on the mixed weather families, and their offline scores against every predictor.
for s in SEEDS:
    reg(f"WX-sector-s{s}", "WX", forecast={"seed": 500_000 + 10_000 * s})
reg("WX-local-s0", "WX", forecast={"local_views": 4})
reg("WX-radar-s0", "WX", forecast={"radar_targets": True})
reg("WX-eval", "E2")

# Baselines on the suite. Delay needs no baseline run: the env measures it against flying direct.
reg("BASE-direct", "BASE", scenarios=SUITE, policy="direct")
reg("BASE-avoid", "BASE", scenarios=SUITE, policy="avoid")
for s in SEEDS:
    reg(f"BASE-qmix-s{s}", "BASE", scenarios=SUITE, **qmix(s))

# E1 (RQ1): does forecasting help? The joint planner, so coordination does not mask the weather.
for name, oracle in (("frozen", FROZEN), ("extrap", EXTRAP), ("tracker", TRACKER), ("perfect", PERFECT), ("radar", forecast("radar"))):
    reg(f"E1-{name}", "E1", scenarios=SUITE, planner=JOINT, oracle=oracle)
for s in SEEDS:
    reg(f"E1-forecast-s{s}", "E1", scenarios=SUITE, planner=JOINT, oracle=forecast(seed=s))

# E3 (RQ2): with the weather known exactly, what coordination is still needed?
SHARED = {"mode": "dec", "assume": "shared"}
for name, planner in (("joint", JOINT), ("direct", {"mode": "dec", "assume": "direct"}), ("shared", SHARED), ("fixed", FIXED)):
    reg(f"E3-N5-{name}", "E3", scenarios=HALF, planner=planner, oracle=EXACT)
for n in (3, 8):
    reg(f"SUITE-N{n}", "SUITE", scenarios=SMALL, n_agents=n)
    reg(f"E3-N{n}-none", "E3", scenarios=SMALL, n_agents=n, policy="direct")
    for name, planner in (("shared", SHARED), ("fixed", FIXED)):
        reg(f"E3-N{n}-{name}", "E3", scenarios=SMALL, n_agents=n, planner=planner, oracle=EXACT)

# The airborne system: each aircraft decides, in a fixed order, with weather it could really have.
reg("AIR-forecast", "AIR", scenarios=HALF, planner=FIXED, oracle=forecast())
reg("AIR-local", "AIR", scenarios=HALF, planner=FIXED, oracle=forecast("local"))
reg("AIR-tracker", "AIR", scenarios=HALF, planner=FIXED, oracle=TRACKER)

# E4: known aircraft motion + learned weather against everything learned.
reg("DATA-avoid", "DATA", data={"policy": "avoid", "name": "avoid-N5"})
reg("E4-s0", "E4", train={"seed": 0, "data": "avoid-N5"})
reg("E4-s0-fly", "E4", scenarios=SMALL, checkpoint=f"{PACKAGES['E4']}/E4-s0/ckpt.pt", planner={"mode": "dec", "assume": "shared", "dynamics": "wm"})
reg("RANK", "E4", checkpoint=f"{PACKAGES['E4']}/E4-s0/ckpt.pt")

# E5: one difficulty factor at a time, on the first 4 chunks. Forecasters and QMIX stay as trained.
FACTORS = {"amount-lo": {"weather": {"lambda_b": 0.04}}, "amount-hi": {"weather": {"lambda_b": 0.16}},
           "wind-lo": {"weather": {"drift_lo": 0.0, "drift_hi": 20.0}}, "wind-hi": {"weather": {"drift_lo": 30.0, "drift_hi": 50.0}},
           "radar-lo": {"obs": {"radar_period_s": 150.0}}, "radar-hi": {"obs": {"radar_period_s": 600.0}},
           "heavy": {"weather": {"lambda_b": 0.24, "drift_lo": 50.0, "drift_hi": 70.0}}}  # fmt: skip  # three times the births and the wind
for name, env in FACTORS.items():
    reg(f"SUITE-{name}", "SUITE", scenarios=SMALL, env=env)
    reg(f"E5-{name}-avoid", "E5", scenarios=SMALL, env=env, policy="avoid")
    reg(f"E5-{name}-qmix", "E5", scenarios=SMALL, env=env, **qmix())
    for short, oracle in (("frozen", FROZEN), ("tracker", TRACKER), ("forecast", forecast())):
        reg(f"E5-{name}-{short}", "E5", scenarios=SMALL, env=env, planner=JOINT, oracle=oracle)


# Heavy weather, with the ceiling and the airborne system: the comparison with the reactive rule and QMIX.
reg("E5-heavy-perfect", "E5", scenarios=SMALL, env=FACTORS["heavy"], planner=JOINT, oracle=PERFECT)
reg("E5-heavy-air-forecast", "E5", scenarios=SMALL, env=FACTORS["heavy"], planner=FIXED, oracle=forecast())
reg("E5-heavy-air-local", "E5", scenarios=SMALL, env=FACTORS["heavy"], planner=FIXED, oracle=forecast("local"))
for n in (3, 8):  # and the traffic level
    reg(f"E5-N{n}-avoid", "E5", scenarios=SMALL, n_agents=n, policy="avoid")
    for short, oracle in (("frozen", FROZEN), ("tracker", TRACKER), ("forecast", forecast())):
        reg(f"E5-N{n}-{short}", "E5", scenarios=SMALL, n_agents=n, planner=JOINT, oracle=oracle)

# Diagnostics behind single numbers in the paper: a forecaster retrained on heavy weather, the true state of only the cells on
# the latest radar picture (what births cost), and a 10 min horizon with the true weather state.
reg("WX-heavy-s0", "WX", env=FACTORS["heavy"])
reg("E5-heavy-forecast-heavy", "E5", scenarios=128, env=FACTORS["heavy"], planner=JOINT, oracle=forecast("heavy"))
reg("E1-seen", "E1", scenarios=128, planner=JOINT, oracle=SEEN)
reg("E1-perfect-H20", "E1", scenarios=192, planner={**JOINT, "horizon": 20}, oracle=PERFECT)


def resolve(run_id: str, overrides: dict | None = None, mock: bool = False) -> tuple[str, DiagCfg]:
    entry = dict(REGISTRY[run_id])
    exp = entry.pop("exp")
    cfg = load(entry, MOCK if mock else None, overrides)
    if mock:
        cfg.out = cfg.out.rstrip("/") + "/mock"
    if cfg.checkpoint and not Path(cfg.checkpoint).is_absolute() and not Path(cfg.checkpoint).exists():
        cfg.checkpoint = str(Path(cfg.out) / cfg.checkpoint)  # checkpoints live under the run root
    w = cfg.oracle.weather_ckpt
    if w and not Path(w).is_absolute() and not Path(w).exists():
        cfg.oracle.weather_ckpt = str(Path(cfg.out) / w)
    return exp, cfg


def run_dir(exp: str, run_id: str, cfg: DiagCfg) -> Path:
    return Path(cfg.out) / PACKAGES[exp] / run_id
