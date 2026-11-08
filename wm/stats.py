"""Per-scenario metrics, paired tests and tables by route-impact class.

    python -m wm.stats E1                      # every E1 run found, overall and by W0-W3
    python -m wm.stats AIR --ref BASE-qmix-s0  # and paired against a reference run

Paired tests use the same scenario rows of two runs: Wilcoxon signed-rank for continuous metrics, exact McNemar
for "any loss of separation in the scenario", 95% bootstrap intervals of the paired difference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats as sps

from wm import experiments
from wm.config import DiagCfg

HEADLINE = ("storm_s", "envelope_s", "delay_s", "extra_nm", "los_per_fh", "los_s", "turns", "turn_deg", "exit_rate", "worst_delay_s", "turn_spread")
BALANCED = {0: 128, 1: 384, 2: 384, 3: 128}  # the evaluation suite's rows per class


def per_scenario(result: dict) -> dict[str, np.ndarray]:
    """Scenario-level values of every metric (NaN where undefined). Delay is against flying direct to the exit."""
    rows = result["scenarios"]
    col = lambda k: np.array([r.get(k, np.nan) for r in rows], dtype=float)
    flights = col("flights")
    per_flight = lambda k: np.array([np.sum(r[k]) for r in rows], dtype=float) / flights if k in rows[0] else np.full(len(rows), np.nan)
    hours = col("flight_hours")
    by_aircraft = lambda k, f: np.array([f(r[k]) for r in rows], dtype=float) if k in rows[0] else np.full(len(rows), np.nan)
    return {
        "storm_s": col("wx_incursion_time_s") / flights,
        "envelope_s": col("wx_envelope_time_s") / flights,
        "severity": col("wx_severity_integral") / flights,
        "delay_s": col("delay_s"),
        "extra_nm": col("extra_track_nm"),
        "los_per_fh": np.where(hours > 0, col("los_events") / np.maximum(hours, 1e-9), np.nan),
        "los_s": col("los_time_s") / flights,
        "turns": per_flight("turns"),
        "turn_deg": per_flight("turn_deg"),
        "exit_rate": col("exit_rate"),
        "worst_delay_s": by_aircraft("delay_by_aircraft", np.max),  # the manoeuvre burden: who pays for the detours
        "turn_spread": by_aircraft("turn_deg", np.std),
        "any_los": (col("los_events") > 0).astype(float),
    }


def summary(result: dict) -> dict[str, tuple[float, float]]:
    """Mean and 95% bootstrap half-width per headline metric."""
    ps, rng, out = per_scenario(result), np.random.default_rng(0), {}
    for k in HEADLINE:
        x = ps[k][~np.isnan(ps[k])]
        if len(x) == 0:
            out[k] = (float("nan"), float("nan"))
            continue
        boots = rng.choice(x, (1000, len(x)), replace=True).mean(1)
        out[k] = (float(x.mean()), float((np.percentile(boots, 97.5) - np.percentile(boots, 2.5)) / 2))
    return out


def headline(result: dict) -> str:
    return f"{result['run_id']}: " + "  ".join(f"{k}={m:.3g}±{h:.2g}" for k, (m, h) in summary(result).items() if not np.isnan(m))


def paired(a: dict, b: dict) -> dict[str, dict]:
    """Run ``b`` minus run ``a`` on their shared scenarios."""
    key = lambda r: (r["chunk_seed"], r["row"])
    ia = {key(r): i for i, r in enumerate(a["scenarios"])}
    idx = [(ia[key(r)], j) for j, r in enumerate(b["scenarios"]) if key(r) in ia]
    pa, pb, rng, out = per_scenario(a), per_scenario(b), np.random.default_rng(0), {}
    ja, jb = np.array([i for i, _ in idx]), np.array([j for _, j in idx])
    for k in HEADLINE:
        x, y = pa[k][ja], pb[k][jb]
        ok = ~(np.isnan(x) | np.isnan(y))
        diff = y[ok] - x[ok]
        if len(diff) < 2:
            continue
        boots = rng.choice(diff, (1000, len(diff)), replace=True).mean(1)
        out[k] = {"n": len(diff), "diff": float(diff.mean()), "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                  "p_wilcoxon": float(sps.wilcoxon(y[ok], x[ok], zero_method="zsplit").pvalue) if np.any(diff != 0) else 1.0}  # fmt: skip
    x, y = pa["any_los"][ja], pb["any_los"][jb]
    b01, b10 = int(((x == 0) & (y == 1)).sum()), int(((x == 1) & (y == 0)).sum())
    out["any_los"] = {"n": len(x), "rate_a": float(x.mean()), "rate_b": float(y.mean()), "b01": b01, "b10": b10,
                      "p_mcnemar": float(sps.binomtest(b01, b01 + b10, 0.5).pvalue) if b01 + b10 else 1.0}  # fmt: skip
    return out


def classes(root: Path) -> dict[str, dict[tuple[int, int], dict]]:
    """Route-impact rows of every screened weather setting (``wm/suite.py``), by env hash, then (chunk seed, row)."""
    out = {}
    for p in sorted((root / experiments.PACKAGES["SUITE"]).glob("*/suite.json")):
        d = json.loads(p.read_text())
        out[d["env_cfg_hash"]] = {(r["chunk_seed"], r["row"]): r for r in d["scenarios"]}
    return out


def subset(result: dict, keep) -> dict:
    return {**result, "scenarios": [r for r in result["scenarios"] if keep(r)]}


def balanced(result: dict, screen: dict) -> dict:
    """The first rows of each class up to ``BALANCED``: the frozen 1,024-scenario suite."""
    left, rows = dict(BALANCED), []
    for r in result["scenarios"]:
        w = screen[(r["chunk_seed"], r["row"])]["w_class"]
        if left[w] > 0:
            left[w] -= 1
            rows.append(r)
    return {**result, "scenarios": rows}


def table(exp: str, ref: str | None, root: Path | None = None) -> str:
    root = root or Path(DiagCfg().out)
    runs = {p.parent.name: json.loads(p.read_text()) for p in sorted((root / experiments.PACKAGES[exp]).glob("*/metrics.json"))}
    screens = classes(root)
    lines = []
    for name, m in runs.items():
        cls = screens.get(m["env_cfg_hash"], {})
        lines.append(f"{name:24s} all n={len(m['scenarios']):4d}  " + _cell(m))
        if len(m["scenarios"]) >= experiments.SUITE and cls:
            lines.append(f"{'':24s} bal n={sum(BALANCED.values()):4d}  " + _cell(balanced(m, cls)))
        for w in range(4):
            part = subset(m, lambda r, w=w: cls.get((r["chunk_seed"], r["row"]), {}).get("w_class") == w)
            if part["scenarios"]:
                lines.append(f"{'':24s} W{w}  n={len(part['scenarios']):4d}  " + _cell(part))
        if ref:
            r = paired(experiments_load(ref), m)
            lines.append(f"{'':24s} vs {ref}: " + "  ".join(f"{k} {r[k]['diff']:+.1f} [{r[k]['ci'][0]:+.1f}, {r[k]['ci'][1]:+.1f}]" for k in ("storm_s", "delay_s") if k in r)
                         + f"  LoS {r['any_los']['rate_a']:.3f}->{r['any_los']['rate_b']:.3f} p={r['any_los']['p_mcnemar']:.2g}")  # fmt: skip
    return "\n".join(lines)


def _cell(m: dict) -> str:
    s, p = summary(m), per_scenario(m)
    return f"weather {s['storm_s'][0]:6.1f}±{s['storm_s'][1]:4.1f}  envelope {s['envelope_s'][0]:6.1f}  delay {s['delay_s'][0]:6.1f}±{s['delay_s'][1]:4.1f}  LoS {int(p['any_los'].sum()):4d}  completed {100 * s['exit_rate'][0]:5.1f}%  turns {s['turns'][0]:4.1f} ({s['turn_deg'][0]:4.0f} deg, spread {s['turn_spread'][0]:3.0f})  worst aircraft {s['worst_delay_s'][0]:5.1f} s"


def experiments_load(run_id: str) -> dict:
    exp, cfg = experiments.resolve(run_id)
    return json.loads((experiments.run_dir(exp, run_id, cfg) / "metrics.json").read_text())


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp")
    ap.add_argument("--ref", default=None)
    args = ap.parse_args(argv)
    print(table(args.exp, args.ref))


if __name__ == "__main__":
    main()
