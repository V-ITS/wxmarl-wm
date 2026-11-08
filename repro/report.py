"""Build the results report from whatever has finished: REPORT.pdf, REPORT.md and one file per figure.

Every number and figure comes from the run files, and a section with no runs yet says so, so the report is valid at
any point. Its numbers are the ones the paper reports.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle

from config import load_config
from core.impact import class_table
from core.weather import FAMILIES, severity
from env import WxEnv
from wm import stats
from wm.experiments import PACKAGES

BLUE, ORANGE, PURPLE, RED, GRAY, DARK = "#1f6fb4", "#e08a1e", "#7a3fa8", "#c0392b", "#8a8a8a", "#222222"
INK = "#333333"
PREDICTORS = (("E1-frozen", "frozen radar", GRAY, "s"), ("E1-extrap", "extrapolation", "#b5b5b5", "D"), ("E1-tracker", "tracker", PURPLE, "v"),
              ("E1-forecast-s0", "forecaster", BLUE, "o"), ("E1-perfect", "true weather state", DARK, "X"))  # fmt: skip
REFERENCES = (("BASE-avoid", "reactive avoid", ORANGE, "^"), ("BASE-qmix-s0", "QMIX", RED, "P"))
AIRBORNE = (("AIR-forecast", "fixed order + forecaster, sector radar", BLUE, "o"), ("AIR-local", "fixed order + forecaster, 20 NM radar", "#6fa8d6", "D"),
            ("AIR-tracker", "fixed order + tracker, sector radar", PURPLE, "v"), *REFERENCES)  # fmt: skip
HEAVY = (("E5-heavy-avoid", "reactive avoid", ORANGE, "^"), ("E5-heavy-qmix", "QMIX", RED, "P"), ("E5-heavy-frozen", "joint + frozen radar", GRAY, "s"), ("E5-heavy-tracker", "joint + tracker", PURPLE, "v"),
         ("E5-heavy-forecast", "joint + forecaster", BLUE, "o"), ("E5-heavy-air-forecast", "fixed order + forecaster, sector radar", "#6fa8d6", "D"), ("E5-heavy-air-local", "fixed order + forecaster, 20 NM radar", "#9c7bb8", "<"), ("E5-heavy-perfect", "true weather state", DARK, "X"))  # fmt: skip
COORD = (("shared", "simultaneous, shared plans", RED, "s"), ("direct", "assume others fly direct", ORANGE, "^"), ("fixed", "fixed order", BLUE, "o"), ("joint", "centralized joint", DARK, "X"))
CONTRASTS = (("RQ1", "E1-frozen", "E1-forecast-s0"), ("RQ1", "E1-extrap", "E1-forecast-s0"), ("RQ1", "E1-tracker", "E1-forecast-s0"), ("RQ1", "BASE-avoid", "E1-forecast-s0"),
             ("RQ1", "BASE-qmix-s0", "E1-forecast-s0"), ("RQ1", "E1-forecast-s0", "E1-perfect"),
             ("RQ2", "E3-N5-shared", "E3-N5-fixed"), ("RQ2", "E3-N5-direct", "E3-N5-fixed"), ("RQ2", "E3-N5-joint", "E3-N5-fixed"),
             ("RQ2", "E3-N3-shared", "E3-N3-fixed"), ("RQ2", "E3-N8-shared", "E3-N8-fixed"),
             ("Airborne", "E1-forecast-s0", "AIR-forecast"), ("Airborne", "AIR-tracker", "AIR-forecast"), ("Airborne", "AIR-forecast", "AIR-local"),
             ("Airborne", "BASE-avoid", "AIR-local"), ("Airborne", "BASE-qmix-s0", "AIR-local"), ("Airborne", "BASE-qmix-s0", "AIR-forecast"))  # fmt: skip

def style() -> None:
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#999999", "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": INK, "ytick.color": INK, "axes.grid": True, "grid.color": "#ececec", "grid.linewidth": 0.6, "axes.axisbelow": True, "pdf.fonttype": 42,
                         "legend.frameon": False, "legend.fontsize": 7.5})  # fmt: skip


def mean_ci(x: np.ndarray, seed: int = 0) -> tuple[float, float, float]:
    x = x[~np.isnan(x)]
    if not len(x):
        return np.nan, np.nan, np.nan
    b = np.random.default_rng(seed).choice(x, (1000, len(x))).mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


class Report:
    def __init__(self, root: Path, out: Path) -> None:
        self.root, self.out = root, out
        self.figs = out / "figures"
        self.figs.mkdir(parents=True, exist_ok=True)
        self.runs = {p.parent.name: json.loads(p.read_text()) for p in sorted(root.glob("*/*/metrics.json"))}
        self.done = {p.stem: json.loads(p.read_text()) for p in (root / "done").glob("*.json")}
        self.screens = stats.classes(root)
        self.md: list[str] = ["# WxMARL results\n"]
        self.pdf: PdfPages | None = None

    def values(self, run: str, key: str, w: int | None = None) -> np.ndarray:
        m = self.runs[run]
        cls = self.screens.get(m["env_cfg_hash"], {})
        part = stats.per_scenario(m)[key]
        if w is None:
            return part
        keep = np.array([cls.get((r["chunk_seed"], r["row"]), {}).get("w_class") == w for r in m["scenarios"]])
        return part[keep]

    def text(self, title: str, lines: list[str], mono: bool = True) -> None:
        self.md += [f"\n## {title}\n", *(["```", *lines, "```"] if mono else lines)]
        wrapped = [piece for ln in lines for piece in (textwrap.wrap(ln, 118, subsequent_indent="  ") or [""])] if not mono else lines
        per_page = 58
        for i in range(0, max(len(wrapped), 1), per_page):
            fig = plt.figure(figsize=(11.69, 8.27))
            fig.text(0.05, 0.94, title + (" (continued)" if i else ""), fontsize=13, weight="bold")
            size = min(8.0, 7.6 * 150 / max(150, max((len(s) for s in wrapped[i : i + per_page]), default=1))) if mono else 8.6
            fig.text(0.05, 0.90, "\n".join(wrapped[i : i + per_page]), va="top", family="monospace" if mono else "sans-serif", fontsize=size, linespacing=1.25)
            self.pdf.savefig(fig)
            plt.close(fig)

    def figure(self, fig, name: str, title: str, caption: str) -> None:
        fig.suptitle(title, fontsize=11, weight="bold", x=0.02, ha="left")
        fig.text(0.02, 0.005, "\n".join(textwrap.wrap(caption, 165)), fontsize=7.5, va="bottom", color="#555555")
        for ext in ("pdf", "png"):
            fig.savefig(self.figs / f"{name}.{ext}", dpi=200)
        self.pdf.savefig(fig)
        plt.close(fig)
        self.md += [f"\n## {title}\n", f"![{name}](figures/{name}.png)\n", caption]

    # ---------- sections ----------
    def gates(self) -> None:
        from repro.jobs import names

        left = [n for n in names() if n not in self.done]
        lines = [f"{len(self.done)} of {len(self.done) + len(left)} jobs finished." + (f" Left: {', '.join(left[:6])}{' ...' if len(left) > 6 else ''}" if left else ""), ""]
        commits = sorted({d.get("git", "?") for d in self.done.values()})
        hashes = {m["env_cfg_hash"] for n, m in self.runs.items() if n.split("-")[0] in ("BASE", "E1", "AIR")}
        lines += ["", "Checks", f"  commits used: {', '.join(commits) or '-'}" + ("   MORE THAN ONE: compare with care" if len(commits) > 1 else ""),
                  f"  weather settings across baselines, E1, airborne: {len(hashes)} distinct" + ("   SHOULD BE 1" if len(hashes) > 1 else ""),
                  "  scenarios: seeds from 200000 at 64 per chunk, classed by core.impact.screen"]  # fmt: skip
        seeds = {s: self.runs[f"E1-forecast-s{s}"] for s in range(3) if f"E1-forecast-s{s}" in self.runs}
        if len(seeds) > 1:
            lines.append("  forecaster seeds, time in weather: " + ", ".join(f"s{s} {np.nanmean(stats.per_scenario(m)['storm_s']):.1f} s" for s, m in seeds.items()))
        qm = {s: self.runs[f"BASE-qmix-s{s}"] for s in range(3) if f"BASE-qmix-s{s}" in self.runs}
        if len(qm) > 1:
            lines.append("  QMIX seeds, time in weather / delay: " + ", ".join(f"s{s} {np.nanmean(stats.per_scenario(m)['storm_s']):.1f} / {np.nanmean(stats.per_scenario(m)['delay_s']):.1f} s" for s, m in qm.items()))
        lines += ["", "Gates"]
        lines.append(f"  benchmark characterized by route impact: {'yes' if (self.root / PACKAGES['SUITE'] / 'SUITE-N5' / 'suite.json').exists() else 'pending'}")
        verdict = "pending"
        if {"E1-frozen", "E1-forecast-s0"} <= set(self.runs) and self.screens:
            good = []
            for w in range(4):
                keep = lambda r, w=w: self.screens.get(self.runs["E1-frozen"]["env_cfg_hash"], {}).get((r["chunk_seed"], r["row"]), {}).get("w_class") == w
                a, b = stats.subset(self.runs["E1-frozen"], keep), stats.subset(self.runs["E1-forecast-s0"], keep)
                if len(a["scenarios"]) > 1 and len(b["scenarios"]) > 1:
                    r = stats.paired(a, b)
                    safe = r["any_los"]["b01"] <= r["any_los"]["b10"] or r["any_los"]["p_mcnemar"] >= 0.05
                    if r["storm_s"]["ci"][1] < 0 and safe:
                        good.append(w)
                    lines.append(f"  forecaster vs frozen, W{w}: {r['storm_s']['diff']:+.1f} s [{r['storm_s']['ci'][0]:+.1f}, {r['storm_s']['ci'][1]:+.1f}]; separation {r['any_los']['b10']} better, {r['any_los']['b01']} worse")
            verdict = "GO: benefit outside the stress class" if any(w < 3 for w in good) else "REVISE CLAIMS: benefit only in W3" if good else "REVISE CLAIMS: no measured benefit"
        lines.append(f"  learned predictor helps without hidden safety loss: {verdict}")
        if {"E3-N5-shared", "E3-N5-fixed"} <= set(self.runs):
            r = stats.paired(self.runs["E3-N5-shared"], self.runs["E3-N5-fixed"])["any_los"]
            lines.append(f"  coordination comparison is paired: {r['rate_a'] * r['n']:.0f} -> {r['rate_b'] * r['n']:.0f} of {r['n']} scenarios, p={r['p_mcnemar']:.2g}")
        else:
            lines.append("  coordination comparison is paired: pending")
        p = self.root / PACKAGES["E4"] / "RANK" / "ranking.json"
        if p.exists():
            lines.append(f"  end-to-end model interpreted fairly: state error {json.loads(p.read_text())['state_error_nm']:.1f} NM before predicting; if large, report it as exploratory, not as evidence against end-to-end")
        else:
            lines.append("  end-to-end model interpreted fairly: pending")
        lines += ["", "Result packages"] + [f"  {v}" for k, v in PACKAGES.items() if k != "DATA"] + ["  report  (this file, figures/)"]
        self.text("Status, checks and gates", lines)

    def environment(self) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(11.69, 4.6), constrained_layout=False)
        fig.subplots_adjust(left=0.03, right=0.98, top=0.86, bottom=0.14, wspace=0.08)
        cmap = LinearSegmentedColormap.from_list("wx", ["#fbe4df", "#e8a598", "#c0392b", "#7a1f15"])
        for ax, fam in zip(axes, FAMILIES, strict=True):
            cfg = load_config("default", {"weather": {"family": fam}})
            cfg.obs.mode = "scan"
            env = WxEnv(cfg, batch_size=16)
            env.reset(seed=810_000)
            b = int((env.cells.alive.sum(1).float() - env.cells.alive.sum(1).float().median()).abs().argmin())
            L = cfg.domain.L
            g = torch.linspace(0, L, 300)
            grid = torch.stack(torch.meshgrid(g, g, indexing="xy"), -1).reshape(1, -1, 2)
            one = type(env.cells)(**{k: v[b : b + 1] for k, v in vars(env.cells).items()})
            sev = severity(grid, one).reshape(300, 300).numpy()
            ax.imshow(np.ma.masked_equal(sev, 0), origin="lower", extent=(0, L, 0, L), cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
            for k in env.cells.alive[b].nonzero().flatten().tolist():
                q, r = env.cells.q[b, k].numpy(), float(env.cells.rho[b, k])
                ax.add_patch(Circle(q, r + cfg.weather.envelope_nm, fill=False, ec=GRAY, lw=0.6, ls=(0, (3, 2))))
            for k in env.radar.alive[b].nonzero().flatten().tolist():
                ax.add_patch(Circle(env.radar.q[b, k].numpy(), float(env.radar.rho[b, k]), fill=False, ec=DARK, lw=0.8))
            for n in range(env.N):
                p, e = env.ac.p0[b, n].numpy(), env.ac.e[b, n].numpy()
                ax.annotate("", xy=e, xytext=p, arrowprops={"arrowstyle": "-|>", "color": BLUE, "lw": 1.0})
                ax.plot(*p, "o", color=BLUE, ms=3.5)
            ax.set(xlim=(0, L), ylim=(0, L), xticks=[], yticks=[], aspect="equal", title=fam)
            ax.grid(False)
        handles = [plt.Rectangle((0, 0), 1, 1, fc="#c0392b", label="weather-affected region (true)"), plt.Line2D([], [], color=DARK, lw=0.8, label="latest radar picture"),
                   plt.Line2D([], [], color=GRAY, lw=0.6, ls=(0, (3, 2)), label="5 NM avoidance envelope"), plt.Line2D([], [], color=BLUE, marker="o", ms=3.5, label="direct routes")]  # fmt: skip
        fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.05))
        self.figure(fig, "fig1_environment", "Fig. 1  Weather families", "Synthetic evolving regions for studying imperfect information, not a weather forecast. Representative moments, 150 x 150 NM, five aircraft.")

    def table_one(self) -> None:
        p = self.root / PACKAGES["SUITE"] / "SUITE-N5" / "suite.json"
        if not p.exists():
            return self.text("Table 1  Scenario suite", ["Not screened yet."])
        rows = json.loads(p.read_text())["scenarios"]
        sc = {k: torch.tensor([r[k] for r in rows]) for k in ("w_class", "tracks_envelope", "detour", "feasible", "coverage", "envelope_coverage", "wind_kt", "cells", "family")}
        lines = [f"{'class':6s}{'scen.':>7s}{'hit rate':>10s}{'detour mean/p90/max':>24s}{'infeasible':>12s}{'coverage':>10s}{'envelope':>10s}{'cells p50':>11s}{'wind p50':>10s}   isolated/clustered/banded"]
        for name, r in class_table(sc).items():
            lines.append(f"{name:6s}{r['scenarios']:7d}{r['hit_rate']:10.2f}{r['detour_mean']:13.3f}/{r['detour_p90']:.3f}/{r['detour_max']:.2f}{r['infeasible_share']:12.3f}{r['coverage']:10.3f}{r['envelope_coverage']:10.3f}{r['cells_p50']:11.1f}{r['wind_kt_p50']:10.1f}   {r['isolated']}/{r['clustered']}/{r['banded']}")
        lines += ["", f"Balanced selection for pooled numbers: {stats.BALANCED}. Classes count direct tracks entering the envelope; set for N = 5 (core/impact.py)."]
        self.text("Table 1  Scenario suite by route impact", lines)

    def by_class(self, series, name: str, title: str, caption: str) -> None:
        series = [s for s in series if s[0] in self.runs]
        if not series:
            return self.text(title, ["No runs yet."])
        metrics = (("storm_s", "time in weather (s per flight)", 1), ("envelope_s", "time in envelope (s per flight)", 1), ("delay_s", "delay (s)", 1), ("any_los", "scenarios losing separation (%)", 100))
        fig, axes = plt.subplots(1, 4, figsize=(11.69, 5.0))
        fig.subplots_adjust(left=0.05, right=0.985, top=0.86, bottom=0.27, wspace=0.28)
        width = 0.72 / max(len(series), 1)
        for ax, (key, label, scale) in zip(axes, metrics, strict=True):
            for i, (run, _, col, mk) in enumerate(series):
                for w in range(4):
                    m, lo, hi = mean_ci(self.values(run, key, w))
                    x = w + (i - (len(series) - 1) / 2) * width
                    ax.errorbar(x, m * scale, yerr=[[(m - lo) * scale], [(hi - m) * scale]], color=col, marker=mk, ms=4.5, lw=1.0, capsize=0, mec="white", mew=0.5)
            ax.set(xticks=range(4), xticklabels=[f"W{w}" for w in range(4)], ylabel=label)
            ax.grid(axis="x", visible=False)
            ax.set_ylim(bottom=0)
        fig.legend(handles=[plt.Line2D([], [], color=c, marker=m, ms=5, lw=1, label=lab) for _, lab, c, m in series], loc="lower center", ncol=min(len(series), 4), bbox_to_anchor=(0.5, 0.08))
        self.figure(fig, name, title, caption)

    def forecast_quality(self) -> None:
        p = self.root / PACKAGES["E2"] / "WX-eval" / "eval.json"
        if not p.exists():
            return self.text("Fig. 3  Forecast quality", ["No runs yet."])
        beliefs = json.loads(p.read_text())["beliefs"]
        avg = lambda name, h, key: float(np.mean([d[key] for d in beliefs[name][h].values() if key in d])) if name in beliefs and any(key in d for d in beliefs[name][h].values()) else np.nan
        hs = list(next(iter(beliefs.values())))
        minutes = [int(h[1:-1]) / 60 for h in hs]
        series = (("frozen radar", GRAY, "s"), ("extrapolation", "#b5b5b5", "D"), ("tracker", PURPLE, "v"), ("WX-sector-s0", BLUE, "o"), ("true state", DARK, "X"))
        fig, axes = plt.subplots(1, 3, figsize=(11.69, 4.6))
        fig.subplots_adjust(left=0.05, right=0.985, top=0.86, bottom=0.25, wspace=0.22)
        for ax, (key, label) in zip(axes, (("iou", "overlap with the true footprint"), ("centre_nm", "centre error, tracked cells (NM)"), ("radius_nm", "radius error, tracked cells (NM)")), strict=True):
            for name, col, mk in series:
                ax.plot(minutes, [avg(name, h, key) for h in hs], color=col, marker=mk, ms=4.5, lw=1.5)
            ax.set(xlabel="minutes ahead", ylabel=label, xticks=minutes)
        names = {"WX-sector-s0": "forecaster"}
        fig.legend(handles=[plt.Line2D([], [], color=c, marker=m, ms=5, lw=1.5, label=names.get(n, n)) for n, c, m in series], loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.08))
        self.figure(fig, "fig3_forecast_quality", "Fig. 3  Forecast quality", "Mean over the three weather families. Overlap is limited by cells born after the latest radar picture, which no radar-based predictor can see; see the table.")
        lines = [f"{'predictor':28s}{'ahead':>8s}{'family':>11s}{'overlap':>9s}{'missed':>8s}{'centre NM':>11s}{'radius NM':>11s}{'survival err':>14s}"]
        for name, per_h in beliefs.items():
            for h, fams in per_h.items():
                for fam, d in fams.items():
                    if name == "born after the picture":
                        lines.append(f"{name:28s}{h:>8s}{fam:>11s}   share of the true footprint {d['share_of_footprint']:.3f}")
                    else:
                        fmt = lambda k, d=d: f"{d[k]:.3f}" if k in d else "-"
                        lines.append(f"{name:28s}{h:>8s}{fam:>11s}{fmt('iou'):>9s}{fmt('miss'):>8s}{fmt('centre_nm'):>11s}{fmt('radius_nm'):>11s}{fmt('survival_err'):>14s}")
        self.text("Table 2  Forecast quality by family", lines)

    def coordination(self) -> None:
        series = [(f"E3-N5-{k}", lab, c, m) for k, lab, c, m in COORD if f"E3-N5-{k}" in self.runs]
        if not series:
            return self.text("Fig. 4  Coordination", ["No runs yet."])
        fig, (a, b, c) = plt.subplots(1, 3, figsize=(11.69, 4.6))
        fig.subplots_adjust(left=0.05, right=0.985, top=0.86, bottom=0.25, wspace=0.22)
        w_ = 0.72 / len(series)
        for i, (run, _, col, mk) in enumerate(series):
            for w in range(4):
                m, lo, hi = mean_ci(self.values(run, "any_los", w))
                a.errorbar(w + (i - (len(series) - 1) / 2) * w_, m * 100, yerr=[[(m - lo) * 100], [(hi - m) * 100]], color=col, marker=mk, ms=5, lw=1)
                m, lo, hi = mean_ci(self.values(run, "delay_s", w))
                b.errorbar(w + (i - (len(series) - 1) / 2) * w_, m, yerr=[[m - lo], [hi - m]], color=col, marker=mk, ms=5, lw=1)
        a.set(xticks=range(4), xticklabels=[f"W{w}" for w in range(4)], ylabel="scenarios losing separation (%), N = 5", ylim=(0, None))
        b.set(xticks=range(4), xticklabels=[f"W{w}" for w in range(4)], ylabel="delay (s), N = 5")
        for k, _, col, mk in COORD:
            pts = [(n, mean_ci(stats.per_scenario(self.runs[f"E3-N{n}-{k}"])["any_los"])) for n in (3, 5, 8) if f"E3-N{n}-{k}" in self.runs]
            if pts:
                c.errorbar([n for n, _ in pts], [v[0] * 100 for _, v in pts], yerr=[[(v[0] - v[1]) * 100 for _, v in pts], [(v[2] - v[0]) * 100 for _, v in pts]], color=col, marker=mk, ms=5, lw=1.5)
        c.set(xticks=(3, 5, 8), xlabel="aircraft", ylabel="scenarios losing separation (%)", ylim=(0, None))
        fig.legend(handles=[plt.Line2D([], [], color=x, marker=m, ms=5, lw=1, label=l) for _, l, x, m in COORD], loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.08))
        self.figure(fig, "fig4_coordination", "Fig. 4  With the weather known exactly, what coordination is needed?", "Same scenarios and exact weather, so weather prediction error is zero. Fixed order is a coordination baseline, not a proposed protocol; it assumes instant, lossless plan sharing and is unequal between aircraft.")

    def attribution(self) -> None:
        p = self.root / PACKAGES["E4"] / "RANK" / "ranking.json"
        if not p.exists():
            return self.text("Fig. 6  Partial vs full world model", ["No runs yet."])
        r = json.loads(p.read_text())
        items = sorted(r["tau_all" if not r["near_weather_moments"] else "tau_near"].items(), key=lambda kv: kv[1])
        fig, ax = plt.subplots(figsize=(11.69, 4.2))
        fig.subplots_adjust(left=0.14, right=0.97, top=0.86, bottom=0.2)
        ax.hlines(range(len(items)), 0, [v for _, v in items], color=BLUE, lw=2)
        ax.plot([v for _, v in items], range(len(items)), "o", color=BLUE, ms=6)
        ax.set(yticks=range(len(items)), yticklabels=[k for k, _ in items], xlabel="agreement with the true plan ranking (Kendall tau; 0 is random)")
        ax.grid(axis="y", visible=False)
        err = "n/a" if r["state_error_nm"] is None else f"{r['state_error_nm']:.1f} NM"
        self.figure(fig, "fig6_attribution", "Fig. 6  Partial vs full world model", f"{r['near_weather_moments']} moments near weather of {r['moments']}. The end-to-end model's aircraft are already {err} from the truth before predicting, so a weak score may reflect reconstruction, not learning everything in principle. One seed; exploratory.")

    def difficulty(self) -> None:
        from wm.experiments import FACTORS

        names = [*FACTORS, "N3", "N8"]
        rows = [(n, c, stats.paired(self.runs[f"E5-{n}-{c}"], self.runs[f"E5-{n}-forecast"])["storm_s"]) for n in names for c in ("frozen", "tracker")
                if f"E5-{n}-{c}" in self.runs and f"E5-{n}-forecast" in self.runs]  # fmt: skip
        if not rows:
            return self.text("Fig. 7  Difficulty factors", ["No runs yet."])
        fig, ax = plt.subplots(figsize=(11.69, 5.0))
        fig.subplots_adjust(left=0.1, right=0.97, top=0.88, bottom=0.2)
        y = {n: i for i, n in enumerate(names)}
        for n, c, r in rows:
            off, col, mk = (0.14, GRAY, "s") if c == "frozen" else (-0.14, PURPLE, "v")
            ax.errorbar(r["diff"], y[n] + off, xerr=[[r["diff"] - r["ci"][0]], [r["ci"][1] - r["diff"]]], color=col, marker=mk, ms=5, lw=1)
        ax.axvline(0, color="#999999", lw=0.8)
        ax.set(yticks=list(y.values()), yticklabels=names, xlabel="forecaster minus comparison: time in weather (s per flight); below zero, the forecaster spends less")
        ax.grid(axis="y", visible=False)
        fig.legend(handles=[plt.Line2D([], [], color=GRAY, marker="s", ms=5, lw=1, label="against frozen radar"), plt.Line2D([], [], color=PURPLE, marker="v", ms=5, lw=1, label="against the tracker")], loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.06))
        self.figure(fig, "fig7_difficulty", "Fig. 7  Which uncertainty makes forecasting valuable?", "One factor at a time on 256 scenarios: hazard amount, wind, radar age, then traffic. Forecasters and QMIX are not retrained for these settings.")

    def contrasts(self) -> None:
        lines = [f"{'group':9s}{'run':18s}{'reference':18s}{'n':>5s}  {'weather (s)':24s}{'envelope (s)':24s}{'delay (s)':24s}separation"]
        for group, ref, run in CONTRASTS:
            if ref in self.runs and run in self.runs:
                r = stats.paired(self.runs[ref], self.runs[run])
                cell = lambda k, r=r: f"{r[k]['diff']:+.1f} [{r[k]['ci'][0]:+.1f}, {r[k]['ci'][1]:+.1f}]" if k in r else "-"
                los = r["any_los"]
                lines.append(f"{group:9s}{run:18s}{ref:18s}{los['n']:5d}  {cell('storm_s'):24s}{cell('envelope_s'):24s}{cell('delay_s'):24s}{los['rate_a'] * los['n']:.0f} -> {los['rate_b'] * los['n']:.0f} (p={los['p_mcnemar']:.2g})")
        self.text("Table 3  Paired comparisons (run minus reference, 95% bootstrap interval)", lines if len(lines) > 1 else ["No pairs yet."])

    def packages(self) -> None:
        for exp, title in (("BASE", "Baselines"), ("E1", "RQ1 forecasting"), ("E3", "RQ2 coordination"), ("AIR", "Airborne system"), ("E4", "End-to-end in flight"), ("E5", "Difficulty factors")):
            if any(self.runs[n]["exp"] == exp for n in self.runs):
                self.text(f"Package {PACKAGES[exp]}: {title}", stats.table(exp, None, self.root).splitlines())

    def build(self) -> Path:
        style()
        path = self.out / "REPORT.pdf"
        with PdfPages(path) as self.pdf:
            self.gates()
            self.environment()
            self.table_one()
            self.by_class((*PREDICTORS, *REFERENCES), "fig2_rq1_forecasting", "Fig. 2  RQ1: when does forecasting help?", "Same scenarios, joint planner, so coordination does not mask the weather. Means with 95% bootstrap intervals, by route-impact class. QMIX and avoid see 20 NM radar; the planner runs see the sector and know all traffic.")
            self.forecast_quality()
            self.coordination()
            self.by_class(AIRBORNE, "fig5_airborne", "Fig. 5  The airborne system against the baselines", "Each aircraft decides, in fixed order, with weather it could really have. The planner also knows all traffic and shares plans instantly at any range, which QMIX does not; only the weather view is matched.")
            self.attribution()
            self.difficulty()
            self.by_class(HEAVY, "fig8_heavy", "Fig. 8  Heavy weather: three times the births and the wind", "256 scenarios. Forecasters and QMIX are as trained, not retrained for this weather. The true weather state is the ceiling a perfect weather model could reach. Only the weather view is matched between QMIX and the 20 NM planner.")
            self.contrasts()
            self.packages()
        (self.out / "REPORT.md").write_text("\n".join(self.md) + "\n")
        return path


def build(root: Path) -> Path:
    return Report(root, root / "report").build()


if __name__ == "__main__":
    from repro.jobs import results_root

    print(build(results_root()))
