"""The paper's figures, from the simulator and the results.

    python -m repro.trace        # once: the replayed encounters behind coord_trace (a few minutes on a GPU)
    python -m repro.figures      # results/redesign/paper_figures/{sim_overview,predictors,levels,route_impact,coordination,coord_trace}.{pdf,png}
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Polygon

from baselines import POLICIES
from config import load_config
from core.observation import GRID_PLANES, build_grid
from core.weather import severity
from env import WxEnv
from repro.jobs import results_root
from repro.report import BLUE, ORANGE, PURPLE, RED, mean_ci
from wm import experiments, stats
from wm.weather import simulate

OUT = results_root() / "paper_figures"
TEXT_W = 7.16
RADAR, AIRCRAFT, TRUTH, DARK = "#333333", "#555555", "#8c2d1f", "#111111"
LEVELS = (("default", "BASE-avoid"), ("births x2", "E5-amount-hi-avoid"), ("wind 30-50 kt", "E5-wind-hi-avoid"), ("heavy: births x3, wind 50-70 kt", "E5-heavy-avoid"))
SERIES = (("reactive avoid, 20 NM radar", ORANGE, "^", ("BASE-avoid", "E5-amount-hi-avoid", "E5-wind-hi-avoid", "E5-heavy-avoid")),
          ("QMIX, 20 NM radar", RED, "P", ("BASE-qmix-s0", "E5-amount-hi-qmix", "E5-wind-hi-qmix", "E5-heavy-qmix")),
          ("planner + tracker", PURPLE, "v", ("E1-tracker", "E5-amount-hi-tracker", "E5-wind-hi-tracker", "E5-heavy-tracker")),
          ("planner + forecaster", BLUE, "o", ("E1-forecast-s0", "E5-amount-hi-forecast", "E5-wind-hi-forecast", "E5-heavy-forecast")),
          ("fixed order + forecaster, 20 NM radar", "#6fa8d6", "D", ("AIR-local", None, None, "E5-heavy-air-local")),
          ("planner + true weather state", DARK, "X", ("E1-perfect", None, None, "E5-heavy-perfect")))  # fmt: skip


def _style() -> None:
    plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 7, "axes.linewidth": 0.5,
                         "xtick.major.width": 0.5, "ytick.major.width": 0.5, "pdf.fonttype": 42, "font.family": "serif"})  # fmt: skip


def _storm_cmap():
    cm = matplotlib.colors.LinearSegmentedColormap.from_list("storm", ["#f4cbc4", "#d9685a", TRUTH])
    cm.set_under("white")
    return cm


def _cells(ax, q, rho, alive, **kw) -> None:
    for i in np.flatnonzero(alive):
        ax.add_patch(Circle(q[i], rho[i], **kw))


def _save(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=250)
    plt.close(fig)


def _busiest_view(cfg, seed: int, B: int, times: tuple[int, ...]) -> tuple[int, int, int]:
    """(row, t, aircraft) with a neighbour and the most weather in its view."""
    G = cfg.obs.grid_n
    env = WxEnv(cfg, batch_size=B, seed=seed)
    *_, avail = env.reset()
    best, pick = -1.0, (0, times[0], 0)
    for t in range(max(times) + 1):
        if t in times:
            obs = build_grid(env.ac, env.radar, env.radar_prev, t, cfg)[..., : GRID_PLANES * G * G]
            planes = obs.reshape(B, -1, GRID_PLANES, G, G)
            score = (planes[:, :, 0] > 0).sum((-1, -2)).float() * (planes[:, :, 2] > 0.5).any(-1).any(-1) * env.ac.alive
            v, i = score.flatten().max(0)
            if float(v) > best:
                best, pick = float(v), (int(i) // planes.shape[1], t, int(i) % planes.shape[1])
        *_, avail, _ = env.step(POLICIES["avoid"](env, avail, None))
    return pick


def _frame(cfg, seed: int, row: int, B: int, t0: int) -> dict:
    env = WxEnv(cfg, batch_size=B, seed=seed)
    *_, avail = env.reset()
    trail = []
    for t in range(t0 + 1):
        trail.append(env.ac.p[row].clone())
        a = POLICIES["avoid"](env, avail, None)
        if t == t0:
            one = lambda c: type(c)(**{k: v[row : row + 1].clone() for k, v in vars(c).items()})
            return {"cells": one(env.cells), "radar": one(env.radar), "prev": one(env.radar_prev), "ac": one(env.ac), "action": a[row].clone(), "trail": torch.stack(trail).numpy()}
        *_, avail, _ = env.step(a)
    raise ValueError(t0)


def _sector(ax, f: dict, cfg, sel: int) -> None:
    L, oob = cfg.domain.L, cfg.domain.oob_margin
    g = torch.linspace(-oob, L + oob, 340)
    pts = torch.stack(torch.meshgrid(g, g, indexing="xy"), -1).reshape(1, -1, 2)
    ax.imshow(severity(pts, f["cells"]).reshape(340, 340).numpy(), origin="lower", extent=(-oob, L + oob, -oob, L + oob), cmap=_storm_cmap(), vmin=1e-6, vmax=1, zorder=0, interpolation="nearest")
    ax.add_patch(plt.Rectangle((0, 0), L, L, fill=False, ec="0.35", lw=0.6, zorder=1))
    ax.add_patch(plt.Rectangle((-oob, -oob), L + 2 * oob, L + 2 * oob, fill=False, ec="0.55", lw=0.5, ls=":", zorder=1))
    for c, ls in ((f["prev"], ":"), (f["radar"], "-")):
        _cells(ax, c.q[0].numpy(), c.rho[0].numpy(), c.alive[0].numpy(), fill=False, ec=RADAR, lw=0.6, ls=ls, zorder=2)
    ac, on = f["ac"], f["ac"].alive[0].numpy()
    p, psi, e = ac.p[0].numpy(), ac.psi[0].numpy(), ac.e[0].numpy()
    for i in np.flatnonzero(on):
        ax.plot(*f["trail"][:, i].T, color=AIRCRAFT, lw=0.45, alpha=0.6, zorder=3)
        ax.plot(*e[i], "x", ms=2.6, mew=0.5, color="0.4", zorder=3)
    R, h = cfg.obs.sense_range_nm, 300.0 * ac.v[0, sel].item() / 3600.0
    ax.add_patch(Circle(p[sel], R, fc=BLUE, alpha=0.07, lw=0, zorder=2))
    ax.add_patch(Circle(p[sel], R, fill=False, ec=BLUE, lw=0.6, zorder=3))
    half = cfg.obs.grid_n * cfg.obs.grid_res_nm / 2
    ca, sa = np.cos(psi[sel]), np.sin(psi[sel])
    corners = np.array([[half, half], [half, -half], [-half, -half], [-half, half]]) @ np.array([[ca, sa], [-sa, ca]]) + p[sel]
    ax.add_patch(Polygon(corners, closed=True, fill=False, ec=BLUE, lw=0.5, ls="--", zorder=3))
    ax.plot(*np.stack((p[sel], e[sel])).T, color="0.3", lw=0.5, ls=":", zorder=3)
    for k, off in enumerate(np.deg2rad(cfg.action.heading_offsets_deg)):
        end = p[sel] + h * np.array([np.cos(psi[sel] + off), np.sin(psi[sel] + off)])
        ax.plot(*np.stack((p[sel], end)).T, color=BLUE, zorder=4, lw=1.3 if k == int(f["action"][sel]) else 0.4)
    tri = np.array([[1.0, 0.0], [-0.65, 0.6], [-0.3, 0.0], [-0.65, -0.6]]) * 2.4
    for i in np.flatnonzero(on):
        ca, sa = np.cos(psi[i]), np.sin(psi[i])
        ax.add_patch(Polygon(tri @ np.array([[ca, sa], [-sa, ca]]) + p[i], closed=True, lw=0, zorder=5, fc=BLUE if i == sel else AIRCRAFT))
    ax.set(xlim=(-oob - 1, L + oob + 1), ylim=(-oob - 1, L + oob + 1), aspect="equal", xticks=[], yticks=[])
    for sp in ax.spines.values():
        sp.set_visible(False)


def sim_overview(seed: int = 200_000, B: int = 64) -> dict:
    """(a) The sector at one moment. (b) The grid observation of one aircraft, read from the simulator."""
    cfg = load_config("default")
    row, t0, sel = _busiest_view(cfg, seed, B, (19, 29, 39, 49))
    f = _frame(cfg, seed, row, B, t0)
    G, R = cfg.obs.grid_n, cfg.obs.sense_range_nm
    obs = build_grid(f["ac"], f["radar"], f["prev"], t0, cfg)[0]
    planes = obs[:, : GRID_PLANES * G * G].reshape(-1, GRID_PLANES, G, G).numpy()
    fig = plt.figure(figsize=(TEXT_W, 2.75))
    gs = GridSpec(2, 4, figure=fig, width_ratios=[2.2, 1, 1, 1], wspace=0.05, hspace=0.22, left=0.005, right=0.995, top=0.85, bottom=0.15)
    _sector(fig.add_subplot(gs[:, 0]), f, cfg, sel)
    storm = _storm_cmap()
    show = [("weather, latest radar", storm, 1e-6, 1), ("weather, previous radar", storm, 1e-6, 1), ("other aircraft", "Greys", 0, 1),
            ("their velocity, right", "RdBu_r", -1, 1), ("their velocity, ahead", "RdBu_r", -1, 1), ("beyond the wall", "Greys", 0, 1)]  # fmt: skip
    half = G * cfg.obs.grid_res_nm / 2
    for n, (lab, cm, lo, hi) in enumerate(show):
        ax = fig.add_subplot(gs[n // 3, n % 3 + 1])
        ax.imshow(planes[sel, n], cmap=cm, vmin=lo, vmax=hi, extent=(-half, half, -half, half), origin="upper", interpolation="nearest")
        ax.add_patch(Circle((0, 0), R, fill=False, ec=BLUE, lw=0.5))
        ax.add_patch(Polygon(np.array([[0, 1.6], [1.1, -1.1], [0, -0.5], [-1.1, -1.1]]), closed=True, fc=BLUE, lw=0))
        ax.set(xticks=[], yticks=[])
        ax.set_title(lab, loc="left", pad=1.5, fontsize=6)
        for sp in ax.spines.values():
            sp.set(visible=True, lw=0.5, color="0.5")
    age = (t0 % cfg.radar_steps) * cfg.domain.dt
    fig.text(0.008, 0.985, f"(a) the sector, radar picture {age:.0f} s old", fontsize=7, va="top")
    fig.text(0.42, 0.985, "(b) what the blue aircraft observes, heading up", fontsize=7, va="top")
    handles = [plt.Rectangle((0, 0), 1, 1, fc="#d9685a", label="true weather"), plt.Line2D([], [], color=RADAR, lw=0.6, label="latest radar"),
               plt.Line2D([], [], color=RADAR, lw=0.6, ls=":", label="previous radar"), plt.Line2D([], [], color=BLUE, lw=0.6, label="20 NM range"),
               plt.Line2D([], [], color=BLUE, lw=0.5, ls="--", label="image window"), plt.Line2D([], [], color=BLUE, lw=1.3, label="7 turns, chosen thick"),
               plt.Line2D([], [], color="0.3", lw=0.5, ls=":", label="direct to exit"), plt.Line2D([], [], color="0.4", marker="x", ms=2.6, mew=0.5, ls="", label="exit"),
               plt.Line2D([], [], color=AIRCRAFT, lw=0.45, label="flown track")]  # fmt: skip
    fig.legend(handles=handles, loc="lower center", ncol=9, frameon=False, fontsize=5.4, bbox_to_anchor=(0.5, 0.0), columnspacing=0.8, handlelength=1.5)
    fig.text(0.42, 0.1, "plus 4 numbers: distance to exit, exit bearing (cos, sin), radar age", fontsize=6, color="0.2")
    _save(fig, "sim_overview")
    return {"seed": seed, "row": row, "t": t0, "aircraft": sel, "radar_age_s": age, "cells": int(f["cells"].alive.sum())}


def _moment(ax, run: str, title: str, seed: int = 410_000, B: int = 256, t0: int = 39) -> dict:
    """A typical moment: the row closest to the setting's median cell count and cover."""
    cfg = experiments.resolve(run)[1].env_cfg()
    L = cfg.domain.L
    g = torch.linspace(1, L - 1, 75)
    pts = torch.stack(torch.meshgrid(g, g, indexing="xy"), -1).reshape(-1, 2).unsqueeze(0).expand(B, -1, -1)
    truth, radar, _ = simulate(cfg, B, t0 + 1, seed)[t0]
    n, a = truth.alive.float().sum(-1), (severity(pts, truth) > 0).float().mean(-1)
    row = int(((n - n.median()).abs() / n.std().clamp_min(1e-6) + (a - a.median()).abs() / a.std().clamp_min(1e-6)).argmin())
    fine = torch.linspace(0, L, 300)
    grid = torch.stack(torch.meshgrid(fine, fine, indexing="xy"), -1).reshape(1, -1, 2)
    one = type(truth)(**{k: v[row : row + 1] for k, v in vars(truth).items()})
    ax.imshow(severity(grid, one).reshape(300, 300).numpy(), origin="lower", extent=(0, L, 0, L), cmap=_storm_cmap(), vmin=1e-6, vmax=1, interpolation="nearest")
    _cells(ax, radar.q[row].numpy(), radar.rho[row].numpy(), radar.alive[row].numpy(), fill=False, ec=RADAR, lw=0.55)
    ax.set(xlim=(0, L), ylim=(0, L), aspect="equal", xticks=[], yticks=[])
    ax.set_title(f"{title}\n{float(n.mean()):.1f} cells, {100 * float(a.mean()):.0f}% of sector", pad=2, fontsize=6.3)
    for sp in ax.spines.values():
        sp.set(visible=True, lw=0.5, color="0.4")
    return {"cells": float(n.mean()), "cover": float(a.mean())}


def sweep() -> dict:
    """Top: a typical moment per weather setting. Bottom: time in weather and delay of each controller, first 256 scenarios."""
    root = results_root()
    runs = {p.parent.name: json.loads(p.read_text()) for p in root.glob("[01]*/*/metrics.json")}
    fig = plt.figure(figsize=(TEXT_W, 3.5))
    top = GridSpec(1, 4, figure=fig, left=0.03, right=0.97, top=0.93, bottom=0.57, wspace=0.05)
    meta = {lab: _moment(fig.add_subplot(top[0, i]), run, lab) for i, (lab, run) in enumerate(LEVELS)}
    low = GridSpec(1, 2, figure=fig, left=0.06, right=0.99, top=0.5, bottom=0.17, wspace=0.17)
    xs = np.arange(len(LEVELS))
    width = 0.12
    for j, (key, ylabel) in enumerate((("storm_s", "time in weather (s per flight)"), ("delay_s", "delay (s)"))):
        ax = fig.add_subplot(low[0, j])
        for k, (label, color, marker, names) in enumerate(SERIES):
            for x, name in zip(xs, names, strict=True):
                if name is None:
                    continue
                first = {**runs[name], "scenarios": runs[name]["scenarios"][: experiments.SMALL]}
                m, lo, hi = mean_ci(stats.per_scenario(first)[key])
                off = x + (k - 2.5) * width
                ax.plot([off, off], [lo, hi], color=color, lw=0.8)
                ax.plot(off, m, marker, color=color, ms=3.4, mec="white", mew=0.3)
        ax.set_xticks(xs, ["default", "births x2", "wind 30-50 kt", "heavy"])
        ax.set_ylabel(ylabel, labelpad=1.5)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", lw=0.3, color="0.88")
        ax.set_axisbelow(True)
    handles = [plt.Line2D([], [], color=c, marker=mk, ms=3.6, lw=0.8, mec="white", mew=0.3, label=lab) for lab, c, mk, _ in SERIES]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=6.3, bbox_to_anchor=(0.5, 0.0), columnspacing=1.5)
    _save(fig, "sweep")
    return meta


QMIX_C = "#8c564b"  # brown: red is reserved for numbers still to rerun
BY_CLASS = (("E1-frozen", "frozen radar", "#7f7f7f", "s"), ("E1-extrap", "extrapolation", "#17becf", "D"), ("E1-tracker", "tracker", PURPLE, "v"),
            ("E1-forecast-s0", "forecaster", BLUE, "o"), ("E1-perfect", "true weather state", DARK, "X"),
            ("BASE-avoid", "reactive avoid (20 NM view)", ORANGE, "^"))  # fmt: skip
COORD = (("none", "fly direct", "#8a8a8a", "s"), ("shared", "simultaneous", BLUE, "o"), ("direct", "assume direct", "#6fa8d6", "D"),
         ("fixed", "fixed order", ORANGE, "^"), ("joint", "joint (reference)", DARK, "X"))  # fmt: skip


def _by_class(runs: dict, screens: dict, run: str, key: str, w: int) -> np.ndarray:
    m = runs[run]
    cls = screens.get(m["env_cfg_hash"], {})
    keep = np.array([cls.get((r["chunk_seed"], r["row"]), {}).get("w_class") == w for r in m["scenarios"]])
    return stats.per_scenario(m)[key][keep]


def route_impact() -> dict:
    """RQ1: each weather predictor and the reactive rule by route-impact class, balanced suite (128/384/384/128)."""
    root = results_root()
    runs = {p.parent.name: json.loads(p.read_text()) for p in root.glob("0[45]*/*/metrics.json")}
    screens = stats.classes(root)
    metrics = (("storm_s", "(a) time in weather (s per flight)"), ("envelope_s", "(b) time in envelope (s per flight)"), ("delay_s", "(c) delay (s per flight)"))
    plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 7.5})
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_W, 2.1))
    fig.subplots_adjust(left=0.045, right=0.995, top=0.89, bottom=0.25, wspace=0.2)
    width, out = 0.8 / len(BY_CLASS), {}
    for ax, (key, title) in zip(axes, metrics, strict=True):
        for i, (run, _, col, mk) in enumerate(BY_CLASS):
            cls = screens.get(runs[run]["env_cfg_hash"], {})
            bal = stats.balanced(runs[run], cls)
            for w in range(4):
                m, lo, hi = mean_ci(_by_class({run: bal}, screens, run, key, w))
                x = w + (i - (len(BY_CLASS) - 1) / 2) * width
                hollow = run == "BASE-avoid"
                ax.plot([x, x], [lo, hi], color=col, lw=0.9)
                ax.plot(x, m, mk, color=col, ms=3.8, mec=col if hollow else "white", mfc="white" if hollow else col, mew=0.7 if hollow else 0.3)
                out[f"{run} {key} W{w}"] = round(m, 2)
        ax.set_title(title, loc="left", pad=3)
        ax.set_xticks(range(4), [f"W{w}" for w in range(4)])
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", lw=0.3, color="0.88")
        ax.set_axisbelow(True)
    handles = [plt.Line2D([], [], color=c, marker=mk, ms=4, lw=0.9, mec=c if r == "BASE-avoid" else "white", mfc="white" if r == "BASE-avoid" else c, mew=0.7 if r == "BASE-avoid" else 0.3, label=lab)
               for r, lab, c, mk in BY_CLASS]  # fmt: skip
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=7.2, bbox_to_anchor=(0.5, 0.0), columnspacing=1.4, handlelength=1.6)
    _save(fig, "route_impact")
    _style()
    return out


def coordination() -> dict:
    """RQ2: exact weather; loss of separation by route-impact class (N = 5) and by traffic. Lines labelled directly, no legend."""
    root = results_root()
    runs = {p.parent.name: json.loads(p.read_text()) for p in root.glob("07*/*/metrics.json")}
    runs["E3-N5-none"] = json.loads((root / "04_baselines" / "BASE-direct" / "metrics.json").read_text())
    screens = stats.classes(root)
    with plt.rc_context({"font.size": 8, "axes.titlesize": 8}):
        fig, (a, b) = plt.subplots(1, 2, figsize=(3.5, 2.2), sharey=True, gridspec_kw={"width_ratios": (1, 1.25)})
        fig.subplots_adjust(left=0.135, right=0.99, top=0.9, bottom=0.2, wspace=0.12)
        dodge, out = 0.07, {}
        for i, (k, _, col, mk) in enumerate(COORD):
            off = (i - 3) * dodge if k in ("direct", "fixed", "joint") else 0.0
            run = f"E3-N5-{k}"
            first = {**runs[run], "scenarios": runs[run]["scenarios"][:512]}
            ms, lo, hi = [], [], []
            for w in range(4):
                m, l, h = mean_ci(_by_class({run: first}, screens, run, "any_los", w))
                ms.append(m * 100), lo.append((m - l) * 100), hi.append((h - m) * 100)
                out[f"{run} W{w}"] = round(m * 100, 1)
            a.errorbar(np.arange(4) + off, ms, yerr=[lo, hi], color=col, marker=mk, ms=4.2, lw=0, elinewidth=1.0, mec="white", mew=0.4, capsize=0)
            pts = []
            for n in (3, 5, 8):
                r = f"E3-N{n}-{k}"
                if r in runs:
                    d = {**runs[r], "scenarios": runs[r]["scenarios"][: 512 if n == 5 else experiments.SMALL]}
                    pts.append((n, mean_ci(stats.per_scenario(d)["any_los"])))
            if pts:
                b.errorbar([n + off * 2.5 for n, _ in pts], [v[0] * 100 for _, v in pts], yerr=[[(v[0] - v[1]) * 100 for _, v in pts], [(v[2] - v[0]) * 100 for _, v in pts]],
                           color=col, marker=mk, ms=4.2, lw=1.0, mec="white", mew=0.4, capsize=0)  # fmt: skip
                out.update({f"E3-N{n}-{k}": round(v[0] * 100, 1) for n, v in pts})
        # direct labels instead of a legend
        b.text(4.55, 50, "fly direct", color=COORD[0][2], ha="right", va="bottom", fontsize=7.5)
        b.text(5.7, 40, "simultaneous", color=COORD[1][2], ha="left", va="top", fontsize=7.5)
        zero = [plt.Line2D([], [], color=c, marker=mk, ms=4.2, lw=0, mec="white", mew=0.4, label=lab) for _, lab, c, mk in COORD[2:]]
        a.legend(handles=zero, loc="lower left", bbox_to_anchor=(0.0, 0.05), frameon=False, fontsize=7, handlelength=1.5, labelspacing=0.25, borderaxespad=0.2)
        a.set_xticks(range(4), [f"W{w}" for w in range(4)])
        a.set_xlabel("route-impact class, $N=5$", labelpad=1.5)
        a.set_ylabel("scenarios with LoS (%)", labelpad=1.5)
        a.set_title("(a) by class", loc="left", pad=3)
        b.set_xticks((3, 5, 8))
        b.set_xlim(2.5, 8.5)
        b.set_xlabel("aircraft $N$", labelpad=1.5)
        b.set_title("(b) by traffic", loc="left", pad=3)
        for ax in (a, b):
            ax.set_ylim(-4, 100)
            ax.grid(axis="y", lw=0.3, color="0.88")
            ax.set_axisbelow(True)
        _save(fig, "coordination")
    return out


# A schematic world for the predictor figure (not to scale): wind to the east; each radar interval moves a cell 4 units.
# Cell A drifts with a noisy own displacement, B grows, C decays, D is born after the latest picture.
_PREV = {"A": (2, 18.5, 4.0), "B": (0, 8, 2.5), "C": (16, 12, 4.6)}
_LATEST = {"A": (6, 20.0, 4.0), "B": (4, 8, 3.3), "C": (20, 12, 3.8)}  # the picture is one interval old
_TRUTH = {"A": (14, 20.0, 4.0), "B": (12, 8, 5.0), "C": (28, 12, 2.0), "D": (33, 24.5, 3.0)}  # 5 min ahead: two intervals after the picture


def _schematic_predictions() -> dict:
    own = {k: np.subtract(_LATEST[k][:2], _PREV[k][:2]) for k in _LATEST}
    mean = np.mean(list(own.values()), 0)
    grow = {k: _LATEST[k][2] - _PREV[k][2] for k in _LATEST}
    pooled = {k: 0.75 * own[k] + 0.25 * mean for k in own}
    rng = np.random.default_rng(3)
    samples = []
    for b in ((24, 3.5, 2.4), (35, 25.5, 2.4), None):  # births are random: a different one, or none, in each future
        cells = [(x + rng.normal(0, 0.5), y + rng.normal(0, 0.5), r * rng.uniform(0.9, 1.1)) for k, (x, y, r) in _TRUTH.items() if k != "D"]
        samples.append(cells + ([b] if b else []))
    return {
        "frozen radar": [_LATEST[k] for k in _LATEST],
        "extrapolation": [(*np.add(_LATEST[k][:2], own[k]), _LATEST[k][2]) for k in _LATEST],
        "tracker": [(*np.add(_LATEST[k][:2], 2 * pooled[k]), _LATEST[k][2] + 2 * grow[k]) for k in _LATEST],
        "forecaster": [(14.2, 20.4, 3.9), (11.8, 8.1, 4.8), (27.7, 12.1, 2.2)],
        "true weather state": samples,
        "exact future": [_TRUTH[k] for k in _TRUTH],
    }, own, pooled


def predictors() -> None:
    """What each weather predictor does, on a schematic world: the truth 5 min ahead (fill) against each prediction (outline)."""
    preds, own, pooled = _schematic_predictions()
    rules = {"frozen radar": "holds the latest\npicture still",
             "extrapolation": "each cell's own last step;\nignores picture age",
             "tracker": "catches up to now;\nshared wind; growth kept",
             "forecaster": "learned: each cell's\nmotion, size, survival",
             "true weather state": "hidden state known now;\nfutures sampled",
             "exact future": "the future that\nwill happen"}  # fmt: skip
    colors = {"frozen radar": "#7f7f7f", "extrapolation": "#17becf", "tracker": PURPLE, "forecaster": BLUE, "true weather state": DARK, "exact future": DARK}
    fig = plt.figure(figsize=(TEXT_W, 1.85))
    gs = GridSpec(1, 6, figure=fig, left=0.005, right=0.995, top=0.8, bottom=0.27, wspace=0.07)
    g = np.linspace(-2, 42, 300)
    X, Y = np.meshgrid(g, np.linspace(0, 30, 205))
    sev = np.zeros_like(X)
    for x, y, r in _TRUTH.values():
        d = np.hypot(X - x, Y - y) / r
        sev = np.maximum(sev, np.where(d <= 1, 0.3 + 0.5 * (1 - d), 0))
    for n, (name, cells) in enumerate(preds.items()):
        ax = fig.add_subplot(gs[0, n])
        ax.imshow(np.ma.masked_less_equal(sev, 0), origin="lower", extent=(-2, 42, 0, 30), cmap=_storm_cmap(), vmin=1e-6, vmax=1, interpolation="bilinear")
        col = colors[name]
        if n < 4:  # the radar pictures a predictor reads
            for pic, ls in ((_PREV, ":"), (_LATEST, "-")):
                for x, y, r in pic.values():
                    ax.add_patch(Circle((x, y), r, fill=False, ec="0.55", lw=0.6, ls=ls))
        if name == "true weather state":
            for k, futs in enumerate(cells):
                for x, y, r in futs:
                    ax.add_patch(Circle((x, y), r, fill=False, ec=col, lw=0.6, ls=("-", "--", ":")[k]))
        else:
            for x, y, r in cells:
                ax.add_patch(Circle((x, y), r, fill=False, ec=col, lw=1.2))
        if name in ("extrapolation", "tracker"):
            step = own if name == "extrapolation" else {k: 2 * v for k, v in pooled.items()}
            for k, d in step.items():
                ax.annotate("", xy=np.add(_LATEST[k][:2], d), xytext=_LATEST[k][:2], arrowprops={"arrowstyle": "-|>", "color": col, "lw": 0.8, "mutation_scale": 6})
        if n == 0:
            ax.annotate("born after\nthe picture", xy=(33, 21.5), xytext=(33, 3.2), fontsize=5.8, ha="center", color="0.25",
                        arrowprops={"arrowstyle": "-", "color": "0.4", "lw": 0.5})  # fmt: skip
            ax.annotate("", xy=(17, 27.5), xytext=(10, 27.5), arrowprops={"arrowstyle": "-|>", "color": "0.4", "lw": 0.6, "mutation_scale": 5})
            ax.text(9.5, 27.5, "wind", fontsize=5.8, ha="right", va="center", color="0.25")
        ax.set(xlim=(-2, 42), ylim=(0, 30), aspect="equal", xticks=[], yticks=[])
        ax.set_title(name, fontsize=7.5, color=col, pad=2)
        ax.text(0.5, -0.06, rules[name], transform=ax.transAxes, ha="center", va="top", fontsize=6.6, linespacing=1.05)
        for sp in ax.spines.values():
            sp.set(visible=True, lw=0.5, color="0.5")
    fig.text(0.335, 0.975, "from the radar pictures", ha="center", va="top", fontsize=7.5)
    fig.text(0.835, 0.975, "references: from the simulator", ha="center", va="top", fontsize=7.5)
    for x0, x1 in ((0.01, 0.66), (0.675, 0.99)):
        fig.add_artist(plt.Line2D([x0, x1], [0.885, 0.885], color="0.5", lw=0.5))
    handles = [plt.Rectangle((0, 0), 1, 1, fc="#d9685a", label="true weather 5 min ahead"), plt.Line2D([], [], color="0.55", lw=0.6, ls=":", label="previous radar picture"),
               plt.Line2D([], [], color="0.55", lw=0.6, label="latest radar picture"), plt.Line2D([], [], color="0.2", lw=1.2, label="prediction 5 min ahead")]  # fmt: skip
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=6.8, bbox_to_anchor=(0.5, -0.01), columnspacing=1.6, handlelength=1.6)
    _save(fig, "predictors")


def levels() -> None:
    """How a weather predictor is scored: (a) forecast overlap; (b, c) agreement of its plan ranking with the exact one."""
    with plt.rc_context({"font.size": 7, "axes.titlesize": 7}):
        fig = plt.figure(figsize=(3.5, 1.45))
        gs = GridSpec(1, 3, figure=fig, left=0.01, right=0.99, top=0.83, bottom=0.04, wspace=0.12, width_ratios=(1, 1.05, 1.1))
        # (a) overlap
        ax = fig.add_subplot(gs[0, 0])
        g = np.linspace(0, 20, 400)
        X, Y = np.meshgrid(g, g)
        true = np.hypot(X - 9, Y - 10) <= 5.5
        pred = np.hypot(X - 12, Y - 10) <= 5
        img = np.zeros((*X.shape, 4))
        img[true] = matplotlib.colors.to_rgba("#e8a198")
        img[true & pred] = matplotlib.colors.to_rgba("#b8483c")
        ax.imshow(img, origin="lower", extent=(0, 20, 0, 20))
        ax.add_patch(Circle((12, 10), 5, fill=False, ec=BLUE, lw=1.1))
        ax.text(10, 1.2, "overlap = $\\frac{\\mathrm{both}}{\\mathrm{either}}$", ha="center", va="bottom", fontsize=7)
        ax.text(3.2, 15.8, "true", color="#b8483c", fontsize=6.5, ha="center")
        ax.text(17.5, 15.8, "predicted", color=BLUE, fontsize=6.5, ha="center")
        ax.set(xlim=(0, 20), ylim=(0, 20), aspect="equal", xticks=[], yticks=[])
        ax.set_title("(a) prediction", loc="left", pad=2)
        # (b) candidate plans
        ax = fig.add_subplot(gs[0, 1])
        ax.plot(20, 15.5, "x", color="0.3", ms=4, mew=0.9)
        ax.text(20, 17, "exit", ha="center", fontsize=6.3, color="0.3")
        ends = {1: (13, 19), 2: (17, 14.6), 3: (18, 7.5), 4: (13, 2)}
        for k, (x, y) in ends.items():
            ax.annotate("", xy=(x, y), xytext=(3, 10), arrowprops={"arrowstyle": "-|>", "color": "0.3", "lw": 0.8, "mutation_scale": 6})
            ax.text(x + 0.6, y, str(k), fontsize=6.5, va="center")
        ax.add_patch(Polygon(np.array([[1.4, 0], [-0.9, 0.8], [-0.4, 0], [-0.9, -0.8]]) * 1.6 + (3, 10), closed=True, fc=BLUE, lw=0))
        ax.set(xlim=(0, 21), ylim=(0, 20), aspect="equal", xticks=[], yticks=[])
        ax.set_title("(b) candidate plans", loc="left", pad=2)
        # (c) ranking agreement
        ax = fig.add_subplot(gs[0, 2])
        predicted, exact = [1, 2, 3, 4], [2, 1, 3, 4]  # best first
        for k in predicted:
            y0, y1 = 4 - predicted.index(k), 4 - exact.index(k)
            swap = k in (1, 2)
            ax.plot([0, 1], [y0, y1], color="#c0392b" if swap else "0.55", lw=1.0 if swap else 0.7)
            ax.text(-0.08, y0, str(k), ha="right", va="center", fontsize=6.5)
            ax.text(1.08, y1, str(k), ha="left", va="center", fontsize=6.5)
        ax.text(0, 4.75, "predicted", ha="center", fontsize=6.3)
        ax.text(1, 4.75, "exact", ha="center", fontsize=6.3)
        ax.text(0.5, -0.35, "1 pair of 6 swapped\n$\\tau = (5-1)/6 = 0.67$", ha="center", va="bottom", fontsize=6.3, linespacing=1.1)
        ax.set(xlim=(-0.45, 1.45), ylim=(-0.5, 5.2), xticks=[], yticks=[])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title("(c) decision", loc="left", pad=2)
        for a in fig.axes[:2]:
            for sp in a.spines.values():
                sp.set(visible=True, lw=0.5, color="0.5")
        _save(fig, "levels")


def _pair_sep(d: dict, row: int) -> np.ndarray:
    """[T, N, N] distance between every pair of aircraft alive at that step (inf otherwise)."""
    p, on = d["p"][row], d["alive"][row]
    sep = np.linalg.norm(p[:, :, None] - p[:, None], axis=-1)
    both = on[:, :, None] & on[:, None]
    n = sep.shape[1]
    return np.where(both & ~np.eye(n, dtype=bool), sep, np.inf)


def _pick_trace(sh: dict, fx: dict) -> tuple[int, int, int, int]:
    """(row, i, j, step of closest approach): a loss of separation under simultaneous replanning, none in a fixed order."""
    best = None
    for row in range(sh["p"].shape[0]):
        s_sh, s_fx = _pair_sep(sh, row), _pair_sep(fx, row)
        if s_sh.min() >= 5 or s_fx.min() < 5:
            continue
        t, i, j = np.unravel_index(np.argmin(s_sh), s_sh.shape)
        # prefer a clean two-aircraft encounter: no third aircraft near either at the closest approach
        third = min(np.delete(s_sh[t, i], [i, j]).min(), np.delete(s_sh[t, j], [i, j]).min())
        key = (third > 15, -s_sh[t, i, j])
        if best is None or key > best[0]:
            best = (key, (row, int(min(i, j)), int(max(i, j)), int(t)))
    assert best is not None, "no row loses separation only under simultaneous replanning"
    return best[1]


def coordination_trace() -> dict:
    """RQ2 mechanism on one replayed scenario: tracks and commanded turns of the pair that loses separation."""
    cache = results_root() / "paper_cache"
    sh, fx = (dict(np.load(cache / f"trace-{m}.npz")) for m in ("shared", "fixed"))
    row, i, j, tc = _pick_trace(sh, fx)
    offs = sh["offsets"]
    t_lo, t_hi = max(tc - 14, 0), tc + 6
    with plt.rc_context({"font.size": 7.5, "axes.titlesize": 7.5}):
        fig = plt.figure(figsize=(3.5, 3.2))
        gs = GridSpec(2, 2, figure=fig, left=0.11, right=0.87, top=0.94, bottom=0.15, wspace=0.12, hspace=0.3, height_ratios=(1.3, 1))
        cols = {i: BLUE, j: ORANGE}
        pts = np.concatenate([d["p"][row, t_lo:t_hi, [i, j]].reshape(-1, 2) for d in (sh, fx)])
        c, half = (pts.min(0) + pts.max(0)) / 2, (pts.max(0) - pts.min(0)).max() / 2 + 4
        for k, (d, title) in enumerate(((sh, "(a) simultaneous"), (fx, "(b) fixed order"))):
            ax = fig.add_subplot(gs[0, k])
            for a in (i, j):
                tr = d["p"][row, t_lo:t_hi, a][d["alive"][row, t_lo:t_hi, a]]
                ax.plot(*tr.T, color=cols[a], lw=1.0)
                ax.plot(*tr[::2].T, "o", color=cols[a], ms=1.8)
                ax.annotate("", xy=tr[-1], xytext=tr[-2], arrowprops={"arrowstyle": "-|>", "color": cols[a], "lw": 1.0, "mutation_scale": 7})
            sep = _pair_sep(d, row)[:, i, j]
            t = int(np.argmin(sep[t_lo:t_hi])) + t_lo
            for a in (i, j):
                ax.add_patch(Circle(d["p"][row, t, a], 2.5, fill=False, ec="#c0392b" if sep[t] < 5 else "0.5", lw=0.7, ls="-" if sep[t] < 5 else ":"))
            ax.set(xlim=(c[0] - half, c[0] + half), ylim=(c[1] - half, c[1] + half), aspect="equal", xticks=[], yticks=[])
            ax.set_title(f"{title}: {sep[t]:.1f} NM", loc="left", pad=2, fontsize=7)
            for sp in ax.spines.values():
                sp.set(visible=True, lw=0.5, color="0.5")
        ax.plot([c[0] + half - 11, c[0] + half - 1], [c[1] - half + 1.5] * 2, color="0.2", lw=1.0)
        ax.text(c[0] + half - 6, c[1] - half + 2.3, "10 NM", ha="center", va="bottom", fontsize=6.5)
        mins = lambda t: (np.asarray(t) - tc) * 0.5
        ax = fig.add_subplot(gs[1, 0])
        for d, ls, lab in ((sh, "-", "simultaneous"), (fx, "--", "fixed order")):
            sep = _pair_sep(d, row)[t_lo:t_hi, i, j]
            ax.plot(mins(np.arange(t_lo, t_hi)), np.where(np.isfinite(sep), sep, np.nan), color="0.2", ls=ls, lw=1.0, label=lab)
        ax.axhline(5, color="#c0392b", lw=0.7)
        ax.text(mins(t_lo), 5.4, "5 NM", color="#c0392b", fontsize=6.5, va="bottom")
        ax.set_ylim(bottom=0)
        ax.set_ylabel("separation (NM)", labelpad=1)
        ax.set_title("(c) separation of the pair", loc="left", pad=2, fontsize=7)
        ax.legend(frameon=False, fontsize=6.3, loc="upper right", handlelength=1.6, borderaxespad=0.1)
        ax = fig.add_subplot(gs[1, 1])
        ts = mins(np.arange(t_lo, t_hi))
        for a in (i, j):
            ax.step(ts, offs[sh["round1"][row, t_lo:t_hi, 0, a]], where="post", color=cols[a], lw=0.8, ls=":")
            ax.step(ts, offs[sh["action"][row, t_lo:t_hi, a]], where="post", color=cols[a], lw=1.1)
        ax.axhline(0, color="0.6", lw=0.5)
        ax.set_yticks((-45, 0, 45))
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.set_ylabel("turn (deg)", labelpad=1)
        fig.text(0.49, 0.005, "minutes from the closest approach in (a)", ha="center", va="bottom")
        ax.set_title("(d) turn in (a): planned, flown", loc="left", pad=2, fontsize=7)
        ax.legend(handles=[plt.Line2D([], [], color="0.3", lw=0.8, ls=":", label="planned"), plt.Line2D([], [], color="0.3", lw=1.1, label="flown")],
                  frameon=False, fontsize=6.3, loc="lower left", handlelength=1.2, borderaxespad=0.1, labelspacing=0.2)  # fmt: skip
        _save(fig, "coord_trace")
    return {"row": row, "pair": (i, j), "t": tc, "sep_shared": float(_pair_sep(sh, row)[:, i, j].min()), "sep_fixed": float(_pair_sep(fx, row)[:, i, j].min())}


def main() -> None:
    _style()
    OUT.mkdir(parents=True, exist_ok=True)
    print(sim_overview(), flush=True)
    predictors()
    levels()
    print(route_impact(), flush=True)
    print(coordination(), flush=True)
    print(coordination_trace(), flush=True)


if __name__ == "__main__":
    main()
