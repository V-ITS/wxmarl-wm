"""Offline matplotlib animation of a saved replay. North-up: +x east, +y north."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Polygon, Rectangle

# EPyMARL sets the root logger to DEBUG, which makes matplotlib (and ffmpeg) very chatty.
logging.getLogger("matplotlib").setLevel(logging.WARNING)

AIRCRAFT, LOS, CELL = "#1f4e79", "#d62728", "#e4572e"
TRAIL_STEPS = 20  # decision steps of trail kept behind each aircraft
MARKER_NM = 3.0  # aircraft marker length
# Nose-right triangle, rotated by the heading.
_TRIANGLE = np.array([[1.0, 0.0], [-0.6, 0.55], [-0.3, 0.0], [-0.6, -0.55]]) * MARKER_NM / 1.6


def _ffmpeg() -> str | None:
    """Path to an ffmpeg binary: system one, else the one bundled with imageio-ffmpeg."""
    path = shutil.which("ffmpeg")
    if path is None:
        try:
            import imageio_ffmpeg

            path = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            pass
    return path


def _scene(d, sep_circles: bool, title: str):
    """Build the figure for replay arrays ``d``; returns ``(fig, draw)`` with ``draw(frame)``."""
    L, dt, d_sep = float(d["L"]), float(d["dt"]), float(d["d_sep"])
    p, psi, alive, e, p0 = d["p"], d["psi"], d["alive"], d["e"], d["p0"]
    N = alive.shape[1]
    K = d["cell_alive"].shape[1]

    fig, ax = plt.subplots(figsize=(7, 7.5), dpi=100)
    pad = 32.0
    ax.set(xlim=(-pad, L + pad), ylim=(-pad, L + pad), aspect="equal", xlabel="East (NM)", ylabel="North (NM)")
    ax.add_patch(Rectangle((0, 0), L, L, fill=False, lw=1.5, ec="black", zorder=1))
    cells = [ax.add_patch(Circle((0, 0), 1, fc=CELL, ec=CELL, lw=1, visible=False, zorder=2)) for _ in range(K)]
    exit_lines = ax.add_collection(LineCollection([], colors="0.6", lw=0.6, alpha=0.5, zorder=3))
    exit_marks = ax.scatter([], [], marker="x", s=18, c="0.4", lw=0.8, zorder=3)
    trails = ax.add_collection(LineCollection([], lw=1.4, zorder=4))
    los_lines = ax.add_collection(LineCollection([], colors=LOS, lw=2.0, zorder=5))
    rings = [ax.add_patch(Circle((0, 0), d_sep / 2, fill=False, ec="0.3", lw=0.6, visible=False, zorder=5))
             for _ in range(N if sep_circles else 0)]  # fmt: skip
    planes = [ax.add_patch(Polygon(_TRIANGLE, closed=True, visible=False, zorder=6)) for _ in range(N)]
    header = ax.set_title("", loc="left", fontsize=10, family="monospace")
    rgb = np.array(matplotlib.colors.to_rgb(AIRCRAFT))

    def draw(f: int):
        for k, patch in enumerate(cells):
            on = bool(d["cell_alive"][f, k])
            patch.set_visible(on)
            if on:
                patch.center = tuple(d["cell_q"][f, k])
                patch.set_radius(float(d["cell_rho"][f, k]))
                patch.set_facecolor((*matplotlib.colors.to_rgb(CELL), 0.1 + 0.6 * float(d["cell_I"][f, k])))

        on = alive[f]
        in_los = d["los"][f].any(-1) & on
        for i, patch in enumerate(planes):
            patch.set_visible(bool(on[i]))
            if on[i]:
                c, s = np.cos(psi[f, i]), np.sin(psi[f, i])
                patch.set_xy(_TRIANGLE @ np.array([[c, s], [-s, c]]) + p[f, i])
                patch.set_color(LOS if in_los[i] else AIRCRAFT)
        for i, ring in enumerate(rings):
            ring.set_visible(bool(on[i]))
            ring.center = tuple(p[f, i])

        exit_lines.set_segments([np.stack((p[f, i], e[f, i])) for i in np.flatnonzero(on)])
        exit_marks.set_offsets(e[f, on].reshape(-1, 2))
        pairs = np.argwhere(np.triu(d["los"][f], 1) & on[:, None] & on[None, :])
        los_lines.set_segments([np.stack((p[f, i], p[f, j])) for i, j in pairs])

        # Fading trails; a trail is cut where the slot was empty or handed to a new flight.
        segs, cols = [], []
        for g in range(max(1, f - TRAIL_STEPS + 1), f + 1):
            same = on & alive[g] & alive[g - 1] & (p0[g] == p0[g - 1]).all(-1) & (p0[g] == p0[f]).all(-1)
            for i in np.flatnonzero(same):
                segs.append(np.stack((p[g - 1, i], p[g, i])))
                cols.append((*rgb, 0.8 * (1 - (f - g) / TRAIL_STEPS)))
        trails.set_segments(segs)
        trails.set_color(cols)

        t = int(f * dt)
        header.set_text(
            f"{title + '  ' if title else ''}t={t // 60:02d}:{t % 60:02d}  active={int(on.sum()):2d}  "
            f"LoS events={int(d['los_events'][f])}  wx incursion={int(d['wx_incursion_time_s'][f])} s"
        )
        return []

    return fig, draw


def animate(replay: str | Path, out: str | Path | None = None, fps: int = 10, sep_circles: bool = False,
            title: str = "") -> Path:  # fmt: skip
    """Render ``replay`` (.npz) to MP4 (ffmpeg) or GIF (pillow). Returns the written path."""
    replay = Path(replay)
    d = np.load(replay)
    fig, draw = _scene(d, sep_circles, title)
    anim = FuncAnimation(fig, draw, frames=d["alive"].shape[0], blit=False)
    ffmpeg = _ffmpeg()
    if ffmpeg:
        plt.rcParams["animation.ffmpeg_path"] = ffmpeg
        out = Path(out or replay.with_suffix(".mp4")).with_suffix(".mp4")
        anim.save(out, writer=FFMpegWriter(fps=fps, codec="h264", extra_args=["-pix_fmt", "yuv420p"]))
    else:
        out = Path(out or replay.with_suffix(".gif")).with_suffix(".gif")
        anim.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out


def save_frame(replay: str | Path, frame: int, out: str | Path, sep_circles: bool = False, title: str = "") -> Path:
    """Write a single frame of ``replay`` as a PNG (for reports)."""
    fig, draw = _scene(np.load(replay), sep_circles, title)
    draw(frame)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return Path(out)
