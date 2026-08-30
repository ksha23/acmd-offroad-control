#!/usr/bin/env python3
"""Generate the manuscript's system and data-flow architecture figure.

The figure is ``sys_arch.png``, labelled ``fig:arch``. Its wide-short two-row
layout is sized for full text width: the six swappable roles read left to
right along the command path (operator -> 5G latency environment -> safety
filter -> Chrono plant), with the autonomous controller and the terrain
estimator beneath the filter and plant they feed, so the diagram's geometry
matches the direction data actually flows. Fonts are set for the roughly 0.46
print scale of a 13-inch canvas placed at 0.95 text width.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import paper_style
paper_style.apply()


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "my_paper" / "paper_figures" / "sys_arch.png"

FONT_KW = dict(family="serif", fontsize=13.4)
LABEL_FS = 13.2

CTRL_EDGE = "#1f3b63"; CTRL_FILL = "#e6f0fb"
SWAP_EDGE = "#9c6b3f"; SWAP_FILL = "#fff6dc"
HUMAN_EDGE = "#4a7a5c"; HUMAN_FILL = "#eaf5ea"
LAT_EDGE = "#9c5a4a"; LAT_FILL = "#f2ebe9"
DATA_C = "#444444"
LAT_C = "#9c5a4a"
FB_C = "#1f3b63"

Y_TOP, Y_BOT = 3.05, 1.10
BOXES = {
    # name: (centre x, centre y, width, height, label, edge, fill)
    # Widths carry the longest line of each label with visible margin, and the
    # columns are spaced so every connector has room for its own annotation.
    "operator": (1.75, Y_TOP, 3.05, 1.22,
                 "Human operator\nphysical wheel + HUD,\ndelayed onboard view",
                 HUMAN_EDGE, HUMAN_FILL),
    "lat": (7.00, Y_TOP, 3.20, 1.22,
            "5G latency environment\nreplayed N-HiTS traffic,\nuplink + downlink",
            LAT_EDGE, LAT_FILL),
    "filter": (11.17, Y_TOP, 2.70, 1.22,
               "Safety filter\nDOB-CBF / MPSF,\ndelay-aware",
               SWAP_EDGE, SWAP_FILL),
    "nmpc": (7.00, Y_BOT, 3.25, 1.22,
             "acados NMPC\nrig surrogate internal\nmodel + $g$\u2013$g$ cap",
             CTRL_EDGE, CTRL_FILL),
    "est": (11.17, Y_BOT, 2.75, 1.22,
            "GRIT $(\\hat n,\\hat\\phi)$\ngated snapshots,\nlow-grip fallback",
            SWAP_EDGE, SWAP_FILL),
    "sim": (15.72, (Y_TOP + Y_BOT) / 2, 2.95, 2.95,
            "Chrono plant\nHMMWV + SCM\nChrono::Sensor\nIMU + camera\nChrono::ROS",
            CTRL_EDGE, CTRL_FILL),
}


def _draw_box(ax, key):
    cx, cy, w, h, label, edge, fill = BOXES[key]
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.16,rounding_size=0.16",
        linewidth=1.8, edgecolor=edge, facecolor=fill, zorder=2))
    ax.text(cx, cy, label, ha="center", va="center", zorder=3,
            color="black", **FONT_KW)


_BOX_PAD = 0.17


def _port(key, side, frac=0.0):
    cx, cy, w, h, *_ = BOXES[key]
    if side in ("L", "R"):
        x = cx + (w / 2 + _BOX_PAD if side == "R" else -(w / 2 + _BOX_PAD))
        return (x, cy + frac * h)
    x = cx + frac * w
    y = cy + (h / 2 + _BOX_PAD if side == "T" else -(h / 2 + _BOX_PAD))
    return (x, y)


def _arrow(ax, p_from, p_to, *, label=None, label_xy=None, color=DATA_C,
           linestyle="-", lw=1.8, label_fs=LABEL_FS):
    ax.add_patch(FancyArrowPatch(
        p_from, p_to, arrowstyle="-|>", mutation_scale=16,
        linewidth=lw, color=color, linestyle=linestyle,
        connectionstyle="arc3,rad=0", zorder=4))
    if label:
        lx, ly = label_xy if label_xy is not None else (
            (p_from[0] + p_to[0]) / 2, (p_from[1] + p_to[1]) / 2)
        ax.text(lx, ly, label, ha="center", va="center", zorder=5,
                color=color, fontsize=label_fs, family="DejaVu Sans",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.88,
                          pad=1.4))


def _poly_arrow(ax, points, *, label=None, label_xy=None, color=DATA_C,
                linestyle="-", lw=1.8, label_fs=LABEL_FS):
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot([start[0], end[0]], [start[1], end[1]], color=color,
                linestyle=linestyle, linewidth=lw, zorder=4)
    _arrow(ax, points[-2], points[-1], color=color, linestyle=linestyle,
           lw=lw)
    if label and label_xy is not None:
        ax.text(*label_xy, label, ha="center", va="center", zorder=5,
                color=color, fontsize=label_fs, family="DejaVu Sans",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.9,
                          pad=1.4))


def _mid(p_from, p_to, dx=0.0, dy=0.0):
    """Label anchor at a connector's midpoint, nudged clear of the line."""
    return ((p_from[0] + p_to[0]) / 2 + dx, (p_from[1] + p_to[1]) / 2 + dy)


CAMERA_RAIL_Y = 4.26
FEEDBACK_RAIL_Y = -0.02


def main():
    fig, ax = plt.subplots(figsize=(16.6, 4.45))
    ax.set_xlim(0.00, 17.45)
    ax.set_ylim(-0.26, 4.42)
    ax.set_aspect("equal")
    ax.axis("off")

    for key in BOXES:
        _draw_box(ax, key)

    # command path, left to right along the top row
    a, b = _port("operator", "R", 0.18), _port("lat", "L", 0.18)
    _arrow(ax, a, b, label="command", color=HUMAN_EDGE, label_fs=12.4,
           label_xy=_mid(a, b, dy=0.24))
    a, b = _port("lat", "L", -0.18), _port("operator", "R", -0.18)
    _arrow(ax, a, b, label="delayed video", color=LAT_C, linestyle=":",
           label_fs=12.4, label_xy=_mid(a, b, dy=-0.26))
    a, b = _port("lat", "R", 0.0), _port("filter", "L", 0.0)
    _arrow(ax, a, b, label="$u_{op}^{\\rm del}$", color=LAT_C, linestyle=":",
           label_fs=14.0, label_xy=_mid(a, b, dy=0.28))
    a, b = _port("filter", "R", 0.0), _port("sim", "L", 0.40)
    _arrow(ax, a, b, label="$u_{safe}$", color=DATA_C, label_fs=14.0,
           label_xy=_mid(a, b, dy=0.28))

    # autonomous command up into the filter
    a, b = _port("nmpc", "T", 0.30), _port("filter", "B", -0.30)
    _arrow(ax, a, b, label="$u_{des}$", color=CTRL_EDGE, label_fs=14.0,
           label_xy=_mid(a, b, dx=-0.40))

    # estimates condition the controller and the filter's authority
    a, b = _port("est", "L", 0.0), _port("nmpc", "R", 0.0)
    _arrow(ax, a, b, label="$(\\hat n,\\hat\\phi)$", color=SWAP_EDGE,
           linestyle="--", label_fs=12.6, label_xy=_mid(a, b, dy=-0.30))
    a, b = _port("est", "T", 0.30), _port("filter", "B", 0.30)
    _arrow(ax, a, b, label="authority", color=SWAP_EDGE, linestyle="--",
           label_fs=12.4, label_xy=_mid(a, b, dx=0.78))

    # plant feedback: signals to the estimator, state to the controller
    a, b = _port("sim", "L", -0.40), _port("est", "R", 0.0)
    _arrow(ax, a, b, label="IMU + state", color=FB_C, label_fs=12.4,
           label_xy=_mid(a, b, dy=-0.30))
    _poly_arrow(
        ax,
        [_port("sim", "B", -0.25), (14.98, FEEDBACK_RAIL_Y),
         (7.00, FEEDBACK_RAIL_Y), _port("nmpc", "B", 0.0)],
        label="state + obstacles*", color=FB_C, label_fs=12.4,
        label_xy=(9.21, FEEDBACK_RAIL_Y))
    # camera downlink back to the latency environment along the top rail
    _poly_arrow(
        ax,
        [_port("sim", "T", 0.0), (15.72, CAMERA_RAIL_Y),
         (7.00, CAMERA_RAIL_Y), _port("lat", "T", 0.0)],
        label="camera", color=FB_C, linestyle=":", label_fs=12.4,
        label_xy=(13.38, CAMERA_RAIL_Y))

    ax.text(0.30, -0.22, "* exact simulator coordinates; no perception model",
            ha="left", va="bottom", fontsize=11.6, color=LAT_C)

    fig.savefig(OUT, dpi=600, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
