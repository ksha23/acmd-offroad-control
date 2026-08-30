#!/usr/bin/env python3
"""Generate the manuscript's teleoperation failure-mode figure.

The figure is ``teleop_scenarios.png``, labelled ``fig:teleopscen``: a plan
view of three failure modes and the evasions they draw.

Each panel is one scenario replayed with identical operator intent under no
filter, DOB-CBF and MPSF, seen from above with the x axis shifted so the hazard
sits at the origin. Vehicles enter from the left. A cross marks a native Chrono
body contact and a dot marks where a run ends still short of the hazard. Signed
clearance is a separate geometric diagnostic and is not used to place contacts.

All three hazards are stationary, which is what makes a plan view the right
picture: the evasion is a spatial quantity and the panel shows it directly. The
moving-traffic modes are longitudinal encounters along a single line, where a
plan view degenerates and the tabulated clearances carry the result instead. The
three span the ways a filter can succeed: steer around or stop short of a rock
in the path, steer around one offset to the side, and fail-stop short of a
stalled vehicle after connection loss.

Drawn from the committed ``teleop_scenario_traces.csv``; see
``publish_teleop_scenario_traces.py`` for how that file is extracted.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "my_paper" / "paper_figures"

PANELS = [
    ("3_missed_obstacle",     "Missed obstacle"),
    ("4_peripheral_hazard",   "Late-revealed peripheral hazard"),
    ("9_freeze_into_stalled", "Connection loss, stalled vehicle"),
]
ARMS = [("none", "no filter", "#c0392b", "-"),
        ("dob_cbf", "DOB-CBF", "#1f77b4", "-"),
        ("mpsf", "MPSF", "#2ca02c", "--")]
X_WINDOW = (-18.0, 8.0)     # along-track offset from the hazard, metres
# simulation/shared/collision_detector.py: the ego radius signed clearance uses.
# Traces are vehicle centres, so the centre stands off by this much at touch.
EGO_RADIUS = 1.5


def relative(d: pd.DataFrame) -> pd.DataFrame | None:
    """Ego position in the hazard frame, with the hazard radius."""
    cols = {c: pd.to_numeric(d[c], errors="coerce")
            for c in ("x", "y", "hazard_x", "hazard_y", "hazard_r")}
    ok = ~(cols["hazard_x"].isna() | cols["hazard_y"].isna())
    if not ok.any():
        return None
    out = pd.DataFrame({
        "dx": cols["x"] - cols["hazard_x"],
        "dy": cols["y"] - cols["hazard_y"],
        "r": cols["hazard_r"],
        "collisions": pd.to_numeric(d.collisions, errors="coerce").fillna(0),
    })[ok.to_numpy()]
    return out if len(out) > 10 else None


def main() -> int:
    src = FIGURES / "teleop_scenario_traces.csv"
    if not src.is_file():
        raise SystemExit(f"missing {src}; run publish_teleop_scenario_traces.py")
    data = pd.read_csv(src)

    fig, axes = plt.subplots(1, len(PANELS), figsize=(6.0, 1.30))
    y_span: list[tuple[float, float]] = []
    for ax, (scen, title) in zip(axes, PANELS):
        radii, ys = [], []
        for arm, label, colour, style in ARMS:
            d = data[(data.scenario == scen) & (data.arm == arm)]
            rel = relative(d) if len(d) else None
            if rel is None:
                continue
            radii.append(float(np.nanmedian(rel.r)))
            w = rel[(rel.dx >= X_WINDOW[0]) & (rel.dx <= X_WINDOW[1])]
            ax.plot(w.dx, w.dy, style, color=colour, lw=1.2, label=label, zorder=3)
            ys += [w.dy.min(), w.dy.max()] if len(w) else []
            hit = rel[rel.collisions > 0]
            if len(hit):
                ax.plot(hit.dx.iloc[0], hit.dy.iloc[0], "x", ms=5.5, mew=1.5,
                        color=colour, zorder=5)
            elif len(w) and float(rel.dx.iloc[-1]) < 0:   # never reached it
                ax.plot(rel.dx.iloc[-1], rel.dy.iloc[-1], "o", ms=3.0,
                        color=colour, zorder=4)
        if not radii:
            continue
        r = float(np.nanmedian(radii))
        ax.add_patch(plt.Circle((0.0, 0.0), r, facecolor="0.55",
                                edgecolor="0.25", lw=0.7, zorder=2))
        # zero of the signed-clearance proxy, in the same units as the traces
        ax.add_patch(plt.Circle((0.0, 0.0), r + EGO_RADIUS, facecolor="none",
                                edgecolor="0.45", lw=0.6, ls=(0, (2.5, 2.0)),
                                zorder=2))
        ys += [-(r + EGO_RADIUS), r + EGO_RADIUS]
        y_lo, y_hi = min(ys) - 1.2, max(ys) + 1.2
        # equal aspect, so keep the box from degenerating into a sliver
        span = (X_WINDOW[1] - X_WINDOW[0]) / 3.4
        if y_hi - y_lo < span:
            mid = 0.5 * (y_lo + y_hi)
            y_lo, y_hi = mid - span / 2, mid + span / 2
        ax.set_xlim(*X_WINDOW)
        y_span.append((y_lo, y_hi))
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title, fontsize=8.2, pad=2.0)
        ax.set_xlabel("along-track offset from hazard (m)", fontsize=7.6, labelpad=1.0)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7.2, pad=1.5)
    # one lateral scale across panels, so an evasion in one is directly
    # comparable with an evasion in another
    lo, hi = min(s[0] for s in y_span), max(s[1] for s in y_span)
    for ax in axes:
        ax.set_ylim(lo, hi)
    axes[0].set_ylabel("lateral (m)", fontsize=7.6, labelpad=1.0)
    axes[0].legend(fontsize=6.2, loc="upper left", framealpha=0.9,
               borderpad=0.25, labelspacing=0.25, handlelength=1.6)
    fig.tight_layout(pad=0.15, w_pad=0.6)
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "teleop_scenarios.png"
    fig.savefig(out, dpi=400, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
