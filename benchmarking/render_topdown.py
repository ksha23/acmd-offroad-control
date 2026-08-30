#!/usr/bin/env python3
"""Top-down animation and closest-approach still for one simulated run.

Renders the qualitative views that accompany the teleoperation failure-mode
battery, which invokes this module under ``--render``. They are inspection
artifacts: the manuscript's quantitative claims come from the recorded
diagnostics, and these renders make an outcome legible by showing how it arose.

The module reads a run's ``sim_diag.csv``, which carries the vehicle trajectory
alongside both the operator's command and the command actually applied, and the
sibling ``obstacles.json`` written by the simulator whenever ``--sim-diag-csv``
is set. It animates a bird's-eye view: the vehicle as an oriented body,
obstacles as circles, the driven trajectory, and a highlight over any interval
in which the applied command departs from the operator's, which is where the
safety filter is intervening.

Animation output uses matplotlib's Pillow writer and requires no external
encoder. A still is also written at the instant of closest approach, the moment
at which the clearance reported for the run is attained.

Usage:
    render_topdown.py <run_dir_or_sim_diag.csv> [--out out.gif] [--fps 12]
                      [--title "..."] [--stride 2]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

# HMMWV-ish footprint (m) for the top-down body.
VEH_L, VEH_W = 4.6, 2.2
INTERVENE_EPS = 1e-2


def _load(path: Path):
    sim_diag = path / "sim_diag.csv" if path.is_dir() else path
    d = pd.read_csv(sim_diag)
    obs_json = sim_diag.with_name("obstacles.json")
    obstacles = []
    if obs_json.exists():
        obstacles = json.loads(obs_json.read_text()).get("obstacles", [])
    return d, obstacles, sim_diag


def _num(d, c, default=0.0):
    return pd.to_numeric(d[c], errors="coerce").fillna(default).to_numpy() if c in d.columns \
        else np.full(len(d), default)


def _heading(d, x, y):
    """True heading if logged; else reconstruct from the direction of travel
    (a non-slipping car points where it moves). Hold the last valid heading when
    nearly stationary so the body doesn't spin at a stop."""
    for col in ("heading", "psi", "yaw"):
        if col in d.columns:
            h = pd.to_numeric(d[col], errors="coerce").to_numpy()
            if np.isfinite(h).any() and np.nanstd(h) > 1e-3:
                return h
    dx = np.gradient(x); dy = np.gradient(y)
    spd = np.hypot(dx, dy)
    psi = np.arctan2(dy, dx)
    last = 0.0
    for i in range(len(psi)):
        if spd[i] < 1e-3:
            psi[i] = last
        else:
            last = psi[i]
    # light smoothing to suppress per-sample jitter
    k = 5
    if len(psi) > k:
        c = np.convolve(np.cos(psi), np.ones(k) / k, mode="same")
        s = np.convolve(np.sin(psi), np.ones(k) / k, mode="same")
        psi = np.arctan2(s, c)
    return psi


def _add_vehicle(ax, x, y, psi, color):
    """Draw the body centered at (x,y) rotated by psi (about its centre), with a
    heading arrow at the front. Returns the artists so callers can remove them."""
    import matplotlib.transforms as mt
    r = Rectangle((x - VEH_L / 2, y - VEH_W / 2), VEH_L, VEH_W,
                  facecolor=color, edgecolor="black", lw=1.2, alpha=0.9, zorder=5)
    r.set_transform(mt.Affine2D().rotate_around(x, y, psi) + ax.transData)
    ax.add_patch(r)
    hx, hy = x + (VEH_L / 2 + 1.0) * math.cos(psi), y + (VEH_L / 2 + 1.0) * math.sin(psi)
    arr, = ax.plot([x, hx], [y, hy], color="black", lw=2.0, zorder=6)
    return [r, arr]


def render(run, out=None, fps=12, stride=2, title=None):
    run = Path(run)
    d, obstacles, sim_diag = _load(run)
    if out is None:
        out = sim_diag.with_name("topdown.gif")
    out = Path(out)

    t = _num(d, "time")
    x, y = _num(d, "x"), _num(d, "y")
    psi = _heading(d, x, y)
    v = _num(d, "speed")
    coll = _num(d, "collisions")
    clr = _num(d, "nearest_clearance_m", np.nan)
    ds = np.abs(_num(d, "steering") - _num(d, "steering_op"))
    dth = np.abs(_num(d, "throttle") - _num(d, "throttle_op"))
    dbr = np.abs(_num(d, "braking") - _num(d, "braking_op"))
    raw_interv = (ds > INTERVENE_EPS) | (dth > INTERVENE_EPS) | (dbr > INTERVENE_EPS)
    # Hold the "filter active" indicator for a few samples after each detection,
    # so that it does not flicker as the correction crosses the deadband and an
    # intervention remains visible at animation frame rates.
    hold = 4
    intervening = np.zeros(len(raw_interv), dtype=bool)
    cd = 0
    for i in range(len(raw_interv)):
        if raw_interv[i]:
            cd = hold
        intervening[i] = cd > 0
        cd = max(0, cd - 1)

    idx = np.arange(0, len(d), max(1, stride))

    # view bounds spanning trajectory + obstacles, with margin
    xs = list(x) + [o["x"] for o in obstacles]
    ys = list(y) + [o["y"] for o in obstacles]
    xmin, xmax = min(xs) - 4, max(xs) + 6
    # clamp the lateral view so a late trajectory drift cannot blow up the frame
    ymin, ymax = max(min(ys) - 4, -11), min(max(ys) + 4, 11)
    ymin, ymax = min(ymin, -6), max(ymax, 6)

    fig, ax = plt.subplots(figsize=(11, max(3.0, (ymax - ymin) / (xmax - xmin) * 11)))
    ax.set_aspect("equal"); ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.25)
    ax.axhline(0.0, color="#888", lw=1.0, ls="--", alpha=0.6, zorder=1)  # nominal path
    for o in obstacles:
        ax.add_patch(Circle((o["x"], o["y"]), o["r"], facecolor="#7a5230",
                            edgecolor="#3d2810", lw=1.2, alpha=0.85, zorder=3))
    ax.plot(x, y, color="#4477cc", lw=1.0, alpha=0.35, zorder=2)  # full path (ghost)
    if title:
        ax.set_title(title, fontsize=12)

    trail, = ax.plot([], [], color="#22aa44", lw=2.2, zorder=4)
    veh_holder = {"artists": []}
    hud = ax.text(0.015, 0.97, "", transform=ax.transAxes, va="top", ha="left",
                  fontsize=10, family="monospace",
                  bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.85), zorder=10)

    def frame(k):
        i = idx[k]
        trail.set_data(x[:i + 1], y[:i + 1])
        for art in veh_holder["artists"]:
            art.remove()
        hit = coll[:i + 1].max() > 0
        color = "#cc3333" if hit else ("#ee9922" if intervening[i] else "#3366cc")
        veh_holder["artists"] = _add_vehicle(ax, x[i], y[i], psi[i], color)
        cl = clr[i]
        hud.set_text(f"t={t[i]:5.1f}s  v={v[i]:4.1f} m/s\n"
                     f"clear={cl:5.2f} m\n"
                     f"filter={'ACTIVE' if intervening[i] else 'idle  '}"
                     f"{'  CONTACT' if hit else ''}")
        return trail, hud

    anim = FuncAnimation(fig, frame, frames=len(idx), blit=False, interval=1000 / fps)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out), writer=PillowWriter(fps=fps))
    plt.close(fig)

    # Still frame at the instant of minimum clearance: the moment that
    # determines the clearance value reported for the run.
    hero = out.with_suffix(".png")
    finite = np.where(np.isfinite(clr))[0]
    hk = int(finite[np.argmin(clr[finite])]) if len(finite) else int(len(d) // 2)
    fig2, ax2 = plt.subplots(figsize=(11, max(3.0, (ymax - ymin) / (xmax - xmin) * 11)))
    ax2.set_aspect("equal"); ax2.set_xlim(xmin, xmax); ax2.set_ylim(ymin, ymax)
    ax2.set_xlabel("x (m)"); ax2.set_ylabel("y (m)"); ax2.grid(True, alpha=0.25)
    ax2.axhline(0.0, color="#888", lw=1.0, ls="--", alpha=0.6)
    for o in obstacles:
        ax2.add_patch(Circle((o["x"], o["y"]), o["r"], facecolor="#7a5230",
                             edgecolor="#3d2810", lw=1.2, alpha=0.85))
    ax2.plot(x[:hk + 1], y[:hk + 1], color="#22aa44", lw=2.2)
    _add_vehicle(ax2, x[hk], y[hk], psi[hk],
                 "#cc3333" if coll[:hk + 1].max() > 0 else "#3366cc")
    ax2.set_title((title + "  " if title else "") +
                  f"closest approach t={t[hk]:.1f}s, clearance {clr[hk]:.2f} m")
    fig2.tight_layout(); fig2.savefig(hero, dpi=130); plt.close(fig2)
    return out, hero


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="run dir containing sim_diag.csv (or the csv path)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--title", default=None)
    a = ap.parse_args()
    gif, png = render(a.run, a.out, a.fps, a.stride, a.title)
    print(f"wrote {gif}\nwrote {png}")


if __name__ == "__main__":
    raise SystemExit(main())
