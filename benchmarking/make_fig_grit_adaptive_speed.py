#!/usr/bin/env python3
"""Generate the manuscript's terrain-adaptive speed figure.

The figure is ``grit_adaptive_speed.png``, labelled ``fig:gritspeed``. It
shows that online terrain estimation converts an estimated soil into an
appropriate speed: fast where the ground is firm, slow where it is not.

(a) Illustrative clay->sand traverse under the live g--g cap: GRIT's estimated
    friction angle and achieved speed climb as it enters firmer soil, while the
    fixed priors hold a constant (too-slow or too-fast) speed.
(b) Full per-terrain matrix (540 cells): mean forward speed per arm per soil.
    GRIT matches the matched-terrain oracle on every soil; the conservative prior
    gives up speed on firm ground.
(c) Worst-case peak crosstrack error per arm per soil: the aggressive prior leaves
    the path on low-grip clay; GRIT and the oracle stay safe everywhere.

Inputs (my_paper/paper_figures/): grit_adaptive_speed_matrix_summary.csv and
grit_adaptive_speed_trace_{grit,conservative,aggressive}.csv.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG = Path(__file__).resolve().parent.parent / "my_paper" / "paper_figures"
TRANSITION_X = 45.0
FALLBACK_PHI_DEG = 13.0  # labelled fixed low-grip fallback (n=0.50, phi=13 deg)
CLAYC, SANDC = "#b8860b", "#1f6f3f"
ARMC = {"oracle": "#555555", "grit": "#1f4e79",
        "conservative": "#b8860b", "aggressive": "#c0392b"}
ARMS = ["oracle", "grit", "conservative", "aggressive"]
TERR = ["clay", "dirt", "sand"]            # soft -> firm


def _trace(name):
    t = pd.read_csv(FIG / f"grit_adaptive_speed_trace_{name}.csv")
    return t[(t.x_fa_meas >= 8) & (t.x_fa_meas <= 135)]


def main() -> int:
    m = pd.read_csv(FIG / "grit_adaptive_speed_matrix_summary.csv")
    m = m.set_index(["variant", "terrain"])
    plt.rcParams.update({"font.size": 14, "axes.labelsize": 14, "axes.titlesize": 14.5,
                     "xtick.labelsize": 12.5, "ytick.labelsize": 12.5})
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(12.0, 3.5))

    # --- (a) adaptation trace ---
    axA.axvspan(0, TRANSITION_X, color=CLAYC, alpha=0.06)
    axA.axvspan(TRANSITION_X, 140, color=SANDC, alpha=0.06)
    axA.axvline(TRANSITION_X, ls="--", color="0.4", lw=1)
    for arm in ("aggressive", "conservative", "grit"):
        t = _trace(arm)
        axA.plot(t.x_fa_meas, t.u_meas, color=ARMC[arm], lw=1.7, label=f"{arm}", zorder=3)
    phi_c = "#6a3d9a"
    axP = axA.twinx()
    tg = _trace("grit")
    # Before its first accepted snapshot GRIT publishes nothing and the
    # controller runs the fixed low-grip fallback. Drawing that stretch as an
    # estimate would credit the estimator with a value it did not produce, so
    # the two are drawn as what they are.
    phi = tg.phi_terrain_est_deg.to_numpy(dtype=float)
    xs = tg.x_fa_meas.to_numpy(dtype=float)
    held = np.isclose(phi, FALLBACK_PHI_DEG, atol=1e-9)
    first = int(np.argmin(held)) if not held.all() else len(phi)
    if first > 0:
        axP.plot(xs[:first], phi[:first], color="0.45", lw=1.6, ls="-",
                 label=f"fallback {FALLBACK_PHI_DEG:.0f}$^\\circ$ (abstaining)")
        axP.axvline(xs[first], color=phi_c, lw=0.9, ls="-", alpha=0.55)
    axP.plot(xs[first:], phi[first:], color=phi_c, lw=1.6, ls=":",
             label="$\\hat\\phi$ published")
    axP.set_ylabel("GRIT $\\hat\\phi$ (deg)", color=phi_c)
    axP.tick_params(axis="y", labelcolor=phi_c)
    axA.set_xlabel("distance along path $x$ (m)")
    axA.set_ylabel("forward speed $u$ (m/s)")
    axA.set_title("(a) clay$\\rightarrow$sand traverse (illustrative)")
    axA.text(10, axA.get_ylim()[1] - 0.12, "clay", color=CLAYC, ha="center", va="top", fontsize=13)
    axA.text(60, axA.get_ylim()[1] - 0.12, "sand", color=SANDC, ha="center", va="top", fontsize=13)
    h1, l1 = axA.get_legend_handles_labels()
    h2, l2 = axP.get_legend_handles_labels()
    axA.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=11, framealpha=0.85,
               handlelength=1.5, borderpad=0.3, labelspacing=0.25, borderaxespad=0.3)

    # --- grouped bars helper (error bars = standard error over the n cells) ---
    def bars(ax, col, sd_col):
        x = np.arange(len(TERR)); w = 0.2
        for i, arm in enumerate(ARMS):
            vals = [m.loc[(arm, t), col] if (arm, t) in m.index else np.nan for t in TERR]
            err = [(m.loc[(arm, t), sd_col] / np.sqrt(m.loc[(arm, t), "n"]))
                   if (arm, t) in m.index else 0 for t in TERR]
            ax.bar(x + (i - 1.5) * w, vals, w, yerr=err, capsize=2,
                   color=ARMC[arm], label=arm, edgecolor="k", linewidth=0.4)
        ax.set_xticks(x); ax.set_xticklabels(list(TERR))

    # --- (b) speed ---
    bars(axB, "mean_u", "mean_u_sd")
    axB.set_ylabel("mean forward speed (m/s)")
    axB.set_title("(b) speed: GRIT $\\approx$ oracle")
    axB.legend(fontsize=11, ncol=2, loc="lower center")
    axB.grid(axis="y", alpha=0.25)

    # --- (c) safety (mean per-run peak crosstrack error) ---
    bars(axC, "max_cte", "max_cte_sd")
    axC.axhline(0.8, ls="--", color="#c0392b", lw=1)
    axC.text(2.42, 0.9, "off path", color="#c0392b", fontsize=13, va="bottom", ha="right")
    axC.set_ylabel("mean peak crosstrack error (m)")
    axC.set_title("(c) safety: only the high-grip prior leaves the path")
    axC.grid(axis="y", alpha=0.25)

    fig.tight_layout(w_pad=2.4)
    out = FIG / "grit_adaptive_speed.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
