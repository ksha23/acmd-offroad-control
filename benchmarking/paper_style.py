"""Shared Matplotlib style for the manuscript's figures.

Applying one style across every figure script keeps the figures visually
consistent with each other and with the manuscript body: a Times-compatible
serif matching the body text, predominantly grayscale series with a single navy
accent reserved for the method under test, light grids, and thin spines. Holding
the accent to one hue means that in any figure the reader can identify the
deployed method without consulting the legend, and the remaining series stay
distinguishable when the page is printed in grayscale.

Import the module and call ``paper_style.apply()`` once at the top of a figure
script, after importing pyplot, then draw using the palette constants below.
"""
from __future__ import annotations
import matplotlib as mpl

# --- palette ---------------------------------------------------------------
INK    = "#1a1a1a"   # text, axes, and primary foreground series
ACCENT = "#1f3b63"   # deep navy: the single accent, reserved for the deployed method
GRID   = "#dcdcdc"

# Muted qualitative set for categorical plots: the navy accent, grays, and two
# restrained earth and teal hues. The saturated red/green/gold triad is avoided
# because it reads as a status signal and is not distinguishable to readers with
# the common colour-vision deficiencies.
MUTED = ["#1f3b63",  # navy: accent and primary
         "#8a8f96",  # mid gray
         "#9c6b3f",  # muted ochre-brown
         "#5b7f6f",  # muted teal
         "#b7bcc2"]  # light gray

# Grayscale ramp, dark to light, for reference arms that should recede behind
# the accented series.
GRAYS = ["#2b2b2b", "#707070", "#a6a6a6", "#cccccc"]

# Terrain-coded hues for the per-terrain figures, kept constant across figures
# so that a soil keeps one colour throughout the manuscript.
TERRAIN = {"clay": "#1f3b63", "dirt": "#9c6b3f", "sand": "#b08a3e"}

# Single-hue sequential colormap for heatmaps, light for low and dark for high,
# so that magnitude survives grayscale reproduction.
SEQ_CMAP = "Blues"


def apply() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Liberation Serif", "Nimbus Roman", "Times New Roman",
                       "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.titlesize": 10.5,
        "axes.labelsize": 10,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.alpha": 1.0,
        "xtick.color": INK, "ytick.color": INK,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "xtick.direction": "out", "ytick.direction": "out",
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "axes.prop_cycle": mpl.cycler(color=MUTED),
    })
