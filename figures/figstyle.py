# -*- coding: utf-8 -*-
"""figstyle.py — shared style for the manuscript figures.

Implements the journal-figure-standards tokens:
  * double column width 183 mm, single column 89 mm
  * Arial, body 7-9 pt, axis labels >= 8 pt, never below 5.5 pt
  * line width 0.5-1.2 pt, ticks outward, <= 8 major ticks
  * left and bottom spines only
  * legend without frame, inside the axes, never overlapping data
  * Okabe-Ito colour-blind-safe palette, one palette for the whole paper
  * error bars report mean +/- SD with n and the test named
  * no 3-D, no gradient backgrounds, no rainbow, no gridlines over data

Data figures only.  Schematics go through fig_compose, not this file.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm

# ---------------------------------------------------------------- palette
# Okabe-Ito, colour-blind safe.  Order fixed for the whole paper.
OI = {
    "black":      "#000000",
    "orange":     "#E69F00",
    "skyblue":    "#56B4E9",
    "green":      "#009E73",
    "yellow":     "#F0E442",
    "blue":       "#0072B2",
    "vermillion": "#D55E00",
    "purple":     "#CC79A7",
    "grey":       "#7F7F7F",
}
# semantic assignment, used identically in every figure
C_SCALAR   = OI["vermillion"]   # the transmitted scalar D
C_BEST1D   = OI["blue"]         # best single projection
C_CEILING  = OI["green"]        # two-number ceiling / H
C_COMP_C   = OI["orange"]       # the C component
C_COMMON   = OI["purple"]       # the common mode M
C_WINDOW   = OI["skyblue"]      # observation window
C_GREY     = OI["grey"]

MM = 1.0 / 25.4
W_DOUBLE = 183 * MM     # 7.205 in
W_SINGLE = 89 * MM      # 3.504 in


def apply_style():
    """Global rcParams.  Called once per figure script."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.0,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "lines.linewidth": 1.1,
        "lines.solid_capstyle": "round",
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.borderpad": 0.2,
        "legend.labelspacing": 0.28,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,        # embed TrueType, editable text in vector output
        "ps.fonttype": 42,
        "axes.grid": False,
        "mathtext.fontset": "dejavusans",
    })


def panel_label(ax, s, x=-0.13, y=1.06, fs=9.5):
    """Bold panel letter, uniform top-left, outside the axes.

    pad=True places the letter just inside the axes instead, for panels built from
    bare text (an axes with axis('off')) where there is no data rectangle to sit
    outside of.
    """
    if y is None:
        y = 1.06
    ax.text(x, y, s, transform=ax.transAxes, fontsize=fs, fontweight="bold",
            va="bottom", ha="left", color=OI["black"])


def tidy(ax, nticks=6):
    """Limit tick count, keep spines light."""
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(nticks))
    ax.yaxis.set_major_locator(MaxNLocator(nticks))
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(0.8)


def save(fig, stem, outdir):
    """Deliver vector PDF + 300 dpi PNG (skill §0 requires both)."""
    from pathlib import Path
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    pdf = out / (stem + ".pdf")
    png = out / (stem + ".png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=400)
    print("wrote", pdf)
    print("wrote", png)
    return pdf, png
