# -*- coding: utf-8 -*-
"""figure_style.py —— compatibility shim.

The released scripts were developed against an internal Matplotlib style module of the
same name that is not part of this release. This shim exposes the same public API so the
scripts import and run; it does NOT reproduce the exact typography of the published
figures. The published figures are the authoritative versions.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FONT_PT = 7.0


def apply_style(font_pt=FONT_PT, **kwargs):
    plt.rcParams.update({
        "font.size": font_pt,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    return plt.rcParams


def mm_fig(width_mm, height_mm):
    return plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4))


def save_figure(fig, path, dpi=600):
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.01)
    return path


def tidy_axes(ax, **kwargs):
    for side in ("top", "right"):
        if side in ax.spines:
            ax.spines[side].set_visible(False)
    return ax
