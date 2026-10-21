"""fig3e_build.py -- Panel e: the two configurations do not fail different participants.

THE CLAIM
    Where a perturbation costs anything, it costs the SAME participants in both configurations, and the
    leaner configuration simply pays a multiple of it. Per condition, the 39 paired differences of the
    skin-only configuration are regressed on those of the full configuration:

        condition          r(dB5, dB3)   slope (dB3 per dB5)   sign agreement
        drift 0 -> 1 C        +0.893            0.94                92.3 %
        noise SD 1.0 C        +0.785            1.02                82.1 %
        20 blank rows         +0.916            1.61                79.5 %
        60 blank rows         +0.799            1.73                69.2 %
        offset +0.3/+0.5 C   degenerate: exactly 0 in both configurations, all 39 participants

    So the multiplier, not the membership, is what separates the two configurations; and the share of
    participants whose cost even changes sign falls from 92 % to 69 % as the corruption gets heavier.

WHY THIS PANEL, AND WHY IT IS NOT OWNED ELSEWHERE
    It is the joint view of the same 468 paired differences panel d reports marginally: d answers "what
    does each configuration pay", e answers "is it the same people". No blueprint claims it -- Fig 2's
    panels are about the observation-to-core relation, Fig 5's about environmental shift and external
    transfer -- and it introduces no new denominator: same folds, same SD_cohort 0.55359, B5 and B3 only,
    which is the delivered convention recorded in FIG3_DESIGN_V2.md section 3.

WHY THE CELLS HAVE EQUAL AXES
    x and y carry the same quantity for two configurations, so each cell uses ONE range on both axes.
    That is what makes the identity line a true 45 degrees and a slope above 1 visibly fan upwards; with
    independent ranges every cell would look alike and the finding would be invisible.

WHAT THE PANEL DOES NOT CLAIM
    A slope above 1 is not evidence that the skin-only configuration is worse in general -- it is the
    within-participant ratio under these nine controlled conditions, this estimator and this cohort. The
    two offset conditions are drawn nowhere because they are (0, 0) for all 39 participants; one note says
    so, and the build asserts the exactness.

Run:  python fig3e_build.py
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

sys.stdout.reconfigure(encoding="ascii", errors="replace")
sys.path.insert(0, "")
from figure_style import apply_style, mm_fig, FONT_PT   # noqa: E402

apply_style(journal="nature", language="en")
mpl.rcParams["savefig.bbox"] = "standard"      # figure_style crops to ink; this panel is written whole

P = Path("")
OUT = P / "fig3_v2"

C_B5, C_B3 = "#5E1B25", "#B5735A"      # the sheet's two configuration colours. NOT used for text here:
                                       # journals print axis labels in black, and the two configurations
                                       # are named in words inside those labels
C_DOT = "#9C7C90"                      # the marks: a light dusty plum. Chosen against the project's own
                                       # vocabulary, not by taste: SKIN is coral, CORE is teal
                                       # (`#0E5F7A`, and this panel's vertical axis IS core temperature),
                                       # AMB is the amber ramp, EAR is periwinkle -- all four are taken,
                                       # and anything from the oxblood/terracotta family would read as B5
                                       # or B3, which a pair of configurations is not
INK = "#000000"                        # every label and the fitted line: black, untinted
C_NOTE = "#8A7F76"                     # panel c's warm grey, for the in-cell values and the key
C_FRAME, C_IDENT = "#DED4CA", "#C4BAB0"  # panel c's hairline colour, one step darker for the identity
                                       # line so the two rules stay distinguishable
W_MM, H_MM = 89.0, 66.5
ABS_TOL = 1e-9
FS_CELL, FS_NOTE, FS_LETTER = 7.5, 7.0, 8.0

# ------------------------------------------------------------------ data (identical ruler to panels c and d)
BASE = {"B3": "B3 multi-site skin", "B5": "B5 skin+HR+env"}
PUB = {}
with open(P / "_twist_results.csv", newline="", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        if r["rep"] == "":
            PUB[(r["axis"], r["level"], r["baseline"])] = float(r["rmse"]), float(r["nrmse"])
_SD = [PUB[("1 subject shift", "LOSO (participant-grouped)", b)][0] /
       PUB[("1 subject shift", "LOSO (participant-grouped)", b)][1] for b in BASE.values()]
SDC = float(np.mean(_SD))
if max(_SD) - min(_SD) > 1e-6:
    raise SystemExit("the cohort SD is not one number -- the baselines are on different targets")

F = defaultdict(lambda: [0.0, 0])
for r in csv.DictReader(open(P / "_twist_folds.csv", encoding="utf-8")):
    k = (r["axis"], r["level"], r["baseline"], r["fold_kind"], r["fold_id"])
    F[k][0] += float(r["sse"])
    F[k][1] += int(r["n"])


def folds(axis, level, base):
    return {fid: np.sqrt(sse / n) / SDC
            for (a, l, b, kd, fid), (sse, n) in F.items()
            if a == axis and l == level and b == base and kd == "participant" and n}


IDOM = {t: folds("1 subject shift", "LOSO (participant-grouped)", b) for t, b in BASE.items()}

# Four conditions carry a real cost and are drawn; the two constant offsets are exact zeros in both
# configurations and are carried by one note.
DRAWN = [("drift 0\u21921 \u00b0C", "4 signal corruption", "drift 0->1C"),
         ("noise 1.0 \u00b0C", "4 signal corruption", "noise sd=1.0"),
         # The two block conditions carry two-line labels: on one line the fitted line ran through the
         # text, and at this cell width a shortened single-line label would have lost the word "blank".
         ("20 blank\nrows", "3 temporal corruption", "gap=20min"),
         ("60 blank\nrows", "3 temporal corruption", "gap=60min")]
EXACT = [("offset +0.3 \u00b0C", "4 signal corruption", "bias +0.3"),
         ("offset +0.5 \u00b0C", "4 signal corruption", "bias +0.5")]

D, STAT = {}, {}
for lab, axis, level in DRAWN:
    cur = {t: folds(axis, level, BASE[t]) for t in BASE}
    common = sorted(set(cur["B5"]) & set(cur["B3"]) & set(IDOM["B5"]) & set(IDOM["B3"]))
    d5 = np.array([cur["B5"][f] - IDOM["B5"][f] for f in common])
    d3 = np.array([cur["B3"][f] - IDOM["B3"][f] for f in common])
    assert len(d5) == 39, "%s has %d paired differences, not 39" % (lab, len(d5))
    D[lab] = (d5, d3)
    r = float(np.corrcoef(d5, d3)[0, 1])
    slope, intercept = (float(x) for x in np.polyfit(d5, d3, 1))
    agree = 100.0 * float(np.mean(np.sign(d5) == np.sign(d3)))
    STAT[lab] = (r, slope, intercept, agree)
    assert slope > 0.5, "%s: the two configurations no longer move together (slope %.3f)" % (lab, slope)
    assert r > 0.5, "%s: concordance collapsed (r %.3f)" % (lab, r)
worst_exact = 0.0
for lab, axis, level in EXACT:
    for t in BASE:
        cur = folds(axis, level, BASE[t])
        common = sorted(set(cur) & set(IDOM[t]))
        v = np.array([cur[f] - IDOM[t][f] for f in common])
        assert len(v) == 39, "%s %s has %d paired differences" % (lab, t, len(v))
        worst_exact = max(worst_exact, float(np.abs(v).max()))
assert worst_exact < ABS_TOL, "the offset conditions are not exact any more: %.3e" % worst_exact

# ------------------------------------------------------------------ layout (millimetres)
XL_L, XR = 15.0, 84.0
# The column joint is 9.5 mm, not 4.5: the right column's y tick labels are about 7.7 mm wide and sit
# 1.2 mm clear of their frame, so at a 4.5 mm joint they reached 2.2 mm INTO the left cell -- over its
# data. The cross-facet gate below now measures exactly that, for both axes.
JOINT_X, JOINT_Y = 9.5, 5.0
CELL_W = (XR - XL_L - JOINT_X) / 2.0        # 29.75 mm
CELL_H = 24.0                               # deliberately not equal to CELL_W: see the note on the frame
TOP = 1.4                                   # the panel letter
GRID_TOP = 6.2
GRID_BOT = GRID_TOP + 2 * CELL_H + JOINT_Y
assert GRID_BOT + 7.0 <= H_MM, "only %.1f mm left under the grid for the tick labels and the axis title" % (
    H_MM - GRID_BOT)
assert CELL_H / CELL_W > 0.8, ("the cells are too wide for their height: the identity line would read "
                              "flatter than about 40 degrees")

def cx(j):
    return XL_L + j * (CELL_W + JOINT_X)

def cy(i):
    return GRID_TOP + i * (CELL_H + JOINT_Y)

# ------------------------------------------------------------------ draw
fig = plt.figure(figsize=mm_fig(W_MM, H_MM))
texts = []
for k, (lab, _a, _l) in enumerate(DRAWN):
    i, j = divmod(k, 2)
    d5, d3 = D[lab]
    lo = float(min(d5.min(), d3.min()))
    hi = float(max(d5.max(), d3.max()))
    pad = 0.06 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    ax = fig.add_axes([cx(j) / W_MM, 1.0 - (cy(i) + CELL_H) / H_MM, CELL_W / W_MM, CELL_H / H_MM])
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)                     # ONE range on both axes: the identity line is a real 45 deg
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C_FRAME)
        ax.spines[side].set_linewidth(0.5)
    ax.plot([lo, hi], [lo, hi], color=C_IDENT, lw=0.6, ls=(0, (2.2, 1.8)), zorder=1)
    r, slope, intercept, agree = STAT[lab]
    ax.plot([lo, hi], [intercept + slope * lo, intercept + slope * hi], color=INK, lw=0.9, zorder=3,
            solid_capstyle="round")
    ax.plot(d5, d3, marker="o", ms=2.4, mfc=C_DOT, mec="white", mew=0.25, ls="none", alpha=1.0, zorder=4)
    for a in (ax.xaxis, ax.yaxis):
        a.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10]))
        a.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f"))
    ax.tick_params(length=2.0, pad=1.2, labelsize=FONT_PT["tick"])
    # Every cell keeps its own labels on BOTH axes: the four cells do not share a range, so a number
    # shown for one cell would be wrong for its neighbour, and hiding it leaves that cell unreadable.
    # The ticks are NOT copied from one axis to the other with set_xticks(): that call is allowed to
    # widen the view limits to include the ticks, which silently broke the equal-range invariant.
    ticks_x, ticks_y = ax.get_xticks(), ax.get_yticks()
    assert np.allclose(ticks_x, ticks_y), \
        "%s: the two axes carry different ticks although their ranges are equal" % lab
    t = ax.text(0.035, 0.955, lab, transform=ax.transAxes, fontsize=FS_CELL, color=INK, ha="left",
                va="top", zorder=6)
    texts.append(t)
    t = ax.text(0.955, 0.06, "r %.2f  \u00b7  \u00d7%.2f" % (r, slope), transform=ax.transAxes,
                fontsize=FS_NOTE, color=C_NOTE, ha="right", va="bottom", zorder=6)
    texts.append(t)

# The two lines in every cell need identifying once, not four times.
# A real legend rather than typed glyphs: Arial has no U+22EF and that entry rendered as a tofu box.
_handles = [Line2D([], [], color=INK, lw=0.9, label="fitted slope"),
            Line2D([], [], color=C_IDENT, lw=0.6, ls=(0, (2.2, 1.8)), label="equal cost (identity)")]
leg = fig.legend(handles=_handles, loc="lower right", bbox_to_anchor=(XR / W_MM, 1.0 - 4.6 / H_MM),
                 ncol=2, frameon=False, fontsize=FS_NOTE, handlelength=1.7, handletextpad=0.4,
                 columnspacing=1.6, borderaxespad=0.0)
texts.append(leg)
t = fig.text((XL_L + XR) / 2.0 / W_MM, 1.0 - (GRID_BOT + 5.6) / H_MM,
             "\u0394 normalised RMSE, full configuration (B5)", fontsize=FONT_PT["axes_label"],
             color=INK, ha="center", va="center")
texts.append(t)
t = fig.text(2.8 / W_MM, 1.0 - (GRID_TOP + CELL_H + JOINT_Y / 2.0) / H_MM,
             "\u0394 normalised RMSE, skin-only (B3)", fontsize=FONT_PT["axes_label"], color=INK,
             ha="center", va="center", rotation=90)
texts.append(t)


# ------------------------------------------------------------------ audits
fig.canvas.draw()
rend = fig.canvas.get_renderer()
DPI_MM = fig.get_dpi() / 25.4
boxes = [t.get_window_extent(renderer=rend) for t in texts]


def label_of(o):
    return o.get_text() if hasattr(o, "get_text") else "?"


for t, bb in zip(texts, boxes):
    x0, y0 = bb.x0 / DPI_MM, bb.y0 / DPI_MM
    w, h = bb.width / DPI_MM, bb.height / DPI_MM
    assert -0.02 <= x0 and x0 + w <= W_MM + 0.02, "text %r leaves the canvas sideways" % label_of(t)
    assert -0.02 <= y0 and y0 + h <= H_MM + 0.02, "text %r leaves the canvas" % label_of(t)
for i in range(len(boxes)):
    for j in range(i + 1, len(boxes)):
        a, b = boxes[i], boxes[j]
        ox = (min(a.x1, b.x1) - max(a.x0, b.x0)) / DPI_MM
        oy = (min(a.y1, b.y1) - max(a.y0, b.y0)) / DPI_MM
        if ox > 0.3 and oy > 0.3:
            raise SystemExit("text %r overlaps %r by %.2f x %.2f mm"
                             % (label_of(texts[i]), label_of(texts[j]), ox, oy))
assert min(FONT_PT["tick"], FONT_PT["axes_label"], FS_CELL, FS_NOTE, FS_LETTER) >= 7.0

# Every point inside its own cell, and the identity line a true 45 degrees (equal axes).
for k, (lab, _a, _l) in enumerate(DRAWN):
    ax = fig.axes[k]
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    assert abs((x1 - x0) - (y1 - y0)) < 1e-9, \
        "%s: the axes are not equal, so the identity line is not 45 degrees" % lab
    d5, d3 = D[lab]
    assert x0 < d5.min() and d5.max() < x1 and y0 < d3.min() and d3.max() < y1, \
        "%s: a point sits on the frame" % lab
print("cells: 4 conditions x 39 participants, equal axes per cell; %d texts audited" % len(texts))

# Each cell's own two annotations inside that cell: a caption-like label that drifts into the neighbouring
# facet is the failure this panel is most exposed to, because the cells are only 26.8 mm wide.
for k, (lab, _a, _l) in enumerate(DRAWN):
    i, j = divmod(k, 2)
    for o in fig.axes[k].texts:
        bb = o.get_window_extent(renderer=rend)
        x0, x1 = bb.x0 / DPI_MM, bb.x1 / DPI_MM
        y0, y1 = H_MM - bb.y1 / DPI_MM, H_MM - bb.y0 / DPI_MM
        assert cx(j) - 0.02 <= x0 and x1 <= cx(j) + CELL_W + 0.02, \
            "%s: %r spans %.2f..%.2f mm, outside its %.2f mm cell" % (lab, o.get_text(), x0, x1, CELL_W)
        assert cy(i) - 0.02 <= y0 and y1 <= cy(i) + CELL_H + 0.02, \
            "%s: %r leaves its cell vertically" % (lab, o.get_text())
print("both in-cell annotations fit inside their own %.1f mm cell in all 4 conditions" % CELL_W)

# Text against the ART, which the text-vs-text audit cannot see: a condition label crossed by the fitted
# line is a defect this panel had (found by eye, fixed by shortening and splitting the label), so it is
# gated here in data units instead.
for k, (lab, _a, _l) in enumerate(DRAWN):
    ax = fig.axes[k]
    lo_x, hi_x = ax.get_xlim()
    r, slope, intercept, agree = STAT[lab]
    d5, d3 = D[lab]
    inv = ax.transData.inverted()
    for o in ax.texts:
        bb = o.get_window_extent(renderer=rend)
        (x0d, y0d) = inv.transform((bb.x0, bb.y0))
        (x1d, y1d) = inv.transform((bb.x1, bb.y1))
        xlo, xhi = sorted((x0d, x1d))
        ylo, yhi = sorted((y0d, y1d))
        for nm, (m_, b_) in (("identity", (1.0, 0.0)), ("fitted", (slope, intercept))):
            xs = np.linspace(max(xlo, lo_x), min(xhi, hi_x), 40)
            ys = m_ * xs + b_
            assert not ((ys > ylo) & (ys < yhi)).any(), \
                "%s: the %s line runs through %r" % (lab, nm, o.get_text())
        n_under = int(((d5 > xlo) & (d5 < xhi) & (d3 > ylo) & (d3 < yhi)).sum())
        assert n_under == 0, "%s: %d point(s) sit under %r" % (lab, n_under, o.get_text())
print("no condition label is crossed by either line, and no point sits under a label, in any cell")

# Text of one facet inside ANOTHER facet's box. The gate above only knows each axes' own artists, so a
# tick label that wandered out of the joint and over the neighbouring cell was invisible to it.
for k, ax in enumerate(fig.axes):
    ab = ax.get_window_extent(renderer=rend)
    ax0, ay0, ax1, ay1 = (ab.x0 / DPI_MM, ab.y0 / DPI_MM, ab.x1 / DPI_MM, ab.y1 / DPI_MM)
    for k2, ax2 in enumerate(fig.axes):
        if k2 == k:
            continue
        for o in list(ax2.texts) + list(ax2.get_yticklabels()) + list(ax2.get_xticklabels()):
            if not o.get_text():
                continue
            b = o.get_window_extent(renderer=rend)
            bx0, by0, bx1, by1 = b.x0 / DPI_MM, b.y0 / DPI_MM, b.x1 / DPI_MM, b.y1 / DPI_MM
            ox = min(ax1, bx1) - max(ax0, bx0)
            oy = min(ay1, by1) - max(ay0, by0)
            assert not (ox > 0.2 and oy > 0.2), \
                "%r of cell %d intrudes into cell %d by %.2f x %.2f mm" % (
                    o.get_text(), k2, k, ox, oy)
print("no facet's text reaches into a neighbouring facet (4 cells checked pairwise)")
for lab, _a, _l in DRAWN:
    r, slope, intercept, agree = STAT[lab]
    print("  %-18s r = %+.3f   slope %+.3f   sign agreement %4.1f %%   n = %d"
          % (lab.encode("ascii", "replace").decode(), r, slope, agree, len(D[lab][0])))
print("the two constant offsets: largest |difference| over both configurations = %.2e (gate 1e-09)"
      % worst_exact)

fig.savefig(OUT / "fig3e.pdf", facecolor="white")
fig.savefig(OUT / "fig3e.svg", facecolor="white")
fig.savefig(OUT / "fig3e.png", dpi=600, facecolor="white")
from PIL import Image
im = Image.open(OUT / "fig3e.png")
got = (im.width / (600 / 25.4), im.height / (600 / 25.4))
assert abs(got[0] - W_MM) < 0.05 and abs(got[1] - H_MM) < 0.05, "PNG is %s mm" % (got,)
print("wrote fig3e.{pdf,svg,png}  %.0f x %.0f mm  PNG %d x %d px" % (W_MM, H_MM, im.width, im.height))
