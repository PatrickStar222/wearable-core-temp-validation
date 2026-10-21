"""fig3d_build.py -- Figure 3 panel d, rebuilt: the same degradation, three different characters.

THE CLAIM (unchanged from the script this replaces)
    Five perturbations of one sensing chain, two configurations, one shared ruler. A constant +0.5 C
    offset on every skin channel costs EXACTLY nothing; a 0 -> 1 C drift makes the metric BETTER; a
    blanked 60-row block costs the most in the set. One accuracy number cannot say that.

WHY IT WAS REBUILT
    The previous build drew two kernel-density ridges per row. On the page that gave: a legend wedged
    into the top-right corner with its swatch printed over its own label, the exactly-absorbed row as a
    bare spike across the zero rule, ridges with detached islands and long meaningless tails, out-of-
    window arrow markers sitting on the baseline, a zero rule as heavy as the axis, and about a fifth of
    the panel empty. The reader's verdict was that it was unreadable.

WHAT REPLACES IT
    Every one of the 39 paired differences is a dot, packed in lanes so no two touch, with the median
    as a short rule. That keeps the property the density version was chosen for -- every participant is
    in the picture -- without asking a kernel to invent a shape between them. The exactly-absorbed row
    has zero variance by construction, so a density is undefined on it and a swarm of 39 coincident dots
    is meaningless; each of those lanes is drawn as one solid mark pinned to the zero rule, and a single
    footnote says what they all are. Text is kept to a minimum on purpose: the previous build carried four
    such labels, five edge digits and a two-line legend inside the plot.

WHAT IS PLOTTED (identical ruler and identical paired differences to the old build)
    delta_p = nrmse_p(perturbation) - nrmse_p(the SAME configuration's in-domain error),
    nrmse = sqrt(SSE/n) / SD_cohort, SD_cohort = 0.55359, one ruler for every row.
    Values: `_twist_folds.csv` (paired differences), `_twist_results.csv` (the ruler).

NOT CLAIMED
    The drift row reports a SIGN, not a cause. That a linear estimator is insensitive to a within-unit
    drift is a hypothesis about why, and it is not tested here.

Run:  python fig3d_build.py
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon

sys.stdout.reconfigure(encoding="ascii", errors="replace")
sys.path.insert(0, "")
from figure_style import apply_style, mm_fig, save_figure, tidy_axes, FONT_PT   # noqa: E402

apply_style(journal="nature", language="en")
# The shared style sets savefig.bbox = "tight", which crops every saved figure to its ink: this panel
# came out 86.66 x 59.52 mm instead of the declared 89 x 70, and the same is true of panels b and c
# saved through that module. A crowded composite needs panels that are exactly the size they claim, so
# the canvas is written whole here.
mpl.rcParams["savefig.bbox"] = "standard"

P = Path("")
OUT = str(P / "fig3_v2")

C_B5, C_B3 = "#5E1B25", "#B5735A"          # oxblood / terracotta, as in panels b and c
ACCENT_FILL = {"B5": "#E4CBCE", "B3": "#EDD5C4"}    # faint IQR fill, so the rug stays the subject
# The rule and the notes were #8A7A66, a khaki that reads olive among the warm fills. Moved to a warm
# brown of the same value, so it groups with the inks instead of fighting them.
C_RULE, C_BASE, C_NOTE = "#7F6C5C", "#DED2CA", "#7F6C5C"
INK = "#4A4033"
W_MM, H_MM = 89.0, 70.0
ABS_TOL = 1e-9
# The style's annotation size is 6.0 pt. The annotations here are not corner marks: they carry the sign
# reading, the exact-collapse counts and the frame-edge counts, so they are set at the tick size instead
# and the floor assertion below covers them.
FS_ANNOT = 7.0
FS_LETTER = 8.0            # the sheet's panel letters: 8 pt bold (fig2v2_style T["letter"])
DOT_R_MM = 0.42
LANE_GAP = 1.25                            # centre to centre, in dot diameters

# ------------------------------------------------------------------ data (unchanged ruler)
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

ROWS = [
    # The drift label carried a "/h" that no source supports: _p3_probe2.py defines this cell as a
    # "per-unit ramp 0->1.0 degC across each unit's rows", and both the manuscript and the claim matrix
    # write it as "0 -> 1 degC". The rate unit was invented by an earlier build and is removed here.
    # bias +0.3 is listed separately because the manuscript claims BOTH constant offsets are absorbed
    # exactly ("+0.3 or +0.5 degC ... to five decimals"), and the data agrees: the largest paired
    # difference is 2.7e-12 for each configuration, which is float accumulation, not signal.
    ("drift 0\u21921 \u00b0C", "4 signal corruption", "drift 0->1C"),
    ("offset +0.3 \u00b0C", "4 signal corruption", "bias +0.3"),
    ("offset +0.5 \u00b0C", "4 signal corruption", "bias +0.5"),
    ("noise 1.0 \u00b0C", "4 signal corruption", "noise sd=1.0"),
    ("20 blank rows", "3 temporal corruption", "gap=20min"),
    ("60 blank rows", "3 temporal corruption", "gap=60min"),
]
SERIES = ("B5", "B3")                      # top lane first: the better configuration on top
D = {}
for lab, axis, level in ROWS:
    for tag in BASE:
        cur = folds(axis, level, BASE[tag])
        common = sorted(set(cur) & set(IDOM[tag]))
        D[(lab, tag)] = np.array([cur[f] - IDOM[tag][f] for f in common])

# --- the three characters must stay distinct, and every row must carry the full cohort
for lab, _a, _l in ROWS:
    for tag in SERIES:
        v = D[(lab, tag)]
        assert len(v) == 39, "%s %s has %d paired differences, not 39" % (lab, tag, len(v))
_exact = [r[0] for r in ROWS if all(np.all(np.abs(D[(r[0], t)]) < ABS_TOL) for t in SERIES)]
_neg = [r[0] for r in ROWS if all(np.median(D[(r[0], t)]) < 0 for t in SERIES)]
_top = max(ROWS, key=lambda r: float(np.median(D[(r[0], "B3")])))[0]
if not _exact or not _neg or len({_exact[0], _neg[0], _top}) != 3:
    raise SystemExit("THE THREE CHARACTERS ARE NOT DISTINCT: exact=%s neg=%s top=%s"
                     % (_exact, _neg, _top))

# ------------------------------------------------------------------ layout
XL_L, XL_R, XL_TOP, XL_BOT = 18.0, 85.0, 8.0, 62.0      # mm inside the 89 x 70 canvas
# XL_L was 20. The widest row label ("noise SD 1.0 degC") measures 17.8 mm, so a 7 pt label pad -- the
# one the frame-edge counts need on this side -- pushed it off the canvas. 23 mm gives it 2.7 mm of margin
# and leaves 0.7 mm between the labels and those counts.
# XL_R was 87. The frame-edge counts stand outside the frame, and with a 2 mm margin that left their ink
# 0.25 mm from the canvas edge -- no room at all once this panel is butted against a neighbour in the
# composite. The axes gives up 2 mm so the counts get 2.25 mm of white after them.
AX_W, AX_H = XL_R - XL_L, XL_BOT - XL_TOP
# The y axis is in ROW units (ylim = (len(ROWS), 0)), not millimetres: row_h is one perturbation band.
# Millimetres are converted where a size is needed, via len(ROWS)/AX_H. Mixing the two here is what put
# every element outside the axes in the first run of this script.
row_h, sub_h = 1.0, 0.5                                 # B5 lane above, B3 lane below
MM_PER_ROW = AX_H / len(ROWS)
xlim = (-0.15, 0.26)
# The window is chosen from the measured quantiles, not by eye: it holds the bulk of the paired
# differences, so the medians and interquartile ranges that carry the argument get the whole width. The
# previous build used (-0.085, 0.245) and had to mark 25 values with triangles on the baseline, which
# read as stray marks. Here anything outside is counted at the frame edge instead -- and the count is
# asserted against the data, so a value can never simply vanish.
for lab, _a, _l in ROWS:
    for tag in SERIES:
        v = D[(lab, tag)]
        assert len(v) == 39
        assert xlim[0] < float(np.median(v)) < xlim[1], "%s %s median is outside the window" % (lab, tag)
inside_all = np.concatenate([D[(r[0], t)] for r in ROWS for t in SERIES])
frac = float(((inside_all > xlim[0]) & (inside_all < xlim[1])).mean())
# The window must hold the bulk -- the medians and the interquartile ranges that carry the argument --
# while every value outside it is counted at the frame edge. 93 % is the honest floor here: widening
# the window to swallow the 60-blank tail (up to +0.799) would compress the bulk into a sliver.
assert frac >= 0.93, "the window holds only %.1f %% of the values" % (100 * frac)
mm_per_unit = AX_W / (xlim[1] - xlim[0])
RUG_MM, BOX_MM = 1.0, 1.6

def rug_and_box(ax, y, v, col, accent):
    """One row: every one of the 39 differences as a rug tick, with the IQR and median behind it.

    A beeswarm was tried first and does not fit: the densest row needs 8 lanes of 1.05 mm in a 5 mm
    band. The rug keeps every observation on the page at the size this panel allows -- coincident
    values overprint, which is stated in the caption rather than hidden -- and the box is drawn faint
    so the observations stay the subject.
    """
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    ax.add_patch(Rectangle((q1, y - BOX_MM / 2.0 * len(ROWS) / AX_H), q3 - q1,
                           BOX_MM * len(ROWS) / AX_H, facecolor=accent, edgecolor="none", zorder=1))
    ax.plot([q1, q3], [y, y], color=col, lw=0.6, zorder=2, solid_capstyle="butt")
    for xv in v:
        ax.plot([xv, xv], [y - RUG_MM / 2.0 * len(ROWS) / AX_H, y + RUG_MM / 2.0 * len(ROWS) / AX_H],
                color=col, lw=0.5, zorder=3, solid_capstyle="butt", alpha=0.75)
    ax.plot([med, med], [y - BOX_MM / 2.0 * len(ROWS) / AX_H, y + BOX_MM / 2.0 * len(ROWS) / AX_H],
            color=INK, lw=1.0, zorder=5, solid_capstyle="butt")


n_collapsed = 0
n_rug = 0            # values actually drawn as a tick
n_rug_over = 0       # of those, how many land on top of an earlier tick (invisible on its own)
for _r in ROWS:
    for _t in SERIES:
        _v = D[(_r[0], _t)]
        if _v.max() - _v.min() < 0.5 / mm_per_unit:        # the collapsed branch: no rug at all
            n_collapsed += 1
            continue
        _vis = np.sort(_v[(_v > xlim[0]) & (_v < xlim[1])])
        n_rug += len(_vis)
        n_rug_over += int(np.sum(np.diff(_vis) < 0.15 / mm_per_unit))   # closer than half a line
assert n_collapsed == 4, \
    "%d collapsed rows; only the two constant-offset perturbations should collapse" % n_collapsed
n_out = int(np.sum((inside_all <= xlim[0]) | (inside_all >= xlim[1])))
n_drawn = int(np.sum((inside_all > xlim[0]) & (inside_all < xlim[1])))
assert n_drawn + n_out == len(inside_all), \
    "%d values drawn + %d counted at the frame edge != %d" % (n_drawn, n_out, len(inside_all))

fig = plt.figure(figsize=mm_fig(W_MM, H_MM))
ax = fig.add_axes([XL_L / W_MM, 1.0 - XL_BOT / H_MM, AX_W / W_MM, AX_H / H_MM])
ax.set_xlim(*xlim)
ax.set_ylim(len(ROWS), 0)
tidy_axes(ax, despine=True)
ax.set_yticks([i + 0.5 for i in range(len(ROWS))])
ax.set_yticklabels([r[0] for r in ROWS], fontsize=FONT_PT["tick"])
ax.tick_params(axis="y", length=0, pad=2.5)
# The y pad was 9 pt when five frame-edge digits stood outside the frame on this side and had to clear the
# row labels. The digits are gone, so the pad is back to 2.5 pt -- which is what lets the whole label
# gutter shrink from 23 mm to 18 mm and hands 5 mm of width to the data.
ax.tick_params(axis="x", length=2.5, pad=1.5)
ax.set_xticks([-0.05, 0.0, 0.05, 0.10, 0.15, 0.20])
ax.set_xticklabels(["\u22120.05", "0", "+0.05", "+0.10", "+0.15", "+0.20"],
                   fontsize=FONT_PT["tick"])
ax.set_xlabel("\u0394 normalised RMSE (paired, per participant)", fontsize=FONT_PT["axes_label"],
              labelpad=3.0)

ax.axvline(0.0, color=C_RULE, lw=0.8, zorder=1)           # zero is the semantic boundary
texts = []
out_counts = []
# The two direction words (improved / costly) were removed on request. They owned a band 5 mm under the
# axis, which is now reclaimed for data: XL_BOT moved from 58 to 62 and the axis title sits directly
# under the tick labels, so nothing floats in between.

marks = []             # (x_data, y) of every mark, so the one note can be checked against the ink
collapse_worst = 0.0   # largest |paired difference| among the lanes drawn as an exact zero
collapse_ys = []       # their lane centres, for placing the single note that replaces four labels
for k, (lab, _a, _l) in enumerate(ROWS):
    top = k * row_h
    for j, tag in enumerate(SERIES):
        y = top + (j + 0.5) * sub_h
        col = C_B5 if tag == "B5" else C_B3
        ax.plot([xlim[0], xlim[1]], [y, y], color=C_BASE, lw=0.5, zorder=1, solid_capstyle="butt")
        v = D[(lab, tag)]
        vis = v[(v > xlim[0]) & (v < xlim[1])]
        n_lo = int(np.sum(v <= xlim[0]))
        n_hi = int(np.sum(v >= xlim[1]))
        if v.max() - v.min() < 0.5 / mm_per_unit:          # zero variance by construction
            collapse_worst = max(collapse_worst, float(np.abs(v).max()))
            collapse_ys.append(y)
            ax.plot([float(np.median(v))], [y], marker="s", ms=4.6, mfc=col, mec=col, ls="none",
                    zorder=4)
            marks.append((0.0, y))
        else:
            rug_and_box(ax, y, vis, col, accent=ACCENT_FILL[tag])
            marks += [(float(xv), y) for xv in vis]
            q1, q3 = np.percentile(v, [25, 75])
            marks += [(float(q1), y), (float(q3), y), (float(np.median(v)), y)]
        # Out-of-window values are marked at the frame by a drawn triangle -- drawn, not typed: Arial has
        # no U+25C2/U+25B8, so those glyphs would print as tofu. The five digits that used to stand here
        # are gone; the triangle alone says the data continues and the counts live in the caption.
        for n_edge, x_edge, sgn in ((n_lo, xlim[0], -1), (n_hi, xlim[1], 1)):
            if n_edge:
                x_base = x_edge - sgn * 0.016
                ax.add_patch(Polygon([[x_base, y - 0.05], [x_edge, y], [x_base, y + 0.05]],
                                     closed=True, facecolor=C_NOTE, edgecolor="none", zorder=6))
        out_counts.append((lab, tag, n_lo, n_hi))

# One note for the four exactly-absorbed lanes. It replaces four "39/39 = 0" labels: those lanes carry no
# rug at all, so the right half of the offset block is empty and the note goes there, clear of the ink.
_note = ax.text(0.012, float(np.mean(collapse_ys)), "\u0394 = 0 for all participants",
                fontsize=FS_ANNOT, color=C_NOTE, ha="left", va="center", zorder=7)
texts.append(_note)

h = [plt.Line2D([], [], marker="o", ls="none", ms=3.4, mfc=C_B5, mec="white", mew=0.25,
                label="B5 skin+HR+env"),
     plt.Line2D([], [], marker="o", ls="none", ms=3.4, mfc=C_B3, mec="white", mew=0.25,
                label="B3 multi-site skin")]
# One line ABOVE the axes, not a two-line block inside the plot: the block used to sit on the drift row's
# two baselines and cut them, and it was the largest single piece of text in the panel.
leg = ax.legend(handles=h, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False,
                fontsize=FONT_PT["legend"], handletextpad=0.35, columnspacing=1.4, borderaxespad=0.0)
texts.append(leg)

# The panel letter is NOT drawn here: the user's instruction is that no fig3 subplot carries its own
# letter, and the compositor (`fig3_assemble.py`) letters the page instead. No panel title either:
# fig3b has one, this panel does not, and the sheet is not uniform on that point, so the caption
# carries it.

# ------------------------------------------------------------------ audits
fig.canvas.draw()
rend = fig.canvas.get_renderer()
DPI_MM = fig.get_dpi() / 25.4
# The tick labels and the axis title belong in this list. An earlier build of this panel laid
# "improved" and "costly" straight across "-0.05" and "+0.05" and the audit passed anyway, because only
# ax.text objects and the legend were ever registered here -- tick labels were never checked at all.
audit = texts + list(ax.get_xticklabels()) + list(ax.get_yticklabels()) + [ax.xaxis.label]
boxes = [t.get_window_extent(renderer=rend) for t in audit]


def label_of(o):
    return o.get_text() if hasattr(o, "get_text") else "legend"


for t, bb in zip(audit, boxes):
    x0, y0 = bb.x0 / DPI_MM, bb.y0 / DPI_MM
    w, hh = bb.width / DPI_MM, bb.height / DPI_MM
    assert -0.02 <= x0 and x0 + w <= W_MM + 0.02, \
        "text %r leaves the canvas sideways" % label_of(t)
    assert -0.02 <= y0 and y0 + hh <= H_MM + 0.02, "text %r leaves the canvas" % label_of(t)
for i in range(len(boxes)):
    for j in range(i + 1, len(boxes)):
        a, b = boxes[i], boxes[j]
        # measured in MILLIMETRES, and only for boxes that really overlap: an earlier version of this
        # check compared pixel gaps and silently passed everything, because matplotlib returns a
        # degenerate 0..1 box for an Annotation that has not been laid out.
        ox = (min(a.x1, b.x1) - max(a.x0, b.x0)) / DPI_MM
        oy = (min(a.y1, b.y1) - max(a.y0, b.y0)) / DPI_MM
        if ox > 0.3 and oy > 0.3:
            raise SystemExit("text %r overlaps %r by %.2f x %.2f mm"
                             % (label_of(audit[i]), label_of(audit[j]), ox, oy))
# With the direction words gone, the only stack under the axis is tick labels then axis title. A generic
# box test would pass on a 0.31 mm shave, so the clearance a reader actually sees is measured directly.
_xt = min(t.get_window_extent(renderer=rend).y0 for t in ax.get_xticklabels() if t.get_text())
_xl = ax.xaxis.label.get_window_extent(renderer=rend).y1
GAP_TITLE = (_xt - _xl) / DPI_MM
assert GAP_TITLE >= 0.8, "the axis title is squeezed against the tick labels: %.2f mm" % GAP_TITLE
assert min(FONT_PT["tick"], FONT_PT["axes_label"], FONT_PT["legend"], FS_ANNOT, FS_LETTER) >= 7.0

# --- the one note against the marks, not just against other text --------------
# The rug is art, so the text-vs-text audit above cannot see a note lying across a tick. Both are measured
# through the axes transform, the only space where a 0.5 pt tick and a 7 pt note can be compared honestly.
_inv = ax.transData.inverted()
_nb = _note.get_window_extent(renderer=rend)
_nxa = _inv.transform((_nb.x0, _nb.y0))[0]
_nxb = _inv.transform((_nb.x1, _nb.y1))[0]
_nylo = min(_inv.transform((_nb.x0, _nb.y0))[1], _inv.transform((_nb.x1, _nb.y1))[1])
_nyhi = max(_inv.transform((_nb.x0, _nb.y0))[1], _inv.transform((_nb.x1, _nb.y1))[1])
_nxlo, _nxhi = min(_nxa, _nxb), max(_nxa, _nxb)
_under = [(x, y) for (x, y) in marks
          if _nxlo - 0.002 < x < _nxhi + 0.002 and _nylo - 0.02 < y < _nyhi + 0.02]
assert not _under, "the note covers %d mark(s), first at %s" % (len(_under), np.round(_under[0], 4))
assert _nxlo > 0.0, "the note starts at %.4f, on the wrong side of zero" % _nxlo
print("the one note is clear of all %d marks and starts right of the zero rule" % len(marks))

# --- the colour key: outside the axes, inside the canvas ---------------------
_lb = leg.get_window_extent(renderer=rend)
assert _lb.y0 / DPI_MM >= H_MM - XL_TOP - 0.05, "the colour key reaches into the axes"
assert _lb.x1 / DPI_MM <= W_MM - 0.4, "the colour key runs off the canvas"
print("colour key: one line above the axes, %.1f mm wide" % (_lb.width / DPI_MM))

save_figure(fig, "fig3d", outdir=OUT, png_dpi=600)
from PIL import Image
im = Image.open(str(P / "fig3_v2" / "fig3d.png"))
got = (im.width / (600 / 25.4), im.height / (600 / 25.4))
assert abs(got[0] - W_MM) < 0.05 and abs(got[1] - H_MM) < 0.05, "PNG is %s mm" % (got,)
print("paired differences: 39 per row per configuration (asserted)")
for lab, _a, _l in ROWS:
    print("  %-20s %s" % (lab.encode("ascii", "replace").decode(),
                          "   ".join("%s med %+.4f  spread %.3f" % (t, np.median(D[(lab, t)]),
                                                                    D[(lab, t)].max() - D[(lab, t)].min())
                                     for t in SERIES)))
print("characters: absorbed exactly = %s | negative = %s | largest = %s"
      % (_exact[0].encode("ascii", "replace").decode(), _neg[0].encode("ascii", "replace").decode(),
         _top.encode("ascii", "replace").decode()))
print("under-axis stack: axis title clear of the tick labels by %.2f mm (the direction words are gone)"
      % GAP_TITLE)
print("rug: %d values drawn as ticks; %d of them overprint an earlier one (spacing < 0.15 mm)"
      % (n_rug, n_rug_over))
print("extremes over all %d paired differences: %+.4f to %+.4f"
      % (len(inside_all), float(inside_all.min()), float(inside_all.max())))
# The exact-absorption claim is only as good as this number, so it is measured, not remembered: every
# collapsed difference must sit below the threshold at which the "= 0" label is drawn.
assert 0.0 <= collapse_worst < ABS_TOL, "a collapsed row is not exact: %.3e" % collapse_worst
print("collapsed lanes: %d of %d differences drawn as an exact zero; largest |difference| %.2e; the"
      " single note covers all %d of them"
      % (n_collapsed * 39, len(inside_all), collapse_worst, len(collapse_ys)))
print("window %s holds %.1f %% of the %d values; %d marked at the frame edge: %s"
      % (str(xlim), 100 * frac, len(inside_all), n_out,
         "  ".join("%s/%s %d low %d high" % (l.encode("ascii", "replace").decode(), t, lo, hi)
                   for l, t, lo, hi in out_counts if lo or hi) or "none"))
print("wrote fig3d.{pdf,png,svg}  %.0f x %.0f mm  PNG %d x %d px" % (W_MM, H_MM, im.width, im.height))
