"""fig3f_build.py -- Panel f: how much a perturbation costs is a property of the perturbation, not of the
person.

THE CLAIM
    Change exactly ONE thing and measure how far a participant's rank moves. Hold the perturbation and change
    the configuration: half the participants move 5 positions or less. Hold the configuration and change the
    perturbation: half of them move 11 positions or more. The shape of the two cumulative distributions --
    where each one turns upwards, how steeply, and how far apart they sit -- IS the result; nothing in this
    panel exists to announce it in words.

THE FORM: an academic ECDF, two weights of ink and no decoration
      * one main ECDF per family: the pooled distribution of every comparison in that family (156 and 468
        participant comparisons), drawn as a staircase, thin and opaque, in the family's colour;
      * every individual comparison is drawn too, uniformly very light and very thin, in the same colour as
        its family. They are the heterogeneity -- no unit is hidden -- and at this weight they read as
        texture behind the two main curves rather than as sixteen objects to track;
      * NO filled area anywhere. Large translucent ribbons were the previous version's defect: they turned a
        statistical figure into an infographic and put more ink into the bands than into the data;
      * the annotations are two short lines of numbers in the one empty triangle, and the family names are
        small (i)/(ii) labels in the strip above the curves -- not titles;
      * white ground, no frame, no grid, one barely-there rule at 0.50.
    Shape first, words last: if the reader takes one thing from this panel it should be that the deep-wine
    staircase turns upward several positions before the grey-purple one.

WHY THIS PANEL
    Panel d shows each participant's cost as a distribution and panel e shows that the two configurations
    fail the same people. Both invite the next question -- "is there a fragile subgroup?" -- and the answer is
    no. The panel exists to stop that reading before a reader makes it.

OWNERSHIP, checked before building
    The six-panel plan lists this slot as "the cost of moving the same probe to 10 C" (Fujii). That content is
    the manuscript's Figure 5c and belongs to Figure 5; the asset on disk (`fig5_translation_layer.*`) is only
    a PDF wrapper for that figure's SVG master, not a panel. The sheet already spends a ladder (a), a matrix
    (b), a slopegraph (c), lane rugs with IQR bands (d) and paired scatters (e); a cumulative curve is none of
    those, and the six earlier versions of this panel that copied one of them are recorded in the caption.

THE UNIT AND THE RULER
    Unit = participant; the |delta| ranked here is |nrmse(perturbed) - nrmse(same configuration in domain)|,
    on the LOSO participant folds with B3/B5 only -- the delivered convention of FIG3_DESIGN_V2.md section 3,
    with SD_cohort = 0.55359. Rank 1 = the largest change. The measure is taken in magnitude rather than
    signed because the drift perturbation IMPROVES the metric, so a signed delta would let participants who
    improved cancel those who worsened; panel d carries the signed view.

WHAT IT DOES NOT CLAIM
    A position of rank change is not a cost: ranks are relative to their own perturbation. A rank correlation
    near zero says this cohort's ordering does not persist across these four perturbations with this
    estimator -- not that individual sensitivity is unreal, since per-participant mean |delta| spans 9-fold
    (B3) to 24-fold (B5). The spread of the light curves describes how much the comparisons inside a family
    disagree with each other; it is not a confidence interval for the main curve.

Run:  python fig3f_build.py
"""
import csv
import itertools
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="ascii", errors="replace")
sys.path.insert(0, "")
from figure_style import apply_style, mm_fig, FONT_PT   # noqa: E402

apply_style(journal="nature", language="en")
mpl.rcParams["savefig.bbox"] = "standard"      # figure_style crops to ink; this panel is written whole

P = Path("")
OUT = P / "fig3_v2"

WINE = "#6B2233"                       # deep wine: the comparison whose ranks survive.
MAUVE = "#9C7C90"                      # grey-purple, panel e's accent: the one whose ranks do not.
INK = "#000000"
GROUND = "#FFFFFF"
C_RULE, C_TICK, C_FRAME = "#EFE8E0", "#000000", "#000000"
W_MM, H_MM = 89.0, 50.0
FS_NOTE, FS_LETTER = 7.0, 8.0
LW_FAINT, AL_FAINT = 0.30, 0.22        # every individual comparison, uniformly
LW_MAIN = 1.15
ANN_BUDGET = 88                        # characters of annotation allowed INSIDE the frame, see the audit

# ------------------------------------------------------------------ data (the same frozen folds)
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

COND = [("drift 0\u21921 \u00b0C", "drift", "4 signal corruption", "drift 0->1C"),
        ("noise 1.0 \u00b0C", "noise", "4 signal corruption", "noise sd=1.0"),
        ("20 blank rows", "20 blank", "3 temporal corruption", "gap=20min"),
        ("60 blank rows", "60 blank", "3 temporal corruption", "gap=60min")]


def build(tag):
    common = None
    for _l, _s, axis, level in COND:
        c = sorted(set(folds(axis, level, BASE[tag])) & set(IDOM[tag]))
        common = c if common is None else [x for x in common if x in set(c)]
    A = np.array([[abs(folds(axis, level, BASE[tag])[f] - IDOM[tag][f])
                   for (_l, _s, axis, level) in COND] for f in common])
    assert A.shape == (39, 4), "%s matrix is %s, expected (39 participants, 4 perturbations)" % (tag, A.shape)
    return A


M = {tag: build(tag) for tag in BASE}


def ranks_desc(x):
    """Rank 1 = LARGEST |delta|. argsort-of-argsort is ascending, which would invert every count printed
    below; the convention is asserted, not commented."""
    return len(x) - np.argsort(np.argsort(x))


RANKS = {tag: np.array([ranks_desc(M[tag][:, i]) for i in range(4)]).T for tag in BASE}
for tag in BASE:
    for i in range(4):
        assert RANKS[tag][int(np.argmax(M[tag][:, i])), i] == 1, "rank 1 is not the largest |delta|"
        assert RANKS[tag][int(np.argmin(M[tag][:, i])), i] == 39, "rank 39 is not the smallest |delta|"


def rho(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


rng = np.random.default_rng(20261004)


def boot(fn, n=4000):
    return np.percentile([fn(rng.integers(0, 39, 39)) for _ in range(n)], [2.5, 97.5])


PAIRS = list(itertools.combinations(range(4), 2))

# FAMILY 1: hold the perturbation, change the configuration. One comparison per perturbation, one rank change
# per participant, so a comparison belongs to neither configuration.
FAM_CFG = []
for i in range(4):
    FAM_CFG.append({"label": COND[i][0], "v": np.abs(RANKS["B5"][:, i] - RANKS["B3"][:, i]),
                    "r": rho(M["B5"][:, i], M["B3"][:, i]),
                    "ci": boot(lambda s, i=i: rho(M["B5"][s][:, i], M["B3"][s][:, i]))})
# FAMILY 2: hold the configuration, change the perturbation. Six pairs in each of the two configurations.
FAM_PERT = []
for tag in BASE:
    for p in PAIRS:
        A = M[tag]
        FAM_PERT.append({"label": "%s %s" % (tag, "-".join(COND[k][1] for k in p)),
                         "v": np.abs(RANKS[tag][:, p[1]] - RANKS[tag][:, p[0]]),
                         "r": rho(A[:, p[0]], A[:, p[1]]),
                         "ci": boot(lambda s, p=p, A=A: rho(A[s][:, p[0]], A[s][:, p[1]]))})

assert len(FAM_CFG) == 4 and len(FAM_PERT) == 12, "the two families are not 4 and 12 comparisons"
assert all(len(c["v"]) == 39 for c in FAM_CFG + FAM_PERT), "a comparison is not 39 participants"
POOL_CFG = np.concatenate([c["v"] for c in FAM_CFG])
POOL_PERT = np.concatenate([c["v"] for c in FAM_PERT])
assert len(POOL_CFG) == 156 and len(POOL_PERT) == 468, "a main curve pools the wrong number of comparisons"

R_CFG = np.array([c["r"] for c in FAM_CFG])
R_PERT = np.array([c["r"] for c in FAM_PERT])
MEAN_CFG = np.array([c["v"].mean() for c in FAM_CFG])
MEAN_PERT = np.array([c["v"].mean() for c in FAM_PERT])
# What the panel claims, asserted: the two families' rank correlations do not overlap, and neither does their
# mean displacement. The medians and 75th percentiles DO overlap (60 blank rows is the perturbation whose
# configuration change displaces ranks as much as a perturbation change does) and are deliberately not
# asserted -- a gate written to fit them would hide that.
assert R_CFG.min() - R_PERT.max() > 0.15, "the rank correlations no longer separate: %s vs %s" % (
    R_CFG.round(2), R_PERT.round(2))
assert MEAN_CFG.max() < MEAN_PERT.min(), "the mean displacement no longer separates: %s vs %s" % (
    MEAN_CFG.round(2), MEAN_PERT.round(2))
assert R_CFG.min() > 0.4, "the within-perturbation agreement collapsed"
assert R_PERT.max() < 0.45, "a stable fragile subgroup appeared: max rho %.3f" % R_PERT.max()
MEAN = {}
for tag in BASE:
    MEAN[tag] = (float(np.mean([c["r"] for c in FAM_PERT if c["label"].startswith(tag)])),
                 boot(lambda s, tag=tag: float(np.mean([rho(M[tag][s][:, p[0]], M[tag][s][:, p[1]])
                                                        for p in PAIRS]))))
POOL = (float(np.mean([c["r"] for c in FAM_PERT])),
        boot(lambda s: float(np.mean([rho(M[t][s][:, p[0]], M[t][s][:, p[1]])
                                      for t in BASE for p in PAIRS]))))
assert POOL[1][1] < 0.35, "a stable fragile subgroup appeared: pooled upper bound %.3f" % POOL[1][1]


def median_of(v):
    x = np.sort(v)
    return float(x[int(np.argmax(np.arange(1, len(x) + 1) / float(len(x)) >= 0.5))])


MED_POOL = (median_of(POOL_CFG), median_of(POOL_PERT))


def lum(hexcol):
    c = [int(hexcol[i:i + 2], 16) / 255.0 for i in (1, 3, 5)]
    c = [(v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4) for v in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


# "One deep, one light, clearly different": the two family colours are asserted to differ in lightness, so a
# later edit cannot quietly bring them together.
assert abs(lum(WINE) - lum(MAUVE)) >= 0.10, "the two family colours are no longer one deep and one light"

print("family 1 -- configuration changed (%d comparisons, pooled over %d):" % (len(FAM_CFG), len(POOL_CFG)))
for c in FAM_CFG:
    print("   %-16s median |d rank| %4.1f   mean %4.2f   rho %+.3f  %+.3f..%+.3f"
          % (c["label"], np.median(c["v"]), c["v"].mean(), c["r"], c["ci"][0], c["ci"][1]))
print("   pooled: median %d  mean %.1f  75th %.0f  90th %.0f   rho range %.2f..%.2f"
      % (MED_POOL[0], POOL_CFG.mean(), np.percentile(POOL_CFG, 75), np.percentile(POOL_CFG, 90),
         R_CFG.min(), R_CFG.max()))
print("family 2 -- perturbation changed (%d comparisons, pooled over %d):" % (len(FAM_PERT), len(POOL_PERT)))
for c in FAM_PERT:
    print("   %-20s median |d rank| %4.1f   mean %4.2f   rho %+.3f  %+.3f..%+.3f"
          % (c["label"], np.median(c["v"]), c["v"].mean(), c["r"], c["ci"][0], c["ci"][1]))
print("   pooled: median %d  mean %.1f  75th %.0f  90th %.0f   rho range %.2f..%.2f"
      % (MED_POOL[1], POOL_PERT.mean(), np.percentile(POOL_PERT, 75), np.percentile(POOL_PERT, 90),
         R_PERT.min(), R_PERT.max()))
print("   share moving more than 10 positions: %.0f %% against %.0f %%"
      % (100.0 * (POOL_CFG > 10).mean(), 100.0 * (POOL_PERT > 10).mean()))
print("   colour lightness: wine %.3f, grey-purple %.3f" % (lum(WINE), lum(MAUVE)))
for tag in BASE:
    print("   mean rho over the six pairs, %s: %+.3f  %+.3f..%+.3f" % (tag, MEAN[tag][0],
                                                                      MEAN[tag][1][0], MEAN[tag][1][1]))
print("   pooled over both configurations: %+.3f  %+.3f..%+.3f" % (POOL[0], POOL[1][0], POOL[1][1]))

# The panel says "nobody stays near the top", so the count behind that sentence is printed every build.
for tag in BASE:
    for topn in (5, 10):
        hits = (RANKS[tag] <= topn).sum(axis=1)
        print("   %s, top %2d most affected: %2d participants in >=2 perturbations, %d in >=3, %d in all four"
              % (tag, topn, int((hits >= 2).sum()), int((hits >= 3).sum()), int((hits == 4).sum())))
worst = int(np.argmax(M["B5"][:, 0]))                   # the participant drift moved most, under B5
print("   the participant drift moved most is ranked %s of 39 (drift, noise, 20, 60), rank 1 = most affected"
      % "/".join(str(int(v)) for v in RANKS["B5"][worst]))
for tag in BASE:                                        # how large individual differences get at all
    per = M[tag].mean(axis=1)
    print("   %s per-participant mean |delta| spans %.4f..%.4f (%.0fx), n=%d, %d paired differences"
          % (tag, per.min(), per.max(), per.max() / per.min(), len(per), M[tag].size))

# ------------------------------------------------------------------ layout (millimetres)
X0, X1 = 14.5, 87.5                                    # the framed plot is 73 of the 89 mm
YT, YB = 5.0, 42.0                                     # and 37 mm tall, all of it data
XMAX = 39.0
fig = plt.figure(figsize=mm_fig(W_MM, H_MM))
fig.patch.set_facecolor(GROUND)
ax = fig.add_axes([X0 / W_MM, 1.0 - YB / H_MM, (X1 - X0) / W_MM, (YB - YT) / H_MM])
ax.set_facecolor(GROUND)
ax.set_xlim(0.0, XMAX)
ax.set_ylim(0.0, 1.0)
ax.axhline(0.5, color=C_RULE, lw=0.5, zorder=1)         # the halfway line, barely there


def draw(v, **kw):
    x = np.sort(v)
    ax.step(np.concatenate([[0.0], x]), np.concatenate([[0.0], np.arange(1, len(x) + 1) / float(len(x))]),
            where="post", **kw)


# Individual comparisons first, all at one weight, then the two main curves over them.
for c in FAM_PERT + FAM_CFG:
    draw(c["v"], color=MAUVE if c in FAM_PERT else WINE, lw=LW_FAINT, alpha=AL_FAINT, zorder=2,
         solid_capstyle="butt")
for pool, col in ((POOL_PERT, MAUVE), (POOL_CFG, WINE)):
    draw(pool, color=col, lw=LW_MAIN, zorder=4, solid_capstyle="butt")
for med, col in zip(MED_POOL, (WINE, MAUVE)):
    ax.plot([med, med], [0.5 - 0.038, 0.5 + 0.038], color=col, lw=0.9, zorder=5, solid_capstyle="butt")

ax.set_xticks([0, 10, 20, 30])
ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f"))
ax.tick_params(length=1.5, width=0.5, color=C_TICK, labelcolor=INK, pad=1.3, labelsize=FONT_PT["tick"])
for side in ("top", "right", "left", "bottom"):         # the frame is back, in black: a boxed panel
    ax.spines[side].set_visible(True)
    ax.spines[side].set_color(C_FRAME)
    ax.spines[side].set_linewidth(0.6)


texts = []
# ONE legend inside the frame, in the empty lower triangle. Each entry's sample is a STEP, not a ruled line:
# the curves are staircases, so a straight swatch describes something the panel never draws. A custom handler
# builds the stepped sample inside the legend's handlebox.
class _StepHandle(mpl.legend_handler.HandlerBase):
    def create_artists(self, legend, orig, xdescent, ydescent, width, height, fontsize, trans):
        x = np.array([0.0, 0.34, 0.34, 0.67, 0.67, 1.0]) * width - xdescent
        y = np.array([0.18, 0.18, 0.52, 0.52, 0.86, 0.86]) * height - ydescent
        ln = mpl.lines.Line2D(x, y, color=orig.get_color(), lw=orig.get_linewidth(),
                              solid_capstyle="butt")
        ln.set_transform(trans)
        return [ln]


leg = ax.legend(handles=[mpl.lines.Line2D([], [], color=WINE, lw=LW_MAIN,
                                          label="configuration changed   \u03c1 0.55\u20130.76"),
                         mpl.lines.Line2D([], [], color=MAUVE, lw=LW_MAIN,
                                          label="perturbation changed   \u03c1 -0.28\u2013+0.35")],
                loc="lower right", bbox_to_anchor=(0.995, 0.045), frameon=False, fontsize=FS_NOTE,
                handlelength=1.9, handletextpad=0.5, labelspacing=0.35, borderaxespad=0.0,
                handler_map={mpl.lines.Line2D: _StepHandle()})
LEG_TEXTS = list(leg.get_texts())
texts.extend(LEG_TEXTS)
# The axis title is symbolic, like panel d's "Delta normalised RMSE": |Delta rank| says the quantity and the
# pair of bars says it is taken in magnitude, which is the whole reason this panel ranks |delta| at all.
texts.append(fig.text((X0 + (X1 - X0) / 2.0) / W_MM, 1.0 - (YB + 5.0) / H_MM,
                      "|\u0394rank| (positions)", fontsize=FONT_PT["axes_label"], color=INK,
                      ha="center", va="center"))
texts.append(fig.text(3.0 / W_MM, 1.0 - (YT + YB) / 2.0 / H_MM, "cumulative share",
                      fontsize=FONT_PT["axes_label"], color=INK, ha="center", va="center", rotation=90))

# ------------------------------------------------------------------ audits
fig.canvas.draw()
rend = fig.canvas.get_renderer()
DPI_MM = fig.get_dpi() / 25.4


def line_boxes(o):
    """Per-LINE boxes: a multi-line label's own box is the union of its lines, which would report collisions
    between lines on different rows and hide one between line 1 of a label and line 2 of another."""
    s = o.get_text()
    bb = o.get_window_extent(renderer=rend)
    parts = s.split("\n")
    if len(parts) == 1:
        return [(s, bb)]
    h = bb.height / len(parts)
    out = []
    for i, ln in enumerate(parts):
        try:
            w = rend.get_text_width_height_descent(ln, o.get_fontproperties(), False)[0]
        except Exception:
            w = bb.width
        out.append((ln, mpl.transforms.Bbox.from_bounds(bb.x0 + (bb.width - w) / 2.0,
                                                        bb.y1 - (i + 1) * h, w, h)))
    return out


objs = []
for _o in texts:
    objs.extend(_o.get_texts() if hasattr(_o, "get_texts") else [_o])
for _ax in fig.axes:
    for _o in list(_ax.get_xticklabels()) + list(_ax.get_yticklabels()):
        if _o.get_text():
            objs.append(_o)
recs = []
for _o in objs:
    for _ln, _bb in line_boxes(_o):
        recs.append(("%s%s" % (_o.get_text(), "" if _ln == _o.get_text() else " / " + _ln), _bb))

for name, bb in recs:
    x0, y0 = bb.x0 / DPI_MM, bb.y0 / DPI_MM
    w, h = bb.width / DPI_MM, bb.height / DPI_MM
    assert -0.02 <= x0 and x0 + w <= W_MM + 0.02, "text %r leaves the canvas sideways" % name
    assert -0.02 <= y0 and y0 + h <= H_MM + 0.02, "text %r leaves the canvas" % name
for i in range(len(recs)):
    for j in range(i + 1, len(recs)):
        a, b = recs[i][1], recs[j][1]
        ox = (min(a.x1, b.x1) - max(a.x0, b.x0)) / DPI_MM
        oy = (min(a.y1, b.y1) - max(a.y0, b.y0)) / DPI_MM
        if ox > 0.3 and oy > 0.3:
            raise SystemExit("text %r overlaps %r by %.2f x %.2f mm" % (recs[i][0], recs[j][0], ox, oy))
assert min(FONT_PT["tick"], FONT_PT["axes_label"], FS_NOTE, FS_LETTER) >= 7.0

# The annotation budget, asserted: all text INSIDE the frame is the legend's two entries, and it must stay
# within ANN_BUDGET characters. Consolidating the four former labels (two titles above the frame, 35
# characters, plus two statistic lines inside it, 39) into one in-plot key leaves two labels and 86 characters;
# the element count is what halved. Anything that pushes past the budget stops the build instead of quietly
# turning the panel back into an infographic.
in_frame = list(LEG_TEXTS)
n_chars = sum(len(t.get_text()) for t in in_frame)
assert n_chars <= ANN_BUDGET, "in-frame annotation is %d characters, budget %d" % (n_chars, ANN_BUDGET)
assert len(in_frame) == 2, "expected exactly two legend entries, found %d" % len(in_frame)
# Every word must be readable on the ground: contrast ratio >= 4.5:1 at 7 pt. The legend's labels are black, so
# this passes trivially today; it stays because a coloured label is the obvious next edit.
for _t in in_frame:
    _l = lum(mpl.colors.to_hex(_t.get_color()))
    _ratio = (max(_l, lum(GROUND)) + 0.05) / (min(_l, lum(GROUND)) + 0.05)
    assert _ratio >= 4.5, "%r is only %.2f:1 against the ground" % (_t.get_text(), _ratio)
# The form the panel is required to have, asserted because a future edit that changed it would be a
# regression rather than a variant: 16 uniformly faint individual staircases, 2 dominant main curves, the
# halfway rule and 2 median markers -- and NO filled areas at all.
assert len(ax.lines) == 1 + 2 + 2 + 16, "expected rule + 2 main + 2 markers + 16 faint, found %d" % len(ax.lines)
assert len(ax.collections) == 0, "the panel must have no filled areas, found %d" % len(ax.collections)
assert all(s.get_visible() for s in ax.spines.values()), "the frame must be drawn on all four sides"
for ln in ax.lines:
    if ln.get_alpha() is not None and ln.get_alpha() < 0.5:
        assert ln.get_linewidth() <= 0.35, "a faint comparison is %.2f pt wide" % ln.get_linewidth()
        assert ln.get_alpha() <= 0.30, "a faint comparison is at alpha %.2f" % ln.get_alpha()
    elif ln.get_linewidth() >= 0.9:
        assert ln.get_alpha() in (None, 1.0), "a main curve is not opaque"
        assert ln.get_linewidth() >= LW_FAINT * 3, "a main curve is not dominant over the faint ones"
# No annotation may be crossed by either MAIN curve (the faint ones are texture and are allowed under text).
for t in texts:
    if t.get_text() in ("|\u0394rank| (positions)", "cumulative share", "f"):
        continue
    bb = t.get_window_extent(renderer=rend)
    (xl, ylo), (xr, yhi) = ax.transData.inverted().transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
    for pool in (POOL_CFG, POOL_PERT):
        s = np.sort(pool)
        lo = np.searchsorted(s, xl, side="right") / len(s)
        hi = np.searchsorted(s, xr, side="right") / len(s)
        assert hi <= ylo + 0.004 or lo >= yhi - 0.004, \
            "%r is crossed by a main curve (%.2f..%.2f against %.2f..%.2f)" % (t.get_text(), lo, hi, ylo, yhi)
# ... and the legend as a whole, handles included, must sit clear of both main curves.
_bb = leg.get_window_extent(renderer=rend)
(_xl, _ylo), (_xr, _yhi) = ax.transData.inverted().transform([(_bb.x0, _bb.y0), (_bb.x1, _bb.y1)])
for pool in (POOL_CFG, POOL_PERT):
    s = np.sort(pool)
    lo = np.searchsorted(s, _xl, side="right") / len(s)
    hi = np.searchsorted(s, _xr, side="right") / len(s)
    assert hi <= _ylo + 0.004 or lo >= _yhi - 0.004, \
        "the legend box is crossed by a main curve (%.2f..%.2f against %.2f..%.2f)" % (lo, hi, _ylo, _yhi)
print("%d faint comparisons + %d main curves, framed, no filled areas; in-frame annotation %d characters "
      "(budget %d) in %d legend entries occupying %.1f x %.1f mm, %d text lines audited; pooled medians %d "
      "and %d are marked on the 0.50 line"
      % (16, 2, n_chars, ANN_BUDGET, len(in_frame),
         leg.get_window_extent(renderer=rend).width / DPI_MM,
         leg.get_window_extent(renderer=rend).height / DPI_MM, len(recs), MED_POOL[0], MED_POOL[1]))

fig.savefig(OUT / "fig3f.pdf", facecolor=GROUND)
fig.savefig(OUT / "fig3f.svg", facecolor=GROUND)
fig.savefig(OUT / "fig3f.png", dpi=600, facecolor=GROUND)
from PIL import Image
im = Image.open(OUT / "fig3f.png")
got = (im.width / (600 / 25.4), im.height / (600 / 25.4))
assert abs(got[0] - W_MM) < 0.05 and abs(got[1] - H_MM) < 0.05, "PNG is %s mm" % (got,)
print("wrote fig3f.{pdf,svg,png}  %.0f x %.0f mm  PNG %d x %d px" % (W_MM, H_MM, im.width, im.height))
