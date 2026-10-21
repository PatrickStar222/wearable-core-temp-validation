"""fig2g_earcanal_build.py -- Figure 2 panel g: how tightly an ear-canal channel tracks core depends
on the ambient. (Built first as "fig3e" for a six-panel experiment sheet; the content is Fig 2's, and the
filenames were changed so the artifact cannot be mistaken for that sheet's panel e.)

THE CLAIM (manuscript Result 1, verbatim numbers; the manuscript's Figure 2 legend carries this panel)
    The within-participant, within-phase correlation between each ear-canal channel and esophageal core,
    computed separately at 10, 20 and 30 C ambient. Infrared falls from +0.459 at 30 C to +0.335 at 20 C
    and +0.135 at 10 C; the internal thermister follows (+0.394, +0.321, +0.137); the external thermister
    crosses zero (+0.223, +0.064, -0.107). At the cool end the share of sessions with a positive
    correlation falls to 47 % (infrared) and 39 % (external), against 75 % and 61 % at 30 C.

OWNERSHIP (added after the fact, and the reason the files were renamed)
    This content is Figure 2's: the seven-panel blueprint puts it at g, `fig2_layout.py` reserves that
    slot, and the manuscript's Figure 2 legend already carries the claim and the numbers. It was built as
    "fig3e" from an experiment-sheet plan whose panels e and f are both content owned elsewhere (f is the
    manuscript's Figure 5c). The panel is sound; the letter and the figure it belongs to were not.
    The sheet's other panels show which PROSPIE configuration survives a stress. This one is the external
    cohort and a different question: the relation between a wearable channel and core is not a fixed
    property of the channel -- it moves with the environment, and for one channel it changes sign.

THE UNIT AND THE ESTIMAND, because two of them exist in this project and they must not be mixed
    Unit = participant x within-session phase x ambient; value = Pearson r between that unit's channel and
    T_es over the unit's samples (len > 2, both s.d. > 0). Frozen means live in
    COOL_END_VALIDATION_RESULTS.md section 3.1 and are asserted here to 2e-3.
    The pooled OLS slope of T_es on the channel per ambient (infrared 0.322 / 0.647 / 0.749, in
    `_fig2_slope_frozen.json`) is a DIFFERENT estimand and belongs to a different panel; nothing in this
    script mixes the two, and the panel states "correlation" on its axis.

WHAT THE PANEL SHOWS
    Nine cells: three channels (rows) x three ambients (columns), one shared r axis, each cell carrying
      * every unit as a dot -- a deterministic beeswarm, so no unit can hide behind another;
      * the cell mean with its 95 % confidence interval, in a lane of its own above the dots, separated
        from them by a hairline that is asserted to contain no dot;
      * n and the share of units above zero, as one small line per cell;
    and a zero rule through every cell, because "crosses zero" is half the claim.

READING WARNING THAT BELONGS IN THE CAPTION, NOT IN THE PANEL
    The nine unit distributions are wide (r runs from about -1 to +1 in every cell), so the mean and the
    median disagree sharply -- infrared at 30 C: mean +0.459, median +0.801; external at 10 C: mean
    -0.107, median -0.307. The mean is what the manuscript quotes and what this panel draws; the dots are
    what stops a reader from taking the mean for the whole story.

COLOUR
    Ambient is an ordered variable, so the three columns carry a ramp ordered by LIGHTNESS: light teal
    #7FB0BF at 10 C, amber #C08A1E at 20 C, deep brick #8C2F1E at 30 C (L* 69.0 / 61.2 / 33.4). The trio
    the panel inherited (teal / amber / vermillion) failed the project's own check -- amber and vermillion
    are 27.8 apart under deuteranopia against a threshold of 40, and its greyscale order did not follow
    temperature. A build-time gate now refuses any edit that breaks the separation or the ordering. The
    hues also stay clear of the oxblood/terracotta that carries the PROSPIE configurations in panels b-d.

Run:  python fig2g_earcanal_build.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.stdout.reconfigure(encoding="ascii", errors="replace")
sys.path.insert(0, "")
from figure_style import apply_style, mm_fig, FONT_PT   # noqa: E402

FS_NOTE = 7.0      # per-cell header: n and the share above zero
FS_HEAD = 8.0      # column headers, the three ambients
FS_LETTER = 8.0    # the sheet draws every panel letter in its own canvas

apply_style(journal="nature", language="en")
mpl.rcParams["savefig.bbox"] = "standard"      # figure_style crops to ink; this panel is written whole

P = Path("")
OUT = P / "fig3_v2"

# ------------------------------------------------------------------ frozen data
# The means come from COOL_END_VALIDATION_RESULTS.md section 3.1; the shares and the channel names are the
# manuscript's own wording for this panel (its Figure 2d legend, which also carries the two caveats the
# caption must keep: the panel is descriptive only, and the ambient order was fixed across days so the
# environmental effect is not separable from session position).
UNITS = json.load(open(P / "_fig2_fujii_units.json"))
FROZEN = {"Infrared": {10: 0.135, 20: 0.335, 30: 0.459},
          "Internal": {10: 0.137, 20: 0.321, 30: 0.394},
          "External": {10: -0.107, 20: 0.064, 30: 0.223}}
SHARE = {"Infrared": {10: 47, 20: 74, 30: 75},
         "Internal": {10: 56, 20: 71, 30: 75},
         "External": {10: 39, 20: 62, 30: 61}}
ROW_LABEL = {"Infrared": "infrared", "Internal": "internal", "External": "external"}
CHANNELS = ("Infrared", "Internal", "External")
AMB = (10, 20, 30)
COL = {10: "#7FB0BF", 20: "#C08A1E", 30: "#8C2F1E"}
INK, C_RULE, C_FRAME = "#1F2A37", "#8C7A6A", "#B9B2A8"

# ------------------------------------------------------------------ colour gate
# The trio this panel inherited (teal #2F7C94 / amber #C08A1E / vermillion #D2542A) failed at the first
# check: under deuteranopia amber and vermillion are 27.8 apart (threshold 40), and its greyscale
# lightness was not monotone in temperature (L* 48.5 / 61.2 / 52.1). Ambient is an ORDERED variable, so
# the ramp is now ordered by lightness and the gate below refuses any future edit that breaks it. The
# matrices are the project's own CVD checker's (_cvd_d.py), so these numbers are comparable with the
# separation already recorded for panels b-d.
def _lin(h):
    c = np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], float) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

_M = {"normal": np.eye(3),
      "protan": np.array([[0.11238, 0.88762, 0.0], [0.11238, 0.88762, 0.0],
                          [0.00401, -0.00401, 1.0]]),
      "deutan": np.array([[0.29275, 0.70725, 0.0], [0.29275, 0.70725, 0.0],
                          [-0.02234, 0.02234, 1.0]]),
      "tritan": np.array([[1.0, 0.14461, -0.14461], [0.0, 0.85924, 0.14076],
                          [0.0, 0.85924, 0.14076]])}

def _vis(h, k):
    v = np.clip(_M[k] @ _lin(h), 0, 1)
    return np.clip(np.where(v <= 0.0031308, v * 12.92, 1.055 * v ** (1 / 2.4) - 0.055), 0, 1)

def _dist(a, b, k):
    return float(np.linalg.norm(_vis(a, k) - _vis(b, k))) * 255.0

def _L(h):
    r, g, b = _lin(h)
    return 116.0 * (0.2126 * r + 0.7152 * g + 0.0722 * b) ** (1 / 3) - 16.0

_pairs = [_dist(COL[a], COL[b], k) for a, b in ((10, 20), (10, 30), (20, 30)) for k in _M]
_ls = [_L(COL[a]) for a in AMB]
_page = min(_dist(COL[a], "#FFFFFF", k) for a in AMB for k in _M)
assert min(_pairs) >= 40.0, "ambient colours are %.1f apart under some vision (threshold 40)" % min(_pairs)
assert _ls[0] > _ls[1] > _ls[2], "the ambient ramp is not monotone in lightness: %s" % np.round(_ls, 1)
assert _page >= 60.0, "an ambient colour is only %.1f from the page" % _page

V, MEAN, CI, SH = {}, {}, {}, {}
for ch in CHANNELS:
    for a in AMB:
        v = np.array(UNITS[ch][str(a)], float)
        V[(ch, a)] = v
        MEAN[(ch, a)] = float(v.mean())
        CI[(ch, a)] = float(1.96 * v.std(ddof=1) / np.sqrt(len(v)))
        SH[(ch, a)] = 100.0 * float((v > 0).mean())
        assert abs(MEAN[(ch, a)] - FROZEN[ch][a]) < 2e-3, \
            "%s %d C: recomputed mean %.4f vs frozen %.3f" % (ch, a, MEAN[(ch, a)], FROZEN[ch][a])
        assert abs(SH[(ch, a)] - SHARE[ch][a]) < 0.6, \
            "%s %d C: share above zero %.1f %% vs manuscript %d %%" % (ch, a, SH[(ch, a)], SHARE[ch][a])
        assert len(v) >= 30, "%s %d C has only %d units" % (ch, a, len(v))
N_UNITS = sum(len(V[k]) for k in V)
# The external channel is the one that crosses zero, and the cool end is where it does it: if a future
# edit of the unit table broke that, the panel's headline would be gone while the page still looked fine.
assert MEAN[("External", 10)] < 0 < MEAN[("External", 20)], "the external channel no longer crosses zero"
assert MEAN[("Infrared", 10)] < MEAN[("Infrared", 20)] < MEAN[("Infrared", 30)], \
    "the infrared coupling no longer rises with ambient"

# ------------------------------------------------------------------ layout (millimetres)
W_MM, H_MM = 89.0, 61.5
XL_L, XR = 15.0, 88.0                  # row-label gutter, then the three cells to the canvas edge
GUT_TOP = 6.6                          # column headers live here
JOINT = 1.5
CELL_W = (XR - XL_L - 2 * JOINT) / 3.0
HEAD_H, SUM_H, DOT_H, PAD_C = 2.6, 4.2, 6.5, 1.2
CELL_H = HEAD_H + SUM_H + DOT_H + PAD_C
GRID_TOP = GUT_TOP
GRID_BOT = GRID_TOP + 3 * CELL_H + 2 * JOINT
assert GRID_BOT + 8.0 <= H_MM, "only %.1f mm left under the grid" % (H_MM - GRID_BOT)
R_LO, R_HI = -1.05, 1.05
DOT_MM, GAP = 0.45, 1.06               # dot diameter and the centre-to-centre gate
# Measured, not guessed: the densest cell (30 C, where units pile up near r = +0.98) needs the dot
# centres to span 5.83 mm plus one dot diameter, i.e. a 6.38 mm band, at this dot size -- and 11.70 mm at
# 0.92 mm. 0.55 mm is the largest dot that keeps the beeswarm inside a 6.5 mm band, so the "no unit hides
# behind another" property survives the 89 mm column width.

def cx(j):
    return XL_L + j * (CELL_W + JOINT)

def cy(i):
    return GRID_TOP + i * (CELL_H + JOINT)

def rx(r, j):
    """r in [-1,1] -> figure x in mm for cell column j."""
    return cx(j) + (np.asarray(r) - R_LO) / (R_HI - R_LO) * CELL_W

# ------------------------------------------------------------------ beeswarm
def beeswarm(vals, j, band_h):
    """Deterministic placement: left to right, nearest free lane, no unit hidden behind another.

    Returns (dx_mm offsets from the lane centre, n_lanes). Raises if a unit cannot be placed inside the
    band -- a build failure is the correct outcome, because an overlapping dot is a false statement about
    how many units there were.
    """
    xs = rx(vals, j)
    s = DOT_MM * GAP
    placed = []
    for x in xs:
        half = band_h / 2.0 - DOT_MM / 2.0
        k = 0
        while True:
            for sign in ((0,) if k == 0 else (1, -1)):
                dy = sign * k * s
                if abs(dy) > half + 1e-9:
                    continue
                if all(abs(x - px) >= DOT_MM or abs(dy - pdy) >= DOT_MM * GAP for px, pdy in placed):
                    placed.append((x, dy))
                    break
            else:
                k += 1
                if k * s > half + 1e-9:
                    raise SystemExit("cell column %d: no free lane for a unit inside %.1f mm" % (j, band_h))
                continue
            break
    placed = np.array(placed)
    d = np.sqrt(((placed[:, None, :] - placed[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    assert d.min() >= DOT_MM - 1e-9, "two units are %.3f mm apart (dot %.2f mm)" % (d.min(), DOT_MM)
    return placed[:, 1], (int(np.ceil(np.ptp(placed[:, 1]) / s)) + 1 if len(placed) > 1 else 1)

# ------------------------------------------------------------------ draw
fig = plt.figure(figsize=mm_fig(W_MM, H_MM))
texts, dots = [], []
lanes = {}
for i, ch in enumerate(CHANNELS):
    for j, a in enumerate(AMB):
        ax = fig.add_axes([cx(j) / W_MM, 1.0 - (cy(i) + CELL_H) / H_MM, CELL_W / W_MM, CELL_H / H_MM])
        ax.set_xlim(R_LO, R_HI)
        ax.set_ylim(CELL_H, 0.0)             # inverted: y = 0 at the TOP of the cell, mm downwards
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(C_FRAME)
            ax.spines[side].set_linewidth(0.5)
        ax.axvline(0.0, color=C_RULE, lw=0.6, zorder=1)
        # summary lane: mean and its 95 % interval, above a hairline that separates it from the dots
        y_sum = HEAD_H + SUM_H / 2.0
        y_dot0 = HEAD_H + SUM_H
        ax.plot([R_LO, R_HI], [y_dot0, y_dot0], color=C_FRAME, lw=0.5, zorder=2, solid_capstyle="butt")
        m, ci = MEAN[(ch, a)], CI[(ch, a)]
        ax.plot([m - ci, m + ci], [y_sum, y_sum], color=COL[a], lw=0.7, zorder=5, solid_capstyle="butt")
        for e in (m - ci, m + ci):
            ax.plot([e, e], [y_sum - 0.55, y_sum + 0.55], color=COL[a], lw=0.7, zorder=5,
                    solid_capstyle="butt")
        ax.plot([m], [y_sum], marker="o", ms=4.2, mfc=COL[a], mec="white", mew=0.35, ls="none", zorder=6)
        # the units
        dys, n_lanes = beeswarm(V[(ch, a)], j, DOT_H)
        y_c = y_dot0 + DOT_H / 2.0
        for r, dy in zip(V[(ch, a)], dys):
            # Plotted at the unit's r in DATA units. rx() is the canvas-millimetre mapping used for the
            # placement and the audits; plotting with it put every dot outside the axes and all 318 were
            # clipped away -- the summary markers were the only marks left, which is how it was caught.
            ax.plot([float(r)], [y_c + dy], marker="o", ms=1.28, mfc=COL[a], mec="white", mew=0.15,
                    ls="none", alpha=0.75, zorder=4)
            dots.append((float(rx(r, j)), cy(i) + y_c + dy))
        lanes[(ch, a)] = n_lanes
        # header: n and the share above zero
        t = ax.text(R_LO + 0.02, HEAD_H * 0.55, "n = %d \u00b7 %d %% > 0" % (len(V[(ch, a)]), round(SH[(ch, a)])),
                    fontsize=FS_NOTE, color=INK, ha="left", va="center", zorder=7)
        texts.append(t)
        if i == len(CHANNELS) - 1:
            # Three major ticks, not five: at 7 pt the "-1" and "-0.5" labels ended up 0.22 mm apart and
            # read as one string. The axis needs its bounds and its zero -- the semantic anchor -- and
            # nothing else; values are read from the dots, not from the ticks.
            ax.set_xticks([-1.0, 0.0, 1.0])
            labs = ax.set_xticklabels(["\u22121", "0", "+1"], fontsize=FONT_PT["tick"])
            # End labels are tucked INSIDE their cell: centred, "+1" of one cell and "-1" of the next were
            # 0.01 mm apart -- the same fix panel a needed.
            labs[0].set_ha("left")
            labs[-1].set_ha("right")
            ax.tick_params(axis="x", length=2.2, pad=1.2)
        else:
            ax.set_xticks([])
        ax.set_yticks([])

# column headers, in the column's own colour
for j, a in enumerate(AMB):
    t = fig.text((cx(j) + CELL_W / 2.0) / W_MM, 1.0 - 2.6 / H_MM, "%d \u00b0C" % a, fontsize=FS_HEAD,
                 fontweight="bold", color=COL[a], ha="center", va="center")
    texts.append(t)
# row labels
for i, ch in enumerate(CHANNELS):
    t = fig.text((XL_L - 2.2) / W_MM, 1.0 - (cy(i) + CELL_H / 2.0) / H_MM, ROW_LABEL[ch],
                 fontsize=FONT_PT["tick"], color=INK, ha="right", va="center")
    texts.append(t)
# shared axis title and the panel letter
t = fig.text(((XL_L + XR) / 2.0) / W_MM, 1.0 - (GRID_BOT + 5.4) / H_MM,
             "correlation with esophageal core, r", fontsize=FONT_PT["axes_label"], color=INK,
             ha="center", va="center")
texts.append(t)
t = fig.text(2.0 / W_MM, 1.0 - 1.6 / H_MM, "e", fontsize=FS_LETTER, fontweight="bold", color=INK,
             ha="left", va="top")
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
assert min(FONT_PT["tick"], FONT_PT["axes_label"], FS_NOTE, FS_HEAD, FS_LETTER) >= 7.0

# The summary lane must contain no unit: that is the whole reason it exists.
for i, ch in enumerate(CHANNELS):
    for j, a in enumerate(AMB):
        y0 = cy(i) + HEAD_H + 0.1
        y1 = cy(i) + HEAD_H + SUM_H - 0.1
        inside = [(x, y) for (x, y) in dots if cx(j) - 0.5 < x < cx(j) + CELL_W + 0.5 and y0 < y < y1]
        assert not inside, "%s %d C: %d unit dot(s) in the summary lane" % (ch, a, len(inside))
print("summary lanes hold no unit dot in any of the 9 cells (checked against all %d dots)" % len(dots))

# Every dot inside its own cell, and every header inside its own cell's width: the y axis of a cell is a
# layout axis, so a sign error there moves marks into the neighbouring cell without touching any datum.
for (x, y) in dots:
    j = min(2, max(0, int((x - XL_L) // (CELL_W + JOINT))))
    i = min(2, max(0, int((y - GRID_TOP) // (CELL_H + JOINT))))
    assert cx(j) - 0.02 <= x <= cx(j) + CELL_W + 0.02, "a dot sits outside the cell columns at x=%.2f" % x
    assert cy(i) - 0.02 <= y <= cy(i) + CELL_H + 0.02, "a dot sits outside the cell rows at y=%.2f" % y
for t, bb in zip(texts, boxes):
    if not label_of(t).startswith("n ="):
        continue
    x0, x1 = bb.x0 / DPI_MM, bb.x1 / DPI_MM
    j = min(2, max(0, int((x0 - XL_L) // (CELL_W + JOINT))))
    assert x0 >= cx(j) - 0.02 and x1 <= cx(j) + CELL_W + 0.02, \
        "a cell header spans %.2f..%.2f mm, outside its %.2f mm cell" % (x0, x1, CELL_W)
print("every unit dot lies inside its own cell, and every header fits inside its %.1f mm cell" % CELL_W)

# Every x tick label inside its own cell as well: with 1.5 mm joints, a centred end label crosses into
# the neighbouring cell and two numbers fuse into one string.
for k, ax in enumerate(fig.axes):
    if not ax.get_xticklabels():
        continue
    j = k % 3
    for t in ax.get_xticklabels():
        bb = t.get_window_extent(renderer=rend)
        x0, x1 = bb.x0 / DPI_MM, bb.x1 / DPI_MM
        assert cx(j) - 0.02 <= x0 and x1 <= cx(j) + CELL_W + 0.02, \
            "tick label %r spans %.2f..%.2f mm, outside cell %d (%.2f..%.2f)" % (
                t.get_text(), x0, x1, j, cx(j), cx(j) + CELL_W)
print("every x tick label stays inside its own cell (the end labels are tucked in, not centred)")

# Margins, measured rather than eyeballed: the room the end tick labels leave at the cell edges, and how
# far the lowest dot of any cell sits above that cell's bottom edge.
_edge = {}
for k, ax in enumerate(fig.axes):
    if not ax.get_xticklabels():
        continue
    j = k % 3
    bs = [t.get_window_extent(renderer=rend) for t in ax.get_xticklabels()]
    _edge[j] = (min(b.x0 for b in bs) / DPI_MM - cx(j), cx(j) + CELL_W - max(b.x1 for b in bs) / DPI_MM)
print("end tick labels leave %.2f mm at the cell's left edge and %.2f mm at its right edge"
      % (min(v[0] for v in _edge.values()), min(v[1] for v in _edge.values())))
_clear = []
for i in range(3):
    for j in range(3):
        ys = [y for (x, y) in dots if cx(j) - 0.5 < x < cx(j) + CELL_W + 0.5
              and cy(i) - 0.5 < y < cy(i) + CELL_H + 0.5]
        _clear.append(cy(i) + CELL_H - (max(ys) + DOT_MM / 2.0))
print("lowest dot in any cell sits %.2f mm above that cell's bottom edge (dot radius %.2f mm)"
      % (min(_clear), DOT_MM / 2.0))

# Neighbouring tick labels INSIDE a cell must be separable too, not only inside the cell boundary.
_gaps = []
for k, ax in enumerate(fig.axes):
    if not ax.get_xticklabels():
        continue
    bs = sorted((t.get_window_extent(renderer=rend) for t in ax.get_xticklabels()), key=lambda b: b.x0)
    _gaps += [(bs[i + 1].x0 - bs[i].x1) / DPI_MM for i in range(len(bs) - 1)]
print("smallest gap between neighbouring tick labels inside a cell: %.2f mm" % min(_gaps))
assert min(_gaps) >= 0.6, "tick labels are only %.2f mm apart" % min(_gaps)

print("cells: %d units total; beeswarm lanes per cell: %s"
      % (N_UNITS, ", ".join("%s/%d=%d" % (c[:4], a, lanes[(c, a)]) for c in CHANNELS for a in AMB)))
for ch in CHANNELS:
    print("  %-9s %s" % (ch, "  ".join("%2d C: r=%+.3f CI+-%.3f (%d %% > 0, n=%d)"
                                       % (a, MEAN[(ch, a)], CI[(ch, a)], round(SH[(ch, a)]),
                                          len(V[(ch, a)])) for a in AMB)))
print("crossing zero: external %.3f at 10 C -> %+.3f at 20 C; infrared rises %.3f -> %.3f"
      % (MEAN[("External", 10)], MEAN[("External", 20)], MEAN[("Infrared", 10)], MEAN[("Infrared", 30)]))

# write the three formats whole-canvas
fig.savefig(OUT / "fig2g_earcanal_coupling.pdf", facecolor="white")
fig.savefig(OUT / "fig2g_earcanal_coupling.svg", facecolor="white")
fig.savefig(OUT / "fig2g_earcanal_coupling.png", dpi=600, facecolor="white")
from PIL import Image
im = Image.open(OUT / "fig2g_earcanal_coupling.png")
got = (im.width / (600 / 25.4), im.height / (600 / 25.4))
assert abs(got[0] - W_MM) < 0.05 and abs(got[1] - H_MM) < 0.05, "PNG is %s mm" % (got,)
print("wrote fig2g_earcanal_coupling.{pdf,svg,png}  %.0f x %.0f mm  PNG %d x %d px" % (W_MM, H_MM, im.width, im.height))
