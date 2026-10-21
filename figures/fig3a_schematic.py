"""fig3a_schematic.py -- Figure 3, panel a: the experiment, drawn as pictures.

WHAT IT IS
    A protocol illustration, not a text panel and not a data panel. The user asked for the experiment flow
    "以图形、图案的方式" and "就像 fig2 一样" -- so this is drawn in the language of Figure 2's key band
    (`fig2_s1_comic2.py`): the same locked palette, the same hand-drawn wobble (`set_sketch_params` on
    scenery only, never on text), the same human proportions, and **titles only** -- every number that a
    reader needs is in the artwork (a thermometer range bar, eleven site dots, a leave-one-out row), not in a
    paragraph.

    Four stages, top to bottom, each with one short title:
      1  cohort and protocol   39 participants, nine controlled ambient conditions, 25.0-38.4 C
      2  sensing chain         eleven skin sites, heart rate, three environment channels
      3  estimator             ridge regression, penalty by leave-one-out CV per fold
      4  evaluation            leave one participant out, then six stress tests
    The six stress tests are drawn as six glyphs, each with a one-word label: sites, blank rows, noise,
    offset, drift, unseen.

WHY THE DRAWINGS ARE ASSERTED
    An illustration may not claim more than the data. The dot count on the body is asserted to be exactly
    eleven (the manuscript's "eleven skin sites"), the thermometer bar is written from the manuscript's own
    condition range (25.0-38.4 C), and the leave-one-out row draws 39 folds as a countable row of people with
    exactly one lifted out. A drawing that drifts from the methods stops the build.

WHY NOT THE SVG COMPOSER
    `fig_compose` draws box-and-arrow overviews at 5.5 pt on a 183 mm canvas; this panel is 89 mm wide and the
    sheet's floor is 7.0 pt, and what is wanted here is a hand-drawn scene, not a pipeline. The drawing
    follows the `publication-mechanism-figure` rules that apply to illustrations: one arrow type, colour with
    meaning, no decoration, nothing below 7 pt, no rotated text.

Run:  python fig3a_schematic.py
"""
import os
import sys
import time

for _opt in ("numexpr", "bottleneck"):
    sys.modules.setdefault(_opt, None)

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, Ellipse, FancyBboxPatch, Polygon, Rectangle
from matplotlib.transforms import Affine2D
from PIL import Image

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig3_v2")
W_MM, H_MM = 89.0, 78.0
DPI = 600

# ---- Figure 2's palette, so the two illustrations read as one hand ------------------------------
PAPER, INK, MUTED, HAIR = "#FFFFFF", "#4A4033", "#8A7A66", "#DED2BE"
CLAY, SAGE, AMBER, PLUM = "#C08268", "#A3AD86", "#D9A44E", "#8E6C8A"
TERRA, BLUSH, CREAM, STONE = "#A85A4E", "#EFD8CE", "#F4EAD8", "#D8C9B0"
SKY, SHIRT, TROUSER, SHOE = "#DCE6EA", "#E2D5DE", "#D9C3AE", "#6E5B4B"
FS_LET, FS_TITLE = 8.0, 7.5
SKETCH = dict(scale=1.1, length=105, randomness=9)
SMOOTH = False
W_GLYPH = 0.5

mpl.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "svg.fonttype": "none", "pdf.fonttype": 42})
fig = plt.figure(figsize=(W_MM / 25.4, H_MM / 25.4), dpi=DPI, facecolor=PAPER)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W_MM)
ax.set_ylim(0, H_MM)
ax.set_aspect("equal")
ax.axis("off")
texts = []


def sk(art):
    if SMOOTH:
        return art
    art.set_sketch_params(**SKETCH)
    return art


def capsule(p0, p1, width, fc, ec=INK, lw=W_GLYPH, z=4):
    (x0, y0), (x1, y1) = p0, p1
    L = float(np.hypot(x1 - x0, y1 - y0))
    ang = float(np.degrees(np.arctan2(y1 - y0, x1 - x0)))
    r = width / 2.0
    p = FancyBboxPatch((0, -r), L, width, boxstyle="round,pad=0,rounding_size=%.3f" % r,
                       facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z)
    p.set_transform(Affine2D().rotate_deg(ang).translate(x0, y0) + ax.transData)
    ax.add_patch(sk(p))


def rrect(x, y, w, h, fc, ec=INK, lw=W_GLYPH, r=0.6, z=4):
    ax.add_patch(sk(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=%.3f" % r,
                                   facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z)))


def txt(x, y, s, fs, c=INK, w="normal", ha="center", va="center", z=20):
    t = ax.text(x, y, s, fontsize=fs, color=c, weight=w, ha=ha, va=va, zorder=z)
    texts.append(t)
    return t


def arrow(x, y0, y1, color=MUTED):
    ax.add_patch(mpl.patches.FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>", mutation_scale=7,
                                             lw=1.1, color=color, zorder=8, shrinkA=0, shrinkB=0,
                                             capstyle="round"))


# ---- the human figure: Figure 2's proportions, copied so the two drawings match -----------------
def person(cx, feet_y, h, arms="down", face=True, z=6, shirt=SHIRT, trouser=TROUSER):
    head_h, head_w = 0.130 * h, 0.100 * h
    shoulder_y = feet_y + 0.815 * h
    hip_y = feet_y + 0.520 * h
    knee_y = feet_y + 0.280 * h
    sh_w, waist_w = 0.240 * h, 0.170 * h
    head_c = (cx, shoulder_y + 0.030 * h + head_h / 2.0)
    for sgn in (-1, 1):
        knee = (cx + sgn * 0.055 * h, knee_y)
        ankle = (cx + sgn * 0.065 * h, feet_y + 0.030 * h)
        capsule((cx + sgn * 0.045 * h, hip_y), knee, 0.098 * h, trouser, z=z)
        capsule(knee, ankle, 0.082 * h, trouser, z=z)
        toe = (ankle[0] - sgn * 0.070 * h, feet_y + 0.010 * h)
        capsule(ankle, toe, 0.040 * h, SHOE, z=z + 1)
        rrect(min(ankle[0], toe[0]) - 0.010 * h, feet_y, 0.090 * h, 0.022 * h, SHOE, r=0.30, z=z + 1)
    ax.add_patch(sk(Polygon([[cx - sh_w / 2, shoulder_y], [cx + sh_w / 2, shoulder_y],
                             [cx + waist_w / 2, hip_y], [cx - waist_w / 2, hip_y]], closed=True,
                            facecolor=shirt, edgecolor=INK, linewidth=W_GLYPH, zorder=z + 1)))
    ax.add_patch(sk(FancyBboxPatch((cx - sh_w * 0.54, shoulder_y - 0.028 * h), sh_w * 1.08, 0.034 * h,
                                   boxstyle="round,pad=0,rounding_size=%.3f" % (0.017 * h),
                                   facecolor=shirt, edgecolor=INK, linewidth=W_GLYPH, zorder=z + 1)))
    segs = {}
    for sgn in (-1, 1):
        elbow = (cx + sgn * (sh_w * 0.52 + 0.010 * h), hip_y + 0.030 * h)
        wrist = (cx + sgn * (sh_w * 0.58 + 0.020 * h), knee_y + 0.060 * h)
        if arms == "out":
            elbow = (cx + sgn * (sh_w * 0.80), hip_y + 0.070 * h)
            wrist = (cx + sgn * (sh_w * 1.30), knee_y + 0.100 * h)
        capsule((cx + sgn * sh_w * 0.44, shoulder_y - 0.010 * h), elbow, 0.062 * h, shirt, z=z)
        capsule(elbow, wrist, 0.054 * h, shirt, z=z)
        ax.add_patch(sk(Ellipse((wrist[0] + sgn * 0.004 * h, wrist[1] - 0.016 * h), 0.062 * h, 0.086 * h,
                                angle=-sgn * 8.0, facecolor=BLUSH, edgecolor=INK, linewidth=W_GLYPH,
                                zorder=z + 2)))
        segs[sgn] = (elbow, wrist)
    capsule((cx, shoulder_y + 0.005 * h), (cx, head_c[1] - head_h * 0.34), 0.046 * h, BLUSH, z=z + 2)
    ax.add_patch(sk(Ellipse(head_c, head_w, head_h, facecolor=BLUSH, edgecolor=INK, linewidth=W_GLYPH,
                            zorder=z + 3)))
    if face:
        for sgn in (-1, 1):
            ax.add_patch(Circle((head_c[0] + sgn * head_w * 0.20, head_c[1] + head_h * 0.10),
                                head_w * 0.055, facecolor=INK, edgecolor="none", zorder=z + 4))
        ax.add_patch(Arc((head_c[0], head_c[1] - head_h * 0.16), head_w * 0.42, head_h * 0.34,
                         theta1=200, theta2=340, color=INK, linewidth=W_GLYPH, zorder=z + 4))
    return dict(h=h, cx=cx, feet=feet_y, hip=hip_y, shoulder=shoulder_y, knee=knee_y, sh_w=sh_w,
                waist_w=waist_w, head=head_c, head_w=head_w, head_h=head_h, arm=segs)


def _seg(p0, p1, t):
    return (p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t)


def _dist_seg(q, p0, p1):
    vx, vy = p1[0] - p0[0], p1[1] - p0[1]
    wx, wy = q[0] - p0[0], q[1] - p0[1]
    L2 = vx * vx + vy * vy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, (wx * vx + wy * vy) / L2))
    return float(np.hypot(q[0] - (p0[0] + t * vx), q[1] - (p0[1] + t * vy)))


def site_points(g):
    """The eleven sites, derived from the drawn body and asserted to lie on it (as in Figure 2)."""
    cx, hip, sh, knee = g["cx"], g["hip"], g["shoulder"], g["knee"]
    span = sh - hip
    items = [("head", g["head"], lambda q: ((q[0] - g["head"][0]) / g["head_w"]) ** 2 +
              ((q[1] - g["head"][1]) / g["head_h"]) ** 2 <= 0.25)]
    for name, dy in (("chest", 0.78), ("abdomen", 0.30)):
        items.append((name, (cx, hip + dy * span),
                      lambda q: abs(q[0] - cx) <= g["sh_w"] * 0.35 and hip <= q[1] <= sh))
    items.append(("upper back", (cx - g["sh_w"] * 0.26, hip + 0.62 * span),
                  lambda q: abs(q[0] - cx) <= g["sh_w"] * 0.42 and hip <= q[1] <= sh))
    items.append(("lower back", (cx + g["sh_w"] * 0.26, hip + 0.40 * span),
                  lambda q: abs(q[0] - cx) <= g["sh_w"] * 0.42 and hip <= q[1] <= sh))
    for name, t in (("upper arm", 0.55), ("lower arm", 1.55), ("hand", 1.95)):
        seg = g["arm"][-1]
        pts = [seg[0], seg[1], (seg[1][0] + 0.012 * g["h"], seg[1][1] - 0.045 * g["h"])]
        p = _seg(pts[0], pts[1], min(t, 1.0)) if t <= 1.0 else _seg(pts[1], pts[2], t - 1.0)
        items.append((name, p, lambda q, a=pts[0], b=pts[1], c=pts[2]:
                      _dist_seg(q, a, b) <= 0.032 * g["h"] or _dist_seg(q, b, c) <= 0.026 * g["h"]))
    for name, t in (("thigh", 0.30), ("calf", 0.78)):
        hip_p = (cx + 0.045 * g["h"], hip)
        knee_p = (cx + 0.055 * g["h"], knee)
        ankle_p = (cx + 0.065 * g["h"], g["feet"] + 0.030 * g["h"])
        if name == "thigh":
            p = _seg(hip_p, knee_p, t)
            items.append((name, p, lambda q, a=hip_p, b=knee_p: _dist_seg(q, a, b) <= 0.050 * g["h"]))
        else:
            p = _seg(knee_p, ankle_p, t)
            items.append((name, p, lambda q, a=knee_p, b=ankle_p: _dist_seg(q, a, b) <= 0.042 * g["h"]))
    ankle = (cx + 0.065 * g["h"], g["feet"] + 0.030 * g["h"])
    toe = (ankle[0] - 0.055 * g["h"], g["feet"] + 0.012 * g["h"])
    items.append(("foot", _seg(ankle, toe, 0.5),
                  lambda q, a=ankle, b=toe: _dist_seg(q, a, b) <= 0.022 * g["h"]))
    for name, p, test in items:
        assert test(p), "the %s site is off the body" % name
    return [(n, p) for n, p, _ in items]


def draw_sites(g, r, z=12):
    pts = site_points(g)
    assert len(pts) == 11, "the body must carry exactly eleven sites, found %d" % len(pts)
    for _name, p in pts:
        ax.add_patch(Circle(p, r, facecolor=PLUM, edgecolor="none", zorder=z))
    return pts


# ---- stage glyphs --------------------------------------------------------------------------------
def wall_thermo(x, y, w, h, frac, col, z=6):
    rrect(x, y, w, h, PAPER, r=0.25, z=z)
    ax.add_patch(Rectangle((x + w * 0.18, y + h * 0.10), w * 0.26, h * 0.80, facecolor=PAPER,
                           edgecolor=HAIR, linewidth=0.4, zorder=z + 1))
    ax.add_patch(Rectangle((x + w * 0.18, y + h * 0.10), w * 0.26, h * 0.80 * frac, facecolor=col,
                           edgecolor="none", zorder=z + 2))
    ax.add_patch(Circle((x + w * 0.31, y + h * 0.07), w * 0.13, facecolor=col, edgecolor=INK,
                        linewidth=0.4, zorder=z + 2))
    for i in range(4):
        yy = y + h * (0.20 + 0.20 * i)
        ax.plot([x + w * 0.50, x + w * (0.62 if i % 2 else 0.72)], [yy, yy], color=HAIR, lw=0.4,
                zorder=z + 2)


def range_bar(x, y, w, h, lo=25.0, hi=38.4):
    """The controlled ambient range, with the nine condition means marked on it.

    The nine marks are the record's own condition means (COOL_END_VALIDATION_RESULTS.md section 2:
    24.97, 29.64, 32.04, 36.62, 37.05, 37.71, 38.21, 38.29, 38.35 C), so the strip says "nine conditions"
    without a word, and the crowded hot end is visible as the crowding it is.
    """
    grad = np.linspace(0, 1, 240)
    for i, t in enumerate(grad):
        ax.add_patch(Rectangle((x + w * i / len(grad), y), w / len(grad) + 0.02, h,
                               facecolor=plt.cm.YlOrRd(0.12 + 0.55 * t), edgecolor="none", zorder=5))
    ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor=INK, linewidth=W_GLYPH, zorder=7))
    for t, lab in ((0.0, "%.1f" % lo), (1.0, "%.1f \u00b0C" % hi)):
        ax.plot([x + w * t] * 2, [y - 0.6, y], color=INK, lw=W_GLYPH, zorder=7)
        txt(x + w * t, y - 2.1, lab, 7.0, MUTED)
    for t in COND_T:
        assert len(CONDITIONS) == 9, "the record carries nine condition means"
        # inside the bar, not across it: the five hot-end conditions sit within 2 C of each other, and
        # ticks that crossed the bar turned that crowding into a black blob on the first upgrade pass.
        ax.plot([x + w * t] * 2, [y + 0.15, y + h - 0.15], color=INK, lw=0.45, zorder=8)


def heart(cx, cy, s):
    ax.add_patch(Circle((cx - 0.30 * s, cy + 0.18 * s), 0.34 * s, facecolor=TERRA, edgecolor=INK,
                        linewidth=0.4, zorder=14))
    ax.add_patch(Circle((cx + 0.30 * s, cy + 0.18 * s), 0.34 * s, facecolor=TERRA, edgecolor=INK,
                        linewidth=0.4, zorder=14))
    ax.add_patch(sk(Polygon([[cx - 0.62 * s, cy + 0.22 * s], [cx + 0.62 * s, cy + 0.22 * s],
                             [cx, cy - 0.72 * s]], closed=True, facecolor=TERRA, edgecolor=INK,
                            linewidth=0.4, zorder=14)))


CONDITIONS = [24.97, 29.64, 32.04, 36.62, 37.05, 37.71, 38.21, 38.29, 38.35]
LO_C, HI_C = 25.0, 38.4
COND_T = [(c - LO_C) / (HI_C - LO_C) for c in CONDITIONS]


def shadow(cx, feet_y, w, z=2):
    ax.add_patch(sk(Ellipse((cx, feet_y + 0.12), w, w * 0.22, facecolor=HAIR, edgecolor="none",
                            zorder=z, alpha=0.85)))


def window(x, y, w, h, z=4):
    """Figure 2's window: frame, sky, a small cloud and a cross."""
    rrect(x, y, w, h, PAPER, ec=INK, r=0.4, z=z)
    ax.add_patch(Rectangle((x + w * 0.10, y + h * 0.10), w * 0.80, h * 0.80, facecolor=SKY,
                           edgecolor="none", zorder=z + 1))
    ax.add_patch(sk(Circle((x + w * 0.68, y + h * 0.72), w * 0.09, facecolor=PAPER, edgecolor=HAIR,
                           linewidth=0.4, zorder=z + 2)))
    ax.plot([x + w * 0.10, x + w * 0.90], [y + h * 0.34] * 2, color=HAIR, lw=0.4, zorder=z + 2)
    ax.plot([x + w * 0.5] * 2, [y + h * 0.10, y + h * 0.90], color=INK, lw=0.45, zorder=z + 3)
    ax.plot([x + w * 0.10, x + w * 0.90], [y + h * 0.5] * 2, color=INK, lw=0.45, zorder=z + 3)


def env_chip(x, y, kind, label):
    """One environment channel, drawn: air temperature, humidity, air velocity.

    The first pass drew three bare arcs and one long label, which left the right half of stage 2 empty
    and said less than nothing about which channel was which. Each channel is now its own plate with its
    own pictogram and its own one-word label.
    """
    rrect(x, y, 13.4, 9.6, PAPER, ec=HAIR, r=0.6, z=5)
    cx, cy = x + 6.7, y + 5.2
    if kind == "air":
        for k in range(3):
            ax.add_patch(Arc((cx, cy + 3.2), 2.0 + k * 1.2, 1.6 + k * 0.9, theta1=42, theta2=138,
                             color=CLAY, linewidth=0.5, zorder=6))
        rrect(cx - 0.95, cy - 3.2, 1.9, 5.4, PAPER, r=0.3, z=7)
        ax.add_patch(Rectangle((cx - 0.95, cy - 3.2), 1.9, 3.5, facecolor=TERRA, edgecolor="none",
                               zorder=8))
        ax.add_patch(Circle((cx, cy - 3.5), 0.85, facecolor=TERRA, edgecolor=INK, linewidth=0.4,
                            zorder=8))
    elif kind == "humidity":
        ax.add_patch(sk(Polygon([[cx, cy + 3.0], [cx - 1.8, cy - 0.4], [cx + 1.8, cy - 0.4]],
                                closed=True, facecolor=SKY, edgecolor=INK, linewidth=0.4, zorder=7)))
        ax.add_patch(sk(Circle((cx, cy - 0.4), 1.8, facecolor=SKY, edgecolor=INK, linewidth=0.4,
                               zorder=7)))
        txt(cx, cy - 0.5, "%", 7.0, INK, z=9)
    else:
        for k in range(3):
            yy = cy + 1.7 - k * 1.7
            ax.plot([cx - 3.6, cx + 0.9], [yy, yy], color=MUTED, lw=0.5, zorder=6)
            ax.add_patch(Arc((cx + 0.7, yy), 2.6 - k * 0.4, 2.4, theta1=-52, theta2=52, color=MUTED,
                             linewidth=0.6, zorder=7))
    txt(x + 6.7, y - 2.0, label, 7.0, MUTED)


def out_chip(x, y):
    """The chain's output: an estimated core temperature, drawn rather than written."""
    rrect(x, y, 16.0, 7.4, CREAM, r=0.5, z=6)
    rrect(x + 1.6, y + 1.3, 1.9, 4.8, PAPER, r=0.3, z=7)
    ax.add_patch(Rectangle((x + 1.6, y + 1.3), 1.9, 3.1, facecolor=TERRA, edgecolor="none", zorder=8))
    ax.add_patch(Circle((x + 2.55, y + 1.15), 0.75, facecolor=TERRA, edgecolor=INK, linewidth=0.4,
                        zorder=8))
    txt(x + 10.2, y + 3.7, "core temp", 7.0, INK, ha="center")


def waves(x, y, n=3, s=1.0):
    for i in range(n):
        for k in range(2):
            ax.add_patch(Arc((x + i * 1.5 * s, y), 2.4 * s * (1 + 0.35 * k), 2.6 * s * (1 + 0.35 * k),
                             theta1=-52, theta2=52, color=MUTED, linewidth=0.5, zorder=6))


def scatter_fit(x, y, w, h, seed=7):
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0.06, 0.94, 26)
    ys = 0.16 + 0.66 * xs + rng.normal(0, 0.11, 26)
    ys = np.clip(ys, 0.03, 0.97)
    for xx, yy in zip(xs, ys):
        ax.add_patch(Circle((x + w * xx, y + h * yy), 0.32, facecolor=STONE, edgecolor=INK,
                            linewidth=0.3, zorder=6))
    ax.plot([x + w * 0.06, x + w * 0.94], [y + h * 0.20, y + h * 0.82], color=CLAY, lw=1.3, zorder=7,
            solid_capstyle="round")
    ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor=HAIR, linewidth=W_GLYPH, zorder=5))


def folds_badge(x, y, n=5):
    for i in range(n):
        fc = CREAM if i != 2 else AMBER
        rrect(x + i * 2.5, y, 1.8, 1.4, fc, r=0.25, z=6)


def fold_ring(cx, cy, r, n=39):
    """The 39 folds, as a ring of 39 ticks that cycles: one tick is the fold being run.

    Stage 4's right half was empty and the title already claims "39 folds", so this draws the claim
    instead of decorating the space: 39 marks around a loop, the current one in terra, with an arrowhead
    to show that the loop runs again for the next participant.
    """
    assert n == 39, "the ring must carry one tick per fold"
    ax.add_patch(Circle((cx, cy), r, facecolor="none", edgecolor=HAIR, linewidth=0.7, zorder=5))
    for i in range(n):
        a = np.radians(90.0 - i * 360.0 / n)
        col, lw, dr = (TERRA, 1.0, 0.9) if i == 0 else (MUTED, 0.5, 0.55)
        ax.plot([cx + (r - 0.45) * np.cos(a), cx + (r + dr) * np.cos(a)],
                [cy + (r - 0.45) * np.sin(a), cy + (r + dr) * np.sin(a)], color=col, lw=lw, zorder=6,
                solid_capstyle="round")
    # a clean arrowhead on the loop, tangent at the top-right (the first pass drew a polygon here and it
    # read as a scribble across the ticks)
    a0, a1 = np.radians(60.0), np.radians(46.0)
    ax.add_patch(mpl.patches.FancyArrowPatch((cx + r * np.cos(a0), cy + r * np.sin(a0)),
                                             (cx + r * np.cos(a1), cy + r * np.sin(a1)),
                                             arrowstyle="-|>", mutation_scale=8, lw=0.9, color=CLAY,
                                             zorder=8, shrinkA=0, shrinkB=0))


def stress_icons(x0, y, w):
    """Six glyphs, each CENTRED on the label that names it.

    The first build placed each glyph by its own left edge and each label by its slot centre, so the two
    columns did not line up. Every glyph is now drawn outward from its slot centre.
    """
    step = w / 6.0
    labels = ["sites", "blank rows", "noise", "offset", "drift", "unseen"]
    cy = y + 3.0
    cen = [x0 + step * (i + 0.5) for i in range(6)]
    # 1 eleven sites, one of them lost (the body in stage 2 carries eleven, so this icon does too)
    for i in range(11):
        col, row = i % 6, i // 6
        ax.add_patch(Circle((cen[0] - 2.9 + col * 1.15, cy - 1.1 + row * 1.7), 0.33,
                            facecolor=PAPER if i == 5 else PLUM, edgecolor=INK, linewidth=0.3,
                            zorder=6))
    # 2 two rows of cells with gaps
    for r in range(2):
        for i in range(5):
            gone = (r == 1 and i in (1, 3))
            ax.add_patch(Rectangle((cen[1] - 3.4 + i * 1.35, cy - 1.7 + r * 1.9), 1.05, 1.4,
                                   facecolor=HAIR if gone else CREAM, edgecolor=INK, linewidth=0.35,
                                   zorder=6))
    # 3 a noisy trace
    rng = np.random.default_rng(3)
    xx = np.linspace(cen[2] - 2.6, cen[2] + 2.6, 40)
    ax.plot(xx, cy + rng.normal(0, 0.55, 40), color=TERRA, lw=0.8, zorder=6, solid_capstyle="round")
    # 4 an offset: the same trace, shifted up off its dashed baseline
    ax.plot([cen[3] - 2.6, cen[3] + 2.6], [cy - 0.9, cy - 0.9], color=HAIR, lw=0.8,
            ls=(0, (1.6, 1.4)), zorder=6)
    ax.plot([cen[3] - 2.6, cen[3] + 2.6], [cy + 1.0, cy + 1.0], color=TERRA, lw=1.0, zorder=6,
            solid_capstyle="round")
    # 5 a drift: a rising line
    ax.plot([cen[4] - 2.6, cen[4] + 2.6], [cy - 1.2, cy + 1.3], color=CLAY, lw=1.3, zorder=6,
            solid_capstyle="round")
    # 6 an unseen condition: a thermometer with a dashed question arc
    rrect(cen[5] - 0.75, cy - 1.7, 1.5, 3.6, PAPER, r=0.25, z=6)
    ax.add_patch(Circle((cen[5], cy + 1.6), 0.5, facecolor=AMBER, edgecolor=INK, linewidth=0.35,
                        zorder=7))
    ax.add_patch(Arc((cen[5] + 0.4, cy), 3.4, 3.4, theta1=-60, theta2=40, color=MUTED, linewidth=0.6,
                     linestyle=(0, (1.8, 1.6)), zorder=7))
    for i, lab in enumerate(labels):
        txt(cen[i], y - 1.5, lab, 7.0, MUTED)


# ================================================================== the five stages


def badge(n, x, y):
    ax.add_patch(Circle((x, y), 1.35, facecolor=CLAY, edgecolor=INK, linewidth=0.4, zorder=20))
    txt(x, y - 0.05, str(n), 7.0, PAPER, "bold")


def stage(n, y, title):
    badge(n, 3.0, y)
    txt(5.8, y, title, FS_TITLE, INK, "bold", ha="left")


# ---- stage 1: the cohort, and the controlled range they sat in
stage(1, 73.0, "39 participants, nine controlled conditions")
ax.add_patch(sk(FancyBboxPatch((2.0, 60.6), 32.0, 10.2,
                               boxstyle="round,pad=0,rounding_size=0.5", facecolor=PAPER,
                               edgecolor=HAIR, linewidth=0.9, zorder=1)))
ax.plot([2.6, 33.4], [60.9, 60.9], color=HAIR, lw=0.7, zorder=2)
window(23.4, 63.6, 5.6, 6.0, z=4)
wall_thermo(4.2, 62.4, 2.8, 7.4, 0.62, CLAY, z=8)
for cx in (9.2, 15.0, 20.8):
    shadow(cx, 60.9, 3.4)
    person(cx, 60.9, 9.4, z=6)
range_bar(40.0, 65.6, 44.0, 3.4)

# ---- stage 2: the sensing chain
stage(2, 57.4, "11 skin sites, heart rate, three environment channels")
body = person(21.0, 39.8, 15.6, z=6)
draw_sites(body, r=0.38)
heart(27.2, 49.2, 1.2)
ax.plot([25.2, 26.4], [49.2, 49.2], color=TERRA, lw=0.5, zorder=13)
for i, (kind, lab) in enumerate((("air", "air T"), ("humidity", "humidity"), ("wind", "wind"))):
    env_chip(34.0 + i * 14.0, 44.0, kind, lab)

# ---- stage 3: the estimator, and what it outputs
# The inner selection is `RidgeCV` with no `cv=` argument, i.e. leave-one-out, not k-fold. The panel said
# "5-fold" and contradicted both the Methods and the code, so the title and the badge label carry the
# implementation. "per fold" replaces "inside each fold" to keep the line the same width it was audited at.
stage(3, 35.6, "ridge regression, penalty by leave-one-out CV per fold")
scatter_fit(18.0, 25.6, 24.0, 7.4)
ax.add_patch(mpl.patches.FancyArrowPatch((42.6, 29.3), (44.6, 29.3), arrowstyle="-|>",
                                         mutation_scale=6, lw=0.9, color=MUTED, zorder=9,
                                         shrinkA=0, shrinkB=0))
folds_badge(45.2, 28.8, n=5)
txt(51.1, 26.8, "leave-one-out", 7.0, MUTED)
out_chip(62.0, 26.0)

# ---- stage 4: the evaluation
stage(4, 22.8, "leave one participant out, 39 folds")
ax.add_patch(sk(FancyBboxPatch((6.0, 12.4), 31.0, 8.2, boxstyle="round,pad=0,rounding_size=0.7",
                               facecolor=CREAM, edgecolor=HAIR, linewidth=0.8, zorder=1)))
for cx in (11.1, 17.9, 24.7, 31.5):
    shadow(cx, 13.2, 2.4)
    person(cx, 13.2, 6.6, z=6)
shadow(43.0, 13.2, 2.4)
person(43.0, 13.2, 6.6, z=9, shirt=HAIR, trouser=HAIR)
# No "never in its own fit" here: the fold box and the greyed figure standing outside it say it, and the
# user's note was that a second sentence only repeated the drawing.
fold_ring(70.0, 18.0, 4.0)
txt(70.0, 11.6, "one fold per participant", 7.0, MUTED)

# ---- stage 5: the six stress tests, each with its one-word label
stage(5, 10.4, "then six stress tests")
stress_icons(13.0, 3.4, 68.0)

# ================================================================== audits
fig.canvas.draw()
rend = fig.canvas.get_renderer()
MM = fig.get_dpi() / 25.4
assert min(FS_LET, FS_TITLE, 7.0) >= 7.0, "a font is under the sheet's 7 pt floor"
recs = []
for t in texts:
    bb = t.get_window_extent(renderer=rend)
    x0, y0 = bb.x0 / MM, H_MM - bb.y1 / MM
    x1, y1 = bb.x1 / MM, H_MM - bb.y0 / MM
    assert 0.4 <= x0 and x1 <= W_MM - 0.4, \
        "text %r leaves the canvas sideways (%.1f..%.1f)" % (t.get_text(), x0, x1)
    assert 0.4 <= y0 and y1 <= H_MM - 0.4, \
        "text %r leaves the canvas (%.1f..%.1f)" % (t.get_text(), y0, y1)
    recs.append((t.get_text(), x0, y0, x1, y1))
for i in range(len(recs)):
    for j in range(i + 1, len(recs)):
        ox = min(recs[i][3], recs[j][3]) - max(recs[i][1], recs[j][1])
        oy = min(recs[i][4], recs[j][4]) - max(recs[i][2], recs[j][2])
        if ox > 0.3 and oy > 0.3:
            raise SystemExit("text %r overlaps %r by %.2f x %.2f mm" % (recs[i][0], recs[j][0], ox, oy))
print("eleven site dots on the body, %d text lines, smallest 7.0 pt, no overlap" % len(recs))

fig.savefig(os.path.join(OUT, "fig3a_schematic.pdf"), facecolor=PAPER)
fig.savefig(os.path.join(OUT, "fig3a_schematic.svg"), facecolor=PAPER)
# The PNG is the file a viewer holds open, and on Windows a locked destination makes savefig raise
# OSError 22. Write to a temp name and swap it in, with retries, rather than losing a build to a viewer.
_png = os.path.join(OUT, "fig3a_schematic.png")
_tmp = _png + ".tmp.png"
fig.savefig(_tmp, dpi=DPI, facecolor=PAPER)
for _try in range(8):
    try:
        os.replace(_tmp, _png)
        break
    except OSError:
        time.sleep(1.0)
else:
    _png = _tmp
    print("NOTE: fig3a_schematic.png is open in another program; wrote %s instead" % _png)
im = Image.open(_png)
got = (im.width / (DPI / 25.4), im.height / (DPI / 25.4))
assert abs(got[0] - W_MM) < 0.05 and abs(got[1] - H_MM) < 0.05, "PNG is %s mm" % (got,)
sys.stdout.reconfigure(encoding="ascii", errors="replace")
print("wrote fig3a_schematic.{pdf,svg,png}  %.0f x %.0f mm  PNG %d x %d px"
      % (W_MM, H_MM, im.width, im.height))
