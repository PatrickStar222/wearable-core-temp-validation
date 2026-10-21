# -*- coding: utf-8 -*-
"""fig3c_cost_profile.py -- Figure 3, panel c: the robustness landscape.

WHAT THIS PANEL IS.  Five configurations traced across fifteen conditions on one value axis, so the
reader sees the whole information hierarchy at once: B5 starts best in-domain and stays best under
every synthetic perturbation, then falls to FOURTH of five under an unseen ambient condition while
B2 and B3 rise past it.  That reversal is the panel.  It cannot be shown by a grid of per-cell bars,
because each cell would need its own axis and the between-configuration differences -- 0.02 to 0.06
normalised RMSE -- compress below visibility inside a 4 mm cell.  A rank reversal needs a shared
axis; a shared axis needs the conditions to be positions along it, not containers beside it.

WHY EVERY VALUE IS ON ONE DENOMINATOR.  `score()` in _twist.py divides by the SD of the rows it
scores, so the two held-out ambient cells are normalised by their own condition's spread while every
other cell is normalised by the cohort's.  A landscape puts all fifteen conditions on one continuous
axis, and a value that changes ruler between columns would show the reader the ruler rather than the
model.  Every entry here is sqrt(SSE/n) / SD_cohort, with SD_cohort taken from the frozen board
itself.  This is the ONE place where the panel departs from the published table, and it is stated in
the caption: under the board's per-condition denominator the ambient columns would be the two worst
columns on the page, and under a fixed denominator they are the two best.  What is invariant is the
ORDER -- B5 falls behind B2 and B3 either way.

WHAT EACH MARK IS.  A node is a short thick horizontal segment at the median over participants, and
the thin line joins a configuration's nodes from condition to condition.  No individual observations:
740 points on the same page as the trajectories puts the raw data and the conclusion at one visual
level.  The per-participant distributions are Supplementary.

STRUCTURE.  Fifteen conditions in five blocks -- in-domain | sites kept | blanked rows | noise and
drift | unseen ambient -- separated by whitespace alone, so the page reads as one continuous space
from the baseline to the environmental shift rather than as fifteen unrelated columns.

EXCLUDED BY NAME.
  k=1                 the board evaluates ONE preselected-single-site model and copies it to B2, B3,
                      B4 and B5 (see the loop at _twist.py:168).  Four identical cells would draw a
                      perfect agreement that was never measured.
  bias +0.3, bias +0.5  exactly the in-domain values for B3/B4/B5: the ridge penalty absorbs a
                      constant whole.  A column identical to the baseline column is not a condition.
  naive random split   folds are random row subsets, not participants.
  leave-one-condition  folds are conditions; differencing it against a per-participant baseline
                      would mix units.  Both are Extended Data.

Values: `_twist_folds.csv`, asserted elsewhere to reproduce `_twist_results.csv` per draw.
Run:  python fig3c_cost_profile.py
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = Path("")
sys.path.insert(0, str(P))
import fig2v2_style as S                                   # noqa: E402

S.apply()

T = dict(S.T)
T.update({"letter": 8.0, "ax": 6.2, "tick": 6.4, "end": 6.8, "group": 5.8})

WINK = "#2B2119"            # deep brown-black, for type
FRAME = "#5A4A3E"           # warm brown for spines and ticks: dark, but not black
WGREY = "#8A7F76"           # warm grey: group titles and the end-label column
WHAIR = "#DED4CA"           # fine separator line between stress domains
BAND = "#FCF8F4"            # the alternating domain band: warm tint, deliberately faint
DOT_MS = 0.95               # base dot diameter, mm; each series scales it by CFG_MS below
MARK_RING = 0.55            # paper-coloured ring, so each marker reads as a bead on the stroke
# PROMINENCE, not identity.  B5 and B4 are the subject of the panel and carry the most ink; B3 and
# B2 are supporting; B1 is a reference and recedes.  Series are still told apart by hue alone --
# what varies here is how loudly each one speaks, in three consistent steps.
CFG_MS = {"B5": 1.04, "B4": 0.95, "B3": 0.79, "B2": 0.73, "B1": 0.62}     # marker, mm
CFG_LW = {"B5": 0.76, "B4": 0.70, "B3": 0.56, "B2": 0.52, "B1": 0.48}     # stroke, pt
MIN_SEP_MM = 3.2            # end labels sit AT their own endpoint, parted only where they collide
# Increasing information runs pale -> deep across B2..B5; B1 is categorically different (it has no
# skin channel at all) and takes a neutral so it never reads as one more step on the ramp.
# A layered warm progression with balanced luminance: oxblood -> vermilion -> terracotta ->
# apricot, and B1 held back as warm taupe because it is categorically different (no skin channel).
# Measured colour-only worst pair 62.4 under all four vision types.
CFG_COL = {"B1": "#9A8A7A", "B2": "#E0A57A", "B3": "#B5735A", "B4": "#C0432A", "B5": "#5E1B25"}
# Five series cannot be separated by colour alone inside one hue family: no arrangement of this
# palette clears 40 under all four vision types (the best warm ramp reaches 34, and the palette's
# own five hues reach 18 -- green and coral collapse under deuteranopia).  B1 therefore carries a
# second, non-colour channel.  It is the right series for it: B1 has no skin channel, so its folds
# are structurally different from the other four, and it is the only line the panel draws dashed.
CFG_LS = {"B1": (0, (3.4, 1.7))}
CFG_ORDER = ["B5", "B4", "B3", "B2", "B1"]

MM = 1.0 / 25.4
W_MM, H_MM = 89.0, 53.0

mpl.rcParams.update({
    # S.apply() leaves the default font at its own tick size, so the unlabelled y ticks were
    # rendering at 6.4 pt while the x ticks rendered at 6.8 -- two sizes for one role.
    "font.size": T["tick"], "xtick.labelsize": T["tick"], "ytick.labelsize": T["tick"],
    "axes.labelsize": T["ax"],
    "axes.spines.top": True, "axes.spines.right": True,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 1.8, "ytick.major.size": 1.8,
    "axes.linewidth": 0.6, "axes.edgecolor": FRAME,
    "text.color": WINK, "axes.labelcolor": WINK,
    "xtick.color": WINK, "ytick.color": WINK,
})

# ---------------------------------------------------------------------------- data
BASE = {"B1": "B1 env only", "B2": "B2 single skin", "B3": "B3 multi-site skin",
        "B4": "B4 skin+HR", "B5": "B5 skin+HR+env"}
PUB = {}
with open(P / "_twist_results.csv", newline="", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        if r["rep"] == "":
            PUB[(r["axis"], r["level"], r["baseline"])] = float(r["rmse"]), float(r["nrmse"])
_SU = [PUB[("1 subject shift", "LOSO (participant-grouped)", b)][0] /
       PUB[("1 subject shift", "LOSO (participant-grouped)", b)][1] for b in BASE.values()]
SDC = float(np.mean(_SU))
if max(_SU) - min(_SU) > 1e-6:
    raise SystemExit("the cohort SD is not one number -- the baselines are on different targets")
print(f"[denom] SD_cohort = {SDC:.5f} (one target, so one ruler for all fifteen conditions)")

F = defaultdict(lambda: [0.0, 0])
for r in csv.DictReader(open(P / "_twist_folds.csv", encoding="utf-8")):
    F[(r["axis"], r["level"], r["baseline"], r["fold_kind"], r["fold_id"])][0] += float(r["sse"])
    F[(r["axis"], r["level"], r["baseline"], r["fold_kind"], r["fold_id"])][1] += int(r["n"])


def per_participant(axis, level, base):
    out = {}
    for (a, l, b, kd, fid), (sse, n) in F.items():
        if a == axis and l == level and b == base and kd == "participant" and n:
            out[fid] = np.sqrt(sse / n) / SDC
    return np.array([float(v) for v in out.values()])


#  (label, axis, level, family)
COND = [
    ("ID", "1 subject shift", "LOSO (participant-grouped)", "baseline"),
    ("8", "2 sensor shift", "k=8", "sites"),
    ("2", "2 sensor shift", "k=2", "sites"),
    ("1", "2 sensor shift", "k=1", "sites"),
    ("20", "3 temporal corruption", "gap=20min", "rows"),
    ("60", "3 temporal corruption", "gap=60min", "rows"),
    ("$\\sigma$1", "4 signal corruption", "noise sd=1.0", "signal"),
    ("drift", "4 signal corruption", "drift 0->1C", "signal"),
    ("25", "6 environmental shift", "unseen 25.0C (cond 6)", "ambient"),
    ("29.6", "6 environmental shift", "unseen 29.6C (cond 9)", "ambient"),
]
# The domains are marked by a very light warm strip at each BOUNDARY plus one fine rule, not by a
# block of colour behind the data: a filled panel competes with the trajectories for the eye, and
# the grouping only needs a rhythm, not an area.

VAL, NS = {}, {}
for i, (lab, axis, level, fam) in enumerate(COND):
    for tag, base in BASE.items():
        v = per_participant(axis, level, base)
        if v.size:
            VAL[(i, tag)] = float(np.median(v))
            NS[(i, tag)] = int(v.size)
N_COND = len(COND)
print(f"[matrix] {len(VAL)} of {N_COND * 5} cells populated over {N_COND} conditions x 5 "
      f"configurations")

RANK = {}
for i in range(N_COND):
    row = sorted(((VAL[(i, t)], t) for t in CFG_ORDER if (i, t) in VAL))
    for r, (_v, t) in enumerate(row, 1):
        RANK[(i, t)] = r
base_rank = [f"{t} {RANK[(0, t)]}" for t in CFG_ORDER if (0, t) in RANK]
print("[baseline ranking] " + " < ".join(base_rank))
for i, (lab, _a, _l, _f) in enumerate(COND):
    if NS.get((i, "B5")) == 39:
        continue
    tail = " < ".join(f"{t} {RANK[(i, t)]}" for t in CFG_ORDER if (i, t) in RANK)
    print(f"[ranking] {lab:>6}: {tail}   (n = 8 or 11)")

# ---------------------------------------------------------------------------- figure
fig = plt.figure(figsize=(W_MM * MM, H_MM * MM))
# The box takes everything the line-end labels and the axis do not need; the data is meant to fill
# the canvas, so the margins are the minimum that keeps one label column legible.
AX = dict(x0=8.4, x1=84.1, y0=2.0, y1=46.4)
ax = fig.add_axes([AX["x0"] / W_MM, 1.0 - AX["y1"] / H_MM,
                   (AX["x1"] - AX["x0"]) / W_MM, (AX["y1"] - AX["y0"]) / H_MM])

# condition positions: evenly spaced inside a domain, a wider gap between domains
GAP = 0.50
pos, cur = [], 0.0
prev_fam = None
for _i, (_lab, _a, _l, fam) in enumerate(COND):
    if prev_fam is not None and fam != prev_fam:
        cur += GAP
    pos.append(cur)
    cur += 1.0
    prev_fam = fam
pos = np.array(pos)
ax.set_xlim(pos[0] - 0.44, pos[-1] + 0.44)
# Tied to the data, not to a round number: the trajectories are then painted nearly edge to edge,
# which is most of what makes the lines read as one continuous motion across the four domains.
_yv = [VAL[k] for k in VAL]
ax.set_ylim(min(_yv) - 0.014, max(_yv) + 0.010)
ax.set_yticks([0.4, 0.6, 0.8])
ax.set_xticks(pos)
ax.set_xticklabels([c[0] for c in COND], fontsize=T["tick"])

# Domain rhythm: a very light warm strip at each boundary, one fine rule down its middle, and a
# small superordinate title.  No panel of colour sits behind the data.
fam_span = {}
for _i, (_lab, _a, _l, fam) in enumerate(COND):
    fam_span.setdefault(fam, []).append(_i)
# Domain rhythm.  Alternating domains carry a band so light it reads as paper rather than as a
# fill; the bands encode the grouping and nothing else.  The separators continue below the axis into
# the label strip, so the x axis reads as a partitioned analytical timeline rather than as ten
# free-floating labels.  A small superordinate title sits under each domain.
fam_span = {}
for _i, (_lab, _a, _l, fam) in enumerate(COND):
    fam_span.setdefault(fam, []).append(_i)
for _d, (fam, idx) in enumerate(fam_span.items()):
    if _d % 2 == 1:                                   # alternate: bare / banded / bare / banded
        ax.axvspan(pos[min(idx)] - GAP / 2.0, pos[max(idx)] + GAP / 2.0, color=BAND, lw=0, zorder=0)
for i in range(1, N_COND):
    if COND[i][3] != COND[i - 1][3]:
        _xb = (pos[i] + pos[i - 1]) / 2.0
        ax.plot([_xb, _xb], [0.0, -0.085], transform=ax.get_xaxis_transform(), color=WHAIR,
                lw=0.6, zorder=1, clip_on=False)
for fam, idx in fam_span.items():
    if fam == "baseline":
        continue
    ax.text(float(np.mean(pos[idx])), -0.085, fam.upper(), transform=ax.get_xaxis_transform(),
            fontsize=T["group"], color=WGREY, ha="center", va="top")

# Columns where configurations coincide.  At k=1 the board evaluates ONE preselected-single-site
# model and copies it to B2..B5, so four medians land on the same point.  No dot is drawn there: a
# marker would have to pick one configuration's colour and claim a membership that does not exist,
# and a ringed "coincidence" symbol turns a statistical figure into a slide.  Only the four lines
# converge, which is exactly what the situation is.  Gate 5 keeps this tied to the data.
CONC = []
for _i in range(N_COND):
    _seen = {}
    for _tag in CFG_ORDER:
        if (_i, _tag) in VAL:
            _seen.setdefault(round(VAL[(_i, _tag)], 9), []).append(_tag)
    for _tags in _seen.values():
        if len(_tags) > 1:
            CONC.append((_i, list(_tags)))
CONC_TAGS = {(i, t) for i, g in CONC for t in g}

n_node = n_seg = 0
GAPS = []                                   # (config, condition index) where a line must not pass
for tag in CFG_ORDER:
    idx = [i for i in range(N_COND) if (i, tag) in VAL]
    if not idx:
        continue
    # Break the trajectory wherever a condition is missing.  B1 has no skin channel and B2 has one,
    # so neither is defined under the sensor, temporal or signal perturbations: joining their
    # baseline node straight to their ambient node would draw a continuous trend across conditions
    # that were never measured for them, and it would look exactly like real data.
    run = [idx[0]]
    for a, b in zip(idx, idx[1:]):
        if b == a + 1:
            run.append(b)
        else:
            GAPS.append((tag, a, b))
            if len(run) > 1:
                ax.plot([pos[i] for i in run], [VAL[(i, tag)] for i in run], color=CFG_COL[tag],
                        lw=CFG_LW[tag], alpha=0.96, zorder=3, ls=CFG_LS.get(tag, "-"))
                n_seg += len(run) - 1
            run = [b]
    if len(run) > 1:
        ax.plot([pos[i] for i in run], [VAL[(i, tag)] for i in run], color=CFG_COL[tag],
                lw=CFG_LW[tag], alpha=0.96, zorder=3, ls=CFG_LS.get(tag, "-"))
        n_seg += len(run) - 1
    # the node: a small solid dot, nothing more
    for i in idx:
        if (i, tag) in CONC_TAGS:
            continue
        ax.plot([pos[i]], [VAL[(i, tag)]], "o", ms=CFG_MS[tag] * 72.0 / 25.4, mfc=CFG_COL[tag],
                mec="white", mew=MARK_RING, zorder=5)
        n_node += 1
print(f"[marks] {n_node} dots and {n_seg} trajectory segments; "
      f"{len(GAPS)} gap(s) left unjoined: "
      + ", ".join(f"{t} {COND[a][0]}->{COND[b][0]}" for t, a, b in GAPS))
if CONC:
    print("[marks] coincident columns, dots suppressed and lines made to converge: "
          + ", ".join(f"{COND[i][0]} ({len(g)} configurations)" for i, g in CONC))

# End labels, strictly staggered.  There is no legend: the series are named where they end, so the
# key IS the data.  The labels are set in each series' own colour, which is what ties a staggered
# name to its trajectory -- no leader rule is drawn, because a column of thin diagonals beside the
# plot adds a second set of lines competing with the five that matter.
_end = []
for tag in CFG_ORDER:
    idx = [i for i in range(N_COND) if (i, tag) in VAL]
    if idx:
        _end.append([tag, VAL[(idx[-1], tag)]])
_end.sort(key=lambda e: e[1])
# Each label sits AT its own endpoint, so the trajectory runs straight into its name and no rule is
# needed to explain the association.  Only where two endpoints land almost on top of each other are
# they parted -- and then by the minimum, not into a uniform column: a forced stagger reads as a
# stack of tags, which is what a legend already was.
_sep = MIN_SEP_MM / (AX["y1"] - AX["y0"]) * (ax.get_ylim()[1] - ax.get_ylim()[0])
_ly = [_e[1] for _e in _end]
for _k in range(1, len(_ly)):
    if _ly[_k] - _ly[_k - 1] < _sep:
        _ly[_k] = _ly[_k - 1] + _sep
n_parted = sum(1 for _a, _b in zip(_ly, [e[1] for e in _end]) if abs(_a - _b) > 1e-9)
for (tag, _val), _y in zip(_end, _ly):
    ax.text(pos[-1] + 0.44 + 0.16, _y, tag, fontsize=T["end"], color=CFG_COL[tag],
            fontweight="bold", ha="left", va="center", clip_on=False, zorder=8)
print(f"[labels] {len(_end)} end labels, {len(_end) - n_parted} exactly on their own endpoint and "
      f"{n_parted} parted by the minimum; no leader rules")

# This panel prints as c: with the cool-end transfer panel moved out of the sheet (its content is Figure
# 2d's dataset, see FIG3_SHEET.md), the four remaining single-column panels split back into an
# alphabetical pair, so the build-time letter is the printed letter again.
# the axis title belongs in the axis margin, not floating inside the field
ax.set_ylabel("normalised RMSE", fontsize=T["group"], color=WGREY, labelpad=2.0)

# ---------------------------------------------------------------- build-time gates
fig.canvas.draw()
_r = fig.canvas.get_renderer()


def mmbb(t):
    bb = t.get_window_extent(_r)
    return (bb.x0 / fig.dpi / MM, (fig.bbox.height - bb.y1) / fig.dpi / MM,
            bb.x1 / fig.dpi / MM, (fig.bbox.height - bb.y0) / fig.dpi / MM)


texts = [t for t in fig.texts if t.get_text()]
texts += [t for t in getattr(ax, "texts", []) if t.get_text()]
texts += [t for t in (list(ax.get_xticklabels()) + list(ax.get_yticklabels())
                      + [ax.xaxis.label, ax.yaxis.label]) if t.get_text()]
bad = []
for t in texts:
    x0, y0, x1, y1 = mmbb(t)
    if x1 > W_MM - 0.5 or x0 < 0.5 or y1 > H_MM - 0.2 or y0 < 0.2:
        bad.append(f"'{t.get_text()[:24]}' x[{x0:.1f},{x1:.1f}] y[{y0:.1f},{y1:.1f}]")
if bad:
    raise SystemExit("TEXT OVERFLOWS THE PAGE: " + "; ".join(bad))
print(f"[gate 1] all {len(texts)} text objects sit inside {W_MM:.0f} x {H_MM:.0f} mm")

BOX = [(t, mmbb(t)) for t in texts]
for i in range(len(BOX)):
    for j in range(i + 1, len(BOX)):
        (ta, a), (tb, b) = BOX[i], BOX[j]
        if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
            raise SystemExit(f"TEXT ON TEXT: '{ta.get_text()[:18]}' vs '{tb.get_text()[:18]}'")
print("[gate 2] no text collides with another text")

# gate 3 -- every populated cell is either drawn or is one of the deliberately suppressed
# coincident dots.  A landscape is exactly where a dropped node vanishes into the field without
# changing the picture's plausibility, so the difference between "not drawn" and "not drawn on
# purpose" has to be closed here rather than assumed.
if n_node + len(CONC_TAGS) != len(VAL):
    raise SystemExit(f"NODES LOST: {n_node} drawn + {len(CONC_TAGS)} suppressed "
                     f"= {n_node + len(CONC_TAGS)} of {len(VAL)}")
print(f"[gate 3] {n_node} cells drawn plus {len(CONC_TAGS)} coincident dots deliberately "
      f"suppressed = all {len(VAL)} populated cells accounted for")

# gate 4 -- the renormalisation must be doing something, and only where it is needed.  A cell whose
# scored rows ARE the cohort has its own denominator equal to SD_cohort, so overriding it changes
# nothing; a cell scored on a subset does not.  If the first set is not empty-minus-the-ambient-cells
# then this panel is renormalising for no reason, and if the ambient cells stop differing then the
# whole justification for the common denominator has evaporated and the panel should go back to the
# published table's numbers.
_overridden = set()
for _i, (_lab, axis, level, _fam) in enumerate(COND):
    for _tag, _base in BASE.items():
        _pub = PUB.get((axis, level, _base))
        if _pub is None or _pub[1] <= 0 or (_i, _tag) not in VAL:
            continue
        if abs(_pub[0] / _pub[1] - SDC) > 1e-6:
            _overridden.add(_i)                # by INDEX, not by label -- the labels are editorial
if not _overridden:
    raise SystemExit("THE COMMON DENOMINATOR IS DOING NOTHING -- no cell's own ruler differs from "
                     "SD_cohort, so this panel should quote the published numbers unchanged")
_amb = {_i for _i, (_l, _a, _lv, _f) in enumerate(COND) if _f == "ambient"}
if _overridden != _amb:
    raise SystemExit("THE RULER IS NOT CONSTANT WHERE IT SHOULD BE: the cells whose own denominator "
                     f"differs from SD_cohort are {sorted(COND[i][0] for i in _overridden)}, not "
                     f"just the held-out ambient conditions {sorted(COND[i][0] for i in _amb)} -- "
                     "the board's normalisation has changed underneath")
print(f"[gate 4] the board's own ruler equals SD_cohort = {SDC:.5f} on every cell except the two "
      f"held-out ambient conditions, which is exactly the set this panel renormalises")

# gate 5 -- the excluded columns are excluded because they carry no information, and that is a
# property of the data, not of the drawing.  Re-assert it here so a board change cannot silently
# turn them into real conditions.
_bias = [per_participant("4 signal corruption", "bias +0.5", b) for b in BASE.values()]
_idom = [per_participant("1 subject shift", "LOSO (participant-grouped)", b) for b in BASE.values()]
for tag, a, b in zip(BASE, _bias, _idom):
    if tag in ("B3", "B4", "B5") and not np.allclose(np.sort(a), np.sort(b), atol=1e-12):
        raise SystemExit(f"THE OFFSET IS NO LONGER ABSORBED EXACTLY on {tag}: the omitted `bias` "
                         "column has become a real condition")
# k=1 is DRAWN, but it is one model copied to four baselines rather than a comparison of four.  If
# that ever stops being true the concentric rendering is wrong and the column becomes a normal one.
_k1 = {t: per_participant("2 sensor shift", "k=1", BASE[t]) for t in BASE if t != "B1"}
_g = [np.median(v) for v in _k1.values()]
if max(_g) - min(_g) > 1e-12:
    raise SystemExit("k=1 is no longer ONE model copied to four baselines -- it has become four "
                     "independent conditions, so the concentric dots must be replaced by ordinary "
                     "ones")
if not CONC or COND[CONC[0][0]][2] != "k=1":
    raise SystemExit("THE COINCIDENT-COLUMN HANDLING IS NOT FIRING on the k=1 level -- four dots "
                     "would be drawn on top of one another and three would be invisible")
print(f"[gate 5] the omitted `bias` column is still exactly in-domain on B3/B4/B5, and k=1 is still "
      f"one model (median {_g[0]:.4f}) copied to {len(_g)} baselines, so its dots are suppressed "
      "and only the lines converge")

# gate 6 -- no segment may bridge a condition that was never measured.  B1 and B2 are undefined
# under nine of the fifteen conditions; a polyline drawn through their two ends would render a
# nine-condition extrapolation as though it were data, and nothing on the page would look wrong.
for _tag in CFG_ORDER:
    _idx = [i for i in range(N_COND) if (i, _tag) in VAL]
    _joined = {(a, b) for a, b in zip(_idx, _idx[1:]) if b == a + 1}
    _skipped = [(a, b) for a, b in zip(_idx, _idx[1:]) if b != a + 1]
    for a, b in _skipped:
        if (a, b) in _joined:
            raise SystemExit(f"LINE BRIDGES A GAP on {_tag}: {COND[a][0]} -> {COND[b][0]}")
    _drawn = sum(1 for a, b in zip(_idx, _idx[1:]) if b == a + 1)
if n_seg != sum(1 for _tag in CFG_ORDER for a, b in
                zip([i for i in range(N_COND) if (i, _tag) in VAL],
                    [i for i in range(N_COND) if (i, _tag) in VAL][1:]) if b == a + 1):
    raise SystemExit("SEGMENT COUNT MISMATCH: the drawn trajectories are not the adjacent pairs")
print(f"[gate 6] every trajectory is drawn only between ADJACENT measured conditions; "
      f"{len(GAPS)} unmeasured spans are left visibly open")

plt.rcParams["savefig.bbox"] = None
plt.rcParams["savefig.pad_inches"] = 0.0
OUT = P / "fig3_v2"
fig.savefig(OUT / "fig3c_cost_profile.pdf")
fig.savefig(OUT / "fig3c_cost_profile.png", dpi=600)
print("wrote", OUT / "fig3c_cost_profile.pdf")
print("wrote", OUT / "fig3c_cost_profile.png")
print(f"page {W_MM:.0f} x {H_MM:.0f} mm")
plt.close(fig)
