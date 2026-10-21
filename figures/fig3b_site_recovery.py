# -*- coding: utf-8 -*-
"""fig3b_site_recovery.py -- new Figure 3, panel b: what one skin site recovers.

THE FORM IS FIXED.  Do not redesign it without asking.
This is the bar form the user approved ("现在挺好").  Four other forms were tried and all
four were worse or wrong: a dot-with-interval dumbbell (too sparse), a delta-from-full-set
bar (the scale stopped being readable), a two-tone outage bar (still eleven near-identical
outlines, and it moved the scale off intuition), and a violin of the per-participant spread
(visually fine, wrong object -- it draws the marginal spread and hides the within-participant
pairing, which is where the effect actually lives: foot beats head in 97 % of participants).
Keep the bar.  If the panel has to say more, add a caption sentence, not a second chart.

Question the panel defends: does a single skin site carry enough information to recover core
temperature, and does adding sites keep buying anything?

Style: the Origin convention settled in FIG2_STATE.md §6.1 after the instruction
"像 origin 画的" -- closed four-sided frame, ticks inward and mirrored on all four sides,
2-4 major ticks per axis, no minor ticks, no gridlines.  Colours, type and stroke weights come
from fig2v2_style so this panel and Figure 2 read as one system.

Values: fig3_v2/_panel_b_sites.csv, written by _panel_b_stats.py; three per-site values are
asserted against the frozen stress board `_twist_results.csv`, so a drift between this panel
and the delivered board stops the build.

Page: 89 x 61 mm (single column).  Run:  python fig3b_site_recovery.py
"""
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

P = Path("")
sys.path.insert(0, str(P))
import fig2v2_style as S                                   # noqa: E402

S.apply()
MM = 1.0 / 25.4
W_MM, H_MM = 89.0, 61.0

# ------------------------------------------------------------------ Origin overrides
# fig2v2_style.apply() ships Figure 2's default: left/bottom spines, ticks outward.
# FIG2_STATE.md §6.1 records the instruction "像 origin 画的" and settles the convention on a
# closed box with inward, mirrored ticks; that supersedes the default.
AX_W = 1.00
TICK_L = 3.0
mpl.rcParams.update({
    "axes.spines.top": True, "axes.spines.right": True,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.major.width": AX_W, "ytick.major.width": AX_W,
    "xtick.major.size": TICK_L, "ytick.major.size": TICK_L,
    "axes.linewidth": AX_W, "axes.edgecolor": S.INK,
})

# ---------------------------------------------------------------------------- values
STATS = P / "fig3_v2" / "_panel_b_sites.csv"
BOARD = P / "_twist_results.csv"
if not STATS.exists():
    raise SystemExit("run _panel_b_stats.py first -- " + str(STATS))

with open(STATS, newline="", encoding="utf-8") as fh:
    SITE = [{"site": r["site"], "v": float(r["nrmse"]),
             "lo": float(r["lo"]), "hi": float(r["hi"])} for r in csv.DictReader(fh)]

board = {}
with open(BOARD, newline="", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        board.setdefault((r["axis"], r["level"], r["baseline"]), []).append(float(r["nrmse"]))


def bval(axis, level, base):
    return float(np.mean(board[(axis, level, base)]))


def site(name):
    return next(s for s in SITE if s["site"] == name)


CHECKS = [
    (site("foot")["v"], bval("2 sensor shift", "k=1", "B2 single skin"), 5e-4),
    (site("hand")["v"], 0.9199, 5e-4),
    (site("head")["v"], 0.9706, 5e-4),
]
for got, want, tol in CHECKS:
    if abs(got - want) > tol:
        raise SystemExit(f"FREEZE MISMATCH: per-site {got:.5f} vs board {want:.5f}")
FULL = bval("1 subject shift", "LOSO (participant-grouped)", "B3 multi-site skin")
print(f"[freeze] {len(CHECKS)} per-site values agree with _twist_results.csv")

# ---------------------------------------------------------------------------- figure
fig = plt.figure(figsize=(W_MM * MM, H_MM * MM))
ax = fig.add_axes([20.5 / W_MM, 9.5 / H_MM, 58.5 / W_MM, 41.0 / H_MM])

SITE.sort(key=lambda s: -s["v"])                # worst at the top, best at the bottom
y = np.arange(len(SITE))

# The bars start at the axis floor, not at zero: the quantity compared is a normalised error
# whose meaningful range is 0.6 to 1.0, and a zero baseline would compress every difference in
# this panel into a few millimetres.  The floor is stated in the caption.
X0, X1 = 0.660, 1.000
BAR_FILL = "#F7DCD4"                            # a tint of SKIN, so one hue carries the panel

# One bar per site with its confidence interval, and the point estimate in a column OUTSIDE
# the frame so that it can never collide with its own interval however wide that interval is.
BAR_W = 0.60
for i, s in enumerate(SITE):
    ax.barh(y[i], s["v"] - X0, BAR_W, left=X0, color=BAR_FILL,
            edgecolor=S.SKIN_D, linewidth=0.9, zorder=3)
    ax.plot([s["lo"], s["hi"]], [y[i], y[i]], "-", color=S.INK, lw=1.3,
            solid_capstyle="butt", zorder=6)
    for x in (s["lo"], s["hi"]):
        ax.plot([x, x], [y[i] - 0.15, y[i] + 0.15], "-", color=S.INK, lw=1.3, zorder=6)
    ax.plot([s["v"]], [y[i]], "|", ms=7.0, color=S.INK, mew=1.5, zorder=7)
    ax.annotate(f"{s['v']:.3f}", xy=(1.0, y[i]), xycoords=("axes fraction", "data"),
                xytext=(5.0, 0.0), textcoords="offset points", fontsize=S.T["tick"],
                color=S.INK, ha="left", va="center", annotation_clip=False, zorder=8)

ax.set_yticks(y)
ax.set_yticklabels([s["site"] for s in SITE], fontsize=S.T["tick"])
for t in ax.get_yticklabels():
    t.set_ha("right")
ax.set_ylim(len(SITE) - 0.45, -0.55)
ax.set_xlim(X0, X1)
ax.set_xticks([0.7, 0.8, 0.9])
ax.set_xticklabels(["0.7", "0.8", "0.9"], fontsize=S.T["tick"])
ax.set_xlabel("normalised RMSE", fontsize=S.T["ax"])

TITLE = "One skin site does most of the work"
# The panel letter is NOT drawn here: the user's instruction is that no fig3 subplot carries its own
# letter, and the compositor (`fig3_assemble.py`) letters the page instead.
title = fig.text(20.5 / W_MM, 1.0 - 2.2 / H_MM, TITLE, fontsize=S.T["title"],
                 fontweight="bold", color=S.INK, ha="left", va="top")

# ---------------------------------------------------------------- build-time gates
fig.canvas.draw()
_r = fig.canvas.get_renderer()


def mmbb(t):
    bb = t.get_window_extent(_r)
    return (bb.x0 / fig.dpi / MM, (fig.bbox.height - bb.y1) / fig.dpi / MM,
            bb.x1 / fig.dpi / MM, (fig.bbox.height - bb.y0) / fig.dpi / MM)


# gate 0 -- no data mark may fall outside the frame
_xlo, _xhi = ax.get_xlim()
_out = [f"{s['site']} [{s['lo']:.3f},{s['hi']:.3f}]" for s in SITE
        if s["lo"] < _xlo or s["hi"] > _xhi]
if _out:
    raise SystemExit("DATA CLIPPED BY THE FRAME: " + "; ".join(_out))
print(f"[gate 0] all {len(SITE)} intervals lie inside the frame x[{_xlo:.3f},{_xhi:.3f}]")

# gate 1 -- nothing outside the page
items = [("title", title), ("xlabel", ax.xaxis.label), ("ylabel", ax.yaxis.label)]
items += [(f"xtick{i}", t) for i, t in enumerate(ax.get_xticklabels())]
items += [(f"ytick{i}", t) for i, t in enumerate(ax.get_yticklabels())]
_over = []
for name, t in items:
    x0, y0, x1, y1 = mmbb(t)
    if x1 > W_MM - 0.6 or x0 < 0.6 or y1 > H_MM - 0.6 or y0 < 0.6:
        _over.append(f"{name} x[{x0:.1f},{x1:.1f}] y[{y0:.1f},{y1:.1f}]")
if _over:
    raise SystemExit("TEXT OVERFLOWS THE PAGE: " + "; ".join(_over))
print(f"[gate 1] every text object sits inside {W_MM:.0f} x {H_MM:.0f} mm")

# gate 2 -- text must not sit on a data mark.  The per-row value labels are exempt: they are
# placed at a fixed 5 pt offset from their own bar end by construction.
marks = []
for i, s in enumerate(SITE):
    for x in (s["lo"], s["hi"], s["v"]):
        marks.append(ax.transData.transform((x, i)))
for name, t in items:
    if name.startswith(("ytick", "x", "y")) and name in ("xlabel", "ylabel"):
        continue
    if name.startswith("ytick"):
        continue
    x0, y0, x1, y1 = mmbb(t)
    for px, py in marks:
        mx, my = px / fig.dpi / MM, (fig.bbox.height - py) / fig.dpi / MM
        if x0 - 0.6 < mx < x1 + 0.6 and y0 - 0.6 < my < y1 + 0.6:
            raise SystemExit(f"TEXT OVERLAPS DATA: {name}")
print("[gate 2] no text sits on a data mark")

plt.rcParams["savefig.bbox"] = None
plt.rcParams["savefig.pad_inches"] = 0.0
OUT = P / "fig3_v2"
fig.savefig(OUT / "fig3b_site_recovery.pdf")
fig.savefig(OUT / "fig3b_site_recovery.png", dpi=600)
print("wrote", OUT / "fig3b_site_recovery.pdf")
print("wrote", OUT / "fig3b_site_recovery.png")
print(f"page {W_MM:.0f} x {H_MM:.0f} mm | full set {FULL:.4f} | best {SITE[-1]['site']} "
      f"{SITE[-1]['v']:.4f} | worst {SITE[0]['site']} {SITE[0]['v']:.4f}")
plt.close(fig)
