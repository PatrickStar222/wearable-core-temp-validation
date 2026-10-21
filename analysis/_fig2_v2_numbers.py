"""Numbers and assets the Figure 2 blueprint cites, computed here so the plan is
not allowed to carry a number that was never measured.
"""
import json
import os
import sys

for _opt in ("numexpr", "bottleneck"):
    sys.modules.setdefault(_opt, None)

import numpy as np
import pandas as pd

SITES = ["abdomen", "calf", "chest", "foot", "hand", "head", "lower arm",
         "lower back", "thigh", "upper arm", "upper back"]
df = pd.read_pickle("_prospie.pkl")
for c in SITES + ["Corerectal", "Temp"]:
    df[c] = pd.to_numeric(df[c], errors="coerce").replace(9999, np.nan)
df["skin_mean"] = df[SITES].mean(axis=1)

gm = df.groupby("Condition")[SITES + ["Temp", "Corerectal", "skin_mean"]].mean()
gm = gm.sort_values("Temp")

print("=== per-condition: pooled r, best/worst site r, gap, sd of gap ===")
rows = []
for cond, sub in df.groupby("Condition"):
    amb = sub.Temp.mean()
    s = sub[["skin_mean", "Corerectal"]].dropna()
    r_pool = float(np.corrcoef(s.skin_mean, s.Corerectal)[0, 1])
    per = {}
    for site in SITES:
        ss = sub[[site, "Corerectal"]].dropna()
        per[site] = float(np.corrcoef(ss[site], ss.Corerectal)[0, 1])
    best = max(per, key=per.get)
    worst = min(per, key=per.get)
    gap = (sub.skin_mean - sub.Corerectal).dropna()
    rows.append(dict(cond=int(cond), ambient=amb, r_pool=r_pool,
                     r_best=per[best], best=best,
                     r_worst=per[worst], worst=worst,
                     spread=per[best] - per[worst],
                     gap=gap.mean(), gap_sd=gap.std(), gap_n=len(gap),
                     n=len(sub), n_par=int(sub.Participant.nunique())))
rows.sort(key=lambda d: d["ambient"])
for d in rows:
    print("  amb %6.2f | r_pool %+.3f | best %s %+.3f | worst %s %+.3f | "
          "spread %.3f | gap %+.3f sd %.3f | n %d"
          % (d["ambient"], d["r_pool"], d["best"], d["r_best"], d["worst"],
             d["r_worst"], d["spread"], d["gap"], d["gap_sd"], d["gap_n"]))

amb = np.array([d["ambient"] for d in rows])
rp = np.array([d["r_pool"] for d in rows])
print("\ncorr(ambient, r_pool) = %+.3f" % np.corrcoef(amb, rp)[0, 1])
print("corr(ambient, gap)    = %+.3f" % np.corrcoef(amb, np.array([d["gap"] for d in rows]))[0, 1])
print("corr(ambient, gap_sd) = %+.3f" % np.corrcoef(amb, np.array([d["gap_sd"] for d in rows]))[0, 1])
print("r_pool span %.3f -> %.3f (drop %.0f%%)" % (rp.max(), rp.min(), 100 * (rp.max() - rp.min()) / rp.max()))

print("\n=== site sensitivity, ranked ===")
sens = []
for site in SITES:
    sl, ic = np.polyfit(gm.Temp, gm[site], 1)
    sens.append((site, float(sl), float(np.corrcoef(gm.Temp, gm[site])[0, 1])))
for site, sl, r in sorted(sens, key=lambda t: -t[1]):
    print("  %-12s slope %+.4f  r %+.3f" % (site, sl, r))

print("\n=== assets present ===")
for p in ["fig3_panels/极简扁平矢量男性人体剪影.png",
          "fig3_panels/figure_asset.png",
          "fig2_panels/A_cool_side.png"]:
    print("  %-46s %s" % (p, "FOUND" if os.path.exists(p) else "missing"))

os.makedirs("fig2_v2", exist_ok=True)
with open("fig2_v2/_panel_numbers.json", "w", encoding="utf-8") as f:
    json.dump(dict(per_condition=rows, sites=[dict(site=s, slope=v, r=r) for s, v, r in sens]),
              f, indent=2)
print("\nwrote fig2_v2/_panel_numbers.json")
