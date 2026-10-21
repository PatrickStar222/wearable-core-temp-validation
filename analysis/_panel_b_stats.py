# -*- coding: utf-8 -*-
"""_panel_b_stats.py -- the numbers behind panel b, with participant bootstrap intervals.

Question the panel answers: does a single skin site carry enough information to recover
core temperature, do the sites differ by more than their intervals, and does adding sites
keep buying anything?

Estimator held fixed: ridge, alpha by 5-fold CV inside each training block, features
standardised on training rows only, 39-fold leave-one-subject-out by participant.
Interval: 2000 participant bootstrap resamples of the pooled squared error.

Writes fig3_v2/_panel_b_sites.csv, which the panel script reads and asserts against.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = Path("")
OUT = P / "fig3_v2" / "_panel_b_sites.csv"
OUT.parent.mkdir(exist_ok=True)

ALPHAS = np.logspace(-2, 4, 25)
B = 2000
SKIN = ["abdomen", "calf", "chest", "foot", "hand", "head",
        "lower arm", "lower back", "thigh", "upper arm", "upper back"]
ENV = ["Temp", "Humidity", "Wind"]

df = pd.read_pickle(P / "_prospie.pkl")
for c in SKIN + ["Corerectal"] + ENV + ["HR"]:
    df[c] = pd.to_numeric(df[c], errors="coerce").replace(9999, np.nan)
D = df.loc[df[SKIN + ["HR"] + ENV + ["Corerectal"]].notna().all(axis=1)].reset_index(drop=True)
Y = D.Corerectal.values.astype(float)
SUBJ = D.Participant.values
print(f"complete-case rows={len(D)}  subjects={D.Participant.nunique()}")

SITE = D[SKIN].values.astype(float)
SUBS = np.unique(SUBJ)
ROWS = [np.where(SUBJ == s)[0] for s in SUBS]
RNG = np.random.default_rng(0)
PICK = [np.concatenate([ROWS[i] for i in RNG.integers(0, len(ROWS), len(ROWS))])
        for _ in range(B)]


def loso(X):
    """leave-one-subject-out predictions for a feature matrix"""
    p = np.full(len(Y), np.nan)
    for s in SUBS:
        te = SUBJ == s
        tr = ~te
        sc = StandardScaler().fit(X[tr])
        m = RidgeCV(alphas=ALPHAS).fit(sc.transform(X[tr]), Y[tr])
        p[te] = m.predict(sc.transform(X[te]))
    return p


def nrmse(rows, err):
    return float(np.sqrt(np.mean(err[rows] ** 2)) / Y[rows].std())


rows = []
for j, nm in enumerate(SKIN):
    err = loso(SITE[:, [j]]) - Y
    point = nrmse(np.arange(len(Y)), err)
    bs = np.array([nrmse(r, err) for r in PICK])
    rows.append(dict(site=nm, nrmse=point,
                     lo=float(np.percentile(bs, 2.5)),
                     hi=float(np.percentile(bs, 97.5))))
    print(f"  {nm:<12} nRMSE={point:.4f}  [{rows[-1]['lo']:.4f}, {rows[-1]['hi']:.4f}]")

# ordered reference: the 11-site value, and the value once two sites are in
order = [r["site"] for r in sorted(rows, key=lambda r: r["nrmse"])]
full_err = loso(SITE) - Y
full = nrmse(np.arange(len(Y)), full_err)
err2 = loso(SITE[:, [SKIN.index(order[0]), SKIN.index(order[1])]]) - Y
two = nrmse(np.arange(len(Y)), err2)
env_err = loso(D[ENV].values.astype(float)) - Y
env = nrmse(np.arange(len(Y)), env_err)

out = pd.DataFrame(rows)
out.to_csv(OUT, index=False, float_format="%.6f")

# Per-participant normalised error for each site: the 39 values behind each interval.  A
# figure that draws only the summary repeats the same statistical shape eleven times; these
# are the observations it summarises.
PERR = Path(str(OUT).replace("_panel_b_sites.csv", "_panel_b_participants.csv"))
prows = []
for nm in SKIN:
    err = loso(SITE[:, [SKIN.index(nm)]]) - Y
    for s_ in SUBS:
        m = SUBJ == s_
        prows.append(dict(site=nm, participant=s_,
                          nrmse=float(np.sqrt(np.mean(err[m] ** 2)) / Y[m].std())))
pd.DataFrame(prows).to_csv(PERR, index=False, float_format="%.6f")
print(f"wrote {PERR}  ({len(prows)} site-participant values)")
print(f"\nwrote {OUT}")
print(f"all 11 sites   {full:.4f}")
print(f"two best sites {two:.4f}   ({order[0]} + {order[1]})")
print(f"environment    {env:.4f}")
print(f"best single    {out.nrmse.min():.4f}  ({out.loc[out.nrmse.idxmin(), 'site']})")
print(f"worst single   {out.nrmse.max():.4f}  ({out.loc[out.nrmse.idxmax(), 'site']})")
print(f"spread/median-interval-width = "
      f"{(out.nrmse.max() - out.nrmse.min()) / np.median(out.hi - out.lo):.2f}")
