# -*- coding: utf-8 -*-
"""_twist_folds.py -- re-run the stress board KEEPING the per-fold values.

Why this exists.  `_twist.py` writes one pooled row per cell: `score()` concatenates every fold's
predictions and reports a single RMSE.  That is the right product for a board, and it is the frozen
product every figure asserts against -- so `_twist.py` is NOT touched here.  But it means the
45-cell board carries no spread: the panel built from it has no uncertainty to show, and an
experimental figure with no uncertainty in it is thin, whatever it looks like.

This script reproduces the same protocol and records, for every fold, the ROW COUNT and the SUM OF
SQUARED ERRORS.  Those two numbers pool exactly:

    pooled RMSE = sqrt( sum_f SSE_f / sum_f n_f )

so the re-run can be checked against the frozen board rather than trusted.  The check is not
optional: if the folds do not reconstruct `_twist_results.csv`, nothing built on them means
anything.

One subtlety that decides whether the reconstruction is exact.  `score()` divides RMSE by the
standard deviation of the rows it scores.  A fold that used its OWN sd as the denominator could not
be pooled back, so the denominator here is the POOLED sd of the scored rows -- the same number for
every fold of a cell -- and each fold stores SSE and n so the pooling stays exact.

Usage:  python _twist_folds.py          (~minutes; writes _twist_folds.csv)
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = ""
sys.path.insert(0, P)
import _flib as F                                            # noqa: E402

ALPHAS = np.logspace(-2, 4, 25)
SKIN = F.SKIN
ENV = ["Temp", "Humidity", "Wind"]

t0 = time.time()
df = pd.read_pickle(os.path.join(P, "_prospie.pkl"))
df["tmin"] = df.Time.apply(
    lambda t: t.hour * 60 + t.minute + t.second / 60 if hasattr(t, "hour") else np.nan)
for c in SKIN + ["Corerectal"] + ENV + ["HR"]:
    df[c] = pd.to_numeric(df[c], errors="coerce").replace(9999, np.nan)
df = df.sort_values(["Participant", "Condition", "tmin"]).reset_index(drop=True)
df["unit"] = df.Participant.astype(str) + "|" + df.Condition.astype(str)
D = df.loc[df[SKIN + ["HR"] + ENV + ["Corerectal"]].notna().all(axis=1)].reset_index(drop=True)
C = D.Corerectal.values.astype(float)
SUBJ = D.Participant.values
COND = D.Condition.values
ENVV = D[ENV].values.astype(float)
HRV = D.HR.values.astype(float).reshape(-1, 1)
S0 = D[SKIN].values.astype(float)
nR = len(D)
uidx = {c: np.where(D.unit.values == c)[0] for c in D.unit.unique()}
print(f"PROSPIE complete-case: rows={nR} subjects={D.Participant.nunique()} "
      f"conditions={D.Condition.nunique()}")


def fit_eval(X, y, tr_mask, te_mask, interactions=None):
    Xi = X if interactions is None else np.column_stack([X, interactions])
    sc = StandardScaler().fit(Xi[tr_mask])
    m = RidgeCV(alphas=ALPHAS).fit(sc.transform(Xi[tr_mask]), y[tr_mask])
    p = np.full(len(y), np.nan)
    if te_mask is not None:
        p[te_mask] = m.predict(sc.transform(Xi[te_mask]))
    else:
        p = m.predict(sc.transform(Xi))
    return p


# --------------------------------------------------------------------------- the fold recorder
FOLDS = []
_POOL = {}


def pooled_sd(mask):
    """the denominator score() uses, computed ONCE per cell so every fold shares it"""
    key = mask.tobytes()
    if key not in _POOL:
        _POOL[key] = float(np.std(C[mask]))
    return _POOL[key]


def record(axis, level, baseline, rep, fold_kind, fold_id, mask, sse, n, sd):
    FOLDS.append(dict(axis=axis, level=level, baseline=baseline, rep=rep, fold_kind=fold_kind,
                      fold_id=fold_id, n=int(n), sse=float(sse),
                      rmse=float(np.sqrt(sse / n)) if n else np.nan,
                      nrmse=float(np.sqrt(sse / n) / sd) if n and sd > 1e-9 else np.nan))


def run_loso(axis, level, baseline, X, inter=None, rep=None, extra_mask=None):
    """LOSO with one recorded fold per held-out participant"""
    p = np.full(nR, np.nan)
    for s in np.unique(SUBJ):
        te = SUBJ == s
        if extra_mask is not None:
            te = te & extra_mask
        if not te.any():
            continue
        q = fit_eval(X, C, ~(SUBJ == s), te, inter)
        p[te] = q[te]
    scored = np.isfinite(p) if extra_mask is None else (extra_mask & np.isfinite(p))
    sd = pooled_sd(scored)
    for s in np.unique(SUBJ):
        te = (SUBJ == s) & scored
        if te.sum() == 0:
            continue
        e = p[te] - C[te]
        record(axis, level, baseline, rep, "participant", int(s), te, float((e ** 2).sum()),
               int(te.sum()), sd)
    return p, scored


def run_condition_heldout(axis, level, baseline, X, inter=None):
    p = np.full(nR, np.nan)
    for c in np.unique(COND):
        te = COND == c
        q = fit_eval(X, C, ~te, te, inter)
        p[te] = q[te]
    scored = np.isfinite(p)
    sd = pooled_sd(scored)
    for c in np.unique(COND):
        te = (COND == c) & scored
        if te.sum() == 0:
            continue
        e = p[te] - C[te]
        record(axis, level, baseline, None, "condition", int(c), te, float((e ** 2).sum()),
               int(te.sum()), sd)
    return p, scored


def run_trainmask(axis, level, baseline, X, train_mask, test_mask, inter=None):
    p = np.full(nR, np.nan)
    for s in np.unique(SUBJ[test_mask]):
        te = test_mask & (SUBJ == s)
        tr = train_mask & (SUBJ != s)
        if tr.sum() < 200 or te.sum() < 5:
            continue
        q = fit_eval(X, C, tr, te, inter)
        p[te] = q[te]
    scored = test_mask & np.isfinite(p)
    sd = pooled_sd(scored)
    for s in np.unique(SUBJ[test_mask]):
        te = scored & (SUBJ == s)
        if te.sum() == 0:
            continue
        e = p[te] - C[te]
        record(axis, level, baseline, None, "participant", int(s), te, float((e ** 2).sum()),
               int(te.sum()), sd)
    return p, scored


def best_single_site(train_mask, y):
    best, bi = np.inf, 0
    for j in range(S0.shape[1]):
        X = S0[:, [j]]
        p = fit_eval(X, y, train_mask, train_mask)
        r = np.sqrt(np.mean((p[train_mask] - y[train_mask]) ** 2))
        if r < best:
            best, bi = r, j
    return bi


# ---------------------------------------------------------------- the protocol, same order/seeds
X_env, X_hr = ENVV, HRV
j_best = best_single_site(np.ones(nR, bool), C)
X_single = S0[:, [j_best]]
print(f"single site chosen on training folds: {SKIN[j_best]}")
FEATS = [("B1 env only", X_env, None), ("B2 single skin", X_single, None),
         ("B3 multi-site skin", S0, None), ("B4 skin+HR", np.column_stack([S0, X_hr]), None),
         ("B5 skin+HR+env", np.column_stack([S0, X_hr, X_env]), None)]

print("AXIS 1 ...")
for name, X, inter in FEATS:
    run_loso("1 subject shift", "LOSO (participant-grouped)", name, X, inter)
    rng = np.random.default_rng(0)
    idx = rng.permutation(nR)
    folds = np.array_split(idx, 10)
    p2 = np.full(nR, np.nan)
    for fi, f in enumerate(folds):
        te = np.zeros(nR, bool)
        te[f] = True
        q = fit_eval(X, C, ~te, te, inter)
        p2[te] = q[te]
    scored = np.isfinite(p2)
    sd = pooled_sd(scored)
    for fi, f in enumerate(folds):
        te = np.zeros(nR, bool)
        te[f] = True
        te &= scored
        if te.sum() == 0:
            continue
        e = p2[te] - C[te]
        record("1 subject shift", "naive random row split", name, None, "random10", fi, te,
               float((e ** 2).sum()), int(te.sum()), sd)
    print(f"   {name} done  ({time.time() - t0:.0f}s)")

print("AXIS 2 ...")
for k in [1, 2, 3, 4, 6, 8, 11]:
    if k == 1:
        for nm in ["B2 single skin", "B3 multi-site skin", "B4 skin+HR", "B5 skin+HR+env"]:
            run_loso("2 sensor shift", f"k={k}", nm, X_single)
    else:
        for rep in range(5):
            rng = np.random.default_rng(100 + rep)
            cols = rng.choice(11, k, replace=False)
            for nm, X in [("B3 multi-site skin", S0[:, cols]),
                          ("B4 skin+HR", np.column_stack([S0[:, cols], X_hr])),
                          ("B5 skin+HR+env", np.column_stack([S0[:, cols], X_hr, X_env]))]:
                run_loso("2 sensor shift", f"k={k}", nm, X, rep=rep)
    print(f"   k={k} done  ({time.time() - t0:.0f}s)")

print("AXIS 3 ...")


def gap_mask(mins, seed):
    r = np.random.default_rng(seed)
    M = np.ones((nR, 11), bool)
    for c, idx in uidx.items():
        L = len(idx)
        if L <= mins + 2:
            continue
        st = int(r.integers(0, L - mins))
        M[idx[st:st + mins], :] = False
    return M


def carry_fill(S, M):
    out = S.copy()
    for j in range(S.shape[1]):
        mm = ~M[:, j]
        if not mm.sum():
            continue
        obs = np.where(M[:, j], S[:, j], np.nan)
        filled = np.full(nR, np.nan)
        for c, idx in uidx.items():
            filled[idx] = pd.Series(obs[idx]).ffill().values
        fb = np.nanmean(S[M[:, j], j])
        out[mm, j] = np.where(np.isfinite(filled), filled, fb)[mm]
    return out


for mins in [5, 20, 60]:
    M = gap_mask(mins, 400 + mins)
    R = carry_fill(S0, M)
    for nm, X in [("B3 multi-site skin", R), ("B4 skin+HR", np.column_stack([R, X_hr])),
                  ("B5 skin+HR+env", np.column_stack([R, X_hr, X_env]))]:
        run_loso("3 temporal corruption", f"gap={mins}min", nm, X)
    print(f"   gap={mins} done  ({time.time() - t0:.0f}s)")

print("AXIS 4 ...")
rng = np.random.default_rng(7)
for lab, Sx in [("noise sd=0.2", S0 + rng.normal(0, 0.2, S0.shape)),
                ("noise sd=0.5", S0 + rng.normal(0, 0.5, S0.shape)),
                ("noise sd=1.0", S0 + rng.normal(0, 1.0, S0.shape)),
                ("bias +0.3", S0 + 0.3), ("bias +0.5", S0 + 0.5)]:
    for nm, X in [("B3 multi-site skin", Sx), ("B4 skin+HR", np.column_stack([Sx, X_hr])),
                  ("B5 skin+HR+env", np.column_stack([Sx, X_hr, X_env]))]:
        run_loso("4 signal corruption", lab, nm, X)
    print(f"   {lab} done  ({time.time() - t0:.0f}s)")
ramp = np.zeros_like(S0)
for c, idx in uidx.items():
    ramp[idx] = np.linspace(0, 1.0, len(idx))[:, None]
for nm, X in [("B3 multi-site skin", S0 + ramp),
              ("B4 skin+HR", np.column_stack([S0 + ramp, X_hr])),
              ("B5 skin+HR+env", np.column_stack([S0 + ramp, X_hr, X_env]))]:
    run_loso("4 signal corruption", "drift 0->1C", nm, X)
print(f"   drift done  ({time.time() - t0:.0f}s)")

print("AXIS 5 ...")
for name, X, inter in FEATS:
    run_condition_heldout("5 condition shift", "leave-one-condition-out", name, X, inter)
print(f"   done  ({time.time() - t0:.0f}s)")

print("AXIS 6 ...")
warm = D.Temp.values >= 32.0
for lab, te_cond in [("unseen 29.6C (cond 9)", 9), ("unseen 25.0C (cond 6)", 6)]:
    test = COND == te_cond
    for name, X, inter in FEATS:
        p, scored = run_trainmask("6 environmental shift", lab, name, X, warm, test, inter)
        if scored.sum() < 20:
            continue
    print(f"   {lab} done  ({time.time() - t0:.0f}s)")

Rf = pd.DataFrame(FOLDS)
Rf.to_csv(os.path.join(P, "_twist_folds.csv"), index=False)
print(f"\nSAVED _twist_folds.csv  folds={len(Rf)}  ({time.time() - t0:.0f}s)")

# --------------------------------------------------------------------------- the reproduction check
frozen = pd.read_csv(os.path.join(P, "_twist_results.csv"))
frozen["baseline"] = frozen.baseline.replace({"B3 multi-site skin": "B3 multi-site skin"})
agg = (Rf.groupby(["axis", "level", "baseline"])
       .agg(n=("n", "sum"), sse=("sse", "sum")).reset_index())
agg["rmse_from_folds"] = np.sqrt(agg.sse / agg.n)

fz = frozen.groupby(["axis", "level", "baseline"]).agg(
    n=("n", "sum"), rmse=("rmse", "mean")).reset_index()
m = agg.merge(fz, on=["axis", "level", "baseline"], how="outer", indicator=True)
print(f"\nREPRODUCTION CHECK against the frozen board")
print(f"  cells in the frozen board : {len(fz)}")
print(f"  cells rebuilt from folds  : {len(agg)}")
print(f"  matched                   : {(m._merge == 'both').sum()}")
bad = m[(m._merge == "both") & (np.abs(m.rmse_from_folds - m.rmse) > 5e-4)]
print(f"  cells differing by >5e-4   : {len(bad)}")
if len(bad):
    print(bad[["axis", "level", "baseline", "rmse", "rmse_from_folds"]].to_string(index=False))
only_f = m[m._merge == "left_only"]
only_b = m[m._merge == "right_only"]
print(f"  only in folds: {len(only_f)}   only in board: {len(only_b)}")
if len(only_b):
    print(only_b[["axis", "level", "baseline"]].head(20).to_string(index=False))
print("\nTWIST FOLDS DONE")
