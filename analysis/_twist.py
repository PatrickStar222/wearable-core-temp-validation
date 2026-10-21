"""_twist.py -- Thermal Wearable Inference Stress Test.

A standardised evaluation protocol, not a model.  Five baselines, one estimator form, seven axes.
Every cell is leave-one-subject-out unless the axis says otherwise; scaler and coefficients are fitted
on training rows only.

Baselines (all `RidgeCV(alphas=logspace(-2,4,25))` on standardised features):
  B1 environment only            [Tamb, RH, wind]
  B2 single skin site            best single site chosen on the training folds only
  B3 multi-site skin             11 sites
  B4 multi-site skin + HR
  B5 multi-site skin + HR + env  (the conventional full model)

Axes: 1 subject shift | 2 sensor shift | 3 temporal corruption | 4 signal corruption
      5 condition shift | 6 environmental shift | 7 protocol validity
"""
import os, sys, time
import numpy as np
import pandas as pd
import _flib as F
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = ""
pd.set_option("display.width", 260)
ALPHAS = np.logspace(-2, 4, 25)
SKIN = F.SKIN
ENV = ["Temp", "Humidity", "Wind"]

df = pd.read_pickle(os.path.join(P, "_prospie.pkl"))
df["tmin"] = df.Time.apply(lambda t: t.hour * 60 + t.minute + t.second / 60 if hasattr(t, "hour") else np.nan)
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
print(f"PROSPIE complete-case: rows={nR} subjects={D.Participant.nunique()} conditions={D.Condition.nunique()}")


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


def score(y, p, mask=None):
    if mask is None:
        mask = np.isfinite(p)
    else:
        mask = mask & np.isfinite(p)
    yy, e = y[mask], p[mask] - y[mask]
    ss = float(np.sum((yy - yy.mean()) ** 2))
    ysd = float(np.std(yy))
    return dict(n=int(mask.sum()), rmse=float(np.sqrt(np.mean(e ** 2))),
                nrmse=float(np.sqrt(np.mean(e ** 2)) / ysd) if ysd > 1e-9 else np.nan,
                mae=float(np.mean(np.abs(e))), bias=float(np.mean(e)),
                r2=float(1 - np.sum(e ** 2) / ss) if ss > 1e-9 else np.nan)


def loso(X, y, inter=None, seed=0):
    """leave-one-subject-out; returns prediction vector"""
    p = np.full(len(y), np.nan)
    for s in np.unique(SUBJ):
        te = SUBJ == s
        q = fit_eval(X, y, ~te, te, inter)
        p[te] = q[te]
    return p


def loso_condition_heldout(X, y, inter=None):
    """one pass per held-out condition; returns prediction vector"""
    p = np.full(len(y), np.nan)
    for c in np.unique(COND):
        te = COND == c
        q = fit_eval(X, y, ~te, te, inter)
        p[te] = q[te]
    return p


def loso_trainmask(X, y, train_mask, test_mask, inter=None):
    """LOSO restricted to a training subpopulation, evaluated on a test subpopulation"""
    p = np.full(len(y), np.nan)
    for s in np.unique(SUBJ[test_mask]):
        te = test_mask & (SUBJ == s)
        tr = train_mask & (SUBJ != s)
        if tr.sum() < 200 or te.sum() < 5:
            continue
        q = fit_eval(X, y, tr, te, inter)
        p[te] = q[te]
    return p


def best_single_site(train_mask, y):
    """choose the single skin site on the training folds only (no test leakage)"""
    best, bi = np.inf, 0
    for j in range(S0.shape[1]):
        X = S0[:, [j]]
        p = fit_eval(X, y, train_mask, train_mask)
        r = np.sqrt(np.mean((p[train_mask] - y[train_mask]) ** 2))
        if r < best:
            best, bi = r, j
    return bi


RESULTS = []


def cell(axis, level, baseline, y, p, mask=None, extra=None):
    s = score(y, p, mask)
    s.update(axis=axis, level=level, baseline=baseline)
    if extra:
        s.update(extra)
    s = pd.Series(s)
    RESULTS.append(s)
    return s


# ---------------------------------------------------------------- baseline feature sets
X_env = ENVV
X_hr = HRV
print("\n" + "=" * 118)
print("AXIS 1 -- SUBJECT SHIFT: participant-grouped (LOSO) versus naive random row split")
print("=" * 118)
# single-site choice on all training data for a stable feature set
j_best = best_single_site(np.ones(nR, bool), C)
X_single = S0[:, [j_best]]
print(f"  single site chosen on training folds: {SKIN[j_best]}")
FEATS = [("B1 env only", X_env, None), ("B2 single skin", X_single, None),
         ("B3 multi-site skin", S0, None), ("B4 skin+HR", np.column_stack([S0, X_hr]), None),
         ("B5 skin+HR+env", np.column_stack([S0, X_hr, X_env]), None)]

for name, X, inter in FEATS:
    p = loso(X, C, inter)
    s1 = cell("1 subject shift", "LOSO (participant-grouped)", name, C, p)
    rng = np.random.default_rng(0)
    idx = rng.permutation(nR)
    folds = np.array_split(idx, 10)
    p2 = np.full(nR, np.nan)
    for f in folds:
        te = np.zeros(nR, bool); te[f] = True
        q = fit_eval(X, C, ~te, te, inter)
        p2[te] = q[te]
    s2 = cell("1 subject shift", "naive random row split", name, C, p2)
    print(f"  {name:<20} LOSO RMSE={s1['rmse']:.4f} R2={s1['r2']:+.4f}  |  "
          f"naive RMSE={s2['rmse']:.4f} R2={s2['r2']:+.4f}  |  dR2={s2['r2']-s1['r2']:+.4f}")

print("\n" + "=" * 118)
print("AXIS 2 -- SENSOR SHIFT: skin-only baselines with k sites (random subsets, 5 draws)")
print("=" * 118)
for k in [1, 2, 3, 4, 6, 8, 11]:
    if k == 1:
        X = X_single
        p = loso(X, C)
        for nm in ["B2 single skin", "B3 multi-site skin", "B4 skin+HR", "B5 skin+HR+env"]:
            cell("2 sensor shift", f"k={k}", nm, C, p)
        print(f"  k={k:<3} (preselected single site) RMSE={score(C,p)['rmse']:.4f} "
              f"nRMSE={score(C,p)['nrmse']:.4f} R2={score(C,p)['r2']:+.4f}")
        continue
    accs = []
    for rep in range(5):
        rng = np.random.default_rng(100 + rep)
        cols = rng.choice(11, k, replace=False)
        for nm, X in [("B3 multi-site skin", S0[:, cols]),
                      ("B4 skin+HR", np.column_stack([S0[:, cols], X_hr])),
                      ("B5 skin+HR+env", np.column_stack([S0[:, cols], X_hr, X_env]))]:
            p = loso(X, C)
            accs.append(cell("2 sensor shift", f"k={k}", nm, C, p, extra=dict(rep=rep)))
    sub = pd.DataFrame(accs)
    print(f"  k={k:<3} " + "  ".join(
        f"{nm.split()[0]} RMSE={g.rmse.mean():.4f}" for nm, g in sub.groupby("baseline")))

print("\n" + "=" * 118)
print("AXIS 3 -- TEMPORAL CORRUPTION: contiguous all-site gaps (carry-forward repair + indicators)")
print("=" * 118)


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
    """carry-forward within unit with a training-fold-free global fallback (the D1 repair, inlined)"""
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
    ind = (~M).astype(float)
    for nm, X in [("B3 multi-site skin", R), ("B4 skin+HR", np.column_stack([R, X_hr])),
                  ("B5 skin+HR+env", np.column_stack([R, X_hr, X_env]))]:
        p = loso(X, C)
        s = cell("3 temporal corruption", f"gap={mins}min", nm, C, p)
    print(f"  gap={mins:<3} min  " + "  ".join(
        f"{r.baseline.split()[0]}={r.rmse:.4f}" for r in RESULTS[-3:]))

print("\n" + "=" * 118)
print("AXIS 4 -- SIGNAL CORRUPTION: noise / bias / drift on the skin channels")
print("=" * 118)
rng = np.random.default_rng(7)
for lab, Sx in [("noise sd=0.2", S0 + rng.normal(0, 0.2, S0.shape)),
                ("noise sd=0.5", S0 + rng.normal(0, 0.5, S0.shape)),
                ("noise sd=1.0", S0 + rng.normal(0, 1.0, S0.shape)),
                ("bias +0.3", S0 + 0.3), ("bias +0.5", S0 + 0.5)]:
    for nm, X in [("B3 multi-site skin", Sx), ("B4 skin+HR", np.column_stack([Sx, X_hr])),
                  ("B5 skin+HR+env", np.column_stack([Sx, X_hr, X_env]))]:
        p = loso(X, C)
        cell("4 signal corruption", lab, nm, C, p)
    print(f"  {lab:<14} " + "  ".join(f"{r.baseline.split()[0]}={r.rmse:.4f}" for r in RESULTS[-3:]))
ramp = np.zeros_like(S0)
for c, idx in uidx.items():
    L = len(idx)
    ramp[idx] = np.linspace(0, 1.0, L)[:, None]
for nm, X in [("B3 multi-site skin", S0 + ramp), ("B4 skin+HR", np.column_stack([S0 + ramp, X_hr])),
              ("B5 skin+HR+env", np.column_stack([S0 + ramp, X_hr, X_env]))]:
    p = loso(X, C)
    cell("4 signal corruption", "drift 0->1C", nm, C, p)
print(f"  {'drift 0->1C':<14} " + "  ".join(f"{r.baseline.split()[0]}={r.rmse:.4f}" for r in RESULTS[-3:]))

print("\n" + "=" * 118)
print("AXIS 5 -- CONDITION SHIFT: leave-one-condition-out")
print("=" * 118)
for name, X, inter in FEATS:
    p = loso_condition_heldout(X, C, inter)
    cell("5 condition shift", "leave-one-condition-out", name, C, p)
    s = score(C, p)
    print(f"  {name:<20} RMSE={s['rmse']:.4f} R2={s['r2']:+.4f}")

print("\n" + "=" * 118)
print("AXIS 6 -- ENVIRONMENTAL SHIFT: train on warm conditions, test on unseen cooler ambient")
print("=" * 118)
warm = D.Temp.values >= 32.0
for lab, te_cond in [("unseen 29.6C (cond 9)", 9), ("unseen 25.0C (cond 6)", 6)]:
    test = COND == te_cond
    for name, X, inter in FEATS:
        p = loso_trainmask(X, C, warm, test, inter)
        m = test & np.isfinite(p)
        if m.sum() < 20:
            continue
        s = cell("6 environmental shift", lab, name, C, p, mask=test)
    print(f"  {lab:<24} " + "  ".join(
        f"{r.baseline.split()[0]}={r.rmse:.4f}" for r in RESULTS[-5:]))
    print(f"      subjects={D.Participant.values[test].astype(int).tolist()[:0] or len(np.unique(SUBJ[test]))}"
          f"  rows={int(test.sum())}  sd(core)={C[test].std():.4f}")

print("\n" + "=" * 118)
print("AXIS 7 -- PROTOCOL VALIDITY diagnostic (not an accuracy axis)")
print("=" * 118)
print("  (a) target-variance dependence of R2 (Trap 2): per condition, sd(core) vs achievable R2")
for c in sorted(np.unique(COND)):
    m = COND == c
    p = loso(X_env if False else S0, C)
    rr = score(C, p, m)
    print(f"      cond {c:<3} n={int(m.sum()):<5} sd(core)={C[m].std():.4f}  "
          f"R2={rr['r2']:+.4f}  RMSE={rr['rmse']:.4f}")
print("  (b) estimator dependence (Trap 3): environment-only under ridge vs OLS")
from sklearn.linear_model import LinearRegression
for lab, mk in [("ridge", lambda: RidgeCV(alphas=ALPHAS)), ("OLS", lambda: LinearRegression())]:
    p = np.full(nR, np.nan)
    for s in np.unique(SUBJ):
        te = SUBJ == s
        sc = StandardScaler().fit(X_env[~te])
        m = mk().fit(sc.transform(X_env[~te]), C[~te])
        p[te] = m.predict(sc.transform(X_env[te]))
    rr = score(C, p)
    print(f"      env-only {lab:<6} RMSE={rr['rmse']:.4f} nRMSE={rr['nrmse']:.4f} R2={rr['r2']:+.4f}")
print("  (c) ambient x session-order collinearity: see FINAL_FALSIFICATION_GATE_RESULTS.md")
print("      (source protocol repeated the same ambient order on both days for 10/12 participants)")

R = pd.DataFrame(RESULTS)
R.to_csv(os.path.join(P, "_twist_results.csv"), index=False)
print(f"\nSAVED _twist_results.csv  cells={len(R)}")
print("TWIST DONE")
