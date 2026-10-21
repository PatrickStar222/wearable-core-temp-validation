# -*- coding: utf-8 -*-
"""_drift_decomp.py -- attribution decomposition of the drift "improvement".

WHAT IT ANSWERS
---------------
The pooled nRMSE moves in the favourable direction by 3.0 / 1.7 / 2.4 % when a
within-record linear ramp 0 -> +1.0 C is added to all eleven skin channels
(B3 / B4 / B5).  The manuscript explains this mechanically ("the ramp adds
within-unit variance").  This script asks whether that explanation is SUFFICIENT:
how much of the drop survives once the re-weighting of cells inside the pooled
metric is removed?

POOLING SEMANTICS AND THE DECOMPOSITION  (verified in-script)
------------------------------------------------------------
_twist.py `score()` concatenates every fold's held-out predictions and reports ONE
nRMSE.  Verified on the frozen board: the denominator is the standard deviation of
the pooled scored rows (0.5536 C), identical for all 45 cells, so the board's
"nrmse" is a fixed linear rescaling of RMSE and relative changes are identical.

    nrmse^2 = MS / sigma^2 ,  MS = sum_u w_u m_u ,  sigma^2 = var of the pooled rows
    w_u = n_u / n  (per-unit row counts),  m_u = per-unit MSE

The unit row counts are the SAME in both conditions -- LOSO holds out whole subjects,
so no fold changes size -- hence w_u is NOT perturbed and the naive "cells were
re-weighted" story is identically zero.  What IS perturbed is sigma^2: the ramp adds
~0.083 C^2 to every unit's variance while the between-unit spread is unchanged, so
sigma^2 rises a little and every cell's nrmse^2 is deflated by the same factor.

    Delta(MS)  = sum_u w_u Delta m_u                       <- WITHIN-UNIT term
               + sum_u Delta w_u m_u^clean                 <- REWEIGHTING (cancellation)
    Delta(sigma^2) = sum_u Delta w_u s_u^2 ,  Delta w_u = (s_u^drift^2 - s_u^clean^2)/sigma_clean^2

    Delta(nrmse^2) ~= Delta(MS)/sigma_clean^2 - nrmse_clean^2 * Delta(sigma^2)/sigma_clean^2
                   = sum_u w_u Delta m_u / sigma_clean^2                      <- WITHIN-UNIT
                   + sum_u Delta w_u m_u^clean / sigma_clean^2               <- REWEIGHTING
                   + sum_u Delta w_u Delta m_u / sigma_clean^2               <- CROSS

The re-weighting term identifies itself: the weight GIVEN to a unit rises with that
unit's own variance increase, and a NEGATIVE re-weighting term means the pool improved
because the metric's denominator moved, not because any cell's error fell.  This is
exact in the limit Delta sigma^2 -> 0 (a 1e-4 relative shift here) and the residual is
reported.  The earlier per-unit-normalised basis (m_u/s_u^2) was tried and REJECTED:
it does not telescope to the pooled metric, because sum_u w_u m_u/s_u^2 is dominated by
units whose target barely moves inside the record.

Sign convention: the metric IMPROVES when Delta < 0.  A NEGATIVE reweighting term
is a cancellation contribution -- the pooled number improves without any cell
truly improving, purely because cells are re-weighted inside the pool.

THE SIGN-FLIP CONTROL
---------------------
Every unit gets the same ramp SHAPE (0 -> 1); only its SIGN is randomised. |ramp|
is unchanged, so the within-unit variance the ramp injects is IDENTICAL for every
draw -- what changes is only whether the ramp rises or falls WITH the target.  If
the improvement persists under random signs, it does not need ramp-target coupling;
if it vanishes, it does.

Usage:  python _drift_decomp.py            (writes _drift_decomp_units.csv)
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, P)
import _flib as F                                              # noqa: E402

ALPHAS = np.logspace(-2, 4, 25)
SKIN = F.SKIN
ENV = ["Temp", "Humidity", "Wind"]
N_FLIP = 10

# ------------------------------------------------------------------ data, verbatim from _twist.py
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
HRV = D.HR.values.reshape(-1, 1)
S0 = D[SKIN].values.astype(float)
nR = len(D)
UNIT = D.unit.values
UNITS = list(pd.unique(UNIT))
uidx = {c: np.where(UNIT == c)[0] for c in UNITS}
SUBJ_U = np.array([c.split("|")[0] for c in UNITS])
COND_U = np.array([c.split("|")[1] for c in UNITS])
LEN_U = np.array([len(uidx[c]) for c in UNITS])
print(f"complete-case: rows={nR} subjects={D.Participant.nunique()} "
      f"conditions={D.Condition.nunique()} UNITS={len(UNITS)}")

# ------------------------------------------------------------------ estimator (identical to _twist.py)
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


def loso(X, inter=None):
    """leave-one-subject-out; returns the full-length prediction vector"""
    p = np.full(nR, np.nan)
    for s in np.unique(SUBJ):
        te = SUBJ == s
        p[te] = fit_eval(X, C, ~te, te, inter)[te]
    return p


X_env, X_hr = ENVV, HRV


def best_single_site(train_mask, y):
    best, bi = np.inf, 0
    for j in range(S0.shape[1]):
        p = fit_eval(S0[:, [j]], y, train_mask, train_mask)
        r = np.sqrt(np.mean((p[train_mask] - y[train_mask]) ** 2))
        if r < best:
            best, bi = r, j
    return bi


j_best = best_single_site(np.ones(nR, bool), C)
X_single = S0[:, [j_best]]
print(f"single site chosen on training folds: {SKIN[j_best]}")

CONFIGS = [("B3 multi-site skin", lambda R: R),
           ("B4 skin+HR", lambda R: np.column_stack([R, X_hr])),
           ("B5 skin+HR+env", lambda R: np.column_stack([R, X_hr, X_env]))]

# ------------------------------------------------------------------ per-unit summary of a prediction vector
def unit_summary(p, ramp=None, sign=None):
    """per-unit n, mse, target variance, slope of target on row index, ramp slope"""
    out = []
    for k, c in enumerate(UNITS):
        i = uidx[c]
        y = C[i]
        e = p[i] - y
        mse = float(np.mean(e ** 2))
        var = float(np.var(y))
        n = len(i)
        r = np.arange(n, dtype=float)
        slope = float(np.polyfit(r, y, 1)[0]) if n >= 3 and np.std(y) > 0 else np.nan
        ramp_i = ramp[i, 0] if (ramp is not None and ramp.ndim == 2) else (
            ramp[i] if ramp is not None else None)
        rs = float(np.polyfit(r, ramp_i, 1)[0]) if (ramp_i is not None and n >= 3) else np.nan
        out.append(dict(unit=c, subject=c.split("|")[0], condition=c.split("|")[1],
                        n_rows=n, unit_mse=mse, unit_rmse=float(np.sqrt(mse)), unit_var=var,
                        unit_sd=float(np.sqrt(var)), target_slope=slope, ramp_slope=rs,
                        sign=float(sign[k]) if sign is not None else 1.0))
    return pd.DataFrame(out)


# ------------------------------------------------------------------ the decomposition
ZERO = dict(within=0.0, reweight=0.0, dw_cross=0.0, total=0.0)


def decompose(b, d, sig_b2, sig_d2):
    """additive decomposition of nrmse_drift^2 - nrmse_clean^2

    b, d       : per-unit frames from unit_summary() for the clean and corrupted runs.
                 Same UNITS order, so positionally aligned -- no merge (a merge would
                 cross-join the duplicated `n` column and silently rescale the weights).
    sig_b2/_d2 : the POOLED variance of the scored rows in each condition, which is the
                 denominator _twist.py's score() actually uses.  NOTE this is var of the
                 pooled rows, NOT the row-weighted mean of per-unit variances: the two
                 differ by the between-unit term (23.7 % of var(C) here).  Using the
                 within-unit part alone inflates nrmse by a factor 1.145.

    Returns components of Delta(nrmse^2); negative == improvement.
    """
    assert (b.unit.values == d.unit.values).all(), "unit order mismatch"
    n = b.n_rows.values.astype(float)
    assert (d.n_rows.values == b.n_rows.values).all(), "fold sizes changed between conditions"
    w = n / n.sum()
    mb, md = b.unit_mse.values, d.unit_mse.values
    dm = md - mb
    # Delta w_u: a unit's weight inside the pooled metric rises with its own share of the
    # pooled outcome variance.  The shifts telescope by construction.
    dw = (d.unit_var.values - b.unit_var.values) / sig_b2
    within = float((w * dm).sum()) / sig_b2
    reweight = float((dw * mb).sum()) / sig_b2
    cross = float((dw * dm).sum()) / sig_b2
    total = within + reweight + cross
    direct = float((w * md).sum() / sig_d2 - (w * mb).sum() / sig_b2)
    return dict(within=within, reweight=reweight, cross=cross, total=total,
                direct=direct, resid=total - direct,
                nrmse_b=float(np.sqrt((w * mb).sum() / sig_b2)),
                nrmse_d=float(np.sqrt((w * md).sum() / sig_d2)),
                sig_b=float(np.sqrt(sig_b2)), sig_d=float(np.sqrt(sig_d2)),
                dsig_rel=(sig_d2 - sig_b2) / sig_b2,
                d_ms=float((w * dm).sum()),
                sum_abs=float((w * np.abs(dm)).sum()) / sig_b2)




# ------------------------------------------------------------------ run: clean + drift + N sign-flips
print("\n" + "=" * 100)
print("RUNNING clean / drift / sign-flip draws")
print("=" * 100)

RAMP_SHAPE = np.zeros((nR, 1))
for c in UNITS:
    i = uidx[c]
    RAMP_SHAPE[i, 0] = np.linspace(0.0, 1.0, len(i))
print(f"ramp shape per unit: 0->1 always; mean added variance per unit = "
      f"{np.mean([np.var(RAMP_SHAPE[uidx[c], 0]) for c in UNITS]):.6f}")

# frozen board values, for the reproduction check
FROZEN = pd.read_csv(os.path.join(P, "_twist_results.csv"))
frozen_nrmse = {}
for nm, _ in CONFIGS:
    m = (FROZEN.axis == "4 signal corruption") & (FROZEN.level == "drift 0->1C") & (FROZEN.baseline == nm)
    frozen_nrmse[nm] = float(FROZEN.loc[m, "nrmse"].iloc[0])
# the clean counterpart of the drift cell, for the pre-flight check
_fz_clean = FROZEN[(FROZEN.axis == "4 signal corruption") & (FROZEN.level == "noise sd=0.2")
                   & (FROZEN.baseline == "B3 multi-site skin")]
assert len(_fz_clean) == 1
FROZEN_CLEAN_B3 = 0.6899312500253638      # _twist_results.csv, axis 1 LOSO / B3, frozen board

flip_rng = np.random.default_rng(20261007)
unit_signs = {r: flip_rng.choice([-1.0, 1.0], len(UNITS)) for r in range(N_FLIP)}

# pre-flight: the pooling arithmetic must reproduce the frozen board on the CLEAN
# cell before any drift number is trusted.  A merge that fans out silently changes
# the weight vector and this is the assertion that catches it.
_pb = loso(S0)
_b0 = unit_summary(_pb, RAMP_SHAPE)
_assert = decompose(_b0, _b0, np.var(C), np.var(C))
# the pooled denominator is the variance of the POOLED rows, which splits exactly into
# the row-weighted within-unit variance plus the row-weighted spread of unit means
_mu = np.array([C[uidx[c]].mean() for c in UNITS])
_w = _b0.n_rows.values / _b0.n_rows.values.sum()
_within = float((_w * _b0.unit_var.values).sum())
_between = float((_w * (_mu - C.mean()) ** 2).sum())
print(f"PRE-FLIGHT  clean B3: recomputed nrmse={_assert['nrmse_b']:.6f}  "
      f"board(clean)={FROZEN_CLEAN_B3:.6f}  "
      f"pandas-pooled={np.sqrt(np.mean((_pb - C) ** 2)) / np.std(C):.6f}  "
      f"sum(n)={int(_b0.n_rows.sum())} (expect {nR})")
print(f"PRE-FLIGHT  var(C)={np.var(C):.6f} = within {_within:.6f} + between {_between:.6f} "
      f"(between share {_between/np.var(C)*100:.1f} %)")
assert abs((_within + _between) - np.var(C)) < 1e-10, "variance split is not exact"
_pooled_mse = float(np.mean((_pb - C) ** 2))
_wmse = float((_b0.n_rows.values * _b0.unit_mse.values).sum() / _b0.n_rows.values.sum())
print(f"PRE-FLIGHT  pooled MSE={_pooled_mse:.8f}   n-weighted mean unit MSE={_wmse:.8f}   "
      f"diff={_wmse-_pooled_mse:+.3e}")
print(f"PRE-FLIGHT  unit n rows min={int(_b0.n_rows.min())} max={int(_b0.n_rows.max())} "
      f"sum={int(_b0.n_rows.sum())}   n_units={len(_b0)}")
assert abs(_assert["nrmse_b"] - np.sqrt(np.mean((_pb - C) ** 2)) / np.std(C)) < 1e-10, \
    "decomposition does not reproduce the pooled metric"
assert abs(_assert["nrmse_b"] - FROZEN_CLEAN_B3) < 5e-6, "clean cell mismatch"
assert int(_b0.n_rows.sum()) == nR, "unit row counts do not sum to the cohort"
print("PRE-FLIGHT OK -- the decomposition reproduces the frozen board\n")

PER_UNIT = []
SUMMARY = []

for nm, mk in CONFIGS:
    # ---- clean
    p_b = loso(mk(S0))
    b = unit_summary(p_b, RAMP_SHAPE)
    # ---- drift (all ramps +1: deterministic, matches the board)
    p_d = loso(mk(S0 + RAMP_SHAPE))
    d = unit_summary(p_d, RAMP_SHAPE)
    # the ramp is applied to the FEATURES, never to the target, so the pooled
    # denominator is the same in both conditions; measured, not assumed.
    sig_b2 = float(np.var(C[np.isfinite(p_b)]))
    sig_d2 = float(np.var(C[np.isfinite(p_d)]))
    assert abs(sig_d2 - sig_b2) < 1e-9, "denominator moved -- decomposition assumption broken"
    dec = decompose(b, d, sig_b2, sig_d2)
    board = frozen_nrmse[nm]
    print(f"\n[{nm}]  board nrmse(drift)={board:.6f}   recomputed={dec['nrmse_d']:.6f}   "
          f"diff={dec['nrmse_d']-board:+.2e}   "
          f"clean={dec['nrmse_b']:.6f}")
    print(f"   nrmse  {dec['nrmse_b']:.6f} -> {dec['nrmse_d']:.6f}   "
          f"rel change = {(dec['nrmse_d']/dec['nrmse_b']-1)*100:+.3f} %")
    print(f"   sigma  {dec['sig_b']:.6f} -> {dec['sig_d']:.6f}   "
          f"({dec['dsig_rel']*100:+.4f} % ; +1C ramp on 11 channels)")
    print(f"   Delta(nrmse^2) = {dec['total']:+.8f}   (direct {dec['direct']:+.8f}, "
          f"residual {dec['resid']:+.2e})")
    print(f"     within-unit          {dec['within']:+.8f}   ({dec['within']/dec['total']*100:6.2f} %)")
    print(f"     reweighting (canc.)  {dec['reweight']:+.8f}   ({dec['reweight']/dec['total']*100:6.2f} %)")
    print(f"     reweight x effect    {dec['cross']:+.8f}   ({dec['cross']/dec['total']*100:6.2f} %)")

    dd = d.merge(b, on="unit", suffixes=("_d", "_b"))
    assert len(dd) == len(UNITS), f"merge fanned out: {len(dd)} != {len(UNITS)}"
    dd["delta_mse"] = dd.unit_mse_d - dd.unit_mse_b
    dd["delta_nmse"] = dd.unit_mse_d / dec["sig_d"] ** 2 - dd.unit_mse_b / dec["sig_b"] ** 2
    dd["delta_var"] = dd.unit_var_d - dd.unit_var_b
    dd["w"] = dd.n_rows_d / dd.n_rows_d.sum()
    dd["ramp_sign"] = 1
    dd["baseline"] = nm
    dd["run"] = "drift(+1)"
    PER_UNIT.append(dd)

    imp = dd.delta_nmse < 0
    print(f"   units: {int(imp.sum())} improve / {int((~imp).sum())} degrade")
    print(f"     sum of improving deltas  {dd.loc[imp,'delta_nmse'].sum():+.6f}  "
          f"(n={int(imp.sum())}, total rows={int(dd.loc[imp,'n_rows_d'].sum())})")
    print(f"     sum of degrading deltas  {dd.loc[~imp,'delta_nmse'].sum():+.6f}  "
          f"(n={int((~imp).sum())}, total rows={int(dd.loc[~imp,'n_rows_d'].sum())})")

    SUMMARY.append(dict(baseline=nm, run="drift(+1)", rel_pct=(dec["nrmse_d"] / dec["nrmse_b"] - 1) * 100,
                        nrmse_clean=dec["nrmse_b"], nrmse_drift=dec["nrmse_d"],
                        d_nrmse_sq=dec["total"], within=dec["within"], reweight=dec["reweight"],
                        dw_cross=dec["cross"], sigma_b=dec["sig_b"], sigma_d=dec["sig_d"],
                        n_improve=int(imp.sum()), n_degrade=int((~imp).sum()),
                        sum_improve=dd.loc[imp, "delta_nmse"].sum(),
                        sum_degrade=dd.loc[~imp, "delta_nmse"].sum()))

    # ---- sign-flip control
    for r in range(N_FLIP):
        sg = unit_signs[r]
        sign_map = dict(zip(UNITS, sg))
        Rf = RAMP_SHAPE.copy()
        for c in UNITS:
            Rf[uidx[c]] *= sign_map[c]
        pf = loso(mk(S0 + Rf))
        dfu = unit_summary(pf, Rf, sign=np.array([sign_map[c] for c in UNITS]))
        decf = decompose(b, dfu, sig_b2, float(np.var(C[np.isfinite(pf)])))
        dd2 = dfu.merge(b, on="unit", suffixes=("_d", "_b"))
        assert len(dd2) == len(UNITS)
        dd2["delta_mse"] = dd2.unit_mse_d - dd2.unit_mse_b
        dd2["delta_nmse"] = dd2.unit_mse_d / decf["sig_d"] ** 2 - dd2.unit_mse_b / decf["sig_b"] ** 2
        dd2["delta_var"] = dd2.unit_var_d - dd2.unit_var_b
        dd2["w"] = dd2.n_rows_d / dd2.n_rows_d.sum()
        dd2["baseline"] = nm
        dd2["run"] = f"flip{r:02d}"
        PER_UNIT.append(dd2)
        SUMMARY.append(dict(baseline=nm, run=f"flip{r:02d}",
                            rel_pct=(decf["nrmse_d"] / decf["nrmse_b"] - 1) * 100,
                            nrmse_clean=decf["nrmse_b"], nrmse_drift=decf["nrmse_d"],
                            d_nrmse_sq=decf["total"], within=decf["within"],
                            reweight=decf["reweight"], dw_cross=decf["cross"],
                            sigma_b=decf["sig_b"], sigma_d=decf["sig_d"]))
    fl = [s for s in SUMMARY if s["baseline"] == nm and s["run"].startswith("flip")]
    rp = np.array([s["rel_pct"] for s in fl])
    print(f"   SIGN-FLIP control over {N_FLIP} draws: rel change % "
          f"mean {rp.mean():+.3f}  sd {rp.std(ddof=1):.3f}  "
          f"min {rp.min():+.3f}  max {rp.max():+.3f}  "
          f"| negative in {int((rp<0).sum())}/{N_FLIP}")
    print(f"   SIGN-FLIP  within {np.mean([s['within'] for s in fl]):+.6f}  "
          f"reweight {np.mean([s['reweight'] for s in fl]):+.6f}")

S = pd.DataFrame(SUMMARY)
U = pd.concat(PER_UNIT, ignore_index=True)
U.to_csv(os.path.join(P, "_drift_decomp_units.csv"), index=False)
S.to_csv(os.path.join(P, "_drift_decomp_summary.csv"), index=False)

# ------------------------------------------------------------------ the headline number
print("\n" + "=" * 100)
print("HEADLINE: how much of the 'improvement' is re-weighting (cancellation)?")
print("=" * 100)
print(f"{'config':<20} {'drop %':>8} {'within':>11} {'reweight':>11} {'rw x eff':>11} "
      f"{'rw share':>10} {'resid':>10}")
for _, r in S[S.run == "drift(+1)"].iterrows():
    share = r.reweight / r.d_nrmse_sq * 100
    print(f"{r.baseline:<20} {r.rel_pct:>+8.3f} {r.within:>+11.6f} {r.reweight:>+11.6f} "
          f"{r.dw_cross:>+11.6f} {share:>9.2f}% {r.within + r.reweight + r.dw_cross - r.d_nrmse_sq:>10.1e}")
print(f"\nSAVED _drift_decomp_units.csv (units x runs = {len(U)}) and _drift_decomp_summary.csv")
print(f"elapsed {time.time()-t0:.0f}s")
print("DRIFT DECOMPOSITION DONE")
