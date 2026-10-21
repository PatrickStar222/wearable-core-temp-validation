"""_flib.py -- shared pipeline for FINAL VALIDATION
(what must be preserved for downstream thermal-state inference?)

FROZEN, unchanged from the programme:
  * complete-case PROSPIE subset, 1-min sampling, LOSO by participant everywhere
  * scaler + imputation + model fitted on training subjects only
  * downstream estimator  =  RidgeCV(alphas=logspace(-2,4,25))   [EMPIRICAL OBSERVABILITY BASELINE]
  * two causal feature sets of the SAME frozen estimator:
        E_row  : per-row features                                  (memoryless)
        E_roll : per-row + causal rolling mean/sd over [t-w, t]    (state-aware)
    E_roll is an estimator-sensitivity check, not a new model: same RidgeCV, causal features only.
  * D3 (physics) kept ONLY as a negative control; not optimised, not renamed

CORRECTED IN THIS ROUND (self-caught bug in the previous round)
  * D2 ("generic data reconstruction") previously built its predictor from S[:, oth], i.e. the
    UNMASKED sibling channels.  It therefore reconstructed a missing value from sensors that were
    themselves missing in the degraded stream.  D2 was thus oracle-assisted and its pointwise
    fidelity was inflated.  D2 now sees only OBSERVED siblings, with per-channel missing
    indicators, and a training-fold mean for the unobserved ones.  It is a genuinely deployable
    spatial+environmental imputer.
"""
import os, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.preprocessing import StandardScaler

P = ""
SKIN = ["abdomen", "calf", "chest", "foot", "hand", "head",
        "lower arm", "lower back", "thigh", "upper arm", "upper back"]
ENV = ["Temp", "Humidity", "Wind"]
NS = len(SKIN)


# ----------------------------------------------------------------------------- data
def load():
    df = pd.read_pickle(os.path.join(P, "_prospie.pkl"))
    df["tmin"] = df.Time.apply(lambda t: t.hour * 60 + t.minute + t.second / 60 if hasattr(t, "hour") else np.nan)
    for c in SKIN + ["Corerectal", "Temp", "Humidity", "Wind", "HR"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").replace(9999, np.nan)
    df = df.sort_values(["Participant", "Condition", "tmin"]).reset_index(drop=True)
    df["unit"] = df.Participant.astype(str) + "|" + df.Condition.astype(str)
    need = SKIN + ["HR"] + ENV + ["Corerectal"]
    D = df.loc[df[need].notna().all(axis=1)].reset_index(drop=True)
    codes, uniq = pd.factorize(D.unit.values)
    uidx = {c: np.where(codes == c)[0] for c in range(len(uniq))}
    return dict(D=D, S=D[SKIN].values.astype(float), Y=D.Corerectal.values.astype(float),
                subj=D.Participant.values, tmin=D.tmin.values,
                codes=codes, uniq=list(uniq), uidx=uidx, nR=len(D))


# ----------------------------------------------------------------------------- masks
def m_spatial(U, k, seed):
    r = np.random.default_rng(seed)
    n = U["nR"]
    M = np.ones((n, NS), bool)
    if k > 0:
        for i in range(n):
            M[i, r.choice(NS, k, replace=False)] = False
    return M


def m_temporal(U, mins, seed):
    r = np.random.default_rng(seed)
    M = np.ones((U["nR"], NS), bool)
    for c, idx in U["uidx"].items():
        L = len(idx)
        if L <= mins + 2:
            continue
        st = int(r.integers(0, L - mins))
        M[idx[st:st + mins], :] = False
    return M


def m_mixed_source(U, seed=12):
    S = U["S"]
    r = np.random.default_rng(seed)
    Sc = S + r.normal(0, 0.15, S.shape)
    Sc[r.choice(U["nR"], U["nR"] // 2, replace=False)] += 0.3
    Sc = Sc + 0.5 * np.sin(2 * np.pi * np.arange(U["nR"]) / 600)[:, None]
    return Sc


COND_SPEC = [("drop1", "spatial", 1, 101), ("drop3", "spatial", 3, 103), ("drop6", "spatial", 6, 106),
             ("drop10", "spatial", 10, 110), ("temporal5", "temporal", 5, 205),
             ("temporal20", "temporal", 20, 220), ("mixed", "mixed", 3, 111),
             # mask-realisation replicates for the focal conditions
             ("drop6b", "spatial", 6, 906), ("drop10b", "spatial", 10, 910),
             ("temporal20b", "temporal", 20, 920)]


def make_cond(U, tag):
    for t, kind, k, sd in COND_SPEC:
        if t == tag:
            if kind == "spatial":
                return m_spatial(U, k, sd), U["S"]
            if kind == "temporal":
                return m_temporal(U, k, sd), U["S"]
            return m_spatial(U, k, sd), m_mixed_source(U)
    raise KeyError(tag)


# ----------------------------------------------------------------------------- repair methods
def _chan_mean(S, M, tr, j):
    m = tr & M[:, j]
    if m.sum() > 0:
        return float(np.nanmean(S[m, j]))
    return float(np.nanmean(S[:, j]))


def recon(U, Sin, M, tr, method):
    """returns a FULL nR x NS matrix: observed entries keep the (possibly corrupted) input value,
    missing entries are filled by `method`."""
    S = Sin
    n = U["nR"]
    out = S.copy()
    if method == "D0":                     # no repair: training-fold channel mean + missing indicator
        for j in range(NS):
            mm = ~M[:, j]
            if mm.sum():
                out[mm, j] = _chan_mean(S, M, tr, j)
    elif method == "D1":                   # carry-forward / persistence within unit
        for j in range(NS):
            mm = ~M[:, j]
            if not mm.sum():
                continue
            obs = np.where(M[:, j], S[:, j], np.nan)
            f = pd.Series(obs).groupby(U["codes"]).ffill().values
            fb = _chan_mean(S, M, tr, j)
            out[mm, j] = np.where(np.isfinite(f), f, fb)[mm]
    elif method == "D2":                   # generic data reconstruction (CORRECTED: observed siblings only)
        ENVD = U["D"][ENV].values
        mu = np.array([_chan_mean(S, M, tr, k) for k in range(NS)])
        Xfill = np.where(M, S, np.broadcast_to(mu, (n, NS)))
        IND = (~M).astype(float)
        for j in range(NS):
            mm = ~M[:, j]
            if not mm.sum():
                continue
            oth = [k for k in range(NS) if k != j]
            Xp = np.column_stack([Xfill[:, oth], IND[:, oth], ENVD, np.ones(n)])
            t2 = tr & (~mm)
            if t2.sum() < 200:
                out[mm, j] = mu[j]
                continue
            out[mm, j] = LinearRegression().fit(Xp[t2], S[t2, j]).predict(Xp[mm])
    elif method == "D3":                   # physics-constrained reconstruction -- NEGATIVE CONTROL
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x1 = np.nanmean(np.where(M, S, np.nan), axis=1)
            x2 = np.nanstd(np.where(M, S, np.nan), axis=1)
        t = U["D"].copy()
        t["_a"] = x1
        t["_b"] = x2
        a1 = t.groupby("unit")["_a"].shift(1).values
        b1 = t.groupby("unit")["_b"].shift(1).values
        Ul = np.column_stack([t.groupby("unit")[c].shift(1).values for c in ENV])
        ok = tr & np.isfinite(a1) & np.isfinite(b1) & np.isfinite(Ul).all(axis=1) & np.isfinite(x1) & np.isfinite(x2)
        if ok.sum() < 200:
            return recon(U, Sin, M, tr, "D1")
        Z = np.column_stack([a1, b1, Ul, np.ones(n)])
        c1, *_ = np.linalg.lstsq(Z[ok], x1[ok], rcond=None)
        c2, *_ = np.linalg.lstsq(Z[ok], x2[ok], rcond=None)
        x1h = x1.copy()
        x2h = x2.copy()
        for c, idx in U["uidx"].items():
            for i in idx[1:]:
                if np.isfinite(Ul[i]).all() and np.isfinite(x1h[i - 1]) and np.isfinite(x2h[i - 1]):
                    x1h[i] = c1[0] * x1h[i - 1] + c1[1] * x2h[i - 1] + c1[2:5] @ Ul[i] + c1[5]
                    x2h[i] = c2[0] * x1h[i - 1] + c2[1] * x2h[i - 1] + c2[2:5] @ Ul[i] + c2[5]
        for j in range(NS):
            mm = ~M[:, j]
            if not mm.sum():
                continue
            tj = tr & M[:, j] & np.isfinite(x1h) & np.isfinite(x2h)
            if tj.sum() < 200:
                out[mm, j] = _chan_mean(S, M, tr, j)
                continue
            W = np.column_stack([x1h[tj], x2h[tj], np.ones(tj.sum())])
            cf, *_ = np.linalg.lstsq(W, S[tj, j], rcond=None)
            out[mm, j] = np.column_stack([x1h[mm], x2h[mm], np.ones(mm.sum())]) @ cf
    else:
        raise KeyError(method)
    return np.asarray(out, float)


# ----------------------------------------------------------------------------- counterfactual surrogate fills
def fill_from(U, base, M, tr):
    """base fill of the missing entries: 'TRUE' (oracle) or 'D1' (carry-forward)."""
    if base == "TRUE":
        return U["S"].copy()
    return recon(U, U["S"], M, tr, "D1")


def f_perm(U, fill, M, seed):
    """T- : permute the fill values among the missing entries of each (unit, channel).
    Preserves the exact multiset / marginal distribution of the fill.  Destroys temporal order."""
    r = np.random.default_rng(seed)
    out = fill.copy()
    for j in range(NS):
        for c, idx in U["uidx"].items():
            mm = idx[~M[idx, j]]
            if len(mm) > 1:
                out[mm, j] = r.permutation(out[mm, j])
    return out


def f_shift_sd(U, fill, M, S, scale=1.0):
    """P- (fixed magnitude): add scale * sd(unit,channel) to the fill.  Preserves within-run order."""
    out = fill.copy()
    for j in range(NS):
        for c, idx in U["uidx"].items():
            mm = idx[~M[idx, j]]
            if len(mm) == 0:
                continue
            sd = float(np.nanstd(S[idx, j]))
            if sd > 1e-8:
                out[mm, j] = out[mm, j] + scale * sd
    return out


def _match_delta(err, target):
    """smallest delta on the increasing branch with mean|err+delta| == target (vectorised)"""
    d = np.asarray(err, float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return 0.0
    m = -float(np.median(d))
    span = max(10.0, abs(m) * 4 + 20.0)
    grid = np.linspace(m, m + span, 241)
    f = np.abs(d[None, :] + grid[:, None]).mean(axis=1)
    if target <= f[0]:
        return float(grid[0])
    if target >= f[-1]:
        return float(grid[-1])
    return float(np.interp(target, f, grid))


def f_shift_matched(U, fill, M, S, target_kind, seed=7):
    """P- (MATCHED): add a per-(unit,channel) constant to the fill so that its pointwise
    MAE (or RMSE) on the missing entries equals that of the permuted fill (T-).
    Equal pointwise error magnitude, different error structure -> the decisive contrast."""
    r = np.random.default_rng(seed)
    out = fill.copy()
    for j in range(NS):
        for c, idx in U["uidx"].items():
            mm = idx[~M[idx, j]]
            if len(mm) < 3:
                continue
            true = S[mm, j]
            tgt = out[mm, j][r.permutation(len(mm))] - true          # T- errors, same multiset
            target = float(np.sqrt(np.mean(tgt ** 2))) if target_kind == "rmse" else float(np.mean(np.abs(tgt)))
            d = out[mm, j] - true
            out[mm, j] = out[mm, j] + _match_delta(d, target)
    return out


def f_meanconst(U, fill, M, S):
    """level-only: every missing entry gets the mean of the true missing values (unbiased level,
    no dynamics at all)"""
    out = fill.copy()
    for j in range(NS):
        for c, idx in U["uidx"].items():
            mm = idx[~M[idx, j]]
            if len(mm) == 0:
                continue
            out[mm, j] = float(np.nanmean(S[mm, j]))
    return out


def f_draw(U, fill, M, S, seed=11):
    """deployable T- : i.i.d. resampling from the unit-channel observed values
    (marginal level preserved, temporal order destroyed)"""
    r = np.random.default_rng(seed)
    out = fill.copy()
    for j in range(NS):
        for c, idx in U["uidx"].items():
            mm = idx[~M[idx, j]]
            if len(mm) == 0:
                continue
            obs = S[idx[M[idx, j]], j]
            if len(obs) < 3:
                continue
            out[mm, j] = r.choice(obs, len(mm), replace=True)
    return out


# ----------------------------------------------------------------------------- downstream estimator
def roll_stats(X, U, w):
    """causal rolling mean / sd over [t-w, t], computed strictly within unit"""
    n = U["nR"]
    m = np.empty_like(X)
    s = np.empty_like(X)
    for c, idx in U["uidx"].items():
        x = X[idx]
        L = len(x)
        cs = np.cumsum(np.vstack([np.zeros((1, x.shape[1])), x]), axis=0)
        cs2 = np.cumsum(np.vstack([np.zeros((1, x.shape[1])), x * x]), axis=0)
        k = np.arange(1, L + 1)
        lo = np.maximum(0, k - w)
        cnt = (k - lo)[:, None]
        m[idx] = (cs[k] - cs[lo]) / cnt
        v = (cs2[k] - cs2[lo]) / cnt - m[idx] ** 2
        s[idx] = np.sqrt(np.maximum(v, 0.0))
    return m, s


def design(U, R, M, kind):
    D = U["D"]
    n = U["nR"]
    parts = [R, (~M).astype(float), D["HR"].values.reshape(-1, 1), D[ENV].values, np.ones((n, 1))]
    if kind == "roll":
        X = np.column_stack([R, D["HR"].values])
        m5, _ = roll_stats(X, U, 5)
        m15, s15 = roll_stats(X, U, 15)
        parts += [m5, m15, s15]
    return np.column_stack(parts)


def loso_eval(U, M, build, kind):
    """build(tr) -> full nR x NS reconstruction for that training split.
    Returns (reconstruction on held-out rows, LOSO prediction).
    Any non-finite entry of a reconstruction (e.g. a leading all-channel gap that the D3
    recursion cannot start from) falls back to the training-fold mean of that channel's
    OBSERVED values, so the downstream design matrix is always finite."""
    n = U["nR"]
    pred = np.full(n, np.nan)
    Rcol = np.full((n, NS), np.nan)
    n_guard = 0
    for s in np.unique(U["subj"]):
        te = U["subj"] == s
        tr = ~te
        if te.sum() < 20 or tr.sum() < 300:
            continue
        R_ = np.asarray(build(tr), float)
        if not np.isfinite(R_).all():
            n_guard += int((~np.isfinite(R_)).sum())
            fb = []
            for j in range(NS):
                m = tr & M[:, j]
                v = float(np.nanmean(R_[m, j])) if m.sum() > 0 else float(np.nanmean(R_[:, j]))
                fb.append(v if np.isfinite(v) else 0.0)
            R_ = np.where(np.isfinite(R_), R_, np.broadcast_to(np.array(fb), R_.shape))
        Rcol[te] = R_[te]
        X = design(U, R_, M, kind)
        sc = StandardScaler().fit(X[tr])
        m = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(sc.transform(X[tr]), U["Y"][tr])
        pred[te] = m.predict(sc.transform(X[te]))
    return Rcol, pred, n_guard


def pooled_row(U, M, Sref, R, pred, cond, method, kind):
    """pooled (row-level) summary, for continuity with the previous round's cell table"""
    pm = ~M
    e = (R - Sref)[pm]
    ok = np.isfinite(pred)
    pe = pred[ok] - U["Y"][ok]
    ss = np.sum((U["Y"][ok] - U["Y"][ok].mean()) ** 2)
    vr, dm, ace, dag, xlg = [], [], [], [], []
    t = U["tmin"]
    adj = np.ones(U["nR"], bool)
    for c, idx in U["uidx"].items():
        if len(idx) > 1:
            d = np.diff(t[idx]) > 1.5
            adj[idx[1:][d]] = False
    for j in range(NS):
        a = R[:, j]
        b = Sref[:, j]
        if np.nanstd(b) > 1e-8 and np.nanstd(a) > 1e-8:
            vr.append(float(np.nanvar(a) / np.nanvar(b)))
            x, y = _ac1(a), _ac1(b)
            if np.isfinite(x) and np.isfinite(y):
                ace.append(abs(x - y))
        da_ = np.diff(a)[adj[1:]]
        db_ = np.diff(b)[adj[1:]]
        if len(db_) > 2:
            dm.append(float(np.mean(np.abs(da_ - db_))))
            ok2 = np.abs(db_) > 0.01
            if ok2.sum() > 2:
                dag.append(float(np.mean(np.sign(da_[ok2]) == np.sign(db_[ok2]))))
    return dict(cond=cond, method=method, est=kind, n=U["nR"],
                pointMAE=float(np.mean(np.abs(e))), pointRMSE=float(np.sqrt(np.mean(e ** 2))),
                pointBias=float(np.mean(e)),
                rmse=float(np.sqrt(np.mean(pe ** 2))), mae=float(np.mean(np.abs(pe))),
                r2=float(1 - np.sum(pe ** 2) / ss) if ss > 1e-8 else np.nan,
                varRatio=float(np.mean(vr)) if vr else np.nan,
                derivMAE=float(np.mean(dm)) if dm else np.nan,
                ac1err=float(np.mean(ace)) if ace else np.nan,
                dirAgree=float(np.mean(dag)) if dag else np.nan)


# ----------------------------------------------------------------------------- unit-level metrics
def _ac1(x):
    x = np.asarray(x, float)
    if len(x) < 4 or np.nanstd(x) < 1e-8:
        return np.nan
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def unit_rows(U, M, Sref, R, pred, cond, method, kind):
    """One row per participant x condition unit: pointwise fidelity, dynamic fidelity, utility."""
    rows = []
    Y = U["Y"]
    for c, idx in U["uidx"].items():
        if len(idx) < 20:
            continue
        Mm = M[idx]
        S = Sref[idx]
        Rr = R[idx]
        pm = ~Mm
        if pm.sum() < 5:
            continue
        e = (Rr - S)[pm]
        # ---- A. pointwise fidelity
        pmae = float(np.mean(np.abs(e)))
        prmse = float(np.sqrt(np.mean(e ** 2)))
        pbias = float(np.mean(e))
        nj = []
        for j in range(NS):
            mj = pm[:, j]
            if mj.sum() < 3:
                continue
            sd = float(np.nanstd(S[:, j]))
            if sd > 1e-8:
                nj.append(float(np.mean(np.abs((Rr - S)[mj, j])) / sd))
        pmaen = float(np.mean(nj)) if nj else np.nan
        # ---- B. dynamic fidelity (simple, interpretable quantities only)
        t = U["tmin"][idx]
        dt = np.diff(t)
        adj = dt <= 1.5
        vr, dmae, ace, dag, slo, xlg, xmx = [], [], [], [], [], [], []
        for j in range(NS):
            a = Rr[:, j]
            b = S[:, j]
            if np.nanstd(b) > 1e-8 and np.nanstd(a) > 1e-8:
                vr.append(float(np.nanvar(a) / np.nanvar(b)))
                x = _ac1(a)
                y = _ac1(b)
                if np.isfinite(x) and np.isfinite(y):
                    ace.append(abs(x - y))
                try:
                    sa = np.polyfit(np.arange(len(a)), a, 1)[0]
                    sb = np.polyfit(np.arange(len(b)), b, 1)[0]
                    slo.append(abs(sa - sb))
                except Exception:
                    pass
            da_ = np.diff(a)[adj]
            db_ = np.diff(b)[adj]
            if len(db_) > 2:
                dmae.append(float(np.mean(np.abs(da_ - db_))))
                ok = np.abs(db_) > 0.01
                if ok.sum() > 2:
                    dag.append(float(np.mean(np.sign(da_[ok]) == np.sign(db_[ok]))))
            L = len(a)
            if L > 12 and np.nanstd(a) > 1e-8 and np.nanstd(b) > 1e-8:
                best = None
                for l in range(-5, 6):
                    if l >= 0:
                        xa, xb = a[0:L - l], b[l:L]
                    else:
                        xa, xb = a[-l:L], b[0:L + l]
                    if len(xa) > 8:
                        cc = np.corrcoef(xa, xb)[0, 1]
                        if np.isfinite(cc) and (best is None or cc > best[0]):
                            best = (cc, l)
                if best:
                    xmx.append(float(best[0]))
                    xlg.append(float(best[1]))
        # ---- C. downstream utility
        okp = np.isfinite(pred[idx])
        if okp.sum() < 10:
            continue
        y = Y[idx][okp]
        pe = pred[idx][okp] - y
        urmse = float(np.sqrt(np.mean(pe ** 2)))
        umae = float(np.mean(np.abs(pe)))
        ysd = float(np.std(y))
        ss = float(np.sum((y - y.mean()) ** 2))
        rows.append(dict(cond=cond, method=method, est=kind, unit=U["uniq"][c],
                         subj=int(U["D"].Participant.values[idx[0]]), n=len(idx),
                         frac_missing=float((~Mm).mean()),
                         pointMAE=pmae, pointRMSE=prmse, pointBias=pbias, pointMAEn=pmaen,
                         varRatio=float(np.mean(vr)) if vr else np.nan,
                         derivMAE=float(np.mean(dmae)) if dmae else np.nan,
                         ac1err=float(np.mean(ace)) if ace else np.nan,
                         dirAgree=float(np.mean(dag)) if dag else np.nan,
                         slopeErr=float(np.mean(slo)) if slo else np.nan,
                         xlag=float(np.mean(xlg)) if xlg else np.nan,
                         xmax=float(np.mean(xmx)) if xmx else np.nan,
                         rmse=urmse, mae=umae, nr=urmse / ysd if ysd > 1e-8 else np.nan,
                         r2=1 - np.sum(pe ** 2) / ss if ss > 1e-8 else np.nan))
    return rows


CF_VARIANTS = ["D0", "D1", "D2", "D3",
               "O_id", "O_perm", "O_draw", "O_mean", "O_shiftSD", "O_shiftMAE", "O_shiftRMSE",
               "C_perm", "C_shiftMAE", "C_shiftSD"]


def build_variant(U, M, Ssrc, variant, tr):
    """full nR x NS reconstruction for one counterfactual variant"""
    if variant in ("D0", "D1", "D2", "D3"):
        return recon(U, Ssrc, M, tr, variant)
    if variant.startswith("O_"):
        base = Ssrc                              # oracle: missing entries carry the truth
    elif variant.startswith("C_"):
        base = recon(U, Ssrc, M, tr, "D1")       # realistic base
    else:
        raise KeyError(variant)
    tail = variant[2:]
    if tail == "id":
        return base
    if tail == "perm":
        return f_perm(U, base, M, 500)
    if tail == "draw":
        return f_draw(U, base, M, Ssrc, 511)
    if tail == "mean":
        return f_meanconst(U, base, M, Ssrc)
    if tail == "shiftSD":
        return f_shift_sd(U, base, M, Ssrc, 1.0)
    if tail == "shiftMAE":
        return f_shift_matched(U, base, M, Ssrc, "mae")
    if tail == "shiftRMSE":
        return f_shift_matched(U, base, M, Ssrc, "rmse")
    raise KeyError(variant)


def append_csv(path, rows):
    if not rows:
        return
    d = pd.DataFrame(rows)
    hdr = not os.path.exists(path)
    d.to_csv(path, mode="a", header=hdr, index=False)


def done_keys(path, keys):
    if not os.path.exists(path):
        return set()
    try:
        d = pd.read_csv(path)
    except Exception:
        return set()
    if len(d) == 0 or any(k not in d.columns for k in keys):
        return set()
    return set(map(tuple, d[keys].astype(str).values.tolist()))


def probe(U):
    print(f"complete-case rows={U['nR']}  subjects={len(np.unique(U['subj']))}  units={len(U['uniq'])}")
    sizes = np.array([len(v) for v in U["uidx"].values()])
    print(f"units with >=20 rows: {(sizes>=20).sum()}   rows covered: {sizes[sizes>=20].sum()}")
    fr = []
    for c, idx in U["uidx"].items():
        if len(idx) < 3:
            continue
        dt = np.diff(U["tmin"][idx])
        fr.append(float(np.mean(dt <= 1.5)))
    print(f"adjacent retained pairs that are genuinely 1-min adjacent: mean={np.mean(fr):.4f}  min={np.min(fr):.4f}")
    for tag, kind, k, sd in COND_SPEC:
        M, _ = make_cond(U, tag)
        print(f"  {tag:<12} missing fraction = {(~M).mean():.4f}")


if __name__ == "__main__":
    probe(load())
