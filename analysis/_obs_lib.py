"""_obs_lib.py -- M0 / M1 / M2 for the environment-conditioned thermal-state observer experiment.

M0  fixed black-box predictor           T_core = g(y_wearable, u)                 (ridge / OLS)
M1  latent state, FIXED observation     x_{t+1} = f(x_t, u_t) ;  y_t = h(x_t) + v
M2  latent state, ENV-CONDITIONED obs.  x_{t+1} = f(x_t, u_t) ;  y_t = h(x_t, u_t, s) + v

The latent state IS the core thermal state, so all three models output the same quantity and are
directly comparable.  f and h are linear-Gaussian and estimated in closed form on the training set;
inference at test time is a scalar-state Kalman filter with a time-varying observation row H_t:

    M1 :  H_j(t) = a_j                      g_j(t) = b_j
    M2 :  H_j(t) = a_j + a2_j * u_t          g_j(t) = b_j + b2_j * u_t

"Same state dynamics, environment-conditioned observation mapping" holds by construction: f is
identical in form for M1 and M2 and never depends on the environment beyond its scalar u input;
only h gains the u-interaction.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

ALPHAS = np.logspace(-2, 4, 25)
RIDGE_TINY = 1e-8


# ------------------------------------------------------------------ data preparation
def make_segments(df, unit_col="unit", time_col="tmin", max_gap=3.0, min_len=15):
    """split each unit wherever the retained sampling grid breaks by more than max_gap minutes."""
    segs = []
    for u, idx in df.groupby(unit_col, sort=False).indices.items():
        t = df[time_col].values[idx]
        brk = np.where(np.diff(t) > max_gap)[0]
        starts = np.r_[0, brk + 1]
        ends = np.r_[brk + 1, len(idx)]
        for s, e in zip(starts, ends):
            if e - s >= min_len:
                segs.append(idx[s:e])
    return segs


def resample_1min(df, idx, cols, time_col="tmin"):
    t = df[time_col].values[idx].astype(float)
    g = np.arange(t[0], t[-1] + 1e-9, 1.0)
    out = np.column_stack([np.interp(g, t, df[c].values[idx].astype(float)) for c in cols])
    return out


# ------------------------------------------------------------------ fitting
def fit_h(Y, U, C, tr, m2):
    """observation mapping y_j = h_j(x, u).  Returns H-coefficients and noise variances."""
    k = Y.shape[1]
    A1 = np.zeros(k); A2 = np.zeros(k); B1 = np.zeros(k); B2 = np.zeros(k)
    R = np.zeros(k)
    for j in range(k):
        if m2:
            Z = np.column_stack([C, C * U, np.ones(len(C)), U])
            coef, *_ = np.linalg.lstsq(Z[tr], Y[tr, j], rcond=None)
            A1[j], A2[j], B1[j], B2[j] = coef
            res = Y[tr, j] - Z[tr] @ coef
        else:
            Z = np.column_stack([C, np.ones(len(C))])
            coef, *_ = np.linalg.lstsq(Z[tr], Y[tr, j], rcond=None)
            A1[j], B1[j] = coef
            res = Y[tr, j] - Z[tr] @ coef
        R[j] = max(float(np.var(res)), 1e-6)
    return A1, A2, B1, B2, R


def run_leaky(S, C, seg_list, tr, grid=np.arange(0.0, 0.99, 0.02)):
    """Leaky observer:  x_hat[t] = (1-lam) * s[t] + lam * x_hat[t-1].
    s = a static observation-based estimate.  lam is fitted by FREE-RUNNING error on the training
    segments (not teacher-forced), so it is the optimal amount of shared temporal dynamics to keep.
    lam = 0 recovers the static model exactly; lam > 0 is 'shared state dynamics'."""
    fit_segs = [i for i in seg_list if tr[i].all()]
    best, bestlam = None, 0.0
    for lam in grid:
        e = []
        for idx in fit_segs:
            x = None
            for t in idx:
                x = S[t] if x is None else (1 - lam) * S[t] + lam * x
                e.append(x - C[t])
        e = np.array(e)
        r = float(np.sqrt(np.mean(e ** 2)))
        if best is None or r < best:
            best, bestlam = r, float(lam)
    pred = np.full(len(C), np.nan)
    for idx in seg_list:
        x = None
        for t in idx:
            x = S[t] if x is None else (1 - bestlam) * S[t] + bestlam * x
            pred[t] = x
    return pred, bestlam


def fit_f(C, U, seg_list, tr):
    """state dynamics x_{t+1} = a*x_t + b*u_t + c over consecutive within-segment pairs."""
    xs, us, ys = [], [], []
    for idx in seg_list:
        m = tr[idx]
        if m.sum() < 3:
            continue
        mm = m[:-1] & m[1:]
        if mm.sum() == 0:
            continue
        xs.append(C[idx][:-1][mm]); us.append(U[idx][:-1][mm]); ys.append(C[idx][1:][mm])
    if not xs:
        return 1.0, 0.0, float(np.mean(C[tr])), float(np.var(C[tr]))
    X = np.concatenate(xs); Uu = np.concatenate(us); Yy = np.concatenate(ys)
    Z = np.column_stack([X, Uu, np.ones(len(X))])
    coef, *_ = np.linalg.lstsq(Z, Yy, rcond=None)
    res = Yy - Z @ coef
    Q = max(float(np.var(res)), 1e-6)
    return coef[0], coef[1], coef[2], Q


def kalman(Yu, Uu, A1, A2, B1, B2, R, a, b, c, Q, m0, P0, m2):
    n, k = Yu.shape
    Rm = np.diag(R)
    m, P = m0, P0
    out = np.empty(n)
    for t in range(n):
        if t > 0:
            m = a * m + b * Uu[t - 1] + c
            P = a * a * P + Q
        u = Uu[t]
        H = (A1 + A2 * u) if m2 else A1
        g = (B1 + B2 * u) if m2 else B1
        S = P * np.outer(H, H) + Rm
        K = P * H @ np.linalg.inv(S)              # 1 x k
        m = m + K @ (Yu[t] - H * m - g)
        P = (1.0 - K @ H) * P
        P = max(P, 1e-9)
        out[t] = m
    return out


def run_state_space(Y, U, C, seg_list, tr, m2, warmup=0, ss_init=True):
    """fit f and h on tr, then filter every segment and return per-row predictions.
    ss_init=True initialises the filter at the model's own steady-state uncertainty (Riccati fixed
    point) instead of the pooled core variance, otherwise the first noisy observation dominates."""
    A1, A2, B1, B2, R = fit_h(Y, U, C, tr, m2)
    a, b, c, Q = fit_f(C, U, seg_list, tr)
    m0 = float(np.mean(C[tr]))
    if ss_init:
        P = float(np.var(C[tr]))
        ub = float(np.mean(U[tr]))
        H = (A1 + A2 * ub) if m2 else A1
        for _ in range(500):
            S = P * np.outer(H, H) + np.diag(R)
            K = P * H @ np.linalg.inv(S)
            P = (1.0 - K @ H) * P
            P = a * a * P + Q
        P0 = float(P)
    else:
        P0 = float(np.var(C[tr]))
    pred = np.full(len(C), np.nan)
    for idx in seg_list:
        p = kalman(Y[idx], U[idx], A1, A2, B1, B2, R, a, b, c, Q, m0, P0, m2)
        if warmup:
            p[:warmup] = np.nan
        pred[idx] = p
    return pred, dict(a=a, b=b, c=c, Q=Q, A1=A1, A2=A2, B1=B1, B2=B2, R=R, P0=P0)


def run_m0(Y, U, HR, C, tr, mode="ridge", interact=False):
    """static black box.  interact=True adds y x u products (black box WITH environment conditioning)."""
    blocks = [Y, U.reshape(-1, 1)]
    if HR is not None:
        blocks.append(HR.reshape(-1, 1))
    if interact:
        blocks.append(Y * U.reshape(-1, 1))
    X = np.column_stack(blocks)
    if mode == "ridge":
        sc = StandardScaler().fit(X[tr])
        m = RidgeCV(alphas=ALPHAS).fit(sc.transform(X[tr]), C[tr])
        return m.predict(sc.transform(X)), m
    Z = np.column_stack([X, np.ones(len(X))])
    coef, *_ = np.linalg.lstsq(Z[tr], C[tr], rcond=None)
    return Z @ coef, coef


def prev_index(seg_list, n):
    prev = np.full(n, -1)
    for idx in seg_list:
        prev[idx[1:]] = idx[:-1]
    return prev


def run_direct(Y, U, C, seg_list, tr, m2, use_hr=None):
    """REDUCED-FORM OBSERVER, fitted in one step (no two-stage inversion):
         x_hat[t] = a*x_hat[t-1] + b*u[t] + sum_j c_j(u[t]) * y_j[t] + d
       with the SAME dynamics terms (a, b) in M1 and M2; only the observation gains gain a u-interaction.
       Fitted with the TRUE lagged core (teacher forcing), run free at test time."""
    n = len(C)
    prev = prev_index(seg_list, n)
    valid = prev >= 0
    Xs = [C[prev], U, Y]
    if use_hr is not None:
        Xs.append(use_hr)
    if m2:
        Xs.append(Y * U.reshape(-1, 1))
    X = np.column_stack(Xs)
    fit = valid & tr
    sc = StandardScaler().fit(X[fit])
    mdl = RidgeCV(alphas=ALPHAS).fit(sc.transform(X[fit]), C[fit])
    m0 = float(np.mean(C[tr]))
    pred = np.full(n, np.nan)
    for idx in seg_list:
        x = m0
        for t in idx:
            if prev[t] < 0:
                pred[t] = x
                x = pred[t]
                continue
            Xt = [np.array([x]), np.array([U[t]]), Y[t]]
            if use_hr is not None:
                Xt.append(np.array([use_hr[t]]))
            if m2:
                Xt.append(Y[t] * U[t])
            row = np.concatenate(Xt).reshape(1, -1)
            x = float(mdl.predict(sc.transform(row))[0])
            pred[t] = x
    return pred, mdl


def static_inversion(Y, U, C, tr, m2):
    """diagnostic: weighted least-squares inversion of h at each timestep, NO dynamics recursion."""
    A1, A2, B1, B2, R = fit_h(Y, U, C, tr, m2)
    H = (A1[None, :] + A2[None, :] * U[:, None]) if m2 else np.broadcast_to(A1, Y.shape)
    Gc = (B1[None, :] + B2[None, :] * U[:, None]) if m2 else np.broadcast_to(B1, Y.shape)
    w = 1.0 / R[None, :]
    return np.sum(w * H * (Y - Gc), axis=1) / np.sum(w * H * H, axis=1)


# ------------------------------------------------------------------ scoring
def score(C, p, mask=None):
    if mask is None:
        mask = np.isfinite(p)
    else:
        mask = mask & np.isfinite(p)
    y = C[mask]; e = p[mask] - y
    ss = float(np.sum((y - y.mean()) ** 2))
    return dict(n=int(mask.sum()), rmse=float(np.sqrt(np.mean(e ** 2))), mae=float(np.mean(np.abs(e))),
                bias=float(np.mean(e)), r2=float(1 - np.sum(e ** 2) / ss) if ss > 1e-9 else np.nan)
