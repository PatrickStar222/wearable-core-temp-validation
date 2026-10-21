"""_final_dyn.py -- dynamic-fidelity block recomputed in PHYSICAL TIME (minutes).

Why: the complete-case subset carries two sampling regimes -- participants 29-50 survive on a
1-minute grid, participants 1-18 (plus one unit of 38) on a 5-minute grid.  Dynamic metrics
defined on SAMPLE lags are therefore not commensurable between them, and the previous pass
silently reduced the regression to 21 of 38 subjects.  Everything here is in minutes:

  pmn        mean_j MAE_j / sd_j                      (scale-free pointwise fidelity)
  pointBias, pointMAE, pointRMSE
  varErr     mean_j |1 - var(rec_j)/var(true_j)|
  derivMAEn  mean over adjacent retained pairs with 1<=dt<=10 min of
             |dx_rec/dt - dx_true/dt|, averaged over channels and divided by the unit's mean
             channel sd  -> per-minute slope error, scale-free and regime-free
  dirAgree   sign agreement of dx over the same pairs (scale-free, time-free)
  ac1err     mean_j |ac1(rec_j) - ac1(true_j)| at the unit's own sampling interval
             (flagged: this lag is 1 min or 5 min depending on regime)
  xmax, xlagmin   max signed cross-correlation of rec vs true over lags -15..+15 MINUTES and
             the lag (minutes) at which it occurs

No model fitting is needed: the fidelity of a reconstruction does not depend on the downstream
estimator.  One pass per (condition, method); units come from the LOSO split of their own subject.
"""
import sys, os, time
import numpy as np
import pandas as pd
import _flib as F

P = F.P
U = F.load()
conds = sys.argv[1].split(",") if len(sys.argv) > 1 else [t for t, _, _, _ in F.COND_SPEC]
OUT = os.path.join(P, "_final_dyn.csv")
done = F.done_keys(OUT, ["cond", "method"])
t0 = time.time()
print(f"resume: {len(done)} cells done", flush=True)

# unit codes grouped by subject
by_subj = {}
for c, idx in U["uidx"].items():
    if len(idx) < 20:
        continue
    s = int(U["D"].Participant.values[idx[0]])
    by_subj.setdefault(s, []).append(c)
LAGS = [-15, -10, -5, -3, -2, -1, 0, 1, 2, 3, 5, 10, 15]


def unit_metrics(U, idx, M, Sref, R_):
    Mm = M[idx]
    S = Sref[idx]
    Rr = R_[idx]
    pm = ~Mm
    if pm.sum() < 5:
        return None
    e = (Rr - S)[pm]
    sd_skin = float(np.mean([np.std(S[:, j]) for j in range(F.NS)]))
    if sd_skin < 1e-8:
        return None
    nj, vr, ace = [], [], []
    for j in range(F.NS):
        mj = pm[:, j]
        sdj = float(np.std(S[:, j]))
        if mj.sum() >= 3 and sdj > 1e-8:
            nj.append(float(np.mean(np.abs((Rr - S)[mj, j])) / sdj))
        if np.std(S[:, j]) > 1e-8 and np.std(Rr[:, j]) > 1e-8:
            vr.append(float(np.var(Rr[:, j]) / np.var(S[:, j])))
        x, y = F._ac1(Rr[:, j]), F._ac1(S[:, j])
        if np.isfinite(x) and np.isfinite(y):
            ace.append(abs(x - y))
    t = U["tmin"][idx]
    dt = np.diff(t)
    okk = (dt >= 0.5) & (dt <= 10.0)
    if not okk.any():
        okk = dt >= 0.5
    dk = dt[okk]
    der, dag = [], []
    if len(dk) > 2:
        for j in range(F.NS):
            da = np.diff(Rr[:, j])[okk] / dk
            db = np.diff(S[:, j])[okk] / dk
            der.append(float(np.mean(np.abs(da - db))))
            sg = np.abs(db) > 0.002
            if sg.sum() > 2:
                dag.append(float(np.mean(np.sign(da[sg]) == np.sign(db[sg]))))
    tol = 0.6 * float(np.median(dt)) if len(dt) else 1.0
    xmx, xlg = [], []
    for j in range(F.NS):
        a, b = Rr[:, j], S[:, j]
        if np.std(a) < 1e-8 or np.std(b) < 1e-8:
            continue
        best = None
        for L in LAGS:
            tgt = t + L
            jj = np.searchsorted(t, tgt)
            jj = np.clip(jj, 0, len(t) - 1)
            prev = np.clip(jj - 1, 0, len(t) - 1)
            take_prev = np.abs(t[prev] - tgt) < np.abs(t[jj] - tgt)
            jj = np.where(take_prev, prev, jj)
            keep = np.abs(t[jj] - tgt) <= max(tol, 0.6)
            if keep.sum() > 8:
                cc = np.corrcoef(a[keep], b[jj[keep]])[0, 1]
                if np.isfinite(cc) and (best is None or cc > best[0]):
                    best = (cc, float(L))
        if best:
            xmx.append(best[0])
            xlg.append(best[1])
    return dict(pointMAE=float(np.mean(np.abs(e))), pointRMSE=float(np.sqrt(np.mean(e ** 2))),
                pointBias=float(np.mean(e)), pmn=float(np.mean(nj)) if nj else np.nan,
                sd_skin=sd_skin, dtmed=float(np.median(dt)) if len(dt) else np.nan,
                npair=int(okk.sum()),
                varErr=float(np.mean([abs(1 - v) for v in vr])) if vr else np.nan,
                derivMAEn=float(np.mean(der) / sd_skin) if der else np.nan,
                dirAgree=float(np.mean(dag)) if dag else np.nan,
                ac1err=float(np.mean(ace)) if ace else np.nan,
                xmax=float(np.mean(xmx)) if xmx else np.nan,
                xlagmin=float(np.mean(xlg)) if xlg else np.nan)


for tag in conds:
    M, Ssrc = F.make_cond(U, tag)
    for method in ["D0", "D1", "D2", "D3"]:
        if (tag, method) in done:
            continue
        rows = []
        for s, codes in by_subj.items():
            tr = U["subj"] != s
            if tr.sum() < 300:
                continue
            R_ = np.asarray(F.recon(U, Ssrc, M, tr, method), float)
            if not np.isfinite(R_).all():
                fb = []
                for j in range(F.NS):
                    m = tr & M[:, j]
                    v = float(np.nanmean(R_[m, j])) if m.sum() > 0 else float(np.nanmean(R_[:, j]))
                    fb.append(v if np.isfinite(v) else 0.0)
                R_ = np.where(np.isfinite(R_), R_, np.broadcast_to(np.array(fb), R_.shape))
            for c in codes:
                m = unit_metrics(U, U["uidx"][c], M, U["S"], R_)
                if m:
                    m.update(cond=tag, method=method, unit=U["uniq"][c], subj=s,
                             n=len(U["uidx"][c]))
                    rows.append(m)
        F.append_csv(OUT, rows)
        print(f"  [{time.time()-t0:7.1f}s] {tag:<12}{method:<4} units={len(rows)} "
              f"mean pmn={np.nanmean([r['pmn'] for r in rows]):.4f} "
              f"derivMAEn={np.nanmean([r['derivMAEn'] for r in rows]):.4f} "
              f"nonNaN_dyn={int(np.isfinite([r['derivMAEn'] for r in rows]).sum())}", flush=True)
print("DYN DONE")
