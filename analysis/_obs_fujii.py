"""_obs_fujii.py -- EXPERIMENT 2: Fujii (Zenodo 19835255).
Training = 20 C + 30 C.  Test = 10 C (unseen ambient).  Participant LOSO throughout.
Same model ladder, same fitting protocol, same preprocessing as Experiment 1.
"""
import os, sys, time
import numpy as np
import pandas as pd
import _obs_lib as L
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = ""
pd.set_option("display.width", 250)
CH = ["Infrared", "Internal thermister", "External thermister"]
MODELS = ["M0", "M0+", "M0f", "M0f+", "M1d", "M2d", "M1k", "M2k", "M1s", "M2s"]
STEP = 10.0  # seconds; 1 Hz data decimated to 10-s means

b = pd.read_pickle(os.path.join(P, "_fujii_lab.pkl"))
b = b.dropna(subset=["T_es"] + CH + ["amb"]).copy()
b["sess"] = (b["participant id"].astype(str) + "|" + b["day #"].astype(str) + "|" +
             b.amb.astype(int).astype(str) + "|" + b.time.dt.date.astype(str))
print(f"rows={len(b):,}  sessions={b.sess.nunique()}  participants={b['participant id'].nunique()}")
segs_raw = [g.sort_values("time").index.values for _, g in b.groupby("sess", sort=False)]

rows = []
for si, idx in enumerate(segs_raw):
    g = b.loc[idx]
    t = (g.time - g.time.iloc[0]).dt.total_seconds().values
    grid = np.arange(0, t[-1], STEP)
    if len(grid) < 12:
        continue
    out = {}
    for c in CH + ["T_es", "amb"]:
        out[c] = np.interp(grid, t, g[c].values.astype(float))
    out["seg"] = si
    out["subj"] = g["participant id"].iloc[0]
    out["amb"] = float(g.amb.iloc[0])
    rows.append(pd.DataFrame(out))
G = pd.concat(rows, ignore_index=True)
G["t"] = G.groupby("seg").cumcount()
Y = G[CH].values.astype(float); C = G.T_es.values.astype(float); U = G.amb.values.astype(float)
seg_list = [g.index.values for _, g in G.groupby("seg", sort=False)]
subj = G.subj.values; amb = G.amb.values
print(f"10-s grid: rows={len(G):,}  segments={len(seg_list)}  per-ambient rows:")
print(G.groupby("amb").size().to_string())
print(G.groupby(["amb"]).T_es.agg(["mean", "std"]).round(3).to_string())


def fit_all(tr):
    out = {}
    out["M0"], _ = L.run_m0(Y, U, None, C, tr, "ridge", interact=False)
    out["M0+"], _ = L.run_m0(Y, U, None, C, tr, "ridge", interact=True)
    out["M0f"], lm0 = L.run_leaky(out["M0"].copy(), C, seg_list, tr)
    out["M0f+"], lm1 = L.run_leaky(out["M0+"].copy(), C, seg_list, tr)
    out["M1d"], _ = L.run_direct(Y, U, C, seg_list, tr, m2=False)
    out["M2d"], _ = L.run_direct(Y, U, C, seg_list, tr, m2=True)
    out["M1k"], _ = L.run_state_space(Y, U, C, seg_list, tr, m2=False)
    out["M2k"], _ = L.run_state_space(Y, U, C, seg_list, tr, m2=True)
    out["M1s"] = L.static_inversion(Y, U, C, tr, m2=False)
    out["M2s"] = L.static_inversion(Y, U, C, tr, m2=True)
    return out, (lm0, lm1)


res, t0 = [], time.time()
print("\n" + "=" * 118)
print("EXPERIMENT 2a -- IN-DOMAIN: all ambients, leave one participant out")
print("=" * 118)
p_in = {m: np.full(len(C), np.nan) for m in MODELS}
for s in np.unique(subj):
    te = subj == s
    pr, _ = fit_all(~te)
    for m in MODELS:
        p_in[m][te] = pr[m][te]
for m in MODELS:
    sc = L.score(C, p_in[m]); res.append(dict(protocol="INDOMAIN", model=m, **sc))
    print(f"  {m:<5} RMSE={sc['rmse']:.4f}  MAE={sc['mae']:.4f}  bias={sc['bias']:+.4f}  R2={sc['r2']:+.4f}")

print("\n" + "=" * 118)
print("EXPERIMENT 2b -- UNSEEN AMBIENT: train on 20 + 30 C, test on 10 C, participant LOSO")
print("=" * 118)
p_cold = {m: np.full(len(C), np.nan) for m in MODELS}
lams = []
for s in np.unique(subj):
    tr = (subj != s) & (amb != 10.0)
    te = (subj == s) & (amb == 10.0)
    pr, lm = fit_all(tr)
    lams.append(lm)
    for m in MODELS:
        p_cold[m][te] = pr[m][te]
mask = amb == 10.0
for m in MODELS:
    sc = L.score(C, p_cold[m], mask); res.append(dict(protocol="COLD10", model=m, **sc))
    print(f"  {m:<5} RMSE={sc['rmse']:.4f}  MAE={sc['mae']:.4f}  bias={sc['bias']:+.4f}  R2={sc['r2']:+.4f}")
print(f"\n  fitted leaky smoothing weights (mean over folds): M0f={np.mean([x[0] for x in lams]):.3f}  "
      f"M0f+={np.mean([x[1] for x in lams]):.3f}")

print("\n" + "=" * 118)
print("EXPERIMENT 2c -- the same protocol for the other two held-out ambients (symmetry check)")
print("=" * 118)
for held in [20.0, 30.0]:
    p_h = {m: np.full(len(C), np.nan) for m in MODELS}
    for s in np.unique(subj):
        tr = (subj != s) & (amb != held)
        te = (subj == s) & (amb == held)
        pr, _ = fit_all(tr)
        for m in MODELS:
            p_h[m][te] = pr[m][te]
    mk = amb == held
    line = f"  held-out {held:.0f} C: "
    for m in MODELS:
        sc = L.score(C, p_h[m], mk); res.append(dict(protocol=f"HELD{int(held)}", model=m, **sc))
        line += f"{m}={sc['rmse']:.4f}  "
    print(line)

R = pd.DataFrame(res)
R.to_csv(os.path.join(P, "_obs_fujii.csv"), index=False)
np.save(os.path.join(P, "_obs_fujii_pred.npy"), {"in": p_in, "cold": p_cold}, allow_pickle=True)
G.to_pickle(os.path.join(P, "_obs_fujii_grid.pkl"))
print("\n  SUMMARY -- RMSE (in-domain vs unseen 10 C) and the OOD degradation ratio:")
tab = R.pivot_table(index="model", columns="protocol", values="rmse")
tab["OOD_degr_%"] = 100 * (tab["COLD10"] - tab["INDOMAIN"]) / tab["INDOMAIN"]
print(tab.round(4).to_string())
print(f"\n[{time.time()-t0:.0f}s] EXPERIMENT 2 DONE")
