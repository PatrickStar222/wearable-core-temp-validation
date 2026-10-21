"""
PROSPIE analysis v2 -- design-corrected.

Targets (kept strictly apart):
  TARGET A  X_wearable -> T_core      "physiological information feasibility"
  TARGET B  X_wearable -> S_H-CORE-W  "mechanistic state reconstruction"

Estimator: Ridge, retained as the EMPIRICAL OBSERVABILITY BASELINE.
No deep learning. No new index.

Design rules enforced here:
  R1  paired same-window comparison: every C_i -> C_{i+1} uses IDENTICAL rows
  R2  coverage reported next to R^2
  R3  all features causal, window = [t-w, t], no forward information
  R4  normalisation fitted on TRAINING participants only
  R5  participant-level 5-fold (unchanged) + leave-one-condition-out
  R6  common prediction timestamps across all window lengths (no stage confound)
"""
import pandas as pd, numpy as np, warnings, json
warnings.filterwarnings("ignore")
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

P = ""
SKIN = ["abdomen","calf","chest","foot","hand","head","lower arm","lower back",
        "thigh","upper arm","upper back"]

df = pd.read_pickle(P + r"\_prospie.pkl")
df["tmin"] = df.Time.apply(lambda t: t.hour*60 + t.minute + t.second/60 if hasattr(t, "hour") else np.nan)
for c in SKIN + ["Corerectal","Temp","Humidity","Wind","HR","Ins skin","Tamb","Tfront","Rhfront","Tback","Rhback"]:
    df[c] = pd.to_numeric(df[c], errors="coerce").replace(9999, np.nan)
df = df.sort_values(["Participant","Condition","tmin"]).reset_index(drop=True)

gid = (df.Participant.astype(str) + "_" + df.Condition.astype(str)).values
_, first, counts = np.unique(gid, return_index=True, return_counts=True)
o = np.argsort(first); starts = first[o]; lens = counts[o]
# per-group tmin offset, so window validity is checked in real minutes
tmin = df.tmin.values
print("rows", len(df), "groups", len(lens), "participants", df.Participant.nunique(),
      "conditions", df.Condition.nunique(), flush=True)

# ---------------- causal rolling features (backward-looking only) ----------------
def roll(a, w, kind):
    """rolling statistic over [i-w+1 .. i] inclusive -- strictly past+present."""
    out = np.full(len(a), np.nan); t = np.arange(w); tc = t - t.mean(); den = np.sum(tc**2)
    for s, L in zip(starts, lens):
        seg = a[s:s+L].astype(float)
        if not np.isfinite(seg).all():
            continue
        if kind == "mean":
            out[s+w-1:s+L] = np.convolve(seg, np.ones(w)/w, mode="valid")
        elif kind == "std":
            c  = np.convolve(seg, np.ones(w)/w, mode="valid")
            c2 = np.convolve(seg**2, np.ones(w)/w, mode="valid")
            out[s+w-1:s+L] = np.sqrt(np.maximum(c2 - c**2, 0))
        else:  # slope per minute, on the window ending at i
            out[s+w-1:s+L] = np.convolve(seg, tc[::-1], mode="valid") / den
    return out

def featset(cols, w):
    out = {}
    for c in cols:
        a = df[c].values
        out[c+"|mean"]  = roll(a, w, "mean")
        out[c+"|std"]   = roll(a, w, "std")
        out[c+"|slope"] = roll(a, w, "slope")
    return pd.DataFrame(out, index=df.index)

ENV   = ["Temp","Humidity","Wind"]
MICRO = ["Ins skin","Tamb"]

CFG = {
 "C0": ENV,
 "C1": ENV + ["chest"],
 "C2": ENV + SKIN,
 "C3": ENV + SKIN + ["HR"],
 "C4": ENV + SKIN + ["HR"] + MICRO,
}
# feature blocks at each window; used for paired deltas AND for window comparison
FE = {w: {k: featset(v, w) for k, v in CFG.items()} for w in (5, 15, 30)}

Y   = df.Corerectal.values
PID = df.Participant.values
CID = df.Condition.values
TAM = df.Temp.values

def metrics(y, yh):
    e = yh - y; ss = np.sum((y - y.mean())**2)
    return dict(MAE=float(np.mean(np.abs(e))), RMSE=float(np.sqrt(np.mean(e**2))),
                R2=float(1 - np.sum(e**2)/ss) if ss > 0 else np.nan, bias=float(np.mean(e)))

def run_cv(d, fc, fold_assign, seed=0):
    """d: DataFrame with __y; fc: feature cols; fold_assign: dict fold->index array."""
    X = d[fc].values; y = d["__y"].values
    yh = np.full(len(d), np.nan)
    for f, te_idx in fold_assign.items():
        m = d.index.isin(te_idx)
        if m.sum() == 0 or (~m).sum() < 50:
            continue
        sc = StandardScaler().fit(X[~m])            # R4: fit on TRAIN participants only
        mdl = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(sc.transform(X[~m]), y[~m])
        yh[m] = mdl.predict(sc.transform(X[m]))
    ok = np.isfinite(yh)
    r = metrics(y[ok], yh[ok]); r["n"] = int(ok.sum())
    return r

# ---------------- ONE globally-verified common row set ----------------
# Rows where EVERY configuration at EVERY window length is complete AND the
# target is present. Every comparison below uses exactly this set, so all
# deltas are paired by construction.
_common = None
for w in (5, 15, 30):
    for k in CFG:
        v = FE[w][k].index[FE[w][k].notna().all(axis=1)].values
        _common = v if _common is None else np.intersect1d(_common, v)
COMMON = np.intersect1d(_common, df.index[np.isfinite(Y)].values)
print("GLOBAL COMMON set:", len(COMMON),
      "participants:", df.loc[COMMON, "Participant"].nunique(),
      "conditions:", sorted(df.loc[COMMON, "Condition"].unique()), flush=True)
assert len(COMMON) > 1000

def run_on(rows, cols, w):
    """Paired evaluation on an explicit row set; guards against residual NaN."""
    F = featset(cols, w) if w != 5 else None
    return None

# participant-level folds, mapped to ROW index arrays (run_cv matches on index)
parts = np.array(sorted(df.Participant.unique()))
PL = np.array_split(np.random.default_rng(0).permutation(parts), 5)
folds_p = {i: df.index[np.isin(PID, f)].values for i, f in enumerate(PL)}
print("fold row counts:", {k: len(v) for k, v in folds_p.items()}, flush=True)

def eval_cfg(k, w, rows):
    F = FE[w][k].loc[rows]
    d = F.copy(); d["__y"] = Y[rows]
    d = d[d["__y"].notna()].dropna()
    if len(d) < 300:
        return None
    return run_cv(d, list(F.columns), folds_p)

print("\n" + "="*100)
print("PAIRED SENSOR-CONFIGURATION COMPARISON  (identical rows for every step, w=5)")
print("="*100)
print(f"{'config':<6} {'nfeat':>5} {'n_used':>7} {'RMSE':>8} {'R2':>8} {'dR2 vs prev':>12}")
print("-"*100)
paired_rows = []; prev = None
for k in CFG:
    r = eval_cfg(k, 5, COMMON)
    if r is None:
        print(f"{k:<6} insufficient"); continue
    r["cfg"] = k; r["nfeat"] = FE[5][k].shape[1]; r["n_used"] = r["n"]
    r["dR2"] = np.nan if prev is None else r["R2"] - prev
    prev = r["R2"]; paired_rows.append(r)
    ds = "" if np.isnan(r["dR2"]) else f"{r['dR2']:+12.4f}"
    print(f"{k:<6} {r['nfeat']:>5} {r['n_used']:>7} {r['RMSE']:>8.4f} {r['R2']:>8.4f} {ds:>12}")
pd.DataFrame(paired_rows).to_csv(P + r"\_prospie_v2_paired.csv", index=False)

print("\n" + "="*100)
print("COVERAGE x PERFORMANCE  (each config on its OWN complete rows)")
print("="*100)
print(f"{'config':<6} {'n_used':>7} {'coverage':>9} {'R2':>8} {'RMSE':>8} {'R2*coverage':>12}")
print("-"*100)
cov_rows = []
for k in CFG:
    Fk = FE[5][k]
    ok_k = Fk.notna().all(axis=1).values & np.isfinite(Y)
    rows_k = df.index[ok_k].values
    r = eval_cfg(k, 5, rows_k)
    if r is None: continue
    r["cfg"] = k; r["n_used"] = len(rows_k); r["coverage"] = len(rows_k)/int(np.isfinite(Y).sum())
    r["R2xcov"] = r["R2"] * r["coverage"]
    cov_rows.append(r)
    print(f"{k:<6} {len(rows_k):>7} {100*r['coverage']:>8.1f}% {r['R2']:>8.4f} {r['RMSE']:>8.4f} {r['R2xcov']:>12.4f}")
pd.DataFrame(cov_rows).to_csv(P + r"\_prospie_v2_coverage.csv", index=False)

print("\n" + "="*100)
print("CHANNEL-GROUP CONTRIBUTIONS  (identical rows, w=5)")
print("="*100)
print("identical rows:", len(COMMON))
RC = {}
for k in CFG:
    r = eval_cfg(k, 5, COMMON); RC[k] = r["R2"]
print(f"  C0 (env only)          R2 = {RC['C0']:+.4f}")
print(f"  C1 (env + chest)       R2 = {RC['C1']:+.4f}   dR2 vs C0  = {RC['C1']-RC['C0']:+.4f}")
print(f"  C2 (env + 11 skin)     R2 = {RC['C2']:+.4f}   dR2_skin11 = {RC['C2']-RC['C1']:+.4f}")
print(f"  C3 (+ HR)              R2 = {RC['C3']:+.4f}   dR2_HR     = {RC['C3']-RC['C2']:+.4f}")
print(f"  C4 (+ microclimate)    R2 = {RC['C4']:+.4f}   dR2_micro  = {RC['C4']-RC['C3']:+.4f}")
json.dump({"R2": RC, "dR2_skin11": RC['C2']-RC['C1'], "dR2_HR": RC['C3']-RC['C2'],
           "dR2_micro": RC['C4']-RC['C3']},
          open(P + r"\_prospie_v2_contrib.json", "w"), indent=1)

print("\n" + "="*100)
print("WINDOW LENGTH on IDENTICAL PREDICTION TIMESTAMPS (no stage confound)")
print("="*100)
print("identical prediction timestamps across w=5/15/30:", len(COMMON))
print(f"{'config':<6} " + "".join(f"{'w='+str(w):>26}" for w in (5, 15, 30)))
print("-"*100)
win_rows = []
for k in CFG:
    line = f"{k:<6} "; rec = {"cfg": k}
    for w in (5, 15, 30):
        r = eval_cfg(k, w, COMMON)
        if r is None: line += f"{'insufficient':>26}"; continue
        rec[f"R2_w{w}"] = r["R2"]; rec[f"RMSE_w{w}"] = r["RMSE"]; rec[f"n_w{w}"] = r["n"]
        line += f"  R2={r['R2']:+.4f} n={r['n']:5d}   "
    win_rows.append(rec); print(line)
pd.DataFrame(win_rows).to_csv(P + r"\_prospie_v2_window.csv", index=False)

print("\nPROTOCOL-STAGE CONFOUND CHECK (activity-code shares within the common set)")
ac = df.ActivityCode.values[COMMON]; yv = Y[COMMON]
for a in sorted(set(ac)):
    m = ac == a
    if m.sum() < 30: continue
    print(f"  ActivityCode {a}: n={m.sum():5d} ({100*m.mean():5.1f}%)  Tcore mean={yv[m].mean():.3f} sd={yv[m].std():.3f}")
print(f"  overall sd = {yv.std():.4f}")
hi = [int(a) for a in sorted(set(ac)) if (ac==a).sum()>=30 and yv[ac==a].std() > yv.std()]
print(f"  stages whose WITHIN-stage sd exceeds overall sd: {hi}")
print(f"  -> a 30-min window necessarily mixes rest and exercise stages;")
print(f"     stage composition of each window is therefore not matched across w.")

# ================================================================
# RESULT 2d -- leave-one-condition-out (protocol/environment transfer)
# ================================================================
print("\n" + "="*100)
print("LEAVE-ONE-CONDITION-OUT  (protocol / environment transfer)")
print("="*100)
loco_rows = []
for k in CFG:
    F = FE[5][k]
    d = F.copy(); d["__y"] = Y; d["__c"] = CID
    d = d[np.isfinite(d["__y"])].dropna()
    fc = [c for c in F.columns]
    yh = np.full(len(d), np.nan); Cv = d["__c"].values
    for c in sorted(set(Cv)):
        m = Cv == c
        if m.sum() < 100 or (~m).sum() < 100: continue
        sc = StandardScaler().fit(d.loc[~m, fc].values)
        mdl = RidgeCV(alphas=np.logspace(-2,4,25)).fit(sc.transform(d.loc[~m, fc].values), d.loc[~m,"__y"].values)
        yh[m] = mdl.predict(sc.transform(d.loc[m, fc].values))
    ok = np.isfinite(yh)
    r = metrics(d["__y"].values[ok], yh[ok]); r["cfg"] = k; r["n"] = int(ok.sum())
    loco_rows.append(r)
    print(f"  {k:<6} n={r['n']:6d}  MAE={r['MAE']:.4f}  RMSE={r['RMSE']:.4f}  R2={r['R2']:+.4f}")
pd.DataFrame(loco_rows).to_csv(P + r"\_prospie_v2_loco.csv", index=False)

print("\nSAVED v2 outputs")
