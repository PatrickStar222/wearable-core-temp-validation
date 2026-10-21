"""
REPRESENTATION-PRESERVING THERMAL-STATE RECONSTRUCTION UNDER DEGRADED SENSING
Data: PROSPIE.  Validation: LOSO (40 subjects).  All scaling/imputation/state fitting
on training subjects only.

Target A : reconstruct missing wearable observation  (skin sites)
Target B : recover latent thermal state (Tcore)

Estimators compared, all sharing the same downstream Tcore model class:
  D0 direct        : observed (degraded) channels -> Tcore, no repair
  D1 carry-forward : missing channel <- its last observed value
  D2 data-only imp : missing channel <- OLS on the other observed channels (train-fit)
  D3 physics-cons  : missing channel <- C @ xhat, where xhat follows a fitted
                     thermophysiological transition model  x_{t+1}=A x_t + B u_t
"""
import pandas as pd, numpy as np, warnings, os, json
warnings.filterwarnings("ignore")
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.preprocessing import StandardScaler

P = ""
SKIN = ["abdomen","calf","chest","foot","hand","head","lower arm","lower back",
        "thigh","upper arm","upper back"]
df = pd.read_pickle(os.path.join(P,"_prospie.pkl"))
df["tmin"]=df.Time.apply(lambda t: t.hour*60+t.minute+t.second/60 if hasattr(t,"hour") else np.nan)
for c in SKIN+["Corerectal","Temp","Humidity","Wind","HR"]:
    df[c]=pd.to_numeric(df[c],errors="coerce").replace(9999,np.nan)
df=df.sort_values(["Participant","Condition","tmin"]).reset_index(drop=True)
df["unit"]=df.Participant.astype(str)+"|"+df.Condition.astype(str)
ENV=["Temp","Humidity","Wind"]
# complete-case frame on the FULL sensor set (so degradation can be applied by us)
FULL = SKIN+["HR"]+ENV
keep = df[FULL+["Corerectal"]].notna().all(axis=1).values
D = df.loc[keep].reset_index(drop=True)
print(f"complete-case rows for the degradation experiment: {len(D)}  subjects={D.Participant.nunique()}")
Y = D.Corerectal.values
pv = D.Participant.values
U = ["unit"]
def grp(s): return D.groupby("unit")[s].shift(1).values
def grp2(s): return D.groupby("unit")[s].shift(2).values

# ---------------- degradation masks ----------------
rng = np.random.default_rng(0)
def mask_spatial(k, rng):
    M = np.ones((len(D), len(SKIN)), bool)
    for i in range(len(D)):
        drop = rng.choice(len(SKIN), size=k, replace=False)
        M[i, drop] = False
    return M
def mask_structured(sites, rng):
    M = np.ones((len(D), len(SKIN)), bool)
    for s_ in sites:
        if s_ in SKIN:
            M[:, SKIN.index(s_)] = False
    return M
def mask_temporal(minutes, rng):
    M = np.ones((len(D), len(SKIN)), bool)
    starts = D.groupby("unit").head(1).index.values
    lens = D.groupby("unit").size().values
    for s, L in zip(starts, lens):
        if L <= minutes+2: continue
        st = rng.integers(0, L-minutes)
        M[s+st : s+st+minutes, :] = False
    return M
def corrupt(X, rng, noise=0.15, bias=0.3, drift=0.5):
    Xc = X.copy()
    Xc = Xc + rng.normal(0, noise, Xc.shape)
    subj = rng.choice(len(D), size=len(D)//2, replace=False)
    Xc[subj] = Xc[subj] + bias
    t = np.arange(len(D))
    Xc = Xc + drift*np.sin(2*np.pi*t/600)[:,None]
    return Xc

# ---------------- reconstruction methods ----------------
def m_carry(S, M):
    """last observed value, else unit mean, else global mean"""
    out = S.copy()
    for j in range(S.shape[1]):
        col = S[:,j].copy(); col[~M[:,j]] = np.nan
        s = pd.Series(col).groupby(D.unit.values).ffill().values
        fill = np.where(np.isfinite(s), s, np.nanmean(S[:,j]))
        out[~M[:,j], j] = fill[~M[:,j]]
    return out

def m_datadriven(S, M, tr):
    """per-channel OLS on the other observed skin channels + env, fitted on TRAIN rows"""
    out = S.copy()
    obs_other = np.ones(S.shape[0], bool)
    for j in range(S.shape[1]):
        miss = ~M[:,j]
        if miss.sum() == 0: continue
        pred_cols = [k for k in range(S.shape[1]) if k != j]
        Xp = np.column_stack([S[:,pred_cols], D[ENV].values, np.ones(len(S))])
        trj = tr & (~miss)
        if trj.sum() < 200: out[miss, j] = np.nanmean(S[trj, j]) if trj.sum()>0 else np.nanmean(S[:,j]); continue
        lm = LinearRegression().fit(Xp[trj], S[trj, j])
        out[miss, j] = lm.predict(Xp[miss])
    return out

def m_physics(S, M, tr, D_):
    """
    Physics-constrained reconstruction.
    Constraint enforced: the reconstructed sensors must equal C @ x where x satisfies
        x_{t+1} = A x_t + B u_t
    fitted on TRAINING rows.  Latent x = [mean-skin, skin-spread]; C maps x back to
    each site via a per-site loadings vector learned on training rows.
    """
    out = S.copy()
    # latent state definition (from observed channels only, train-estimated loadings)
    # x1 = mean of AVAILABLE skin sites; x2 = spread of AVAILABLE skin sites
    avail = M
    x1 = np.where(avail, S, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x1m = np.nanmean(x1, axis=1)
        x2m = np.nanstd(x1, axis=1)
    X = np.column_stack([x1m, x2m])
    Xlag1 = np.column_stack([D.groupby("unit")[c].shift(1).values for c in ["_x1","_x2"]]) if False else None
    tmp = D.copy(); tmp["_x1"]=x1m; tmp["_x2"]=x2m
    x1l = tmp.groupby("unit")["_x1"].shift(1).values
    x2l = tmp.groupby("unit")["_x2"].shift(1).values
    Ul  = np.column_stack([tmp.groupby("unit")[c].shift(1).values for c in ENV])
    # fit transition A,B on TRAINING rows
    Xl = np.column_stack([x1l, x2l]); y1 = X[:,0]; y2 = X[:,1]
    ok = tr & np.isfinite(Xl).all(axis=1) & np.isfinite(Ul).all(axis=1) & np.isfinite(y1) & np.isfinite(y2)
    if ok.sum() < 200:
        return out
    Z = np.column_stack([Xl, Ul, np.ones(len(D))])
    b1,*_ = np.linalg.lstsq(Z[ok], y1[ok], rcond=None)
    b2,*_ = np.linalg.lstsq(Z[ok], y2[ok], rcond=None)
    # learned latent trajectory, recursively propagated from the last complete state
    x1h = x1m.copy(); x2h = x2m.copy()
    # per-site loadings: site_j ~ a_j*x1 + b_j*x2 + c_j   (train rows where observed)
    for j in range(S.shape[1]):
        miss = ~M[:,j]
        if miss.sum()==0: continue
        tj = tr & M[:,j] & np.isfinite(x1h) & np.isfinite(x2h)
        if tj.sum() < 200: out[miss,j]=np.nanmean(S[tj,j]) if tj.sum()>0 else np.nanmean(S[:,j]); continue
        W = np.column_stack([x1h[tj], x2h[tj], np.ones(tj.sum())])
        cf,*_ = np.linalg.lstsq(W, S[tj,j], rcond=None)
        # propagate the latent state forward within each unit using A,B and fill
        for s,L in zip(D.groupby("unit").head(1).index.values, D.groupby("unit").size().values):
            for i in range(s, s+L):
                if i>s and np.isfinite(Ul[i]).all():
                    x1h[i] = b1[0]*x1h[i-1] + b1[1]*x2h[i-1] + np.dot(b1[2:5], Ul[i]) + b1[5]
                    x2h[i] = b2[0]*x2h[i-1] + b2[1]*x2h[i-1] + np.dot(b2[2:5], Ul[i]) + b2[5]
        Wm = np.column_stack([x1h[miss], x2h[miss], np.ones(miss.sum())])
        out[miss,j] = Wm @ cf
    return out

def run(dmask, tag, corrupt_X=None, reps=1, seed=0):
    r = np.random.default_rng(seed)
    S0 = D[SKIN].values.astype(float).copy()
    if corrupt_X is not None: S0 = corrupt_X
    res={'tag':tag}
    preds={k: np.full(len(D), np.nan) for k in ["D0","D1","D2","D3"]}
    fid ={k: [] for k in ["D1","D2","D3"]}
    for s in np.unique(pv):
        te = pv==s; tr = ~te
        if te.sum()<20 or tr.sum()<300: continue
        S = S0.copy(); M = dmask.copy()
        # ffill within unit already handled by construction; carry/env
        S1 = m_carry(S, M)
        S2 = m_datadriven(S, M, tr)
        S3 = m_physics(S, M, tr, D)
        for name, Sig in [("D1",S1),("D2",S2),("D3",S3)]:
            fid[name].append(np.nanmean(np.abs(Sig[te][~M[te]] - D[SKIN].values[te][~M[te]])) if (~M[te]).sum()>0 else np.nan)
        def fitpred(Sig, use_obs_only=False):
            cols = [Sig[:,j] for j in range(len(SKIN))]
            Xf = np.column_stack(cols + [D[c].values for c in ["HR"]+ENV])
            sc = StandardScaler().fit(Xf[tr])
            m = RidgeCV(alphas=np.logspace(-2,4,25)).fit(sc.transform(Xf[tr]), Y[tr])
            return m.predict(sc.transform(Xf[te]))
        # D0: observed only (missing channels replaced by training mean)
        S0d = S.copy()
        for j in range(len(SKIN)):
            S0d[~M[:,j], j] = np.nanmean(S0[tr][M[tr][:,j], j]) if M[tr][:,j].sum()>0 else np.nanmean(S0[:,j])
        preds["D0"][te] = fitpred(S0d)
        preds["D1"][te] = fitpred(S1)
        preds["D2"][te] = fitpred(S2)
        preds["D3"][te] = fitpred(S3)
    out={}
    for k,v in preds.items():
        ok=np.isfinite(v)
        if ok.sum()<200: out[k]=None; continue
        e=v[ok]-Y[ok]; ss=np.sum((Y[ok]-Y[ok].mean())**2)
        out[k]=dict(n=int(ok.sum()),RMSE=float(np.sqrt(np.mean(e**2))),
                    nRMSE=float(np.sqrt(np.mean(e**2))/Y[ok].std()),
                    R2=float(1-np.sum(e**2)/ss))
    out["_fidelity"]={k: float(np.nanmean(v)) for k,v in fid.items() if len(v)}
    return out

print("\n"+"="*104); print("DEGRADATION GRID (LOSO; nRMSE of Tcore; fidelity = MAE of reconstructed missing skin)"); print("="*104)
base = run(np.ones((len(D),len(SKIN)),bool), "full")
print(f"  FULL   D0 nRMSE={base['D0']['nRMSE']:.4f}  (reference)")
rows=[]
for k in (1,3,6,8,10):
    m = mask_spatial(k, np.random.default_rng(100+k))
    r = run(m, f"spatial drop {k}")
    line=f"  drop {k:>2} sites "
    for key in ["D0","D1","D2","D3"]:
        line += f" {key}={r[key]['nRMSE']:.4f}" if r[key] else f" {key}=  --  "
    line += "   | fid " + " ".join(f"{kk}={r['_fidelity'].get(kk,float('nan')):.3f}" for kk in ["D1","D2","D3"])
    print(line); rows.append(dict(cond=f"spatial_{k}", **{kk:(r[kk]['nRMSE'] if r[kk] else np.nan) for kk in ["D0","D1","D2","D3"]}, fid=r["_fidelity"]))
for nm, sites in [("distal lost",["foot","hand","calf","lower arm"]),
                  ("proximal lost",["chest","upper back","lower back","abdomen"]),
                  ("chest lost",["chest"]), ("hand lost",["hand"])]:
    m = mask_structured(sites, np.random.default_rng(7))
    r = run(m, nm)
    line=f"  {nm:<16} "
    for key in ["D0","D1","D2","D3"]:
        line += f" {key}={r[key]['nRMSE']:.4f}" if r[key] else f" {key}=  --  "
    print(line); rows.append(dict(cond=nm, **{kk:(r[kk]['nRMSE'] if r[kk] else np.nan) for kk in ["D0","D1","D2","D3"]}, fid=r["_fidelity"]))
for mins in (1,5,10,20):
    m = mask_temporal(mins, np.random.default_rng(200+mins))
    r = run(m, f"temporal {mins}min")
    line=f"  temporal {mins:>2} min "
    for key in ["D0","D1","D2","D3"]:
        line += f" {key}={r[key]['nRMSE']:.4f}" if r[key] else f" {key}=  --  "
    print(line); rows.append(dict(cond=f"temporal_{mins}", **{kk:(r[kk]['nRMSE'] if r[kk] else np.nan) for kk in ["D0","D1","D2","D3"]}, fid=r["_fidelity"]))
# mixed
m = mask_spatial(3, np.random.default_rng(11))
Sc = corrupt(D[SKIN].values.astype(float), np.random.default_rng(12))
r = run(m, "noise+bias+drift+3 missing", corrupt_X=Sc)
line="  mixed (noise+bias+drift, 3 missing) "
for key in ["D0","D1","D2","D3"]:
    line += f" {key}={r[key]['nRMSE']:.4f}" if r[key] else f" {key}=  --  "
print(line); rows.append(dict(cond="mixed", **{kk:(r[kk]['nRMSE'] if r[kk] else np.nan) for kk in ["D0","D1","D2","D3"]}, fid=r["_fidelity"]))
pd.DataFrame(rows).to_csv(os.path.join(P,"_degrade_grid.csv"),index=False)
print("\nSAVED")
