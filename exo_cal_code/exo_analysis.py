#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, warnings, argparse, math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------- utils -----------------------------

def log(msg, quiet=False):
    if not quiet:
        print(msg, flush=True)

def ensure_dir(d):
    Path(d).mkdir(parents=True, exist_ok=True)

def read_archive_csv(path):
    """Read NASA Exoplanet Archive CSVs that start with comment lines (# ...)."""
    with open(path,'r',encoding='utf-8',errors='ignore') as f:
        lines = [ln for ln in f if not ln.startswith('#')]
    from io import StringIO
    return pd.read_csv(StringIO(''.join(lines)))

def to_str(x):
    try: s = str(x)
    except Exception: s = "NA"
    return s.replace('/','_').replace('\\','_').replace(':','_').replace(' ','')

def minmax01(x):
    x = np.asarray(x,float)
    a=np.nanmin(x); b=np.nanmax(x)
    if not np.isfinite(a) or not np.isfinite(b) or a==b: return np.zeros_like(x)
    return (x-a)/(b-a)

# ----------------------- label inference -------------------------

def labels_from_TOI(T):
    col = [c for c in T.columns if c.lower() in ('tfopwg_disp','tfopwgdisp','koi_pdisposition','koi_disposition')]
    y = np.full(len(T), np.nan); src=""
    if col:
        c = col[0]
        s = T[c].astype(str).str.upper().str.strip()
        pos = s.isin(['CP','KP','PC','CANDIDATE','CONFIRMED'])
        neg = s.isin(['FP','FALSE POSITIVE','FALSE_POSITIVE'])
        y[pos]=1; y[neg]=0; src=c
    return y, src

def labels_from_KOI(T):
    col = [c for c in T.columns if c.lower() in ('koi_disposition','koi_pdisposition')]
    y = np.full(len(T), np.nan); src=""
    if col:
        c=col[0]; s=T[c].astype(str).str.upper().str.strip()
        pos = s.isin(['CANDIDATE','CONFIRMED'])
        neg = s.isin(['FALSE POSITIVE'])
        y[pos]=1; y[neg]=0; src=c
    return y, src

def labels_from_K2(T):
    y = np.full(len(T), np.nan); src=""
    for c in T.columns:
        cl=c.lower()
        if cl in ('disposition','k2_disposition'):
            s=T[c].astype(str).str.upper().str.strip()
            pos=s.isin(['CANDIDATE','CONFIRMED'])
            neg=s.isin(['FALSE POSITIVE','FALSE_POSITIVE'])
            y[pos]=1; y[neg]=0; src=c; break
    return y, src

# ----------------------- feature builders ------------------------

def colfirst(T, keys):
    for k in keys:
        for c in T.columns:
            if c.lower()==k.lower():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    return pd.to_numeric(T[c], errors='coerce')
    return pd.Series(np.full(len(T), np.nan))

def build_features_TOI(T):
    P    = colfirst(T, ['pl_orbper','koi_period'])
    Durh = colfirst(T, ['pl_trandurh','koi_duration'])
    Depth_ppm = colfirst(T, ['pl_trandep','koi_depth'])
    Rp_Re = colfirst(T, ['pl_rade','koi_prad'])
    Teff  = colfirst(T, ['st_teff','koi_steff'])
    logg  = colfirst(T, ['st_logg','koi_slogg'])
    Rstar = colfirst(T, ['st_rad','koi_srad'])
    Vmag  = colfirst(T, ['sy_vmag','koi_kepmag','st_tmag'])
    Jmag  = colfirst(T, ['sy_kmag'])
    Kmag  = colfirst(T, ['sy_kmag'])
    ecc   = colfirst(T, ['pl_orbeccen'])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        depth_frac = Depth_ppm/1e6
        RpRs = np.sqrt(np.clip(depth_frac,0,np.inf))
        aR = np.where(np.isfinite(P) & np.isfinite(Rstar) & (Rstar>0),
                      np.cbrt((P/365.25)**2) / np.maximum(Rstar,1e-3), np.nan)

    X = np.column_stack([P, Durh, Depth_ppm, Rp_Re, Teff, logg, Rstar, Vmag, Jmag, Kmag, ecc, aR, depth_frac, RpRs])
    names = ['P','Dur','Depth','Rp','Teff','logg','Rstar','Vmag','Jmag','Kmag','ecc','aR','depth_frac','Rp/Rs']
    id_col = None
    for c in T.columns:
        if c.lower() in ('toi','koi_name','kepoi_name','pl_name','designation','pl_name'):
            id_col = T[c].astype(str); break
    if id_col is None:
        id_col = pd.Series([f'ROW{i}' for i in range(len(T))])
    return X, names, id_col, {'P_days':P,'Dur_hr':Durh,'Depth_ppm':Depth_ppm,'Rp_Re':Rp_Re,'Teff_K':Teff,'Rstar_Rsun':Rstar}

def build_features_KOI(T): return build_features_TOI(T)
def build_features_K2(T):  return build_features_TOI(T)

# ----------------- impute -> scale -> sanitize -------------------

def impute_and_scale(X):
    """
    Median-impute; columns that are entirely NaN get 0.
    RobustScale; any residual non-finite -> 0.
    """
    X = np.asarray(X, float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(X, axis=0)
    med[~np.isfinite(med)] = 0.0               # fully-NaN columns -> 0
    Ximp = np.where(np.isfinite(X), X, med)    # impute
    scaler = RobustScaler()
    Z = scaler.fit_transform(Ximp)
    Z[~np.isfinite(Z)] = 0.0                   # final sanitize
    return Z

# --------------------- model & calibration -----------------------

def train_stack(Z, y):
    """Returns (stack_func, calib_func, coef_for_explain or None)."""
    pos = np.where(y==1)[0]; neg = np.where(y==0)[0]
    if len(pos)>0 and len(neg)>0:
        lr = LogisticRegression(max_iter=4000, class_weight='balanced')
        lr.fit(Z, y)
        gb = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=2)
        gb.fit(Z, y)
        coef = getattr(lr, 'coef_', None)
        def stack(X):
            p1 = lr.predict_proba(X)[:,1]
            p2 = gb.predict_proba(X)[:,1]
            return 0.5*p1 + 0.5*p2
        f = IsotonicRegression(out_of_bounds='clip')
        z = stack(Z)
        f.fit(z, y)
        return stack, (lambda s: f.transform(s)), coef
    if len(pos)>0 and len(neg)==0:
        oc = IsolationForest(n_estimators=300, random_state=0, contamination=0.15)
        oc.fit(Z[pos])
        def stack(X):
            s = -oc.score_samples(X)
            return minmax01(-s)
        return stack, (lambda s: s), None
    return (lambda X: np.full(X.shape[0], 0.5)), (lambda s: s), None

def kfold_perf(Z, y, stack, calib):
    pos = (y==1); neg=(y==0)
    if pos.sum()<10 or neg.sum()<10:
        return {'auc_roc': np.nan, 'auc_pr': np.nan, 'ece': np.nan}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    p = np.zeros_like(y, float)
    for tr, te in skf.split(Z, y):
        stack_tr, calib_tr, _ = train_stack(Z[tr], y[tr])
        p[te] = calib_tr(stack_tr(Z[te]))
    try:
        auc = roc_auc_score(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
    except Exception:
        auc, ap = np.nan, np.nan
    # ECE
    bins = np.linspace(0,1,11)
    e = []
    for i in range(10):
        m = (p>=bins[i]) & (p<bins[i+1])
        if m.any():
            e.append(abs(p[m].mean() - y[m].mean()))
    ece = float(np.mean(e)) if e else np.nan
    return {'auc_roc':float(auc), 'auc_pr':float(ap), 'ece':ece}

# --------------------- plotting helpers -------------------------

def plot_roc_pr_reliability(y, p, outdir, tag):
    ensure_dir(outdir)
    # ROC
    try:
        fpr, tpr, _ = roc_curve(y, p)
        plt.figure(figsize=(5,3)); plt.plot(fpr,tpr,'-')
        plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'{tag} ROC (AUC={roc_auc_score(y,p):.3f})')
        plt.grid(True,alpha=.3); plt.tight_layout()
        plt.savefig(os.path.join(outdir,'roc.png'),dpi=150); plt.close()
    except Exception:
        pass
    # PR
    try:
        prec, rec, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y,p)
        plt.figure(figsize=(5,3)); plt.plot(rec,prec,'-')
        plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(f'{tag} PR (AP={ap:.3f})')
        plt.grid(True,alpha=.3); plt.tight_layout()
        plt.savefig(os.path.join(outdir,'pr.png'),dpi=150); plt.close()
    except Exception:
        pass
    # Reliability
    bins = np.linspace(0,1,11); xs=[]; ys=[]
    for i in range(10):
        m = (p>=bins[i]) & (p<bins[i+1])
        if m.any():
            xs.append(p[m].mean()); ys.append(y[m].mean())
    if xs:
        plt.figure(figsize=(5,3))
        plt.plot([0,1],[0,1],'k--',alpha=.4)
        plt.plot(xs,ys,'o-')
        plt.xlabel('Predicted'); plt.ylabel('Observed'); plt.title(f'{tag} reliability')
        plt.grid(True,alpha=.3); plt.tight_layout()
        plt.savefig(os.path.join(outdir,'reliability.png'),dpi=150); plt.close()

def plot_std_features(z, names, out_png):
    plt.figure(figsize=(5,3))
    ix = np.argsort(z); plt.barh(np.array(names)[ix], z[ix])
    plt.xlabel('z-score'); plt.title('Standardized features'); plt.tight_layout()
    plt.savefig(out_png, dpi=150); plt.close()

def plot_contrib_proxy(z, coef, names, out_png):
    c = z * (coef if coef is not None else np.ones_like(z))
    rank = np.argsort(-np.abs(c))[:12]
    plt.figure(figsize=(5,3))
    plt.barh(np.array(names)[rank], c[rank])
    plt.title('Top feature contributions'); plt.xlabel('local contribution proxy'); plt.tight_layout()
    plt.savefig(out_png, dpi=150); plt.close()

def plot_mc_hist(samples, out_png, title='Monte Carlo probability'):
    plt.figure(figsize=(5,3))
    plt.hist(samples, bins=24)
    plt.title(title); plt.xlabel('p'); plt.ylabel('count')
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def plot_pca_local(Xz, i, out_png, tag='PCA neighborhood (proxy)'):
    try:
        n = min(1500, Xz.shape[0])
        idx = np.random.RandomState(0).choice(Xz.shape[0], n, replace=False)
        pcs = PCA(n_components=2, random_state=0).fit_transform(Xz[idx])
        pt = PCA(n_components=2, random_state=0).fit_transform(Xz[i][None,:])
        plt.figure(figsize=(5,3))
        plt.scatter(pcs[:,0], pcs[:,1], s=6, alpha=.35, c='gray')
        plt.scatter(pt[:,0], pt[:,1], s=60, marker='*', c='gold')
        plt.title(tag); plt.xlabel('PC1'); plt.ylabel('PC2'); plt.tight_layout()
        plt.savefig(out_png, dpi=150); plt.close()
    except Exception:
        pass

# --------------------- physics validation -----------------------

def duration_proxy_hours(P_days, Rstar_Rsun, Mstar_Msun=None, RpRs=None):
    if not np.isfinite(P_days) or not np.isfinite(Rstar_Rsun):
        return np.nan
    M = 1.0 if not np.isfinite(Mstar_Msun) else Mstar_Msun
    a_over_R = (M**(1/3)) * ((P_days/365.25)**(2/3)) * (215.032/max(Rstar_Rsun,1e-6))
    if not np.isfinite(a_over_R) or a_over_R<=0: return np.nan
    Dur = (P_days*24/np.pi) * (1.0/ a_over_R) * (1 + (RpRs if np.isfinite(RpRs) else 0))
    return Dur

def physics_checks(one_row):
    P = float(one_row.get('P_days', np.nan))
    Dur = float(one_row.get('Dur_hr', np.nan))
    Rstar = float(one_row.get('Rstar_Rsun', np.nan))
    RpRs = float(one_row.get('RpRs', np.nan))
    Mstar = float(one_row.get('Mstar_Msun', np.nan)) if 'Mstar_Msun' in one_row else np.nan
    expected = duration_proxy_hours(P, Rstar, Mstar, RpRs)
    if not np.isfinite(Dur) or not np.isfinite(expected):
        return {'dur_expected_hr': expected, 'dur_obs_hr': Dur, 'dur_ratio': np.nan, 'flag': 'insufficient_data'}
    ratio = Dur / max(expected,1e-6)
    flag = 'ok'
    if ratio>2.5 or ratio<0.4: flag = 'inconsistent'
    return {'dur_expected_hr': expected, 'dur_obs_hr': Dur, 'dur_ratio': ratio, 'flag': flag}

# ----------------------- target report --------------------------

def write_target_report(out_dir, designation, tag, z_row, names, coef, mc_samps, prob, phys_dict):
    d = Path(out_dir)/'targets'/to_str(designation)
    ensure_dir(d)
    plot_std_features(z_row, names, d/'std_feat.png')
    plot_contrib_proxy(z_row, coef, names, d/'contrib.png')
    plot_mc_hist(mc_samps, d/'mc_hist.png')
    plot_pca_local(z_row[None,:], 0, d/'pca_local.png')
    phys = phys_dict.copy()
    try:
        phys['RpRs'] = math.sqrt(max(phys.get('Depth_ppm',np.nan)/1e6,0))
    except Exception:
        pass
    pcheck = physics_checks(phys)
    summ = {
        'dataset': tag,
        'designation': str(designation),
        'pred_prob': float(prob),
        'entropy': float(- (prob*math.log(max(prob,1e-12)) + (1-prob)*math.log(max(1-prob,1e-12)))),
        'margin': float(abs(prob-0.5)),
        'top_features': [
            {'feature': names[i], 'contrib_proxy': float((coef if coef is not None else np.ones_like(z_row))[i]*z_row[i])}
            for i in np.argsort(-np.abs((coef if coef is not None else np.ones_like(z_row))*z_row))[:5]
        ],
        'physics': phys,
        'physics_check': pcheck,
        'mc': {'mean': float(np.mean(mc_samps)), 'std': float(np.std(mc_samps)),
               'p05': float(np.percentile(mc_samps,5)), 'p95': float(np.percentile(mc_samps,95)),
               'n': int(len(mc_samps))}
    }
    with open(d/'summary.json','w',encoding='utf-8') as f:
        json.dump(summ, f, indent=2)
    try:
        fig = plt.figure(figsize=(5,9))
        ax = fig.add_subplot(411)
        ix = np.argsort(z_row); ax.barh(np.array(names)[ix], z_row[ix]); ax.set_title(f'{tag} – {designation}')
        ax = fig.add_subplot(412)
        c = z_row*(coef if coef is not None else np.ones_like(z_row))
        ix = np.argsort(-np.abs(c))[:12]; ax.barh(np.array(names)[ix], c[ix]); ax.set_title('Top feature contributions')
        ax = fig.add_subplot(413); ax.hist(mc_samps, bins=24); ax.set_title('Monte Carlo probability')
        ax = fig.add_subplot(414); ax.scatter(np.arange(len(z_row)), z_row, s=8, alpha=.6); ax.axhline(0,color='k',lw=.5); ax.set_title('z-features (proxy)')
        fig.tight_layout(); fig.savefig(d/'diag.png', dpi=150); plt.close(fig)
    except Exception:
        pass

# ----------------------- dashboard ------------------------------

def write_dashboard(outdir, tag, perf):
    fh = Path(outdir)/'index.html'
    html = f"""<!doctype html><html><head><meta charset="utf-8">
    <title>{tag} results</title>
    <style>
    body{{font-family:system-ui,Segoe UI,Arial; margin:18px; background:#0b0e12; color:#dfe8f3}}
    a{{color:#8bd3ff}} .row{{display:flex; gap:18px; flex-wrap:wrap}}
    .card{{background:#151a23; padding:12px; border-radius:10px}}
    img{{max-width:520px; height:auto; border-radius:8px; background:#10141b}}
    </style></head><body>
    <h2>{tag} results</h2>
    <div class="row">
      <div class="card"><img src="fig/roc.png"><br>ROC</div>
      <div class="card"><img src="fig/pr.png"><br>Precision–Recall</div>
      <div class="card"><img src="fig/reliability.png"><br>Reliability</div>
      <div class="card"><img src="fig/pca_dataset.png"><br>PCA (dataset)</div>
    </div>
    <p>
      <a href="dataset_predictions.csv">dataset_predictions.csv</a> ·
      <a href="threshold_table.csv">threshold_table.csv</a> ·
      <a href="feature_importance.csv">feature_importance.csv</a> ·
      <a href="active_learning_queue.csv">active_learning_queue.csv</a> ·
      <a href="top_candidates.csv">top_candidates.csv</a> ·
      <a href="perf.csv">perf.csv</a>
    </p>
    <h3>Key metrics</h3>
    <pre>{json.dumps(perf, indent=2)}</pre>
    </body></html>"""
    with open(fh,'w',encoding='utf-8') as f: f.write(html)

# ----------------------- main runner ----------------------------

def run_dataset(path, tag, outdir, limit_targets=None, seed=7, quiet=False):
    ensure_dir(outdir); figdir = Path(outdir)/'fig'; ensure_dir(figdir); ensure_dir(Path(outdir)/'targets')

    T = read_archive_csv(path)
    if tag=='TOI':
        y, src = labels_from_TOI(T); X,names,ids,phys = build_features_TOI(T)
    elif tag=='KOI':
        y, src = labels_from_KOI(T); X,names,ids,phys = build_features_KOI(T)
    else:
        y, src = labels_from_K2(T);  X,names,ids,phys = build_features_K2(T)

    log(f"[{tag}] labels src={src} | pos={np.nansum(y==1):.0f} neg={np.nansum(y==0):.0f} unl={np.nansum(~np.isfinite(y)):.0f}", quiet)

    # Robust impute -> scale -> sanitize (and double-check)
    Z = impute_and_scale(X)
    if np.isnan(Z).any() or (~np.isfinite(Z)).any():
        Z[~np.isfinite(Z)] = 0.0

    # Train
    is_lab = np.isfinite(y)
    if is_lab.any():
        stack, calib, coef = train_stack(Z[is_lab], y[is_lab].astype(int))
        oof_perf = kfold_perf(Z[is_lab], y[is_lab].astype(int), stack, calib)
    else:
        stack, calib, coef = (lambda X: np.full(X.shape[0],0.5)), (lambda s: s), None
        oof_perf = {'auc_roc':np.nan,'auc_pr':np.nan,'ece':np.nan}

    # Score all
    s_raw = stack(Z); p = calib(s_raw)

    # Dataset PCA figure
    try:
        pcs = PCA(n_components=2, random_state=0).fit_transform(Z)
        plt.figure(figsize=(5,3)); plt.scatter(pcs[:,0], pcs[:,1], s=6, alpha=.35, c='gray')
        plt.title(f'{tag} PCA (dataset)'); plt.tight_layout()
        plt.savefig(figdir/'pca_dataset.png', dpi=150); plt.close()
    except Exception:
        pass

    if is_lab.sum()>=5 and (y[is_lab]==1).sum()>0 and (y[is_lab]==0).sum()>0:
        plot_roc_pr_reliability(y[is_lab].astype(int), p[is_lab], figdir, tag)

    # threshold sweep
    thr_table=[]
    if is_lab.any():
        tgrid = np.linspace(0.05,0.95,19)
        for t in tgrid:
            yl = (p[is_lab]>=t).astype(int); yt = y[is_lab].astype(int)
            tp = int(((yl==1)&(yt==1)).sum()); fp = int(((yl==1)&(yt==0)).sum())
            fn = int(((yl==0)&(yt==1)).sum()); tn = int(((yl==0)&(yt==0)).sum())
            prec = tp/max(tp+fp,1); rec = tp/max(tp+fn,1); f1 = 2*prec*rec/max(prec+rec,1e-12)
            thr_table.append({'threshold':float(t),'tp':tp,'fp':fp,'fn':fn,'tn':tn,
                              'precision':float(prec),'recall':float(rec),'f1':float(f1)})
        pd.DataFrame(thr_table).to_csv(Path(outdir)/'threshold_table.csv', index=False)

    # feature importance
    if coef is not None:
        df_imp = pd.DataFrame({'feature':names, 'weight':coef[0] if hasattr(coef,'ndim') and coef.ndim>1 else coef})
        df_imp.sort_values('weight', key=np.abs, ascending=False).to_csv(Path(outdir)/'feature_importance.csv', index=False)

    # predictions CSV (+ physics cols)
    label_col = pd.Series(np.where(is_lab, y, np.nan), name='label')
    df_pred = pd.DataFrame({'dataset':tag, 'designation':ids.astype(str), 'prob':p})
    df_pred['label']=label_col
    for k in {'P_days','Dur_hr','Depth_ppm','Rp_Re','Teff_K','Rstar_Rsun'}:
        if k in phys: df_pred[k]=phys[k]
    df_pred.to_csv(Path(outdir)/'dataset_predictions.csv', index=False)

    # active learning queue (FIX: correct clip)
    if (~is_lab).any():
        ent = -(p*np.log(np.clip(p,1e-12,1)) + (1-p)*np.log(np.clip(1-p,1e-12,1)))
        u = np.where(~is_lab)[0]; u_sorted = u[np.argsort(ent[u])[::-1]]
        k = min(200, len(u_sorted))
        pd.DataFrame({'designation':ids.iloc[u_sorted[:k]].astype(str), 'score':p[u_sorted[:k]],
                      'entropy':ent[u_sorted[:k]]}).to_csv(Path(outdir)/'active_learning_queue.csv', index=False)

    # top candidates
    pd.DataFrame(df_pred.sort_values('prob',ascending=False).head(200)).to_csv(Path(outdir)/'top_candidates.csv', index=False)

    # perf.csv
    pd.DataFrame([{'metric':'auc_roc','value':oof_perf['auc_roc']},
                  {'metric':'auc_pr','value':oof_perf['auc_pr']},
                  {'metric':'ece','value':oof_perf['ece']}]).to_csv(Path(outdir)/'perf.csv', index=False)

    # per-target reports
    N = len(df_pred) if limit_targets is None else min(limit_targets, len(df_pred))
    rng = np.random.RandomState(seed)
    for i,(idx,row) in enumerate(df_pred.head(N).iterrows(), start=1):
        desig = row['designation']; log(f"[{tag}] Working on {i}/{N}: {desig}", quiet)
        zrow = Z[idx].copy()
        eps = rng.normal(0, 0.02, size=(256, Z.shape[1]))
        s_mc = calib(stack((Z[idx][None,:] + eps).clip(-8,8)))
        write_target_report(outdir, desig, tag, zrow, names,
                            (coef[0] if (coef is not None and hasattr(coef,'ndim') and coef.ndim>1) else coef),
                            s_mc, row['prob'],
                            {k: (phys[k].iloc[idx] if isinstance(phys[k], pd.Series) else phys[k]) for k in phys})

    # dashboard
    write_dashboard(outdir, tag, {'cv_auc_roc':oof_perf['auc_roc'], 'cv_auc_pr':oof_perf['auc_pr'], 'ece':oof_perf['ece']})

# -------------------------- CLI ---------------------------------

def main():
    p = argparse.ArgumentParser(description="Exoplanet Overkill + validation + static dashboard")
    p.add_argument('--toi', type=str, default=None)
    p.add_argument('--koi', type=str, default=None)
    p.add_argument('--k2',  type=str, default=None)
    p.add_argument('--out', type=str, required=True)
    p.add_argument('--limit-targets', type=int, default=None)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--quiet', action='store_true')
    args = p.parse_args()

    np.random.seed(args.seed)
    out_root = args.out; ensure_dir(out_root)

    if args.toi:
        log("\n=== TOI ===", args.quiet)
        run_dataset(args.toi, 'TOI', os.path.join(out_root,'TOI'),
                    limit_targets=args.limit_targets, seed=args.seed, quiet=args.quiet)
    if args.koi:
        log("\n=== KOI ===", args.quiet)
        run_dataset(args.koi, 'KOI', os.path.join(out_root,'KOI'),
                    limit_targets=args.limit_targets, seed=args.seed, quiet=args.quiet)
    if args.k2:
        log("\n=== K2 ===", args.quiet)
        run_dataset(args.k2, 'K2', os.path.join(out_root,'K2'),
                    limit_targets=args.limit_targets, seed=args.seed, quiet=args.quiet)

if __name__ == "__main__":
    main()
