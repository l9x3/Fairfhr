import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from models import (train_mlp, gga_optimize, refine_chromosome, np_forward, mae)
from debias import shap_values, bias_sensitive_features
from prep import AXES, GROUP, TARGET

GCMAP = {"Age": "age_grp", "BMI": "bmi_grp", "Gestation": "ga_grp"}


def build_global(work, feat, seed=0):
    X = work[feat].values.astype(np.float32); y = work[TARGET].values.astype(np.float32)
    idx = np.arange(len(X))
    tr, te = train_test_split(idx, test_size=0.25, random_state=0,
                              stratify=work[GROUP].values)
    sc = StandardScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
    my, sy = y[tr].mean(), y[tr].std(); inv = lambda z: z * sy + my
    itr, iva = train_test_split(np.arange(len(tr)), test_size=0.2, random_state=1)
    base = train_mlp(Xtr[itr], (y[tr][itr]-my)/sy, Xtr[iva], (y[tr][iva]-my)/sy,
                     batch=512, lr=1e-3, epochs=200, seed=0)
    chrom, _ = gga_optimize(base, Xtr[iva], (y[tr][iva]-my)/sy, generations=30, seed=0)
    chrom = refine_chromosome(chrom, Xtr[itr], (y[tr][itr]-my)/sy,
                              Xtr[iva], (y[tr][iva]-my)/sy, epochs=120, seed=0)
    sub = np.random.RandomState(0).choice(len(Xtr), min(800, len(Xtr)), replace=False)
    sv = shap_values(chrom, Xtr[sub], Xtr, bg_k=12, nsamples=120)
    print("[global] model MAE (in-dist test) =",
          round(mae(y[te], inv(np_forward(chrom, Xte))), 3))
    return dict(chrom=chrom, tr=tr, te=te, sub=sub, shap=sv,
                scaler_mean=sc.mean_, scaler_scale=sc.scale_, my=my, sy=sy)


def k_selection(work, feat, art, seed=7):
    X = work[feat].values.astype(np.float32); y = work[TARGET].values.astype(np.float32)
    gc = {ax: work[GCMAP[ax]].values for ax in AXES}
    tr, sub, sv, chrom = art["tr"], art["sub"], art["shap"], art["chrom"]
    Xtr = (X[tr] - art["scaler_mean"]) / art["scaler_scale"]
    inv = lambda z: z * art["sy"] + art["my"]
    opt_tr = inv(np_forward(chrom, Xtr.astype(np.float32)))
    resid_all = opt_tr[sub] - y[tr][sub]
    grp_all = {ax: gc[ax][tr][sub] for ax in AXES}
    n = len(sub); ks = list(range(3, 13))
    fit_i, ev_i = train_test_split(np.arange(n), test_size=0.35, random_state=seed)

    per_axis = {}
    for ax, (col, order) in AXES.items():
        labs = grp_all[ax]; mu = []
        for g in order:
            sel = np.where(labs == g)[0]
            if len(sel): mu.append(np.mean(np.abs(sv[sel]), axis=0))
        disp = np.vstack(mu).max(0) - np.vstack(mu).min(0)
        ranked = np.argsort(disp)[::-1]
        curves = {sg: [] for sg in order}
        for k in ks:
            fi = ranked[:k]
            reg = Ridge(alpha=1.0).fit(sv[fit_i][:, fi], resid_all[fit_i])
            corr = np.clip(reg.predict(sv[ev_i][:, fi]), -15, 15)
            e = np.abs(y[tr][sub][ev_i] - (opt_tr[sub][ev_i] - corr))
            for sg in order:
                mm = labs[ev_i] == sg
                curves[sg].append(float(np.mean(e[mm])) if mm.sum() else float("nan"))
        per_axis[ax] = (ks, curves)

    grp_fit = {ax: grp_all[ax][fit_i] for ax in AXES}
    elbow = []
    for k in ks:
        fi, _ = bias_sensitive_features(sv[fit_i], grp_fit, AXES, k=k)
        reg = Ridge(alpha=1.0).fit(sv[fit_i][:, fi], resid_all[fit_i])
        corr = np.clip(reg.predict(sv[ev_i][:, fi]), -15, 15)
        elbow.append(float(np.mean(np.abs(y[tr][sub][ev_i] - (opt_tr[sub][ev_i] - corr)))))

    rng = np.random.RandomState(0); chosen = []
    for _ in range(500):
        bi = rng.choice(len(sub), len(sub), replace=True)
        f2, e2 = train_test_split(np.arange(len(bi)), test_size=0.35,
                                  random_state=int(rng.randint(1e6)))
        svb, rb = sv[bi], resid_all[bi]; grpb = {ax: grp_all[ax][bi] for ax in AXES}
        best_k, best_e = ks[0], np.inf
        for k in ks:
            fi, _ = bias_sensitive_features(svb[f2], {ax: grpb[ax][f2] for ax in AXES}, AXES, k=k)
            reg = Ridge(alpha=1.0).fit(svb[f2][:, fi], rb[f2])
            e = np.mean(np.abs(rb[e2] - np.clip(reg.predict(svb[e2][:, fi]), -15, 15)))
            if e < best_e - 1e-6: best_e, best_k = e, k
        chosen.append(best_k)
    vals, cnts = np.unique(chosen, return_counts=True)
    stability = {int(v): round(100 * c / len(chosen), 1) for v, c in zip(vals, cnts)}

    art["kcurves"] = per_axis
    art["elbow"] = {"k": ks, "mae": elbow}
    art["kstability"] = stability
    print("[k-selection] elbow min k =", ks[int(np.argmin(elbow))],
          "| bootstrap mode k =", int(vals[np.argmax(cnts)]))
    return art
