import warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, train_test_split
from models import (train_mlp, gga_optimize, refine_chromosome, np_forward, mae, all_metrics)
from debias import shap_values, bias_sensitive_features, fit_corrector, apply_correction
from prep import AXES, GROUP, TARGET, load_full
from config import TABLES_DIR

GCMAP = {"Age": "age_grp", "BMI": "bmi_grp", "Gestation": "ga_grp"}


def _fresh_test(feat, per_subject=150):
    full, _ = load_full()
    parts = [sub.sample(n=min(per_subject, len(sub)), random_state=999)
             for _, sub in full.groupby(GROUP)]
    return pd.concat(parts).reset_index(drop=True)


def make_tables(work, feat, oof):
    Xtr = work[feat].values.astype(np.float32); ytr = work[TARGET].values.astype(np.float32)
    fresh = _fresh_test(feat)
    Xte = fresh[feat].values.astype(np.float32); yte = fresh[TARGET].values.astype(np.float32)
    gte = {ax: fresh[GCMAP[ax]].values for ax in AXES}
    sc = StandardScaler().fit(Xtr); Zt, Ze = sc.transform(Xtr), sc.transform(Xte)
    my, sy = ytr.mean(), ytr.std(); inv = lambda z: z*sy+my

    gkf = GroupKFold(5); rows = []
    for fold, (tri, _) in enumerate(gkf.split(Zt, ytr, work[GROUP].values), 1):
        Xf, yf = Zt[tri], ytr[tri]
        itr, iva = train_test_split(np.arange(len(tri)), test_size=0.2, random_state=fold)
        base = train_mlp(Xf[itr], (yf[itr]-my)/sy, Xf[iva], (yf[iva]-my)/sy,
                         batch=512, lr=1e-3, epochs=150, seed=fold)
        chrom, _ = gga_optimize(base, Xf[iva], (yf[iva]-my)/sy, generations=25, seed=fold)
        chrom = refine_chromosome(chrom, Xf[itr], (yf[itr]-my)/sy, Xf[iva], (yf[iva]-my)/sy,
                                  epochs=90, seed=fold)
        m = all_metrics(yte, inv(np_forward(chrom, Ze)))
        rows.append(dict(Fold=fold,
                         MAE_train=round(mae(yf[itr], inv(np_forward(chrom, Xf[itr]))), 2),
                         MAE_val=round(mae(yf[iva], inv(np_forward(chrom, Xf[iva]))), 2),
                         MAE_test=round(m["MAE"], 2), RMSE=round(m["RMSE"], 2),
                         R2=round(m["R2"], 2), MAPE=round(m["MAPE"], 2), PPA10=round(m["PPA10"], 2)))
    t11 = pd.DataFrame(rows)
    t11.to_csv(TABLES_DIR / "table11_crossdataset_folds.csv", index=False)
    pd.DataFrame([t11.drop(columns="Fold").mean().round(2), t11.drop(columns="Fold").std().round(2)],
                 index=["Mean", "Std"]).to_csv(TABLES_DIR / "table11_crossdataset_summary.csv")

    # mitigated model training 
    itr, iva = train_test_split(np.arange(len(Xtr)), test_size=0.2, random_state=0)
    base = train_mlp(Zt[itr], (ytr[itr]-my)/sy, Zt[iva], (ytr[iva]-my)/sy,
                     batch=512, lr=1e-3, epochs=200, seed=0)
    chrom, _ = gga_optimize(base, Zt[iva], (ytr[iva]-my)/sy, generations=30, seed=0)
    chrom = refine_chromosome(chrom, Zt[itr], (ytr[itr]-my)/sy, Zt[iva], (ytr[iva]-my)/sy,
                              epochs=120, seed=0)
    opt_tr, opt_te = inv(np_forward(chrom, Zt)), inv(np_forward(chrom, Ze))
    sub = np.random.RandomState(0).choice(len(Zt), min(700, len(Zt)), replace=False)
    sv_tr = shap_values(chrom, Zt[sub], Zt, bg_k=10, nsamples=100)
    grp = {ax: work[GCMAP[ax]].values[sub] for ax in AXES}
    fi, _ = bias_sensitive_features(sv_tr, grp, AXES, k=10)
    corr = fit_corrector(sv_tr, opt_tr[sub]-ytr[sub], fi, alpha=1.0)
    sv_te = shap_values(chrom, Ze, Zt, bg_k=10, nsamples=100)
    mit_te, _ = apply_correction(opt_te, sv_te, corr, fi, eps=0.05, delta=0.005, clip=15)

    err = np.abs(yte - mit_te); rows12, disps = [], []
    for ax, (col, order) in AXES.items():
        g = pd.Series(err).groupby(gte[ax]).mean().reindex(order)
        for sg in order: rows12.append((ax, sg, round(float(g[sg]), 2)))
        disps.append(round(float(g.max()-g.min()), 2))
    pd.DataFrame(rows12, columns=["Demographic", "Subgroup", "MAE_BPM"]).to_csv(
        TABLES_DIR / "crossdataset_subgroups.csv", index=False)

    ind_err = np.abs(oof["y"] - oof["mit"])
    ind_disp = np.mean([pd.Series(ind_err.values).groupby(oof[GCMAP[ax]].values).mean()
                        .reindex(o).pipe(lambda s: s.max()-s.min()) for ax, (c, o) in AXES.items()])
    cm = all_metrics(yte, mit_te)
    pd.DataFrame([
        dict(Dataset="IIScFHSDB (in-domain)", MAE=round(float(ind_err.mean()), 2),
             RMSE=round(float(np.sqrt(np.mean((oof['y']-oof['mit'])**2))), 2),
             InterGroup_Disparity=round(float(ind_disp), 2)),
        dict(Dataset="Fresh recordings (cross-test)", MAE=round(cm["MAE"], 2),
             RMSE=round(cm["RMSE"], 2), InterGroup_Disparity=round(float(np.mean(disps)), 2)),
    ]).to_csv(TABLES_DIR / "table13_crossdataset_compare.csv", index=False)
    print("[crossdataset] Tables written")
