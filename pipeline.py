import time, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

from models import (train_mlp, predict, gga_optimize, np_forward,
                    model_to_chromosome, all_metrics, mae, refine_chromosome)
from debias import (shap_values, bias_sensitive_features, fit_corrector,
                    apply_correction)
from prep import AXES, GROUP, TARGET, SEED

K_BIAS = 10
EPS, DELTA = 0.05, 0.005        # paper Table 3
CORR_TRAIN_CAP = 700            # SHAP samples used to fit the corrector
GCMAP = {"Age": "age_grp", "BMI": "bmi_grp", "Gestation": "ga_grp"}


def inner_split(subjects_tr, frac_val=0.25, seed=SEED):
    """Group-level 75:25 train/val split within a training fold."""
    uniq = np.unique(subjects_tr)
    r = np.random.RandomState(seed); r.shuffle(uniq)
    n_val = max(1, int(round(frac_val * len(uniq))))
    val_mask = np.isin(subjects_tr, list(uniq[:n_val]))
    return ~val_mask, val_mask


def hp_search(Xtr, ytr, Xval, yval):
    """Grid over batch x lr (target standardised); returns (df, best_cfg)."""
    my, sy = ytr.mean(), ytr.std()
    ztr, zval = (ytr - my) / sy, (yval - my) / sy
    grid = [(b, lr) for b in (128, 256, 512) for lr in (1e-3, 1e-2)]
    rows, best, best_mae = [], None, np.inf
    for b, lr in grid:
        m = train_mlp(Xtr, ztr, Xval, zval, batch=b, lr=lr, epochs=200, seed=0)
        vm = mae(yval, predict(m, Xval) * sy + my)
        rows.append(dict(batch=b, lr=lr, val_MAE=round(vm, 4)))
        if vm < best_mae:
            best_mae, best = vm, (b, lr)
    return pd.DataFrame(rows), best


def run(work, feat_cols, verbose=True):
    X = work[feat_cols].values.astype(np.float32)
    y = work[TARGET].values.astype(np.float32)
    groups = work[GROUP].values
    grp_cols = {ax: work[GCMAP[ax]].values for ax in AXES}

    gkf = GroupKFold(n_splits=5)
    fold_rows, oof, feat_sel = [], [], {}
    t0 = time.time()

    # hyper-parameter selection on the first fold
    tr0, _ = next(gkf.split(X, y, groups))
    sc0 = StandardScaler().fit(X[tr0])
    itr, iva = inner_split(groups[tr0])
    hp_df, best_cfg = hp_search(sc0.transform(X[tr0])[itr], y[tr0][itr],
                                sc0.transform(X[tr0])[iva], y[tr0][iva])
    batch, lr = best_cfg
    if verbose:
        print(f"[HP] best (batch, lr) = {best_cfg}")

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups), 1):
        ft = time.time()
        scaler = StandardScaler().fit(X[tr])
        Xtr_all, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
        ytr_all, yte = y[tr], y[te]
        itr, iva = inner_split(groups[tr])
        Xtr, ytr = Xtr_all[itr], ytr_all[itr]
        Xva, yva = Xtr_all[iva], ytr_all[iva]
        my, sy = ytr.mean(), ytr.std()
        ztr, zva = (ytr - my) / sy, (yva - my) / sy
        inv = lambda z: z * sy + my

        # 1. baseline MLP
        base = train_mlp(Xtr, ztr, Xva, zva, batch=batch, lr=lr, epochs=200, seed=fold)
        base_tr, base_va, base_te = (inv(predict(base, Xtr)), inv(predict(base, Xva)),
                                     inv(predict(base, Xte)))

        # 2. GGA-optimised MLP (global GA search + greedy local refine)
        best_chrom, _ = gga_optimize(base, Xva, zva, pop_size=30, generations=40, seed=fold)
        best_chrom = refine_chromosome(best_chrom, Xtr, ztr, Xva, zva, lr=1e-3,
                                       epochs=120, seed=fold)
        base_chrom = model_to_chromosome(base).astype(np.float64)
        if mae(yva, inv(np_forward(base_chrom, Xva))) < mae(yva, inv(np_forward(best_chrom, Xva))):
            best_chrom = refine_chromosome(base_chrom, Xtr, ztr, Xva, zva, lr=1e-3,
                                           epochs=120, seed=fold)
        pred_bpm = lambda Z: inv(np_forward(best_chrom, Z))
        opt_tr, opt_va, opt_te = pred_bpm(Xtr), pred_bpm(Xva), pred_bpm(Xte)

        # 3. SHAP-residual mitigated model
        n_ct = min(CORR_TRAIN_CAP, Xtr.shape[0])
        ct_idx = np.random.RandomState(fold).choice(Xtr.shape[0], n_ct, replace=False)
        shap_ct = shap_values(best_chrom, Xtr[ct_idx], Xtr, bg_k=10, nsamples=100)
        tr_all_idx = tr[itr]
        grp_ct = {ax: grp_cols[ax][tr_all_idx][ct_idx] for ax in AXES}
        feat_idx, _ = bias_sensitive_features(shap_ct, grp_ct, AXES, k=K_BIAS)
        corrector = fit_corrector(shap_ct, opt_tr[ct_idx] - ytr[ct_idx], feat_idx, alpha=1.0)
        feat_sel[fold] = [int(i) for i in feat_idx]

        shap_te = shap_values(best_chrom, Xte, Xtr, bg_k=10, nsamples=100)
        mit_te, _ = apply_correction(opt_te, shap_te, corrector, feat_idx, eps=EPS, delta=DELTA)
        mit_tr, _ = apply_correction(opt_tr[ct_idx], shap_ct, corrector, feat_idx, EPS, DELTA)
        shap_va = shap_values(best_chrom, Xva, Xtr, bg_k=10, nsamples=100)
        mit_va, _ = apply_correction(opt_va, shap_va, corrector, feat_idx, EPS, DELTA)

        for name, tr_p, va_p, te_p, tr_y in [
            ("baseline", base_tr[ct_idx], base_va, base_te, ytr[ct_idx]),
            ("optimized", opt_tr[ct_idx], opt_va, opt_te, ytr[ct_idx]),
            ("mitigated", mit_tr, mit_va, mit_te, ytr[ct_idx])]:
            m = all_metrics(yte, te_p)
            fold_rows.append(dict(model=name, fold=fold,
                                  MAE_train=round(mae(tr_y, tr_p), 2),
                                  MAE_val=round(mae(yva, va_p), 2),
                                  MAE_test=round(m["MAE"], 2), RMSE=round(m["RMSE"], 2),
                                  R2=round(m["R2"], 2), MAPE=round(m["MAPE"], 2),
                                  PPA10=round(m["PPA10"], 2)))

        for i, gi in enumerate(te):
            oof.append(dict(idx=int(gi), subject=int(groups[te][i]), y=float(yte[i]),
                            base=float(base_te[i]), opt=float(opt_te[i]), mit=float(mit_te[i]),
                            age_grp=grp_cols["Age"][te][i], bmi_grp=grp_cols["BMI"][te][i],
                            ga_grp=grp_cols["Gestation"][te][i]))
        if verbose:
            print(f"[fold {fold}] base={mae(yte,base_te):.2f} opt={mae(yte,opt_te):.2f} "
                  f"mit={mae(yte,mit_te):.2f}  ({time.time()-ft:.0f}s)")

    if verbose:
        print(f"[pipeline] done in {time.time()-t0:.0f}s")
    return dict(oof=pd.DataFrame(oof), fold_metrics=pd.DataFrame(fold_rows),
                best_cfg=best_cfg, hp_search=hp_df, feat_sel=feat_sel)
