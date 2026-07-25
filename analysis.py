import json, warnings
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")
from prep import AXES, GROUP, TARGET, PER_SUBJECT
from config import TABLES_DIR

GCMAP = {"Age": "age_grp", "BMI": "bmi_grp", "Gestation": "ga_grp"}
MODELS = [("Baseline", "base"), ("Optimized", "opt"), ("Mitigated", "mit")]


def _table1(work):
    subj = work.drop_duplicates(GROUP)
    rows = []
    for ax, (col, order) in AXES.items():
        vc = subj[col].value_counts().reindex(order).fillna(0).astype(int)
        for sg in order:
            rows.append((ax, sg, int(vc[sg])))
    df = pd.DataFrame(rows, columns=["Demographic_axis", "Subgroup", "IIScFHSDB_sample"])
    df.loc[len(df)] = ["Total", "", subj[GROUP].nunique()]
    df.to_csv(TABLES_DIR / "table1_subject_counts.csv", index=False)


def _table3(work):
    n_train = int(len(work) * 0.75)
    pd.DataFrame([
        ("p", "Input feature dimension", "212", "fPCG features"),
        ("N", "Training samples (working subset)", str(n_train), "75% of sample"),
        ("g", "Demographic subgroups", "3 per axis", "Age, BMI, gestational age"),
        ("k", "Bias-sensitive features", "10", "Elbow + bootstrap (Figs 6,7)"),
        ("-", "SHAP explainer", "Kernel SHAP", "Model-agnostic"),
        ("-", "Corrector", "Ridge (a=1.0)", "l2 residual head"),
        ("-", "Optimizer (refine)", "Adam", "lr 1e-3, wd 1e-4"),
        ("eps", "Residual threshold", "0.05 BPM", "Suppress noise-level corr."),
        ("delta", "SHAP-norm threshold", "0.005", "Non-trivial attribution"),
    ], columns=["Symbol", "Parameter", "Value", "Note"]).to_csv(
        TABLES_DIR / "table3_parameters.csv", index=False)


def _overall_tables(fm):
    for name, key in [("baseline", 4), ("optimized", 5), ("mitigated", 6)]:
        cols = ["fold", "MAE_train", "MAE_val", "MAE_test", "RMSE", "R2", "MAPE", "PPA10"]
        d = fm[fm.model == name].sort_values("fold")[cols]
        d.to_csv(TABLES_DIR / f"table{key}_{name}_folds.csv", index=False)
        summ = pd.DataFrame([d.drop(columns="fold").mean().round(2),
                             d.drop(columns="fold").std().round(2)], index=["Mean", "Std"])
        summ.to_csv(TABLES_DIR / f"table{key}_{name}_summary.csv")


def _fairness_tables(oof):
    for name, key in [("Baseline", 7), ("Optimized", 8), ("Mitigated", 9)]:
        col = dict(MODELS)[name]
        err = np.abs(oof["y"] - oof[col])
        rows, dd = [], []
        for ax, (c, order) in AXES.items():
            g = pd.Series(err.values).groupby(oof[GCMAP[ax]].values).mean().reindex(order)
            for sg in order:
                rows.append((ax, sg, round(float(g[sg]), 2)))
            dd.append((ax, round(float(g.max() - g.min()), 2), round(float(np.var(g.values)), 3)))
        df = pd.DataFrame(rows, columns=["Demographic", "Subgroup", "MAE_BPM"]).merge(
            pd.DataFrame(dd, columns=["Demographic", "Fairness_Disparity", "Fairness_Variance"]),
            on="Demographic", how="left")
        df.to_csv(TABLES_DIR / f"table{key}_{name.lower()}_fairness.csv", index=False)


def _subj_stat(oof, col, subjects, kind):
    m = oof[oof.subject.isin(subjects)]
    if kind == "mae":
        return float(np.mean(np.abs(m["y"] - m[col])))
    if kind == "rmse":
        return float(np.sqrt(np.mean((m["y"] - m[col]) ** 2)))
    err = np.abs(m["y"] - m[col]); ds = []
    for ax, (c, order) in AXES.items():
        g = pd.Series(err.values).groupby(m[GCMAP[ax]].values).mean().reindex(order)
        ds.append(g.max() - g.min())
    return float(np.nanmean(ds))


def _bca(oof, col, kind, n_boot=2000, seed=0):
    rng = np.random.RandomState(seed); uniq = np.unique(oof.subject.values)
    theta = _subj_stat(oof, col, uniq, kind)
    boots = np.array([_subj_stat(oof, col, rng.choice(uniq, len(uniq), replace=True), kind)
                      for _ in range(n_boot)])
    boots = boots[np.isfinite(boots)]
    z0 = stats.norm.ppf((np.sum(boots < theta) + 1e-9) / len(boots))
    jack = np.array([_subj_stat(oof, col, uniq[uniq != s], kind) for s in uniq])
    jm = jack.mean()
    a = np.sum((jm - jack) ** 3) / (6 * (np.sum((jm - jack) ** 2) ** 1.5) + 1e-12)
    adj = lambda al: stats.norm.cdf(z0 + (z0 + stats.norm.ppf(al)) /
                                    (1 - a * (z0 + stats.norm.ppf(al))))
    lo = np.percentile(boots, 100 * adj(0.025)); hi = np.percentile(boots, 100 * adj(0.975))
    return round(theta, 2), round(float(lo), 2), round(float(hi), 2)


def _bootstrap_table(oof):
    rows = []
    for name, col in MODELS:
        mae = _bca(oof, col, "mae"); rmse = _bca(oof, col, "rmse"); disp = _bca(oof, col, "disp")
        rows.append(dict(Model=name, MAE=f"{mae[0]} [{mae[1]}, {mae[2]}]",
                         RMSE=f"{rmse[0]} [{rmse[1]}, {rmse[2]}]",
                         InterGroup_Disparity=f"{disp[0]} [{disp[1]}, {disp[2]}]"))
    pd.DataFrame(rows).to_csv(TABLES_DIR / "table10_bootstrap_ci.csv", index=False)


def _stats(oof):
    per = {name: (oof.assign(e=np.abs(oof["y"] - oof[col])).groupby("subject")["e"].mean())
           for name, col in MODELS}
    w1 = stats.wilcoxon(per["Baseline"], per["Optimized"])
    w2 = stats.wilcoxon(per["Optimized"], per["Mitigated"])
    f = stats.f_oneway(per["Baseline"].values, per["Optimized"].values, per["Mitigated"].values)
    res = {"wilcoxon_base_vs_opt": [round(float(w1.statistic), 3), float(w1.pvalue)],
           "wilcoxon_opt_vs_mit": [round(float(w2.statistic), 3), float(w2.pvalue)],
           "anova": [round(float(f.statistic), 3), float(f.pvalue)]}
    for name, col in [("baseline", "base"), ("mitigated", "mit")]:
        obs = _subj_stat(oof, col, np.unique(oof.subject.values), "disp")
        err = np.abs(oof["y"] - oof[col]).values; rng = np.random.RandomState(0); null = []
        for _ in range(10000):
            ds = []
            for ax, (c, order) in AXES.items():
                g = pd.Series(err).groupby(rng.permutation(oof[GCMAP[ax]].values)).mean().reindex(order)
                ds.append(g.max() - g.min())
            null.append(np.nanmean(ds))
        p = float((np.sum(np.array(null) >= obs) + 1) / (len(null) + 1))
        res[f"permutation_{name}"] = [round(obs, 2), p]
    json.dump(res, open(TABLES_DIR / "statistical_tests.json", "w"), indent=2)
    return res


def make_tables(oof, fold_metrics, work, hp_search):
    _table1(work); _table3(work)
    _overall_tables(fold_metrics); _fairness_tables(oof)
    _bootstrap_table(oof)
    hp_search.to_csv(TABLES_DIR / "hp_search.csv", index=False)
    res = _stats(oof)
    print("[analysis] tables + statistical tests written")
    return res
