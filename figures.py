import warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
from models import np_forward, train_mlp, model_to_chromosome
from debias import shap_values
from prep import AXES, GROUP, TARGET, AGE_ORDER, BMI_ORDER, GA_ORDER
from config import FIGURES_DIR

plt.rcParams.update({"figure.dpi": 120, "font.size": 9})
GCMAP = {"Age": "age_grp", "BMI": "bmi_grp", "Gestation": "ga_grp"}


def _demographics(subj):
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    ax[0].hist(subj["Age"], bins=8, color="#6699e6", edgecolor="w")
    ax[0].set_title("Distribution of Maternal Age"); ax[0].set_xlabel("Age (years)"); ax[0].set_ylabel("Count")
    ax[1].hist(subj["BMI"], bins=8, color="#3fb6a8", edgecolor="w")
    ax[1].set_title("Distribution of Maternal BMI"); ax[1].set_xlabel("BMI (kg/m$^2$)")
    ax[2].hist(subj["Gestational_age"], bins=8, color="#f0ad4e", edgecolor="w")
    ax[2].set_title("Distribution of Gestational Age"); ax[2].set_xlabel("Gestational Age (weeks)")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "demographics.png"); plt.close(fig)


def _fhr(work):
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    for a, (order, gcol, ttl) in zip(ax, [
        (AGE_ORDER, "age_grp", "FHR by Age Group"),
        (BMI_ORDER, "bmi_grp", "FHR by BMI Group"),
        (GA_ORDER, "ga_grp", "FHR by Gestational Age Group")]):
        a.boxplot([work[work[gcol] == g][TARGET].values for g in order], tick_labels=order)
        a.set_title(ttl); a.set_ylabel("FHR (bpm)")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "fhr_by_subgroup.png"); plt.close(fig)


def _correlation(work):
    d = work[["Age", "BMI", "Gestational_age", TARGET]].rename(
        columns={"Gestational_age": "Gest. age", TARGET: "FHR"})
    c = d.corr().values
    fig, ax = plt.subplots(figsize=(5, 4.2)); im = ax.imshow(c, cmap="viridis", vmin=-0.1, vmax=1)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(d.columns, rotation=30, ha="right"); ax.set_yticklabels(d.columns)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{c[i,j]:.2f}", ha="center", va="center",
                    color="w" if c[i, j] < 0.6 else "k")
    ax.set_title("Correlation Matrix of Key Variables"); fig.colorbar(im, shrink=0.8)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "correlation.png"); plt.close(fig)



def _kval(art):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    titles = {"Age": "Maternal Age Subgroups", "BMI": "BMI Subgroups", "Gestation": "Gestational Age Subgroups"}
    for a, axis in zip(ax, ["Age", "BMI", "Gestation"]):
        ks, cv = art["kcurves"][axis]
        for sg, ys in cv.items(): a.plot(ks, ys, marker="o", label=sg)
        a.axvline(10, ls="--", color="gray"); a.set_title(titles[axis])
        a.set_xlabel("K-value"); a.set_ylabel("MAE (bpm)"); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "kvalue_peraxis.png"); plt.close(fig)


def _elbow(art):
    e = art["elbow"]; fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(e["k"], e["mae"], marker="o", color="#1f77b4")
    for k, m in zip(e["k"], e["mae"]):
        ax.annotate(f"{m:.2f}", (k, m), textcoords="offset points", xytext=(0, 6), fontsize=7)
    ax.axvline(10, ls="--", color="r"); ax.set_xlabel("k (Number of Bias-Sensitive Features)")
    ax.set_ylabel("Mean held-out MAE"); ax.set_title("Elbow analysis: mean MAE vs k")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "elbow.png"); plt.close(fig)


def _kstability(art):
    freq = art["kstability"]; ks = sorted(freq); vals = [freq[k] for k in ks]
    fig, ax = plt.subplots(figsize=(6.5, 3.6)); bars = ax.bar(ks, vals, color="#4c8fd6", edgecolor="w")
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.3, f"{v}%", ha="center", fontsize=7)
    ax.set_xlabel("k"); ax.set_ylabel("Selection Frequency (%)")
    ax.set_title("Bootstrap stability of selected k (500 resamples)")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "kstability.png"); plt.close(fig)


def _importance(work, feat, art):
    sv, tr, sub = art["shap"], art["tr"], art["sub"]
    X = work[feat].values.astype(np.float32)
    Xz = ((X[tr] - art["scaler_mean"]) / art["scaler_scale"])[sub]
    plt.figure(figsize=(7, 8))
    shap.summary_plot(sv, Xz, feature_names=feat, max_display=25, show=False, plot_type="dot")
    plt.title("SHAP beeswarm (feature effects on model output)")
    plt.tight_layout(); plt.savefig(FIGURES_DIR / "fig8a_beeswarm.png"); plt.close()
    plt.figure(figsize=(6.5, 8))
    shap.summary_plot(sv, Xz, feature_names=feat, max_display=30, show=False, plot_type="bar")
    plt.title("SHAP feature importance (mean |SHAP|)")
    plt.tight_layout(); plt.savefig(FIGURES_DIR / "importance.png"); plt.close()


def _force(work, feat, art):
    sv, chrom, tr, sub = art["shap"], art["chrom"], art["tr"], art["sub"]
    my, sy = art["my"], art["sy"]
    X = work[feat].values.astype(np.float32); y = work[TARGET].values.astype(np.float32)
    Xz = ((X[tr] - art["scaler_mean"]) / art["scaler_scale"])[sub]
    ypred = np_forward(chrom, Xz) * sy + my; base_val = float(np.mean(ypred))
    fig, ax = plt.subplots(2, 1, figsize=(11, 4.6))
    for r, s in enumerate([np.argmin(ypred), np.argmax(ypred)]):
        contrib = sv[s] * sy; top = np.argsort(np.abs(contrib))[::-1][:8]
        vals = contrib[top]; colors = ["#d9534f" if v > 0 else "#4c8fd6" for v in vals]
        ax[r].barh(range(len(top))[::-1], vals, color=colors)
        ax[r].set_yticks(range(len(top))[::-1]); ax[r].set_yticklabels([feat[i] for i in top], fontsize=7)
        ax[r].set_title(f"Sample {r+1}: Actual FHR {y[tr][sub][s]:.0f}, Predicted {ypred[s]:.1f} "
                        f"(base {base_val:.1f})", fontsize=9)
        ax[r].set_xlabel("SHAP contribution (bpm)")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "force.png"); plt.close(fig)


def _subgroupshap(work, feat, art):
    sv_opt, tr, sub = art["shap"], art["tr"], art["sub"]
    X = work[feat].values.astype(np.float32); y = work[TARGET].values.astype(np.float32)
    Xz = (X[tr] - art["scaler_mean"]) / art["scaler_scale"]; my, sy = art["my"], art["sy"]
    base = train_mlp(Xz, (y[tr]-my)/sy, Xz, (y[tr]-my)/sy, batch=512, lr=1e-3, epochs=80, seed=1)
    base_chrom = model_to_chromosome(base)
    sv_base = shap_values(base_chrom, Xz[sub], Xz, bg_k=10, nsamples=80)
    gc = {ax: work[GCMAP[ax]].values[tr][sub] for ax in AXES}
    top = np.argsort(np.mean(np.abs(sv_opt), 0))[::-1][:13]; names = [feat[i] for i in top]
    fig, axall = plt.subplots(3, 2, figsize=(12, 12))
    for row, (axis, order) in enumerate([("Age", AGE_ORDER), ("BMI", BMI_ORDER), ("Gestation", GA_ORDER)]):
        for col, (sv, tag) in enumerate([(sv_base, "Baseline"), (sv_opt, "Mitigated")]):
            a = axall[row][col]; yy = np.arange(len(top)); wbar = 0.25
            for gi, g in enumerate(order):
                m = gc[axis] == g
                if m.sum() == 0: continue
                a.barh(yy + gi*wbar, np.mean(np.abs(sv[m][:, top]), 0), height=wbar, label=g)
            a.set_yticks(yy + wbar); a.set_yticklabels(names, fontsize=6); a.invert_yaxis()
            a.set_xlabel("Mean |SHAP| value"); a.set_title(f"{axis} subgroups ({tag})", fontsize=9)
            a.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "subgroup_shap.png"); plt.close(fig)


def make_figures(work, feat, art):
    subj = work.drop_duplicates(GROUP)
    _demographics(subj); _fhr(work); _correlation(work);
    _kval(art); _elbow(art); _kstability(art)
    _importance(work, feat, art); _force(work, feat, art); _subgroupshap(work, feat, art)
    print("[figures] figures 1-10 written")
