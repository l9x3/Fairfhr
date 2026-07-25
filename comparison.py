import warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import torch, torch.nn as nn
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from models import MLP, all_metrics
from prep import AXES, GROUP, TARGET
from config import TABLES_DIR
torch.set_num_threads(1)

GCMAP = {"Age": "age_grp", "BMI": "bmi_grp", "Gestation": "ga_grp"}


class GradRev(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam): ctx.lam = lam; return x.view_as(x)
    @staticmethod
    def backward(ctx, g): return -ctx.lam * g, None


class AdvMLP(nn.Module):
    def __init__(self, n_sub=3):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(212, 128), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(64, 32), nn.ReLU())
        self.reg = nn.Linear(32, 1)
        self.adv = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, n_sub))
    def forward(self, x, lam=0.0):
        h = self.enc(x); return self.reg(h).squeeze(-1), self.adv(GradRev.apply(h, lam))


def _reweight(Xtr, ztr, w, epochs=150, seed=0):
    torch.manual_seed(seed); m = MLP(); opt = torch.optim.Adam(m.parameters(), 1e-3)
    Xt, zt, wt = torch.tensor(Xtr), torch.tensor(ztr), torch.tensor(w, dtype=torch.float32)
    for _ in range(epochs):
        m.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 256):
            idx = perm[i:i+256]; opt.zero_grad()
            (wt[idx] * torch.abs(m(Xt[idx]) - zt[idx])).mean().backward(); opt.step()
    m.eval(); return m


def _varpen(Xtr, ztr, sub_lab, lam=1.0, epochs=150, seed=0):
    torch.manual_seed(seed); m = MLP(); opt = torch.optim.Adam(m.parameters(), 1e-3)
    Xt, zt, sub = torch.tensor(Xtr), torch.tensor(ztr), torch.tensor(sub_lab)
    for _ in range(epochs):
        m.train(); ae = torch.abs(m(Xt) - zt)
        gmae = torch.stack([ae[sub == g].mean() for g in sub.unique() if (sub == g).sum() > 0])
        loss = ae.mean() + lam * torch.var(gmae)
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); return m


def _adv(Xtr, ztr, sub_lab, epochs=150, lam=0.5, seed=0):
    torch.manual_seed(seed); m = AdvMLP()
    opt = torch.optim.Adam(m.parameters(), 1e-3); l1 = nn.L1Loss(); ce = nn.CrossEntropyLoss()
    Xt, zt = torch.tensor(Xtr), torch.tensor(ztr); sub = torch.tensor(sub_lab, dtype=torch.long)
    for _ in range(epochs):
        m.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 256):
            idx = perm[i:i+256]; opt.zero_grad()
            pred, adv = m(Xt[idx], lam)
            (l1(pred, zt[idx]) + ce(adv, sub[idx])).backward(); opt.step()
    m.eval(); return m


def _pred(m, X, adv=False):
    with torch.no_grad():
        xt = torch.tensor(X)
        return (m(xt, 0.0)[0] if adv else m(xt)).numpy()


def _run_method(work, feat, method):
    X = work[feat].values.astype(np.float32); y = work[TARGET].values.astype(np.float32)
    groups = work[GROUP].values; grp = {ax: work[GCMAP[ax]].values for ax in AXES}
    gkf = GroupKFold(5); oof = {}
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups), 1):
        sc = StandardScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        my, sy = y[tr].mean(), y[tr].std(); ztr = (y[tr]-my)/sy; inv = lambda z: z*sy+my
        codes = pd.Categorical(grp["Age"][tr]).codes
        if method == "reweight":
            w = np.ones(len(tr))
            for ax in AXES:
                lab = grp[ax][tr]; vc = pd.Series(lab).value_counts()
                w *= (len(lab) / (len(vc) * vc.reindex(lab).values))
            m = _reweight(Xtr, ztr, w / w.mean(), seed=fold); pred = inv(_pred(m, Xte))
        elif method == "varpen":
            m = _varpen(Xtr, ztr, codes, seed=fold); pred = inv(_pred(m, Xte))
        else:
            m = _adv(Xtr, ztr, codes, seed=fold); pred = inv(_pred(m, Xte, adv=True))
        for i, gi in enumerate(te): oof[int(gi)] = pred[i]
    return np.array([oof[i] for i in range(len(X))])


def _summary(work, feat, pred):
    y = work[TARGET].values.astype(np.float32); grp = {ax: work[GCMAP[ax]].values for ax in AXES}
    err = np.abs(y - pred); ov = all_metrics(y, pred); per, disp = {}, {}
    for ax, (col, order) in AXES.items():
        g = pd.Series(err).groupby(grp[ax]).mean().reindex(order)
        per[ax] = {sg: round(float(g[sg]), 2) for sg in order}
        disp[ax] = round(float(g.max() - g.min()), 2)
    return ov, per, disp


def make_tables(work, feat, oof):
    prop = np.full(len(work), np.nan)
    for _, r in oof.iterrows():
        prop[int(r["idx"])] = r["mit"]
    methods = {"Reweighting": _run_method(work, feat, "reweight"),
               "Variance Penalty": _run_method(work, feat, "varpen"),
               "Adversarial": _run_method(work, feat, "adv"),
               "SHAP-Guided (Ours)": prop}
    rows15, rows16 = [], []
    for name, pred in methods.items():
        ov, per, disp = _summary(work, feat, pred)
        rows15.append(dict(Method=name, MAE=round(ov["MAE"], 2), RMSE=round(ov["RMSE"], 2),
                           Age_dFair=disp["Age"], BMI_dFair=disp["BMI"], Gest_dFair=disp["Gestation"],
                           InterGroup=round(float(np.mean(list(disp.values()))), 2)))
        row = {"Method": name}
        for ax, order in [("Age", AXES["Age"][1]), ("BMI", AXES["BMI"][1]), ("Gestation", AXES["Gestation"][1])]:
            for sg in order: row[f"{ax}:{sg}"] = per[ax][sg]
        rows16.append(row)
    pd.DataFrame(rows15).to_csv(TABLES_DIR / "table15_method_comparison.csv", index=False)
    pd.DataFrame(rows16).to_csv(TABLES_DIR / "table16_per_subgroup_comparison.csv", index=False)
    pd.DataFrame([
        {"Method": "Reweighting", "Type": "Pre-processing", "Retraining": "Yes", "Interpretable": "Low"},
        {"Method": "Adversarial", "Type": "In-processing", "Retraining": "Yes", "Interpretable": "Low"},
        {"Method": "Variance Penalty", "Type": "In-processing", "Retraining": "Yes", "Interpretable": "Medium"},
        {"Method": "SHAP-Guided (Ours)", "Type": "Post-processing", "Retraining": "No", "Interpretable": "High"},
    ]).to_csv(TABLES_DIR / "table17_operational.csv", index=False)
    print("[comparison] Tables 15-17 written")
