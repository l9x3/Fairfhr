import numpy as np
import shap
from sklearn.linear_model import Ridge
from models import np_forward

shap.utils._show_progress.show_progress = lambda *a, **k: a[0]  # silence bars


def shap_values(chrom, X_explain, X_background, bg_k=10, nsamples=100, seed=0):
    """KernelExplainer SHAP values for a chromosome-encoded MLP."""
    f = lambda Z: np_forward(chrom, Z)
    bg = shap.kmeans(X_background, bg_k)
    expl = shap.KernelExplainer(f, bg, silent=True)
    sv = expl.shap_values(X_explain, nsamples=nsamples, silent=True)
    return np.asarray(sv)


def bias_sensitive_features(shap_tr, grp_labels, axes, k=10):
    """
    shap_tr    : (n, p) SHAP matrix on training samples
    grp_labels : dict axis -> array(n,) subgroup label per sample
    axes       : dict axis -> (col, order)
    Returns    : indices of the top-k features by mean inter-subgroup SHAP
                 disparity, averaged over the three demographic axes.
    """
    p = shap_tr.shape[1]
    disparity = np.zeros(p)
    for axis, (_, order) in axes.items():
        labs = grp_labels[axis]
        mu = []
        for g in order:
            m = labs == g
            if m.sum() == 0:
                continue
            mu.append(np.mean(np.abs(shap_tr[m]), axis=0))     # mu_{g,j}
        mu = np.vstack(mu)
        disparity += mu.max(0) - mu.min(0)                     # Delta_j
    disparity /= len(axes)
    top = np.argsort(disparity)[::-1][:k]
    return top, disparity


def fit_corrector(shap_tr, resid_tr, feat_idx, alpha=1.0):
    """Ridge corrector on the restricted SHAP subvector."""
    reg = Ridge(alpha=alpha)
    reg.fit(shap_tr[:, feat_idx], resid_tr)
    return reg


def apply_correction(yhat, shap_te, corrector, feat_idx, eps=0.5, delta=0.1,
                     clip=15.0):
    """y_fair = yhat - r_hat, with residual/norm safeguards (Sec 2.4.3)."""
    phi = shap_te[:, feat_idx]
    r_hat = corrector.predict(phi)
    r_hat = np.clip(r_hat, -clip, clip)             # guard against extrapolation
    norm = np.linalg.norm(phi, axis=1)
    suppress = (np.abs(r_hat) < eps) | (norm < delta)
    r_hat = np.where(suppress, 0.0, r_hat)
    return yhat - r_hat, r_hat


from sklearn.neural_network import MLPRegressor
def fit_corrector_mlp(shap_tr, resid_tr, feat_idx, seed=0):
    """Lightweight MLP residual head (32->16->1), paper Table 3."""
    reg = MLPRegressor(hidden_layer_sizes=(32, 16), activation="relu",
                       solver="adam", learning_rate_init=1e-3, alpha=1e-4,
                       max_iter=400, random_state=seed, early_stopping=True,
                       n_iter_no_change=20)
    reg.fit(shap_tr[:, feat_idx], resid_tr)
    return reg
