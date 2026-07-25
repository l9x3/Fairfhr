import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)
LAYERS = [212, 128, 64, 32, 1]

# ---- metrics ----------------------------------------------------------------
def mae(y, p):   return float(np.mean(np.abs(y - p)))
def rmse(y, p):  return float(np.sqrt(np.mean((y - p) ** 2)))
def r2(y, p):
    ss_res = np.sum((y - p) ** 2); ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
def mape(y, p):  return float(100 * np.mean(np.abs((y - p) / y)))
def ppa10(y, p): return float(np.mean(np.abs((y - p) / y) <= 0.10) * 100)
def all_metrics(y, p):
    return dict(MAE=mae(y, p), RMSE=rmse(y, p), R2=r2(y, p),
                MAPE=mape(y, p), PPA10=ppa10(y, p))

# ---- torch MLP --------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, layers=LAYERS, p_drop=0.3):
        super().__init__()
        blocks = []
        for i in range(len(layers) - 2):
            blocks += [nn.Linear(layers[i], layers[i + 1]), nn.ReLU(), nn.Dropout(p_drop)]
        blocks += [nn.Linear(layers[-2], layers[-1])]
        self.net = nn.Sequential(*blocks)
    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_mlp(Xtr, ytr, Xval, yval, batch=512, lr=1e-2, epochs=120, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    model = MLP(); opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    lossf = nn.L1Loss()
    Xt = torch.tensor(Xtr, dtype=torch.float32); yt = torch.tensor(ytr, dtype=torch.float32)
    Xv = torch.tensor(Xval, dtype=torch.float32); yv = torch.tensor(yval, dtype=torch.float32)
    n = len(Xt); best_val, best_state, patience, bad = np.inf, None, 25, 0
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]; opt.zero_grad()
            loss = lossf(model(Xt[idx]), yt[idx]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vmae = float(torch.mean(torch.abs(model(Xv) - yv)))
        if vmae < best_val - 1e-4:
            best_val = vmae; best_state = {k: v.clone() for k, v in model.state_dict().items()}; bad = 0
        else:
            bad += 1
            if bad >= patience: break
    if best_state is not None: model.load_state_dict(best_state)
    model.eval(); return model

def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).numpy()

# ---- chromosome utilities ---------------------------------------------------
def _shapes(layers=LAYERS):
    shapes = []
    for i in range(len(layers) - 1):
        shapes.append((layers[i + 1], layers[i])); shapes.append((layers[i + 1],))
    return shapes

def model_to_chromosome(model):
    return np.concatenate([p.detach().numpy().ravel() for _, p in model.named_parameters()])

def chromosome_to_arrays(chrom, layers=LAYERS):
    arrays, off = [], 0
    for shp in _shapes(layers):
        size = int(np.prod(shp)); arrays.append(chrom[off:off + size].reshape(shp)); off += size
    return arrays

def np_forward(chrom, X, layers=LAYERS):
    a = X; arrs = chromosome_to_arrays(chrom, layers); nl = len(layers) - 1
    for li in range(nl):
        W, b = arrs[2 * li], arrs[2 * li + 1]; a = a @ W.T + b
        if li < nl - 1: a = np.maximum(a, 0.0)
    return a.ravel()

CHROM_LEN = sum(int(np.prod(s)) for s in _shapes())

# ---- GGA-MLP (Sec 2.4.2) ----------------------------------------------------
def gga_optimize(base_model, Xval, yval, pop_size=30, generations=40,
                 elite_frac=0.30, init_sigma=0.02, mut_sigma=0.02,
                 mut_frac=0.05, tol=1e-3, seed=0, verbose=False):
    rng = np.random.RandomState(seed)
    base = model_to_chromosome(base_model).astype(np.float64)
    def fitness(ch): return float(np.mean(np.abs(np_forward(ch, Xval) - yval)))
    pop = [base.copy()]
    for _ in range(pop_size - 1):
        pop.append(base + rng.normal(0, init_sigma, size=base.shape))
    pop = np.array(pop); fit = np.array([fitness(c) for c in pop])
    n_elite = max(1, int(elite_frac * pop_size)); best_hist = [fit.min()]
    for gen in range(generations):
        order = np.argsort(fit); pop, fit = pop[order], fit[order]
        elites = pop[:n_elite].copy(); new_pop = [pop[0].copy()]
        while len(new_pop) < pop_size:
            i, j = rng.randint(n_elite), rng.randint(n_elite)
            w = rng.uniform(0.3, 0.7); child = w * elites[i] + (1 - w) * elites[j]
            parent_fit = min(fit[i], fit[j]); mask = rng.rand(child.size) < mut_frac
            cand = child.copy(); cand[mask] += rng.normal(0, mut_sigma, size=int(mask.sum()))
            if fitness(cand) <= parent_fit: child = cand
            new_pop.append(child)
        pop = np.array(new_pop); fit = np.array([fitness(c) for c in pop])
        best_hist.append(fit.min())
        if verbose and gen % 5 == 0: print(f"    gen {gen:2d}  best val MAE={fit.min():.4f}")
        if gen > 5 and abs(best_hist[-6] - best_hist[-1]) < tol: break
    best = pop[int(np.argmin(fit))]; return best, best_hist

def chromosome_to_model(chrom):
    model = MLP(); arrs = chromosome_to_arrays(chrom); sd = model.state_dict()
    for k, arr in zip(list(sd.keys()), arrs):
        sd[k] = torch.tensor(arr, dtype=torch.float32)
    model.load_state_dict(sd); model.eval(); return model


# ---- memetic local refinement of a GA solution (Adam + MAE, early stop) -----
def refine_chromosome(chrom, Xtr, ytr, Xval, yval, lr=1e-3, epochs=120, seed=0):
    """Gradient fine-tuning of a GA chromosome -- standard memetic GA local
    search that turns the GA's good global init into a locally optimal MLP."""
    torch.manual_seed(seed)
    model = chromosome_to_model(chrom)
    for p in model.parameters(): p.requires_grad_(True)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.L1Loss()
    Xt = torch.tensor(Xtr, dtype=torch.float32); yt = torch.tensor(ytr, dtype=torch.float32)
    Xv = torch.tensor(Xval, dtype=torch.float32); yv = torch.tensor(yval, dtype=torch.float32)
    n = len(Xt); best, best_state, bad, patience = np.inf, None, 0, 20
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n)
        for i in range(0, n, 512):
            idx = perm[i:i+512]; opt.zero_grad()
            loss = lossf(model(Xt[idx]), yt[idx]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vm = float(torch.mean(torch.abs(model(Xv) - yv)))
        if vm < best - 1e-4:
            best, best_state, bad = vm, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience: break
    if best_state is not None: model.load_state_dict(best_state)
    model.eval()
    return model_to_chromosome(model)
