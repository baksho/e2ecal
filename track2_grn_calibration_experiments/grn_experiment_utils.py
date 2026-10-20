"""Shared utilities for the lightweight GRN calibration experiments."""
from __future__ import annotations
import os, random, math
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

SEED = 42
def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def make_pair_sequences(expr, edges, tf_indices=None, max_conditions=64, max_pairs=12000,
                        negatives_per_positive=5, seed=SEED):
    """
    expr: genes x conditions numeric matrix; edges: iterable of (regulator, target, label)
    Returns one sequence per directed pair: [conditions, 2] (regulator/target expression).
    A fixed random subset of conditions makes the experiment lightweight.
    """
    rng = np.random.default_rng(seed)
    expr = np.asarray(expr, dtype=np.float32)
    expr = np.nan_to_num(expr, nan=0.0, posinf=0.0, neginf=0.0)
    # Log transform only if values are nonnegative and strongly right-skewed/count-like.
    if np.min(expr) >= 0 and np.percentile(expr, 99) > 20:
        expr = np.log1p(expr)
    # Per-gene standardisation across conditions.
    mu = expr.mean(axis=1, keepdims=True)
    sd = expr.std(axis=1, keepdims=True)
    expr = (expr - mu) / np.maximum(sd, 1e-5)
    n_genes, n_cond = expr.shape
    cond_idx = np.arange(n_cond)
    if n_cond > max_conditions:
        cond_idx = np.sort(rng.choice(cond_idx, size=max_conditions, replace=False))
    expr = expr[:, cond_idx]
    edge_map = {}
    for a,b,y in edges:
        if a != b and 0 <= int(a) < n_genes and 0 <= int(b) < n_genes:
            edge_map[(int(a),int(b))] = int(y)
    regulators = list(range(n_genes)) if tf_indices is None else sorted(set(int(i) for i in tf_indices if 0 <= int(i) < n_genes))
    positives = [(a,b,1) for (a,b),y in edge_map.items() if y == 1 and a in regulators]
    if not positives:
        raise ValueError("No positive regulator-target edges matched the expression matrix gene order.")
    positive_keys = {(a,b) for a,b,_ in positives}
    neg_pool = [(a,b,0) for a in regulators for b in range(n_genes) if a != b and (a,b) not in positive_keys]
    n_neg = min(len(neg_pool), max(len(positives)*negatives_per_positive, 100), max_pairs-len(positives))
    if n_neg > 0:
        take = rng.choice(len(neg_pool), size=n_neg, replace=False)
        negatives = [neg_pool[i] for i in take]
    else: negatives=[]
    pairs = positives + negatives
    rng.shuffle(pairs)
    if len(pairs) > max_pairs:
        # Retain all positives where possible, sample negatives to the cap.
        pos = [p for p in pairs if p[2] == 1]
        neg = [p for p in pairs if p[2] == 0]
        keep_neg = max(0, max_pairs-len(pos))
        neg = [neg[i] for i in rng.choice(len(neg), keep_neg, replace=False)] if keep_neg < len(neg) else neg
        pairs = pos + neg
        rng.shuffle(pairs)
    X = np.stack([np.stack([expr[a], expr[b]], axis=-1) for a,b,y in pairs]).astype("float32")
    y = np.asarray([p[2] for p in pairs], dtype="float32")
    pair_ids = np.asarray([f"{a}->{b}" for a,b,_ in pairs])
    return X, y, pair_ids

class PairDataset(Dataset):
    def __init__(self, X, y):
        self.X=torch.tensor(X, dtype=torch.float32); self.y=torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

class PairTransformer(nn.Module):
    """Small permutation-invariant Transformer over condition-level TF/target expression pairs."""
    def __init__(self, d_model=16, nhead=2, layers=1, dropout=0.1):
        super().__init__()
        self.proj=nn.Linear(2,d_model)
        enc_layer=nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
            dim_feedforward=2*d_model, dropout=dropout, batch_first=True, norm_first=True)
        self.encoder=nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head=nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model),
                                nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model,1))
    def forward(self,x):
        h=self.encoder(self.proj(x))
        h=h.mean(dim=1)
        return self.head(h).squeeze(-1)

def soft_calibration_penalty(logits, y, n_bins=12, bandwidth=0.08):
    p=torch.sigmoid(logits)
    centers=torch.linspace(0,1,n_bins,device=p.device)
    # Smooth, normalized memberships; detach weights to avoid unstable bin-assignment gradients.
    w=torch.softmax(-((p[:,None]-centers[None,:])**2)/(2*bandwidth**2), dim=1).detach()
    mass=w.sum(dim=0).clamp_min(1e-6)
    conf=(w*p[:,None]).sum(dim=0)/mass
    acc=(w*y[:,None]).sum(dim=0)/mass
    occupied=(w.sum(dim=0)>1e-3).float()
    return (((conf-acc)**2)*occupied).sum()/occupied.sum().clamp_min(1)

def ece_score(y, p, n_bins=10):
    y=np.asarray(y).astype(float); p=np.asarray(p).astype(float)
    edges=np.linspace(0,1,n_bins+1); out=0.0
    for lo,hi in zip(edges[:-1],edges[1:]):
        m=(p>=lo)&(p<(hi if hi<1 else hi+1e-9))
        if m.any(): out += m.mean()*abs(y[m].mean()-p[m].mean())
    return float(out)

def evaluate(y, p):
    y=np.asarray(y).astype(int); p=np.clip(np.asarray(p),1e-6,1-1e-6)
    return {"AUROC": float(roc_auc_score(y,p)) if len(np.unique(y))>1 else float("nan"),
            "AUPR": float(average_precision_score(y,p)),
            "Brier": float(brier_score_loss(y,p)), "ECE10": ece_score(y,p)}

def split_indices(y, seed=SEED):
    idx=np.arange(len(y))
    tr, rem=train_test_split(idx, test_size=0.30, random_state=seed, stratify=y)
    va, te=train_test_split(rem, test_size=0.50, random_state=seed, stratify=np.asarray(y)[rem])
    return tr,va,te

def train_model(X, y, train_idx, val_idx, lam=1.0, base_loss="bce",
                model_size="small", epochs=35, batch_size=64, lr=1e-3,
                seed=SEED, patience=6, device=None):
    """Train a Transformer with a predictive loss and a smooth calibration penalty.

    Objective: (1-lam)*soft_calibration_penalty + lam*base_loss.
    `base_loss` is "bce" or "brier"; `model_size` is "small" or "medium".
    Lambda=1 is predictive-only training; lower lambda gives more weight to the
    calibration surrogate. Validation Brier score is used for early stopping.
    """
    set_seed(seed)
    device=device or ("cuda" if torch.cuda.is_available() else "cpu")
    if model_size == "small":
        model=PairTransformer(d_model=16, nhead=2, layers=1, dropout=0.1).to(device)
    elif model_size == "medium":
        model=PairTransformer(d_model=32, nhead=4, layers=2, dropout=0.1).to(device)
    else:
        raise ValueError("model_size must be 'small' or 'medium'")
    if base_loss not in {"bce", "brier"}:
        raise ValueError("base_loss must be 'bce' or 'brier'")
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    train_loader=DataLoader(PairDataset(X[train_idx],y[train_idx]),batch_size=batch_size,shuffle=True)
    val_loader=DataLoader(PairDataset(X[val_idx],y[val_idx]),batch_size=batch_size,shuffle=False)
    best=float("inf"); best_state=None; wait=0; history=[]
    for ep in range(epochs):
        model.train(); train_losses=[]
        for xb,yb in train_loader:
            xb=xb.to(device); yb=yb.to(device)
            opt.zero_grad(); logits=model(xb)
            if base_loss == "bce":
                predictive = nn.functional.binary_cross_entropy_with_logits(logits, yb)
            else:
                predictive = torch.mean((torch.sigmoid(logits) - yb) ** 2)
            cal=soft_calibration_penalty(logits,yb)
            # Convex combination in the same convention as the thesis.
            loss=(1.0-lam)*cal + lam*predictive
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),2.0); opt.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval(); vp=[]; vy=[]
        with torch.no_grad():
            for xb,yb in val_loader:
                vp.extend(torch.sigmoid(model(xb.to(device))).cpu().numpy()); vy.extend(yb.numpy())
        val_brier=evaluate(vy,vp)["Brier"]
        history.append({"epoch":ep+1,"train_loss":float(np.mean(train_losses)),"val_brier":val_brier})
        if val_brier < best-1e-5:
            best=val_brier; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; wait=0
        else:
            wait+=1
            if wait>=patience: break
    if best_state: model.load_state_dict(best_state)
    return model, pd.DataFrame(history)

def predict(model, X, batch_size=128, device=None):
    device=device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval(); preds=[]
    with torch.no_grad():
        for start in range(0,len(X),batch_size):
            xb=torch.tensor(X[start:start+batch_size],dtype=torch.float32,device=device)
            preds.extend(torch.sigmoid(model(xb)).cpu().numpy())
    return np.asarray(preds)

def fit_isotonic(y, p):
    """Fit non-parametric isotonic recalibration on validation probabilities."""
    from sklearn.isotonic import IsotonicRegression
    model=IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    model.fit(np.asarray(p), np.asarray(y))
    return model

def fit_temperature(y, p):
    """Fit a single temperature on validation probabilities by minimizing NLL."""
    from scipy.optimize import minimize_scalar
    y=np.asarray(y).astype(float); p=np.clip(np.asarray(p),1e-6,1-1e-6)
    logits=np.log(p/(1-p))
    def obj(logT):
        z=logits/np.exp(logT)
        return float(np.mean(np.logaddexp(0,z)-y*z))
    result=minimize_scalar(obj,bounds=(-3,3),method="bounded")
    return float(np.exp(result.x))

def apply_isotonic(p, calibrator):
    return np.clip(calibrator.predict(np.asarray(p)), 1e-6, 1-1e-6)

def apply_temperature(p, T):
    p=np.clip(np.asarray(p),1e-6,1-1e-6); z=np.log(p/(1-p))/T
    return 1/(1+np.exp(-z))
