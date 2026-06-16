"""
Training / evaluation pipeline for one experimental run.

run_experiment() reproduces one row of Table 4 (with ACFD) or one
configuration of Table 5 (with/without ACFD). The critical methodological
points enforced here:

  * the 70/15/15 split is made on REAL data;
  * the Min-Max scaler is fit on the TRAIN split only (no leakage);
  * windows are labelled by their terminal flow;
  * ACFD synthetic samples are added to the TRAIN split ONLY; the validation
    and test splits are 100% real;
  * early stopping is on the validation F1.
"""
import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

from .models import ACFD, LongformerAPTDetector


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def make_windows(X, y, window, sub=None):
    """Window = rows i..i+w-1; label = the LAST flow of the window."""
    xs = np.lib.stride_tricks.sliding_window_view(X, (window, X.shape[1]))[:, 0]
    ys = y[window - 1:]
    out = [torch.tensor(xs.copy(), dtype=torch.float32),
           torch.tensor(ys, dtype=torch.long)]
    if sub is not None:
        out.append(torch.tensor(sub[window - 1:], dtype=torch.long))
    return out


@torch.no_grad()
def evaluate(model, dl, device):
    model.eval()
    ys, ps, probs = [], [], []
    for xb, yb in dl:
        logits = model(xb.to(device))
        ys.append(yb)
        ps.append(logits.argmax(-1).cpu())
        probs.append(logits.softmax(-1)[:, 1].cpu())
    return (torch.cat(ys).numpy(), torch.cat(ps).numpy(), torch.cat(probs).numpy())


def count_params(input_dim, cfg):
    m = LongformerAPTDetector(input_dim, cfg)
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    del m
    return n


def run_experiment(seed, data, cfg, device, use_acfd=True, keep_artifacts=False):
    """Run one full train/eval cycle. `data` is the tuple returned by
    load_or_build_dataset. Returns (metrics_dict, artifacts_or_None)."""
    X_all, y_all, sub_codes, features, sub_names = data
    set_seed(seed)
    print(f"\n{'=' * 64}\n  SEED {seed} | ACFD = {use_acfd}\n{'=' * 64}")

    # ---- split 70/15/15 on REAL data ------------------------------------
    X_tr, X_tmp, y_tr, y_tmp, s_tr, _ = train_test_split(
        X_all, y_all, sub_codes, test_size=0.30, stratify=y_all, random_state=seed)
    X_va, X_te, y_va, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=seed)[:4]

    scaler = MinMaxScaler().fit(X_tr)              # fit on TRAIN only
    X_tr, X_va, X_te = scaler.transform(X_tr), scaler.transform(X_va), scaler.transform(X_te)

    w = cfg.window
    Xw_tr, yw_tr, sw_tr = make_windows(X_tr, y_tr, w, s_tr)
    Xw_va, yw_va = make_windows(X_va, y_va, w)[:2]
    Xw_te, yw_te = make_windows(X_te, y_te, w)[:2]
    print(f"Windows: train {tuple(Xw_tr.shape)} | val {tuple(Xw_va.shape)} | "
          f"test {tuple(Xw_te.shape)} | train APT-rate {yw_tr.float().mean():.4f}")

    # ---- sanity check: the data must carry a learnable signal -----------
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(Xw_tr[:, -1, :].numpy(), yw_tr.numpy())
    print(f"[Sanity] LogReg val F1 = "
          f"{f1_score(yw_va.numpy(), clf.predict(Xw_va[:, -1, :].numpy())):.4f}")

    # ---- ACFD augmentation on the TRAIN split ONLY ----------------------
    acfd = None
    if use_acfd:
        x_dim = w * len(features)
        mask = yw_tr == 1
        cond_pool = sw_tr[mask]
        n_cond = len(sub_names)
        print(f"Pre-training ACFD on {int(mask.sum())} real attack windows, "
              f"{n_cond} conditions...")
        acfd = ACFD(x_dim, n_cond, cfg, device)
        acfd.pretrain(Xw_tr[mask].reshape(-1, x_dim), cond_pool)
        n_need = int((yw_tr == 0).sum() - (yw_tr == 1).sum())
        if n_need > 0:
            print(f"Sampling {n_need:,} synthetic minority windows (T={cfg.t_steps})...")
            t0 = time.time()
            synth = acfd.sample(n_need, cond_pool, x_dim).reshape(-1, w, len(features))
            print(f"   done in {(time.time() - t0) / 60:.1f} min")
            Xw_bal = torch.cat([Xw_tr, synth])
            yw_bal = torch.cat([yw_tr, torch.ones(len(synth), dtype=torch.long)])
        else:
            print("Training set already balanced; no synthetic samples needed.")
            Xw_bal, yw_bal = Xw_tr, yw_tr
    else:
        Xw_bal, yw_bal = Xw_tr, yw_tr

    # ---- train the Longformer detector ----------------------------------
    model = LongformerAPTDetector(len(features), cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.max_epochs)
    crit = nn.CrossEntropyLoss()
    train_dl = DataLoader(TensorDataset(Xw_bal, yw_bal),
                          batch_size=cfg.batch_size, shuffle=True)
    val_dl = DataLoader(TensorDataset(Xw_va, yw_va), batch_size=1024)

    best_f1, best_state, patience = -1.0, None, cfg.patience
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        tot = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item()
        sched.step()
        yv, pv, _ = evaluate(model, val_dl, device)
        vf1 = f1_score(yv, pv)
        print(f"  Epoch {epoch:02d} | loss {tot / len(train_dl):.4f} | "
              f"val F1 {vf1:.4f} | pred-APT {pv.mean():.3f}")
        if vf1 > best_f1:
            best_f1, patience = vf1, cfg.patience
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience -= 1
            if patience == 0:
                print(f"  Early stop (best val F1 {best_f1:.4f})")
                break
    model.load_state_dict(best_state)

    # ---- evaluate on the REAL test set ----------------------------------
    test_dl = DataLoader(TensorDataset(Xw_te, yw_te), batch_size=1024)
    y, p, prob = evaluate(model, test_dl, device)
    metrics = {
        "seed": seed,
        "use_acfd": use_acfd,
        "accuracy_%": accuracy_score(y, p) * 100,
        "precision": precision_score(y, p, zero_division=0),
        "recall": recall_score(y, p, zero_division=0),
        "f1": f1_score(y, p, zero_division=0),
        "auc_roc": roc_auc_score(y, prob),
    }
    print("  TEST: " + " | ".join(
        f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, float)))
    print(confusion_matrix(y, p))

    artifacts = None
    if keep_artifacts:
        artifacts = {"model": model, "acfd": acfd, "scaler": scaler,
                     "Xw_tr": Xw_tr, "yw_tr": yw_tr, "Xw_te": Xw_te, "yw_te": yw_te}
    else:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return metrics, artifacts
