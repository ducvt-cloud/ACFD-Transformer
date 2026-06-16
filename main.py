"""
Entry point for the ACFD-Transformer experiments.

Examples
--------
# Main result (Table 4, Longformer row): 3 seeds, with ACFD augmentation
python main.py --data-dir data --out-dir outputs

# Also run the ablation (Table 5): "Without ACFD" on the first seed
python main.py --data-dir data --out-dir outputs --ablation

# Quick single-seed smoke test
python main.py --seeds 42

All metrics are written to JSON files in the output directory, and the best
seed-42 model checkpoint is saved. Every reported number in the paper is
traceable to one of these JSON files.
"""
import os
import json
import argparse
import numpy as np
import torch

from src.config import Config
from src.data import load_or_build_dataset
from src.pipeline import run_experiment, count_params


def parse_args():
    cfg = Config()
    ap = argparse.ArgumentParser(description="ACFD-Transformer APT detection pipeline")
    ap.add_argument("--data-dir", default=cfg.data_dir,
                    help="directory containing the two CICAPT-IIoT network CSV files")
    ap.add_argument("--out-dir", default=cfg.out_dir,
                    help="directory for cache, metrics JSON, and checkpoints")
    ap.add_argument("--seeds", type=int, nargs="+", default=cfg.seeds,
                    help="random seeds (default: 42 43 44)")
    ap.add_argument("--window", type=int, default=cfg.window,
                    help="sliding-window size W (default: 10)")
    ap.add_argument("--ablation", action="store_true",
                    help="also run the 'Without ACFD' configuration (Table 5)")
    return ap.parse_args()


def summarise(results, keys):
    summary = {}
    for k in keys:
        vals = np.array([r[k] for r in results])
        std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        summary[k] = {"mean": float(vals.mean()), "std": std}
        print(f"  {k:12s}: {vals.mean():.4f} +/- {std:.4f}   "
              f"(seeds: {np.round(vals, 4)})")
    return summary


def main():
    args = parse_args()
    cfg = Config(data_dir=args.data_dir, out_dir=args.out_dir,
                 seeds=args.seeds, window=args.window)
    os.makedirs(cfg.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device: {device} | {dev_name}")

    # ---- load real data --------------------------------------------------
    data = load_or_build_dataset(cfg)
    X_all, y_all, sub_codes, features, sub_names = data
    print(f"\nReal dataset: X={X_all.shape} | benign={(y_all == 0).sum():,} | "
          f"attack={(y_all == 1).sum():,}")
    print(f"{len(features)} features | sub-types: {sub_names}")

    n_params = count_params(len(features), cfg)
    print(f"Longformer detector parameters: {n_params:,} ({n_params / 1e6:.2f}M)")

    keys = ["accuracy_%", "precision", "recall", "f1", "auc_roc"]

    # ---- main run: multi-seed, WITH ACFD (Table 4) ----------------------
    results, art = [], None
    for i, seed in enumerate(cfg.seeds):
        m, a = run_experiment(seed, data, cfg, device,
                              use_acfd=True, keep_artifacts=(i == 0))
        results.append(m)
        if a is not None:
            art = a

    print("\n" + "=" * 64)
    print(f"RESULTS over {len(cfg.seeds)} seeds - ACFD-Transformer (Longformer), "
          f"{n_params / 1e6:.2f}M params")
    print("=" * 64)
    summary = summarise(results, keys)

    with open(os.path.join(cfg.out_dir, "table4_longformer_real.json"), "w") as f:
        json.dump({"per_seed": results, "summary": summary,
                   "params_M": n_params / 1e6, "n_features": len(features),
                   "features": features,
                   "real_benign": int((y_all == 0).sum()),
                   "real_attack": int((y_all == 1).sum())}, f, indent=2)
    if art is not None:
        torch.save(art["model"].state_dict(),
                   os.path.join(cfg.out_dir, "acfd_longformer_seed%d.pth" % cfg.seeds[0]))
    print(f"\nSaved metrics and checkpoint to {cfg.out_dir}")

    # ---- optional ablation: WITHOUT ACFD (Table 5) ----------------------
    if args.ablation:
        abl, _ = run_experiment(cfg.seeds[0], data, cfg, device, use_acfd=False)
        with_acfd = results[0]
        print("\n" + "=" * 64)
        print("TABLE 5 - Impact of the ACFD module (same seed)")
        print("=" * 64)
        print(f"{'Configuration':32s} {'Acc.(%)':>8s} {'Prec.':>7s} "
              f"{'Rec.':>7s} {'F1':>7s}")
        for name, m in (("Without ACFD (Imbalanced)", abl),
                        ("With ACFD (Balanced)", with_acfd)):
            print(f"{name:32s} {m['accuracy_%']:8.2f} {m['precision']:7.4f} "
                  f"{m['recall']:7.4f} {m['f1']:7.4f}")
        with open(os.path.join(cfg.out_dir, "table5_ablation_real.json"), "w") as f:
            json.dump({"without_acfd": abl, "with_acfd": with_acfd}, f, indent=2)
        print(f"\nSaved ablation to {cfg.out_dir}")


if __name__ == "__main__":
    main()
