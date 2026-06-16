"""
Data loading and preprocessing for the CICAPT-IIoT network-traffic subset.

This mirrors Section 4.1-4.2 of the paper:
  * the two network-traffic CSV files (phase 1 + phase 2) are read in chunks;
  * every real attack flow is kept; benign flows are sub-sampled;
  * 63 informative numeric features are selected (constant columns removed);
  * the attack sub-type (subLabelCat) is encoded as the conditioning signal
    for the ACFD diffusion module.

IMPORTANT: synthetic samples are NEVER created here. This module produces only
real data. Synthetic augmentation happens later, on the training split only
(see pipeline.run_experiment).

The processed arrays are cached to a compressed .npz file so that subsequent
runs load in seconds instead of re-reading the multi-GB CSVs.
"""
import os
import gc
import time
import numpy as np
import pandas as pd

# Columns that are identifiers / labels rather than behavioural features.
META_COLS = {
    "ts", "Source IP", "Destination IP", "Source Port", "Destination Port",
    "Protocol_name", "label", "subLabel", "subLabelCat", "Flow ID", "id",
}


def _attack_mask(sub: pd.Series) -> pd.Series:
    """A row is an attack iff its subLabelCat is a non-zero, non-empty label."""
    s = sub.astype(str).str.strip()
    return (s != "0") & (s != "0.0") & (s != "") & (s.str.lower() != "nan")


def load_or_build_dataset(cfg):
    """
    Return (X_all, y_all, sub_codes, features, sub_names) of REAL data.

    X_all      : float32 array  [N, n_features]
    y_all      : int64 array    [N]    (0 = benign, 1 = APT)
    sub_codes  : int64 array    [N]    (0 = benign, 1..K = attack sub-type)
    features   : list[str]             selected feature column names
    sub_names  : list[str]             ['benign', <attack sub-types...>]
    """
    os.makedirs(cfg.out_dir, exist_ok=True)

    if os.path.exists(cfg.cache_path):
        print(f"Found cache -> loading {cfg.cache_path}")
        z = np.load(cfg.cache_path, allow_pickle=True)
        return (z["X"], z["y"], z["sub"],
                list(z["features"]), list(z["sub_names"]))

    # ---- locate the raw CSV files ---------------------------------------
    for p in (cfg.phase2_path, cfg.phase1_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Could not find '{p}'.\n"
                "The CICAPT-IIoT network-traffic CSV files are not bundled with "
                "this repository because of their size. Please download them and "
                "place them in the data directory as described in README.md "
                "(section 'Data'). Expected files:\n"
                f"  - {cfg.phase1_path}\n  - {cfg.phase2_path}"
            )

    print("No cache yet -> reading the two CSV files in chunks "
          "(first run takes ~10-20 minutes)...")
    head = pd.read_csv(cfg.phase2_path, nrows=5)
    print(f"phase2 has {len(head.columns)} columns")
    feat_candidates = [c for c in head.columns if c not in META_COLS]

    rng = np.random.default_rng(123)   # fixed seed for the data-collection step
    attack_parts, benign_parts = [], []
    benign_cap = cfg.target_benign * 3  # over-collect, then sub-sample exactly

    def harvest(path, phase):
        """Keep every attack row; sample a small fraction of benign per chunk."""
        n_rows = 0
        usecols = feat_candidates + [c for c in ("subLabelCat",) if c in head.columns]
        dtypes = {"subLabelCat": "str"} if "subLabelCat" in head.columns else None
        for chunk in pd.read_csv(path, chunksize=cfg.chunksize,
                                 usecols=lambda c: c in set(usecols),
                                 dtype=dtypes):
            n_rows += len(chunk)
            if phase == 2 and "subLabelCat" in chunk.columns:
                m = _attack_mask(chunk["subLabelCat"])
                if m.any():
                    attack_parts.append(chunk[m].copy())
                ben = chunk[~m]
            else:
                ben = chunk
            if sum(len(b) for b in benign_parts) < benign_cap and len(ben):
                take = max(1, int(len(ben) * 0.005))
                idx = rng.choice(len(ben), size=min(take, len(ben)), replace=False)
                benign_parts.append(ben.iloc[idx].copy())
            del chunk
            gc.collect()
        print(f"   phase{phase}: scanned {n_rows:,} rows")

    t0 = time.time()
    harvest(cfg.phase2_path, 2)
    harvest(cfg.phase1_path, 1)
    print(f"Reading finished in {(time.time() - t0) / 60:.1f} min")

    attack_df = pd.concat(attack_parts, ignore_index=True)
    benign_df = pd.concat(benign_parts, ignore_index=True)
    if len(benign_df) > cfg.target_benign:
        benign_df = benign_df.sample(n=cfg.target_benign, random_state=123)
    print(f"Real attack flows: {len(attack_df):,} | benign sampled: {len(benign_df):,}")

    # ---- attack sub-type as the ACFD conditioning signal ----------------
    sub_attack, sub_uniques = pd.factorize(attack_df["subLabelCat"].astype(str))
    sub_codes = np.concatenate([
        np.zeros(len(benign_df), dtype=np.int64),
        sub_attack + 1,                      # shift so 0 stays reserved for benign
    ])
    sub_names = ["benign"] + list(sub_uniques)

    df = pd.concat([benign_df, attack_df], ignore_index=True)
    y_all = np.concatenate([
        np.zeros(len(benign_df), dtype=np.int64),
        np.ones(len(attack_df), dtype=np.int64),
    ])

    # ---- select features: numeric, drop constant columns, top variance --
    num = df[[c for c in feat_candidates if c in df.columns]].apply(
        pd.to_numeric, errors="coerce")
    num = num.replace([np.inf, -np.inf], np.nan).fillna(0)
    num = num.loc[:, num.nunique() > 1]
    print(f"Usable numeric features: {num.shape[1]}")
    features = num.var().sort_values(ascending=False).index[:cfg.n_features].tolist()
    X_all = num[features].values.astype(np.float32)

    np.savez_compressed(
        cfg.cache_path, X=X_all, y=y_all, sub=sub_codes,
        features=np.array(features),
        sub_names=np.array(sub_names, dtype=object),
    )
    print(f"Cached processed dataset -> {cfg.cache_path}")
    del df, attack_df, benign_df, num
    gc.collect()
    return X_all, y_all, sub_codes, features, sub_names
