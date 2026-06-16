"""
Configuration for the ACFD-Transformer pipeline.

All hyperparameters correspond to Table 3 and Sections 3.5 / 4.2 of the paper.
Paths default to local ./data and ./outputs directories but can be overridden
from the command line (see main.py) or by editing the defaults below.
"""
from dataclasses import dataclass, field
from typing import List
import os


@dataclass
class Config:
    # ---- Paths -----------------------------------------------------------
    # The two CICAPT-IIoT *network traffic* CSV files. See README (Data) for
    # how to obtain them. They are NOT included in the repository because of
    # their size (several GB each).
    data_dir: str = "data"
    phase1_csv: str = "phase1_NetworkData.csv"
    phase2_csv: str = "phase2_NetworkData.csv"
    out_dir: str = "outputs"
    cache_file: str = "cicapt_network_cache.npz"

    # ---- Dataset construction (Section 4.1 / 4.2, Table 1) ---------------
    n_features: int = 63          # informative numeric features kept
    target_benign: int = 50_000   # benign windows sampled from both phases
    chunksize: int = 1_000_000    # CSV is read in chunks to bound memory

    # ---- Sliding window (Section 4.2) ------------------------------------
    window: int = 10              # W = 10; label = terminal flow of the window

    # ---- ACFD diffusion module (Table 3) ---------------------------------
    t_steps: int = 1000           # diffusion steps T
    beta_start: float = 1e-4
    beta_end: float = 0.02
    acfd_hidden: int = 128        # 3-layer MLP denoiser, 128 hidden units
    acfd_epochs: int = 100
    acfd_lr: float = 1e-3

    # ---- Longformer backbone (Table 3) -----------------------------------
    embed_dim: int = 128
    n_layers: int = 2
    n_heads: int = 8
    ffn_dim: int = 512
    dropout: float = 0.1

    # ---- Training (Section 3.5) ------------------------------------------
    batch_size: int = 64
    lr: float = 1e-4
    max_epochs: int = 50
    patience: int = 7             # early stopping on validation F1

    # ---- Reproducibility -------------------------------------------------
    seeds: List[int] = field(default_factory=lambda: [42, 43, 44])

    # ---- Derived ---------------------------------------------------------
    @property
    def phase1_path(self) -> str:
        return os.path.join(self.data_dir, self.phase1_csv)

    @property
    def phase2_path(self) -> str:
        return os.path.join(self.data_dir, self.phase2_csv)

    @property
    def cache_path(self) -> str:
        return os.path.join(self.out_dir, self.cache_file)
