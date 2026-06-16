# ACFD-Transformer: APT Detection with Conditional Diffusion Augmentation and a Longformer Backbone

Reference implementation for the paper:

> **Optimizing APT Attack Detection with Long-Sequence Transformer Variants and Attention-based Contextual Diffusion**
> Thanh Duc Vu, Xuan Cho Do, Long Giang Nguyen.

The framework couples a conditional denoising-diffusion module (**ACFD**) that
synthesises minority-class (APT) windows for training-set balancing with a
**Longformer** classification backbone, evaluated on the network-traffic subset
of the CICAPT-IIoT dataset.

This repository reproduces the main results of the paper end-to-end: dataset
construction, ACFD pre-training and augmentation, Longformer training, and
evaluation on a strictly real test set over multiple random seeds.

---

## Methodological notes

A few points are enforced in the code and are worth highlighting, because they
are central to the integrity of the reported numbers:

- **Synthetic samples are used in training only.** ACFD generates synthetic
  minority windows *after* the train/validation/test split, and they are merged
  into the **training split only**. The validation and test splits contain 100%
  real data.
- **No leakage.** The Min-Max scaler is fit on the training split only and then
  applied to validation and test.
- **Windows are labelled by their terminal flow** (the last flow in the window).
- **Multi-seed.** Results are averaged over three seeds (42, 43, 44) and
  reported as mean ± standard deviation, computed directly from the run outputs.

---

## Repository structure

```
ACFD-Transformer/
├── main.py              # entry point (CLI)
├── requirements.txt
├── src/
│   ├── config.py        # all hyperparameters (Table 3) and paths
│   ├── data.py          # CSV loading, feature selection, caching (real data only)
│   ├── models.py        # ACFD diffusion module + Longformer detector
│   └── pipeline.py      # split, windowing, training, evaluation
├── data/                # place the dataset CSVs here (not tracked)
└── outputs/             # metrics JSON + checkpoints (generated)
```

---

## Installation

```bash
git clone https://github.com/ducvt-cloud/ACFD-Transformer.git
cd ACFD-Transformer
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended (the full diffusion sampling is the slowest
step). The code also runs on CPU for small smoke tests.

---

## Data

The experiments use the **network-traffic** portion of the **CICAPT-IIoT**
dataset (Ghiasvand et al., 2024). The raw CSV files are several GB each and are
therefore **not** included in this repository.

1. Obtain the CICAPT-IIoT network-traffic CSV files (publicly available; e.g.
   from the dataset's Kaggle mirror:
   <https://www.kaggle.com/datasets/waqarkha/cicapt-iiot>).
2. Place the two phase files in the `data/` directory so that the following
   paths exist:
   ```
   data/phase1_NetworkData.csv
   data/phase2_NetworkData.csv
   ```
   (If your file names differ, either rename them or pass the correct names via
   `src/config.py`.)

On the first run the pipeline reads the CSVs in chunks, selects the 63
informative numeric features, sub-samples benign flows, and writes a compressed
cache (`outputs/cicapt_network_cache.npz`). Subsequent runs load the cache in
seconds.

The cross-dataset generalisation study in the paper additionally uses
**UNSW-NB15** (Moustafa & Slay, 2015), available from UNSW:
<https://research.unsw.edu.au/projects/unsw-nb15-dataset>.

---

## Usage

```bash
# Main result (Table 4, Longformer row): 3 seeds, with ACFD augmentation
python main.py --data-dir data --out-dir outputs

# Also run the ablation (Table 5): "Without ACFD" on the first seed
python main.py --data-dir data --out-dir outputs --ablation

# Quick single-seed smoke test
python main.py --seeds 42
```

### Outputs

| File | Content |
|------|---------|
| `outputs/table4_longformer_real.json` | per-seed and mean ± std metrics, parameter count, feature list |
| `outputs/table5_ablation_real.json`   | With-ACFD vs. Without-ACFD comparison (if `--ablation`) |
| `outputs/acfd_longformer_seed42.pth`  | trained model checkpoint (seed 42) |

Every number reported in the paper's Table 4 and Table 5 corresponds to a field
in these JSON files.

---

## Configuration

All hyperparameters live in `src/config.py` and correspond to Table 3 of the
paper: ACFD diffusion steps `T=1000`, linear β schedule `1e-4 → 0.02`, 3-layer
MLP denoiser with 128 hidden units; Longformer with 2 layers, 8 heads,
embedding dimension 128; AdamW with learning rate `1e-4` and cosine-annealing
schedule, dropout 0.1, early stopping (patience 7) on the validation F1; sliding
window `W=10`.

---

## Citation

If you use this code, please cite the paper (full bibliographic details will be
added upon publication).

## License

See `LICENSE`.
