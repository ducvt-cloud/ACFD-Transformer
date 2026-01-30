<div align="center">
  <h1>🛡️ ACFD-Transformer</h1>
  <p><b>Advanced APT Detection via Adaptive Conditional Feature Diffusion and Longformer</b></p>

  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=for-the-badge&logo=pytorch" />
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB.svg?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge" />
</div>

---

## 📖 Overview
This repository provides the official implementation of the **ACFD-Transformer** framework. Our model is specifically designed to detect stealthy Advanced Persistent Threats (APTs) by combining state-of-the-art diffusion-based data synthesis with long-sequence modeling.

### 🌟 Key Highlights
* **High Accuracy:** Achieved **98.59%** on the CIC-APT-2024 dataset.
* **Lightweight:** Only **0.84M parameters**, making it suitable for real-time SOC deployment.
* **Explainable AI:** Uses SHAP values to explain model decisions for security analysts.

---

## 🏗️ Architecture
The system follows a three-stage pipeline as described in our **Journal of Combinatorial Optimization (JOCO)** submission:

1. **ACFD Module:** Synthesizes minority APT class features using a conditional diffusion process.
2. **Sliding Window:** Transforms network flows into temporal sequences of size $W=10$.
3. **Longformer:** Captures multi-stage attack patterns using sliding window attention.



---

## 📂 Project Structure
```bash
ACFD-Transformer/
├── models/             # Architecture definitions (ACFD & Longformer)
│   └── model.py
├── utils/              # Data processing & Sliding window logic
│   └── preprocess.py
├── data/               # Dataset directory (CSVs here)
│   └── .gitkeep
├── main.py             # Main entry for training and evaluation
├── requirements.txt    # Required Python libraries
└── README.md           # Project documentation
```
## 🚀 Quick Start
1. Installation
```bash
git clone https://github.com/ducvt-cloud/ACFD-Transformer.git
cd ACFD-Transformer
pip install -r requirements.txt
```
2. Training & Evaluation
```bash
python main.py
```
## 📊 Performance Comparison

| Model | Accuracy | F1-Score | Parameters |
|-------|----------|----------|------------|
| **ACFD-Longformer (Ours)** | **98.59%** | **0.979** | **0.84M** |
| Original Transformer | 97.24% | 0.958 | 1.83M |

## 📬 Contact
**Thanh Duc Vu**<br>
📧 Email: ducvt@haui.edu.vn<br>
🏫 Hanoi University of Industry (HaUI)
