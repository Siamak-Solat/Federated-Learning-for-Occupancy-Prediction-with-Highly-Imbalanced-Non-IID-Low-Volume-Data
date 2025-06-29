# Federated-Learning-for-Occupancy-Prediction-with-Highly-Imbalanced-Non-IID-Low-Volume-Data

Below is a **drop-in-ready** `README.md`.
The directory tree is now inside an explicit Markdown code-block, so it will render correctly on GitHub, GitLab, Bitbucket, etc.

---

````markdown
# Federated Learning for Occupancy Prediction  
**Highly-Imbalanced · Non-IID · Low-Volume IoT Logs**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](#requirements)

> **Paper:** S. Solat & N. Georgantas,  
> *Federated Learning for Occupancy Prediction with Highly Imbalanced, Non-IID, Low-Volume Data* (CoopIS 2025)

---

## 1. Overview
This repository contains the full data-engineering and **federated-learning (FL)** pipeline described in the paper above.  
It forecasts the number of connected devices (a proxy for human presence) in eight building zones while keeping raw data **on-premise**.

* **Data challenges:** extreme sparsity and imbalance, short history, strong non-IID behaviour between zones.  
* **Key ideas:** focal-MSE / Huber / Pinball losses, dynamic FedProx aggregation, synthetic data expansion, seasonal drift correction.  
* **Outcome:** correlations ≥ 0.85 on 6 / 8 zones and < 7 % absolute load error in the busiest areas—without centralising any raw logs.

---

## 2. Repository layout

```text
github_repository/
├── federated_learning/
│   ├── server_coordinator.py
│   ├── client_train_val_test.py
│   ├── client_inference.py
│   └── all_zones.yaml
├── helpers_for-data-engineering/
│   ├── restore_missing_rows.py
│   ├── synthetic_generator.py
│   ├── patterns.py
│   ├── add_column.py
│   └── utils.py          # (optional) shared helpers
└── zones_datasets/
    ├── Accueil_dataset.xlsx
    ├── BTIC_Office_dataset.xlsx
    ├── Casablanca_dataset.xlsx
    ├── Discovery_dataset.xlsx
    ├── Experience_dataset.xlsx
    ├── Regie_dataset.xlsx
    ├── Team_Office_dataset.xlsx
    ├── Tech_Area_dataset.xlsx
    └── Zone_heatmap.csv   # raw Juniper export
````

---

## 3. Quick start

### 3.1 Clone & install

```bash
git clone https://github.com/<ORG>/Federated-Learning-for-Occupancy-Prediction-with-Highly-Imbalanced-Non-IID-Low-Volume-Data.git
cd Federated-Learning-*
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` reference:

```
# Core
numpy>=1.23
pandas>=1.5
PyYAML>=6.0
tqdm
joblib
absl-py

# ML / DL
tensorflow>=2.11,<3.0
scikit-learn>=1.4

# Excel I/O
openpyxl>=3.1
```

### 3.2 Prepare data (run once)

```bash
python helpers_for-data-engineering/add_column.py
python helpers_for-data-engineering/restore_missing_rows.py
python helpers_for-data-engineering/synthetic_generator.py
python helpers_for-data-engineering/patterns.py
```

### 3.3 Train federated models

```bash
# ── on each client ──
python federated_learning/client_train_val_test.py --zone Accueil --round 0
# …repeat for other zones…

# ── on the central server ──
python federated_learning/server_coordinator.py
```

* Round 0 = local warm-start.
* Global rounds controlled by `common.federated.rounds` in `all_zones.yaml` (default = 3).
* Aggregated checkpoints saved in `global_weights/`.

### 3.4 Run inference / forecast 2026

```bash
python federated_learning/client_inference.py --zone Accueil
```

Creates `Accueil_forecast_2026.xlsx` plus evaluation metrics.

---

## 4. Key YAML parameters (`all_zones.yaml`)

| Parameter                    | Default     | Meaning                                 |
| ---------------------------- | ----------- | --------------------------------------- |
| `common.federated.rounds`    | `3`         | Global aggregation rounds               |
| `common.federated.mu_base`   | `0.01`      | Base FedProx μ (scaled by round spread) |
| `zones.<ZONE>.loss`          | `focal_mse` | Loss: `focal_mse`, `huber`, `pinball`   |
| `zones.<ZONE>.window_length` | `96`        | LSTM look-back (96×15 min = 24 h)       |
| `zones.<ZONE>.batch_size`    | `32`        | Local minibatch size                    |
| `zones.<ZONE>.epochs`        | `30`        | Local epochs before early-stop          |

---

## 5. Results snapshot (test horizon 2025 H2)

| Zone        | Hourly ρ | Hourly nRMSE | Monthly ρ | MAPE % |
| ----------- | :------: | :----------: | :-------: | :----: |
| Accueil     |   0.89   |     0.86     |    0.88   |  26.5  |
| BTIC Office |   0.93   |     0.45     |    0.83   |  71.9  |
| Casablanca  |   0.98   |     0.19     |    0.96   |  24.9  |
| Discovery   |   0.94   |     0.58     |    0.96   |  25.4  |
| Experience  |   0.99   |     0.16     |    0.96   |   6.6  |
| Regie       |   0.08   |     3.84     |   −0.17   |  186.0 |
| Team Office |   0.96   |     0.29     |    0.90   |  41.7  |
| Tech Area   |   0.85   |     0.91     |    0.14   |  79.7  |

See the paper for full discussion.

---

## 6. Requirements

* **Python ≥ 3.9**
* One NVIDIA GPU (A100 on Colab is fine; CPU also works but is slower)
* Disk: \~200 MB (datasets + checkpoints)

---

## 7. License

This project is released under the **MIT License** — see `LICENSE`.

---

## 8. Citation

```bibtex
@inproceedings{Solat2025OccFL,
  title     = {Federated Learning for Occupancy Prediction with Highly Imbalanced, Non-IID, Low-Volume Data},
  author    = {Siamak Solat and Nikolaos Georgantas},
  booktitle = {Proceedings of CoopIS 2025},
  year      = {2025},
  url       = {https://github.com/<ORG>/Federated-Learning-for-Occupancy-Prediction-with-Highly-Imbalanced-Non-IID-Low-Volume-Data}
}
```

---

## 9. Contact

**Siamak Solat** — [siamak.solat@inria.fr](mailto:siamak.solat@inria.fr) · [siamak.solat@gmail.com](mailto:siamak.solat@gmail.com)
Issues and pull requests are welcome!

```

---

### How to use it
1. Copy everything between the triple-backtick blocks into a new `README.md`.
2. Add a `LICENSE` file containing the MIT text.
3. Supply a `requirements.txt` (list above) or an `environment.yml`.
4. Commit and push—done!
```
