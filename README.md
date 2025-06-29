# Federated-Learning-for-Occupancy-Prediction-with-Highly-Imbalanced-Non-IID-Low-Volume-Data

Below is a **clean, plain-Markdown** README ready to drop into `README.md`.
I removed the collapsible `<details>` tags and any other HTML so everything renders the same on GitHub, GitLab, Bitbucket, etc.

---

```markdown
# Federated Learning for Occupancy Prediction  
**Highly-Imbalanced · Non-IID · Low-Volume IoT Logs**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](#requirements)

> **Paper:** S. Solat & N. Georgantas,  
> *Federated Learning for Occupancy Prediction with Highly Imbalanced, Non-IID, Low-Volume Data* (CoopIS 2025)

---

## 1. Overview
This repository contains the complete data-engineering and **federated learning (FL)** workflow described in the paper above.  
It predicts the number of connected devices (a proxy for human presence) for eight building zones while keeping raw data local to each zone.

* **Data challenges:** extreme sparsity and class imbalance, short history, strong non-IID distribution across zones.  
* **Key ideas:** focal-MSE / Huber / Pinball losses, dynamic FedProx aggregation, synthetic data expansion, seasonal drift correction.  
* **Outcome:** correlations ≥ 0.85 on 6 / 8 zones and < 7 % absolute load error in the busiest areas—without centralising any raw logs.

---

## 2. Repository layout

```

github\_repository/
├── federated\_learning/
│   ├── server\_coordinator.py
│   ├── client\_train\_val\_test.py
│   ├── client\_inference.py
│   └── all\_zones.yaml
├── helpers\_for-data-engineering/
│   ├── restore\_missing\_rows.py
│   ├── synthetic\_generator.py
│   ├── patterns.py
│   ├── add\_column.py
│   └── utils.py            # extra helpers if added later
└── zones\_datasets/
├── Accueil\_dataset.xlsx
├── BTIC\_Office\_dataset.xlsx
├── … (8 total)
└── Zone\_heatmap.csv    # raw Juniper dump

````

---

## 3. Quick start

### 3.1 Clone and install

```bash
git clone https://github.com/<ORG>/Federated-Learning-for-Occupancy-Prediction-with-Highly-Imbalanced-Non-IID-Low-Volume-Data.git
cd Federated-Learning-*
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
````

**requirements.txt (reference)**

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

### 3.2 Prepare data (run once, in this order)

```bash
python helpers_for-data-engineering/add_column.py          # normalise headers
python helpers_for-data-engineering/restore_missing_rows.py # fill 15-min gaps
python helpers_for-data-engineering/synthetic_generator.py  # extend to 5 yrs
python helpers_for-data-engineering/patterns.py             # seasonality tables
```

### 3.3 Train federated models

```bash
# ---------- on each client machine ----------
python federated_learning/client_train_val_test.py --zone Accueil --round 0
# …repeat for other zones…

# ---------- on the central server ----------
python federated_learning/server_coordinator.py
```

* Round 0 = local warm-start.
* Number of global rounds is set in `all_zones.yaml` (`common.federated.rounds`, default = 3).
* Global checkpoints are saved to `global_weights/`.

### 3.4 Run inference / forecast 2026

```bash
python federated_learning/client_inference.py --zone Accueil
```

Creates `Accueil_forecast_2026.xlsx` plus evaluation metrics.

---

## 4. Configuration quick reference (`all_zones.yaml`)

| Key                          | Default     | Purpose                                     |
| ---------------------------- | ----------- | ------------------------------------------- |
| `common.federated.rounds`    | `3`         | Number of global aggregation rounds         |
| `common.federated.mu_base`   | `0.01`      | Base FedProx μ (scaled by round dispersion) |
| `zones.<ZONE>.loss`          | `focal_mse` | Loss: `focal_mse`, `huber`, or `pinball`    |
| `zones.<ZONE>.window_length` | `96`        | LSTM look-back (96 × 15 min = 24 h)         |
| `zones.<ZONE>.batch_size`    | `32`        | Local mini-batch size                       |
| `zones.<ZONE>.epochs`        | `30`        | Local epochs before early stopping          |

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

See Section 9 of the paper for full discussion.

---

## 6. Requirements

* **Python ≥ 3.9**
* One NVIDIA GPU (A100 on Colab is fine; CPU also works but is slow)
* Disk: \~200 MB (raw data + checkpoints)

---

## 7. License

This project is released under the MIT License — see `LICENSE`.

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

**Siamak Solat** — [siamak.solat@inria.fr](mailto:siamak.solat@inria.fr) • [siamak.solat@gmail.com](mailto:siamak.solat@gmail.com)
Issues and pull requests are welcome!

```

---

### How to use it
1. Copy everything between the triple-backtick blocks (inclusive) into a new `README.md`.
2. Add a `LICENSE` file with the MIT text.
3. Create `requirements.txt` from the list above or via `pip freeze`.
4. Commit and push — done!
```
