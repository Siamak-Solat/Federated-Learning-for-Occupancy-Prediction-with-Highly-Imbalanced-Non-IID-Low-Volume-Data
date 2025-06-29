# Federated Learning for Occupancy Prediction with Highly Imbalanced, Non-IID, Low-Volume Data: An Empirical Study  

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](#requirements)

> **Paper:** Siamak Solat and Nikolaos Georgantas,  
> *Federated Learning for Occupancy Prediction with Highly Imbalanced, Non-IID, Low-Volume Data: An Empirical Study*  
> (Submitted to CoopIS 2025 Conference)


---

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Synthetic Data Generation](#synthetic-data-generation)
- [Federated Learning Pipeline](#federated-learning-pipeline)
- [Experimental Setup & Results](#experimental-setup--results)
- [Repository Layout](#repository-layout)

---

## Overview
In this work, we tackle the problem of accurate short-term occupancy prediction in smart buildings using connected-device logs—data that, despite being a convenient proxy for human presence, suffer from four core challenges: (1) low overall volume; (2) extreme sparsity and severe class imbalance; (3) pronounced non-IID heterogeneity across physical zones; and (4) privacy constraints that preclude raw-data sharing.

## Dataset
The initial CSV file, named `Zone_heatmap.csv`, was provided by Juniper Networks in collaboration with Atos's BTIC team during the CP4SC French research project. It is a time-series matrix with 24,831 rows and 9 columns, representing 511 days of 15-minute-interval logs across eight distinct zones.

![BTIC zones map](assets/BTICzones.png)

## Synthetic Data Generation
Since reliable forecasting of device-connectivity counts requires training data that capture hourly, weekly and monthly seasonal cycles, we extend these logs into a five-year span via a probability-matched synthetic-data generator that back- and forward-fills gaps while preserving each zone’s base rate and seasonal fingerprints within tight tolerances.

## Federated Learning Pipeline
To address non-IID heterogeneity, we begin by applying statistical tests that confirm distributional skew across clients. Our end-to-end federated-learning pipeline then integrates:

- **A dynamic FedProx-style aggregation server**, which adaptively tunes its proximal coefficient based on client dispersion and deploys zone-specific optimizers and learning schedules to ensure stable convergence;

- **A family of imbalance-aware loss functions**—including focal-MSE, Huber, and pinball losses—calibrated to each zone’s zero/non-zero ratio, with guarded calibration maps to stabilize updates and offset bias.

## Experimental Setup & Results
We implemented the entire workflow on Google Colab using an NVIDIA A100 GPU. In experiments spanning eight heterogeneous zones, our method achieves Pearson correlations of at least 0.85 for both hourly and monthly occupancy in six zones, reduces July–December 2025 aggregate-load forecasting error to below 7 % in the busiest areas, and maintains negligible forecast bias—all while keeping raw observations entirely local. These results confirm the practical viability of federated learning for IoT time-series forecasting under extreme data scarcity, distribution skew, and privacy restrictions.

---

## Repository layout

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
```

---





