# Jakarta Flood Risk Prediction — Phase 1 MVP

> **Status Pengembangan: Work In Progress (Fase Proposal)**  
> Repository ini saat ini berada di Fase Proposal Dicoding AI Impact Challenge 2026.  
> Implementasi kode (model, API Azure Functions, dan dashboard Azure Static Web Apps) masih dalam pengembangan aktif dan belum lengkap.

XGBoost-based flood risk prediction for 15 pilot kelurahan in South and East Jakarta. Outputs calibrated flood probabilities at 6, 12, and 24-hour horizons with SHAP explainability for BPBD operators.

---

## Overview

Jakarta experiences severe annual flooding driven by monsoon rainfall, tidal backflow, and limited drainage capacity in low-lying urban areas. This MVP targets the Ciliwung corridor (South/East Jakarta) where flood risk is highest and telemetry data coverage is most reliable.

**Target:** F1-score ≥ 0.65 on 2024 hold-out validation  
**Training period:** 2018–2023 (time-based split, no leakage)  
**Scope:** 15 pilot kelurahan across Tebet, Pancoran, Kramat Jati, Jatinegara, Pasar Minggu, and Makasar

### Risk levels

| Level | Probability | Recommended action |
|---|---|---|
| **Aman** | < 20% | Normal monitoring |
| **Waspada** | 20–50% | Heightened watch, pre-position teams |
| **Siaga** | 50–80% | Deploy field teams, open evacuation centres |
| **Bahaya** | ≥ 80% | Immediate evacuation |

---

## Architecture

```
flood_risk_mvp/
├── flood_risk/
│   ├── config.py                  # kelurahan coords, thresholds, train/val split
│   ├── data/
│   │   ├── bmkg.py                # BMKG hourly rainfall API client
│   │   ├── water_level.py         # Jakarta Open Data pintu air loader + label builder
│   │   ├── dem.py                 # DEMNAS/SRTM static terrain feature extractor
│   │   └── pipeline.py            # master pipeline → model-ready DataFrames
│   ├── models/
│   │   ├── xgb_flood.py           # FloodRiskModel + MultiHorizonFloodModel
│   │   └── tuner.py               # Optuna HPO (50 trials per horizon)
│   └── evaluation/
│       ├── metrics.py             # F1, ROC-AUC, PR-AUC, threshold calibration
│       └── explainability.py      # SHAP TreeExplainer: global + local + alert narrative
├── train.py                       # training entrypoint
├── predict.py                     # inference CLI
└── requirements.txt
```
## Azure Architecture (Draft)

This project is designed to run end-to-end on Microsoft Azure (currently in draft, WIP for proposal phase):

- **Azure Functions (HTTP trigger)**  
  - Serves a REST API endpoint `/api/predict` that loads the trained XGBoost models (`flood_xgb_{6,12,24}h.joblib`) and returns calibrated flood probabilities for a given kelurahan and timestamp.  
  - Code scaffold lives in `azure-function/` (binding configuration and handler are under active development).

- **Azure Static Web Apps**  
  - Hosts a static dashboard (React/JS) located in `static-web-app/`.  
  - The dashboard calls the Azure Functions API to display 6/12/24-hour flood risk levels (Aman, Waspada, Siaga, Bahaya) for the 15 pilot kelurahan.  
  - Deployment target: Azure Static Web Apps Free tier with GitHub Actions for CI/CD.

- **(Planned) Azure Storage**  
  - Optional blob storage for archiving daily prediction outputs and logs for later analysis and model monitoring.

> During the Dicoding AI Impact Challenge 2026 proposal phase, the Azure deployment is being implemented and iterated. Public URLs for the Static Web App and Function API will be added here once stable.
---

### Data sources

| Source | Data | Frequency | Coverage |
|---|---|---|---|
| [BMKG](https://data.bmkg.go.id) | Rainfall (mm) | Hourly | 3 stations near pilot area |
| [Jakarta Open Data](https://data.jakarta.go.id) | Pintu air water level (cm) | Hourly | 5 floodgates: Manggarai, Karet, Kampung Melayu, Rawajati, Cawang |
| DEMNAS / SRTM | Elevation, slope, TWI, flow accumulation | Static | Per kelurahan centroid |

### Feature groups (~50 features per row)

| Group | Features |
|---|---|
| **Rainfall** | Rolling sums at 1/3/6/12/24/48/72h, station max, heavy/extreme flags, antecedent moisture proxy |
| **Water level** | Raw levels, hourly delta, lag 1–24h, rolling max/trend, hours above warning threshold |
| **DEM (static)** | Elevation (m), slope (°), topographic wetness index, distance to river, log flow accumulation |
| **Calendar** | Hour/month sin–cos encoding, wet season flag, peak flood month (Jan–Feb), storm hour (17–21h) |

### Labelling

`flood_Nh = 1` if any pintu air station reaches ≥ 750 cm (Siaga 2 / Manggarai alert level) within the next N hours. Max-pooled across stations so a single exceedance anywhere in the network triggers the label.

### Model

One `XGBClassifier` per horizon. Class imbalance (flood events ≈ 2–8% of hours) is handled via `scale_pos_weight` + `compute_sample_weight("balanced")`. Post-training threshold sweep maximises F1 under a minimum 70% recall constraint — catching real floods takes priority over false-alarm reduction.

---

## Pilot Kelurahan

| Kelurahan | Kecamatan | Elevation |
|---|---|---|
| Bidara Cina | Jatinegara | 4.1 m |
| Kampung Melayu | Jatinegara | 4.5 m |
| Pengadegan | Pancoran | 5.2 m |
| Rawajati | Pancoran | 5.5 m |
| Bukit Duri | Tebet | 5.8 m |
| Kebon Baru | Tebet | 6.2 m |
| Duren Tiga | Pancoran | 6.5 m |
| Cawang | Kramat Jati | 6.8 m |
| Balekambang | Kramat Jati | 7.2 m |
| Cililitan | Kramat Jati | 7.8 m |
| Cipinang Melayu | Makasar | 8.4 m |
| Pejaten Timur | Pasar Minggu | 9.1 m |
| Halim Perdanakusuma | Makasar | 9.5 m |
| Ragunan | Pasar Minggu | 12.3 m |
| Batu Ampar | Kramat Jati | 7.5 m |

---

## Setup

```bash
git clone https://github.com/suryalionael/jakarta-flood-risk.git
cd jakarta-flood-risk
pip install -r requirements.txt
```

Python 3.11+ recommended.

---

## Usage

### Train

```bash
# Full training run using synthetic data (no credentials needed)
python train.py

# With Optuna hyperparameter optimisation (50 trials per horizon)
python train.py --tune --trials 50

# Generate SHAP summary plots and importance CSVs
python train.py --shap

# Shorter history (faster for dev)
python train.py --start 2020-01-01
```

Outputs:
- `models/flood_xgb_{6,12,24}h.joblib` — trained models
- `reports/validation_metrics.csv` — per-horizon F1, ROC-AUC, PR-AUC
- `reports/shap/` — SHAP plots and importance CSVs (if `--shap`)

### Predict

```bash
# All 15 kelurahan, latest available data
python predict.py --all

# Single kelurahan
python predict.py --kelurahan "Kampung Melayu"

# Specific timestamp
python predict.py --kelurahan "Bidara Cina" --timestamp "2024-02-15 18:00"
```

Sample output:
```
==================================================
Kelurahan: Kampung Melayu | Time: 2024-02-15 18:00
   6h: [████████████░░░░░░░░] 62.3%  →  Siaga
  12h: [██████████░░░░░░░░░░] 51.8%  →  Siaga
  24h: [███████░░░░░░░░░░░░░] 38.1%  →  Waspada
```

---

## Wiring Real Data

Both data loaders fall back to statistically-plausible synthetic stubs when the API is unreachable, so the full pipeline runs without credentials during development.

**BMKG rainfall**: Replace `_api_fetch` in `flood_risk/data/bmkg.py` with your BMKG FTP or API credentials. BMKG provides historical hourly data via their Climate Data Portal or upon institutional request.

**Jakarta Open Data water levels**: Register at [data.jakarta.go.id](https://data.jakarta.go.id) and update `_RESOURCE_IDS` in `flood_risk/data/water_level.py` with the correct Socrata resource IDs for each pintu air station.

**DEM**: Download DEMNAS tiles from [tanahair.indonesia.go.id](https://tanahair.indonesia.go.id) (8 m resolution) or use SRTM 30 m as fallback. Pass the GeoTIFF path to `DEMFeatureExtractor(dem_path=...)`.

---

## SHAP Explainability

Every prediction can be explained at the feature level. The `alert_narrative()` method generates Indonesian-language text for BPBD operators:

```
[2024-02-15 18:00] PERINGATAN BANJIR — Level: Siaga
Probabilitas: 62.3% dalam 6 jam ke depan
Faktor utama:
  • rain sum 6h: 48.2mm (↑ risk)
  • wl max cm: 712.0cm (↑ risk)
  • antecedent rain 72h: 183.4mm (↑ risk)
```

Global feature importance and waterfall plots are saved to `reports/shap/` during training.

---

## Evaluation

Validation uses a strict time-based split: train on 2018–2023, validate on 2024. No shuffling, no cross-contamination.

Key metrics tracked:
- **F1** (primary target: ≥ 0.65)
- **ROC-AUC** and **PR-AUC** (imbalanced-class safe)
- **Recall** (miss rate — safety-critical; floor at 70%)
- **False alarm rate** (operator trust)

The threshold calibration step (`_calibrate_threshold`) sweeps the decision boundary on the validation set and picks the value that maximises F1, which in practice pushes recall higher than a naive 0.5 cutoff.

---

## Project Status

This repository is currently in **Proposal Phase** for the Dicoding AI Impact Challenge 2026:

- Core Python package (`flood_risk/`) for data processing and model training is in early MVP stage.  
- Azure Functions API (`azure-function/`) and Static Web App dashboard (`static-web-app/`) are under active development and not feature-complete.  
- Synthetic data stubs are used by default so the pipeline can be executed without access to BMKG / Jakarta Open Data credentials.  
- Model performance targets and evaluation protocol are defined, but final 2018–2024 retraining and metric reporting for all horizons are still in progress.

Feedback and issue reports are welcome while the implementation is being completed during the competition timeline.

