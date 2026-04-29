# Jakarta Flood Risk Prediction — Phase 1 MVP

> **Status Pengembangan: Work In Progress (Fase Proposal)**  
> Repository ini saat ini berada di Fase Proposal Dicoding AI Impact Challenge 2026.  
> Implementasi kode (model, API Azure Functions, dan dashboard Azure Static Web Apps) masih dalam pengembangan aktif dan belum lengkap.

Sistem prediksi risiko banjir berbasis XGBoost untuk 15 kelurahan pilot di Jakarta Selatan dan Timur. Output berupa probabilitas banjir terkalibrasi pada horizon 6, 12, dan 24 jam dengan SHAP explainability untuk operator BPBD.

---

## Latar Belakang

Jakarta mengalami banjir parah setiap tahun yang dipicu oleh curah hujan musiman, banjir kiriman dari hulu Ciliwung (Bogor), dan kapasitas drainase terbatas di area dataran rendah. MVP ini fokus pada koridor Ciliwung (Jakarta Selatan/Timur) dimana risiko banjir tertinggi dan cakupan data telemetri paling reliable.

**Target:** F1-score ≥ 0.65 pada validation hold-out 2024
**Periode training:** 2018–2023 (time-based split, tanpa data leakage)
**Cakupan:** 15 kelurahan pilot di Tebet, Pancoran, Kramat Jati, Jatinegara, Pasar Minggu, dan Makasar

### Tingkat Risiko

| Level | Probabilitas | Tindakan yang Direkomendasikan |
|---|---|---|
| **Aman** | < 20% | Monitoring normal |
| **Waspada** | 20–50% | Peningkatan kewaspadaan, pre-position tim |
| **Siaga** | 50–80% | Deploy tim lapangan, buka posko evakuasi |
| **Awas** | ≥ 80% | Evakuasi segera |

---

## Arsitektur Sistem

### Struktur Repositori (saat ini)

```
jakarta-flood-risk/
├── flood_risk/
│   ├── config.py                  # koordinat kelurahan, threshold, train/val split
│   ├── data/
│   │   ├── bmkg.py                # client API curah hujan BMKG (hourly)
│   │   ├── water_level.py         # loader pintu air Jakarta Open Data + label builder
│   │   ├── dem.py                 # ekstraktor fitur terrain DEMNAS/SRTM
│   │   └── pipeline.py            # pipeline master → DataFrame siap-model
│   ├── models/
│   │   ├── xgb_flood.py           # FloodRiskModel + MultiHorizonFloodModel
│   │   └── tuner.py               # Optuna HPO (50 trial per horizon)
│   └── evaluation/
│       ├── metrics.py             # F1, ROC-AUC, PR-AUC, threshold calibration
│       └── explainability.py      # SHAP TreeExplainer: global + local + alert narrative
├── train.py                       # entrypoint training
├── predict.py                     # CLI inference
└── requirements.txt
```

### Rencana Arsitektur Azure (Fase Implementasi)

Sistem dirancang untuk berjalan end-to-end di Microsoft Azure. Komponen-komponen berikut adalah bagian dari roadmap implementasi setelah fase proposal:

- **Azure Functions (HTTP trigger)** — REST API endpoint `/api/predict` yang me-load model XGBoost terlatih (`flood_xgb_{6,12,24}h.joblib`) dan mengembalikan probabilitas banjir terkalibrasi untuk kelurahan dan timestamp tertentu.

- **Azure Static Web Apps** — Dashboard statis berbasis HTML/JavaScript yang memanggil Azure Functions API untuk menampilkan tingkat risiko banjir 6/12/24 jam (Aman, Waspada, Siaga, Awas) untuk 15 kelurahan pilot. Target deployment: Static Web Apps Free tier dengan GitHub Actions untuk CI/CD.

- **Azure OpenAI Service** — Generative advisory multi-audience yang menerjemahkan output prediksi menjadi pesan kontekstual untuk warga (bahasa sederhana), BPBD (terminologi operasional), dan perencana kota (analisis struktural).

- **Azure Maps** — Visualisasi peta interaktif dengan choropleth risiko banjir per kelurahan.

- **(Direncanakan) Azure Storage** — Blob storage untuk arsip output prediksi harian dan log untuk monitoring model.

> URL publik untuk Static Web App dan Function API akan ditambahkan ke README ini setelah deployment Phase 1 stabil.

---

## Sumber Data

| Sumber | Data | Frekuensi | Cakupan |
|---|---|---|---|
| [BMKG](https://data.bmkg.go.id) | Curah hujan (mm) | Per jam | 3 stasiun di sekitar area pilot |
| [Jakarta Open Data](https://data.jakarta.go.id) | Tinggi muka air pintu air (cm) | Per jam | 5 pintu air: Manggarai, Karet, Kampung Melayu, Rawajati, Cawang |
| DEMNAS / SRTM | Elevasi, slope, TWI, flow accumulation | Statis | Per centroid kelurahan |

### Kelompok Fitur (~50 fitur per row)

| Kelompok | Fitur |
|---|---|
| **Curah Hujan** | Rolling sum pada window 1/3/6/12/24/48/72 jam, max antar stasiun, flag heavy/extreme, antecedent moisture proxy |
| **Tinggi Muka Air** | Level mentah, delta per jam, lag 1–24h, rolling max/trend, jam di atas warning threshold |
| **DEM (statis)** | Elevasi (m), slope (°), topographic wetness index, jarak ke sungai, log flow accumulation |
| **Kalender** | Sin–cos encoding jam/bulan, flag musim hujan, bulan puncak banjir (Jan–Feb), jam puncak storm (17–21) |

### Labelling

`flood_Nh = 1` jika ada satu pintu air mencapai ≥ 750 cm (Siaga 2 / level alert Manggarai) dalam N jam ke depan. Max-pooled antar stasiun sehingga satu exceedance dimanapun di network akan trigger label.

### Model

Satu `XGBClassifier` per horizon. Class imbalance (event banjir ≈ 2–8% dari total jam) ditangani via `scale_pos_weight` + `compute_sample_weight("balanced")`. Threshold sweep post-training memaksimalkan F1 dengan constraint minimum recall 70% — menangkap banjir sungguhan diprioritaskan di atas pengurangan false alarm.

---

## Kelurahan Pilot

| Kelurahan | Kecamatan | Elevasi |
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
| Batu Ampar | Kramat Jati | 7.5 m |
| Cililitan | Kramat Jati | 7.8 m |
| Cipinang Melayu | Makasar | 8.4 m |
| Pejaten Timur | Pasar Minggu | 9.1 m |
| Halim Perdanakusuma | Makasar | 9.5 m |
| Ragunan | Pasar Minggu | 12.3 m |

---

## Setup

```bash
git clone https://github.com/suryalionael/jakarta-flood-risk.git
cd jakarta-flood-risk
pip install -r requirements.txt
```

Direkomendasikan Python 3.11+.

---

## Penggunaan

### Training Model

```bash
# Training run lengkap dengan synthetic data (tanpa kredensial API)
python train.py

# Dengan Optuna hyperparameter optimization (50 trial per horizon)
python train.py --tune --trials 50

# Generate plot SHAP summary dan CSV importance
python train.py --shap

# History lebih pendek (lebih cepat untuk development)
python train.py --start 2020-01-01
```

Output:
- `models/flood_xgb_{6,12,24}h.joblib` — model terlatih
- `reports/validation_metrics.csv` — F1, ROC-AUC, PR-AUC per horizon
- `reports/shap/` — plot SHAP dan CSV importance (jika `--shap`)

### Inference

```bash
# Semua 15 kelurahan, data terbaru
python predict.py --all

# Satu kelurahan
python predict.py --kelurahan "Kampung Melayu"

# Timestamp spesifik
python predict.py --kelurahan "Bidara Cina" --timestamp "2024-02-15 18:00"
```

Contoh output:
```
==================================================
Kelurahan: Kampung Melayu | Time: 2024-02-15 18:00
   6h: [████████████░░░░░░░░] 62.3%  →  Siaga
  12h: [██████████░░░░░░░░░░] 51.8%  →  Siaga
  24h: [███████░░░░░░░░░░░░░] 38.1%  →  Waspada
```

---

## Integrasi Data Riil

Kedua loader data memiliki fallback ke synthetic stub yang statistically plausible saat API tidak dapat dijangkau, sehingga pipeline lengkap dapat dijalankan tanpa kredensial selama development.

**Curah hujan BMKG**: Ganti `_api_fetch` di `flood_risk/data/bmkg.py` dengan kredensial BMKG FTP atau API. BMKG menyediakan data historis per jam via Climate Data Portal mereka atau atas permintaan institusional.

**Tinggi muka air Jakarta Open Data**: Daftar di [data.jakarta.go.id](https://data.jakarta.go.id) dan update `_RESOURCE_IDS` di `flood_risk/data/water_level.py` dengan resource ID Socrata yang sesuai untuk setiap stasiun pintu air.

**DEM**: Download tile DEMNAS dari [tanahair.indonesia.go.id](https://tanahair.indonesia.go.id) (resolusi 8 m) atau gunakan SRTM 30 m sebagai fallback. Pass path GeoTIFF ke `DEMFeatureExtractor(dem_path=...)`.

---

## SHAP Explainability

Setiap prediksi dapat dijelaskan pada level fitur. Method `alert_narrative()` menghasilkan teks bahasa Indonesia untuk operator BPBD:

```
[2024-02-15 18:00] PERINGATAN BANJIR — Level: Siaga
Probabilitas: 62.3% dalam 6 jam ke depan
Faktor utama:
  • rain sum 6h: 48.2mm (↑ risiko)
  • wl max cm: 712.0cm (↑ risiko)
  • antecedent rain 72h: 183.4mm (↑ risiko)
```

Global feature importance dan waterfall plot disimpan ke `reports/shap/` selama training.

---

## Evaluasi

Validasi menggunakan time-based split yang ketat: training pada 2018–2023, validasi pada 2024. Tanpa shuffling, tanpa kontaminasi silang.

Metrik utama yang dilacak:
- **F1** (target utama: ≥ 0.65)
- **ROC-AUC** dan **PR-AUC** (aman untuk class imbalanced)
- **Recall** (miss rate — kritikal keselamatan; floor 70%)
- **False alarm rate** (kepercayaan operator)

Step kalibrasi threshold (`_calibrate_threshold`) melakukan sweep decision boundary pada validation set dan memilih value yang memaksimalkan F1, yang dalam praktiknya mendorong recall lebih tinggi dari cutoff naif 0.5.

---

## Status Pengembangan

Repositori ini saat ini berada dalam **Fase Proposal** untuk Dicoding AI Impact Challenge 2026:

- Core Python package (`flood_risk/`) untuk data processing dan model training berada pada tahap MVP awal
- Azure Functions API dan dashboard Static Web App tertulis dalam roadmap arsitektur dan akan diimplementasikan pada fase setelah seleksi proposal
- Synthetic data stub digunakan secara default sehingga pipeline dapat dieksekusi tanpa akses kredensial BMKG / Jakarta Open Data
- Target performa model dan protokol evaluasi sudah didefinisikan, namun retraining final 2018–2024 dan pelaporan metrik untuk semua horizon masih dalam proses

Feedback dan issue report dipersilakan selama implementasi diselesaikan dalam timeline kompetisi.

---
