"""Central configuration for Jakarta flood risk MVP."""
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- 15 pilot kelurahan (South & East Jakarta) ---
PILOT_KELURAHAN = {
    "Pengadegan":       {"kec": "Pancoran",        "lat": -6.2447, "lon": 106.8456},
    "Cawang":           {"kec": "Kramat Jati",      "lat": -6.2575, "lon": 106.8700},
    "Bidara Cina":      {"kec": "Jatinegara",       "lat": -6.2253, "lon": 106.8714},
    "Kampung Melayu":   {"kec": "Jatinegara",       "lat": -6.2175, "lon": 106.8669},
    "Bukit Duri":       {"kec": "Tebet",            "lat": -6.2264, "lon": 106.8519},
    "Kebon Baru":       {"kec": "Tebet",            "lat": -6.2280, "lon": 106.8588},
    "Pejaten Timur":    {"kec": "Pasar Minggu",     "lat": -6.2889, "lon": 106.8392},
    "Ragunan":          {"kec": "Pasar Minggu",     "lat": -6.3147, "lon": 106.8197},
    "Duren Tiga":       {"kec": "Pancoran",         "lat": -6.2600, "lon": 106.8417},
    "Rawajati":         {"kec": "Pancoran",         "lat": -6.2547, "lon": 106.8503},
    "Balekambang":      {"kec": "Kramat Jati",      "lat": -6.2697, "lon": 106.8628},
    "Cililitan":        {"kec": "Kramat Jati",      "lat": -6.2619, "lon": 106.8758},
    "Cipinang Melayu":  {"kec": "Makasar",          "lat": -6.2503, "lon": 106.8869},
    "Halim Perdanakusuma": {"kec": "Makasar",       "lat": -6.2664, "lon": 106.8914},
    "Batu Ampar":       {"kec": "Kramat Jati",      "lat": -6.2742, "lon": 106.8717},
}

# --- BMKG rainfall stations covering pilot area ---
BMKG_STATIONS = {
    "Halim":      {"id": "96745", "lat": -6.2664, "lon": 106.8914},
    "Pasar_Minggu": {"id": "96747", "lat": -6.2900, "lon": 106.8400},
    "Tanjung_Priok": {"id": "96749", "lat": -6.1078, "lon": 106.8681},
}

# --- Jakarta Open Data pintu air (floodgates) ---
PINTU_AIR_STATIONS = [
    "Manggarai", "Karet", "Kampung Melayu", "Rawajati", "Cawang",
]

# --- Forecast horizons (hours) ---
HORIZONS = [6, 12, 24]

# --- Risk classification thresholds ---
RISK_LEVELS = {
    "Aman":    (0.00, 0.20),
    "Waspada": (0.20, 0.50),
    "Siaga":   (0.50, 0.80),
    "Bahaya":  (0.80, 1.00),
}

# --- Time split ---
TRAIN_END = "2023-12-31"
VAL_START = "2024-01-01"
VAL_END   = "2024-12-31"

# --- Feature engineering ---
RAINFALL_WINDOWS_H = [1, 3, 6, 12, 24, 48, 72]  # rolling sum windows
WATER_LEVEL_LAGS_H = [1, 2, 3, 6, 12, 24]        # lag steps
FLOOD_THRESHOLD_CM = 750                            # Manggarai alert level (cm)

# --- XGBoost defaults (tuned via Optuna) ---
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": ["logloss", "aucpr"],
    "tree_method": "hist",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
}
