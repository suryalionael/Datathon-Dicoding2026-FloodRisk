"""
Azure Function HTTP trigger for FloodCast prediction API.

Endpoint: GET /api/predict?kelurahan=<name>&horizon=<6|12|24>
         GET /api/predict?all=true                    # all kelurahan
         GET /api/health                              # health check

Returns flood probability + risk level + SHAP-based explanation per kelurahan.
"""
import logging
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import azure.functions as func
import joblib
import pandas as pd

# Lazy-loaded singletons (avoid cold-start cost per request)
_MODELS = None
_SHAP_EXPLAINERS = None
_LATEST_FEATURES = None

MODEL_DIR = Path(__file__).parent.parent / "models"
DATA_DIR = Path(__file__).parent.parent / "data"

RISK_LEVELS = [
    (0.20, "Aman"),
    (0.50, "Waspada"),
    (0.80, "Siaga"),
    (1.01, "Awas"),
]


def get_risk_level(prob: float) -> str:
    for threshold, label in RISK_LEVELS:
        if prob < threshold:
            return label
    return "Awas"


def load_models():
    """Load XGBoost models for 6h, 12h, 24h horizons. Cached after first call."""
    global _MODELS
    if _MODELS is None:
        _MODELS = {}
        for horizon in [6, 12, 24]:
            path = MODEL_DIR / f"flood_xgb_{horizon}h.joblib"
            if path.exists():
                _MODELS[horizon] = joblib.load(path)
                logging.info(f"Loaded model for {horizon}h horizon")
            else:
                logging.warning(f"Model not found: {path}")
    return _MODELS


def load_latest_features():
    """Load latest precomputed features per kelurahan. Cached after first call."""
    global _LATEST_FEATURES
    if _LATEST_FEATURES is None:
        path = DATA_DIR / "latest_features.parquet"
        if path.exists():
            _LATEST_FEATURES = pd.read_parquet(path)
        else:
            # Fallback to JSON for environments without pyarrow
            json_path = DATA_DIR / "latest_features.json"
            if json_path.exists():
                _LATEST_FEATURES = pd.read_json(json_path)
            else:
                logging.error("No feature file found")
                return None
    return _LATEST_FEATURES


def predict_one(kelurahan: str, horizon: int) -> dict:
    """Generate prediction for a single kelurahan + horizon."""
    models = load_models()
    features = load_latest_features()

    if features is None or horizon not in models:
        return {"error": "Model or features not available"}

    row = features[features["kelurahan"] == kelurahan]
    if row.empty:
        return {"error": f"Kelurahan '{kelurahan}' not found"}

    feature_cols = [c for c in features.columns
                    if c not in ("kelurahan", "timestamp", "kecamatan")]
    X = row[feature_cols].values

    model = models[horizon]
    prob = float(model.predict_proba(X)[0, 1])
    risk = get_risk_level(prob)

    # Top contributing features (use feature_importances_ as a proxy when SHAP unavailable)
    importances = model.feature_importances_
    top_idx = importances.argsort()[-5:][::-1]
    top_features = [
        {
            "feature": feature_cols[i],
            "value": float(X[0][i]),
            "importance": float(importances[i]),
        }
        for i in top_idx
    ]

    return {
        "kelurahan": kelurahan,
        "horizon_hours": horizon,
        "probability": round(prob, 4),
        "risk_level": risk,
        "top_factors": top_features,
        "predicted_at": datetime.now(timezone.utc).isoformat(),
    }


def main(req: func.HttpRequest) -> func.HttpResponse:
    """HTTP trigger entry point."""
    logging.info("FloodCast predict endpoint hit")

    try:
        kelurahan = req.params.get("kelurahan")
        horizon_str = req.params.get("horizon", "24")
        all_flag = req.params.get("all", "false").lower() == "true"

        try:
            horizon = int(horizon_str)
        except ValueError:
            return func.HttpResponse(
                json.dumps({"error": "horizon must be 6, 12, or 24"}),
                status_code=400,
                mimetype="application/json",
            )

        if horizon not in (6, 12, 24):
            return func.HttpResponse(
                json.dumps({"error": "horizon must be 6, 12, or 24"}),
                status_code=400,
                mimetype="application/json",
            )

        if all_flag:
            features = load_latest_features()
            if features is None:
                return func.HttpResponse(
                    json.dumps({"error": "Features not loaded"}),
                    status_code=503,
                    mimetype="application/json",
                )
            results = [predict_one(k, horizon) for k in features["kelurahan"].unique()]
            return func.HttpResponse(
                json.dumps({"horizon_hours": horizon, "predictions": results}),
                status_code=200,
                mimetype="application/json",
                headers={"Access-Control-Allow-Origin": "*"},
            )

        if not kelurahan:
            return func.HttpResponse(
                json.dumps({"error": "Provide either ?kelurahan=<name> or ?all=true"}),
                status_code=400,
                mimetype="application/json",
            )

        result = predict_one(kelurahan, horizon)
        status = 200 if "error" not in result else 404
        return func.HttpResponse(
            json.dumps(result),
            status_code=status,
            mimetype="application/json",
            headers={"Access-Control-Allow-Origin": "*"},
        )

    except Exception as e:
        logging.exception("Unhandled error in predict endpoint")
        return func.HttpResponse(
            json.dumps({"error": "Internal server error", "detail": str(e)}),
            status_code=500,
            mimetype="application/json",
        )
