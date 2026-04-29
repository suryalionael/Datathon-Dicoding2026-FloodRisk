"""Health check endpoint."""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import azure.functions as func


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Health check hit")

    model_dir = Path(__file__).parent.parent / "models"
    expected_models = ["flood_xgb_6h.joblib", "flood_xgb_12h.joblib", "flood_xgb_24h.joblib"]
    model_status = {m: (model_dir / m).exists() for m in expected_models}

    all_healthy = all(model_status.values())

    payload = {
        "status": "healthy" if all_healthy else "degraded",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "floodcast-prediction-api",
        "version": "0.1.0",
        "models_available": model_status,
    }

    return func.HttpResponse(
        json.dumps(payload),
        status_code=200 if all_healthy else 503,
        mimetype="application/json",
        headers={"Access-Control-Allow-Origin": "*"},
    )
