"""
Inference script — generate flood risk alerts for current/future timestamps.

Usage:
    python predict.py --kelurahan "Kampung Melayu" --timestamp "2024-02-15 18:00"
    python predict.py --all  # all 15 pilot kelurahan, latest data
"""
import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("predict")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kelurahan", default=None)
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    from flood_risk.config import MODELS_DIR, PILOT_KELURAHAN
    from flood_risk.data.pipeline import FloodDataPipeline
    from flood_risk.models.xgb_flood import MultiHorizonFloodModel
    from flood_risk.evaluation.explainability import FloodSHAPExplainer

    ts = args.timestamp or datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    end = ts[:10]
    start = (datetime.strptime(end, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")

    log.info("Loading recent data %s → %s", start, end)
    pipeline = FloodDataPipeline()
    combined = pipeline.build_combined(start=start, end=end)

    model = MultiHorizonFloodModel.load(MODELS_DIR)

    kelurahan_list = (
        list(PILOT_KELURAHAN.keys()) if args.all
        else ([args.kelurahan] if args.kelurahan else list(PILOT_KELURAHAN.keys()))
    )

    from flood_risk.models.xgb_flood import _EXCLUDE_COLS
    import numpy as np

    for kel in kelurahan_list:
        subset = combined[combined["kelurahan"] == kel] if "kelurahan" in combined.columns else combined
        if subset.empty:
            log.warning("No data for %s", kel)
            continue

        feature_cols = [c for c in subset.columns if c not in _EXCLUDE_COLS and c != "kelurahan"]
        X = subset[feature_cols].select_dtypes(include=[np.number]).tail(1)

        alerts = model.predict_all(X)
        print(f"\n{'='*50}")
        print(f"Kelurahan: {kel} | Time: {X.index[-1]}")
        for h in [6, 12, 24]:
            prob = alerts[f"prob_{h}h"].iloc[0]
            level = alerts[f"risk_level_{h}h"].iloc[0]
            bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
            print(f"  {h:2d}h: [{bar}] {prob:.1%}  →  {level}")


if __name__ == "__main__":
    main()
