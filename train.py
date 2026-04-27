"""
Main training script for Jakarta Flood Risk MVP.

Usage:
    python train.py                        # full run, synthetic data
    python train.py --tune --trials 50     # with Optuna HPO
    python train.py --start 2020-01-01     # shorter history
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--tune", action="store_true", help="Run Optuna HPO")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--shap", action="store_true", help="Generate SHAP reports")
    args = parser.parse_args()

    from flood_risk.config import TRAIN_END, VAL_START, VAL_END
    from flood_risk.data.pipeline import FloodDataPipeline
    from flood_risk.models.xgb_flood import MultiHorizonFloodModel
    from flood_risk.evaluation.metrics import evaluate_all_horizons
    from flood_risk.evaluation.explainability import run_full_shap_report

    # 1. Build dataset
    log.info("Building dataset %s → %s", args.start, args.end)
    pipeline = FloodDataPipeline()
    combined = pipeline.build_combined(start=args.start, end=args.end)

    train_df, val_df = pipeline.train_val_split(combined)

    log.info("Train flood rates (24h horizon):")
    for h in [6, 12, 24]:
        rate = train_df[f"flood_{h}h"].mean()
        log.info("  %dh: %.2f%%", h, rate * 100)

    # 2. Hyperparameter tuning (optional)
    best_params = None
    if args.tune:
        log.info("Running Optuna HPO (%d trials per horizon)", args.trials)
        from flood_risk.models.tuner import tune_all_horizons
        best_params_by_horizon = tune_all_horizons(train_df, val_df, args.trials)
        # Use 6h best params as global default (simplification for MVP)
        best_params = best_params_by_horizon.get(6)

    # 3. Train
    log.info("Training MultiHorizonFloodModel")
    model = MultiHorizonFloodModel(params=best_params)
    model.fit(train_df, val_df)

    # 4. Evaluate
    log.info("Evaluating on 2024 validation set")
    summary = evaluate_all_horizons(model, val_df)
    summary.to_csv(ROOT / "reports" / "validation_metrics.csv")
    log.info("\n%s", summary.to_string())

    # Check target F1
    target = 0.65
    passing = summary["f1"] >= target
    if passing.all():
        log.info("All horizons meet F1 ≥ %.2f target ✓", target)
    else:
        failing = summary[~passing].index.tolist()
        log.warning("Horizons failing F1 target: %s", failing)

    # 5. Save models
    model.save()
    log.info("Models saved to %s", ROOT / "models")

    # 6. SHAP explainability
    if args.shap:
        log.info("Generating SHAP reports")
        run_full_shap_report(model, val_df)

    log.info("Done.")
    return summary


if __name__ == "__main__":
    main()
