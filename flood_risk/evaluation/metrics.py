"""Evaluation metrics for flood risk models."""
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

from flood_risk.config import HORIZONS

log = logging.getLogger(__name__)


def evaluate_model(model, val_df: pd.DataFrame) -> dict:
    """
    Full evaluation suite for a single FloodRiskModel.
    Returns metrics dict; also logs a human-readable summary.
    """
    from flood_risk.models.xgb_flood import _EXCLUDE_COLS

    feature_cols = [c for c in val_df.columns if c not in _EXCLUDE_COLS]
    X_val = val_df[feature_cols].select_dtypes(include=[np.number])
    y_val = val_df[model.target_col]

    probs = model.predict_proba(X_val)
    preds = model.predict(X_val)

    pos_rate = y_val.mean()
    if pos_rate == 0 or pos_rate == 1:
        log.warning("Degenerate label distribution (pos_rate=%.3f) — skip eval", pos_rate)
        return {}

    metrics = {
        "horizon_h": model.horizon_h,
        "n_samples": len(y_val),
        "flood_rate": float(pos_rate),
        "f1": float(f1_score(y_val, preds, zero_division=0)),
        "f1_weighted": float(f1_score(y_val, preds, average="weighted", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_val, probs)),
        "pr_auc": float(average_precision_score(y_val, probs)),
        "threshold": model.threshold,
    }

    cm = confusion_matrix(y_val, preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    metrics.update({
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "false_alarm_rate": float(fp / max(fp + tn, 1)),
        "miss_rate": float(fn / max(fn + tp, 1)),
    })

    target_f1 = 0.65
    passed = metrics["f1"] >= target_f1
    log.info(
        "Horizon %dh | F1=%.3f (%s) | ROC-AUC=%.3f | PR-AUC=%.3f | Recall=%.3f | FAR=%.3f",
        model.horizon_h,
        metrics["f1"],
        "PASS ✓" if passed else f"FAIL — target {target_f1}",
        metrics["roc_auc"],
        metrics["pr_auc"],
        metrics["recall"],
        metrics["false_alarm_rate"],
    )
    return metrics


def evaluate_all_horizons(multi_model, val_df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate MultiHorizonFloodModel and return a summary DataFrame."""
    rows = []
    for h, model in multi_model.models.items():
        metrics = evaluate_model(model, val_df)
        if metrics:
            rows.append(metrics)
    summary = pd.DataFrame(rows).set_index("horizon_h")
    log.info("\n%s", summary[["f1", "roc_auc", "pr_auc", "recall", "false_alarm_rate"]].to_string())
    return summary


def find_optimal_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    beta: float = 1.0,
    min_recall: float = 0.70,
) -> float:
    """
    Find threshold maximising F-beta under a minimum recall constraint.
    Default beta=1 → F1. For flood safety, consider beta=2 (recall-weighted).
    min_recall=0.70 ensures we catch ≥70% of actual floods regardless of precision.
    """
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    valid = recall[:-1] >= min_recall
    if not valid.any():
        log.warning("No threshold satisfies min_recall=%.2f — relaxing", min_recall)
        valid = np.ones(len(thresholds), dtype=bool)

    b2 = beta ** 2
    fbeta = (1 + b2) * precision[:-1] * recall[:-1] / (b2 * precision[:-1] + recall[:-1] + 1e-9)
    fbeta[~valid] = -1
    best_idx = np.argmax(fbeta)
    return float(thresholds[best_idx])
