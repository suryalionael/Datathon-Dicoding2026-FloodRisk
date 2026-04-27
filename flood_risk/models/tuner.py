"""Optuna hyperparameter search for XGBoost flood models."""
import logging

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight

from flood_risk.config import HORIZONS, MODELS_DIR
from flood_risk.models.xgb_flood import FloodRiskModel, _EXCLUDE_COLS

log = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune_horizon(
    horizon_h: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    n_trials: int = 50,
) -> dict:
    """Run Optuna search; return best params dict."""
    assert horizon_h in HORIZONS

    target = f"flood_{horizon_h}h"
    feature_cols = [c for c in train_df.columns if c not in _EXCLUDE_COLS]

    X_train = train_df[feature_cols].select_dtypes(include=[np.number])
    y_train = train_df[target]
    X_val = val_df[feature_cols].select_dtypes(include=[np.number])
    y_val = val_df[target]

    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    sample_weight = compute_sample_weight("balanced", y_train)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary:logistic",
            "tree_method": "hist",
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma": trial.suggest_float("gamma", 0, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "scale_pos_weight": pos_weight,
            "random_state": 42,
            "n_jobs": -1,
            "early_stopping_rounds": 30,
        }
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            sample_weight=sample_weight,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        probs = model.predict_proba(X_val)[:, 1]
        # Optimise over F1 at best threshold
        best_f1 = max(
            f1_score(y_val, (probs >= t).astype(int), zero_division=0)
            for t in np.arange(0.1, 0.9, 0.05)
        )
        return best_f1

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    log.info("Horizon %dh | best F1=%.4f | params=%s", horizon_h, study.best_value, best)
    return best


def tune_all_horizons(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    n_trials: int = 50,
) -> dict[int, dict]:
    """Tune all horizons; return dict horizon → best_params."""
    return {h: tune_horizon(h, train_df, val_df, n_trials) for h in HORIZONS}
