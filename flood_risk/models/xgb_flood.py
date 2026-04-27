"""XGBoost flood risk model — one model per forecast horizon."""
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight

from flood_risk.config import HORIZONS, MODELS_DIR, RISK_LEVELS, XGB_PARAMS

log = logging.getLogger(__name__)

# Feature columns that are NOT targets or metadata
_EXCLUDE_COLS = {"kelurahan"} | {f"flood_{h}h" for h in HORIZONS}


class FloodRiskModel:
    """
    Wrapper around XGBClassifier for a single forecast horizon.
    Handles class imbalance via sample weights, early stopping,
    and threshold calibration.
    """

    def __init__(self, horizon_h: int, params: dict | None = None):
        assert horizon_h in HORIZONS, f"horizon_h must be one of {HORIZONS}"
        self.horizon_h = horizon_h
        self.target_col = f"flood_{horizon_h}h"
        self.params = {**XGB_PARAMS, **(params or {})}
        self.model: xgb.XGBClassifier | None = None
        self.feature_names: list[str] = []
        self.threshold: float = 0.5  # calibrated post-training

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        early_stopping_rounds: int = 50,
    ) -> "FloodRiskModel":
        X_train, y_train = self._split_xy(train_df)
        X_val, y_val = self._split_xy(val_df)
        self.feature_names = X_train.columns.tolist()

        # Class imbalance — flood events ~2-8% of hours
        sample_weight = compute_sample_weight("balanced", y_train)
        pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        log.info(
            "Horizon %dh | train flood rate: %.2f%% | scale_pos_weight≈%.1f",
            self.horizon_h, y_train.mean() * 100, pos_weight,
        )

        self.model = xgb.XGBClassifier(
            **self.params,
            scale_pos_weight=pos_weight,
            early_stopping_rounds=early_stopping_rounds,
        )
        self.model.fit(
            X_train, y_train,
            sample_weight=sample_weight,
            eval_set=[(X_val, y_val)],
            verbose=100,
        )
        self.threshold = self._calibrate_threshold(X_val, y_val)
        log.info("Horizon %dh | best iteration: %d | threshold: %.3f",
                 self.horizon_h, self.model.best_iteration, self.threshold)
        return self

    def _calibrate_threshold(self, X_val: pd.DataFrame, y_val: pd.Series) -> float:
        """Find threshold maximising F1 on validation set."""
        from sklearn.metrics import f1_score
        probs = self.predict_proba(X_val)
        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.1, 0.9, 0.02):
            preds = (probs >= t).astype(int)
            f1 = f1_score(y_val, preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        log.info("Horizon %dh | calibrated threshold %.3f → F1 %.4f", self.horizon_h, best_t, best_f1)
        return best_t

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_aligned = self._align_features(X)
        return self.model.predict_proba(X_aligned)[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def classify(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame with prob and risk level string."""
        prob = self.predict_proba(X)
        levels = pd.cut(
            prob,
            bins=[0, 0.20, 0.50, 0.80, 1.01],
            labels=["Aman", "Waspada", "Siaga", "Bahaya"],
            right=False,
        )
        return pd.DataFrame(
            {"prob": prob, "risk_level": levels, "horizon_h": self.horizon_h},
            index=X.index if hasattr(X, "index") else None,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | None = None) -> Path:
        path = path or MODELS_DIR / f"flood_xgb_{self.horizon_h}h.joblib"
        joblib.dump(self, path)
        log.info("Saved model → %s", path)
        return path

    @classmethod
    def load(cls, path: Path) -> "FloodRiskModel":
        return joblib.load(path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _split_xy(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        feature_cols = [c for c in df.columns if c not in _EXCLUDE_COLS]
        X = df[feature_cols].select_dtypes(include=[np.number])
        y = df[self.target_col]
        return X, y

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ensure feature columns match training order; fill missing with 0."""
        X = X.reindex(columns=self.feature_names, fill_value=0)
        return X


class MultiHorizonFloodModel:
    """Container for 6h, 12h, 24h models with a unified prediction API."""

    def __init__(self, params: dict | None = None):
        self.models: dict[int, FloodRiskModel] = {
            h: FloodRiskModel(h, params) for h in HORIZONS
        }

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> "MultiHorizonFloodModel":
        for h, model in self.models.items():
            log.info("=" * 50)
            log.info("Training %d-hour horizon model", h)
            model.fit(train_df, val_df)
        return self

    def predict_all(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return wide DataFrame: cols = (prob_6h, risk_6h, prob_12h, ...) per row."""
        frames = []
        for h, model in self.models.items():
            out = model.classify(X)
            out.columns = [f"{c}_{h}h" if c != "horizon_h" else c for c in out.columns]
            frames.append(out.drop(columns=["horizon_h"]))
        return pd.concat(frames, axis=1)

    def save(self, directory: Path = MODELS_DIR) -> None:
        for h, model in self.models.items():
            model.save(directory / f"flood_xgb_{h}h.joblib")

    @classmethod
    def load(cls, directory: Path = MODELS_DIR) -> "MultiHorizonFloodModel":
        obj = cls.__new__(cls)
        obj.models = {h: FloodRiskModel.load(directory / f"flood_xgb_{h}h.joblib") for h in HORIZONS}
        return obj
