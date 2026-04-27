"""SHAP-based explainability for flood risk predictions."""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from flood_risk.config import REPORTS_DIR

log = logging.getLogger(__name__)


class FloodSHAPExplainer:
    """
    Wraps shap.TreeExplainer for a FloodRiskModel.
    Provides:
      - global feature importance (summary plot, mean |SHAP|)
      - local explanation for a single prediction event
      - kelurahan-level aggregated importance
    """

    def __init__(self, model, background_df: pd.DataFrame | None = None):
        import shap

        self.model = model
        self.feature_names = model.feature_names
        self.explainer = shap.TreeExplainer(
            model.model,
            feature_perturbation="tree_path_dependent",
        )
        self._background = background_df

    # ------------------------------------------------------------------
    # Global explanations
    # ------------------------------------------------------------------

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Return SHAP values array (n_samples, n_features)."""
        X_aligned = self.model._align_features(X)
        sv = self.explainer.shap_values(X_aligned)
        # TreeExplainer returns list[array] for binary; take positive class
        if isinstance(sv, list):
            sv = sv[1]
        return sv

    def global_importance(self, X: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
        """Mean absolute SHAP values — global feature importance table."""
        sv = self.shap_values(X)
        importance = pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_shap": np.abs(sv).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False).head(top_n)
        return importance

    def summary_plot(self, X: pd.DataFrame, save_path: Path | None = None) -> None:
        import shap
        import matplotlib.pyplot as plt

        X_aligned = self.model._align_features(X)
        sv = self.shap_values(X)
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(sv, X_aligned, feature_names=self.feature_names, show=False, plot_size=None)
        plt.title(f"SHAP Summary — {self.model.horizon_h}h Flood Risk")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            log.info("SHAP summary plot → %s", save_path)
        plt.close()

    def importance_bar_plot(self, X: pd.DataFrame, top_n: int = 20, save_path: Path | None = None) -> None:
        import matplotlib.pyplot as plt

        imp = self.global_importance(X, top_n)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(imp["feature"][::-1], imp["mean_abs_shap"][::-1])
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"Top {top_n} Features — {self.model.horizon_h}h Horizon")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    # ------------------------------------------------------------------
    # Local explanations
    # ------------------------------------------------------------------

    def explain_event(self, X_event: pd.DataFrame) -> pd.DataFrame:
        """
        Return SHAP values for a single event (one or few rows).
        Useful for explaining a specific flood alert to operators.
        """
        sv = self.shap_values(X_event)
        explanation = pd.DataFrame(
            sv, columns=self.feature_names, index=X_event.index
        )
        # Add feature values alongside SHAP
        top = (
            explanation.abs().mean()
            .sort_values(ascending=False)
            .head(10)
            .index.tolist()
        )
        out = []
        for feat in top:
            out.append({
                "feature": feat,
                "value": float(X_event[feat].iloc[0]) if feat in X_event.columns else np.nan,
                "shap": float(explanation[feat].mean()),
                "direction": "↑ risk" if explanation[feat].mean() > 0 else "↓ risk",
            })
        return pd.DataFrame(out)

    def waterfall_plot(self, X_event: pd.DataFrame, save_path: Path | None = None) -> None:
        import shap
        import matplotlib.pyplot as plt

        X_aligned = self.model._align_features(X_event.iloc[:1])
        sv = self.explainer(X_aligned)
        shap.waterfall_plot(sv[0], show=False)
        plt.title(f"SHAP Waterfall — {self.model.horizon_h}h | {X_event.index[0]}")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    # ------------------------------------------------------------------
    # Kelurahan-level aggregation
    # ------------------------------------------------------------------

    def kelurahan_importance(
        self, df: pd.DataFrame, top_n: int = 10
    ) -> pd.DataFrame:
        """
        Compute mean |SHAP| per kelurahan.
        df must have a 'kelurahan' column.
        Returns MultiIndex DataFrame: (kelurahan, feature).
        """
        results = []
        for kel, group in df.groupby("kelurahan"):
            feat_cols = [c for c in group.columns if c not in {"kelurahan"} and c in self.feature_names]
            X = group[feat_cols].select_dtypes(include=[np.number])
            imp = self.global_importance(X, top_n=top_n)
            imp["kelurahan"] = kel
            results.append(imp)
        return pd.concat(results).set_index(["kelurahan", "feature"])

    # ------------------------------------------------------------------
    # Operator-facing alert narrative
    # ------------------------------------------------------------------

    def alert_narrative(self, X_event: pd.DataFrame, prob: float) -> str:
        """
        Generate a plain-text explanation of a flood alert for BPBD operators.
        """
        risk_level = _prob_to_level(prob)
        explanation = self.explain_event(X_event)

        drivers = explanation[explanation["shap"] > 0].head(3)
        driver_txt = "\n".join(
            f"  • {r['feature'].replace('_', ' ')}: {r['value']:.1f} ({r['direction']})"
            for _, r in drivers.iterrows()
        )
        ts = X_event.index[0].strftime("%Y-%m-%d %H:%M") if hasattr(X_event.index[0], "strftime") else str(X_event.index[0])
        return (
            f"[{ts}] PERINGATAN BANJIR — Level: {risk_level}\n"
            f"Probabilitas: {prob:.1%} dalam {self.model.horizon_h} jam ke depan\n"
            f"Faktor utama:\n{driver_txt}"
        )


def _prob_to_level(prob: float) -> str:
    from flood_risk.config import RISK_LEVELS
    for level, (lo, hi) in RISK_LEVELS.items():
        if lo <= prob < hi:
            return level
    return "Bahaya"


def run_full_shap_report(multi_model, val_df: pd.DataFrame) -> None:
    """Generate SHAP summary plots and importance CSVs for all horizons."""
    shap_dir = REPORTS_DIR / "shap"
    shap_dir.mkdir(exist_ok=True)

    from flood_risk.models.xgb_flood import _EXCLUDE_COLS

    feature_cols = [c for c in val_df.columns if c not in _EXCLUDE_COLS and c != "kelurahan"]
    X_val = val_df[feature_cols].select_dtypes(include=[np.number])
    X_sample = X_val.sample(min(2000, len(X_val)), random_state=42)

    for h, model in multi_model.models.items():
        log.info("Generating SHAP report for %dh horizon", h)
        explainer = FloodSHAPExplainer(model)
        explainer.summary_plot(X_sample, save_path=shap_dir / f"shap_summary_{h}h.png")
        explainer.importance_bar_plot(X_sample, save_path=shap_dir / f"shap_importance_{h}h.png")
        imp = explainer.global_importance(X_sample)
        imp.to_csv(shap_dir / f"shap_importance_{h}h.csv", index=False)
        log.info("  Top 5 features (h=%d): %s", h, imp["feature"].head(5).tolist())
