"""Master data pipeline — assembles all sources into a model-ready DataFrame."""
import logging

import pandas as pd

from flood_risk.config import HORIZONS, PILOT_KELURAHAN, TRAIN_END, VAL_END, VAL_START
from flood_risk.data.bmkg import BMKGLoader
from flood_risk.data.dem import DEMFeatureExtractor
from flood_risk.data.water_level import WaterLevelLoader

log = logging.getLogger(__name__)


class FloodDataPipeline:
    """
    Orchestrates ingestion from BMKG, Jakarta Open Data, and DEM.

    Output schema (per kelurahan, per hour, per horizon):
      - Rainfall features (rolling sums, intensity flags)
      - Water level features (levels, deltas, lag history)
      - Static DEM features (broadcast to each row)
      - Calendar features (hour, month, is_wet_season)
      - Target labels: flood_6h, flood_12h, flood_24h
    """

    def __init__(
        self,
        bmkg: BMKGLoader | None = None,
        water: WaterLevelLoader | None = None,
        dem: DEMFeatureExtractor | None = None,
    ):
        self.bmkg = bmkg or BMKGLoader()
        self.water = water or WaterLevelLoader()
        self.dem = dem or DEMFeatureExtractor()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        start: str = "2018-01-01",
        end: str = VAL_END,
    ) -> dict[str, pd.DataFrame]:
        """
        Return dict keyed by kelurahan name, each value a model-ready DataFrame.
        Rows: hourly timestamps in [start, end].
        """
        log.info("Loading BMKG rainfall %s → %s", start, end)
        rain_df = self.bmkg.load(start, end)

        log.info("Loading water level data")
        wl_df = self.water.load(start, end)

        log.info("Loading DEM static features")
        dem_df = self.dem.load()

        log.info("Building per-kelurahan feature matrices")
        result = {}
        for kelurahan in PILOT_KELURAHAN:
            df = self._build_for_kelurahan(kelurahan, rain_df, wl_df, dem_df)
            result[kelurahan] = df
            log.debug("  %s → shape %s", kelurahan, df.shape)

        return result

    def build_combined(self, start: str = "2018-01-01", end: str = VAL_END) -> pd.DataFrame:
        """
        Like build() but stacks all kelurahan into a single DataFrame with
        a 'kelurahan' column. Useful for training one global model.
        """
        per_kel = self.build(start, end)
        frames = []
        for name, df in per_kel.items():
            df = df.copy()
            df["kelurahan"] = name
            frames.append(df)
        combined = pd.concat(frames).sort_index()
        log.info("Combined dataset: %d rows × %d cols", *combined.shape)
        return combined

    def train_val_split(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        train = df.loc[:TRAIN_END].copy()
        val = df.loc[VAL_START:VAL_END].copy()
        log.info("Train: %d rows | Val: %d rows", len(train), len(val))
        return train, val

    # ------------------------------------------------------------------
    # Per-kelurahan assembly
    # ------------------------------------------------------------------

    def _build_for_kelurahan(
        self,
        kelurahan: str,
        rain_df: pd.DataFrame,
        wl_df: pd.DataFrame,
        dem_df: pd.DataFrame,
    ) -> pd.DataFrame:
        df = pd.DataFrame(index=rain_df.index)

        # --- Rainfall features ---
        df = pd.concat([df, self._rainfall_features(rain_df)], axis=1)

        # --- Water level features ---
        df = pd.concat([df, self._water_level_features(wl_df)], axis=1)

        # --- Calendar features ---
        df = pd.concat([df, self._calendar_features(df.index)], axis=1)

        # --- Static DEM (broadcast scalar to all rows) ---
        if kelurahan in dem_df.index:
            for col, val in dem_df.loc[kelurahan].items():
                df[f"dem_{col}"] = val

        # --- Target labels ---
        for h in HORIZONS:
            df[f"flood_{h}h"] = self.water.flood_label(wl_df, h).reindex(df.index).fillna(0).astype(int)

        # Drop rows at the head that have NaN from lag/rolling (warm-up period)
        df = df.dropna()
        return df

    # ------------------------------------------------------------------
    # Feature builders
    # ------------------------------------------------------------------

    @staticmethod
    def _rainfall_features(rain_df: pd.DataFrame) -> pd.DataFrame:
        from flood_risk.config import RAINFALL_WINDOWS_H

        feat = pd.DataFrame(index=rain_df.index)
        rain_total = rain_df.sum(axis=1)  # station-average
        rain_mean = rain_df.mean(axis=1)

        feat["rain_total_mm"] = rain_total
        feat["rain_max_station_mm"] = rain_df.max(axis=1)

        for w in RAINFALL_WINDOWS_H:
            feat[f"rain_sum_{w}h"] = rain_total.rolling(w, min_periods=1).sum()
            feat[f"rain_mean_{w}h"] = rain_mean.rolling(w, min_periods=1).mean()
            feat[f"rain_max_{w}h"] = rain_total.rolling(w, min_periods=1).max()

        # Intensity flag: is it raining heavily RIGHT NOW across >1 station?
        feat["rain_heavy_flag"] = (rain_total > 20).astype(int)  # >20mm/h combined
        feat["rain_extreme_flag"] = (rain_total > 50).astype(int)

        # Antecedent soil moisture proxy (72h cumulative)
        feat["antecedent_rain_72h"] = rain_total.rolling(72).sum()

        return feat

    @staticmethod
    def _water_level_features(wl_df: pd.DataFrame) -> pd.DataFrame:
        from flood_risk.config import WATER_LEVEL_LAGS_H

        feat = pd.DataFrame(index=wl_df.index)
        level_cols = [c for c in wl_df.columns if c.endswith("_level_cm")]
        delta_cols = [c for c in wl_df.columns if c.endswith("_delta_cm")]

        # Raw levels
        feat[level_cols] = wl_df[level_cols]
        feat[delta_cols] = wl_df[delta_cols]

        # Aggregate across stations
        feat["wl_max_cm"] = wl_df[level_cols].max(axis=1)
        feat["wl_mean_cm"] = wl_df[level_cols].mean(axis=1)
        feat["wl_rising_flag"] = (wl_df[delta_cols].max(axis=1) > 5).astype(int)

        # Lag features for max station level
        max_level = wl_df[level_cols].max(axis=1)
        for lag in WATER_LEVEL_LAGS_H:
            feat[f"wl_max_lag_{lag}h"] = max_level.shift(lag)

        # Rolling statistics
        for w in [3, 6, 12, 24]:
            feat[f"wl_max_roll_{w}h"] = max_level.rolling(w).max()
            feat[f"wl_trend_{w}h"] = max_level - max_level.shift(w)

        # How long has level been above warning threshold?
        feat["hours_above_warning"] = (
            (max_level >= 700)  # Siaga 2 at Manggarai
            .groupby((max_level < 700).cumsum())
            .cumcount()
        )

        return feat

    @staticmethod
    def _calendar_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
        feat = pd.DataFrame(index=idx)
        feat["hour"] = idx.hour
        feat["day_of_week"] = idx.dayofweek
        feat["month"] = idx.month
        feat["is_wet_season"] = ((idx.month >= 10) | (idx.month <= 4)).astype(int)
        # Peak flood risk: Jan–Feb and during evening convective storms (17-21h)
        feat["is_peak_month"] = idx.month.isin([1, 2]).astype(int)
        feat["is_storm_hour"] = idx.hour.isin(range(17, 22)).astype(int)
        # Cyclical encoding
        feat["hour_sin"] = pd.Series(idx.hour, index=idx).apply(lambda h: __import__("math").sin(2 * 3.14159 * h / 24))
        feat["hour_cos"] = pd.Series(idx.hour, index=idx).apply(lambda h: __import__("math").cos(2 * 3.14159 * h / 24))
        feat["month_sin"] = pd.Series(idx.month, index=idx).apply(lambda m: __import__("math").sin(2 * 3.14159 * m / 12))
        feat["month_cos"] = pd.Series(idx.month, index=idx).apply(lambda m: __import__("math").cos(2 * 3.14159 * m / 12))
        return feat
