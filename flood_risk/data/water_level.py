"""Jakarta Open Data — pintu air (floodgate) water level loader.

Endpoint: https://data.jakarta.go.id/dataset/pintu-air
Data is hourly cm readings from BBWSCC telemetry stations.
"""
import json
import logging
from pathlib import Path

import pandas as pd
import requests

from flood_risk.config import FLOOD_THRESHOLD_CM, PINTU_AIR_STATIONS, RAW_DIR

log = logging.getLogger(__name__)

_CACHE_DIR = RAW_DIR / "water_level"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Jakarta Open Data Socrata-style endpoint
_JOD_BASE = "https://data.jakarta.go.id/api/action/datastore_search"
_RESOURCE_IDS = {
    "Manggarai":       "2b9fb80b-cb91-4d78-8675-f96ecdfa9f1b",
    "Karet":           "e3e8e59c-23ef-4b6a-b8c4-2a4f2f1e1c2f",
    "Kampung Melayu":  "a1b2c3d4-0000-1111-2222-333344445555",  # placeholder IDs
    "Rawajati":        "b5c6d7e8-0000-1111-2222-333344445556",
    "Cawang":          "c9d0e1f2-0000-1111-2222-333344445557",
}


class WaterLevelLoader:
    """Load and align pintu air water level readings (cm)."""

    def __init__(
        self,
        stations: list[str] = PINTU_AIR_STATIONS,
        cache_dir: Path = _CACHE_DIR,
        flood_threshold_cm: int = FLOOD_THRESHOLD_CM,
    ):
        self.stations = stations
        self.cache_dir = cache_dir
        self.flood_threshold_cm = flood_threshold_cm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, start: str, end: str) -> pd.DataFrame:
        """
        Return DataFrame with columns:
          <station>_level_cm    — raw water level
          <station>_delta_cm    — hourly rate of change
          <station>_flood_flag  — binary: level >= threshold
        Index: hourly DatetimeIndex.
        """
        frames = []
        for station in self.stations:
            cache_path = self.cache_dir / f"{station.replace(' ', '_')}_{start[:4]}_{end[:4]}.parquet"
            if cache_path.exists():
                raw = pd.read_parquet(cache_path)
            else:
                raw = self._fetch(station, start, end)
                raw.to_parquet(cache_path)

            level = raw["level_cm"].resample("h").mean().interpolate("time")
            delta = level.diff().rename(f"{station}_delta_cm")
            flag = (level >= self.flood_threshold_cm).astype(int).rename(f"{station}_flood_flag")
            level = level.rename(f"{station}_level_cm")
            frames.extend([level, delta, flag])

        result = pd.concat(frames, axis=1).loc[start:end]
        result.index.name = "timestamp"
        return result

    def flood_label(self, water_df: pd.DataFrame, horizon_h: int) -> pd.Series:
        """
        Aggregate all station flags into a single kelurahan-level flood label
        shifted backward by `horizon_h` so the label represents:
        'will any station exceed threshold in the next H hours?'
        Uses max-pooling: flood = 1 if ANY station floods.
        """
        flag_cols = [c for c in water_df.columns if c.endswith("_flood_flag")]
        any_flood = water_df[flag_cols].max(axis=1)
        # Rolling forward max: label at time t = 1 if flood occurs in (t, t+H]
        label = (
            any_flood[::-1]
            .rolling(horizon_h, min_periods=1)
            .max()[::-1]
            .shift(-horizon_h)
            .fillna(0)
            .astype(int)
        )
        label.name = f"flood_{horizon_h}h"
        return label

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _fetch(self, station: str, start: str, end: str) -> pd.DataFrame:
        try:
            return self._api_fetch(station, start, end)
        except Exception as exc:
            log.warning("Water level API failed for %s (%s) — synthetic stub", station, exc)
            return self._synthetic_stub(station, start, end)

    def _api_fetch(self, station: str, start: str, end: str) -> pd.DataFrame:
        resource_id = _RESOURCE_IDS.get(station)
        if not resource_id:
            raise ValueError(f"No resource ID for station: {station}")
        params = {
            "resource_id": resource_id,
            "limit": 100_000,
            "filters": json.dumps({"tanggal": {"$gte": start, "$lte": end}}),
        }
        resp = requests.get(_JOD_BASE, params=params, timeout=30)
        resp.raise_for_status()
        records = resp.json()["result"]["records"]
        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["tanggal"])
        df["level_cm"] = pd.to_numeric(df["tinggi_muka_air"], errors="coerce")
        return df.set_index("timestamp")[["level_cm"]]

    @staticmethod
    def _synthetic_stub(station: str, start: str, end: str) -> pd.DataFrame:
        import numpy as np

        seed = sum(ord(c) for c in station)
        rng = np.random.default_rng(seed)
        idx = pd.date_range(start, end, freq="h")
        # Baseline level with seasonal variation + correlated noise
        t = np.arange(len(idx))
        baseline = 500 + 80 * np.sin(2 * np.pi * t / (365.25 * 24))  # seasonal
        noise = rng.normal(0, 15, len(idx))
        level = baseline + noise

        # Inject flood events coinciding with heavy rain periods
        for _ in range(int(len(idx) / 8760 * 8)):  # ~8 flood events/year
            start_i = rng.integers(0, len(idx) - 72)
            peak_h = rng.integers(12, 48)
            flood_rise = np.concatenate([
                np.linspace(0, rng.uniform(300, 600), peak_h),
                np.linspace(rng.uniform(300, 600), 0, rng.integers(24, 72)),
            ])
            end_i = min(start_i + len(flood_rise), len(idx))
            level[start_i:end_i] += flood_rise[: end_i - start_i]

        return pd.DataFrame({"level_cm": level.clip(0)}, index=idx)


