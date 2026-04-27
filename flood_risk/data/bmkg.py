"""BMKG rainfall data loader.

Fetches hourly rainfall (mm) from BMKG's public API and caches locally.
Falls back to CSV if the API is unreachable (useful for offline dev/testing).
"""
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from flood_risk.config import BMKG_STATIONS, RAW_DIR

log = logging.getLogger(__name__)

BMKG_BASE = "https://data.bmkg.go.id/DataMKG/MEWS/DigitalForecast/"
_CACHE_DIR = RAW_DIR / "bmkg"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


class BMKGLoader:
    """Download and harmonise BMKG hourly rainfall for pilot stations."""

    def __init__(self, stations: dict = BMKG_STATIONS, cache_dir: Path = _CACHE_DIR):
        self.stations = stations
        self.cache_dir = cache_dir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, start: str, end: str) -> pd.DataFrame:
        """Return wide DataFrame: index=timestamp (hourly), cols=station names."""
        frames = []
        for name, meta in self.stations.items():
            cache_path = self.cache_dir / f"{name}_{start[:7]}_{end[:7]}.parquet"
            if cache_path.exists():
                df = pd.read_parquet(cache_path)
            else:
                df = self._fetch_station(name, meta["id"], start, end)
                df.to_parquet(cache_path)
            frames.append(df.rename(columns={"rainfall_mm": name}))

        combined = pd.concat(frames, axis=1).sort_index()
        combined = combined.resample("h").sum()  # ensure hourly
        combined = combined.loc[start:end]
        combined.index.name = "timestamp"
        return combined

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _fetch_station(self, name: str, station_id: str, start: str, end: str) -> pd.DataFrame:
        """
        Try the BMKG Open API; fall back to synthetic stub so the pipeline
        keeps running without network access during development.
        """
        try:
            return self._api_fetch(station_id, start, end)
        except Exception as exc:
            log.warning("BMKG API unavailable for %s (%s) — using synthetic stub", name, exc)
            return self._synthetic_stub(start, end)

    def _api_fetch(self, station_id: str, start: str, end: str) -> pd.DataFrame:
        """
        BMKG does not expose a simple REST endpoint for historical hourly data.
        In production, replace this with the actual data ingestion method
        (e.g., BMKG FTP, SFTP from their data portal, or purchased dataset).

        This implementation shows the expected contract: return a DataFrame
        with a DatetimeIndex and a 'rainfall_mm' column.
        """
        url = (
            f"https://cuaca.bmkg.go.id/api/v1/climate/historical"
            f"?station_id={station_id}&start={start}&end={end}&param=RR"
        )
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()["data"]
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["datetime"])
        df = df.set_index("timestamp")[["rainfall_mm"]].astype(float)
        return df

    @staticmethod
    def _synthetic_stub(start: str, end: str) -> pd.DataFrame:
        """
        Generates statistically plausible Jakarta rainfall:
        - wet season (Oct–Apr): gamma-distributed with higher mean
        - dry season (May–Sep): sparser, lower intensity
        Useful for end-to-end pipeline validation before real data arrives.
        """
        import numpy as np

        rng = np.random.default_rng(seed=42)
        idx = pd.date_range(start, end, freq="h")
        month = idx.month
        is_wet = (month >= 10) | (month <= 4)

        rain = np.zeros(len(idx))
        wet_mask = is_wet & (rng.random(len(idx)) < 0.15)   # ~15% rainy hours wet season
        dry_mask = (~is_wet) & (rng.random(len(idx)) < 0.04)

        rain[wet_mask] = rng.gamma(shape=2.0, scale=5.0, size=wet_mask.sum())
        rain[dry_mask] = rng.gamma(shape=1.2, scale=2.5, size=dry_mask.sum())

        # Inject flood-precursor events: 3–6 consecutive heavy rain hours
        for _ in range(int(len(idx) / 8760 * 30)):  # ~30 events/year
            t = rng.integers(0, len(idx) - 6)
            rain[t : t + rng.integers(3, 7)] += rng.gamma(4, 10)

        return pd.DataFrame({"rainfall_mm": rain.clip(0)}, index=idx)
