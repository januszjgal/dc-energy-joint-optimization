"""Canonical hourly market-panel validation and train-only statistics."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from env.ramp_v6.models import FORECAST_HOURS, FrozenRampStats, HORIZONS


BASE_COLUMNS = (
    "timestamp_utc",
    "market_id",
    "gross_demand_mw",
    "net_load_mw",
    "wind_mw",
    "solar_mw",
    "market_scale_mw",
    "da_lmp_usd_per_mwh",
    "forecast_issue_time_utc",
    "forecast_vintage_id",
    "quality_ok",
)
FORECAST_COLUMNS = tuple(
    f"forecast_{quantity}_h{hour}_mw"
    for quantity in ("gross", "net")
    for hour in FORECAST_HOURS
)
REQUIRED_COLUMNS = BASE_COLUMNS + FORECAST_COLUMNS


class CanonicalMarketPanel:
    """Validated long-form, complete hourly UTC market panel."""

    def __init__(self, frame: pd.DataFrame):
        self.frame = self._validate(frame)
        self.markets = tuple(sorted(self.frame["market_id"].unique()))
        self.timestamps = pd.DatetimeIndex(
            self.frame["timestamp_utc"].drop_duplicates().sort_values()
        )
        indexed = self.frame.set_index(["timestamp_utc", "market_id"])
        self._observation_cache = tuple(
            indexed.loc[timestamp].loc[list(self.markets)]
            for timestamp in self.timestamps
        )

    @classmethod
    def from_csv(cls, path: Path) -> CanonicalMarketPanel:
        return cls(pd.read_csv(path))

    @staticmethod
    def _validate(frame: pd.DataFrame) -> pd.DataFrame:
        missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
        if missing:
            raise ValueError(f"canonical panel missing columns: {missing}")
        result = frame.loc[:, REQUIRED_COLUMNS].copy()
        for column in ("market_id", "forecast_vintage_id"):
            valid = result[column].map(
                lambda value: isinstance(value, str) and bool(value.strip())
            )
            if not bool(valid.all()):
                raise ValueError(f"{column} must contain non-blank strings")
        valid_quality = result["quality_ok"].map(
            lambda value: isinstance(value, (bool, np.bool_)) and bool(value)
        )
        if not bool(valid_quality.all()):
            raise ValueError("quality_ok must contain the boolean value true")
        result["timestamp_utc"] = pd.to_datetime(
            result["timestamp_utc"], utc=True, errors="raise"
        )
        result["forecast_issue_time_utc"] = pd.to_datetime(
            result["forecast_issue_time_utc"], utc=True, errors="raise"
        )
        result["market_id"] = result["market_id"].astype(str)
        result["forecast_vintage_id"] = result["forecast_vintage_id"].astype(str)
        if result.duplicated(["timestamp_utc", "market_id"]).any():
            raise ValueError("canonical panel has duplicate market-hours")
        numeric = [
            column
            for column in REQUIRED_COLUMNS
            if column not in {
                "timestamp_utc",
                "forecast_issue_time_utc",
                "market_id",
                "forecast_vintage_id",
                "quality_ok",
            }
        ]
        if not np.isfinite(result[numeric].to_numpy(dtype=np.float64)).all():
            raise ValueError("canonical panel numeric fields must be finite")
        if (result["market_scale_mw"] <= 0.0).any():
            raise ValueError("market_scale_mw must be positive")
        if (result["forecast_issue_time_utc"] > result["timestamp_utc"]).any():
            raise ValueError("forecast issue time cannot be later than controller time")
        result["market_id"] = result["market_id"].str.strip()
        result["forecast_vintage_id"] = result["forecast_vintage_id"].str.strip()
        result["quality_ok"] = True
        result = result.sort_values(["timestamp_utc", "market_id"]).reset_index(drop=True)
        markets = sorted(result["market_id"].unique())
        counts = result.groupby("timestamp_utc")["market_id"].nunique()
        if not (counts == len(markets)).all():
            raise ValueError("canonical panel must contain every market every hour")
        timestamps = pd.DatetimeIndex(result["timestamp_utc"].drop_duplicates())
        expected = pd.date_range(timestamps.min(), timestamps.max(), freq="h", tz="UTC")
        if not timestamps.equals(expected):
            raise ValueError("canonical panel must be contiguous at hourly UTC cadence")
        return result

    def at(self, timestamp: pd.Timestamp) -> pd.DataFrame:
        rows = self.frame[self.frame["timestamp_utc"] == timestamp]
        return rows.set_index("market_id").loc[list(self.markets)]

    def observation_rows(self, index: int) -> pd.DataFrame:
        """Return only controller-time data; future realized rows are inaccessible."""
        if index < 0 or index >= len(self.timestamps):
            raise IndexError(index)
        return self._observation_cache[index]

    def fit_stats(self, train_months: Iterable[str]) -> FrozenRampStats:
        months = tuple(train_months)
        if not months:
            raise ValueError("at least one training month is required")
        month_labels = self.frame["timestamp_utc"].dt.strftime("%Y-%m")
        train = self.frame[month_labels.isin(months)].copy()
        if set(train["timestamp_utc"].dt.strftime("%Y-%m").unique()) != set(months):
            raise ValueError("every requested training month must exist")
        _require_complete_months(train, months, self.markets)

        gross_q95: dict[str, float] = {}
        gross_mean: dict[str, float] = {}
        gross_std: dict[str, float] = {}
        net_mean: dict[str, float] = {}
        net_std: dict[str, float] = {}
        thresholds: dict[str, dict[int, float]] = {}
        for market, rows in train.groupby("market_id", sort=True):
            rows = rows.sort_values("timestamp_utc")
            gross = rows["gross_demand_mw"].to_numpy(dtype=np.float64)
            net = rows["net_load_mw"].to_numpy(dtype=np.float64)
            scale = float(np.quantile(gross, 0.95))
            gross_q95[market] = scale
            gross_mean[market] = float(np.mean(gross))
            gross_std[market] = max(float(np.std(gross)), 1e-9)
            net_mean[market] = float(np.mean(net))
            net_std[market] = max(float(np.std(net)), 1e-9)
            thresholds[market] = {}
            for horizon in HORIZONS:
                ramp = (net[horizon:] - net[:-horizon]) / (scale * horizon)
                thresholds[market][horizon] = float(
                    np.quantile(np.abs(ramp), 0.90)
                )
        return FrozenRampStats(
            fit_start_utc=train["timestamp_utc"].min().isoformat(),
            fit_end_utc=train["timestamp_utc"].max().isoformat(),
            gross_q95_mw=gross_q95,
            gross_level_mean_mw=gross_mean,
            gross_level_std_mw=gross_std,
            net_level_mean_mw=net_mean,
            net_level_std_mw=net_std,
            native_abs_ramp_q90_fraction_s_per_hour=thresholds,
            fit_months=months,
        )

    def split_complete_months(
        self,
        train_months: Iterable[str],
        validation_months: Iterable[str],
        test_months: Iterable[str],
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        groups = tuple(
            tuple(months)
            for months in (train_months, validation_months, test_months)
        )
        flat = [month for group in groups for month in group]
        if len(flat) != len(set(flat)):
            raise ValueError("month split groups must be disjoint")
        if flat != sorted(flat):
            raise ValueError("primary month split must be chronological")
        labels = self.frame["timestamp_utc"].dt.strftime("%Y-%m")
        outputs = []
        for months in groups:
            subset = self.frame[labels.isin(months)].copy()
            _require_complete_months(subset, months, self.markets)
            outputs.append(subset)
        return tuple(outputs)  # type: ignore[return-value]

    @staticmethod
    def stats_dict(stats: FrozenRampStats) -> dict:
        return asdict(stats)


def _require_complete_months(
    frame: pd.DataFrame,
    months: Iterable[str],
    markets: Iterable[str],
) -> None:
    markets = tuple(markets)
    for month in months:
        start = pd.Timestamp(f"{month}-01", tz="UTC")
        end = start + pd.offsets.MonthBegin(1)
        expected_hours = len(pd.date_range(start, end, freq="h", inclusive="left"))
        rows = frame[
            (frame["timestamp_utc"] >= start) & (frame["timestamp_utc"] < end)
        ]
        if len(rows) != expected_hours * len(markets):
            raise ValueError(f"{month} is not a complete chronological month")
        if set(rows["market_id"].unique()) != set(markets):
            raise ValueError(f"{month} does not contain every panel market")
