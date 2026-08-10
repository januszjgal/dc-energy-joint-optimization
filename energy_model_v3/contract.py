"""Canonical schemas and fail-closed validation for energy model v3."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

import numpy as np
import pandas as pd


class ContractError(ValueError):
    """Raised when source data cannot satisfy the v3 evidence contract."""


PHYSICAL_COLUMNS = {
    "interval_start_utc",
    "interval_end_utc",
    "market",
    "gross_demand_mw",
    "wind_mw",
    "solar_mw",
    "net_load_mw",
    "net_load_method",
    "market_scale_mw",
    "source_quality_flags",
}

PRICE_COLUMNS = {
    "interval_start_utc",
    "interval_end_utc",
    "market",
    "price_market",
    "price_location",
    "da_lmp_usd_mwh",
    "rt_lmp_usd_mwh",
    "energy_component_usd_mwh",
    "congestion_component_usd_mwh",
    "loss_component_usd_mwh",
    "source_quality_flags",
}

FORECAST_COLUMNS = {
    "interval_start_utc",
    "market",
    "forecast_target",
    "forecast_value_mw",
    "forecast_issue_utc",
    "forecast_vintage_utc",
    "forecast_horizon_hours",
    "forecast_capability",
    "source_quality_flags",
}


@dataclass(frozen=True)
class SourceDescriptor:
    market: str
    source: str
    feed: str
    product: str
    location: str
    authentication: str
    licensing: str
    cadence: str
    units: str
    interval_semantics: str
    source_timezone: str
    dst_rule: str
    status_field: str
    revision_policy: str
    query: dict[str, Any]
    redistribution: str
    retrieval_time_utc: str | None = None
    raw_sha256: str | None = None
    quality_flags: tuple[str, ...] = field(default_factory=tuple)

    def validate(self) -> None:
        required = {
            "market": self.market,
            "source": self.source,
            "feed": self.feed,
            "product": self.product,
            "location": self.location,
            "authentication": self.authentication,
            "licensing": self.licensing,
            "cadence": self.cadence,
            "units": self.units,
            "interval_semantics": self.interval_semantics,
            "source_timezone": self.source_timezone,
            "dst_rule": self.dst_rule,
            "status_field": self.status_field,
            "revision_policy": self.revision_policy,
            "redistribution": self.redistribution,
        }
        missing = sorted(key for key, value in required.items() if not value)
        if missing:
            raise ContractError(
                f"{self.market} source descriptor missing: {', '.join(missing)}"
            )
        if self.interval_semantics not in {"interval_beginning", "interval_ending"}:
            raise ContractError(
                f"{self.market} has unsupported interval semantics "
                f"{self.interval_semantics!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class Capability:
    market: str
    status: str
    capability: str
    reason: str
    credential: str | None = None
    checked_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_utc(series: pd.Series, label: str) -> pd.Series:
    parsed = pd.to_datetime(series, utc=True, errors="raise")
    if parsed.isna().any():
        raise ContractError(f"{label} contains missing timestamps")
    return parsed


def validate_native_table(
    frame: pd.DataFrame,
    required: Iterable[str],
    *,
    table_name: str,
    expected_market: str | None = None,
    non_nullable: Iterable[str] | None = None,
    unique_columns: Iterable[str] = ("market", "interval_start_utc"),
    numeric_columns: Iterable[str] = (),
    non_negative_columns: Iterable[str] = (),
) -> pd.DataFrame:
    """Validate a native long-form table without filling or coercing evidence."""
    required_set = set(required)
    missing = sorted(required_set - set(frame.columns))
    if missing:
        raise ContractError(
            f"{table_name} missing required columns: {', '.join(missing)}"
        )
    result = frame.copy()
    result["interval_start_utc"] = _validate_utc(
        result["interval_start_utc"], f"{table_name}.interval_start_utc"
    )
    if "interval_end_utc" in result:
        result["interval_end_utc"] = _validate_utc(
            result["interval_end_utc"], f"{table_name}.interval_end_utc"
        )
        if (
            result["interval_end_utc"] <= result["interval_start_utc"]
        ).any():
            raise ContractError(f"{table_name} has non-positive intervals")
    unique = list(unique_columns)
    if result.duplicated(unique).any():
        raise ContractError(f"{table_name} has duplicate market/timestamp rows")
    if expected_market is not None:
        actual = set(result["market"].dropna().astype(str))
        if actual != {expected_market}:
            raise ContractError(
                f"{table_name} geography mismatch: expected "
                f"{expected_market}, got {sorted(actual)}"
            )
    required_values = set(non_nullable or required_set)
    if result[list(required_values)].isna().any().any():
        missing_columns = result[list(required_values)].columns[
            result[list(required_values)].isna().any()
        ].tolist()
        raise ContractError(
            f"{table_name} has missing required values in "
            f"{', '.join(sorted(missing_columns))}"
        )
    if "record_status" in result:
        statuses = set(result["record_status"].astype(str).str.lower())
        if len(statuses) > 1:
            raise ContractError(
                f"{table_name} mixes record statuses: {sorted(statuses)}"
            )
    for column in numeric_columns:
        numeric = pd.to_numeric(result[column], errors="raise")
        present = numeric.dropna()
        if not np.isfinite(present.to_numpy()).all():
            raise ContractError(f"{table_name}.{column} is not finite")
        result[column] = numeric
    for column in non_negative_columns:
        if (result[column].dropna() < 0).any():
            raise ContractError(f"{table_name}.{column} must be non-negative")
    return result.sort_values(["market", "interval_start_utc"]).reset_index(
        drop=True
    )


def validate_forecasts(frame: pd.DataFrame) -> pd.DataFrame:
    result = validate_native_table(
        frame,
        FORECAST_COLUMNS,
        table_name="forecast",
        unique_columns=(
            "market",
            "interval_start_utc",
            "forecast_target",
            "forecast_horizon_hours",
            "forecast_issue_utc",
        ),
    )
    result["forecast_issue_utc"] = _validate_utc(
        result["forecast_issue_utc"], "forecast.forecast_issue_utc"
    )
    result["forecast_vintage_utc"] = _validate_utc(
        result["forecast_vintage_utc"], "forecast.forecast_vintage_utc"
    )
    values = pd.to_numeric(result["forecast_value_mw"], errors="raise")
    if not np.isfinite(values.to_numpy()).all():
        raise ContractError("forecast values must be finite")
    result["forecast_value_mw"] = values
    if (
        result["forecast_issue_utc"] > result["interval_start_utc"]
    ).any():
        raise ContractError("forecast issue time occurs after its target")
    if (
        result["forecast_vintage_utc"] > result["forecast_issue_utc"]
    ).any():
        raise ContractError("forecast vintage occurs after issue time")
    horizon = pd.to_numeric(
        result["forecast_horizon_hours"], errors="raise"
    )
    if not horizon.isin({1, 3}).all():
        raise ContractError("forecast horizon must be one or three hours")
    actual_horizon = (
        result["interval_start_utc"] - result["forecast_issue_utc"]
    ).dt.total_seconds() / 3600.0
    if not (actual_horizon.sub(horizon).abs() < 1e-9).all():
        raise ContractError(
            "forecast horizon does not match target minus issue time"
        )
    forbidden = {
        "realized_future",
        "future_rt",
        "realized_load",
        "realized_future_rt",
        "realized_future_load",
    }
    capabilities = set(
        result["forecast_capability"].astype(str).str.strip().str.lower()
    )
    if capabilities & forbidden:
        raise ContractError("realized future values cannot be used as forecasts")
    return result


def validate_hourly_index(
    frame: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    table_name: str,
) -> pd.DataFrame:
    expected = pd.date_range(start, end, freq="1h", inclusive="left")
    actual = pd.DatetimeIndex(
        pd.to_datetime(frame["interval_start_utc"], utc=True, errors="raise")
    )
    if not actual.is_unique:
        raise ContractError(f"{table_name} has duplicate hourly timestamps")
    if not actual.equals(expected):
        missing = expected.difference(actual)
        extra = actual.difference(expected)
        raise ContractError(
            f"{table_name} incomplete common calendar: "
            f"missing={len(missing)}, extra={len(extra)}, "
            f"expected={len(expected)}, actual={len(actual)}"
        )
    if "interval_end_utc" not in frame:
        raise ContractError(f"{table_name} lacks hourly interval ends")
    ends = pd.DatetimeIndex(
        pd.to_datetime(frame["interval_end_utc"], utc=True, errors="raise")
    )
    expected_ends = actual + pd.Timedelta(hours=1)
    if not ends.equals(expected_ends):
        raise ContractError(f"{table_name} interval ends are not exact hours")
    return frame
