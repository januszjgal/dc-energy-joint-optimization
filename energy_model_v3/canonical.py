"""Canonicalization helpers shared by source-specific adapters."""

from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .contract import ContractError


def canonical_interval(
    values: pd.Series,
    *,
    timezone_name: str,
    semantics: str,
    cadence: pd.Timedelta,
    ambiguous: pd.Series | str = "raise",
) -> pd.DataFrame:
    parsed = pd.to_datetime(values, errors="raise")
    if getattr(parsed.dt, "tz", None) is None:
        parsed = parsed.dt.tz_localize(
            ZoneInfo(timezone_name),
            ambiguous=ambiguous,
            nonexistent="raise",
        )
    utc = parsed.dt.tz_convert("UTC")
    if semantics == "interval_beginning":
        start = utc
        end = utc + cadence
    elif semantics == "interval_ending":
        start = utc - cadence
        end = utc
    else:
        raise ContractError(f"unsupported interval semantics {semantics!r}")
    return pd.DataFrame(
        {"interval_start_utc": start, "interval_end_utc": end}
    )


def derive_net_load(
    frame: pd.DataFrame,
    *,
    method: str,
) -> pd.DataFrame:
    result = frame.copy()
    if method == "published":
        if "published_net_load_mw" not in result:
            raise ContractError("published net load column is missing")
        result["net_load_mw"] = pd.to_numeric(
            result["published_net_load_mw"], errors="raise"
        )
        result["net_load_method"] = "published_native"
    elif method == "gross-minus-wind-solar":
        required = {"gross_demand_mw", "wind_mw", "solar_mw"}
        missing = required - set(result.columns)
        if missing:
            raise ContractError(
                f"net load derivation missing {sorted(missing)}"
            )
        result["net_load_mw"] = (
            pd.to_numeric(result["gross_demand_mw"], errors="raise")
            - pd.to_numeric(result["wind_mw"], errors="raise")
            - pd.to_numeric(result["solar_mw"], errors="raise")
        )
        result["net_load_method"] = "derived:gross-wind-solar"
    elif method == "pal-reflects-btm":
        if "gross_demand_mw" not in result:
            raise ContractError("NYISO PAL load column is missing")
        result["net_load_mw"] = pd.to_numeric(
            result["gross_demand_mw"], errors="raise"
        )
        result["net_load_method"] = (
            "native:PAL_already_reflects_BTM_solar_no_double_subtraction"
        )
    else:
        raise ContractError(f"unsupported net load method {method!r}")
    return result


def aggregate_native_hourly(
    frame: pd.DataFrame,
    *,
    value_columns: list[str],
    native_minutes: int,
) -> pd.DataFrame:
    """Average complete native intervals; never forward-fill missing evidence."""
    if 60 % native_minutes:
        raise ContractError("native cadence must divide one hour")
    expected = 60 // native_minutes
    result = frame.copy()
    result["interval_start_utc"] = pd.to_datetime(
        result["interval_start_utc"], utc=True, errors="raise"
    )
    if result["interval_start_utc"].duplicated().any():
        raise ContractError("native input contains duplicate timestamps")
    result["hour_utc"] = result["interval_start_utc"].dt.floor("1h")
    counts = result.groupby("hour_utc").size()
    if not (counts == expected).all():
        bad = counts[counts != expected]
        raise ContractError(
            f"native-to-hourly aggregation has {len(bad)} incomplete hours; "
            "forward filling is forbidden"
        )
    numeric = result[value_columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ContractError("native input has non-finite values")
    result[value_columns] = numeric
    hourly = result.groupby("hour_utc", as_index=False)[value_columns].mean()
    hourly = hourly.rename(columns={"hour_utc": "interval_start_utc"})
    hourly["interval_end_utc"] = (
        hourly["interval_start_utc"] + pd.Timedelta(hours=1)
    )
    return hourly


def require_single_product_status(
    frame: pd.DataFrame,
    *,
    column: str = "record_status",
) -> None:
    if column not in frame:
        raise ContractError(f"missing product status column {column}")
    statuses = set(frame[column].dropna().astype(str).str.lower())
    if len(statuses) != 1:
        raise ContractError(
            f"mixed preliminary/final product status: {sorted(statuses)}"
        )
