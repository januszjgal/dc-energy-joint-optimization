"""Authoritative no-key acquisition for the six-market live v3 panel."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from .builder import MARKETS
from .calendar import StudyCalendar, make_calendar
from .contract import ContractError, validate_hourly_index

USER_AGENT = "dc-energy-joint-optimization/energy-model-v3"
EIA_BA = {
    "CAISO_NP15": "CISO",
    "ERCOT_LZ_NORTH": "ERCO",
    "NYISO_NYC_J": "NYIS",
    "MISO_MINN_HUB": "MISO",
    "SPP_NORTH_HUB": "SWPP",
    "ISONE_NEMA": "ISNE",
}
PRICE_LOCATIONS = {
    "CAISO_NP15": "TH_NP15_GEN-APND",
    "ERCOT_LZ_NORTH": "LZ_NORTH",
    "NYISO_NYC_J": "N.Y.C. PTID 61761",
    "MISO_MINN_HUB": "MINN.HUB",
    "SPP_NORTH_HUB": "SPPNORTH_HUB",
    "ISONE_NEMA": ".Z.NEMASSBOST location 4008",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path, *, timeout: int = 180) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size:
        return {
            "url": url,
            "path": str(destination),
            "bytes": destination.stat().st_size,
            "sha256": sha256(destination),
            "retrieved_at_utc": None,
            "cache_hit": True,
        }
    partial = destination.with_suffix(destination.suffix + ".partial")
    response = None
    for attempt in range(7):
        response = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
            stream=True,
        )
        if response.status_code != 429:
            break
        response.close()
        if attempt == 6:
            raise ContractError(f"rate limit persisted for {url}")
        time.sleep(min(2**attempt, 60))
    assert response is not None
    response.raise_for_status()
    with partial.open("wb") as handle:
        for chunk in response.iter_content(1024 * 1024):
            if chunk:
                handle.write(chunk)
    if not partial.stat().st_size:
        raise ContractError(f"empty response from {url}")
    partial.replace(destination)
    return {
        "url": response.url,
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "cache_hit": False,
        "http_headers": {
            key: response.headers.get(key)
            for key in ("Content-Type", "ETag", "Last-Modified")
        },
    }


def _download_many(
    tasks: Iterable[tuple[str, Path]], *, workers: int = 12
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_download, url, destination): (url, destination)
            for url, destination in tasks
        }
        for future in as_completed(futures):
            url, destination = futures[future]
            try:
                records.append(future.result())
            except Exception as exc:
                raise ContractError(
                    f"download failed for {url} -> {destination}: {exc}"
                ) from exc
    return sorted(records, key=lambda item: str(item["path"]))


def _dates(calendar: StudyCalendar) -> pd.DatetimeIndex:
    start = pd.Timestamp(calendar.start_utc).tz_localize(None)
    end = pd.Timestamp(calendar.end_utc).tz_localize(None)
    return pd.date_range(start, end, freq="1D", inclusive="left")


def _months(calendar: StudyCalendar) -> pd.DatetimeIndex:
    start = pd.Timestamp(calendar.start_utc).tz_localize(None)
    end = pd.Timestamp(calendar.end_utc).tz_localize(None)
    return pd.date_range(start, end, freq="MS", inclusive="left")


def _local_source_dates(calendar: StudyCalendar) -> pd.DatetimeIndex:
    dates = _dates(calendar)
    return dates.insert(0, dates[0] - pd.Timedelta(days=1))


def _local_source_months(calendar: StudyCalendar) -> pd.DatetimeIndex:
    months = _months(calendar)
    return months.insert(0, months[0] - pd.offsets.MonthBegin(1))


def _utc_from_local(
    local: pd.Series,
    timezone_name: str,
    *,
    repeated: pd.Series | None = None,
) -> pd.Series:
    naive = pd.to_datetime(local, errors="raise")
    if repeated is None:
        occurrence = naive.groupby(naive).cumcount()
        repeated = occurrence.astype(bool)
    zone = ZoneInfo(timezone_name)
    values = [
        timestamp.to_pydatetime().replace(
            tzinfo=zone,
            fold=1 if bool(is_repeated) else 0,
        )
        for timestamp, is_repeated in zip(naive, repeated, strict=True)
    ]
    return pd.Series(pd.to_datetime(values, utc=True), index=local.index)


def _utc_from_eastern_abbreviation(
    local: pd.Series, abbreviation: pd.Series
) -> pd.Series:
    naive = pd.to_datetime(local, errors="raise")
    labels = abbreviation.astype(str).str.upper()
    invalid = ~labels.isin({"EST", "EDT"})
    if invalid.any():
        raise ContractError(
            f"unsupported Eastern timezone labels: {sorted(labels[invalid].unique())}"
        )
    offsets = labels.map({"EDT": 4, "EST": 5})
    return pd.Series(
        pd.to_datetime(naive + pd.to_timedelta(offsets, unit="h"), utc=True),
        index=local.index,
    )


def extract_eia_physical(
    archive: Path,
    *,
    calendar: StudyCalendar | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Select only the required UTC demand/wind/solar series from EBA.zip."""
    calendar = calendar or make_calendar()
    start = pd.Timestamp(calendar.start_utc)
    end = pd.Timestamp(calendar.end_utc)
    wanted = {
        f"EBA.{ba}-ALL.{suffix}.H": (market, column)
        for market, ba in EIA_BA.items()
        for suffix, column in (
            ("D", "gross_demand_mw"),
            ("NG.WND", "wind_mw"),
            ("NG.SUN", "solar_mw"),
        )
    }
    selected: dict[str, dict[str, pd.Series]] = {market: {} for market in MARKETS}
    series_meta: dict[str, Any] = {}
    with zipfile.ZipFile(archive) as zipped, zipped.open("EBA.txt") as handle:
        for raw in handle:
            marker = raw.find(b'","name"')
            if marker < 0:
                continue
            series_id = raw[len(b'{"series_id":"') : marker].decode()
            target = wanted.get(series_id)
            if target is None:
                continue
            payload = json.loads(raw)
            market, column = target
            pairs = payload["data"]
            values = {
                pd.to_datetime(timestamp, format="%Y%m%dT%H", utc=True): (
                    np.nan if value is None else float(value)
                )
                for timestamp, value in pairs
            }
            selected[market][column] = pd.Series(values, dtype=float)
            series_meta[series_id] = {
                key: payload.get(key)
                for key in ("name", "units", "start", "end", "last_updated")
            }
    expected = pd.date_range(start, end, freq="1h", inclusive="left")
    frames: dict[str, pd.DataFrame] = {}
    negative_renewable_normalization: dict[str, dict[str, Any]] = {}
    for market in MARKETS:
        if set(selected[market]) != {"gross_demand_mw", "wind_mw", "solar_mw"}:
            raise ContractError(f"EIA bulk archive lacks required {market} series")
        frame = pd.DataFrame(selected[market]).reindex(expected)
        missing = frame.isna().sum()
        if missing.any() and market in {"MISO_MINN_HUB", "ISONE_NEMA"}:
            raise ContractError(
                f"EIA {market} has missing primary hours: {missing.to_dict()}"
            )
        negative_renewable_normalization[market] = {}
        for column in ("wind_mw", "solar_mw"):
            negative = frame[column] < 0
            negative_renewable_normalization[market][column] = {
                "count": int(negative.sum()),
                "raw_min_mw": (
                    float(frame.loc[negative, column].min())
                    if negative.any()
                    else None
                ),
                "rule": "clip_negative_reported_generation_to_zero",
            }
            frame.loc[negative, column] = 0.0
        frame.index.name = "interval_start_utc"
        frame = frame.reset_index()
        frame["interval_end_utc"] = (
            frame["interval_start_utc"] + pd.Timedelta(hours=1)
        )
        frame["market"] = market
        frame["net_load_mw"] = (
            frame["gross_demand_mw"] - frame["wind_mw"] - frame["solar_mw"]
        )
        frame["net_load_method"] = "derived:gross-wind-solar"
        frame["source_quality_flags"] = (
            "EIA_EBA_same_balancing_authority_fallback"
        )
        frames[market] = frame
    return frames, {
        "source": "https://api.eia.gov/bulk/EBA.zip",
        "raw_path": str(archive),
        "bytes": archive.stat().st_size,
        "sha256": sha256(archive),
        "selected_series": series_meta,
        "known_missing_hours": {
            market: {
                column: int(frame[column].isna().sum())
                for column in ("gross_demand_mw", "wind_mw", "solar_mw")
            }
            for market, frame in frames.items()
        },
        "negative_renewable_normalization": negative_renewable_normalization,
        "interpolation": False,
    }


def _aggregate_complete_5min(
    frame: pd.DataFrame,
    *,
    value_columns: list[str],
    market: str,
) -> pd.DataFrame:
    ordered = frame.sort_values("interval_start_utc").copy()
    ordered["interval_start_utc"] = pd.to_datetime(
        ordered["interval_start_utc"], utc=True
    )
    if ordered["interval_start_utc"].duplicated().any():
        raise ContractError(f"{market} native physical timestamps are duplicated")
    ordered["hour"] = ordered["interval_start_utc"].dt.floor("1h")
    counts = ordered.groupby("hour").size()
    bad = counts[counts != 12]
    if len(bad):
        raise ContractError(
            f"{market} has {len(bad)} incomplete native 5-minute hours"
        )
    result = ordered.groupby("hour", as_index=False)[value_columns].mean()
    return result.rename(columns={"hour": "interval_start_utc"})


def _time_weighted_hourly(
    frame: pd.DataFrame,
    *,
    value_columns: list[str],
    market: str,
    maximum_gap: pd.Timedelta = pd.Timedelta(minutes=10),
) -> pd.DataFrame:
    ordered = frame.sort_values("interval_start_utc").copy()
    ordered["interval_start_utc"] = pd.to_datetime(
        ordered["interval_start_utc"], utc=True
    )
    ordered = ordered.drop_duplicates("interval_start_utc", keep="last")
    starts = pd.DatetimeIndex(ordered["interval_start_utc"])
    ends = starts[1:]
    starts = starts[:-1]
    values = ordered.iloc[:-1].reset_index(drop=True)
    buckets: dict[pd.Timestamp, dict[str, float]] = {}
    for position, (start, end) in enumerate(zip(starts, ends, strict=True)):
        gap = end - start
        if gap <= pd.Timedelta(0) or gap > maximum_gap:
            raise ContractError(f"{market} native execution gap {gap} at {start}")
        cursor = start
        while cursor < end:
            hour = cursor.floor("1h")
            boundary = min(end, hour + pd.Timedelta(hours=1))
            duration = (boundary - cursor).total_seconds() / 3600.0
            bucket = buckets.setdefault(hour, {"duration": 0.0})
            bucket["duration"] += duration
            for column in value_columns:
                bucket[column] = bucket.get(column, 0.0) + (
                    float(values.loc[position, column]) * duration
                )
            cursor = boundary
    rows = []
    for hour, bucket in sorted(buckets.items()):
        if bucket["duration"] < 0.999:
            continue
        row = {"interval_start_utc": hour}
        for column in value_columns:
            row[column] = bucket[column] / bucket["duration"]
        rows.append(row)
    return pd.DataFrame(rows)


def acquire_caiso_physical(
    raw_root: Path, *, calendar: StudyCalendar
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    tasks = []
    for date in _dates(calendar):
        for product in ("netdemand", "fuelsource"):
            tasks.append(
                (
                    "https://www.caiso.com/outlook/history/"
                    f"{date:%Y%m%d}/{product}.csv",
                    raw_root / "caiso" / "physical" / f"{date:%Y%m%d}_{product}.csv",
                )
            )
    metadata = _download_many(tasks, workers=20)
    pieces = []
    for date in _dates(calendar):
        demand = pd.read_csv(
            raw_root / "caiso" / "physical" / f"{date:%Y%m%d}_netdemand.csv"
        )
        fuel = pd.read_csv(
            raw_root / "caiso" / "physical" / f"{date:%Y%m%d}_fuelsource.csv"
        )
        if len(demand) != len(fuel) or not demand["Time"].equals(fuel["Time"]):
            raise ContractError(f"CAISO outlook physical files differ on {date:%F}")
        local = pd.to_datetime(
            date.strftime("%Y-%m-%d") + " " + demand["Time"].astype(str)
        )
        pieces.append(
            pd.DataFrame(
                {
                    "local": local,
                    "gross_demand_mw": pd.to_numeric(
                        demand["Current demand"], errors="raise"
                    ),
                    "published_net_load_mw": pd.to_numeric(
                        demand["Net demand"], errors="raise"
                    ),
                    "wind_mw": pd.to_numeric(fuel["Wind"], errors="raise"),
                    "solar_mw": pd.to_numeric(fuel["Solar"], errors="raise"),
                }
            )
        )
    native = pd.concat(pieces, ignore_index=True)
    native["interval_start_utc"] = _utc_from_local(
        native["local"], "America/Los_Angeles"
    )
    hourly = _aggregate_complete_5min(
        native,
        value_columns=[
            "gross_demand_mw",
            "published_net_load_mw",
            "wind_mw",
            "solar_mw",
        ],
        market="CAISO_NP15",
    )
    hourly["net_load_mw"] = hourly["published_net_load_mw"]
    hourly["net_load_method"] = "published_native_CAISO_net_demand"
    hourly["source_quality_flags"] = (
        "CAISO_Todays_Outlook_history_complete_5min_mean"
    )
    return _physical_frame("CAISO_NP15", hourly), metadata


def acquire_nyiso_physical(
    raw_root: Path,
    *,
    calendar: StudyCalendar,
    eia_context: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    tasks = []
    for month in _local_source_months(calendar):
        products = (
            ("palIntegrated", "palIntegrated"),
            ("btmactualforecast", "BTMEstimatedActual"),
        )
        for directory, filename in products:
            tasks.append(
                (
                    f"https://mis.nyiso.com/public/csv/{directory}/"
                    f"{month:%Y%m}01{filename}_csv.zip",
                    raw_root / "nyiso" / "physical"
                    / f"{month:%Y%m}_{directory}.zip",
                )
            )
    metadata = _download_many(tasks, workers=12)
    pal_pieces = []
    solar_pieces = []
    for month in _local_source_months(calendar):
        pal_path = (
            raw_root / "nyiso" / "physical" / f"{month:%Y%m}_palIntegrated.zip"
        )
        with zipfile.ZipFile(pal_path) as zipped:
            for name in zipped.namelist():
                if name.lower().endswith(".csv"):
                    frame = pd.read_csv(zipped.open(name))
                    pal_pieces.append(frame.loc[frame["PTID"] == 61761])
        solar_path = (
            raw_root
            / "nyiso"
            / "physical"
            / f"{month:%Y%m}_btmactualforecast.zip"
        )
        with zipfile.ZipFile(solar_path) as zipped:
            for name in zipped.namelist():
                if name.lower().endswith(".csv"):
                    frame = pd.read_csv(zipped.open(name))
                    frame.columns = frame.columns.str.strip()
                    solar_pieces.append(
                        frame.loc[frame["Zone Name"].astype(str) == "SYSTEM"]
                    )
    pal = pd.concat(pal_pieces, ignore_index=True)
    pal["interval_start_utc"] = _utc_from_eastern_abbreviation(
        pal["Time Stamp"], pal["Time Zone"]
    )
    pal = pal.rename(columns={"Integrated Load": "gross_demand_mw"})[
        ["interval_start_utc", "gross_demand_mw"]
    ]
    solar = pd.concat(solar_pieces, ignore_index=True)
    timestamp_column = (
        "Time Stamp" if "Time Stamp" in solar.columns else "Timestamp"
    )
    solar["interval_start_utc"] = _utc_from_eastern_abbreviation(
        solar[timestamp_column], solar["Time Zone"]
    )
    solar_value = (
        "MW Value" if "MW Value" in solar.columns else "MW"
    )
    solar = solar.rename(columns={solar_value: "solar_mw"})[
        ["interval_start_utc", "solar_mw"]
    ]
    if eia_context is None:
        raise ContractError("NYISO physical build requires EIA NYCA wind context")
    renewable_context = eia_context[
        ["interval_start_utc", "wind_mw", "solar_mw"]
    ].copy()
    result = pal.merge(
        renewable_context, on="interval_start_utc", validate="one_to_one"
    )
    result["net_load_mw"] = result["gross_demand_mw"]
    result["net_load_method"] = (
        "native:NYISO_PAL_reflects_BTM_solar_no_double_subtraction"
    )
    result["source_quality_flags"] = (
        "Zone_J_hourly_integrated_PAL_with_EIA_NYCA_wind_solar_context"
    )
    metadata.append(
        {
            "source_role": "diagnostic_not_primary",
            "product": "NYISO BTMEstimatedActual",
            "candidate_hour_rows": int(
                solar["interval_start_utc"].between(
                    pd.Timestamp(calendar.start_utc),
                    pd.Timestamp(calendar.end_utc),
                    inclusive="left",
                ).sum()
            ),
            "reason": (
                "operator BTM solar lacks a complete locked UTC calendar; "
                "same-BA EIA solar is used explicitly without interpolation"
            ),
        }
    )
    return _physical_frame("NYISO_NYC_J", result), metadata


def acquire_spp_physical(
    raw_root: Path, *, calendar: StudyCalendar
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    tasks = [
        (
            "https://portal.spp.org/file-browser-api/download/"
            "generation-mix-historical?path=/SPP/GenMix365_SPP.csv",
            raw_root / "spp" / "GenMix365_SPP.csv",
        )
    ]
    metadata = _download_many(tasks, workers=1)
    native = pd.read_csv(tasks[0][1])
    native["interval_start_utc"] = pd.to_datetime(
        native["GMT MKT Interval"], utc=True, errors="raise"
    )
    native["wind_mw"] = (
        pd.to_numeric(native["Wind Market"], errors="raise")
        + pd.to_numeric(native["Wind Self"], errors="raise")
    )
    native["solar_mw"] = (
        pd.to_numeric(native["Solar Market"], errors="raise")
        + pd.to_numeric(native["Solar Self"], errors="raise")
    )
    native["gross_demand_mw"] = pd.to_numeric(native["Load"], errors="raise")
    start = pd.Timestamp(calendar.start_utc)
    end = pd.Timestamp(calendar.end_utc)
    native = native.loc[
        (native["interval_start_utc"] >= start)
        & (native["interval_start_utc"] < end)
    ]
    hourly = _aggregate_complete_5min(
        native,
        value_columns=["gross_demand_mw", "wind_mw", "solar_mw"],
        market="SPP_NORTH_HUB",
    )
    hourly["net_load_mw"] = (
        hourly["gross_demand_mw"] - hourly["wind_mw"] - hourly["solar_mw"]
    )
    hourly["net_load_method"] = "derived:gross-wind-solar"
    hourly["source_quality_flags"] = (
        "SPP_GenMix365_snapshot_complete_5min_mean"
    )
    return _physical_frame("SPP_NORTH_HUB", hourly), metadata


def _load_ercot_native_load(
    raw_root: Path, *, calendar: StudyCalendar
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    urls = {
        2025: "https://www.ercot.com/files/docs/2025/02/11/Native_Load_2025.zip",
        2026: "https://www.ercot.com/files/docs/2026/02/10/Native_Load_2026.zip",
    }
    tasks = [
        (url, raw_root / "ercot" / f"Native_Load_{year}.zip")
        for year, url in urls.items()
    ]
    metadata = _download_many(tasks, workers=2)
    pieces = []
    for _, path in tasks:
        with zipfile.ZipFile(path) as zipped:
            member = zipped.namelist()[0]
            frame = pd.read_excel(io.BytesIO(zipped.read(member)))
            pieces.append(frame)
    source = pd.concat(pieces, ignore_index=True)
    hour_text = source["Hour Ending"].astype(str)
    parts = hour_text.str.extract(
        r"(?P<date>\d{2}/\d{2}/\d{4})\s+(?P<hour>\d{1,2}):"
    )
    if parts.isna().any().any():
        raise ContractError("ERCOT Native Load has unparseable Hour Ending values")
    local_end = pd.to_datetime(parts["date"], errors="raise") + pd.to_timedelta(
        pd.to_numeric(parts["hour"], errors="raise"), unit="h"
    )
    local_start = local_end - pd.Timedelta(hours=1)
    repeated = local_start.groupby(local_start).cumcount().astype(bool)
    source["interval_start_utc"] = _utc_from_local(
        local_start, "America/Chicago", repeated=repeated
    )
    source = source.rename(columns={"ERCOT": "gross_demand_mw"})
    start = pd.Timestamp(calendar.start_utc)
    end = pd.Timestamp(calendar.end_utc)
    source = source.loc[
        (source["interval_start_utc"] >= start)
        & (source["interval_start_utc"] < end),
        ["interval_start_utc", "gross_demand_mw"],
    ]
    return source.sort_values("interval_start_utc"), metadata


def _ercot_sced_documents(
    calendar: StudyCalendar,
    *,
    start: pd.Timestamp | None = None,
    require_all: bool = True,
) -> list[dict[str, Any]]:
    response = requests.get(
        "https://www.ercot.com/misapp/servlets/IceDocListJsonWS",
        params={"reportTypeId": 13052},
        headers={"User-Agent": USER_AGENT},
        timeout=60,
    )
    response.raise_for_status()
    documents = [
        item["Document"]
        for item in response.json()["ListDocsByRptTypeRes"]["DocumentList"]
        if item["Document"]["Extension"].lower() == "zip"
        and item["Document"]["FriendlyName"] == "60_Day_SCED_Disclosure"
    ]
    by_publish_date: dict[pd.Timestamp, dict[str, Any]] = {}
    for document in documents:
        date = pd.Timestamp(document["PublishDate"]).tz_localize(None).normalize()
        previous = by_publish_date.get(date)
        if previous is None or int(document["ContentSize"]) > int(
            previous["ContentSize"]
        ):
            by_publish_date[date] = document
    selected = []
    for operating_date in _dates(calendar):
        if start is not None and operating_date < pd.Timestamp(start):
            continue
        publish_date = operating_date + pd.Timedelta(days=60)
        document = by_publish_date.get(publish_date.normalize())
        if document is None:
            if require_all:
                raise ContractError(
                    "ERCOT report13052 lacks disclosure for "
                    f"{operating_date:%Y-%m-%d}"
                )
            continue
        selected.append({"operating_date": operating_date, **document})
    return selected


def _time_weighted_sced_hourly(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        frame.groupby(["interval_start_utc", "Resource Type"], as_index=False)[
            "Telemetered Net Output"
        ]
        .sum()
        .pivot(
            index="interval_start_utc",
            columns="Resource Type",
            values="Telemetered Net Output",
        )
        .fillna(0.0)
        .sort_index()
    )
    for resource_type in ("WGR", "PVGR"):
        if resource_type not in grouped:
            grouped[resource_type] = 0.0
    first_anchor = grouped.index[0].floor("1h")
    if grouped.index[0] > first_anchor:
        grouped.loc[first_anchor] = grouped.iloc[0]
        grouped = grouped.sort_index()
    rows: dict[pd.Timestamp, dict[str, float]] = {}
    index = pd.DatetimeIndex(grouped.index)
    if len(index) < 2:
        raise ContractError("ERCOT SCED day has insufficient executions")
    starts = index
    ends = index[1:].append(
        pd.DatetimeIndex([index[-1].ceil("1h")])
    )
    for position, (start, end) in enumerate(zip(starts, ends, strict=True)):
        if end <= start or end - start > pd.Timedelta(hours=1):
            raise ContractError(
                f"ERCOT SCED execution gap is invalid at {start}: {end - start}"
            )
        cursor = start
        while cursor < end:
            hour = cursor.floor("1h")
            boundary = min(end, hour + pd.Timedelta(hours=1))
            duration_hours = (boundary - cursor).total_seconds() / 3600.0
            bucket = rows.setdefault(
                hour, {"wind_energy": 0.0, "solar_energy": 0.0, "duration": 0.0}
            )
            bucket["wind_energy"] += grouped.iloc[position]["WGR"] * duration_hours
            bucket["solar_energy"] += grouped.iloc[position]["PVGR"] * duration_hours
            bucket["duration"] += duration_hours
            cursor = boundary
    result = pd.DataFrame(
        [
            {
                "interval_start_utc": hour,
                "wind_mw": values["wind_energy"] / values["duration"],
                "solar_mw": values["solar_energy"] / values["duration"],
                "covered_fraction": values["duration"],
            }
            for hour, values in sorted(rows.items())
        ]
    )
    if (result["covered_fraction"] < 0.98).any():
        raise ContractError("ERCOT SCED hourly time coverage is below 98%")
    return result.drop(columns="covered_fraction")


def _parse_ercot_sced_tasks(
    tasks: list[tuple[str, Path]],
    documents: list[dict[str, Any]],
) -> pd.DataFrame:
    renewable_pieces = []
    required = {
        "SCED Time Stamp",
        "Repeated Hour Flag",
        "Resource Name",
        "Resource Type",
        "Telemetered Net Output",
    }
    for (_, path), document in zip(tasks, documents, strict=True):
        hourly_cache = path.with_suffix(".hourly.csv")
        if hourly_cache.exists():
            cached = pd.read_csv(hourly_cache)
            cached["interval_start_utc"] = pd.to_datetime(
                cached["interval_start_utc"], utc=True
            )
            renewable_pieces.append(cached)
            continue
        with zipfile.ZipFile(path) as zipped:
            members = [
                name
                for name in zipped.namelist()
                if "SCED_Gen_Resource_Data" in name
            ]
            if len(members) != 1:
                raise ContractError(f"ERCOT SCED resource member missing in {path}")
            source = pd.read_csv(
                zipped.open(members[0]),
                usecols=lambda name: str(name).strip() in required,
                low_memory=False,
            )
        source.columns = source.columns.str.strip()
        if set(source.columns) != required:
            raise ContractError(
                f"ERCOT SCED required columns differ in {path}: "
                f"{sorted(source.columns)}"
            )
        source["Resource Type"] = source["Resource Type"].astype(str).str.strip()
        source["Resource Type"] = source["Resource Type"].replace({"WIND": "WGR"})
        source = source.loc[source["Resource Type"].isin(["WGR", "PVGR"])].copy()
        local = pd.to_datetime(source["SCED Time Stamp"], errors="raise")
        repeated = source["Repeated Hour Flag"].astype(str).str.upper().eq("Y")
        source["interval_start_utc"] = _utc_from_local(
            local, "America/Chicago", repeated=repeated
        )
        source["Telemetered Net Output"] = pd.to_numeric(
            source["Telemetered Net Output"], errors="raise"
        )
        hourly = _time_weighted_sced_hourly(source)
        hourly.to_csv(hourly_cache, index=False)
        renewable_pieces.append(hourly)
    return (
        pd.concat(renewable_pieces, ignore_index=True)
        .drop_duplicates("interval_start_utc")
        .sort_values("interval_start_utc")
    )


def _ercot_annual_renewables(
    raw_root: Path,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    document = _ercot_document(
        13424, "ERCOT_2025_Hourly_WindSolar_Output"
    )
    path = (
        raw_root
        / "ercot"
        / f"ERCOT_2025_Hourly_WindSolar_Output_{document['DocID']}.xlsx"
    )
    metadata = [
        _download(
            "https://www.ercot.com/misdownload/servlets/mirDownload?"
            f"doclookupId={document['DocID']}",
            path,
        )
    ]
    wind = pd.read_excel(path, sheet_name="Wind Data")
    solar = pd.read_excel(path, sheet_name="Solar Data")
    local_end = pd.to_datetime(wind["Time (Hour-Ending)"], errors="raise")
    repeated = local_end.groupby(local_end).cumcount().astype(bool)
    utc = (
        _utc_from_local(local_end, "America/Chicago", repeated=repeated)
        - pd.Timedelta(hours=1)
    )
    result = pd.DataFrame(
        {
            "interval_start_utc": utc,
            "wind_mw": pd.to_numeric(wind["ERCOT.WIND.GEN"], errors="raise"),
            "solar_mw": pd.to_numeric(solar["ERCOT.PVGR.GEN"], errors="raise"),
        }
    )
    metadata[0]["ercot_document"] = document
    return result, metadata


def _ercot_overlap_diagnostics(
    annual: pd.DataFrame,
    sced: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any]:
    overlap = annual.merge(
        sced,
        on="interval_start_utc",
        suffixes=("_annual13424", "_sced13052"),
        validate="one_to_one",
    )
    overlap = overlap.loc[
        (overlap["interval_start_utc"] >= start)
        & (overlap["interval_start_utc"] < end)
    ].copy()
    expected_index = pd.date_range(start, end, freq="1h", inclusive="left")
    missing = expected_index.difference(
        pd.DatetimeIndex(overlap["interval_start_utc"])
    )
    expected_rows = len(expected_index)
    metrics: dict[str, dict[str, float]] = {}
    for resource in ("wind_mw", "solar_mw"):
        annual_values = overlap[f"{resource}_annual13424"]
        sced_values = overlap[f"{resource}_sced13052"]
        errors = sced_values - annual_values
        metrics[resource] = {
            "annual13424_mean_mw": float(annual_values.mean()),
            "sced13052_mean_mw": float(sced_values.mean()),
            "bias_mw_sced_minus_annual": float(errors.mean()),
            "mae_mw": float(errors.abs().mean()),
            "rmse_mw": float(np.sqrt(np.mean(np.square(errors)))),
            "correlation": float(annual_values.corr(sced_values)),
        }
    return {
        "period_start_utc": start.isoformat(),
        "period_end_exclusive_utc": end.isoformat(),
        "rows": len(overlap),
        "expected_rows": expected_rows,
        "complete_hourly_overlap": len(missing) == 0,
        "missing_hours": len(missing),
        "missing_hour_examples_utc": [value.isoformat() for value in missing[:24]],
        "comparison": (
            "report13052 time-weighted SCED WGR/PVGR versus report13424 "
            "published hourly wind/solar"
        ),
        "metrics": metrics,
    }


def acquire_ercot_physical(
    raw_root: Path, *, calendar: StudyCalendar
) -> tuple[pd.DataFrame, dict[str, Any]]:
    load, load_metadata = _load_ercot_native_load(
        raw_root, calendar=calendar
    )
    annual, annual_metadata = _ercot_annual_renewables(raw_root)
    documents = _ercot_sced_documents(calendar, require_all=False)
    primary_documents = _ercot_sced_documents(
        calendar, start=pd.Timestamp("2026-01-01"), require_all=True
    )
    if {
        document["operating_date"] for document in primary_documents
    } - {document["operating_date"] for document in documents}:
        raise ContractError("ERCOT primary SCED document validation is inconsistent")
    tasks = [
        (
            "https://www.ercot.com/misdownload/servlets/mirDownload?"
            f"doclookupId={document['DocID']}",
            raw_root / "ercot" / "sced"
            / f"{document['operating_date']:%Y%m%d}_{document['DocID']}.zip",
        )
        for document in documents
    ]
    download_metadata = _download_many(tasks, workers=12)
    sced = _parse_ercot_sced_tasks(tasks, documents)
    sced_primary_start = pd.Timestamp(
        "2026-01-01", tz="America/Chicago"
    ).tz_convert("UTC")
    overlap = _ercot_overlap_diagnostics(
        annual,
        sced,
        start=max(pd.Timestamp(calendar.start_utc), sced["interval_start_utc"].min()),
        end=sced_primary_start,
    )
    renewable = pd.concat(
        [
            annual.loc[
                annual["interval_start_utc"] < sced_primary_start
            ],
            sced.loc[sced["interval_start_utc"] >= sced_primary_start],
        ],
        ignore_index=True,
    ).sort_values("interval_start_utc")
    result = load.merge(renewable, on="interval_start_utc", validate="one_to_one")
    result["net_load_mw"] = (
        result["gross_demand_mw"] - result["wind_mw"] - result["solar_mw"]
    )
    result["net_load_method"] = "derived:gross-wind-solar"
    result["source_quality_flags"] = (
        "ERCOT_Native_Load_plus_report13424_then_report13052_time_weighted_SCED"
    )
    return _physical_frame("ERCOT_LZ_NORTH", result), {
        "native_load": load_metadata,
        "annual_renewables": annual_metadata,
        "sced_downloads": download_metadata,
        "sced_documents": [
            {
                **document,
                "operating_date": document["operating_date"].date().isoformat(),
            }
            for document in documents
        ],
        "hourly_weighting": "piecewise_constant_between_irregular_SCED_executions",
        "overlap_validation": overlap,
    }


def _physical_frame(market: str, source: pd.DataFrame) -> pd.DataFrame:
    result = source.copy()
    result["interval_start_utc"] = pd.to_datetime(
        result["interval_start_utc"], utc=True
    )
    result["interval_end_utc"] = (
        result["interval_start_utc"] + pd.Timedelta(hours=1)
    )
    result["market"] = market
    columns = [
        "interval_start_utc",
        "interval_end_utc",
        "market",
        "gross_demand_mw",
        "wind_mw",
        "solar_mw",
        "net_load_mw",
        "net_load_method",
        "source_quality_flags",
    ]
    return result[columns].sort_values("interval_start_utc").reset_index(drop=True)


def acquire_physical(
    raw_root: Path,
    *,
    calendar: StudyCalendar | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    calendar = calendar or make_calendar()
    eia_frames, eia_metadata = extract_eia_physical(
        raw_root / "EBA.zip", calendar=calendar
    )
    frames: dict[str, pd.DataFrame] = {}
    metadata: dict[str, Any] = {"EIA_EBA": eia_metadata}
    functions = {"ERCOT_LZ_NORTH": acquire_ercot_physical}
    for market in MARKETS:
        if market == "NYISO_NYC_J":
            frame, records = acquire_nyiso_physical(
                raw_root,
                calendar=calendar,
                eia_context=eia_frames[market],
            )
            metadata[market] = records
        elif market in functions:
            frame, records = functions[market](raw_root, calendar=calendar)
            metadata[market] = records
        else:
            frame = eia_frames[market]
            metadata[market] = {
                "source_role": "same_balancing_authority_EIA_bulk_fallback",
                "operator_current_only_history_forbidden": True,
                "fallback_reason": {
                    "CAISO_NP15": (
                        "Today's Outlook historical CSV has no authoritative UTC "
                        "field and always exposes 288 display rows on DST days"
                    ),
                    "MISO_MINN_HUB": (
                        "MISO historical physical Data Exchange requires a subscription"
                    ),
                    "SPP_NORTH_HUB": (
                        "GenMix365 snapshot has 67 incomplete 5-minute hours in "
                        "the locked candidate; primary interpolation is forbidden"
                    ),
                    "ISONE_NEMA": (
                        "ISO-NE static physical reports are not complete and "
                        "harmonized for the locked candidate"
                    ),
                }[market],
            }
        start = pd.Timestamp(calendar.start_utc)
        end = pd.Timestamp(calendar.end_utc)
        frame = frame.loc[
            (frame["interval_start_utc"] >= start)
            & (frame["interval_start_utc"] < end)
        ].reset_index(drop=True)
        validate_hourly_index(
            frame,
            start=start,
            end=end,
            table_name=f"{market}.physical",
        )
        frames[market] = frame
    return frames, metadata


def acquire_caiso_price(
    raw_root: Path, *, calendar: StudyCalendar
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    tasks = []
    for month in _months(calendar):
        next_month = month + pd.offsets.MonthBegin(1)
        chunk_start = month
        while chunk_start < next_month:
            chunk_end = min(chunk_start + pd.Timedelta(days=30), next_month)
            query = (
                "https://oasis.caiso.com/oasisapi/SingleZip?"
                "queryname=PRC_LMP&version=12&market_run_id=DAM&"
                f"startdatetime={chunk_start:%Y%m%d}T00:00-0000&"
                f"enddatetime={chunk_end:%Y%m%d}T00:00-0000&"
                "node=TH_NP15_GEN-APND&resultformat=6"
            )
            tasks.append(
                (
                    query,
                    raw_root
                    / "caiso"
                    / f"{chunk_start:%Y%m%d}_{chunk_end:%Y%m%d}.zip",
                )
            )
            chunk_start = chunk_end
    metadata = _download_many(tasks, workers=1)
    pieces = []
    for _, path in tasks:
        with zipfile.ZipFile(path) as zipped:
            names = [name for name in zipped.namelist() if name.lower().endswith(".csv")]
            if len(names) != 1:
                raise ContractError(f"unexpected CAISO archive members in {path}")
            pieces.append(pd.read_csv(zipped.open(names[0])))
    source = pd.concat(pieces, ignore_index=True)
    source = source[
        (source["NODE"] == "TH_NP15_GEN-APND")
        & (source["LMP_TYPE"] == "LMP")
        & (source["MARKET_RUN_ID"] == "DAM")
    ].copy()
    return _price_frame(
        "CAISO_NP15",
        pd.to_datetime(source["INTERVALSTARTTIME_GMT"], utc=True),
        source["MW"],
        source_flag="CAISO_OASIS_PRC_LMP_v12_DAM_final_query",
    ), metadata


def acquire_nyiso_price(
    raw_root: Path, *, calendar: StudyCalendar
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    tasks = [
        (
            "https://mis.nyiso.com/public/csv/damlbmp/"
            f"{month:%Y%m}01damlbmp_zone_csv.zip",
            raw_root / "nyiso" / f"{month:%Y%m}_damlbmp.zip",
        )
        for month in _local_source_months(calendar)
    ]
    metadata = _download_many(tasks, workers=8)
    pieces = []
    for _, path in tasks:
        with zipfile.ZipFile(path) as zipped:
            for name in zipped.namelist():
                if name.lower().endswith(".csv"):
                    frame = pd.read_csv(zipped.open(name))
                    pieces.append(frame.loc[frame["PTID"] == 61761])
    source = pd.concat(pieces, ignore_index=True).sort_values("Time Stamp")
    utc = _utc_from_local(source["Time Stamp"], "America/New_York")
    return _price_frame(
        "NYISO_NYC_J",
        utc,
        source["LBMP ($/MWHr)"],
        energy=source["LBMP ($/MWHr)"]
        - source["Marginal Cost Losses ($/MWHr)"]
        - source["Marginal Cost Congestion ($/MWHr)"],
        congestion=source["Marginal Cost Congestion ($/MWHr)"],
        loss=source["Marginal Cost Losses ($/MWHr)"],
        source_flag="NYISO_DAM_zonal_LBMP_archive",
    ), metadata


def _ercot_document(report_type: int, friendly_name: str) -> dict[str, Any]:
    response = requests.get(
        "https://www.ercot.com/misapp/servlets/IceDocListJsonWS",
        params={"reportTypeId": report_type},
        headers={"User-Agent": USER_AGENT},
        timeout=60,
    )
    response.raise_for_status()
    documents = [
        item["Document"]
        for item in response.json()["ListDocsByRptTypeRes"]["DocumentList"]
    ]
    matches = [item for item in documents if item["FriendlyName"] == friendly_name]
    if not matches:
        raise ContractError(f"ERCOT report {report_type} lacks {friendly_name}")
    return max(matches, key=lambda item: item["PublishDate"])


def acquire_ercot_price(
    raw_root: Path, *, calendar: StudyCalendar
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    tasks: list[tuple[str, Path]] = []
    records = []
    for year in sorted({date.year for date in _dates(calendar)}):
        document = _ercot_document(13060, f"DAMLZHBSPP_{year}")
        destination = (
            raw_root
            / "ercot"
            / f"DAMLZHBSPP_{year}_{document['DocID']}.zip"
        )
        tasks.append(
            (
                "https://www.ercot.com/misdownload/servlets/mirDownload?"
                f"doclookupId={document['DocID']}",
                destination,
            )
        )
        records.append(document)
    metadata = _download_many(tasks, workers=2)
    pieces = []
    for _, path in tasks:
        with zipfile.ZipFile(path) as zipped:
            member = zipped.namelist()[0]
            workbook = pd.ExcelFile(io.BytesIO(zipped.read(member)))
            for sheet in workbook.sheet_names:
                frame = pd.read_excel(workbook, sheet_name=sheet)
                pieces.append(frame.loc[frame["Settlement Point"] == "LZ_NORTH"])
    source = pd.concat(pieces, ignore_index=True)
    hour = source["Hour Ending"].astype(str).str.slice(0, 2).astype(int)
    local_start = pd.to_datetime(source["Delivery Date"]) + pd.to_timedelta(
        hour - 1, unit="h"
    )
    repeated = source["Repeated Hour Flag"].astype(str).str.upper().eq("Y")
    utc = _utc_from_local(
        local_start, "America/Chicago", repeated=repeated
    )
    frame = _price_frame(
        "ERCOT_LZ_NORTH",
        utc,
        source["Settlement Point Price"],
        source_flag="ERCOT_MIS_report13060_annual_DAM",
    )
    for item, document in zip(metadata, records, strict=True):
        item["ercot_document"] = document
    return frame, metadata


def acquire_miso_price(
    raw_root: Path, *, calendar: StudyCalendar
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    tasks = [
        (
            f"https://docs.misoenergy.org/marketreports/{date:%Y%m%d}_da_expost_lmp.csv",
            raw_root / "miso" / f"{date:%Y%m%d}_da_expost_lmp.csv",
        )
        for date in _local_source_dates(calendar)
    ]
    metadata = _download_many(tasks, workers=16)
    pieces = []
    for _, path in tasks:
        frame = pd.read_csv(path, skiprows=4)
        row = frame.loc[
            (frame["Node"] == "MINN.HUB")
            & (frame["Type"] == "Hub")
            & (frame["Value"] == "LMP")
        ]
        if len(row) != 1:
            raise ContractError(f"MISO MINN.HUB LMP row missing in {path}")
        values = row.filter(regex=r"^HE \d+$").iloc[0]
        date = pd.Timestamp(path.name[:8])
        local_start = date + pd.to_timedelta(
            np.arange(len(values)), unit="h"
        )
        utc = local_start.tz_localize("Etc/GMT+5").tz_convert("UTC")
        pieces.append(pd.DataFrame({"utc": utc, "value": values.to_numpy()}))
    source = pd.concat(pieces, ignore_index=True)
    return _price_frame(
        "MISO_MINN_HUB",
        source["utc"],
        source["value"],
        source_flag="MISO_public_daily_DA_expost_final_fixed_EST",
    ), metadata


def acquire_spp_price(
    raw_root: Path, *, calendar: StudyCalendar
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    tasks = [
        (
            "https://portal.spp.org/file-browser-api/download/"
            "da-lmp-by-settlement-location?path="
            f"/{date:%Y}/{date:%m}/By_Day/DA-LMP-SL-{date:%Y%m%d}0100.csv",
            raw_root / "spp" / f"{date:%Y%m%d}_da_lmp.csv",
        )
        for date in _local_source_dates(calendar)
    ]
    metadata = _download_many(tasks, workers=16)
    pieces = []
    for _, path in tasks:
        frame = pd.read_csv(path)
        row = frame.loc[frame["Settlement Location"] == "SPPNORTH_HUB"]
        if len(row) not in {23, 24, 25}:
            raise ContractError(
                f"SPP SPPNORTH_HUB expected 23-25 rows in {path}, got {len(row)}"
            )
        pieces.append(row)
    source = pd.concat(pieces, ignore_index=True)
    return _price_frame(
        "SPP_NORTH_HUB",
        pd.to_datetime(source["GMTIntervalEnd"], utc=True, format="mixed")
        - pd.Timedelta(hours=1),
        source["LMP"],
        energy=source["MEC"],
        congestion=source["MCC"],
        loss=source["MLC"],
        source_flag="SPP_public_DA_LMP_by_settlement_location",
    ), metadata


def acquire_isone_price(
    raw_root: Path, *, calendar: StudyCalendar
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    tasks = [
        (
            "https://www.iso-ne.com/static-transform/csv/histRpts/da-lmp/"
            f"WW_DALMP_ISO_{date:%Y%m%d}.csv",
            raw_root / "isone" / f"{date:%Y%m%d}_da_lmp.csv",
        )
        for date in _local_source_dates(calendar)
    ]
    metadata = _download_many(tasks, workers=16)
    pieces = []
    for _, path in tasks:
        frame = pd.read_csv(
            path,
            skiprows=6,
            names=[
                "record",
                "date",
                "hour_ending",
                "location_id",
                "location_name",
                "location_type",
                "lmp",
                "energy",
                "congestion",
                "loss",
            ],
        )
        row = frame.loc[
            (pd.to_numeric(frame["location_id"], errors="coerce") == 4008)
            & (frame["record"] == "D")
        ].copy()
        if len(row) not in {23, 24, 25}:
            raise ContractError(
                f"ISO-NE location4008 has {len(row)} rows in {path}"
            )
        pieces.append(row)
    source = pd.concat(pieces, ignore_index=True)
    hour_text = source["hour_ending"].astype(str).str.strip()
    hour_number = pd.to_numeric(
        hour_text.str.extract(r"(\d+)", expand=False), errors="raise"
    )
    local_start = pd.to_datetime(source["date"]) + pd.to_timedelta(
        hour_number - 1, unit="h"
    )
    utc = _utc_from_local(
        local_start,
        "America/New_York",
        repeated=hour_text.str.upper().str.endswith("X"),
    )
    return _price_frame(
        "ISONE_NEMA",
        utc,
        source["lmp"],
        energy=source["energy"],
        congestion=source["congestion"],
        loss=source["loss"],
        source_flag="ISO_NE_static_final_DA_LMP_location4008",
    ), metadata


def _price_frame(
    market: str,
    interval_start: pd.Series | pd.DatetimeIndex,
    lmp: pd.Series | np.ndarray,
    *,
    energy: pd.Series | np.ndarray | None = None,
    congestion: pd.Series | np.ndarray | None = None,
    loss: pd.Series | np.ndarray | None = None,
    source_flag: str,
) -> pd.DataFrame:
    start = pd.Series(pd.to_datetime(interval_start, utc=True)).reset_index(drop=True)
    length = len(start)
    frame = pd.DataFrame(
        {
            "interval_start_utc": start,
            "interval_end_utc": start + pd.Timedelta(hours=1),
            "market": market,
            "price_market": "day_ahead",
            "price_location": PRICE_LOCATIONS[market],
            "da_lmp_usd_mwh": pd.to_numeric(
                pd.Series(lmp).reset_index(drop=True), errors="raise"
            ),
            "rt_lmp_usd_mwh": np.full(length, np.nan),
            "energy_component_usd_mwh": (
                np.full(length, np.nan)
                if energy is None
                else pd.to_numeric(pd.Series(energy).reset_index(drop=True))
            ),
            "congestion_component_usd_mwh": (
                np.full(length, np.nan)
                if congestion is None
                else pd.to_numeric(pd.Series(congestion).reset_index(drop=True))
            ),
            "loss_component_usd_mwh": (
                np.full(length, np.nan)
                if loss is None
                else pd.to_numeric(pd.Series(loss).reset_index(drop=True))
            ),
            "source_quality_flags": source_flag,
        }
    )
    duplicates = frame.duplicated(["market", "interval_start_utc"], keep=False)
    if duplicates.any():
        examples = (
            frame.loc[duplicates, "interval_start_utc"]
            .drop_duplicates()
            .head(5)
            .astype(str)
            .tolist()
        )
        raise ContractError(
            f"{market} price source has duplicate UTC hours: {examples}"
        )
    return frame.sort_values("interval_start_utc").reset_index(drop=True)


def acquire_prices(
    raw_root: Path,
    *,
    calendar: StudyCalendar | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    calendar = calendar or make_calendar()
    functions = {
        "CAISO_NP15": acquire_caiso_price,
        "ERCOT_LZ_NORTH": acquire_ercot_price,
        "NYISO_NYC_J": acquire_nyiso_price,
        "MISO_MINN_HUB": acquire_miso_price,
        "SPP_NORTH_HUB": acquire_spp_price,
        "ISONE_NEMA": acquire_isone_price,
    }
    frames: dict[str, pd.DataFrame] = {}
    metadata: dict[str, Any] = {}
    for market in MARKETS:
        frame, records = functions[market](raw_root, calendar=calendar)
        start = pd.Timestamp(calendar.start_utc)
        end = pd.Timestamp(calendar.end_utc)
        frame = frame.loc[
            (frame["interval_start_utc"] >= start)
            & (frame["interval_start_utc"] < end)
        ].reset_index(drop=True)
        validate_hourly_index(
            frame,
            start=start,
            end=end,
            table_name=f"{market}.price",
        )
        frames[market] = frame
        metadata[market] = records
    return frames, metadata


def write_native_inputs(
    data_root: Path,
    physical: dict[str, pd.DataFrame],
    prices: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    hashes: dict[str, Any] = {}
    for market in MARKETS:
        root = data_root / "native" / market
        root.mkdir(parents=True, exist_ok=True)
        physical_path = root / "physical_hourly.csv"
        price_path = root / "price_hourly.csv"
        physical[market].to_csv(physical_path, index=False)
        prices[market].to_csv(price_path, index=False)
        hashes[market] = {
            "physical_hourly": {
                "path": str(physical_path),
                "bytes": physical_path.stat().st_size,
                "sha256": sha256(physical_path),
            },
            "price_hourly": {
                "path": str(price_path),
                "bytes": price_path.stat().st_size,
                "sha256": sha256(price_path),
            },
        }
    return hashes
