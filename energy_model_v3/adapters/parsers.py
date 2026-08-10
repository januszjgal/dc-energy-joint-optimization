"""Source-specific primary day-ahead price parsers."""

from __future__ import annotations

import io
import re
import xml.etree.ElementTree as ET
import zipfile
from datetime import date, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from ..canonical import canonical_interval
from ..contract import PRICE_COLUMNS, ContractError, validate_native_table

XLSX_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


def _column(frame: pd.DataFrame, *names: str) -> str:
    for name in names:
        if name in frame.columns:
            return name
    raise ContractError(f"missing source column; expected one of {names}")


def _truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(
        {"true", "t", "1", "yes", "y"}
    )


def _canonical_price(
    *,
    market: str,
    start: pd.Series,
    end: pd.Series,
    price_market: str,
    location: str,
    lmp: pd.Series,
    energy: pd.Series | None = None,
    congestion: pd.Series | None = None,
    loss: pd.Series | None = None,
    quality: str,
) -> pd.DataFrame:
    size = len(start)
    result = pd.DataFrame(
        {
            "interval_start_utc": pd.to_datetime(start, utc=True),
            "interval_end_utc": pd.to_datetime(end, utc=True),
            "market": market,
            "price_market": price_market,
            "price_location": location,
            "da_lmp_usd_mwh": pd.to_numeric(lmp, errors="raise"),
            "rt_lmp_usd_mwh": np.full(size, np.nan),
            "energy_component_usd_mwh": (
                pd.to_numeric(energy, errors="raise")
                if energy is not None
                else np.full(size, np.nan)
            ),
            "congestion_component_usd_mwh": (
                pd.to_numeric(congestion, errors="raise")
                if congestion is not None
                else np.full(size, np.nan)
            ),
            "loss_component_usd_mwh": (
                pd.to_numeric(loss, errors="raise")
                if loss is not None
                else np.full(size, np.nan)
            ),
            "source_quality_flags": quality,
        }
    )
    return validate_native_table(
        result,
        PRICE_COLUMNS,
        table_name=f"{market}.parsed_price",
        expected_market=market,
        non_nullable={
            "interval_start_utc",
            "interval_end_utc",
            "market",
            "price_market",
            "price_location",
            "da_lmp_usd_mwh",
            "source_quality_flags",
        },
        numeric_columns={
            "da_lmp_usd_mwh",
            "rt_lmp_usd_mwh",
            "energy_component_usd_mwh",
            "congestion_component_usd_mwh",
            "loss_component_usd_mwh",
        },
    )


def parse_pjm_da(frame: pd.DataFrame) -> pd.DataFrame:
    pnode = _column(frame, "pnode_id", "pnodeid")
    filtered = frame.loc[
        pd.to_numeric(frame[pnode], errors="raise") == 34964545
    ].copy()
    if filtered.empty:
        raise ContractError("PJM response has no DOM pnode 34964545")
    verified = _column(filtered, "is_verified", "verified")
    current = _column(filtered, "row_is_current", "current")
    version = _column(filtered, "version_nbr", "version")
    if not _truthy(filtered[verified]).all():
        raise ContractError("PJM DAM rows are not all verified")
    if not _truthy(filtered[current]).all():
        raise ContractError("PJM DAM rows are not all current")
    if filtered[version].isna().any() or (
        filtered[version].astype(str).str.strip() == ""
    ).any():
        raise ContractError("PJM DAM rows lack version provenance")
    timestamp = _column(
        filtered,
        "datetime_beginning_utc",
        "datetime_beginning_ept",
    )
    if timestamp.endswith("_utc"):
        start = pd.to_datetime(filtered[timestamp], utc=True, errors="raise")
    else:
        interval = canonical_interval(
            filtered[timestamp],
            timezone_name="America/New_York",
            semantics="interval_beginning",
            cadence=pd.Timedelta(hours=1),
        )
        start = interval["interval_start_utc"]
    lmp = _column(filtered, "total_lmp_da", "da_lmp")
    return _canonical_price(
        market="PJM_DOM",
        start=start,
        end=start + pd.Timedelta(hours=1),
        price_market="PJM DAM settlement-final",
        location="DOM pnode 34964545",
        lmp=filtered[lmp],
        energy=filtered.get("system_energy_price_da"),
        congestion=filtered.get("congestion_price_da"),
        loss=filtered.get("marginal_loss_price_da"),
        quality="PJM_VERSION_FIELDS_REQUIRED_IN_PROVENANCE",
    )


def parse_nyiso_da(
    frame: pd.DataFrame,
    *,
    ambiguous: pd.Series | str = "infer",
) -> pd.DataFrame:
    ptid = _column(frame, "PTID")
    filtered = frame.loc[
        pd.to_numeric(frame[ptid], errors="raise") == 61761
    ].copy()
    if filtered.empty:
        raise ContractError("NYISO archive has no Zone J PTID 61761")
    interval = canonical_interval(
        filtered[_column(filtered, "Time Stamp")],
        timezone_name="America/New_York",
        semantics="interval_beginning",
        cadence=pd.Timedelta(hours=1),
        ambiguous=ambiguous,
    )
    return _canonical_price(
        market="NYISO_NYC_J",
        start=interval["interval_start_utc"],
        end=interval["interval_end_utc"],
        price_market="NYISO DAM zonal LBMP",
        location="N.Y.C. Zone J PTID 61761",
        lmp=filtered[_column(filtered, "LBMP ($/MWHr)")],
        congestion=filtered.get("Marginal Cost Congestion ($/MWHr)"),
        loss=filtered.get("Marginal Cost Losses ($/MWHr)"),
        quality="NYISO_MONTHLY_ARCHIVE_HASH_REQUIRED",
    )


def parse_caiso_da(frame: pd.DataFrame) -> pd.DataFrame:
    node_column = _column(frame, "NODE", "NODE_ID")
    filtered = frame.loc[
        (frame[node_column].astype(str) == "TH_NP15_GEN-APND")
        & (frame[_column(frame, "MARKET_RUN_ID")].astype(str) == "DAM")
    ].copy()
    if filtered.empty:
        raise ContractError("CAISO response has no NP15 DAM rows")
    value_column = _column(filtered, "MW")
    duplicate_key = [
        "INTERVALSTARTTIME_GMT",
        "INTERVALENDTIME_GMT",
        "LMP_TYPE",
    ]
    if filtered.duplicated(duplicate_key).any():
        raise ContractError(
            "CAISO response has duplicate interval/LMP_TYPE rows"
        )
    pivot = filtered.pivot_table(
        index=["INTERVALSTARTTIME_GMT", "INTERVALENDTIME_GMT"],
        columns="LMP_TYPE",
        values=value_column,
        aggfunc="first",
    ).reset_index()
    if "LMP" not in pivot:
        raise ContractError("CAISO response has no LMP_TYPE=LMP")
    return _canonical_price(
        market="CAISO_NP15",
        start=pivot["INTERVALSTARTTIME_GMT"],
        end=pivot["INTERVALENDTIME_GMT"],
        price_market="CAISO OASIS DAM PRC_LMP v12",
        location="TH_NP15_GEN-APND",
        lmp=pivot["LMP"],
        energy=pivot.get("MCE"),
        congestion=pivot.get("MCC"),
        loss=pivot.get("MCL"),
        quality="CAISO_INTERVALSTARTTIME_GMT_AUTHORITATIVE",
    )


def parse_ercot_da(frame: pd.DataFrame) -> pd.DataFrame:
    filtered = frame.loc[
        frame[_column(frame, "Settlement Point")].astype(str) == "LZ_NORTH"
    ].copy()
    if filtered.empty:
        raise ContractError("ERCOT workbook has no LZ_NORTH rows")
    delivery = pd.to_datetime(
        filtered[_column(filtered, "Delivery Date")], errors="raise"
    ).dt.normalize()
    hour_text = filtered[_column(filtered, "Hour Ending")].astype(str)
    hour = pd.to_timedelta(
        hour_text.str.split(":").str[0].astype(int), unit="h"
    )
    local_start = delivery + hour - pd.Timedelta(hours=1)
    repeated = filtered[
        _column(filtered, "Repeated Hour Flag", "DSTFlag")
    ].astype(str).str.upper()
    ambiguous = ~(repeated.isin({"Y", "1", "TRUE"}))
    localized_start = local_start.dt.tz_localize(
        "America/Chicago",
        ambiguous=ambiguous.to_numpy(),
        nonexistent="raise",
    ).dt.tz_convert("UTC")
    return _canonical_price(
        market="ERCOT_LZ_NORTH",
        start=localized_start,
        end=localized_start + pd.Timedelta(hours=1),
        price_market="ERCOT DAM report 13060",
        location="LZ_NORTH",
        lmp=filtered[_column(filtered, "Settlement Point Price")],
        quality="ERCOT_DOCID_FILENAME_HASH_REQUIRED",
    )


def read_ercot_xlsx_sheet(
    workbook: bytes,
    *,
    sheet_number: int,
) -> pd.DataFrame:
    """Read the five-column ERCOT annual workbook without an Excel dependency."""
    with zipfile.ZipFile(io.BytesIO(workbook)) as archive:
        shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        shared = [
            "".join(
                node.text or ""
                for node in item.iter(f"{{{XLSX_NS}}}t")
            )
            for item in shared_root.findall(f"{{{XLSX_NS}}}si")
        ]
        sheet_path = f"xl/worksheets/Sheet{sheet_number}.xml"
        sheet_root = ET.fromstring(archive.read(sheet_path))
    rows: list[dict[int, str]] = []
    for row in sheet_root.findall(f".//{{{XLSX_NS}}}sheetData/{{{XLSX_NS}}}row"):
        values: dict[int, str] = {}
        for cell in row.findall(f"{{{XLSX_NS}}}c"):
            reference = cell.get("r", "")
            match = re.match(r"([A-Z]+)", reference)
            if match is None:
                raise ContractError(f"invalid ERCOT XLSX cell {reference!r}")
            letters = match.group(1)
            column = 0
            for letter in letters:
                column = column * 26 + ord(letter) - ord("A") + 1
            value_node = cell.find(f"{{{XLSX_NS}}}v")
            value = "" if value_node is None else value_node.text or ""
            if cell.get("t") == "s" and value:
                value = shared[int(value)]
            values[column - 1] = value
        rows.append(values)
    if not rows:
        raise ContractError("ERCOT workbook sheet is empty")
    headers = rows[0]
    expected = {
        0: "Delivery Date",
        1: "Hour Ending",
        2: "Repeated Hour Flag",
        3: "Settlement Point",
        4: "Settlement Point Price",
    }
    if headers != expected:
        raise ContractError(f"unexpected ERCOT workbook headers: {headers}")
    return pd.DataFrame(
        [
            {headers[index]: row.get(index, "") for index in headers}
            for row in rows[1:]
        ]
    )


def _read_xlsx_worksheet(workbook: bytes, sheet_path: str) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(workbook)) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = [
                "".join(
                    node.text or ""
                    for node in item.iter(f"{{{XLSX_NS}}}t")
                )
                for item in shared_root.findall(f"{{{XLSX_NS}}}si")
            ]
        sheet_root = ET.fromstring(archive.read(sheet_path))
    rows: list[dict[int, str]] = []
    for row in sheet_root.findall(
        f".//{{{XLSX_NS}}}sheetData/{{{XLSX_NS}}}row"
    ):
        values: dict[int, str] = {}
        for cell in row.findall(f"{{{XLSX_NS}}}c"):
            reference = cell.get("r", "")
            match = re.match(r"([A-Z]+)", reference)
            if match is None:
                raise ContractError(f"invalid XLSX cell {reference!r}")
            column = 0
            for letter in match.group(1):
                column = column * 26 + ord(letter) - ord("A") + 1
            value_node = cell.find(f"{{{XLSX_NS}}}v")
            value = "" if value_node is None else value_node.text or ""
            if cell.get("t") == "s" and value:
                value = shared[int(value)]
            elif cell.get("t") == "inlineStr":
                value = "".join(
                    node.text or ""
                    for node in cell.iter(f"{{{XLSX_NS}}}t")
                )
            values[column - 1] = value
        rows.append(values)
    if not rows:
        raise ContractError(f"XLSX worksheet is empty: {sheet_path}")
    headers = {
        index: value.strip()
        for index, value in rows[0].items()
        if value.strip()
    }
    if len(set(headers.values())) != len(headers):
        raise ContractError(f"XLSX worksheet has duplicate headers: {sheet_path}")
    records = [
        {header: row.get(index, "") for index, header in headers.items()}
        for row in rows[1:]
        if any(row.get(index, "") != "" for index in headers)
    ]
    return pd.DataFrame(records, columns=list(headers.values()))


def _central_start_to_utc(
    local_start: pd.Series,
    repeated: pd.Series,
) -> pd.Series:
    repeated_values = repeated.astype(str).str.strip().str.upper()
    invalid = ~repeated_values.isin({"N", "Y"})
    if invalid.any():
        raise ContractError(
            "ERCOT repeated-hour flag must contain only N or Y"
        )
    ambiguous = ~repeated_values.eq("Y")
    return local_start.dt.tz_localize(
        "America/Chicago",
        ambiguous=ambiguous.to_numpy(),
        nonexistent="raise",
    ).dt.tz_convert("UTC")


def read_ercot_native_load_xlsx(
    workbook: bytes,
    *,
    start_utc: pd.Timestamp | str | None = None,
    end_utc: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Parse the public Native_Load_YYYY.xlsx hourly ERCOT-system series."""
    frame = _read_xlsx_worksheet(workbook, "xl/worksheets/sheet1.xml")
    expected = {
        "Hour Ending",
        "COAST",
        "EAST",
        "FWEST",
        "NORTH",
        "NCENT",
        "SOUTH",
        "SCENT",
        "WEST",
        "ERCOT",
    }
    if set(frame.columns) != expected:
        raise ContractError(
            f"unexpected ERCOT native-load workbook headers: {frame.columns.tolist()}"
        )
    hour_ending = frame["Hour Ending"].astype(str).str.strip()
    parsed = hour_ending.str.extract(
        r"^(?P<day>\d{2}/\d{2}/\d{4}) "
        r"(?P<hour>\d{2}):00(?P<repeated> DST)?$"
    )
    if parsed[["day", "hour"]].isna().any().any():
        raise ContractError("ERCOT native-load hour-ending value is invalid")
    hour = pd.to_numeric(parsed["hour"], errors="raise")
    if (~hour.between(1, 24)).any():
        raise ContractError("ERCOT native-load hour must be 01 through 24")
    local_start = (
        pd.to_datetime(parsed["day"], format="%m/%d/%Y", errors="raise")
        + pd.to_timedelta(hour - 1, unit="h")
    )
    # The archive's explicit DST suffix is the earlier daylight occurrence.
    repeated = parsed["repeated"].fillna("").ne("").map({False: "Y", True: "N"})
    start = _central_start_to_utc(local_start, repeated)
    result = pd.DataFrame(
        {
            "interval_start_utc": start,
            "interval_end_utc": start + pd.Timedelta(hours=1),
            "market": "ERCOT_LZ_NORTH",
            "gross_demand_mw": pd.to_numeric(frame["ERCOT"], errors="raise"),
            "source_quality_flags": (
                "ERCOT_PUBLIC_NATIVE_LOAD_ARCHIVE;"
                "PRICE_PHYSICAL_BOUNDARY_MISMATCH_EXPLICIT"
            ),
        }
    )
    if start_utc is not None:
        result = result.loc[
            result["interval_start_utc"] >= pd.Timestamp(start_utc)
        ]
    if end_utc is not None:
        result = result.loc[
            result["interval_start_utc"] < pd.Timestamp(end_utc)
        ]
    if result.empty:
        raise ContractError("ERCOT native-load archive has no requested rows")
    numeric = result["gross_demand_mw"]
    if not np.isfinite(numeric.to_numpy()).all() or (numeric < 0).any():
        raise ContractError("ERCOT native-load archive has invalid system load")
    if result["interval_start_utc"].duplicated().any():
        raise ContractError(
            "ERCOT native-load archive has duplicate requested UTC intervals"
        )
    return result.sort_values("interval_start_utc").reset_index(drop=True)


def select_ercot_sced_document(
    documents: list[dict[str, Any]],
    *,
    sced_date: date | str,
) -> dict[str, Any]:
    """Select the public 60-day disclosure for a SCED operating date."""
    operating_date = pd.Timestamp(sced_date).date()
    publication_date = operating_date + timedelta(days=60)
    matches: list[dict[str, Any]] = []
    for item in documents:
        document = item.get("Document", {})
        try:
            published = pd.Timestamp(document["PublishDate"]).date()
        except (KeyError, TypeError, ValueError):
            continue
        if (
            str(document.get("ReportTypeID")) == "13052"
            and document.get("FriendlyName") == "60_Day_SCED_Disclosure"
            and document.get("SecurityStatus") == "P"
            and str(document.get("Extension")).lower() == "zip"
            and published == publication_date
        ):
            matches.append(document)
    if not matches:
        raise ContractError(
            "ERCOT discovery has no public report-13052 disclosure for "
            f"SCED date {operating_date.isoformat()}"
        )
    return max(matches, key=lambda document: pd.Timestamp(document["PublishDate"]))


def read_ercot_sced_generation_zip(raw_archive: bytes) -> pd.DataFrame:
    """Read the generation-resource table from one report-13052 ZIP."""
    with zipfile.ZipFile(io.BytesIO(raw_archive)) as archive:
        names = [
            name
            for name in archive.namelist()
            if re.search(
                r"(?:^|/)60d_SCED_Gen_Resource_Data-\d{2}-[A-Z]{3}-\d{2}\.csv$",
                name,
                re.IGNORECASE,
            )
        ]
        if len(names) != 1:
            raise ContractError(
                "report-13052 archive must contain exactly one generation "
                "resource CSV"
            )
        with archive.open(names[0]) as handle:
            frame = pd.read_csv(
                handle,
                usecols=lambda value: value.strip()
                in {
                    "SCED Time Stamp",
                    "Repeated Hour Flag",
                    "Resource Name",
                    "Resource Type",
                    "Telemetered Net Output",
                },
                low_memory=False,
            )
    frame.columns = frame.columns.str.strip()
    required = {
        "SCED Time Stamp",
        "Repeated Hour Flag",
        "Resource Name",
        "Resource Type",
        "Telemetered Net Output",
    }
    if set(frame.columns) != required:
        raise ContractError(
            f"report-13052 generation columns are invalid: {frame.columns.tolist()}"
        )
    return frame


def parse_ercot_sced_executions(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate WGR/PVGR telemetry at each irregular SCED execution."""
    source = frame.copy()
    source.columns = source.columns.str.strip()
    required = {
        "SCED Time Stamp",
        "Repeated Hour Flag",
        "Resource Name",
        "Resource Type",
        "Telemetered Net Output",
    }
    missing = required - set(source.columns)
    if missing:
        raise ContractError(
            f"report-13052 generation table missing {sorted(missing)}"
        )
    source["Resource Type"] = (
        source["Resource Type"].astype(str).str.strip().str.upper()
    )
    source = source.loc[
        source["Resource Type"].isin({"WGR", "WIND", "PVGR"})
    ].copy()
    source["Resource Type"] = source["Resource Type"].replace({"WIND": "WGR"})
    if source.empty or set(source["Resource Type"]) != {"WGR", "PVGR"}:
        raise ContractError("report-13052 lacks WGR or PVGR resource telemetry")
    source["Resource Name"] = source["Resource Name"].astype(str).str.strip()
    if source["Resource Name"].eq("").any():
        raise ContractError("report-13052 has an empty renewable resource name")
    resource_types = source.groupby("Resource Name")["Resource Type"].nunique()
    if (resource_types != 1).any():
        raise ContractError(
            "report-13052 changes renewable classification for a resource"
        )
    timestamp_text = source["SCED Time Stamp"].astype(str).str.strip()
    local = pd.to_datetime(
        timestamp_text, format="%m/%d/%Y %H:%M:%S", errors="raise"
    )
    source["timestamp_utc"] = _central_start_to_utc(
        local, source["Repeated Hour Flag"]
    )
    source["mw"] = pd.to_numeric(
        source["Telemetered Net Output"], errors="raise"
    )
    if not np.isfinite(source["mw"].to_numpy()).all():
        raise ContractError("report-13052 has non-finite renewable telemetry")
    duplicate_key = ["timestamp_utc", "Resource Name"]
    if source.duplicated(duplicate_key).any():
        raise ContractError(
            "report-13052 duplicates a renewable resource execution"
        )
    grouped = (
        source.groupby(["timestamp_utc", "Resource Type"])
        .agg(mw=("mw", "sum"), resource_count=("Resource Name", "nunique"))
        .reset_index()
    )
    value = grouped.pivot(
        index="timestamp_utc", columns="Resource Type", values="mw"
    )
    count = grouped.pivot(
        index="timestamp_utc", columns="Resource Type", values="resource_count"
    )
    if value.isna().any().any() or count.isna().any().any():
        raise ContractError(
            "report-13052 execution is missing all WGR or PVGR resources"
        )
    result = pd.DataFrame(
        {
            "timestamp_utc": value.index,
            "wind_mw": value["WGR"].to_numpy(),
            "solar_mw": value["PVGR"].to_numpy(),
            "wind_resource_count": count["WGR"].to_numpy(dtype=int),
            "solar_resource_count": count["PVGR"].to_numpy(dtype=int),
        }
    ).sort_values("timestamp_utc")
    if result["timestamp_utc"].duplicated().any():
        raise ContractError("report-13052 has duplicate UTC executions")
    return result.reset_index(drop=True)


def time_weight_ercot_sced_hourly(
    executions: pd.DataFrame,
    *,
    start_utc: pd.Timestamp | str,
    end_utc: pd.Timestamp | str,
    gap_warning: pd.Timedelta = pd.Timedelta(minutes=20),
) -> pd.DataFrame:
    """Integrate SCED step values across UTC hours without circular padding."""
    start = pd.Timestamp(start_utc)
    end = pd.Timestamp(end_utc)
    if start.tzinfo is None or end.tzinfo is None:
        raise ContractError("SCED hourly bounds must be timezone aware")
    start = start.tz_convert("UTC")
    end = end.tz_convert("UTC")
    if end <= start or start != start.floor("1h") or end != end.floor("1h"):
        raise ContractError("SCED hourly bounds must be increasing UTC hours")
    required = {
        "timestamp_utc",
        "wind_mw",
        "solar_mw",
        "wind_resource_count",
        "solar_resource_count",
    }
    missing = required - set(executions.columns)
    if missing:
        raise ContractError(f"SCED executions missing {sorted(missing)}")
    source = executions.copy()
    source["timestamp_utc"] = pd.to_datetime(
        source["timestamp_utc"], utc=True, errors="raise"
    )
    source = source.sort_values("timestamp_utc").reset_index(drop=True)
    if source["timestamp_utc"].duplicated().any():
        raise ContractError("SCED executions have duplicate timestamps")
    if source.iloc[0]["timestamp_utc"] > start:
        raise ContractError("SCED reconstruction lacks real warm history")
    if source.iloc[-1]["timestamp_utc"] < end:
        raise ContractError("SCED reconstruction lacks a real terminal tail")
    for column in ("wind_mw", "solar_mw"):
        source[column] = pd.to_numeric(source[column], errors="raise")
        if not np.isfinite(source[column].to_numpy()).all():
            raise ContractError(f"SCED executions have invalid {column}")
    hours = pd.date_range(start, end, freq="1h", inclusive="left")
    accumulators = {
        hour: {
            "wind_mw_seconds": 0.0,
            "solar_mw_seconds": 0.0,
            "covered_seconds": 0.0,
            "execution_count": 0,
            "max_gap_seconds": 0.0,
            "min_wind_resources": np.inf,
            "min_solar_resources": np.inf,
        }
        for hour in hours
    }
    timestamps = source["timestamp_utc"].tolist()
    for index in range(len(source) - 1):
        segment_start = max(timestamps[index], start)
        segment_end = min(timestamps[index + 1], end)
        if segment_end <= segment_start:
            continue
        gap_seconds = (timestamps[index + 1] - timestamps[index]).total_seconds()
        if gap_seconds <= 0:
            raise ContractError("SCED execution order is invalid")
        cursor = segment_start
        while cursor < segment_end:
            hour = cursor.floor("1h")
            boundary = min(hour + pd.Timedelta(hours=1), segment_end)
            seconds = (boundary - cursor).total_seconds()
            bucket = accumulators[hour]
            bucket["wind_mw_seconds"] += source.at[index, "wind_mw"] * seconds
            bucket["solar_mw_seconds"] += source.at[index, "solar_mw"] * seconds
            bucket["covered_seconds"] += seconds
            bucket["max_gap_seconds"] = max(
                bucket["max_gap_seconds"], gap_seconds
            )
            bucket["min_wind_resources"] = min(
                bucket["min_wind_resources"],
                source.at[index, "wind_resource_count"],
            )
            bucket["min_solar_resources"] = min(
                bucket["min_solar_resources"],
                source.at[index, "solar_resource_count"],
            )
            cursor = boundary
        if start <= timestamps[index] < end:
            accumulators[timestamps[index].floor("1h")]["execution_count"] += 1
    rows: list[dict[str, Any]] = []
    for hour, bucket in accumulators.items():
        if not np.isclose(bucket["covered_seconds"], 3600.0):
            raise ContractError(
                f"SCED reconstruction does not cover full UTC hour {hour}"
            )
        flags = [
            "ERCOT_SCED_13052_TIME_WEIGHTED",
            "RESOURCE_CLASSIFICATION_WGR_PVGR",
            "SOURCE_WIND_CLASS_NORMALIZED_TO_WGR",
        ]
        if bucket["max_gap_seconds"] > gap_warning.total_seconds():
            flags.append("SCED_EXECUTION_GAP_GT_20_MINUTES")
        rows.append(
            {
                "interval_start_utc": hour,
                "interval_end_utc": hour + pd.Timedelta(hours=1),
                "market": "ERCOT_LZ_NORTH",
                "wind_mw": bucket["wind_mw_seconds"] / 3600.0,
                "solar_mw": bucket["solar_mw_seconds"] / 3600.0,
                "execution_count": bucket["execution_count"],
                "max_execution_gap_seconds": bucket["max_gap_seconds"],
                "min_wind_resource_count": int(bucket["min_wind_resources"]),
                "min_solar_resource_count": int(bucket["min_solar_resources"]),
                "source_quality_flags": ";".join(flags),
            }
        )
    return pd.DataFrame(rows)


def parse_ercot_annual_renewables_xlsx(workbook: bytes) -> pd.DataFrame:
    """Parse report 13424 for independent SCED-reconstruction validation."""
    wind = _read_xlsx_worksheet(workbook, "xl/worksheets/sheet1.xml")
    solar = _read_xlsx_worksheet(workbook, "xl/worksheets/sheet3.xml")
    expected_wind = {"Time (Hour-Ending)", "ERCOT.WIND.GEN"}
    expected_solar = {"Time (Hour-Ending)", "ERCOT.PVGR.GEN"}
    if not expected_wind.issubset(wind.columns):
        raise ContractError("report 13424 wind worksheet schema is invalid")
    if not expected_solar.issubset(solar.columns):
        raise ContractError("report 13424 solar worksheet schema is invalid")

    def parse_sheet(
        frame: pd.DataFrame, value_column: str, output_column: str
    ) -> pd.DataFrame:
        serial = pd.to_numeric(frame["Time (Hour-Ending)"], errors="raise")
        offsets = pd.to_timedelta(serial, unit="D").dt.round("s")
        local_end = pd.Timestamp("1899-12-30") + offsets
        utc_end = local_end.dt.tz_localize(
            "America/Chicago", ambiguous="infer", nonexistent="raise"
        ).dt.tz_convert("UTC")
        utc = utc_end - pd.Timedelta(hours=1)
        result = pd.DataFrame(
            {
                "interval_start_utc": utc,
                output_column: pd.to_numeric(frame[value_column], errors="raise"),
            }
        )
        if result["interval_start_utc"].duplicated().any():
            raise ContractError("report 13424 has duplicate UTC intervals")
        return result

    wind_values = parse_sheet(wind, "ERCOT.WIND.GEN", "wind_mw")
    solar_values = parse_sheet(solar, "ERCOT.PVGR.GEN", "solar_mw")
    result = wind_values.merge(
        solar_values, on="interval_start_utc", how="outer", validate="one_to_one"
    )
    if result[["wind_mw", "solar_mw"]].isna().any().any():
        raise ContractError("report 13424 wind and solar hours do not align")
    result["interval_end_utc"] = (
        result["interval_start_utc"] + pd.Timedelta(hours=1)
    )
    result["market"] = "ERCOT_LZ_NORTH"
    result["source_quality_flags"] = "ERCOT_REPORT_13424_VALIDATION_ONLY"
    return result.sort_values("interval_start_utc").reset_index(drop=True)


def compare_ercot_renewables(
    reconstructed: pd.DataFrame,
    annual: pd.DataFrame,
    *,
    start_utc: pd.Timestamp | str,
    end_utc: pd.Timestamp | str,
) -> dict[str, Any]:
    """Return non-calibrating Sep-Dec validation diagnostics."""
    start = pd.Timestamp(start_utc)
    end = pd.Timestamp(end_utc)
    left = reconstructed.loc[
        reconstructed["interval_start_utc"].between(
            start, end, inclusive="left"
        ),
        ["interval_start_utc", "wind_mw", "solar_mw"],
    ]
    right = annual.loc[
        annual["interval_start_utc"].between(start, end, inclusive="left"),
        ["interval_start_utc", "wind_mw", "solar_mw"],
    ]
    expected_index = pd.date_range(start, end, freq="1h", inclusive="left")
    left_index = pd.DatetimeIndex(
        pd.to_datetime(left["interval_start_utc"], utc=True, errors="raise")
    )
    right_index = pd.DatetimeIndex(
        pd.to_datetime(right["interval_start_utc"], utc=True, errors="raise")
    )
    missing_sced_hours = len(expected_index.difference(left_index))
    missing_annual_hours = len(expected_index.difference(right_index))
    merged = left.merge(
        right,
        on="interval_start_utc",
        how="outer",
        suffixes=("_sced", "_annual"),
        indicator=True,
        validate="one_to_one",
    )
    matched = merged.loc[merged["_merge"] == "both"].copy()
    if matched.empty:
        raise ContractError("ERCOT renewable validation has no matched hours")
    metrics: dict[str, dict[str, float]] = {}
    for component in ("wind", "solar"):
        difference = (
            matched[f"{component}_mw_sced"]
            - matched[f"{component}_mw_annual"]
        )
        metrics[component] = {
            "bias_mw": float(difference.mean()),
            "mae_mw": float(difference.abs().mean()),
            "rmse_mw": float(np.sqrt(np.mean(np.square(difference)))),
            "difference_q05_mw": float(difference.quantile(0.05)),
            "difference_q50_mw": float(difference.quantile(0.50)),
            "difference_q95_mw": float(difference.quantile(0.95)),
            "correlation": float(
                matched[
                    [f"{component}_mw_sced", f"{component}_mw_annual"]
                ].corr().iloc[0, 1]
            ),
        }
    return {
        "schema_version": "energy-model-v3-ercot-renewable-validation",
        "validation_only": True,
        "calibration_applied": False,
        "start_utc": start.isoformat(),
        "end_utc": end.isoformat(),
        "matched_hours": len(matched),
        "missing_sced_hours": missing_sced_hours,
        "missing_annual_hours": missing_annual_hours,
        "metrics": metrics,
        "caveats": [
            "SCED telemetry uses WGR/PVGR resource classifications.",
            "Irregular executions are duration weighted as step values.",
            "Differences are diagnostics only and never reconcile the primary series.",
        ],
    }


def select_ercot_document(
    documents: list[dict[str, Any]],
    *,
    friendly_name: str,
) -> dict[str, Any]:
    """Select the newest discovered public ERCOT revision by metadata."""
    matches = [
        item["Document"]
        for item in documents
        if item.get("Document", {}).get("FriendlyName") == friendly_name
        and item.get("Document", {}).get("SecurityStatus") == "P"
    ]
    if not matches:
        raise ContractError(
            f"ERCOT discovery has no public {friendly_name} document"
        )
    return max(
        matches,
        key=lambda document: pd.Timestamp(document["PublishDate"]),
    )


def parse_miso_da(frame: pd.DataFrame, delivery_date: str) -> pd.DataFrame:
    filtered = frame.loc[
        (frame[_column(frame, "Node")].astype(str) == "MINN.HUB")
        & (frame[_column(frame, "Type")].astype(str) == "Hub")
    ]
    values: dict[str, pd.Series] = {}
    for component in ("LMP", "MCC", "MLC"):
        rows = filtered.loc[
            filtered[_column(filtered, "Value")].astype(str) == component
        ]
        if len(rows) != 1:
            raise ContractError(
                "MISO report must contain one "
                f"MINN.HUB/Hub/{component} row"
            )
        values[component] = rows.iloc[0]
    hours = [f"HE {hour}" for hour in range(1, 25)]
    missing = [column for column in hours if column not in frame]
    if missing:
        raise ContractError(f"MISO report missing hours: {missing}")
    fixed_est = timezone(timedelta(hours=-5))
    delivery = pd.Timestamp(delivery_date).tz_localize(fixed_est)
    end = pd.Series(
        [delivery + pd.Timedelta(hours=hour) for hour in range(1, 25)]
    )
    return _canonical_price(
        market="MISO_MINN_HUB",
        start=end - pd.Timedelta(hours=1),
        end=end,
        price_market="MISO DA ex-post LMP",
        location="MINN.HUB",
        lmp=pd.Series([values["LMP"][column] for column in hours]),
        congestion=pd.Series([values["MCC"][column] for column in hours]),
        loss=pd.Series([values["MLC"][column] for column in hours]),
        quality="MISO_FIXED_EST_HOUR_ENDING",
    )


def parse_spp_da(frame: pd.DataFrame) -> pd.DataFrame:
    filtered = frame.loc[
        frame[_column(frame, "Settlement Location")].astype(str)
        == "SPPNORTH_HUB"
    ].copy()
    if filtered.empty:
        raise ContractError("SPP report has no SPPNORTH_HUB rows")
    end = pd.to_datetime(
        filtered[_column(filtered, "GMTIntervalEnd")],
        utc=True,
        errors="raise",
    )
    return _canonical_price(
        market="SPP_NORTH_HUB",
        start=end - pd.Timedelta(hours=1),
        end=end,
        price_market="SPP DAM",
        location="SPPNORTH_HUB",
        lmp=filtered[_column(filtered, "LMP")],
        energy=filtered.get("MEC"),
        congestion=filtered.get("MCC"),
        loss=filtered.get("MLC"),
        quality="SPP_GMT_INTERVAL_END_AUTHORITATIVE",
    )
