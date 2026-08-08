"""Source-specific primary day-ahead price parsers."""

from __future__ import annotations

import io
import re
import xml.etree.ElementTree as ET
import zipfile
from datetime import timedelta, timezone

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
