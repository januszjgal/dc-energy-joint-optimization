"""Command-line interface for v3 provenance, probes, and fail-closed builds."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .adapters import (
    ADAPTERS,
    compare_ercot_renewables,
    get_adapter,
    parse_caiso_da,
    parse_ercot_annual_renewables_xlsx,
    parse_ercot_da,
    parse_ercot_sced_executions,
    parse_miso_da,
    parse_nyiso_da,
    parse_spp_da,
    read_ercot_native_load_xlsx,
    read_ercot_sced_generation_zip,
    read_ercot_xlsx_sheet,
    select_ercot_document,
    select_ercot_sced_document,
    time_weight_ercot_sced_hourly,
)
from .adapters.base import download_raw
from .acquisition import (
    acquire_physical,
    acquire_prices,
    sha256 as acquisition_sha256,
    write_native_inputs,
)
from .builder import (
    BLOCKED_MARKETS,
    MARKETS,
    blocked_preflight,
    build_panel,
    fixture_market_frame,
    load_scenario,
    sha256,
)
from .calendar import make_calendar
from .canonical import derive_net_load
from .contract import ContractError
from .coverage import assess_candidate_coverage
from .derivations import (
    add_grid_features,
    calibrate_scale,
    diagnostic_summary,
)
from .diagnostics import plot_market_diagnostics
from .forecasts import write_forecasts

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data" / "energy_model_v3"
DEFAULT_OUTPUT = ROOT / "output" / "energy_model_v3"
DEFAULT_SCENARIO = (
    ROOT / "env" / "scenarios" / "us_six_market_v3_primary.yaml"
)


def build_measured_dc_power(
    data_root: Path,
    scenario_path: Path,
    *,
    calendar: object,
) -> tuple[dict[str, pd.Series], dict[str, object]]:
    scenario = load_scenario(scenario_path)
    power_model_path = ROOT / "data" / "power_model_params.json"
    power_models = json.loads(
        power_model_path.read_text(encoding="utf-8")
    )["per_cell_cpu_model"]
    index = pd.date_range(
        calendar.start_utc, calendar.end_utc, freq="1h", inclusive="left"
    )
    output_root = data_root / "modeled_dc_power"
    output_root.mkdir(parents=True, exist_ok=True)
    power_by_market: dict[str, pd.Series] = {}
    records: dict[str, object] = {}
    for site in scenario["sites"]:
        market = site["market"]
        cell = site["borg_cell"]
        cell_path = ROOT / "data" / "cells" / f"cell_{cell}.csv"
        source = pd.read_csv(cell_path)
        if list(source.columns) != ["timestep", "cpu_demand_norm"]:
            raise ContractError(f"{cell_path} has an unexpected workload schema")
        expected_steps = np.arange(len(source))
        if not np.array_equal(source["timestep"].to_numpy(), expected_steps):
            raise ContractError(f"{cell_path} workload timesteps are not contiguous")
        utilization = pd.to_numeric(
            source["cpu_demand_norm"], errors="raise"
        )
        if utilization.isna().any() or not utilization.between(0.0, 1.0).all():
            raise ContractError(f"{cell_path} workload values are invalid")
        if len(utilization) % 12 == 1:
            aggregation_values = utilization.iloc[:-1]
        elif len(utilization) % 12 == 0:
            aggregation_values = utilization
        else:
            raise ContractError(
                f"{cell_path} cannot be aggregated into complete hours"
            )
        hourly_utilization = (
            aggregation_values.groupby(
                np.arange(len(aggregation_values)) // 12
            ).mean().to_numpy()
        )
        repetitions = int(np.ceil(len(index) / len(hourly_utilization)))
        candidate_utilization = np.tile(
            hourly_utilization, repetitions
        )[: len(index)]
        model = power_models[cell]
        rating = float(site["rated_power_mw"])
        power = pd.Series(
            (
                float(model["idle_power"])
                + float(model["slope"]) * candidate_utilization
            )
            * rating,
            dtype=float,
        )
        artifact = output_root / f"{market}.csv"
        pd.DataFrame(
            {
                "interval_start_utc": index,
                "market": market,
                "borg_cell": cell,
                "dc_power_mw": power,
                "source_quality_flags": (
                    "measured_Borg_ClusterData2019_cell_5min_to_hourly_mean_"
                    "trace_tiled_from_candidate_start"
                ),
            }
        ).to_csv(artifact, index=False)
        power_by_market[market] = power
        records[market] = {
            "borg_cell": cell,
            "rated_power_mw": rating,
            "source": str(cell_path),
            "source_sha256": sha256(cell_path),
            "source_5min_rows": len(utilization),
            "aggregation_5min_rows": len(aggregation_values),
            "terminal_boundary_row_excluded": (
                len(aggregation_values) != len(utilization)
            ),
            "hourly_profile_rows": len(hourly_utilization),
            "candidate_rows": len(index),
            "trace_repetitions": repetitions,
            "power_model": {
                "idle_power": float(model["idle_power"]),
                "slope": float(model["slope"]),
                "peak_power": float(model["peak_power"]),
            },
            "artifact": str(artifact.resolve()),
            "artifact_bytes": artifact.stat().st_size,
            "artifact_sha256": sha256(artifact),
        }
    manifest = {
        "source_role": "measured_workload_power_model",
        "mapping": scenario["mapping"],
        "temporal_mapping": (
            "5-minute measured month aggregated to hourly means, tiled from "
            "candidate UTC start without market-data substitution"
        ),
        "power_model_path": str(power_model_path),
        "power_model_sha256": sha256(power_model_path),
        "markets": records,
    }
    write_json(data_root / "provenance" / "workload-power-manifest.json", manifest)
    return power_by_market, manifest


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def command_describe(args: argparse.Namespace) -> int:
    output = Path(args.output)
    descriptors = {
        market: [
            descriptor.to_dict()
            for descriptor in get_adapter(market).descriptors()
        ]
        for market in MARKETS
    }
    write_json(
        output,
        {
            "schema_version": "energy-model-v3",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "sources": descriptors,
        },
    )
    return 0


def command_probe(args: argparse.Namespace) -> int:
    capabilities = [
        get_adapter(market).probe(timeout=args.timeout).to_dict()
        for market in MARKETS
    ]
    write_json(Path(args.output), capabilities)
    unavailable = [item for item in capabilities if item["status"] != "AVAILABLE"]
    return 2 if unavailable else 0


def command_download_samples(args: argparse.Namespace) -> int:
    records: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory() as temporary:
        temporary_root = Path(temporary)
        for market in MARKETS:
            adapter = get_adapter(market)
            capability = adapter.capability()
            if capability.status == "BLOCKED":
                records.append(capability.to_dict())
                continue
            url, request = adapter.probe_url()
            request = dict(request)
            headers = {
                "User-Agent": "dc-energy-joint-optimization/energy-model-v3",
                **request.pop("headers", {}),
            }
            metadata = download_raw(
                url,
                temporary_root / f"{market}.sample",
                params=request.pop("params", None),
                headers=headers,
                timeout=args.timeout,
            )
            records.append(
                {
                    "market": market,
                    "status": "SAMPLE_HASHED",
                    "scope": "single locked probe request, not full study",
                    "raw_persisted": False,
                    "redistribution": adapter.redistribution,
                    **metadata,
                }
            )
    write_json(Path(args.output), records)
    return 2 if any(item["status"] == "BLOCKED" for item in records) else 0


def _live_parse_record(
    market: str,
    raw: bytes,
    parsed: pd.DataFrame,
    *,
    source: str,
) -> dict[str, object]:
    return {
        "market": market,
        "status": "PARSED",
        "source": source,
        "raw_bytes": len(raw),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "raw_persisted": False,
        "canonical_rows": len(parsed),
        "start_utc": parsed["interval_start_utc"].min().isoformat(),
        "end_utc": parsed["interval_end_utc"].max().isoformat(),
    }


def _collect_live_parser_records(
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    headers = {
        "User-Agent": "dc-energy-joint-optimization/energy-model-v3"
    }
    records: list[dict[str, object]] = []

    nyiso_url = (
        "https://mis.nyiso.com/public/csv/damlbmp/"
        "20250901damlbmp_zone_csv.zip"
    )
    nyiso_raw = requests.get(
        nyiso_url, headers=headers, timeout=args.timeout
    ).content
    with zipfile.ZipFile(io.BytesIO(nyiso_raw)) as archive:
        nyiso_source = pd.concat(
            [pd.read_csv(archive.open(name)) for name in archive.namelist()],
            ignore_index=True,
        )
    records.append(
        _live_parse_record(
            "NYISO_NYC_J",
            nyiso_raw,
            parse_nyiso_da(nyiso_source),
            source=nyiso_url,
        )
    )

    caiso_url = "https://oasis.caiso.com/oasisapi/SingleZip"
    caiso_params = {
        "queryname": "PRC_LMP",
        "version": 12,
        "market_run_id": "DAM",
        "startdatetime": "20250901T00:00-0000",
        "enddatetime": "20250901T01:00-0000",
        "node": "TH_NP15_GEN-APND",
        "resultformat": 6,
    }
    caiso_response = requests.get(
        caiso_url,
        params=caiso_params,
        headers=headers,
        timeout=args.timeout,
    )
    caiso_response.raise_for_status()
    caiso_raw = caiso_response.content
    with zipfile.ZipFile(io.BytesIO(caiso_raw)) as archive:
        caiso_source = pd.read_csv(archive.open(archive.namelist()[0]))
    records.append(
        _live_parse_record(
            "CAISO_NP15",
            caiso_raw,
            parse_caiso_da(caiso_source),
            source=caiso_response.url,
        )
    )

    ercot_discovery = requests.get(
        "https://www.ercot.com/misapp/servlets/IceDocListJsonWS",
        params={"reportTypeId": 13060},
        headers=headers,
        timeout=args.timeout,
    )
    ercot_discovery.raise_for_status()
    documents = ercot_discovery.json()["ListDocsByRptTypeRes"]["DocumentList"]
    annual = select_ercot_document(
        documents,
        friendly_name="DAMLZHBSPP_2025",
    )
    ercot_response = requests.get(
        "https://www.ercot.com/misdownload/servlets/mirDownload",
        params={"doclookupId": annual["DocID"]},
        headers=headers,
        timeout=args.timeout,
    )
    ercot_response.raise_for_status()
    ercot_raw = ercot_response.content
    with zipfile.ZipFile(io.BytesIO(ercot_raw)) as archive:
        workbook = archive.read(archive.namelist()[0])
    ercot_source = read_ercot_xlsx_sheet(workbook, sheet_number=9)
    records.append(
        {
            **_live_parse_record(
                "ERCOT_LZ_NORTH",
                ercot_raw,
                parse_ercot_da(ercot_source),
                source=ercot_response.url,
            ),
            "doc_id": annual["DocID"],
            "filename": annual["ConstructedName"],
        }
    )

    miso_url = (
        "https://docs.misoenergy.org/marketreports/"
        "20250901_da_expost_lmp.csv"
    )
    miso_response = requests.get(
        miso_url, headers=headers, timeout=args.timeout
    )
    miso_response.raise_for_status()
    miso_raw = miso_response.content
    miso_source = pd.read_csv(io.BytesIO(miso_raw), skiprows=4)
    records.append(
        _live_parse_record(
            "MISO_MINN_HUB",
            miso_raw,
            parse_miso_da(miso_source, "2025-09-01"),
            source=miso_url,
        )
    )

    spp_url = (
        "https://portal.spp.org/file-browser-api/download/"
        "da-lmp-by-settlement-location"
    )
    spp_response = requests.get(
        spp_url,
        params={
            "path": "/2025/09/By_Day/DA-LMP-SL-202509010100.csv"
        },
        headers=headers,
        timeout=args.timeout,
    )
    spp_response.raise_for_status()
    spp_raw = spp_response.content
    spp_source = pd.read_csv(io.BytesIO(spp_raw))
    records.append(
        _live_parse_record(
            "SPP_NORTH_HUB",
            spp_raw,
            parse_spp_da(spp_source),
            source=spp_response.url,
        )
    )

    records.insert(
        0,
        get_adapter("PJM_DOM").capability().to_dict(),
    )
    return records


def command_probe_parsers(args: argparse.Namespace) -> int:
    output = Path(args.output)
    generated_at = datetime.now(timezone.utc).isoformat()
    write_json(
        output,
        {
            "schema_version": "energy-model-v3",
            "generated_at_utc": generated_at,
            "status": "RUNNING",
            "records": [],
        },
    )
    try:
        records = _collect_live_parser_records(args)
    except (
        requests.RequestException,
        OSError,
        KeyError,
        IndexError,
        StopIteration,
        ET.ParseError,
        zipfile.BadZipFile,
        ContractError,
        ValueError,
    ) as exc:
        write_json(
            output,
            {
                "schema_version": "energy-model-v3",
                "generated_at_utc": generated_at,
                "status": "FAILED",
                "records": [],
                "failure": f"{type(exc).__name__}: {exc}",
            },
        )
        return 2
    write_json(
        output,
        {
            "schema_version": "energy-model-v3",
            "generated_at_utc": generated_at,
            "status": "COMPLETED_WITH_CREDENTIAL_BLOCKERS",
            "records": records,
        },
    )
    return 2


def command_probe_ercot_physical(args: argparse.Namespace) -> int:
    """Probe the public load/SCED path without retaining source bulk data."""
    headers = {
        "User-Agent": "dc-energy-joint-optimization/energy-model-v3"
    }
    session = requests.Session()
    discovery_url = (
        "https://www.ercot.com/misapp/servlets/IceDocListJsonWS"
    )
    download_url = (
        "https://www.ercot.com/misdownload/servlets/mirDownload"
    )
    retrieved_at = datetime.now(timezone.utc).isoformat()

    discovery = session.get(
        discovery_url,
        params={"reportTypeId": 13052},
        headers=headers,
        timeout=args.timeout,
    )
    discovery.raise_for_status()
    documents = discovery.json()["ListDocsByRptTypeRes"]["DocumentList"]
    selected = select_ercot_sced_document(
        documents, sced_date=args.sced_date
    )
    response = session.get(
        download_url,
        params={"doclookupId": selected["DocID"]},
        headers=headers,
        timeout=args.timeout,
    )
    response.raise_for_status()
    sced_raw = response.content
    executions = parse_ercot_sced_executions(
        read_ercot_sced_generation_zip(sced_raw)
    )
    validation_start = executions["timestamp_utc"].min().ceil("1h")
    validation_end = executions["timestamp_utc"].max().floor("1h")
    if validation_end <= validation_start:
        raise ContractError(
            "SCED disclosure lacks an interior full-hour validation window"
        )
    reconstructed = time_weight_ercot_sced_hourly(
        executions,
        start_utc=validation_start,
        end_utc=validation_end,
    )

    load_urls = {
        2025: (
            "https://www.ercot.com/files/docs/2025/02/11/"
            "Native_Load_2025.zip"
        ),
        2026: (
            "https://www.ercot.com/files/docs/2026/02/10/"
            "Native_Load_2026.zip"
        ),
    }
    load_frames: list[pd.DataFrame] = []
    load_records: list[dict[str, object]] = []
    for year, url in load_urls.items():
        load_response = session.get(
            url, headers=headers, timeout=args.timeout
        )
        load_response.raise_for_status()
        raw = load_response.content
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            names = archive.namelist()
            expected_name = f"Native_Load_{year}.xlsx"
            if names != [expected_name]:
                raise ContractError(
                    f"ERCOT load ZIP contents changed for {year}: {names}"
                )
            workbook = archive.read(expected_name)
        frame = read_ercot_native_load_xlsx(
            workbook,
            start_utc="2025-09-01T00:00:00Z",
            end_utc="2026-05-01T00:00:00Z",
        )
        load_frames.append(frame)
        load_records.append(
            {
                "year": year,
                "url": url,
                "http_last_modified": load_response.headers.get(
                    "Last-Modified"
                ),
                "http_etag": load_response.headers.get("ETag"),
                "raw_bytes": len(raw),
                "raw_sha256": hashlib.sha256(raw).hexdigest(),
                "canonical_rows_in_candidate": len(frame),
                "start_utc": frame["interval_start_utc"].min().isoformat(),
                "end_utc": frame["interval_end_utc"].max().isoformat(),
            }
        )
    load = (
        pd.concat(load_frames, ignore_index=True)
        .sort_values("interval_start_utc")
        .reset_index(drop=True)
    )
    candidate_index = pd.date_range(
        "2025-09-01T00:00:00Z",
        "2026-05-01T00:00:00Z",
        freq="1h",
        inclusive="left",
    )
    load_index = pd.DatetimeIndex(load["interval_start_utc"])
    if not load_index.equals(candidate_index):
        raise ContractError(
            "public ERCOT native-load archives do not form the exact "
            "5,808-hour candidate index"
        )

    annual_discovery = session.get(
        discovery_url,
        params={"reportTypeId": 13424},
        headers=headers,
        timeout=args.timeout,
    )
    annual_discovery.raise_for_status()
    annual_documents = annual_discovery.json()[
        "ListDocsByRptTypeRes"
    ]["DocumentList"]
    annual_document = select_ercot_document(
        annual_documents,
        friendly_name="ERCOT_2025_Hourly_WindSolar_Output",
    )
    annual_response = session.get(
        download_url,
        params={"doclookupId": annual_document["DocID"]},
        headers=headers,
        timeout=args.timeout,
    )
    annual_response.raise_for_status()
    annual_raw = annual_response.content
    annual = parse_ercot_annual_renewables_xlsx(annual_raw)
    comparison = compare_ercot_renewables(
        reconstructed,
        annual,
        start_utc=validation_start,
        end_utc=validation_end,
    )

    publication_dates = [
        pd.Timestamp(item["Document"]["PublishDate"])
        for item in documents
        if item.get("Document", {}).get("FriendlyName")
        == "60_Day_SCED_Disclosure"
        and item.get("Document", {}).get("SecurityStatus") == "P"
    ]
    operating_dates = [
        timestamp.date() - pd.Timedelta(days=60)
        for timestamp in publication_dates
    ]
    write_json(
        Path(args.output),
        {
            "schema_version": "energy-model-v3-ercot-physical-probe",
            "generated_at_utc": retrieved_at,
            "status": "PUBLIC_PRIMARY_PATH_CONFIRMED",
            "raw_persisted": False,
            "candidate_native_load": {
                "status": "EXACT_CANDIDATE_INDEX_CONFIRMED",
                "rows": len(load),
                "start_utc": load["interval_start_utc"].min().isoformat(),
                "end_utc": load["interval_end_utc"].max().isoformat(),
                "archives": load_records,
            },
            "sced_discovery": {
                "report_type_id": 13052,
                "documents": len(publication_dates),
                "earliest_operating_date": min(operating_dates).isoformat(),
                "latest_operating_date": max(operating_dates).isoformat(),
                "selection_rule": (
                    "public report 13052, publication date equals operating "
                    "date plus 60 days, newest same-day revision"
                ),
            },
            "sced_sample": {
                "operating_date": args.sced_date,
                "doc_id_observed_not_hard_coded": selected["DocID"],
                "filename": selected["ConstructedName"],
                "publish_date": selected["PublishDate"],
                "raw_bytes": len(sced_raw),
                "raw_sha256": hashlib.sha256(sced_raw).hexdigest(),
                "execution_rows": len(executions),
                "first_execution_utc": (
                    executions["timestamp_utc"].min().isoformat()
                ),
                "last_execution_utc": (
                    executions["timestamp_utc"].max().isoformat()
                ),
                "interior_hourly_rows": len(reconstructed),
                "classification": (
                    "source WIND class normalized to WGR; PVGR retained"
                ),
            },
            "report_13424_validation": {
                "doc_id_observed_not_hard_coded": annual_document["DocID"],
                "filename": annual_document["ConstructedName"],
                "raw_bytes": len(annual_raw),
                "raw_sha256": hashlib.sha256(annual_raw).hexdigest(),
                **comparison,
            },
            "retrospective_reports_forbidden": [
                13028,
                13483,
                14787,
                21809,
            ],
            "eia_bulk_sensitivity_only": {
                "url": "https://api.eia.gov/bulk/EBA.zip",
                "series": [
                    "EBA.ERCO-ALL.D.H",
                    "EBA.ERCO-ALL.NG.WND.H",
                    "EBA.ERCO-ALL.NG.SUN.H",
                ],
                "known_gap": "24-hour renewable gap around 2025-12-05/06",
                "primary_substitution": False,
            },
        },
    )
    return 0


def _download_bytes(
    session: requests.Session,
    *,
    url: str,
    destination: Path,
    headers: dict[str, str],
    timeout: int,
    params: dict[str, object] | None = None,
) -> tuple[bytes, bool]:
    if destination.is_file():
        return destination.read_bytes(), True
    response = session.get(
        url, params=params, headers=headers, timeout=timeout
    )
    response.raise_for_status()
    raw = response.content
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_bytes(raw)
    temporary.replace(destination)
    return raw, False


def _write_canonical_product(
    frame: pd.DataFrame,
    *,
    path: Path,
    source: str,
    feed: str,
    product: str,
    location: str,
) -> None:
    output = frame.copy()
    output["source"] = source
    output["feed"] = feed
    output["product"] = product
    output["location"] = location
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    output.to_csv(temporary, index=False)
    temporary.replace(path)


def _ercot_native_load_years(
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
) -> list[int]:
    first_year = start_utc.tz_convert("America/Chicago").year
    last_year = (
        end_utc - pd.Timedelta(microseconds=1)
    ).tz_convert("America/Chicago").year
    return list(range(first_year, last_year + 1))


def command_download_ercot_physical(args: argparse.Namespace) -> int:
    """Stage the exact public ERCOT physical tuple and provenance."""
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    if start.tzinfo is None or end.tzinfo is None:
        raise ContractError("ERCOT download bounds must be timezone aware")
    start = start.tz_convert("UTC")
    end = end.tz_convert("UTC")
    expected_index = pd.date_range(start, end, freq="1h", inclusive="left")
    if end <= start or start != start.floor("1h") or end != end.floor("1h"):
        raise ContractError("ERCOT download bounds must be increasing UTC hours")
    data_root = Path(args.data_root)
    raw_root = data_root / "raw" / "ERCOT_LZ_NORTH"
    native_root = data_root / "native" / "ERCOT_LZ_NORTH"
    provenance_root = (
        data_root
        / "provenance"
        / "raw_manifests"
        / "ERCOT_LZ_NORTH"
    )
    headers = {
        "User-Agent": "dc-energy-joint-optimization/energy-model-v3"
    }
    session = requests.Session()
    discovery_url = (
        "https://www.ercot.com/misapp/servlets/IceDocListJsonWS"
    )
    download_url = (
        "https://www.ercot.com/misdownload/servlets/mirDownload"
    )
    discovery = session.get(
        discovery_url,
        params={"reportTypeId": 13052},
        headers=headers,
        timeout=args.timeout,
    )
    discovery.raise_for_status()
    documents = discovery.json()["ListDocsByRptTypeRes"]["DocumentList"]
    first_operating_date = start.tz_convert("America/Chicago").date()
    last_operating_date = (
        end - pd.Timedelta(microseconds=1)
    ).tz_convert("America/Chicago").date()
    operating_dates = pd.date_range(
        first_operating_date, last_operating_date, freq="1D"
    )
    execution_frames: list[pd.DataFrame] = []
    sced_files: dict[str, str] = {}
    sced_documents: list[dict[str, object]] = []
    resumed = 0
    for operating_timestamp in operating_dates:
        operating_date = operating_timestamp.date()
        document = select_ercot_sced_document(
            documents, sced_date=operating_date
        )
        destination = (
            raw_root
            / "report_13052"
            / str(document["ConstructedName"])
        )
        raw, was_resumed = _download_bytes(
            session,
            url=download_url,
            destination=destination,
            headers=headers,
            timeout=args.timeout,
            params={"doclookupId": document["DocID"]},
        )
        resumed += int(was_resumed)
        execution_frames.append(
            parse_ercot_sced_executions(
                read_ercot_sced_generation_zip(raw)
            )
        )
        relative = str(destination.relative_to(ROOT)).replace("\\", "/")
        raw_hash = hashlib.sha256(raw).hexdigest()
        sced_files[relative] = raw_hash
        sced_documents.append(
            {
                "operating_date": operating_date.isoformat(),
                "doc_id": document["DocID"],
                "filename": document["ConstructedName"],
                "publish_date": document["PublishDate"],
                "expired_date": document.get("ExpiredDate"),
                "content_size": document.get("ContentSize"),
                "sha256": raw_hash,
            }
        )
    executions = (
        pd.concat(execution_frames, ignore_index=True)
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )
    if executions["timestamp_utc"].duplicated().any():
        raise ContractError(
            "adjacent report-13052 disclosures conflict at a SCED execution"
        )
    renewables = time_weight_ercot_sced_hourly(
        executions, start_utc=start, end_utc=end
    )
    if not pd.DatetimeIndex(
        renewables["interval_start_utc"]
    ).equals(expected_index):
        raise ContractError("SCED reconstruction does not match requested index")

    load_urls = {
        2025: (
            "https://www.ercot.com/files/docs/2025/02/11/"
            "Native_Load_2025.zip"
        ),
        2026: (
            "https://www.ercot.com/files/docs/2026/02/10/"
            "Native_Load_2026.zip"
        ),
    }
    load_frames: list[pd.DataFrame] = []
    load_files: dict[str, str] = {}
    load_archives: list[dict[str, object]] = []
    for year in _ercot_native_load_years(start, end):
        if year not in load_urls:
            raise ContractError(
                f"no locked public native-load archive URL for {year}"
            )
        url = load_urls[year]
        destination = raw_root / "native_load" / f"Native_Load_{year}.zip"
        raw, was_resumed = _download_bytes(
            session,
            url=url,
            destination=destination,
            headers=headers,
            timeout=args.timeout,
        )
        resumed += int(was_resumed)
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            expected_name = f"Native_Load_{year}.xlsx"
            if archive.namelist() != [expected_name]:
                raise ContractError(
                    f"ERCOT load ZIP contents changed for {year}"
                )
            workbook = archive.read(expected_name)
        load_frames.append(
            read_ercot_native_load_xlsx(
                workbook, start_utc=start, end_utc=end
            )
        )
        relative = str(destination.relative_to(ROOT)).replace("\\", "/")
        raw_hash = hashlib.sha256(raw).hexdigest()
        load_files[relative] = raw_hash
        load_archives.append(
            {"year": year, "url": url, "sha256": raw_hash}
        )
    load = (
        pd.concat(load_frames, ignore_index=True)
        .sort_values("interval_start_utc")
        .reset_index(drop=True)
    )
    if not pd.DatetimeIndex(load["interval_start_utc"]).equals(expected_index):
        raise ContractError(
            "native-load archives do not match requested exact index"
        )

    load_product = native_root / "ercot_native_load_archive.csv"
    renewable_product = native_root / "ercot_sced_renewables_13052.csv"
    _write_canonical_product(
        load,
        path=load_product,
        source="ERCOT",
        feed="Native_Load_YYYY.zip",
        product="ercot_native_load_archive",
        location="ERCOT system",
    )
    _write_canonical_product(
        renewables,
        path=renewable_product,
        source="ERCOT MIS",
        feed="report 13052 NP3-965-ER",
        product="ercot_sced_renewables_13052",
        location="ERCOT system WGR/PVGR resources",
    )
    physical = load.merge(
        renewables,
        on=["interval_start_utc", "interval_end_utc", "market"],
        how="inner",
        validate="one_to_one",
        suffixes=("_load", "_renewables"),
    )
    physical["source_quality_flags"] = (
        physical["source_quality_flags_load"]
        + ";"
        + physical["source_quality_flags_renewables"]
    )
    physical = derive_net_load(
        physical[
            [
                "interval_start_utc",
                "interval_end_utc",
                "market",
                "gross_demand_mw",
                "wind_mw",
                "solar_mw",
                "source_quality_flags",
            ]
        ],
        method="gross-minus-wind-solar",
    )
    if (
        physical[["gross_demand_mw", "wind_mw", "solar_mw"]] < 0
    ).any().any():
        raise ContractError(
            "ERCOT physical reconstruction has negative primary values"
        )
    physical_path = native_root / "physical_hourly.csv"
    physical_path.parent.mkdir(parents=True, exist_ok=True)
    physical.to_csv(physical_path, index=False)

    provenance_root.mkdir(parents=True, exist_ok=True)
    load_manifest = provenance_root / "ercot_native_load_archive.json"
    sced_manifest = provenance_root / "ercot_sced_renewables_13052.json"
    write_json(
        load_manifest,
        {
            "schema_version": "energy-model-v3-raw-hash-manifest",
            "product": "ercot_native_load_archive",
            "source": "ERCOT",
            "feed": "Native_Load_YYYY.zip",
            "location": "ERCOT system",
            "start_utc": start.isoformat(),
            "end_utc": end.isoformat(),
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "archives": load_archives,
            "files": load_files,
        },
    )
    write_json(
        sced_manifest,
        {
            "schema_version": "energy-model-v3-raw-hash-manifest",
            "product": "ercot_sced_renewables_13052",
            "source": "ERCOT MIS",
            "feed": "report 13052 NP3-965-ER",
            "location": "ERCOT system WGR/PVGR resources",
            "start_utc": start.isoformat(),
            "end_utc": end.isoformat(),
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "selection_rule": (
                "report 13052 public ZIP published operating_date + 60 days; "
                "newest same-day revision"
            ),
            "documents": sced_documents,
            "files": sced_files,
        },
    )

    validation: dict[str, object] = {
        "status": "NOT_REQUESTED_OUTSIDE_2025"
    }
    if start.year <= 2025 and end > pd.Timestamp("2025-09-01T00:00:00Z"):
        annual_discovery = session.get(
            discovery_url,
            params={"reportTypeId": 13424},
            headers=headers,
            timeout=args.timeout,
        )
        annual_discovery.raise_for_status()
        annual_document = select_ercot_document(
            annual_discovery.json()["ListDocsByRptTypeRes"]["DocumentList"],
            friendly_name="ERCOT_2025_Hourly_WindSolar_Output",
        )
        annual_path = (
            raw_root
            / "report_13424_validation"
            / str(annual_document["ConstructedName"])
        )
        annual_raw, was_resumed = _download_bytes(
            session,
            url=download_url,
            destination=annual_path,
            headers=headers,
            timeout=args.timeout,
            params={"doclookupId": annual_document["DocID"]},
        )
        resumed += int(was_resumed)
        annual = parse_ercot_annual_renewables_xlsx(annual_raw)
        validation_start = max(start, pd.Timestamp("2025-09-01T00:00:00Z"))
        validation_end = min(end, pd.Timestamp("2026-01-01T00:00:00Z"))
        validation = {
            "doc_id": annual_document["DocID"],
            "filename": annual_document["ConstructedName"],
            "raw_sha256": hashlib.sha256(annual_raw).hexdigest(),
            **compare_ercot_renewables(
                renewables,
                annual,
                start_utc=validation_start,
                end_utc=validation_end,
            ),
        }
        if (
            validation["missing_sced_hours"] != 0
            or validation["missing_annual_hours"] != 0
        ):
            raise ContractError(
                "report 13424 validation does not cover the exact requested "
                "2025 validation index"
            )
    write_json(
        Path(args.output),
        {
            "schema_version": "energy-model-v3-ercot-physical-download",
            "status": "COMPLETE",
            "start_utc": start.isoformat(),
            "end_utc": end.isoformat(),
            "hourly_rows": len(physical),
            "sced_documents": len(sced_documents),
            "resumed_files": resumed,
            "canonical_files": {
                str(load_product.relative_to(ROOT)).replace("\\", "/"):
                    sha256(load_product),
                str(renewable_product.relative_to(ROOT)).replace("\\", "/"):
                    sha256(renewable_product),
                str(physical_path.relative_to(ROOT)).replace("\\", "/"):
                    sha256(physical_path),
            },
            "raw_manifests": {
                str(load_manifest.relative_to(ROOT)).replace("\\", "/"):
                    sha256(load_manifest),
                str(sced_manifest.relative_to(ROOT)).replace("\\", "/"):
                    sha256(sced_manifest),
            },
            "report_13424_validation": validation,
        },
    )
    return 0


def command_preflight(args: argparse.Namespace) -> int:
    capability_path = Path(args.capabilities)
    capabilities = (
        json.loads(capability_path.read_text(encoding="utf-8"))
        if capability_path.exists()
        else [
            get_adapter(market).capability().to_dict() for market in MARKETS
        ]
    )
    coverage_path = Path(args.coverage_evidence)
    if not coverage_path.exists():
        raise ContractError(f"coverage evidence is missing: {coverage_path}")
    report = blocked_preflight(
        Path(args.data_root),
        capabilities,
        coverage_evidence_path=coverage_path,
        repository_root=ROOT,
        eia_api_key_available=bool(os.environ.get("EIA_API_KEY")),
        ercot_eia_fallback_activated=args.enable_eia_ercot_fallback,
    )
    write_json(Path(args.output), report)
    return 2 if report["status"] == "BLOCKED" else 0


def command_fixture_diagnostics(args: argparse.Namespace) -> int:
    calendar = make_calendar()
    frame = fixture_market_frame("CAISO_NP15", calendar=calendar, seed=303)
    scale = calibrate_scale(
        frame,
        market="CAISO_NP15",
        train_start=pd.Timestamp(calendar.train_start_utc),
        train_end=pd.Timestamp(calendar.train_end_utc),
    )
    hour = np.arange(len(frame))
    dc_power = pd.Series(75.0 + 10.0 * np.sin(2 * np.pi * hour / 24))
    panel = add_grid_features(frame, dc_power, scale)
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    plot_market_diagnostics(panel, output / "fixture_quality_diagnostics.png")
    write_json(
        output / "fixture_diagnostics.json",
        {
            "schema_version": "energy-model-v3",
            "status": "SYNTHETIC_FIXTURE_ONLY_NOT_MARKET_EVIDENCE",
            "market_label": "CAISO_NP15 fixture schema",
            "calendar": calendar.to_dict(),
            "scale": scale.to_dict(),
            "diagnostics": diagnostic_summary(panel),
            "plot": "fixture_quality_diagnostics.png",
        },
    )
    return 0


def command_manifest(args: argparse.Namespace) -> int:
    paths = [
        ROOT / "env" / "protocols" / "independent_us_v3.yaml",
        ROOT / "env" / "scenarios" / "us_six_market_v3_primary.yaml",
        ROOT / "env" / "scenarios" / "us_six_market_v3_robustness.yaml",
        ROOT / "env" / "scenarios" / "us_six_market_v3_stress_6gw.yaml",
        DEFAULT_DATA / "README.md",
        DEFAULT_DATA / "scale-config.yaml",
        DEFAULT_DATA / "provenance" / "source_contract.json",
        DEFAULT_DATA / "provenance" / "forecast_capabilities.json",
        DEFAULT_DATA / "provenance" / "coverage_evidence.json",
        DEFAULT_DATA / "provenance" / "workload-power-manifest.json",
        DEFAULT_DATA / "provenance" / "live-acquisition-manifest.json",
        DEFAULT_OUTPUT / "capabilities.json",
        DEFAULT_OUTPUT / "sample-download-manifest.json",
        DEFAULT_OUTPUT / "live-parser-probes.json",
        DEFAULT_OUTPUT / "ercot-physical-probe.json",
        DEFAULT_OUTPUT / "blocked-capability-report.json",
        DEFAULT_OUTPUT / "pjm-blocked-report.json",
        DEFAULT_OUTPUT / "live-campaign-summary.json",
        DEFAULT_OUTPUT / "fixture-only" / "fixture_diagnostics.json",
        DEFAULT_OUTPUT / "fixture-only" / "fixture_quality_diagnostics.png",
    ]
    paths.extend(
        DEFAULT_OUTPUT / "live-diagnostics" / f"{market}.png"
        for market in MARKETS
    )
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.exists()]
    if missing:
        raise ContractError(f"manifest inputs missing: {', '.join(missing)}")
    blocked_report = json.loads(
        (DEFAULT_OUTPUT / "blocked-capability-report.json").read_text(
            encoding="utf-8"
        )
    )
    coverage_path = DEFAULT_DATA / "provenance" / "coverage_evidence.json"
    assessed_coverage_hash = (
        blocked_report.get("coverage_assessment", {}).get("evidence_sha256")
    )
    current_coverage_hash = sha256(coverage_path)
    if assessed_coverage_hash != current_coverage_hash:
        raise ContractError(
            "blocked report is stale relative to coverage evidence"
        )
    current_coverage_evidence = json.loads(
        coverage_path.read_text(encoding="utf-8")
    )
    current_coverage_assessment = assess_candidate_coverage(
        current_coverage_evidence,
        evidence_sha256=current_coverage_hash,
        repository_root=ROOT,
        eia_api_key_available=bool(os.environ.get("EIA_API_KEY")),
        ercot_eia_fallback_activated=bool(
            blocked_report.get("coverage_assessment", {}).get(
                "ercot_eia_fallback_activated"
            )
        ),
    )
    previous_assessment = blocked_report.get("coverage_assessment", {})
    for field in ("status", "blockers", "markets", "evidence_sha256"):
        if previous_assessment.get(field) != current_coverage_assessment.get(field):
            raise ContractError(
                "blocked report coverage assessment is stale or inconsistent"
            )
    capabilities_path = DEFAULT_OUTPUT / "capabilities.json"
    current_capabilities = json.loads(
        capabilities_path.read_text(encoding="utf-8")
    )
    if blocked_report.get("capabilities") != current_capabilities:
        raise ContractError(
            "blocked report is stale relative to capability artifact"
        )
    current_preflight = blocked_preflight(
        DEFAULT_DATA,
        current_capabilities,
        coverage_evidence_path=coverage_path,
        repository_root=ROOT,
        eia_api_key_available=bool(os.environ.get("EIA_API_KEY")),
        ercot_eia_fallback_activated=bool(
            current_coverage_assessment.get(
                "ercot_eia_fallback_activated"
            )
        ),
    )
    for field in (
        "status",
        "missing_primary_inputs",
        "invalid_primary_inputs",
        "capability_failures",
        "coverage_failures",
        "validated_primary_files",
    ):
        if blocked_report.get(field) != current_preflight.get(field):
            raise ContractError(
                "blocked report is stale relative to current preflight"
            )
    for relative, expected_hash in current_coverage_assessment.get(
        "validated_files", {}
    ).items():
        proof_path = ROOT / relative
        if not proof_path.is_file() or sha256(proof_path) != expected_hash:
            raise ContractError(f"coverage proof changed: {relative}")
        paths.append(proof_path)
    for relative, expected_hash in current_preflight.get(
        "validated_primary_files", {}
    ).items():
        input_path = ROOT / relative
        if not input_path.is_file() or sha256(input_path) != expected_hash:
            raise ContractError(f"primary input changed: {relative}")
        paths.append(input_path)
    blocked_report = current_preflight
    live_panel_manifest = DEFAULT_OUTPUT / "live-panel" / "manifest.json"
    ready = blocked_report["status"] == "READY"
    panel_error: str | None = None
    panel_paths: list[Path] = []
    if ready:
        try:
            panel = json.loads(live_panel_manifest.read_text(encoding="utf-8"))
            if panel.get("schema_version") != "energy-model-v3":
                raise ContractError("panel schema version is invalid")
            if panel.get("fixture_mode") is not False:
                raise ContractError("panel is marked as fixture data")
            if panel.get("market_order") != list(MARKETS):
                raise ContractError("panel market order is invalid")
            if panel.get("exact_common_index") is not True:
                raise ContractError("panel common-index claim is absent")
            if panel.get("inputs") != current_preflight.get(
                "validated_primary_files"
            ):
                raise ContractError(
                    "panel inputs do not match current preflight inputs"
                )
            primary_scenario = (
                ROOT / "env" / "scenarios" / "us_six_market_v3_primary.yaml"
            )
            if panel.get("scenario_sha256") != sha256(primary_scenario):
                raise ContractError(
                    "panel scenario is not the locked primary scenario"
                )
            outputs = panel.get("outputs")
            if not isinstance(outputs, dict) or set(outputs) != {
                f"{market}.csv" for market in MARKETS
            }:
                raise ContractError("panel output set is invalid")
            for filename, expected_hash in outputs.items():
                panel_path = live_panel_manifest.parent / filename
                if not panel_path.exists() or sha256(panel_path) != expected_hash:
                    raise ContractError(
                        f"panel output hash mismatch: {filename}"
                    )
                panel_paths.append(panel_path)
            panel_paths.append(live_panel_manifest)
        except (OSError, json.JSONDecodeError, ContractError) as exc:
            panel_error = str(exc)
    live_panel_persisted = ready and panel_error is None and bool(panel_paths)
    paths.extend(panel_paths)
    files = {
        str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
        for path in paths
    }
    if live_panel_persisted:
        reason = "Validated all-six live panel manifest is present."
    elif ready:
        reason = f"Preflight is ready, but live panel validation failed: {panel_error}"
    else:
        failures = blocked_report.get("capability_failures", {})
        coverage_failures = blocked_report.get("coverage_failures", {})
        missing_inputs = blocked_report.get("missing_primary_inputs", {})
        invalid_inputs = blocked_report.get("invalid_primary_inputs", {})
        combined = {
            **{
                f"capability:{market}": message
                for market, message in failures.items()
            },
            **{
                f"coverage:{market}": message
                for market, message in coverage_failures.items()
            },
            **{
                f"missing:{market}": f"{len(paths)} primary files absent"
                for market, paths in missing_inputs.items()
            },
            **{
                f"invalid:{market}": message
                for market, message in invalid_inputs.items()
            },
        }
        reason = (
            "All-six primary calendar is blocked: "
            + "; ".join(
                f"{market}: {message}"
                for market, message in sorted(combined.items())
            )
        )
    write_json(
        Path(args.output),
        {
            "schema_version": "energy-model-v3",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "study_status": blocked_report["status"],
            "live_panel_persisted": live_panel_persisted,
            "reason": reason,
            "files": files,
        },
    )
    return 0


def command_build(args: argparse.Namespace) -> int:
    scenario = Path(args.scenario)
    calendar = make_calendar()
    index = pd.date_range(
        calendar.start_utc, calendar.end_utc, freq="1h", inclusive="left"
    )
    dc_power: dict[str, pd.Series] = {}
    scenario_data = __import__("yaml").safe_load(
        scenario.read_text(encoding="utf-8")
    )
    for site in scenario_data["sites"]:
        power_path = (
            Path(args.data_root)
            / "modeled_dc_power"
            / f"{site['market']}.csv"
        )
        if not power_path.exists():
            raise ContractError(f"missing modeled DC power: {power_path}")
        power = pd.read_csv(power_path)
        timestamps = pd.DatetimeIndex(
            pd.to_datetime(power["interval_start_utc"], utc=True)
        )
        if not timestamps.equals(index):
            raise ContractError(
                f"{site['market']} modeled DC power index is not exact"
            )
        dc_power[site["market"]] = pd.to_numeric(
            power["dc_power_mw"], errors="raise"
        )
    build_panel(
        data_root=Path(args.data_root),
        output_root=Path(args.output_root),
        scenario_path=scenario,
        dc_power=dc_power,
        calendar=calendar,
        fixture_mode=False,
    )
    return 0


def command_acquire_live(args: argparse.Namespace) -> int:
    calendar = make_calendar()
    data_root = Path(args.data_root)
    raw_root = data_root / "raw"
    output_root = Path(args.output_root)
    eia_path = raw_root / "EBA.zip"
    eia_metadata = download_raw(
        "https://api.eia.gov/bulk/EBA.zip",
        eia_path,
        timeout=args.timeout,
        retries=6,
    ) if not eia_path.exists() else {
        "url": "https://api.eia.gov/bulk/EBA.zip",
        "bytes": eia_path.stat().st_size,
        "sha256": acquisition_sha256(eia_path),
        "cache_hit": True,
    }
    physical, physical_metadata = acquire_physical(
        raw_root, calendar=calendar
    )
    prices, price_metadata = acquire_prices(raw_root, calendar=calendar)
    native_hashes = write_native_inputs(data_root, physical, prices)
    forecast_manifest = write_forecasts(
        data_root / "forecasts", physical, calendar=calendar
    )
    capabilities = [
        {
            "market": market,
            "status": "AVAILABLE",
            "capability": "historical_primary_tuple",
            "reason": "authoritative no-key price and physical inputs staged",
        }
        for market in MARKETS
    ]
    write_json(DEFAULT_OUTPUT / "capabilities.json", capabilities)
    blocked = [
        get_adapter(market).capability().to_dict()
        for market in BLOCKED_MARKETS
    ]
    write_json(DEFAULT_OUTPUT / "pjm-blocked-report.json", {
        "status": "BLOCKED_EXCLUDED_FROM_LIVE_ROSTER",
        "prominent_notice": (
            "PJM DOM / Northern Virginia was not evaluated and is excluded "
            "from all live results and claims until PJM_API_KEY is supplied."
        ),
        "capabilities": blocked,
    })
    preflight = blocked_preflight(data_root, capabilities, calendar)
    write_json(DEFAULT_OUTPUT / "blocked-capability-report.json", preflight)
    if preflight["status"] != "READY":
        raise ContractError(
            "live acquisition staged inputs but preflight is blocked: "
            f"{preflight}"
        )
    dc_power, workload_power_manifest = build_measured_dc_power(
        data_root, DEFAULT_SCENARIO, calendar=calendar
    )
    panel_manifest = build_panel(
        data_root=data_root,
        output_root=output_root / "live-panel",
        scenario_path=DEFAULT_SCENARIO,
        dc_power=dc_power,
        calendar=calendar,
        fixture_mode=False,
    )
    for market in MARKETS:
        panel = pd.read_csv(output_root / "live-panel" / f"{market}.csv")
        plot_market_diagnostics(
            panel, output_root / "live-diagnostics" / f"{market}.png"
        )
    acquisition_manifest = {
        "schema_version": "energy-model-v3",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "calendar": calendar.to_dict(),
        "live_roster": list(MARKETS),
        "PJM": {
            "status": "BLOCKED_EXCLUDED",
            "evaluated": False,
            "required_credential": "PJM_API_KEY",
        },
        "eia_bulk": eia_metadata,
        "physical_sources": physical_metadata,
        "price_sources": price_metadata,
        "native_inputs": native_hashes,
        "forecast_manifest": forecast_manifest,
        "workload_power_manifest": workload_power_manifest,
        "panel_manifest": panel_manifest,
        "raw_redistribution": "local_ignored",
        "interpolation": False,
        "synthetic_market_data": False,
    }
    write_json(
        data_root / "provenance" / "live-acquisition-manifest.json",
        acquisition_manifest,
    )
    acquisition_manifest_path = (
        data_root / "provenance" / "live-acquisition-manifest.json"
    )
    raw_files = [path for path in raw_root.rglob("*") if path.is_file()]
    panel_files = [
        output_root / "live-panel" / filename
        for filename in panel_manifest["outputs"]
    ]
    panel_files.append(output_root / "live-panel" / "manifest.json")
    summary = {
        "schema_version": "energy-model-v3",
        "status": "READY",
        "prominent_notice": (
            "PJM DOM / Northern Virginia was not evaluated. PJM remains "
            "BLOCKED and excluded until PJM_API_KEY is provided."
        ),
        "calendar": calendar.to_dict(),
        "markets": list(MARKETS),
        "rows_per_market": calendar.rows,
        "total_market_rows": calendar.rows * len(MARKETS),
        "native_inputs": native_hashes,
        "scales": panel_manifest["scales"],
        "diagnostics": panel_manifest["diagnostics"],
        "forecasts": {
            market: forecast_manifest["markets"][market]["errors"]
            for market in MARKETS
        },
        "workload_power": workload_power_manifest,
        "raw_evidence": {
            "path": str(raw_root.resolve()),
            "files": len(raw_files),
            "total_bytes": sum(path.stat().st_size for path in raw_files),
            "acquisition_manifest_sha256": acquisition_sha256(
                acquisition_manifest_path
            ),
        },
        "local_panel": {
            "path": str((output_root / "live-panel").resolve()),
            "files": panel_manifest["outputs"],
            "total_bytes": sum(path.stat().st_size for path in panel_files),
            "manifest_sha256": acquisition_sha256(
                output_root / "live-panel" / "manifest.json"
            ),
        },
    }
    write_json(output_root / "live-campaign-summary.json", summary)
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Independent six-market US energy model v3"
    )
    commands = result.add_subparsers(dest="command", required=True)
    describe = commands.add_parser("describe-sources")
    describe.add_argument(
        "--output",
        default=str(DEFAULT_DATA / "provenance" / "source_contract.json"),
    )
    describe.set_defaults(func=command_describe)
    probe = commands.add_parser("probe")
    probe.add_argument("--timeout", type=int, default=30)
    probe.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT / "capabilities.json"),
    )
    probe.set_defaults(func=command_probe)
    samples = commands.add_parser("download-samples")
    samples.add_argument("--timeout", type=int, default=120)
    samples.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT / "sample-download-manifest.json"),
    )
    samples.set_defaults(func=command_download_samples)
    parser_probes = commands.add_parser("probe-parsers")
    parser_probes.add_argument("--timeout", type=int, default=180)
    parser_probes.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT / "live-parser-probes.json"),
    )
    parser_probes.set_defaults(func=command_probe_parsers)
    ercot_physical = commands.add_parser("probe-ercot-physical")
    ercot_physical.add_argument("--timeout", type=int, default=300)
    ercot_physical.add_argument(
        "--sced-date",
        default="2025-09-01",
        help="SCED operating date used for the live schema/weighting probe",
    )
    ercot_physical.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT / "ercot-physical-probe.json"),
    )
    ercot_physical.set_defaults(func=command_probe_ercot_physical)
    ercot_download = commands.add_parser("download-ercot-physical")
    ercot_download.add_argument("--timeout", type=int, default=300)
    ercot_download.add_argument(
        "--start", default="2025-09-01T00:00:00Z"
    )
    ercot_download.add_argument(
        "--end", default="2026-05-01T00:00:00Z"
    )
    ercot_download.add_argument("--data-root", default=str(DEFAULT_DATA))
    ercot_download.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT / "ercot-physical-download.json"),
    )
    ercot_download.set_defaults(func=command_download_ercot_physical)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--data-root", default=str(DEFAULT_DATA))
    preflight.add_argument(
        "--capabilities",
        default=str(DEFAULT_OUTPUT / "capabilities.json"),
    )
    preflight.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT / "blocked-capability-report.json"),
    )
    preflight.add_argument(
        "--coverage-evidence",
        default=str(
            DEFAULT_DATA / "provenance" / "coverage_evidence.json"
        ),
    )
    preflight.add_argument(
        "--enable-eia-ercot-fallback",
        action="store_true",
        help=(
            "Record explicit ERCO EIA-930 sensitivity intent; this can never "
            "clear the primary native-load/SCED coverage gate"
        ),
    )
    preflight.set_defaults(func=command_preflight)
    fixture = commands.add_parser("fixture-diagnostics")
    fixture.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT / "fixture-only"),
    )
    fixture.set_defaults(func=command_fixture_diagnostics)
    manifest = commands.add_parser("manifest")
    manifest.add_argument(
        "--output",
        default=str(DEFAULT_DATA / "manifest.json"),
    )
    manifest.set_defaults(func=command_manifest)
    build = commands.add_parser("build")
    build.add_argument("--data-root", default=str(DEFAULT_DATA))
    build.add_argument("--output-root", default=str(DEFAULT_OUTPUT / "panel"))
    build.add_argument("--scenario", default=str(DEFAULT_SCENARIO))
    build.set_defaults(func=command_build)
    acquire = commands.add_parser("acquire-live")
    acquire.add_argument("--data-root", default=str(DEFAULT_DATA))
    acquire.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    acquire.add_argument("--timeout", type=int, default=600)
    acquire.set_defaults(func=command_acquire_live)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return int(args.func(args))
    except ContractError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
