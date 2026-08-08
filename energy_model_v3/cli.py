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
    get_adapter,
    parse_caiso_da,
    parse_ercot_da,
    parse_miso_da,
    parse_nyiso_da,
    parse_spp_da,
    read_ercot_xlsx_sheet,
    select_ercot_document,
)
from .adapters.base import download_raw
from .builder import (
    MARKETS,
    blocked_preflight,
    build_panel,
    fixture_market_frame,
    sha256,
)
from .calendar import make_calendar
from .contract import ContractError
from .coverage import assess_candidate_coverage
from .derivations import (
    add_grid_features,
    calibrate_scale,
    diagnostic_summary,
)
from .diagnostics import plot_market_diagnostics

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data" / "energy_model_v3"
DEFAULT_OUTPUT = ROOT / "output" / "energy_model_v3"
DEFAULT_SCENARIO = (
    ROOT / "env" / "scenarios" / "us_six_market_v3_primary.yaml"
)


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
        DEFAULT_OUTPUT / "capabilities.json",
        DEFAULT_OUTPUT / "sample-download-manifest.json",
        DEFAULT_OUTPUT / "live-parser-probes.json",
        DEFAULT_OUTPUT / "blocked-capability-report.json",
        DEFAULT_OUTPUT / "fixture-only" / "fixture_diagnostics.json",
        DEFAULT_OUTPUT / "fixture-only" / "fixture_quality_diagnostics.png",
    ]
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
    live_panel_manifest = DEFAULT_OUTPUT / "panel" / "manifest.json"
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
            "Explicitly enable same-BA EIA-930 retrieval for the ERCOT "
            "physical gap; also requires EIA_API_KEY"
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
