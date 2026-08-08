"""Candidate-calendar coverage evidence and explicit fallback gating."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from .builder import MARKETS
from .contract import ContractError

TRUSTED_PRODUCT_CONTRACTS: dict[str, dict[str, dict[str, Any]]] = {
    "PJM_DOM": {
        "pjm_dom_da_lmp": {
            "source_type": "authoritative_native",
            "source": "PJM Data Miner 2", "feed": "rt_da_monthly_lmps",
            "location": "DOM pnode 34964545", "cadence_minutes": 60,
            "data_columns": ["da_lmp_usd_mwh"],
        },
        "pjm_rto_physical": {
            "source_type": "authoritative_native",
            "source": "PJM Data Miner 2", "feed": "PJM RTO load/wind/solar",
            "location": "PJM RTO", "cadence_minutes": 60,
            "data_columns": ["gross_demand_mw", "wind_mw", "solar_mw", "net_load_mw"],
        },
    },
    "NYISO_NYC_J": {
        "nyiso_zone_j_dam": {
            "source_type": "authoritative_native",
            "source": "NYISO MIS", "feed": "damlbmp",
            "location": "Zone J PTID 61761", "cadence_minutes": 60,
            "data_columns": ["da_lmp_usd_mwh"],
        },
        "nyiso_zone_j_pal": {
            "source_type": "authoritative_native",
            "source": "NYISO MIS", "feed": "pal",
            "location": "Zone J PTID 61761", "cadence_minutes": 60,
            "data_columns": ["gross_demand_mw", "net_load_mw"],
        },
        "nyiso_system_fuelmix": {
            "source_type": "authoritative_native",
            "source": "NYISO MIS", "feed": "rtfuelmix",
            "location": "NYCA system", "cadence_minutes": 60,
            "data_columns": ["wind_mw", "solar_mw"],
        },
    },
    "CAISO_NP15": {
        "caiso_np15_dam": {
            "source_type": "authoritative_native",
            "source": "CAISO OASIS", "feed": "PRC_LMP v12",
            "location": "TH_NP15_GEN-APND", "cadence_minutes": 60,
            "data_columns": ["da_lmp_usd_mwh"],
        },
        "caiso_outlook_netdemand": {
            "source_type": "authoritative_native",
            "source": "CAISO Today's Outlook", "feed": "netdemand.csv",
            "location": "CAISO system", "cadence_minutes": 5,
            "data_columns": ["gross_demand_mw", "net_load_mw"],
        },
        "caiso_outlook_fuelsource": {
            "source_type": "authoritative_native",
            "source": "CAISO Today's Outlook", "feed": "fuelsource.csv",
            "location": "CAISO system", "cadence_minutes": 5,
            "data_columns": ["wind_mw", "solar_mw"],
        },
    },
    "ERCOT_LZ_NORTH": {
        "ercot_lz_north_dam": {
            "source_type": "authoritative_native",
            "source": "ERCOT MIS", "feed": "report 13060",
            "location": "LZ_NORTH", "cadence_minutes": 60,
            "data_columns": ["da_lmp_usd_mwh"],
        },
        "ercot_load_13101": {
            "source_type": "authoritative_native",
            "source": "ERCOT MIS", "feed": "report 13101",
            "location": "ERCOT system", "cadence_minutes": 60,
            "data_columns": ["gross_demand_mw"],
        },
        "ercot_renewables_13424": {
            "source_type": "authoritative_native",
            "source": "ERCOT MIS", "feed": "report 13424",
            "location": "ERCOT system", "cadence_minutes": 60,
            "data_columns": ["wind_mw", "solar_mw"],
        },
        "ercot_eia930_physical": {
            "source_type": "EIA-930",
            "source": "EIA-930", "feed": "ERCO balancing authority",
            "location": "ERCO", "cadence_minutes": 60,
            "data_columns": ["gross_demand_mw", "wind_mw", "solar_mw", "net_load_mw"],
        },
    },
    "MISO_MINN_HUB": {
        "miso_minn_hub_dam": {
            "source_type": "authoritative_native",
            "source": "MISO Market Reports", "feed": "da_expost_lmp",
            "location": "MINN.HUB", "cadence_minutes": 60,
            "data_columns": ["da_lmp_usd_mwh"],
        },
        "miso_system_physical": {
            "source_type": "authoritative_native",
            "source": "MISO Data Exchange", "feed": "historical physical tuple",
            "location": "MISO system", "cadence_minutes": 60,
            "data_columns": ["gross_demand_mw", "wind_mw", "solar_mw", "net_load_mw"],
        },
    },
    "SPP_NORTH_HUB": {
        "spp_north_dam": {
            "source_type": "authoritative_native",
            "source": "SPP Marketplace Public Data",
            "feed": "da-lmp-by-settlement-location",
            "location": "SPPNORTH_HUB", "cadence_minutes": 60,
            "data_columns": ["da_lmp_usd_mwh"],
        },
        "spp_genmix365": {
            "source_type": "authoritative_native",
            "source": "SPP Marketplace Public Data", "feed": "GenMix365_SPP.csv",
            "location": "SPP balancing authority", "cadence_minutes": 5,
            "data_columns": ["gross_demand_mw", "wind_mw", "solar_mw", "net_load_mw"],
        },
    },
}


def assess_candidate_coverage(
    evidence: dict[str, Any],
    *,
    evidence_sha256: str,
    repository_root: Path,
    eia_api_key_available: bool,
    ercot_eia_fallback_activated: bool,
) -> dict[str, Any]:
    if evidence.get("schema_version") != "energy-model-v3-coverage":
        raise ContractError("coverage evidence schema is invalid")
    markets = evidence.get("markets")
    if not isinstance(markets, dict) or set(markets) != set(MARKETS):
        raise ContractError("coverage evidence must describe exactly six markets")
    results: dict[str, dict[str, Any]] = {}
    blockers: dict[str, str] = {}
    validated_files: dict[str, str] = {}
    candidate_start = pd.Timestamp(evidence["candidate_start_utc"])
    candidate_end = pd.Timestamp(evidence["candidate_end_utc"])
    if pd.isna(candidate_start) or pd.isna(candidate_end):
        raise ContractError("coverage evidence candidate timestamps are missing")
    if candidate_end <= candidate_start:
        raise ContractError("coverage evidence candidate interval is invalid")
    if not re.fullmatch(r"[0-9a-f]{64}", evidence_sha256):
        raise ContractError("coverage evidence SHA-256 is invalid")
    for market in MARKETS:
        record = markets[market]
        complete = record.get("candidate_complete")
        if not isinstance(complete, bool):
            raise ContractError(
                f"{market} candidate_complete must be an explicit boolean"
            )
        if complete:
            required_products = record.get("required_products")
            proof = record.get("coverage_proof", {})
            products = proof.get("products")
            if (
                not isinstance(required_products, list)
                or not required_products
                or not isinstance(products, list)
            ):
                raise ContractError(
                    f"{market} complete coverage lacks product proof"
                )
            contracts = TRUSTED_PRODUCT_CONTRACTS[market]
            expected_names = set(contracts)
            if market == "ERCOT_LZ_NORTH" and "ercot_eia930_physical" in set(
                required_products or []
            ):
                expected_names = {
                    "ercot_lz_north_dam",
                    "ercot_eia930_physical",
                }
            if set(required_products or []) != expected_names:
                raise ContractError(
                    f"{market} required product set is not trusted"
                )
            proved_names = {item.get("name") for item in products}
            if proved_names != expected_names:
                raise ContractError(
                    f"{market} coverage proof does not match required products"
                )
            for product in products:
                trusted = contracts[product["name"]]
                source_type = product.get("source_type")
                if source_type != trusted["source_type"]:
                    raise ContractError(
                        f"{market} {product['name']} source type mismatch"
                    )
                if trusted["source_type"] == "EIA-930":
                    fallback = record.get("eia_930_fallback", {})
                    if market != "ERCOT_LZ_NORTH":
                        raise ContractError("EIA-930 fallback is ERCOT-only")
                    if not ercot_eia_fallback_activated:
                        raise ContractError(
                            "ERCOT EIA-930 proof requires explicit activation"
                        )
                    if not eia_api_key_available:
                        raise ContractError(
                            "ERCOT EIA-930 proof requires EIA_API_KEY"
                        )
                    if (
                        not fallback.get("same_balancing_authority_only")
                        or product.get("balancing_authority") != "ERCO"
                    ):
                        raise ContractError(
                            "ERCOT EIA-930 proof must be same-BA ERCO"
                        )
                start = pd.Timestamp(product["start_utc"])
                end = pd.Timestamp(product["end_utc"])
                if pd.isna(start) or pd.isna(end):
                    raise ContractError(
                        f"{market} {product['name']} coverage timestamps are missing"
                    )
                if start > candidate_start or end < candidate_end:
                    raise ContractError(
                        f"{market} {product['name']} does not span candidate"
                    )
                if product.get("missing_intervals") != 0:
                    raise ContractError(
                        f"{market} {product['name']} has missing intervals"
                    )
                if product.get("duplicate_intervals") != 0:
                    raise ContractError(
                        f"{market} {product['name']} has duplicate intervals"
                    )
                validated_files.update(_validate_raw_hash_manifest(
                    product.get("raw_hash_manifest"),
                    repository_root=repository_root,
                    market=market,
                    product_name=product["name"],
                    trusted=trusted,
                    candidate_start=candidate_start,
                    candidate_end=candidate_end,
                ))
                validated_files.update(_validate_canonical_coverage_file(
                    product.get("canonical_file"),
                    repository_root=repository_root,
                    market=market,
                    product=product,
                    trusted=trusted,
                    candidate_start=candidate_start,
                    candidate_end=candidate_end,
                ))
            results[market] = {
                "status": "CANDIDATE_COVERAGE_VERIFIED",
                "reason": record["reason"],
            }
            continue
        reason = str(record.get("reason", "candidate coverage not proven"))
        if market == "ERCOT_LZ_NORTH" and ercot_eia_fallback_activated:
            fallback = record.get("eia_930_fallback", {})
            if not fallback.get("same_balancing_authority_only"):
                raise ContractError("ERCOT EIA fallback is not same-BA constrained")
            if not eia_api_key_available:
                reason = (
                    "ERCOT EIA-930 fallback explicitly activated but "
                    "EIA_API_KEY is absent"
                )
            else:
                results[market] = {
                    "status": "FALLBACK_READY_TO_RETRIEVE",
                    "reason": (
                        "Explicit same-BA EIA-930 fallback is keyed; canonical "
                        "inputs must still be retrieved and validated"
                    ),
                }
                reason = (
                    "ERCOT EIA-930 fallback is ready to retrieve but no exact "
                    "ERCO candidate panel with source provenance is staged"
                )
        blockers[market] = reason
        results[market] = {"status": "BLOCKED", "reason": reason}
    return {
        "schema_version": "energy-model-v3-coverage-assessment",
        "assessed_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_start_utc": evidence["candidate_start_utc"],
        "candidate_end_utc": evidence["candidate_end_utc"],
        "evidence_sha256": evidence_sha256,
        "ercot_eia_fallback_activated": ercot_eia_fallback_activated,
        "eia_api_key_available": eia_api_key_available,
        "markets": results,
        "blockers": blockers,
        "validated_files": validated_files,
        "status": "BLOCKED" if blockers else "COVERAGE_VERIFIED",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_raw_hash_manifest(
    reference: Any,
    *,
    repository_root: Path,
    market: str,
    product_name: str,
    trusted: dict[str, Any],
    candidate_start: pd.Timestamp,
    candidate_end: pd.Timestamp,
) -> dict[str, str]:
    if not isinstance(reference, dict):
        raise ContractError(f"{product_name} raw hash manifest reference is invalid")
    relative = reference.get("path")
    expected_manifest_hash = reference.get("sha256")
    if (
        not isinstance(relative, str)
        or not re.fullmatch(r"[0-9a-f]{64}", str(expected_manifest_hash))
    ):
        raise ContractError(f"{product_name} raw hash manifest reference is invalid")
    manifest_path = (repository_root / relative).resolve()
    root = repository_root.resolve()
    expected_manifest_parent = (
        root / "data" / "energy_model_v3" / "provenance"
        / "raw_manifests" / market
    ).resolve()
    if (
        expected_manifest_parent != manifest_path.parent
        and expected_manifest_parent not in manifest_path.parents
    ):
        raise ContractError(f"{product_name} raw hash manifest path is invalid")
    if not manifest_path.is_file() or _sha256(manifest_path) != expected_manifest_hash:
        raise ContractError(f"{product_name} raw hash manifest is missing or stale")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "energy-model-v3-raw-hash-manifest":
        raise ContractError(f"{product_name} raw hash manifest schema is invalid")
    if payload.get("product") != product_name:
        raise ContractError(f"{product_name} raw hash manifest product mismatch")
    for field in ("source", "feed", "location"):
        if payload.get(field) != trusted[field]:
            raise ContractError(
                f"{product_name} raw hash manifest {field} mismatch"
            )
    if (
        pd.isna(manifest_start := pd.Timestamp(payload.get("start_utc")))
        or pd.isna(manifest_end := pd.Timestamp(payload.get("end_utc")))
    ):
        raise ContractError(f"{product_name} raw hash manifest timestamps are missing")
    if manifest_start > candidate_start or manifest_end < candidate_end:
        raise ContractError(f"{product_name} raw hash manifest calendar is incomplete")
    files = payload.get("files")
    if not isinstance(files, dict) or not files:
        raise ContractError(f"{product_name} raw hash manifest has no files")
    validated = {
        str(manifest_path.relative_to(root)).replace("\\", "/"):
        expected_manifest_hash
    }
    expected_raw_parent = (
        root / "data" / "energy_model_v3" / "raw" / market
    ).resolve()
    for file_relative, expected_hash in files.items():
        if not re.fullmatch(r"[0-9a-f]{64}", str(expected_hash)):
            raise ContractError(f"{product_name} raw file hash is invalid")
        source_path = (repository_root / file_relative).resolve()
        if (
            expected_raw_parent != source_path.parent
            and expected_raw_parent not in source_path.parents
        ):
            raise ContractError(f"{product_name} raw file path is invalid")
        if not source_path.is_file() or _sha256(source_path) != expected_hash:
            raise ContractError(f"{product_name} raw file is missing or stale")
        validated[
            str(source_path.relative_to(root)).replace("\\", "/")
        ] = expected_hash
    return validated


def _validate_canonical_coverage_file(
    reference: Any,
    *,
    repository_root: Path,
    market: str,
    product: dict[str, Any],
    trusted: dict[str, Any],
    candidate_start: pd.Timestamp,
    candidate_end: pd.Timestamp,
) -> dict[str, str]:
    if not isinstance(reference, dict):
        raise ContractError(
            f"{product['name']} canonical coverage reference is invalid"
        )
    relative = reference.get("path")
    expected_hash = reference.get("sha256")
    cadence_minutes = reference.get("cadence_minutes")
    if (
        not isinstance(relative, str)
        or not re.fullmatch(r"[0-9a-f]{64}", str(expected_hash))
        or not isinstance(cadence_minutes, int)
        or cadence_minutes <= 0
    ):
        raise ContractError(
            f"{product['name']} canonical coverage reference is invalid"
        )
    if cadence_minutes != trusted["cadence_minutes"]:
        raise ContractError(
            f"{product['name']} canonical cadence does not match contract"
        )
    root = repository_root.resolve()
    path = (root / relative).resolve()
    expected_parent = (
        root / "data" / "energy_model_v3" / "native" / market
    ).resolve()
    if expected_parent != path.parent and expected_parent not in path.parents:
        raise ContractError(
            f"{product['name']} canonical coverage file is outside its market"
        )
    if not path.is_file() or _sha256(path) != expected_hash:
        raise ContractError(
            f"{product['name']} canonical coverage file is missing or stale"
        )
    frame = pd.read_csv(path)
    required = {
        "interval_start_utc",
        "market",
        "source",
        "feed",
        "product",
        "location",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ContractError(
            f"{product['name']} canonical coverage columns missing: {sorted(missing)}"
        )
    if set(frame["market"].astype(str)) != {market}:
        raise ContractError(f"{product['name']} canonical market mismatch")
    expected_identity = {
        "source": trusted["source"],
        "feed": trusted["feed"],
        "product": product["name"],
        "location": trusted["location"],
    }
    for field, expected in expected_identity.items():
        if set(frame[field].astype(str)) != {str(expected)}:
            raise ContractError(
                f"{product['name']} canonical {field} identity mismatch"
            )
    actual = pd.DatetimeIndex(
        pd.to_datetime(frame["interval_start_utc"], utc=True, errors="raise")
    )
    if not actual.is_unique:
        raise ContractError(
            f"{product['name']} canonical coverage has duplicate intervals"
        )
    expected_index = pd.date_range(
        candidate_start,
        candidate_end,
        freq=pd.Timedelta(minutes=cadence_minutes),
        inclusive="left",
    )
    if not actual.equals(expected_index):
        raise ContractError(
            f"{product['name']} canonical coverage index is incomplete"
        )
    missing_data = set(trusted["data_columns"]) - set(frame.columns)
    if missing_data:
        raise ContractError(
            f"{product['name']} canonical data columns missing: "
            f"{sorted(missing_data)}"
        )
    numeric = frame[trusted["data_columns"]].apply(
        pd.to_numeric, errors="raise"
    )
    if numeric.isna().any().any():
        raise ContractError(
            f"{product['name']} canonical data contains missing values"
        )
    if not np.isfinite(numeric.to_numpy()).all():
        raise ContractError(
            f"{product['name']} canonical data contains non-finite values"
        )
    return {str(path.relative_to(root)).replace("\\", "/"): expected_hash}
