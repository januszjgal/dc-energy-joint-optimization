"""Build and validate self-contained direct-K=1 four-market factory artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from energy_model_v3.four_market_v2 import (  # noqa: E402
    CELLS,
    COMPUTE_CAPACITY,
    FACTORY_ROOT,
    MARKETS,
    MARKET_TO_CELL,
    RATED_POWER_MW,
    TOTAL_RATED_POWER_MW,
    VARIANTS,
)
from env.ramp_v6.models import SiteConfig  # noqa: E402


ACTIVE_SOURCE_ROOT = ROOT / "data" / "four_market_v2"
ACTIVE_SOURCE_MANIFEST = ACTIVE_SOURCE_ROOT / "source_manifest.json"
TRACE_START = pd.Timestamp("2025-09-01T00:00:00Z")


def canonical_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _hourly_curve(cell: str) -> tuple[np.ndarray, np.ndarray]:
    rows = pd.read_csv(ROOT / "data" / "cells" / f"cell_{cell}_tiers.csv")
    if len(rows) % 12 == 1:
        rows = rows.iloc[:-1]
    if len(rows) % 12:
        raise ValueError(f"cell {cell} cannot form whole hourly intervals")
    hourly = rows.groupby(np.arange(len(rows)) // 12)[
        ["service_demand_norm", "batch_demand_norm"]
    ].mean()
    return (
        hourly["service_demand_norm"].to_numpy(dtype=np.float64),
        hourly["batch_demand_norm"].to_numpy(dtype=np.float64),
    )


def _site_configs() -> list[SiteConfig]:
    models = json.loads((ROOT / "data" / "power_model_params.json").read_text())[
        "per_cell_cpu_model"
    ]
    return [
        SiteConfig(
            site_id=f"site-{cell}",
            market_id=market,
            rated_power_mw=RATED_POWER_MW,
            compute_capacity=COMPUTE_CAPACITY,
            idle_power_fraction=float(models[cell]["idle_power"]),
            dynamic_power_fraction=float(models[cell]["slope"]),
        )
        for market, cell in MARKET_TO_CELL
    ]


def _raw_arrivals(times: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    service_columns: list[np.ndarray] = []
    batch_columns: list[np.ndarray] = []
    offsets = np.asarray((times - TRACE_START) / pd.Timedelta(hours=1), dtype=np.int64)
    for cell in CELLS:
        service, batch = _hourly_curve(cell)
        service_columns.append(service[offsets % len(service)])
        batch_columns.append(batch[offsets % len(batch)])
    return np.column_stack(service_columns), np.column_stack(batch_columns)


def _workload(
    timestamps: pd.DatetimeIndex, variant: str, sites: list[SiteConfig]
) -> dict[str, Any]:
    active_service, raw_batch = _raw_arrivals(timestamps[3:-3])
    if variant == "envelope_on":
        cap = 0.10 * len(sites)
        totals = raw_batch.sum(axis=1)
        scale = np.minimum(1.0, cap / np.where(totals > 0.0, totals, 1.0))
        batch = raw_batch * scale[:, None]
        service = active_service + raw_batch - batch
        if service.sum(axis=1).max() > 0.75 * len(sites) + 1e-12:
            raise ValueError("envelope_on service exceeds the 75% fleet envelope")
    elif variant == "envelope_off":
        service = active_service
        batch = raw_batch
    else:
        raise ValueError(f"unsupported variant {variant}")
    if np.any(service.sum(axis=1) > len(sites) + 1e-12):
        raise ValueError("immediate service is not feasible")
    warm_service, warm_batch = _raw_arrivals(timestamps[:3])
    warm_work = warm_service + warm_batch
    warm_power = np.asarray(
        [
            [site.power_mw(float(work)) for site, work in zip(sites, row)]
            for row in warm_work
        ],
        dtype=np.float64,
    )
    tail = np.zeros((3, len(sites)), dtype=np.float64)
    return {
        "source": (
            "raw measured per-cell a-d hourly arrivals"
            if variant == "envelope_off"
            else "raw measured per-cell a-d arrivals with fleet batch admission envelope"
        ),
        "admission_envelope_enabled": variant == "envelope_on",
        "service_arrivals": np.vstack((service, tail)).tolist(),
        "batch_arrivals": np.vstack((batch, tail)).tolist(),
        "batch_deadline_hours": np.tile(np.asarray([2, 1, 2, 3]), (27, 1)).tolist(),
        "warm_power_mw": warm_power.tolist(),
    }


def build() -> dict[str, Any]:
    if FACTORY_ROOT.exists():
        return validate()
    source_manifest = ACTIVE_SOURCE_MANIFEST
    if not source_manifest.is_file():
        raise FileNotFoundError("active four-market physical source is unavailable")
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    if tuple(source["markets"]) != MARKETS:
        raise ValueError("active physical source market mapping is invalid")
    if set(source["windows"]) != {"train", "validation"}:
        raise ValueError("active physical source must contain train and validation only")
    if len(source["windows"]["train"]) != 114 or len(source["windows"]["validation"]) != 28:
        raise ValueError("active physical source has the wrong learning calendar")
    sites = _site_configs()
    stats_path = ROOT / source["frozen_stats_path"]
    if canonical_hash(stats_path) != source["frozen_stats_sha256"]:
        raise ValueError("active physical source frozen statistics hash mismatch")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    stats_hash = hashlib.sha256(
        json.dumps(stats, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    try:
        for variant in VARIANTS:
            records: dict[str, dict[str, Any]] = {"train": {}, "validation": {}}
            for split in ("train", "validation"):
                for window_id, source_record in sorted(source["windows"][split].items()):
                    source_root = ROOT / source_record["artifact_root"]
                    source_panel = source_root / "canonical_panel.csv"
                    if canonical_hash(source_panel) != source_record["canonical_panel_sha256"]:
                        raise ValueError(f"{window_id} active physical panel hash mismatch")
                    destination = FACTORY_ROOT / variant / "windows" / window_id
                    destination.mkdir(parents=True, exist_ok=False)
                    panel_path = destination / "canonical_panel.csv"
                    shutil.copyfile(source_panel, panel_path)
                    frame = pd.read_csv(panel_path)
                    timestamps = pd.DatetimeIndex(
                        pd.to_datetime(frame["timestamp_utc"].unique(), utc=True)
                    ).sort_values()
                    if len(timestamps) != 30 or set(frame["market_id"]) != set(MARKETS):
                        raise ValueError(f"{window_id} has invalid physical panel rows")
                    fixture_path = destination / "fixture.json"
                    write_json(
                        fixture_path,
                        {
                            "fixture_id": f"four-market-v2-{variant}-{window_id}",
                            "sites": [asdict(site) for site in sites],
                            "workload": _workload(timestamps, variant, sites),
                            "frozen_stats": stats,
                        },
                    )
                    records[split][window_id] = {
                        "artifact_root": str(destination.relative_to(ROOT)).replace("\\", "/"),
                        "month": int(source_record["month"]),
                        "period": str(source_record["period"]),
                        "day": str(source_record["day"]),
                        "forecast_vintage": str(source_record["forecast_vintage"]),
                        "source_hashes": {
                            "canonical_panel": canonical_hash(panel_path),
                            "fixture": canonical_hash(fixture_path),
                        },
                    }
            manifest = {
                "schema_version": "four-market-direct-k1-factory-v2",
                "status": "large_validation_campaign_non_final",
                "markets": list(MARKETS),
                "workload_cells": list(CELLS),
                "variant": {
                    "id": variant,
                    "admission_envelope_enabled": variant == "envelope_on",
                    "service_envelope_fraction_of_fleet": 0.75,
                    "batch_arrival_envelope_fraction_of_fleet": 0.10,
                    "batch_handling": (
                        "conserve_and_reclassify_excess_as_immediate_service"
                        if variant == "envelope_on"
                        else "preserve_raw_measured_service_and_batch"
                    ),
                },
                "fleet": {
                    "site_count": 4,
                    "rated_power_mw_per_site": RATED_POWER_MW,
                    "compute_capacity_per_site": COMPUTE_CAPACITY,
                    "total_rated_power_mw": TOTAL_RATED_POWER_MW,
                    "power_model": "P=500*(idle_fraction+dynamic_fraction*utilization)",
                },
                "deadline_hours": [2, 1, 2, 3],
                "forecast_model": str(source["forecast_model"]),
                "frozen_stats_sha256": stats_hash,
                "physical_source": {
                    "manifest_path": str(source_manifest.relative_to(ROOT)).replace("\\", "/"),
                    "manifest_sha256": canonical_hash(source_manifest),
                },
                "split_periods": {
                    "train": ["2025-10", "2025-11", "2025-12", "2026-01"],
                    "validation": ["2026-02"],
                    "test": [],
                },
                "sealed_test_access": False,
                "windows": records,
            }
            write_json(FACTORY_ROOT / variant / "factory_manifest.json", manifest)
    except Exception:
        shutil.rmtree(FACTORY_ROOT, ignore_errors=True)
        raise
    return validate()


def validate() -> dict[str, Any]:
    results: dict[str, Any] = {}
    for variant in VARIANTS:
        manifest_path = FACTORY_ROOT / variant / "factory_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            tuple(manifest["markets"]) != MARKETS
            or tuple(manifest["workload_cells"]) != CELLS
            or manifest["split_periods"]["test"]
            or manifest["sealed_test_access"]
        ):
            raise ValueError(f"{variant} manifest design contract is invalid")
        if len(manifest["windows"]["train"]) != 114 or len(manifest["windows"]["validation"]) != 28:
            raise ValueError(f"{variant} learning calendar is invalid")
        for split in ("train", "validation"):
            for window_id, record in manifest["windows"][split].items():
                root = ROOT / record["artifact_root"]
                for name in ("canonical_panel", "fixture"):
                    path = root / f"{name}.{'csv' if name == 'canonical_panel' else 'json'}"
                    if canonical_hash(path) != record["source_hashes"][name]:
                        raise ValueError(f"{variant}/{window_id} {name} hash mismatch")
        results[variant] = {
            "manifest": str(manifest_path.relative_to(ROOT)).replace("\\", "/"),
            "manifest_sha256": canonical_hash(manifest_path),
            "train_windows": 114,
            "validation_windows": 28,
        }
    return results


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, sort_keys=True))
