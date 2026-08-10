"""Build the real six-market energy-v3 handoff for the ramp-v6 campaign."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from energy_model_v3.calendar import make_calendar
from energy_model_v3.forecasts import reconstruct_forecasts
from env.ramp_v6.models import FrozenRampStats, RampProtocol
from env.ramp_v6.panel import CanonicalMarketPanel
from ramp_rl.provenance import historical_text_sha256_matches

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data" / "energy_model_v3"
LIVE_ROOT = ROOT / "output" / "energy_model_v3" / "live-panel"
OUTPUT_ROOT = ROOT / "output" / "energy_model_v3" / "ramp_v6"
PANEL_MANIFEST_SHA256 = (
    "489cb39c61c19952fe90b213c1ea20e4f5eb904563425ec8b751a15ceed446df"
)
ACQUISITION_MANIFEST_SHA256 = (
    "64fd78254dabefa8d525f3e43144b2a50bc12c3050bad21efcc8836e3d11b7e7"
)
FORECAST_MODEL = "energy-v3-causal-h3-trajectory-gross-net-ridge-v1"
CANONICAL_TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".md",
    ".py",
    ".yaml",
    ".yml",
}
MARKET_TO_CELL = {
    "CAISO_NP15": "a",
    "ERCOT_LZ_NORTH": "b",
    "NYISO_NYC_J": "c",
    "MISO_MINN_HUB": "d",
    "SPP_NORTH_HUB": "e",
    "ISONE_NEMA": "f",
}
PHYSICAL_COLUMNS = {
    "interval_start_utc": "timestamp_utc",
    "market": "market_id",
    "da_lmp_usd_mwh": "da_lmp_usd_per_mwh",
}


def sha256_file(path: Path) -> str:
    if path.suffix.lower() in CANONICAL_TEXT_SUFFIXES:
        payload = path.read_bytes().replace(b"\r\n", b"\n")
        return hashlib.sha256(payload).hexdigest()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_matches(path: Path, expected: str) -> bool:
    if path.suffix.lower() in CANONICAL_TEXT_SUFFIXES:
        return historical_text_sha256_matches(path, expected)
    return sha256_file(path) == expected


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _verify_live_inputs() -> dict[str, Any]:
    panel_manifest_path = LIVE_ROOT / "manifest.json"
    acquisition_path = DATA_ROOT / "provenance" / "live-acquisition-manifest.json"
    if not sha256_matches(panel_manifest_path, PANEL_MANIFEST_SHA256):
        raise ValueError("live panel manifest SHA-256 does not match the verified handoff")
    if not sha256_matches(acquisition_path, ACQUISITION_MANIFEST_SHA256):
        raise ValueError("live acquisition manifest SHA-256 does not match the verified handoff")
    manifest = json.loads(panel_manifest_path.read_text(encoding="utf-8"))
    if manifest["market_order"] != list(MARKET_TO_CELL):
        raise ValueError("live panel market order does not match the frozen six-market mapping")
    for name, expected in manifest["outputs"].items():
        if not sha256_matches(LIVE_ROOT / name, expected):
            raise ValueError(f"live panel artifact hash mismatch: {name}")
    return manifest


def _load_physical_panel() -> tuple[pd.DataFrame, dict[str, str]]:
    frames = []
    hashes: dict[str, str] = {}
    for market in MARKET_TO_CELL:
        path = LIVE_ROOT / f"{market}.csv"
        source = pd.read_csv(path)
        source["interval_start_utc"] = pd.to_datetime(
            source["interval_start_utc"], utc=True, errors="raise"
        )
        selected = source[
            [
                "interval_start_utc",
                "market",
                "gross_demand_mw",
                "net_load_mw",
                "wind_mw",
                "solar_mw",
                "market_scale_mw",
                "da_lmp_usd_mwh",
            ]
        ].rename(columns=PHYSICAL_COLUMNS)
        if set(selected["market_id"]) != {market}:
            raise ValueError(f"live panel market mismatch: {market}")
        frames.append(selected)
        hashes[f"live_panel:{market}"] = sha256_file(path)
    panel = pd.concat(frames, ignore_index=True).sort_values(
        ["timestamp_utc", "market_id"]
    )
    return panel.reset_index(drop=True), hashes


def _h3_forecast_rows(
    physical_market: pd.DataFrame, market: str
) -> tuple[pd.DataFrame, dict[str, str]]:
    net_path = DATA_ROOT / "forecasts" / f"{market}.csv"
    forecast_input = physical_market.rename(
        columns={"timestamp_utc": "interval_start_utc"}
    )
    net, _ = reconstruct_forecasts(
        forecast_input,
        market=market,
        calendar=make_calendar(),
        target_column="net_load_mw",
    )
    gross, _ = reconstruct_forecasts(
        forecast_input,
        market=market,
        calendar=make_calendar(),
        target_column="gross_demand_mw",
    )
    net = net[net["forecast_horizon_hours"] == 3].copy()
    gross = gross[gross["forecast_horizon_hours"] == 3].copy()
    return (
        pd.concat([net, gross], ignore_index=True),
        {
            f"net_forecast:{market}": sha256_file(net_path),
            f"gross_forecast_model_input:{market}": sha256_file(
                LIVE_ROOT / f"{market}.csv"
            ),
            "forecast_implementation": sha256_file(
                ROOT / "energy_model_v3" / "forecasts.py"
            ),
        },
    )


def _attach_causal_forecasts(
    physical: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str]]:
    outputs = []
    hashes: dict[str, str] = {}
    for market in MARKET_TO_CELL:
        rows = physical[physical["market_id"] == market].copy()
        forecasts, market_hashes = _h3_forecast_rows(rows, market)
        hashes.update(market_hashes)
        indexed = {
            target: group.set_index("interval_start_utc")
            for target, group in forecasts.groupby("forecast_target")
        }
        complete_rows = []
        for row in rows.itertuples(index=False):
            timestamp = row.timestamp_utc
            components: list[pd.Series] = []
            missing = False
            for quantity in ("gross_demand_mw", "net_load_mw"):
                table = indexed[quantity]
                for horizon in (1, 2, 3):
                    target = timestamp + pd.Timedelta(hours=horizon)
                    if target not in table.index:
                        missing = True
                        break
                    components.append(table.loc[target])
                if missing:
                    break
            if missing:
                continue
            values = row._asdict()
            gross = components[:3]
            net = components[3:]
            for horizon, component in enumerate(gross, start=1):
                values[f"forecast_gross_h{horizon}_mw"] = float(
                    component["forecast_value_mw"]
                )
            for horizon, component in enumerate(net, start=1):
                values[f"forecast_net_h{horizon}_mw"] = float(
                    component["forecast_value_mw"]
                )
            latest_issue = max(
                pd.Timestamp(component["forecast_issue_utc"])
                for component in components
            )
            if latest_issue > timestamp:
                raise ValueError("forecast bundle contains a future-issued component")
            identity = [
                f"{component['forecast_model_id']}@"
                f"{pd.Timestamp(component['forecast_vintage_utc']).isoformat()}"
                for component in components
            ]
            values["forecast_issue_time_utc"] = latest_issue
            values["forecast_vintage_id"] = "bundle-" + hashlib.sha256(
                "|".join(identity).encode()
            ).hexdigest()[:24]
            values["quality_ok"] = True
            complete_rows.append(values)
        outputs.append(pd.DataFrame(complete_rows))
    result = pd.concat(outputs, ignore_index=True)
    common_times = set.intersection(
        *(
            set(result.loc[result["market_id"] == market, "timestamp_utc"])
            for market in MARKET_TO_CELL
        )
    )
    result = result[result["timestamp_utc"].isin(common_times)].copy()
    result = result.sort_values(["timestamp_utc", "market_id"]).reset_index(drop=True)
    CanonicalMarketPanel(result)
    return result, hashes


def _fit_frozen_stats(physical: pd.DataFrame, train_months: list[str]) -> FrozenRampStats:
    labels = physical["timestamp_utc"].dt.strftime("%Y-%m")
    train = physical[labels.isin(train_months)].copy()
    markets = set(MARKET_TO_CELL)
    if set(train["market_id"]) != markets:
        raise ValueError("training statistics do not cover all six markets")
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
        gross_mean[market] = float(gross.mean())
        gross_std[market] = max(float(gross.std()), 1e-9)
        net_mean[market] = float(net.mean())
        net_std[market] = max(float(net.std()), 1e-9)
        thresholds[market] = {
            horizon: float(
                np.quantile(
                    np.abs((net[horizon:] - net[:-horizon]) / (scale * horizon)),
                    0.90,
                )
            )
            for horizon in (1, 3)
        }
    return FrozenRampStats(
        fit_start_utc=train["timestamp_utc"].min().isoformat(),
        fit_end_utc=train["timestamp_utc"].max().isoformat(),
        gross_q95_mw=gross_q95,
        gross_level_mean_mw=gross_mean,
        gross_level_std_mw=gross_std,
        net_level_mean_mw=net_mean,
        net_level_std_mw=net_std,
        native_abs_ramp_q90_fraction_s_per_hour=thresholds,
        fit_months=tuple(train_months),
        stats_id="energy-v3-live-train-only-ramp-stats-v1",
    )


def _load_workload(
    timestamps: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int], dict[str, str]]:
    service: dict[str, np.ndarray] = {}
    batch: dict[str, np.ndarray] = {}
    deadlines: dict[str, int] = {}
    hashes: dict[str, str] = {}
    for market, cell in MARKET_TO_CELL.items():
        tier_path = ROOT / "data" / "cells" / f"cell_{cell}_tiers.csv"
        tier = pd.read_csv(tier_path)
        if len(tier) % 12 == 1:
            tier = tier.iloc[:-1]
        if len(tier) % 12:
            raise ValueError(f"tier trace cannot be aggregated to hours: cell {cell}")
        hourly = tier.groupby(np.arange(len(tier)) // 12)[
            ["service_demand_norm", "batch_demand_norm"]
        ].mean()
        repetitions = math.ceil(len(timestamps) / len(hourly))
        service[market] = np.tile(
            hourly["service_demand_norm"].to_numpy(), repetitions
        )[: len(timestamps)]
        batch[market] = np.tile(
            hourly["batch_demand_norm"].to_numpy(), repetitions
        )[: len(timestamps)]
        distribution_path = ROOT / "data" / "jobs" / f"batch_distributions_{cell}.json"
        distribution = json.loads(distribution_path.read_text(encoding="utf-8"))
        deadlines[market] = max(
            1, math.ceil(2.0 * float(distribution["duration"]["mean"]) / 3600.0)
        )
        hashes[f"tier_workload:cell_{cell}"] = sha256_file(tier_path)
        hashes[f"deadline_model:cell_{cell}"] = sha256_file(distribution_path)
    service_frame = pd.DataFrame(service, index=timestamps)
    batch_frame = pd.DataFrame(batch, index=timestamps)
    batch_limit = RampProtocol().batch_arrival_envelope_fraction_of_fleet * len(
        MARKET_TO_CELL
    )
    totals = batch_frame.sum(axis=1)
    scale = np.minimum(1.0, batch_limit / totals.where(totals > 0.0, 1.0))
    admitted_batch = batch_frame.mul(scale, axis=0)
    service_frame += batch_frame - admitted_batch
    if float(service_frame.sum(axis=1).max()) > 0.75 * len(MARKET_TO_CELL) + 1e-12:
        raise ValueError("measured service workload exceeds the frozen fleet envelope")
    if float(admitted_batch.sum(axis=1).max()) > batch_limit + 1e-12:
        raise ValueError("admitted batch workload exceeds the frozen fleet envelope")
    return service_frame, admitted_batch, deadlines, hashes


def _load_warm_power(
    timestamps: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, dict[str, str]]:
    values: dict[str, pd.Series] = {}
    hashes: dict[str, str] = {}
    for market in MARKET_TO_CELL:
        path = DATA_ROOT / "modeled_dc_power" / f"{market}.csv"
        frame = pd.read_csv(path)
        frame["interval_start_utc"] = pd.to_datetime(
            frame["interval_start_utc"], utc=True, errors="raise"
        )
        values[market] = frame.set_index("interval_start_utc")["dc_power_mw"]
        hashes[f"modeled_power:{market}"] = sha256_file(path)
    result = pd.DataFrame(values).reindex(timestamps)
    if result.isna().any().any():
        raise ValueError("modeled warm-power inputs do not cover the study calendar")
    return result, hashes


def _sites() -> list[dict[str, Any]]:
    manifest = json.loads(
        (DATA_ROOT / "provenance" / "workload-power-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    sites = []
    for market, cell in MARKET_TO_CELL.items():
        model = manifest["markets"][market]["power_model"]
        sites.append(
            {
                "site_id": f"site-{cell}",
                "market_id": market,
                "rated_power_mw": 100.0,
                "compute_capacity": 1.0,
                "idle_power_fraction": float(model["idle_power"]),
                "dynamic_power_fraction": float(model["slope"]),
            }
        )
    return sites


def build_factory(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    live_manifest = _verify_live_inputs()
    physical, live_hashes = _load_physical_panel()
    canonical, forecast_hashes = _attach_causal_forecasts(physical)
    protocol_path = ROOT / "env" / "protocols" / "v6_pure_ramp_rl.yaml"
    campaign = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    split_months = {
        split: [str(value) for value in campaign["data"]["split"][split]["months"]]
        for split in ("train", "validation", "test")
    }
    stats = _fit_frozen_stats(physical, split_months["train"])
    stats_payload = asdict(stats)
    frozen_stats_sha256 = sha256_json(stats_payload)
    timestamps = pd.DatetimeIndex(
        physical["timestamp_utc"].drop_duplicates().sort_values()
    )
    service, batch, deadlines, workload_hashes = _load_workload(timestamps)
    warm_power, power_hashes = _load_warm_power(timestamps)
    all_input_hashes = {
        "live_panel_manifest": PANEL_MANIFEST_SHA256,
        "live_acquisition_manifest": ACQUISITION_MANIFEST_SHA256,
        **live_hashes,
        **forecast_hashes,
        **workload_hashes,
        **power_hashes,
    }
    windows_root = output_root / "windows"
    if windows_root.exists():
        shutil.rmtree(windows_root)
    windows_root.mkdir(parents=True)
    windows: dict[str, dict[str, Any]] = {
        "train": {},
        "validation": {},
        "test": {},
    }
    sites = _sites()
    available = set(canonical["timestamp_utc"].drop_duplicates())
    study_start = timestamps.min().normalize()
    study_end = timestamps.max().normalize()
    for day in pd.date_range(study_start, study_end, freq="D", tz="UTC"):
        period = day.strftime("%Y-%m")
        split = next(
            (
                name
                for name, months in split_months.items()
                if period in months
            ),
            None,
        )
        if split is None:
            continue
        panel_times = pd.date_range(
            day - pd.Timedelta(hours=3),
            day + pd.Timedelta(hours=26),
            freq="h",
            tz="UTC",
        )
        if not set(panel_times).issubset(available):
            continue
        window_id = f"{day:%Y-%m-%d}-daily"
        artifact_root = windows_root / window_id
        artifact_root.mkdir()
        panel = canonical[canonical["timestamp_utc"].isin(panel_times)].copy()
        CanonicalMarketPanel(panel)
        panel_path = artifact_root / "canonical_panel.csv"
        panel.to_csv(panel_path, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
        active_times = pd.date_range(day, periods=24, freq="h", tz="UTC")
        tail = np.zeros((3, len(MARKET_TO_CELL)), dtype=float)
        service_values = np.vstack(
            [service.loc[active_times, list(MARKET_TO_CELL)].to_numpy(), tail]
        )
        batch_values = np.vstack(
            [batch.loc[active_times, list(MARKET_TO_CELL)].to_numpy(), tail]
        )
        deadline_values = np.tile(
            np.asarray(
                [deadlines[market] for market in MARKET_TO_CELL], dtype=np.int64
            ),
            (27, 1),
        )
        warm_times = pd.date_range(
            day - pd.Timedelta(hours=3), periods=3, freq="h", tz="UTC"
        )
        fixture = {
            "fixture_id": f"energy-v3-ramp-v6-{window_id}",
            "sites": sites,
            "workload": {
                "source": "measured Borg a-f tier curves; excess batch is conserved as immediate service",
                "deadline_semantics": "ceil(2x measured mean completed no-SLO duration / 1h)",
                "service_arrivals": service_values.tolist(),
                "batch_arrivals": batch_values.tolist(),
                "batch_deadline_hours": deadline_values.tolist(),
                "warm_power_mw": warm_power.loc[
                    warm_times, list(MARKET_TO_CELL)
                ].to_numpy().tolist(),
            },
            "frozen_stats": stats_payload,
        }
        fixture_path = artifact_root / "fixture.json"
        fixture_path.write_text(
            json.dumps(fixture, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        vintages = sorted(set(panel["forecast_vintage_id"]))
        source_hashes = {
            "canonical_panel": sha256_file(panel_path),
            "fixture": sha256_file(fixture_path),
        }
        windows[split][window_id] = {
            "artifact_root": str(artifact_root.relative_to(ROOT)),
            "month": int(day.month),
            "period": period,
            "day": day.strftime("%Y-%m-%d"),
            "forecast_vintage": "window-" + sha256_json(vintages)[:24],
            "source_hashes": source_hashes,
            "input_hashes_sha256": sha256_json(all_input_hashes),
        }
    if any(not windows[split] for split in windows):
        raise ValueError("factory generation produced an empty split")
    manifest = {
        "schema_version": "energy-model-v3-ramp-v6-factory-v1",
        "forecast_model": FORECAST_MODEL,
        "frozen_stats_sha256": frozen_stats_sha256,
        "source_panel_manifest_sha256": PANEL_MANIFEST_SHA256,
        "raw_acquisition_manifest_sha256": ACQUISITION_MANIFEST_SHA256,
        "source_panel_generated_at_utc": live_manifest["generated_at_utc"],
        "split_periods": split_months,
        "input_hashes": all_input_hashes,
        "workload_policy": {
            "source_cells": list(MARKET_TO_CELL.values()),
            "batch_excess_handling": "conserve_and_reclassify_as_immediate_service",
            "batch_arrival_fleet_envelope_fraction": 0.10,
            "service_fleet_envelope_fraction": 0.75,
        },
        "windows": windows,
    }
    manifest_path = output_root / "factory_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def validate_factory(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    manifest_path = output_root / "factory_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    counts: dict[str, int] = {}
    for split, windows in payload["windows"].items():
        counts[split] = len(windows)
        for window in windows.values():
            root = ROOT / window["artifact_root"]
            for name, filename in (
                ("canonical_panel", "canonical_panel.csv"),
                ("fixture", "fixture.json"),
            ):
                if not sha256_matches(
                    root / filename,
                    window["source_hashes"][name],
                ):
                    raise ValueError(f"{split} window {name} hash mismatch")
            panel = CanonicalMarketPanel.from_csv(root / "canonical_panel.csv")
            fixture = json.loads((root / "fixture.json").read_text(encoding="utf-8"))
            if len(panel.markets) != 6 or len(fixture["sites"]) != 6:
                raise ValueError("factory window is not six-market")
            if sha256_json(fixture["frozen_stats"]) != payload["frozen_stats_sha256"]:
                raise ValueError("factory window frozen statistics hash mismatch")
            service = np.asarray(fixture["workload"]["service_arrivals"])
            batch = np.asarray(fixture["workload"]["batch_arrivals"])
            if service.shape != (27, 6) or batch.shape != (27, 6):
                raise ValueError("factory workload arrays are not daily plus tail")
            if np.any(service[-3:] != 0.0) or np.any(batch[-3:] != 0.0):
                raise ValueError("factory terminal tail contains new arrivals")
            if float(service.sum(axis=1).max()) > 4.5 + 1e-12:
                raise ValueError("factory service envelope violation")
            if float(batch.sum(axis=1).max()) > 0.6 + 1e-12:
                raise ValueError("factory batch envelope violation")
    return {
        "schema_version": payload["schema_version"],
        "manifest_sha256": sha256_file(manifest_path),
        "window_counts": counts,
        "frozen_stats_sha256": payload["frozen_stats_sha256"],
    }
