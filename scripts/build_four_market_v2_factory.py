"""Build the one raw-workload four-market factory."""

from __future__ import annotations

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
    CELLS, COMPUTE_CAPACITY, FACTORY_ROOT, MARKETS, MARKET_TO_CELL,
    RATED_POWER_MW, TOTAL_RATED_POWER_MW,
)
from env.ramp_v6.models import SiteConfig  # noqa: E402


CALENDAR_PATH = ROOT / "data" / "four_market_v2" / "calendar.json"
TRACE_START = pd.Timestamp("2025-09-01T00:00:00Z")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _hourly_curve(cell: str) -> tuple[np.ndarray, np.ndarray]:
    rows = pd.read_csv(ROOT / "data" / "cells" / f"cell_{cell}_tiers.csv")
    if len(rows) % 12 == 1:
        rows = rows.iloc[:-1]
    if len(rows) % 12:
        raise ValueError(f"cell {cell} cannot form whole hourly intervals")
    hourly = rows.groupby(np.arange(len(rows)) // 12)[["service_demand_norm", "batch_demand_norm"]].mean()
    return (
        hourly["service_demand_norm"].to_numpy(dtype=np.float64),
        hourly["batch_demand_norm"].to_numpy(dtype=np.float64),
    )


def _site_configs() -> list[SiteConfig]:
    models = json.loads((ROOT / "data" / "power_model_params.json").read_text())["per_cell_cpu_model"]
    return [
        SiteConfig(
            site_id=f"site-{cell}", market_id=market, rated_power_mw=RATED_POWER_MW,
            compute_capacity=COMPUTE_CAPACITY, idle_power_fraction=float(models[cell]["idle_power"]),
            dynamic_power_fraction=float(models[cell]["slope"]),
        )
        for market, cell in MARKET_TO_CELL
    ]


def _raw_arrivals(times: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.asarray((times - TRACE_START) / pd.Timedelta(hours=1), dtype=np.int64)
    service_columns, batch_columns = [], []
    for cell in CELLS:
        service, batch = _hourly_curve(cell)
        service_columns.append(service[offsets % len(service)])
        batch_columns.append(batch[offsets % len(batch)])
    return np.column_stack(service_columns), np.column_stack(batch_columns)


def _workload(timestamps: pd.DatetimeIndex, sites: list[SiteConfig]) -> dict[str, Any]:
    service, batch = _raw_arrivals(timestamps[3:-3])
    capacity = sum(site.compute_capacity for site in sites)
    if np.any((service + batch).sum(axis=1) > capacity + 1e-12):
        raise ValueError("raw hourly arrivals exceed hard fleet capacity")
    warm_service, warm_batch = _raw_arrivals(timestamps[:3])
    warm_power = np.asarray([
        [site.power_mw(float(work)) for site, work in zip(sites, row)]
        for row in warm_service + warm_batch
    ])
    tail = np.zeros((3, len(sites)))
    return {
        "source": "raw measured per-cell a-d hourly arrivals",
        "service_arrivals": np.vstack((service, tail)).tolist(),
        "batch_arrivals": np.vstack((batch, tail)).tolist(),
        "batch_deadline_hours": np.tile(np.asarray([2, 1, 2, 3]), (27, 1)).tolist(),
        "warm_power_mw": warm_power.tolist(),
    }


def _load_calendar() -> dict[str, Any]:
    calendar = json.loads(CALENDAR_PATH.read_text(encoding="utf-8"))
    if tuple(calendar["markets"]) != MARKETS or calendar.get("test") != []:
        raise ValueError("calendar market or test split contract is invalid")
    if len(calendar["train"]) != 114 or len(calendar["validation"]) != 28:
        raise ValueError("calendar must contain 114 train and 28 validation days")
    if not (ROOT / calendar["frozen_stats_path"]).is_file():
        raise FileNotFoundError("calendar frozen statistics are missing")
    return calendar


def build() -> dict[str, Any]:
    if FACTORY_ROOT.exists():
        return validate()
    calendar = _load_calendar()
    sites = _site_configs()
    windows: dict[str, dict[str, Any]] = {"train": {}, "validation": {}, "test": {}}
    try:
        for split in ("train", "validation"):
            for source in calendar[split]:
                window_id = source["window_id"]
                panel_path = ROOT / source["panel_path"]
                if not panel_path.is_file():
                    raise FileNotFoundError(f"{window_id} canonical panel is missing")
                frame = pd.read_csv(panel_path)
                timestamps = pd.DatetimeIndex(pd.to_datetime(frame["timestamp_utc"].unique(), utc=True)).sort_values()
                if len(timestamps) != 30 or set(frame["market_id"]) != set(MARKETS):
                    raise ValueError(f"{window_id} canonical panel shape is invalid")
                fixture = FACTORY_ROOT / "windows" / window_id / "fixture.json"
                fixture.parent.mkdir(parents=True, exist_ok=False)
                _write_json(fixture, {"sites": [asdict(site) for site in sites], "workload": _workload(timestamps, sites)})
                windows[split][window_id] = {
                    "day": source["day"], "month": source["month"],
                    "panel_path": source["panel_path"],
                    "fixture_path": str(fixture.relative_to(ROOT)).replace("\\", "/"),
                }
        _write_json(FACTORY_ROOT / "factory.json", {
            "markets": list(MARKETS), "workload_cells": list(CELLS),
            "frozen_stats_path": calendar["frozen_stats_path"],
            "forecast_model": calendar["forecast_model"], "windows": windows,
        })
    except Exception:
        shutil.rmtree(FACTORY_ROOT, ignore_errors=True)
        raise
    return validate()


def validate() -> dict[str, Any]:
    factory_path = FACTORY_ROOT / "factory.json"
    factory = json.loads(factory_path.read_text(encoding="utf-8"))
    if tuple(factory["markets"]) != MARKETS or tuple(factory["workload_cells"]) != CELLS:
        raise ValueError("factory market mapping is invalid")
    if factory["windows"].get("test") != {}:
        raise ValueError("factory must not contain a test split")
    result = {"factory": str(factory_path.relative_to(ROOT)).replace("\\", "/")}
    for split, expected in (("train", 114), ("validation", 28)):
        records = factory["windows"][split]
        if len(records) != expected:
            raise ValueError(f"factory has wrong {split} count")
        for window_id, record in records.items():
            panel = ROOT / record["panel_path"]
            fixture = ROOT / record["fixture_path"]
            if not panel.is_file() or not fixture.is_file():
                raise FileNotFoundError(f"{split}/{window_id} factory input is missing")
            payload = json.loads(fixture.read_text(encoding="utf-8"))
            workload = payload["workload"]
            if tuple(np.asarray(workload["batch_deadline_hours"])[0]) != (2, 1, 2, 3):
                raise ValueError(f"{split}/{window_id} deadline contract is invalid")
            arrivals = (
                np.asarray(workload["service_arrivals"])
                + np.asarray(workload["batch_arrivals"])
            ).sum(axis=1)
            if np.any(arrivals > len(payload["sites"]) + 1e-12):
                raise ValueError(f"{split}/{window_id} violates hard capacity")
        result[f"{split}_windows"] = expected
    return result


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, sort_keys=True))
