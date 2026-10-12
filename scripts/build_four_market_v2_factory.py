"""Build the four-market factories: one per split specification.

    python scripts/build_four_market_v2_factory.py            # build any missing factory
    python scripts/build_four_market_v2_factory.py --rebuild  # rebuild every factory

Each cross-validation fold (cv-k1 .. cv-k4) has a factory whose windows are its
nine training months and its three validation months; the final factory has
all twelve 2025 months for training and the May 2026 test month. 2025 windows
are served by the training fleet (cells a-d); the test window by the test fleet
(cells e-h). The objective references C_R and C_Q are calibrated on the spec's
training months with the training fleet under no flexibility.
"""

from __future__ import annotations

import argparse
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
    CELL_CAPACITY_PATH, DEADLINE_WINDOW_SLOTS, MARKETS, REFERENCE_POWER_MW,
    factory_root, month_hours, site_capacities,
)
from energy_model_v3.splits import (  # noqa: E402
    CV_FOLDS, CV_SPECS, DEVELOPMENT_MONTHS, FINAL_SPEC, SPECS, TEST_FLEET, TEST_MONTHS,
    TRAIN_FLEET, fleet_for_split, spec_months,
)
from env.ramp_v6 import clock  # noqa: E402
from env.ramp_v6.models import HISTORY_HOURS, SiteConfig  # noqa: E402
from env.ramp_v6.objective import (  # noqa: E402
    JointObjective, NORMALIZATION_METHOD, OBJECTIVE_VERSION, reference_trajectory_scores,
)
from env.ramp_v6.panel import CanonicalMarketPanel  # noqa: E402
from scripts.build_data_splits import (  # noqa: E402
    CALENDAR_PATH, development_panel_path, fold_data_root, test_panel_path, write_fold_data,
)


POWER_MODEL_PATH = ROOT / "data" / "power_model_params.json"
SPLITS = ("train", "validation", "test")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


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


def _cells(fleet: tuple[tuple[str, str], ...]) -> tuple[str, ...]:
    return tuple(cell for _, cell in fleet)


def _site_configs(fleet: tuple[tuple[str, str], ...] = TRAIN_FLEET) -> list[SiteConfig]:
    models = json.loads(POWER_MODEL_PATH.read_text())["per_cell_cpu_model"]
    return [
        SiteConfig(
            site_id=f"site-{cell}", market_id=market,
            rated_power_mw=REFERENCE_POWER_MW * capacity, compute_capacity=capacity,
            idle_power_fraction=float(models[cell]["idle_power"]),
            dynamic_power_fraction=float(models[cell]["slope"]),
        )
        for (market, cell), capacity in zip(fleet, site_capacities(_cells(fleet)))
    ]


def _raw_arrivals(
    times: pd.DatetimeIndex, fleet: tuple[tuple[str, str], ...] = TRAIN_FLEET,
) -> tuple[np.ndarray, np.ndarray]:
    """Map each hour onto the workload trace, restarting the trace every month.

    Months run on the trace's own clock (UTC-7; ``env/ramp_v6/clock.py``), so
    each UTC hour draws on the trace hour that was recorded at the same clock
    time: the month's first hour, 07:00 UTC on the 1st, is trace hour 0 (00:00
    PDT on 1 May 2019). Day D of a month draws on trace day D. Months shorter
    than 31 days simply stop early, so February never reaches trace days 29-31
    and the 30-day months never reach day 31.

    The curves are fractions of each cell's own capacity; multiplying by the
    site's compute capacity turns them into work units that are the same
    amount of computation at every site, so arrivals from different sites add.
    """
    offsets = clock.trace_hours(times)
    service_columns, batch_columns = [], []
    for cell, capacity in zip(_cells(fleet), site_capacities(_cells(fleet))):
        service, batch = _hourly_curve(cell)
        service_columns.append(capacity * service[offsets % len(service)])
        batch_columns.append(capacity * batch[offsets % len(batch)])
    return np.column_stack(service_columns), np.column_stack(batch_columns)


def _workload(
    timestamps: pd.DatetimeIndex, sites: list[SiteConfig], fleet: tuple[tuple[str, str], ...],
) -> dict[str, Any]:
    """Arrivals for every hour of the month plus in-place warm power before it."""
    history = timestamps[:HISTORY_HOURS]
    active = timestamps[HISTORY_HOURS:]
    service, batch = _raw_arrivals(active, fleet)
    capacity = sum(site.compute_capacity for site in sites)
    if np.any((service + batch).sum(axis=1) > capacity + 1e-12):
        raise ValueError("raw hourly arrivals exceed hard fleet capacity")
    if np.any(service + batch > np.asarray([site.compute_capacity for site in sites]) + 1e-12):
        raise ValueError("no-flexibility arrivals exceed an origin site's capacity")
    warm_service, warm_batch = _raw_arrivals(history, fleet)
    warm_power = np.asarray([
        [site.power_mw(float(work)) for site, work in zip(sites, row)]
        for row in warm_service + warm_batch
    ])
    return {
        "source": (
            f"raw measured per-cell {'-'.join(_cells(fleet)[::3])} hourly arrivals; the warm "
            "rows carry the in-place power of the hours before the month and no queue"
        ),
        "service_arrivals": service.tolist(),
        "batch_arrivals": batch.tolist(),
        "batch_deadline_hours": np.tile(np.asarray(DEADLINE_WINDOW_SLOTS), (len(active), 1)).tolist(),
        "warm_power_mw": warm_power.tolist(),
    }


def _load_calendar() -> dict[str, Any]:
    calendar = json.loads(CALENDAR_PATH.read_text(encoding="utf-8"))
    if tuple(calendar["markets"]) != MARKETS:
        raise ValueError("calendar market order is invalid")
    if [row["month_id"] for row in calendar["development"]] != list(DEVELOPMENT_MONTHS):
        raise ValueError("calendar must list the twelve 2025 development months")
    if calendar["cv_folds"] != {fold: list(months) for fold, months in CV_FOLDS.items()}:
        raise ValueError("calendar folds differ from energy_model_v3/splits.py")
    if [row["month_id"] for row in calendar["test"]] != list(TEST_MONTHS):
        raise ValueError("calendar test month differs from energy_model_v3/splits.py")
    if any(tuple(row["workload_cells"]) != _cells(TEST_FLEET) for row in calendar["test"]):
        raise ValueError("calendar test month must be served by the test fleet")
    if int(calendar.get("history_hours", -1)) != HISTORY_HOURS:
        raise ValueError(f"calendar must carry {HISTORY_HOURS} warm hours per month")
    if not (ROOT / calendar["frozen_stats_path"]).is_file():
        raise FileNotFoundError("calendar frozen statistics are missing")
    return calendar


def spec_inputs(spec: str) -> dict[str, Any]:
    """Where a spec's panels, statistics, and forecast vintage live."""
    calendar = _load_calendar()
    months = spec_months(spec)
    if spec == FINAL_SPEC:
        panels = {month: development_panel_path(month) for month in months["train"]}
        panels.update({month: test_panel_path(month) for month in months["test"]})
        return {"panels": panels, "frozen_stats_path": ROOT / calendar["frozen_stats_path"],
                "forecast_model": calendar["forecast_model"]}
    root = fold_data_root(spec)
    panels = {month: root / "months" / month / "canonical_panel.csv"
              for month in months["train"] + months["validation"]}
    return {"panels": panels, "frozen_stats_path": root / "frozen_stats.json",
            "forecast_model": f"{calendar['forecast_model']}+{spec}"}


def _input_digest(spec: str) -> str:
    inputs = spec_inputs(spec)
    months = spec_months(spec)
    cells = set(_cells(TRAIN_FLEET)) | (set(_cells(TEST_FLEET)) if months["test"] else set())
    sources = [
        CALENDAR_PATH, inputs["frozen_stats_path"], POWER_MODEL_PATH, CELL_CAPACITY_PATH,
        *(ROOT / "data" / "cells" / f"cell_{cell}_tiers.csv" for cell in sorted(cells)),
        *inputs["panels"].values(),
        # Fold panels are derived from the committed panels, so a change there
        # must invalidate the fold factories too.
        *(development_panel_path(month) for month in DEVELOPMENT_MONTHS),
    ]
    digest = hashlib.sha256(spec.encode("utf-8") + b"\0")
    for path in sorted(set(sources)):
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def calibrate_objective(spec: str) -> dict[str, Any]:
    """Freeze positive absolute baseline scores using the spec's training months only."""
    inputs = spec_inputs(spec)
    stats = json.loads(inputs["frozen_stats_path"].read_text(encoding="utf-8"))
    months = sorted(spec_months(spec)["train"])
    if months != sorted(stats["fit_months"]):
        raise ValueError("objective calibration training months are inconsistent")
    sites = _site_configs(TRAIN_FLEET)
    scales = np.asarray([stats["gross_q95_mw"][market] for market in MARKETS])
    reference_scores: dict[str, dict[str, float]] = {}
    for month in months:
        panel = CanonicalMarketPanel.from_csv(inputs["panels"][month])
        service, batch = _raw_arrivals(panel.timestamps, TRAIN_FLEET)
        if np.any(service + batch > np.asarray([site.compute_capacity for site in sites]) + 1e-12):
            raise ValueError("objective calibration requires a feasible no-flexibility baseline")
        power = np.asarray([
            [site.power_mw(float(work)) for site, work in zip(sites, row)]
            for row in service + batch
        ])
        net = panel.frame.pivot(
            index="timestamp_utc", columns="market_id", values="net_load_mw"
        ).loc[panel.timestamps, list(MARKETS)].to_numpy(dtype=np.float64)
        ramp, peak = reference_trajectory_scores(net, power, scales)
        reference_scores[month] = {"ramp_squared_sum": ramp, "normalized_peak": peak}
    ramp_reference = float(np.mean([row["ramp_squared_sum"] for row in reference_scores.values()]))
    peak_reference = float(np.mean([row["normalized_peak"] for row in reference_scores.values()]))
    JointObjective(ramp_reference=ramp_reference, peak_reference=peak_reference).validate()
    return {
        "normalization": NORMALIZATION_METHOD,
        "fit_months": months,
        "ramp_reference": ramp_reference,
        "peak_reference": peak_reference,
        "reference_scores_by_month": reference_scores,
    }


def build(spec: str = FINAL_SPEC, *, rebuild: bool = False) -> dict[str, Any]:
    if spec not in SPECS:
        raise ValueError(f"unknown split specification {spec!r}")
    root = factory_root(spec)
    if rebuild and root.exists():
        shutil.rmtree(root)
    if root.exists():
        return validate(spec)
    if spec in CV_SPECS:
        write_fold_data(spec)
    inputs = spec_inputs(spec)
    calibration = calibrate_objective(spec)
    months = spec_months(spec)
    windows: dict[str, dict[str, Any]] = {split: {} for split in SPLITS}
    try:
        for split in SPLITS:
            fleet = fleet_for_split(split)
            sites = _site_configs(fleet)
            for month in months[split]:
                panel_path = inputs["panels"][month]
                if not panel_path.is_file():
                    raise FileNotFoundError(f"{spec}/{month} canonical panel is missing")
                frame = pd.read_csv(panel_path)
                timestamps = pd.DatetimeIndex(pd.to_datetime(frame["timestamp_utc"].unique(), utc=True)).sort_values()
                hours = month_hours(month)
                if len(timestamps) != HISTORY_HOURS + hours or set(frame["market_id"]) != set(MARKETS):
                    raise ValueError(f"{spec}/{month} canonical panel shape is invalid")
                fixture = root / "windows" / split / month / "fixture.json"
                fixture.parent.mkdir(parents=True, exist_ok=False)
                _write_json(fixture, {"sites": [asdict(site) for site in sites],
                                      "workload": _workload(timestamps, sites, fleet)})
                windows[split][month] = {
                    "month_id": month, "month": int(month[-2:]), "hours": hours,
                    "panel_path": panel_path.relative_to(ROOT).as_posix(),
                    "fixture_path": fixture.relative_to(ROOT).as_posix(),
                    "workload_cells": list(_cells(fleet)),
                }
        _write_json(root / "factory.json", {
            "version": OBJECTIVE_VERSION,
            "spec": spec,
            "split_months": {split: list(months[split]) for split in SPLITS},
            "input_digest": _input_digest(spec),
            "objective_calibration": calibration,
            "markets": list(MARKETS),
            "fleets": {"train": list(_cells(TRAIN_FLEET)), "test": list(_cells(TEST_FLEET))},
            "history_hours": HISTORY_HOURS,
            "episode_unit": "one continuous month on the trace clock (UTC-7)",
            "frozen_stats_path": inputs["frozen_stats_path"].relative_to(ROOT).as_posix(),
            "forecast_model": inputs["forecast_model"],
            "windows": windows,
        })
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise
    return validate(spec)


def validate(spec: str = FINAL_SPEC) -> dict[str, Any]:
    root = factory_root(spec)
    factory_path = root / "factory.json"
    factory = json.loads(factory_path.read_text(encoding="utf-8"))
    if factory.get("version") != OBJECTIVE_VERSION or factory.get("spec") != spec:
        raise ValueError("stale objective factory; rebuild with --rebuild")
    if factory.get("input_digest") != _input_digest(spec):
        raise ValueError("factory inputs changed; rebuild with --rebuild")
    months = spec_months(spec)
    calibration = factory["objective_calibration"]
    if (
        calibration["normalization"] != NORMALIZATION_METHOD
        or sorted(calibration["fit_months"]) != sorted(months["train"])
        or sorted(factory["windows"]["train"]) != sorted(months["train"])
    ):
        raise ValueError("factory objective calibration is inconsistent")
    JointObjective(
        ramp_reference=calibration["ramp_reference"],
        peak_reference=calibration["peak_reference"],
    ).validate()
    if tuple(factory["markets"]) != MARKETS:
        raise ValueError("factory market mapping is invalid")
    if int(factory.get("history_hours", -1)) != HISTORY_HOURS:
        raise ValueError("factory warm-history contract is invalid")
    result: dict[str, Any] = {"factory": factory_path.relative_to(ROOT).as_posix()}
    for split in SPLITS:
        records = factory["windows"][split]
        if sorted(records) != sorted(months[split]):
            raise ValueError(f"factory has the wrong {split} months")
        fleet = fleet_for_split(split)
        for month, record in records.items():
            panel = ROOT / record["panel_path"]
            fixture = ROOT / record["fixture_path"]
            if not panel.is_file() or not fixture.is_file():
                raise FileNotFoundError(f"{split}/{month} factory input is missing")
            if tuple(record["workload_cells"]) != _cells(fleet):
                raise ValueError(f"{split}/{month} is served by the wrong fleet")
            payload = json.loads(fixture.read_text(encoding="utf-8"))
            workload = payload["workload"]
            hours = month_hours(record["month_id"])
            service = np.asarray(workload["service_arrivals"])
            batch = np.asarray(workload["batch_arrivals"])
            if service.shape != (hours, len(MARKETS)) or batch.shape != service.shape:
                raise ValueError(f"{split}/{month} arrival shape is invalid")
            if np.asarray(workload["warm_power_mw"]).shape != (HISTORY_HOURS, len(MARKETS)):
                raise ValueError(f"{split}/{month} warm power shape is invalid")
            if tuple(np.asarray(workload["batch_deadline_hours"])[0]) != DEADLINE_WINDOW_SLOTS:
                raise ValueError(f"{split}/{month} deadline contract is invalid")
            capacity = np.asarray([site["compute_capacity"] for site in payload["sites"]])
            if np.any(service + batch > capacity + 1e-12):
                raise ValueError(f"{split}/{month} violates hard capacity")
        result[f"{split}_windows"] = len(records)
    return result


def build_all(*, rebuild: bool = False) -> dict[str, Any]:
    return {spec: build(spec, rebuild=rebuild) for spec in SPECS}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="replace the generated factories")
    parser.add_argument("--spec", choices=SPECS, help="build only one split specification")
    args = parser.parse_args()
    result = (build(args.spec, rebuild=args.rebuild) if args.spec
              else build_all(rebuild=args.rebuild))
    print(json.dumps(result, indent=2, sort_keys=True))
