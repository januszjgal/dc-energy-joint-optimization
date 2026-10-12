"""Four-market factories: one continuous episode per calendar month.

There is one factory per split specification (see ``energy_model_v3/splits.py``):
four cross-validation folds and the final factory with the test month.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
import yaml

from env.ramp_v6 import clock
from env.ramp_v6.environment import RampAwareEnv
from env.ramp_v6.models import (
    HISTORY_HOURS,
    FrozenRampStats,
    RampProtocol,
    SiteConfig,
    WorkloadTrace,
)
from env.ramp_v6.panel import CanonicalMarketPanel
from env.ramp_v6.objective import JointObjective, NORMALIZATION_METHOD, OBJECTIVE_VERSION
from ramp_rl.contract import EnvRequest
from energy_model_v3.splits import (
    FINAL_SPEC, REFERENCE_CELL, SPECS, TRAIN_FLEET, fleet_for_split,
)


ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_NAMESPACE = "four_market_joint_v3"
PROTOCOL_PATH = ROOT / "env" / "protocols" / "four_market_v2.yaml"
MARKET_TO_CELL = TRAIN_FLEET
MARKETS = tuple(market for market, _ in MARKET_TO_CELL)
CELLS = tuple(cell for _, cell in MARKET_TO_CELL)
CELL_CAPACITY_PATH = ROOT / "data" / "cell_capacity.json"
# Cell c's site, the largest training site, has compute capacity 1 and this
# reference power; every other site, training or test, scales both by its
# cell's CPU capacity relative to cell c.
REFERENCE_POWER_MW = 500.0
DEADLINE_WINDOW_SLOTS = (24, 24, 24, 24)


def factory_root(spec: str = FINAL_SPEC) -> Path:
    """Generated factory directory of one split specification."""
    if spec not in SPECS:
        raise ValueError(f"unknown split specification {spec!r}; expected one of {SPECS}")
    return ROOT / "output" / ARTIFACT_NAMESPACE / spec / "factory"


def site_capacities(cells: tuple[str, ...] = CELLS) -> tuple[float, ...]:
    """Compute capacity of each site in ``cells`` order, relative to cell c."""
    capacities = json.loads(CELL_CAPACITY_PATH.read_text(encoding="utf-8"))["cells"]
    reference = float(capacities[REFERENCE_CELL]["cpu_capacity"])
    return tuple(float(capacities[cell]["cpu_capacity"]) / reference for cell in cells)


def objective_from_factory(
    payload: dict[str, Any], ramp_weight: float | None = None
) -> JointObjective:
    if payload.get("version") != OBJECTIVE_VERSION:
        raise ValueError("stale factory: rebuild the joint-objective factory")
    calibration = payload["objective_calibration"]
    if calibration["normalization"] != NORMALIZATION_METHOD:
        raise ValueError("factory objective normalization is incompatible")
    if sorted(calibration["fit_months"]) != sorted(payload["windows"]["train"]):
        raise ValueError("objective references must use exactly the training months")
    configuration = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))["objective"]
    if (
        configuration["version"] != OBJECTIVE_VERSION
        or configuration["normalization"] != NORMALIZATION_METHOD
        or configuration["ramp_horizons_hours"] != [1]
        or configuration["ramp_weights"] != {1: 1.0}
        or configuration["market_aggregation"] != "sum"
        or configuration["ramp_time_aggregation"] != "sum_over_month"
        or configuration["ramp_quantity"] != "incremental_squared_impact"
        or configuration["peak_time_aggregation"] != "maximum_per_region_over_decision_month"
        or configuration["peak_quantity"] != "adjusted_maximum_minus_original_maximum"
    ):
        raise ValueError("protocol objective is incompatible with the factory")
    weights = configuration["component_weights"]
    objective = JointObjective(
        ramp_weight=float(weights["ramp"]) if ramp_weight is None else ramp_weight,
        peak_weight=float(weights["net_load_peak"]) if ramp_weight is None else 1.0 - ramp_weight,
        ramp_reference=float(calibration["ramp_reference"]),
        peak_reference=float(calibration["peak_reference"]),
    )
    objective.validate()
    return objective


def month_hours(month_id: str) -> int:
    """Number of hourly decision slots in a simulated month such as ``2025-05``."""
    return clock.month_hours(month_id)


def _load_window(
    panel_path: Path, fixture_path: Path, frozen_stats_path: Path
) -> tuple[CanonicalMarketPanel, list[SiteConfig], WorkloadTrace, FrozenRampStats]:
    if not panel_path.is_file() or not fixture_path.is_file() or not frozen_stats_path.is_file():
        raise FileNotFoundError("four-market factory references a missing input")
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    workload = fixture["workload"]
    stats = json.loads(frozen_stats_path.read_text(encoding="utf-8"))
    stats["fit_months"] = tuple(stats["fit_months"])
    return (
        CanonicalMarketPanel.from_csv(panel_path),
        [SiteConfig(**site) for site in fixture["sites"]],
        WorkloadTrace(
            service_arrivals=np.asarray(workload["service_arrivals"], dtype=np.float64),
            batch_arrivals=np.asarray(workload["batch_arrivals"], dtype=np.float64),
            batch_deadline_hours=np.asarray(workload["batch_deadline_hours"], dtype=np.int64),
            warm_power_mw=np.asarray(workload["warm_power_mw"], dtype=np.float64),
        ),
        FrozenRampStats(**stats),
    )


class FourMarketV2WindowEnv(gym.Env):
    """Factory wrapper that chooses a raw-workload train, validation, or test month."""

    metadata = {"render_modes": []}

    def __init__(
        self, payload: dict[str, Any], request: EnvRequest, objective: JointObjective
    ) -> None:
        super().__init__()
        self.payload = payload
        self.request = request
        self.objective = objective
        self._window_ids = tuple(sorted(payload["windows"][request.split]))
        if request.window_id is not None and request.window_id not in self._window_ids:
            raise ValueError(f"factory has no {request.window_id!r} in {request.split}")
        self._rng = np.random.default_rng(request.seed + 1009 * request.rank)
        self._order: list[str] = []
        self._current = self._new_window(self._initial_window_id())
        self.action_space = self._current.action_space
        self.observation_space = self._current.observation_space

    def _initial_window_id(self) -> str:
        if self.request.window_id is not None:
            return self.request.window_id
        return self._window_ids[(self.request.seed + self.request.rank) % len(self._window_ids)]

    def _next_window_id(self, options: dict[str, Any]) -> str:
        requested = options.get("window_id", self.request.window_id)
        if requested is not None:
            if requested not in self._window_ids:
                raise ValueError(f"window {requested!r} is outside {self.request.split}")
            return str(requested)
        if self.request.split != "train":
            return self._initial_window_id()
        if not self._order:
            self._order = [self._window_ids[index] for index in self._rng.permutation(len(self._window_ids))]
        return self._order.pop()

    def _new_window(self, window_id: str) -> RampAwareEnv:
        record = self.payload["windows"][self.request.split][window_id]
        panel, sites, workload, stats = _load_window(
            ROOT / record["panel_path"],
            ROOT / record["fixture_path"],
            ROOT / self.payload["frozen_stats_path"],
        )
        fleet = fleet_for_split(self.request.split)
        if tuple(site.market_id for site in sites) != MARKETS:
            raise ValueError("site mapping is invalid")
        if tuple(record["workload_cells"]) != tuple(cell for _, cell in fleet):
            raise ValueError(f"{self.request.split} windows must be served by their own fleet")
        capacities = site_capacities(tuple(cell for _, cell in fleet))
        if len(sites) != 4 or any(
            not np.isclose(site.compute_capacity, capacity, rtol=0.0, atol=1e-12)
            or not np.isclose(site.rated_power_mw, REFERENCE_POWER_MW * capacity, rtol=0.0, atol=1e-9)
            for site, capacity in zip(sites, capacities)
        ):
            raise ValueError("four-market sites must match the measured cell capacities")
        if tuple(workload.batch_deadline_hours[0]) != DEADLINE_WINDOW_SLOTS:
            raise ValueError("batch execution-window semantics are invalid")
        month_id = str(record["month_id"])
        hours = month_hours(month_id)
        if len(panel.timestamps) != HISTORY_HOURS + hours:
            raise ValueError("window panel must hold the warm history plus the whole month")
        active = panel.timestamps[HISTORY_HOURS:]
        if (
            len(active) != hours
            or set(clock.month_labels(active)) != {month_id}
            or active[0] != clock.month_start(month_id)
        ):
            raise ValueError("window month is invalid")
        return RampAwareEnv(
            panel,
            sites,
            workload,
            stats,
            RampProtocol(
                protocol_id="four-market-joint-net-load-peak-impact-ramp-impact-v3",
                objective=self.objective,
            ),
            episode_context={
                "split": self.request.split,
                "window_id": window_id,
                "month_id": month_id,
                "month": int(record["month"]),
                "hours": hours,
                "chronological": True,
                "forecast_model": self.payload["forecast_model"],
                "future_realized_features_exposed": False,
                "markets": list(MARKETS),
                "workload_cells": list(record["workload_cells"]),
                "factory_spec": self.payload["spec"],
                "total_rated_power_mw": float(sum(site.rated_power_mw for site in sites)),
                "factory_id": self.payload["input_digest"],
            },
        )

    def ramp_rl_contract(self) -> dict[str, Any]:
        return self._current.ramp_rl_contract()

    def evaluation_action(self, name: str) -> np.ndarray:
        return self._current.evaluation_action(name)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed + 1009 * self.request.rank)
            self._order = []
        window_id = self._next_window_id(dict(options or {}))
        self._current.close()
        self._current = self._new_window(window_id)
        if self._current.action_space != self.action_space or self._current.observation_space != self.observation_space:
            raise ValueError("factory windows have inconsistent spaces")
        requested = dict(options or {})
        requested.update({"split": self.request.split, "window_id": window_id})
        return self._current.reset(seed=seed, options=requested)

    def step(self, action: np.ndarray):
        return self._current.step(action)

    def close(self) -> None:
        self._current.close()


def make_four_market_env(
    request: EnvRequest, *, spec: str = FINAL_SPEC, ramp_weight: float | None = None
) -> FourMarketV2WindowEnv:
    """Build the four-market environment of one split specification."""
    if request.split not in {"train", "validation", "test"}:
        raise ValueError("four-market splits are train, validation, and test")
    payload = json.loads((factory_root(spec) / "factory.json").read_text(encoding="utf-8"))
    if payload.get("spec") != spec:
        raise ValueError("factory specification does not match the request")
    if not payload.get("windows", {}).get(request.split):
        raise ValueError(f"factory {spec!r} has no {request.split} windows")
    return FourMarketV2WindowEnv(payload, request, objective_from_factory(payload, ramp_weight))
