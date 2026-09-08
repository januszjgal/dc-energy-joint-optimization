"""One raw-workload four-market factory: one continuous episode per calendar month."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd

from env.ramp_v6.environment import RampAwareEnv
from env.ramp_v6.models import (
    HISTORY_HOURS,
    FrozenRampStats,
    RampProtocol,
    SiteConfig,
    WorkloadTrace,
)
from env.ramp_v6.panel import CanonicalMarketPanel
from ramp_rl.contract import EnvRequest


ROOT = Path(__file__).resolve().parent.parent
FACTORY_ROOT = ROOT / "output" / "four_market_v2" / "factory"
MARKET_TO_CELL = (
    ("CAISO_NP15", "a"),
    ("MISO_MINN_HUB", "b"),
    ("SPP_NORTH_HUB", "c"),
    ("ISONE_NEMA", "d"),
)
MARKETS = tuple(market for market, _ in MARKET_TO_CELL)
CELLS = tuple(cell for _, cell in MARKET_TO_CELL)
RATED_POWER_MW = 500.0
COMPUTE_CAPACITY = 1.0
TOTAL_RATED_POWER_MW = 2_000.0
DEADLINE_WINDOW_SLOTS = (24, 24, 24, 24)


def month_hours(month_id: str) -> int:
    """Number of hourly decision slots in a calendar month such as ``2025-05``."""
    start = pd.Timestamp(f"{month_id}-01T00:00:00Z")
    end = start + pd.offsets.MonthBegin(1)
    return int((end - start) / pd.Timedelta(hours=1))


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
    """Factory wrapper that chooses a raw-workload train or validation month."""

    metadata = {"render_modes": []}

    def __init__(self, payload: dict[str, Any], request: EnvRequest) -> None:
        super().__init__()
        self.payload = payload
        self.request = request
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
        if tuple(site.market_id for site in sites) != MARKETS:
            raise ValueError("site mapping is invalid")
        if len(sites) != 4 or any(
            site.compute_capacity != COMPUTE_CAPACITY or site.rated_power_mw != RATED_POWER_MW
            for site in sites
        ):
            raise ValueError("four-market factory requires four direct 500 MW / K=1 sites")
        if tuple(workload.batch_deadline_hours[0]) != DEADLINE_WINDOW_SLOTS:
            raise ValueError("batch execution-window semantics are invalid")
        month_id = str(record["month_id"])
        hours = month_hours(month_id)
        if len(panel.timestamps) != HISTORY_HOURS + hours:
            raise ValueError("window panel must hold the warm history plus the whole month")
        active = panel.timestamps[HISTORY_HOURS:]
        if len(active) != hours or set(active.strftime("%Y-%m")) != {month_id}:
            raise ValueError("window month is invalid")
        return RampAwareEnv(
            panel,
            sites,
            workload,
            stats,
            RampProtocol(protocol_id="four-market-v2-continuous-month"),
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
                "workload_cells": list(CELLS),
                "total_rated_power_mw": TOTAL_RATED_POWER_MW,
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


def make_four_market_env(request: EnvRequest) -> FourMarketV2WindowEnv:
    """Build the only active raw-workload four-market environment."""
    if request.split not in {"train", "validation"}:
        raise ValueError("four-market permits train and validation only")
    payload = json.loads((FACTORY_ROOT / "factory.json").read_text(encoding="utf-8"))
    if payload.get("windows", {}).get("test"):
        raise ValueError("four-market factory must not expose a test split")
    return FourMarketV2WindowEnv(payload, request)
