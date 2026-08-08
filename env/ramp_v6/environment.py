"""Gymnasium-compatible hourly ramp-aware pure-RL environment."""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from env.ramp_v6.models import (
    FORECAST_HOURS,
    HORIZONS,
    EDFQueue,
    FrozenRampStats,
    RampProtocol,
    SiteConfig,
    WorkloadTrace,
)
from env.ramp_v6.panel import CanonicalMarketPanel
from env.ramp_v6.projection import ProjectedAction, project_action
from env.ramp_v6.reward import closed_window_terms


class RampAwareEnv(gym.Env):
    """Hourly controller with preference-only actions and a hard decoder.

    The policy never sees realized future market values. The only forward
    signals are timestamped forecasts whose issue time is no later than the
    current controller timestamp.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        panel: CanonicalMarketPanel,
        sites: list[SiteConfig],
        workload: WorkloadTrace,
        frozen_stats: FrozenRampStats,
        protocol: RampProtocol | None = None,
    ):
        super().__init__()
        self.panel = panel
        self.sites = tuple(sites)
        self.workload = workload
        self.stats = frozen_stats
        self.protocol = protocol or RampProtocol()
        self.protocol.validate()
        if not self.sites:
            raise ValueError("at least one site is required")
        if len({site.site_id for site in self.sites}) != len(self.sites):
            raise ValueError("site IDs must be unique")
        for site in self.sites:
            site.validate()
        site_markets = {site.market_id for site in self.sites}
        panel_markets = set(panel.markets)
        if site_markets != panel_markets:
            raise ValueError("sites must cover every panel market")
        self.stats.validate(panel_markets)
        self.n_sites = len(self.sites)
        self.action_steps = len(panel.timestamps) - self.protocol.history_hours
        if self.action_steps <= self.protocol.terminal_tail_hours:
            raise ValueError("panel must include history, active hours, and tail")
        self.main_steps = self.action_steps - self.protocol.terminal_tail_hours
        workload.validate(
            self.action_steps,
            self.n_sites,
            self.protocol.history_hours,
        )
        tail = slice(self.main_steps, self.action_steps)
        if np.any(workload.service_arrivals[tail] != 0.0) or np.any(
            workload.batch_arrivals[tail] != 0.0
        ):
            raise ValueError("terminal 3h tail must contain no new arrivals")

        self._capacity = np.asarray(
            [site.compute_capacity for site in self.sites], dtype=np.float64
        )
        self._market_site_indices = {
            market: np.asarray(
                [
                    index
                    for index, site in enumerate(self.sites)
                    if site.market_id == market
                ],
                dtype=np.int64,
            )
            for market in panel.markets
        }
        self.observation_schema = tuple(self._build_observation_schema())
        self.action_schema = tuple(
            [f"service_preference:{site.site_id}" for site in self.sites]
            + ["optional_total_batch_execution"]
            + [f"batch_destination_preference:{site.site_id}" for site in self.sites]
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.observation_schema),),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-6.0,
            high=6.0,
            shape=(len(self.action_schema),),
            dtype=np.float32,
        )
        self.queue = EDFQueue()
        self._step = 0
        self._arrivals_loaded = False
        self._site_power_history: list[np.ndarray] = []
        self._market_power_history: dict[str, list[float]] = {}
        self._episode_history: list[dict[str, Any]] = []

    @property
    def current_timestamp(self) -> pd.Timestamp:
        return self.panel.timestamps[self.protocol.history_hours + self._step]

    def _build_observation_schema(self) -> list[str]:
        names: list[str] = []
        for market in self.panel.markets:
            for quantity in ("gross", "net"):
                for lag in range(4):
                    names.append(f"{market}:{quantity}_level_z_lag{lag}")
            for horizon in HORIZONS:
                names.append(f"{market}:native_ramp_{horizon}h_fraction_s_per_hour")
            for quantity in ("gross", "net"):
                for hour in FORECAST_HOURS:
                    names.append(f"{market}:forecast_{quantity}_h{hour}_z")
                names.append(f"{market}:forecast_{quantity}_max_up_3h_fraction_s")
            names.extend(
                [
                    f"{market}:da_lmp_usd_per_mwh_scaled",
                    f"{market}:forecast_vintage_age_hours",
                    f"{market}:quality_ok",
                ]
            )
        for index, site in enumerate(self.sites):
            names.extend(
                [
                    f"{site.site_id}:service_arrival_capacity_fraction",
                    f"{site.site_id}:batch_arrival_capacity_fraction",
                    f"{site.site_id}:queued_batch_capacity_fraction",
                    f"{site.site_id}:previous_power_rated_fraction",
                    f"{site.site_id}:compute_capacity",
                    f"{site.site_id}:rated_power_100mw_units",
                    f"{site.site_id}:idle_power_fraction",
                    f"{site.site_id}:dynamic_power_fraction",
                ]
            )
        names.extend(
            f"batch_queue_deadline_le_{edge}h_capacity_fraction"
            for edge in self.protocol.deadline_bucket_hours
        )
        names.extend(
            [
                "batch_queue_deadline_beyond_24h_capacity_fraction",
                "episode_progress",
                "terminal_tail_active",
                "hour_sin",
                "hour_cos",
                "day_of_week_sin",
                "day_of_week_cos",
            ]
        )
        return names

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options
        self.queue = EDFQueue()
        self._step = 0
        self._arrivals_loaded = False
        self._site_power_history = [
            self.workload.warm_power_mw[index].astype(np.float64).copy()
            for index in range(self.protocol.history_hours)
        ]
        self._market_power_history = {
            market: [
                float(
                    self.workload.warm_power_mw[index, site_indices].sum()
                )
                for index in range(self.protocol.history_hours)
            ]
            for market, site_indices in self._market_site_indices.items()
        }
        self._episode_history = []
        self._load_current_arrivals()
        return self._observation(), {
            "protocol_id": self.protocol.protocol_id,
            "timestamp_utc": self.current_timestamp.isoformat(),
        }

    def _load_current_arrivals(self) -> None:
        if self._arrivals_loaded:
            return
        for origin in range(self.n_sites):
            amount = float(self.workload.batch_arrivals[self._step, origin])
            deadline = min(
                self._step
                + int(self.workload.batch_deadline_hours[self._step, origin]),
                self.action_steps - 1,
            )
            self.queue.add(amount, origin, deadline)
        self._arrivals_loaded = True

    def _observation(self) -> np.ndarray:
        panel_index = self.protocol.history_hours + self._step
        current = self.panel.observation_rows(panel_index)
        values: list[float] = []
        for market in self.panel.markets:
            row = current.loc[market]
            scale = self.stats.gross_q95_mw[market]
            for quantity, mean_map, std_map in (
                (
                    "gross_demand_mw",
                    self.stats.gross_level_mean_mw,
                    self.stats.gross_level_std_mw,
                ),
                (
                    "net_load_mw",
                    self.stats.net_level_mean_mw,
                    self.stats.net_level_std_mw,
                ),
            ):
                for lag in range(4):
                    history_row = self.panel.observation_rows(panel_index - lag)
                    values.append(
                        (
                            float(history_row.loc[market, quantity])
                            - mean_map[market]
                        )
                        / std_map[market]
                    )
            for horizon in HORIZONS:
                then = self.panel.observation_rows(panel_index - horizon)
                values.append(
                    (
                        float(row["net_load_mw"])
                        - float(then.loc[market, "net_load_mw"])
                    )
                    / (scale * horizon)
                )
            for quantity, mean_map, std_map, current_name in (
                (
                    "gross",
                    self.stats.gross_level_mean_mw,
                    self.stats.gross_level_std_mw,
                    "gross_demand_mw",
                ),
                (
                    "net",
                    self.stats.net_level_mean_mw,
                    self.stats.net_level_std_mw,
                    "net_load_mw",
                ),
            ):
                forecasts = [
                    float(row[f"forecast_{quantity}_h{hour}_mw"])
                    for hour in FORECAST_HOURS
                ]
                values.extend(
                    (forecast - mean_map[market]) / std_map[market]
                    for forecast in forecasts
                )
                path = [float(row[current_name]), *forecasts]
                max_up = max(
                    (path[index + 1] - path[index]) / scale
                    for index in range(3)
                )
                values.append(max_up)
            values.extend(
                [
                    float(row["da_lmp_usd_per_mwh"]) / 100.0,
                    (
                        self.current_timestamp - row["forecast_issue_time_utc"]
                    ).total_seconds()
                    / 3600.0,
                    float(bool(row["quality_ok"])),
                ]
            )

        queued_by_origin = self.queue.by_origin(self.n_sites)
        previous_power = self._site_power_history[-1]
        for index, site in enumerate(self.sites):
            capacity = site.compute_capacity
            values.extend(
                [
                    float(self.workload.service_arrivals[self._step, index])
                    / capacity,
                    float(self.workload.batch_arrivals[self._step, index])
                    / capacity,
                    float(queued_by_origin[index]) / capacity,
                    float(previous_power[index]) / site.rated_power_mw,
                    capacity,
                    site.rated_power_mw / 100.0,
                    site.idle_power_fraction,
                    site.dynamic_power_fraction,
                ]
            )
        total_capacity = float(self._capacity.sum())
        values.extend(
            self.queue.histogram(
                self._step, self.protocol.deadline_bucket_hours
            )
            / total_capacity
        )
        timestamp = self.current_timestamp
        hour_angle = 2.0 * math.pi * timestamp.hour / 24.0
        day_angle = 2.0 * math.pi * timestamp.dayofweek / 7.0
        values.extend(
            [
                self._step / max(self.action_steps - 1, 1),
                float(self._step >= self.main_steps),
                math.sin(hour_angle),
                math.cos(hour_angle),
                math.sin(day_angle),
                math.cos(day_angle),
            ]
        )
        observation = np.asarray(values, dtype=np.float32)
        if observation.shape != self.observation_space.shape:
            raise RuntimeError(
                f"observation schema mismatch: {observation.shape} "
                f"!= {self.observation_space.shape}"
            )
        return observation

    def _guaranteed_future_batch_capacity_by_deadline(
        self,
    ) -> dict[int, float]:
        total_capacity = float(self._capacity.sum())
        carried_fraction = (
            self.protocol.guaranteed_batch_capacity_fraction
            - self.protocol.batch_arrival_envelope_fraction_of_fleet
        )
        result: dict[int, float] = {}
        for deadline in self.queue.deadlines:
            capacity = 0.0
            stop = min(deadline, self.action_steps - 1)
            for future_step in range(self._step + 1, stop + 1):
                capacity += (
                    total_capacity
                    if future_step >= self.main_steps
                    else total_capacity * carried_fraction
                )
            result[deadline] = capacity
        return result

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != self.action_space.shape:
            raise ValueError(f"action must have shape {self.action_space.shape}")
        if np.any(~np.isfinite(action)):
            raise ValueError("action must contain only finite values")
        if np.any(action < self.action_space.low) or np.any(
            action > self.action_space.high
        ):
            raise ValueError("action is outside the frozen [-6, 6] bounds")
        self._load_current_arrivals()
        service_total = float(self.workload.service_arrivals[self._step].sum())
        fleet_capacity = float(self._capacity.sum())
        if self._step < self.main_steps:
            service_limit = (
                fleet_capacity
                * self.protocol.service_envelope_fraction_of_fleet
            )
            batch_arrival = float(
                self.workload.batch_arrivals[self._step].sum()
            )
            batch_arrival_limit = (
                fleet_capacity
                * self.protocol.batch_arrival_envelope_fraction_of_fleet
            )
            if service_total > service_limit + self.protocol.tolerance:
                raise ValueError(
                    "service arrival exceeds frozen causal feasibility envelope"
                )
            if batch_arrival > batch_arrival_limit + self.protocol.tolerance:
                raise ValueError(
                    "batch arrival exceeds frozen causal feasibility envelope"
                )
        projected = project_action(
            action,
            service_total,
            self._capacity,
            self.queue,
            self._step,
            self.action_steps - 1,
            self.n_sites,
            guaranteed_future_batch_capacity_by_deadline=(
                self._guaranteed_future_batch_capacity_by_deadline()
            ),
        )
        if self.queue.due_by(self._step + 1) > self.protocol.tolerance:
            raise RuntimeError("EDF deadline work remained after mandatory execution")
        work = projected.service + projected.batch_by_destination
        if np.any(work > self._capacity + self.protocol.tolerance):
            raise RuntimeError("decoder emitted a capacity violation")
        site_power = np.asarray(
            [site.power_mw(work[index]) for index, site in enumerate(self.sites)],
            dtype=np.float64,
        )
        self._site_power_history.append(site_power)
        market_power = {
            market: float(site_power[site_indices].sum())
            for market, site_indices in self._market_site_indices.items()
        }
        for market, power in market_power.items():
            self._market_power_history[market].append(power)
        info, reward = self._step_info(projected, work, site_power, market_power)
        self._episode_history.append(info)

        if abs(self.queue.conservation_error()) > self.protocol.tolerance:
            raise RuntimeError("batch work conservation failed")
        terminated = self._step == self.action_steps - 1
        if terminated:
            if self.queue.total > self.protocol.tolerance:
                raise RuntimeError("terminal batch queue is not empty")
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            self._step += 1
            self._arrivals_loaded = False
            self._load_current_arrivals()
            observation = self._observation()
        return observation, reward, terminated, False, info

    def _step_info(
        self,
        projected: ProjectedAction,
        work: np.ndarray,
        site_power: np.ndarray,
        market_power: dict[str, float],
    ) -> tuple[dict[str, Any], float]:
        panel_index = self.protocol.history_hours + self._step
        current = self.panel.observation_rows(panel_index)
        per_market: dict[str, dict[str, Any]] = {}
        weighted_impact = 0.0
        weighted_tail = 0.0
        total_da_cost = 0.0
        for market in self.panel.markets:
            row = current.loc[market]
            scale = self.stats.gross_q95_mw[market]
            windows: dict[str, dict[str, float]] = {}
            for horizon in HORIZONS:
                past_row = self.panel.observation_rows(panel_index - horizon).loc[
                    market
                ]
                terms = closed_window_terms(
                    net_now_mw=float(row["net_load_mw"]),
                    net_then_mw=float(past_row["net_load_mw"]),
                    power_now_mw=market_power[market],
                    power_then_mw=self._market_power_history[market][
                        -(horizon + 1)
                    ],
                    gross_q95_scale_mw=scale,
                    horizon_hours=horizon,
                    tail_q90_fraction_s_per_hour=(
                        self.stats.native_abs_ramp_q90_fraction_s_per_hour[
                            market
                        ][horizon]
                    ),
                )
                windows[f"{horizon}h"] = {
                    "native_ramp_mw": (
                        float(row["net_load_mw"])
                        - float(past_row["net_load_mw"])
                    ),
                    "adjusted_ramp_mw": (
                        float(row["net_load_mw"])
                        + market_power[market]
                        - float(past_row["net_load_mw"])
                        - self._market_power_history[market][-(horizon + 1)]
                    ),
                    **terms.__dict__,
                }
                weighted_impact += (
                    self.protocol.ramp_weights[horizon]
                    * terms.incremental_squared_impact
                )
                weighted_tail += (
                    self.protocol.ramp_weights[horizon]
                    * terms.incremental_tail_burden
                )
            da_cost = (
                market_power[market] * float(row["da_lmp_usd_per_mwh"])
            )
            total_da_cost += da_cost
            forecast_errors = {}
            for horizon in HORIZONS:
                issue_index = panel_index - horizon
                if issue_index >= 0:
                    issue = self.panel.observation_rows(issue_index).loc[market]
                    forecast_errors[f"net_h{horizon}_abs_error_mw"] = abs(
                        float(issue[f"forecast_net_h{horizon}_mw"])
                        - float(row["net_load_mw"])
                    )
            gross = float(row["gross_demand_mw"])
            per_market[market] = {
                "power_mw": market_power[market],
                "gross_demand_mw": gross,
                "net_load_mw": float(row["net_load_mw"]),
                "market_scale_mw": float(row["market_scale_mw"]),
                "power_over_frozen_s": market_power[market] / scale,
                "power_over_gross_demand": market_power[market] / gross,
                "power_over_market_scale": (
                    market_power[market] / float(row["market_scale_mw"])
                ),
                "da_lmp_usd_per_mwh": float(row["da_lmp_usd_per_mwh"]),
                "da_energy_cost_usd": da_cost,
                "windows": windows,
                "forecast_errors": forecast_errors,
            }
        macro_divisor = len(self.panel.markets)
        weighted_impact /= macro_divisor
        weighted_tail /= macro_divisor
        scalar_objective = weighted_impact + self.protocol.tail_weight * weighted_tail
        reward = -self.protocol.ramp_reward_scale * scalar_objective
        info: dict[str, Any] = {
            "protocol_id": self.protocol.protocol_id,
            "timestamp_utc": self.current_timestamp.isoformat(),
            "step": self._step,
            "terminal_tail_active": self._step >= self.main_steps,
            "new_arrivals": float(
                self.workload.service_arrivals[self._step].sum()
                + self.workload.batch_arrivals[self._step].sum()
            ),
            "scalar_reward": reward,
            "weighted_incremental_ramp_impact": weighted_impact,
            "weighted_incremental_tail_burden": weighted_tail,
            "da_energy_cost_usd": total_da_cost,
            "service_completed": float(projected.service.sum()),
            "service_arrived": float(
                self.workload.service_arrivals[self._step].sum()
            ),
            "batch_completed": float(projected.batch_by_destination.sum()),
            "batch_arrived": float(
                self.workload.batch_arrivals[self._step].sum()
            ),
            "deadline_missed_work": 0.0,
            "batch_mandatory": projected.mandatory_batch,
            "batch_requested": projected.requested_batch,
            "batch_queue": self.queue.as_dict(),
            "work_conservation_error": self.queue.conservation_error(),
            "capacity_slack": (self._capacity - work).tolist(),
            "service_allocation": projected.service.tolist(),
            "batch_by_origin": projected.batch_by_origin.tolist(),
            "batch_by_destination": projected.batch_by_destination.tolist(),
            "batch_transport": projected.transport.tolist(),
            "per_site": {
                site.site_id: {
                    "market_id": site.market_id,
                    "work": float(work[index]),
                    "capacity": site.compute_capacity,
                    "power_mw": float(site_power[index]),
                }
                for index, site in enumerate(self.sites)
            },
            "per_market": per_market,
        }
        return info, reward

    @property
    def episode_history(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._episode_history)
