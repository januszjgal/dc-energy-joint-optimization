"""Deterministic semantic-action fixture; it does not duplicate v6 internals."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ramp_rl.contract import CONTRACT_VERSION, SEMANTIC_ACTION_ID, EnvRequest

INTERVALS_PER_HOUR = 12
HISTORY_STEPS = 3 * INTERVALS_PER_HOUR
TAIL_STEPS = 3 * INTERVALS_PER_HOUR
SPLIT_MONTHS = {
    "train": (1, 2, 3, 4, 5, 6),
    "validation": (7, 8),
    "test": (9, 10),
}


class DeterministicRampFixtureEnv(gym.Env[np.ndarray, np.ndarray]):
    """Small causal environment used only to test the trainer/evidence boundary."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        request: EnvRequest,
        *,
        decision_steps: int = 48,
    ):
        super().__init__()
        self.request = request
        self.decision_steps = int(decision_steps)
        self.action_space = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.full(9, -20.0, dtype=np.float32),
            high=np.full(9, 20.0, dtype=np.float32),
            dtype=np.float32,
        )
        self._rng = np.random.default_rng(request.seed)
        self._step = 0
        self._backlog = 0.0
        self._power_history: list[float] = []
        self._month = SPLIT_MONTHS[request.split][0]
        self._window_id = ""
        self._energy_cost = 0.0
        self._status_quo_cost = 0.0
        self._terminal = False
        self._next_action_provenance = "agent_semantic"
        self._lagrangian_multiplier = float(request.lagrangian_multiplier)
        self._make_series()

    def ramp_rl_contract(self) -> dict[str, Any]:
        return {
            "version": CONTRACT_VERSION,
            "protocol_id": "ramp-v6-pure-rl-frozen-v1",
            "semantic_feasible_action": True,
            "semantic_action_id": SEMANTIC_ACTION_ID,
            "raw_redundant_projected_logits": False,
            "history_hours": 3,
            "terminal_tail_hours": 3,
            "actual_terminal": True,
            "interval_minutes": 5,
            "decision_steps": self.decision_steps,
            "action_shape": list(self.action_space.shape),
            "action_low": self.action_space.low.tolist(),
            "action_high": self.action_space.high.tolist(),
        }

    def set_lagrangian_multiplier(self, value: float) -> None:
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("Lagrangian multiplier must be finite and non-negative")
        self._lagrangian_multiplier = float(value)

    def _make_series(self) -> None:
        total = HISTORY_STEPS + self.decision_steps + TAIL_STEPS
        index = np.arange(total, dtype=np.float64)
        phase = 0.19 * self._month
        grid = 2.0 + 0.42 * np.sin(index / 9.0 + phase)
        grid += 0.28 * (index >= HISTORY_STEPS + self.decision_steps // 2)
        forecast = grid + 0.04 * np.sin(index / 5.0 + 0.7)
        price = 1.0 + 0.15 * np.cos(index / 13.0 + phase)
        self._grid = grid
        self._forecast = forecast
        self._price = price
        source_payload = json.dumps(
            {
                "fixture_version": 1,
                "month": self._month,
                "grid": np.round(grid, 10).tolist(),
                "forecast": np.round(forecast, 10).tolist(),
            },
            sort_keys=True,
        ).encode("utf-8")
        self._source_hash = hashlib.sha256(source_payload).hexdigest()

    def _observation(self) -> np.ndarray:
        absolute = HISTORY_STEPS + min(self._step, self.decision_steps - 1)
        history = self._power_history
        last = history[-1]
        lag_1h = history[-INTERVALS_PER_HOUR] if len(history) >= INTERVALS_PER_HOUR else history[0]
        lag_3h = history[-HISTORY_STEPS] if len(history) >= HISTORY_STEPS else history[0]
        remaining = (self.decision_steps - self._step) / self.decision_steps
        return np.asarray(
            [
                self._grid[absolute],
                self._forecast[absolute],
                self._price[absolute],
                self._backlog,
                last,
                last - lag_1h,
                last - lag_3h,
                remaining,
                np.sin(2.0 * np.pi * absolute / 288.0),
            ],
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed + 1009 * self.request.rank)
        options = dict(options or {})
        split = str(options.get("split", self.request.split))
        if split != self.request.split:
            raise ValueError("fixture request split is immutable")
        months = SPLIT_MONTHS[split]
        if "window_id" in options:
            token = str(options["window_id"])
            month = int(token.split("-")[1])
            if month not in months:
                raise ValueError("requested window is outside the split")
            self._window_id = token
        elif split == "train":
            month = int(self._rng.choice(months))
            self._window_id = f"m-{month:02d}-random-{int(self._rng.integers(0, 10_000)):04d}"
        else:
            month = months[0]
            self._window_id = f"m-{month:02d}-sealed-0000"
        self._month = month
        self._make_series()
        warm_grid = self._grid[:HISTORY_STEPS]
        self._power_history = (warm_grid + 0.20).tolist()
        self._step = 0
        self._backlog = 0.0
        self._energy_cost = 0.0
        self._status_quo_cost = 0.0
        self._terminal = False
        self._next_action_provenance = "agent_semantic"
        return self._observation(), {
            "episode_context": {
                "split": split,
                "window_id": self._window_id,
                "month": self._month,
                "history_hours": 3,
                "terminal_tail_hours": 3,
                "chronological": True,
                "forecast_model": "deterministic-fixture-causal-v1",
                "forecast_vintage": "fixture-2026-08-08",
                "source_hashes": {"fixture_timeseries": self._source_hash},
                "future_realized_features_exposed": False,
            }
        }

    def _metrics_for_power(self, power: float, absolute: int) -> tuple[float, float, float]:
        one_hour = self._power_history[-INTERVALS_PER_HOUR]
        three_hour = self._power_history[-HISTORY_STEPS]
        h1 = abs((self._grid[absolute] + power) - (self._grid[absolute - INTERVALS_PER_HOUR] + one_hour))
        h3 = abs((self._grid[absolute] + power) - (self._grid[absolute - HISTORY_STEPS] + three_hour))
        baseline_h1 = abs(self._grid[absolute] - self._grid[absolute - INTERVALS_PER_HOUR])
        baseline_h3 = abs(self._grid[absolute] - self._grid[absolute - HISTORY_STEPS])
        incremental = 0.5 * ((h1 - baseline_h1) + (h3 - baseline_h3))
        return float(h1), float(h3), float(incremental)

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._terminal:
            raise RuntimeError("step called after terminal")
        fraction = float(np.asarray(action, dtype=np.float32)[0])
        arrival = 0.24 + 0.03 * np.sin((self._step + self._month) / 7.0)
        self._backlog += max(arrival, 0.0)
        remaining_steps = self.decision_steps - self._step
        mandatory = max(0.0, self._backlog - 0.55 * (remaining_steps - 1))
        maximum = min(0.55, self._backlog)
        service = mandatory + fraction * max(maximum - mandatory, 0.0)
        self._backlog = max(self._backlog - service, 0.0)
        absolute = HISTORY_STEPS + self._step
        power = 0.20 + 0.65 * service
        h1, h3, incremental = self._metrics_for_power(power, absolute)
        self._power_history.append(power)
        cost = float(self._price[absolute] * power)
        status_quo_power = 0.20 + 0.65 * (mandatory + 0.5 * max(maximum - mandatory, 0.0))
        status_quo_cost = float(self._price[absolute] * status_quo_power)
        self._energy_cost += cost
        self._status_quo_cost += status_quo_cost
        budget = status_quo_cost + abs(status_quo_cost) * (
            self.request.epsilon_pct / 100.0
        )
        violation = max(cost - budget, 0.0)
        reward = -(incremental + self._lagrangian_multiplier * violation)
        self._step += 1
        terminated = self._step == self.decision_steps
        tail_h1: list[float] = []
        tail_h3: list[float] = []
        tail_incremental: list[float] = []
        tail_energy: list[float] = []
        tail_status_quo_energy: list[float] = []
        if terminated:
            for tail in range(TAIL_STEPS):
                tail_absolute = HISTORY_STEPS + self.decision_steps + tail
                tail_power = 0.20
                tail_h1_value, tail_h3_value, tail_value = self._metrics_for_power(
                    tail_power, tail_absolute
                )
                tail_h1.append(tail_h1_value)
                tail_h3.append(tail_h3_value)
                tail_incremental.append(tail_value)
                tail_cost = float(self._price[tail_absolute] * tail_power)
                tail_energy.append(tail_cost)
                tail_status_quo_energy.append(tail_cost)
                self._power_history.append(tail_power)
            reward -= sum(tail_incremental)
            self._terminal = True
        provenance = self._next_action_provenance
        self._next_action_provenance = "agent_semantic"
        info = {
            "window_id": self._window_id,
            "month": self._month,
            "ramp_h1_adjusted": h1,
            "ramp_h3_adjusted": h3,
            "incremental_ramp_impact": incremental,
            "energy_cost": cost,
            "status_quo_energy_cost": status_quo_cost,
            "energy_budget_violation": violation,
            "service_unserved": 0.0,
            "batch_unfinished": self._backlog if terminated else 0.0,
            "batch_expired": 0.0,
            "certificate_violations": 0,
            "emergency_feasibility": False,
            "semantic_adjustment_l2": 0.0,
            "action_provenance": provenance,
            "tail_complete": terminated,
            "actual_terminal": terminated,
            "terminal_work": 0.0,
            "terminal_tail_ramp_h1_adjusted": tail_h1,
            "terminal_tail_ramp_h3_adjusted": tail_h3,
            "terminal_tail_incremental_ramp_impact": tail_incremental,
            "terminal_tail_energy_cost": tail_energy,
            "terminal_tail_status_quo_energy_cost": tail_status_quo_energy,
            "terminal_tail_service_unserved": [0.0] * len(tail_h1),
            "terminal_tail_batch_unfinished": [0.0] * len(tail_h1),
            "terminal_tail_batch_expired": [0.0] * len(tail_h1),
            "terminal_tail_certificate_violations": [0] * len(tail_h1),
            "terminal_tail_emergency_feasibility": [False] * len(tail_h1),
            "terminal_tail_semantic_adjustment_l2": [0.0] * len(tail_h1),
            "deferrable_pre_service": service if absolute < HISTORY_STEPS + self.decision_steps // 2 else 0.0,
            "dc_power_during_realized_ramp": power if absolute >= HISTORY_STEPS + self.decision_steps // 2 else 0.0,
            "step_ramp_h1_adjusted": [h1],
            "step_ramp_h3_adjusted": [h3],
            "step_incremental_ramp_impact": [incremental],
            "step_energy_cost": [cost],
            "step_service_unserved": [0.0],
            "step_batch_unfinished": [self._backlog],
            "step_batch_expired": [0.0],
            "step_certificate_violations": [0],
        }
        observation = self._observation() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        return observation, float(reward), terminated, False, info

    def evaluation_action(self, name: str) -> np.ndarray:
        if name != "status_quo":
            raise ValueError("fixture exposes only the evaluation-only status_quo controller")
        self._next_action_provenance = "evaluation_status_quo"
        return np.asarray([0.5], dtype=np.float32)


def make_fixture_env(request: EnvRequest) -> DeterministicRampFixtureEnv:
    return DeterministicRampFixtureEnv(request)
