"""Frozen data structures for the additive v6 ramp environment."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


HORIZONS = (1, 3)
FORECAST_HOURS = (1, 2, 3)


@dataclass(frozen=True)
class SiteConfig:
    """A compute site attached to one electricity market."""

    site_id: str
    market_id: str
    rated_power_mw: float = 100.0
    compute_capacity: float = 1.0
    idle_power_fraction: float = 0.55
    dynamic_power_fraction: float = 0.45

    def validate(self) -> None:
        values = {
            "rated_power_mw": self.rated_power_mw,
            "compute_capacity": self.compute_capacity,
            "idle_power_fraction": self.idle_power_fraction,
            "dynamic_power_fraction": self.dynamic_power_fraction,
        }
        if not self.site_id or not self.market_id:
            raise ValueError("site_id and market_id must be non-empty")
        if any(not math.isfinite(value) for value in values.values()):
            raise ValueError("site configuration values must be finite")
        if self.rated_power_mw <= 0.0 or self.compute_capacity <= 0.0:
            raise ValueError("site power and compute capacity must be positive")
        if self.idle_power_fraction < 0.0 or self.dynamic_power_fraction < 0.0:
            raise ValueError("power fractions must be non-negative")
        if self.idle_power_fraction + self.dynamic_power_fraction > 1.0 + 1e-12:
            raise ValueError("site power fractions cannot exceed rated power")

    def power_mw(self, work: float) -> float:
        utilization = np.clip(work / self.compute_capacity, 0.0, 1.0)
        return float(
            self.rated_power_mw
            * (
                self.idle_power_fraction
                + self.dynamic_power_fraction * utilization
            )
        )

    def scaled(self, multiplier: float) -> SiteConfig:
        """Scale physical power and compute throughput together."""
        if not math.isfinite(multiplier) or multiplier <= 0.0:
            raise ValueError("scale multiplier must be finite and positive")
        return SiteConfig(
            site_id=self.site_id,
            market_id=self.market_id,
            rated_power_mw=self.rated_power_mw * multiplier,
            compute_capacity=self.compute_capacity * multiplier,
            idle_power_fraction=self.idle_power_fraction,
            dynamic_power_fraction=self.dynamic_power_fraction,
        )


@dataclass(frozen=True)
class FrozenRampStats:
    """Train-only normalizers and native-ramp tail thresholds."""

    fit_start_utc: str
    fit_end_utc: str
    gross_q95_mw: dict[str, float]
    gross_level_mean_mw: dict[str, float]
    gross_level_std_mw: dict[str, float]
    net_level_mean_mw: dict[str, float]
    net_level_std_mw: dict[str, float]
    native_abs_ramp_q90_fraction_s_per_hour: dict[str, dict[int, float]]
    fit_months: tuple[str, ...]
    stats_id: str = "ramp-v6-train-only-q95-q90-v1"

    def validate(self, markets: set[str]) -> None:
        mappings = (
            self.gross_q95_mw,
            self.gross_level_mean_mw,
            self.gross_level_std_mw,
            self.net_level_mean_mw,
            self.net_level_std_mw,
            self.native_abs_ramp_q90_fraction_s_per_hour,
        )
        if any(set(mapping) != markets for mapping in mappings):
            raise ValueError("frozen statistics must cover exactly the panel markets")
        if not self.fit_months:
            raise ValueError("frozen statistics require complete training months")
        for market in markets:
            if self.gross_q95_mw[market] <= 0.0:
                raise ValueError("gross Q95 scale must be positive")
            if self.gross_level_std_mw[market] <= 0.0:
                raise ValueError("gross level standard deviation must be positive")
            if self.net_level_std_mw[market] <= 0.0:
                raise ValueError("net level standard deviation must be positive")
            thresholds = self.native_abs_ramp_q90_fraction_s_per_hour[market]
            if set(thresholds) != set(HORIZONS):
                raise ValueError("tail thresholds must contain 1h and 3h")


@dataclass(frozen=True)
class RampProtocol:
    """Frozen environment and scalar-reward semantics."""

    protocol_id: str = "ramp-v6-pure-rl-frozen-v1"
    history_hours: int = 3
    terminal_tail_hours: int = 3
    deadline_bucket_hours: tuple[int, ...] = (1, 3, 6, 12, 24)
    ramp_weights: dict[int, float] = field(
        default_factory=lambda: {1: 0.40, 3: 0.60}
    )
    tail_weight: float = 0.15
    ramp_reward_scale: float = 1.0
    cost_budget_fraction: float = 0.05
    guaranteed_batch_capacity_fraction: float = 0.25
    service_envelope_fraction_of_fleet: float = 0.75
    batch_arrival_envelope_fraction_of_fleet: float = 0.10
    admission_envelope_enabled: bool = True
    tolerance: float = 1e-9

    def validate(self) -> None:
        if self.history_hours != 3 or self.terminal_tail_hours != 3:
            raise ValueError("v6 requires exactly 3h warm history and terminal tail")
        if set(self.ramp_weights) != set(HORIZONS):
            raise ValueError("ramp weights must contain exactly 1h and 3h")
        if not math.isclose(
            sum(self.ramp_weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("ramp weights must sum to one")
        if self.tail_weight < 0.0 or self.ramp_reward_scale <= 0.0:
            raise ValueError("reward weights must be non-negative with positive scale")
        if self.cost_budget_fraction < 0.0 or self.tolerance <= 0.0:
            raise ValueError("cost budget must be non-negative and tolerance positive")
        if not 0.0 <= self.guaranteed_batch_capacity_fraction <= 1.0:
            raise ValueError("guaranteed batch capacity fraction must be in [0, 1]")
        if not 0.0 <= self.service_envelope_fraction_of_fleet <= 1.0:
            raise ValueError("service envelope fraction must be in [0, 1]")
        if not 0.0 <= self.batch_arrival_envelope_fraction_of_fleet <= 1.0:
            raise ValueError("batch arrival envelope fraction must be in [0, 1]")
        if self.admission_envelope_enabled:
            expected_batch_capacity = 1.0 - self.service_envelope_fraction_of_fleet
            if not math.isclose(
                self.guaranteed_batch_capacity_fraction,
                expected_batch_capacity,
                rel_tol=0.0,
                abs_tol=self.tolerance,
            ):
                raise ValueError(
                    "guaranteed batch capacity must equal one minus service envelope"
                )
            if (
                self.batch_arrival_envelope_fraction_of_fleet
                > self.guaranteed_batch_capacity_fraction + self.tolerance
            ):
                raise ValueError("batch arrival envelope exceeds guaranteed capacity")
        if tuple(sorted(self.deadline_bucket_hours)) != self.deadline_bucket_hours:
            raise ValueError("deadline buckets must be strictly increasing")


@dataclass(frozen=True)
class WorkloadTrace:
    """Hourly service/batch arrivals and warm modeled power.

    Arrays use absolute compute-work units. Scaling a study multiplies arrivals,
    warm power, site compute capacity, and site rated power together.
    """

    service_arrivals: np.ndarray
    batch_arrivals: np.ndarray
    batch_deadline_hours: np.ndarray
    warm_power_mw: np.ndarray

    def validate(self, n_steps: int, n_sites: int, history_hours: int = 3) -> None:
        expected = (n_steps, n_sites)
        if self.service_arrivals.shape != expected:
            raise ValueError(f"service_arrivals must have shape {expected}")
        if self.batch_arrivals.shape != expected:
            raise ValueError(f"batch_arrivals must have shape {expected}")
        if self.batch_deadline_hours.shape != expected:
            raise ValueError(f"batch_deadline_hours must have shape {expected}")
        if self.warm_power_mw.shape != (history_hours, n_sites):
            raise ValueError(
                f"warm_power_mw must have shape {(history_hours, n_sites)}"
            )
        for values in (
            self.service_arrivals,
            self.batch_arrivals,
            self.warm_power_mw,
        ):
            if np.any(~np.isfinite(values)) or np.any(values < 0.0):
                raise ValueError("workload and warm power must be finite/non-negative")
        if np.any(self.batch_deadline_hours < 1):
            raise ValueError("batch deadlines must be at least one hour")

    def scaled(self, multiplier: float) -> WorkloadTrace:
        if not math.isfinite(multiplier) or multiplier <= 0.0:
            raise ValueError("scale multiplier must be finite and positive")
        return WorkloadTrace(
            service_arrivals=self.service_arrivals * multiplier,
            batch_arrivals=self.batch_arrivals * multiplier,
            batch_deadline_hours=self.batch_deadline_hours.copy(),
            warm_power_mw=self.warm_power_mw * multiplier,
        )


@dataclass
class BatchEntry:
    amount: float
    origin: int
    deadline_step: int
    sequence: int


class EDFQueue:
    """Exact batch-work ledger with deterministic EDF origin drainage."""

    def __init__(self) -> None:
        self.entries: list[BatchEntry] = []
        self._sequence = 0
        self.arrived = 0.0
        self.completed = 0.0

    @property
    def total(self) -> float:
        return float(sum(entry.amount for entry in self.entries))

    def add(self, amount: float, origin: int, deadline_step: int) -> None:
        if amount <= 0.0:
            return
        self.entries.append(
            BatchEntry(float(amount), int(origin), int(deadline_step), self._sequence)
        )
        self._sequence += 1
        self.arrived += float(amount)

    def due_by(self, deadline_step: int) -> float:
        return float(
            sum(
                entry.amount
                for entry in self.entries
                if entry.deadline_step <= deadline_step
            )
        )

    def histogram(self, current_step: int, edges: tuple[int, ...]) -> np.ndarray:
        buckets = np.zeros(len(edges) + 1, dtype=np.float64)
        for entry in self.entries:
            remaining = max(entry.deadline_step - current_step, 0)
            index = int(np.searchsorted(edges, remaining, side="left"))
            buckets[index] += entry.amount
        return buckets

    def by_origin(self, n_origins: int) -> np.ndarray:
        totals = np.zeros(n_origins, dtype=np.float64)
        for entry in self.entries:
            totals[entry.origin] += entry.amount
        return totals

    @property
    def deadlines(self) -> tuple[int, ...]:
        return tuple(sorted({entry.deadline_step for entry in self.entries}))

    def drain(self, amount: float, n_origins: int) -> np.ndarray:
        target = min(max(float(amount), 0.0), self.total)
        drained = np.zeros(n_origins, dtype=np.float64)
        remaining: list[BatchEntry] = []
        ordered = sorted(
            self.entries,
            key=lambda value: (value.deadline_step, value.origin, value.sequence),
        )
        amount_left = target
        for entry in ordered:
            take = min(entry.amount, amount_left)
            if take > 0.0:
                drained[entry.origin] += take
                entry.amount -= take
                amount_left -= take
            if entry.amount > 1e-12:
                remaining.append(entry)
        self.entries = sorted(
            remaining,
            key=lambda value: (value.deadline_step, value.origin, value.sequence),
        )
        actual = float(drained.sum())
        self.completed += actual
        return drained

    def conservation_error(self) -> float:
        return self.arrived - self.completed - self.total

    def as_dict(self) -> dict[str, Any]:
        return {
            "arrived": self.arrived,
            "completed": self.completed,
            "queued": self.total,
            "conservation_error": self.conservation_error(),
        }
