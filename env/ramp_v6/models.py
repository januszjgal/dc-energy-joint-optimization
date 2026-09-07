"""Frozen data structures for the additive ramp environment (continuous-month protocol)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


HORIZONS = (1, 3)
FORECAST_HOURS = (1, 2, 3)
# The hour-t decision reads grid rows through hour t-1. Level features use
# four lags of that last completed hour, and the three-hour closed ramp needs
# the hour four steps back, so four warm hours precede the first decision.
HISTORY_HOURS = 4


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
    """Train-only normalizers for canonical market panels."""

    fit_start_utc: str
    fit_end_utc: str
    gross_q95_mw: dict[str, float]
    gross_level_mean_mw: dict[str, float]
    gross_level_std_mw: dict[str, float]
    net_level_mean_mw: dict[str, float]
    net_level_std_mw: dict[str, float]
    fit_months: tuple[str, ...]
    stats_id: str = "ramp-v6-train-only-q95-v1"

    def validate(self, markets: set[str]) -> None:
        mappings = (
            self.gross_q95_mw,
            self.gross_level_mean_mw,
            self.gross_level_std_mw,
            self.net_level_mean_mw,
            self.net_level_std_mw,
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


@dataclass(frozen=True)
class RampProtocol:
    """Frozen environment and scalar-reward semantics.

    One episode is one continuous panel. There is no terminal run-out: work
    that arrives in the last slots has its window cut at the final decision
    slot, so every episode ends with empty queues and every slot is scored.
    """

    protocol_id: str = "ramp-v7-continuous-month-v1"
    history_hours: int = HISTORY_HOURS
    terminal_tail_hours: int = 0
    deadline_bucket_hours: tuple[int, ...] = (1, 3)
    ramp_weights: dict[int, float] = field(
        default_factory=lambda: {1: 0.40, 3: 0.60}
    )
    ramp_reward_scale: float = 1.0
    tolerance: float = 1e-9

    def validate(self) -> None:
        if self.history_hours != HISTORY_HOURS:
            raise ValueError(
                f"continuous-month protocol requires exactly {HISTORY_HOURS}h of warm history"
            )
        if self.terminal_tail_hours != 0:
            raise ValueError("continuous-month protocol scores every slot; no terminal run-out")
        if set(self.ramp_weights) != set(HORIZONS):
            raise ValueError("ramp weights must contain exactly 1h and 3h")
        if not math.isclose(
            sum(self.ramp_weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("ramp weights must sum to one")
        if self.ramp_reward_scale <= 0.0 or self.tolerance <= 0.0:
            raise ValueError("reward scale and tolerance must be positive")
        if self.deadline_bucket_hours != (1, 3):
            raise ValueError("policy observations use only <=1h and <=3h urgency")


@dataclass(frozen=True)
class WorkloadTrace:
    """Hourly service/batch arrivals and warm modeled power.

    Arrays use absolute compute-work units. Scaling a study multiplies arrivals,
    warm power, site compute capacity, and site rated power together.
    ``batch_deadline_hours[t, i]`` is the inclusive execution-window length
    ``H_i``: an arrival in slot ``t`` may run in slots ``t .. t + H_i - 1``,
    so ``H_i = 1`` allows no delay and ``H_i = 3`` allows at most two hours.
    """

    service_arrivals: np.ndarray
    batch_arrivals: np.ndarray
    batch_deadline_hours: np.ndarray
    warm_power_mw: np.ndarray

    def validate(
        self, n_steps: int, n_sites: int, history_hours: int = HISTORY_HOURS
    ) -> None:
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
            raise ValueError("batch execution windows must be at least one slot")

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
    # First slot in which the work would be late. Work with ``deadline_step``
    # D must execute in some slot <= D - 1, so an arrival in slot t with
    # window H_i gets D = t + H_i, i.e. its last permitted slot is t + H_i - 1.
    deadline_step: int
    sequence: int


class EDFQueue:
    """Exact batch-work ledger with deterministic EDF origin drainage.

    Drainage order is earliest ``deadline_step`` first, then lowest origin
    index, then arrival order (``sequence``). The rule is a total order, so
    the same request always drains the same work.
    """

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

    @staticmethod
    def _drain_key(entry: BatchEntry) -> tuple[int, int, int]:
        return (entry.deadline_step, entry.origin, entry.sequence)

    def drain(self, amount: float, n_origins: int) -> np.ndarray:
        target = min(max(float(amount), 0.0), self.total)
        drained = np.zeros(n_origins, dtype=np.float64)
        remaining: list[BatchEntry] = []
        ordered = sorted(self.entries, key=self._drain_key)
        amount_left = target
        for entry in ordered:
            take = min(entry.amount, amount_left)
            if take > 0.0:
                drained[entry.origin] += take
                entry.amount -= take
                amount_left -= take
            if entry.amount > 1e-12:
                remaining.append(entry)
        self.entries = sorted(remaining, key=self._drain_key)
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
