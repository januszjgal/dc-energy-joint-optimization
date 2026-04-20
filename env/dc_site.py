"""Data center site: holds timeseries data and mutable state for one DC."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from env.workload_generator import BatchArrivalGenerator, BatchPool


@dataclass
class DataCenterSite:
    """One data center in the multi-DC simulation.

    Attributes:
        name: Human-readable identifier (e.g., "US-West")
        workload: Normalized CPU demand timeseries [0, 1]
        solar: Solar capacity factor timeseries [0, 1]
        price: Electricity price timeseries ($/kWh)
        solar_capacity_mw: Installed solar capacity in MW
        rated_power_mw: Total DC rated power capacity in MW
        capacity: Max normalized CPU utilization (from fleet data or 1.0)
        memory_capacity: Normalized memory capacity (from fleet data or 1.0)
        memory_cpu_ratio: Ratio of memory demand to CPU demand for workloads
        batch_fraction: Fraction of workload that is deferrable batch [0, 1]
        batch_mean_duration_sec: Mean batch job duration (for deadline computation)

    Mutable state (reset each episode):
        backlog: Accumulated unserved service work (CPU)
        memory_backlog: Accumulated unserved memory demand
        current_load: Current CPU utilization assigned to this DC
        current_memory_load: Current memory utilization
        batch_pool: Pool of deferrable batch work with deadlines
    """

    name: str
    workload: np.ndarray  # shape: (T,)
    solar: np.ndarray  # shape: (T,)
    price: np.ndarray  # shape: (T,)
    duck_score: np.ndarray = field(default_factory=lambda: np.zeros(0))  # shape: (T,)
    solar_capacity_mw: float = 50.0
    rated_power_mw: float = 100.0
    capacity: float = 1.0
    memory_capacity: float = 1.0
    memory_cpu_ratio: float = 0.7
    batch_fraction: float = 0.0
    batch_mean_duration_sec: float = 2100.0

    # Fleet info (raw totals from machine data, informational)
    fleet_cpu_total: float = field(default=0.0, repr=False)
    fleet_memory_total: float = field(default=0.0, repr=False)
    fleet_machine_count: int = field(default=0, repr=False)

    # Mutable state
    backlog: float = field(default=0.0, init=False, repr=False)
    memory_backlog: float = field(default=0.0, init=False, repr=False)
    current_load: float = field(default=0.0, init=False, repr=False)
    current_memory_load: float = field(default=0.0, init=False, repr=False)
    batch_pool: BatchPool = field(default_factory=BatchPool, init=False, repr=False)
    batch_generator: BatchArrivalGenerator | None = field(
        default=None, init=False, repr=False
    )

    def reset(self, seed: int | None = None) -> None:
        """Reset mutable state for a new episode."""
        self.backlog = 0.0
        self.memory_backlog = 0.0
        self.current_load = 0.0
        self.current_memory_load = 0.0
        self.batch_pool.reset()
        if self.batch_generator is not None:
            self.batch_generator.reset(seed=seed)

    @property
    def num_timesteps(self) -> int:
        return min(len(self.workload), len(self.solar), len(self.price))

    def get_local_demand(self, t: int) -> float:
        """Return the natural workload demand at timestep t."""
        return float(self.workload[t])

    def get_service_demand(self, t: int) -> float:
        """Return the immediate (non-deferrable) demand at timestep t."""
        return float(self.workload[t]) * (1.0 - self.batch_fraction)

    def get_batch_demand(self, t: int) -> float:
        """Return the deferrable batch CPU demand arriving at timestep t."""
        if self.batch_generator is not None:
            return self.batch_generator.get_batch_cpu_demand(t)
        return float(self.workload[t]) * self.batch_fraction

    def get_batch_memory_demand(self, t: int) -> float:
        """Return the deferrable batch memory demand at timestep t."""
        if self.batch_generator is not None:
            return self.batch_generator.get_batch_memory_demand(t)
        return self.get_batch_demand(t) * self.memory_cpu_ratio

    def get_memory_demand(self, t: int) -> float:
        """Return the service memory demand at timestep t."""
        return self.get_service_demand(t) * self.memory_cpu_ratio

    def get_solar_fraction(self, t: int) -> float:
        """Return solar availability fraction at timestep t."""
        return float(self.solar[t])

    def get_price(self, t: int) -> float:
        """Return electricity price at timestep t."""
        return float(self.price[t])

    def get_duck_score(self, t: int) -> float:
        """Return duck curve stress score at timestep t.

        Positive = grid more stressed than recent average (expensive to draw grid power).
        Negative = grid less stressed than average (cheap/off-peak).
        Range: approximately [-3, 3] (clipped z-score).
        """
        if len(self.duck_score) == 0 or t >= len(self.duck_score):
            return 0.0
        return float(self.duck_score[t])
