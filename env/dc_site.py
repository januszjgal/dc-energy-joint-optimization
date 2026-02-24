"""Data center site: holds timeseries data and mutable state for one DC."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from env.workload_generator import BatchPool


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
        capacity: Max normalized CPU utilization this DC can sustain (typically 1.0)
        batch_fraction: Fraction of workload that is deferrable batch [0, 1]
        batch_mean_duration_sec: Mean batch job duration (for deadline computation)

    Mutable state (reset each episode):
        backlog: Accumulated unserved service work
        current_load: Current CPU utilization assigned to this DC
        batch_pool: Pool of deferrable batch work with deadlines
    """

    name: str
    workload: np.ndarray  # shape: (T,)
    solar: np.ndarray  # shape: (T,)
    price: np.ndarray  # shape: (T,)
    solar_capacity_mw: float = 50.0
    rated_power_mw: float = 100.0
    capacity: float = 1.0
    batch_fraction: float = 0.0
    batch_mean_duration_sec: float = 2100.0

    # Mutable state
    backlog: float = field(default=0.0, init=False, repr=False)
    current_load: float = field(default=0.0, init=False, repr=False)
    batch_pool: BatchPool = field(default_factory=BatchPool, init=False, repr=False)

    def reset(self) -> None:
        """Reset mutable state for a new episode."""
        self.backlog = 0.0
        self.current_load = 0.0
        self.batch_pool.reset()

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
        """Return the deferrable batch demand arriving at timestep t."""
        return float(self.workload[t]) * self.batch_fraction

    def get_solar_fraction(self, t: int) -> float:
        """Return solar availability fraction at timestep t."""
        return float(self.solar[t])

    def get_price(self, t: int) -> float:
        """Return electricity price at timestep t."""
        return float(self.price[t])
