"""Data center site: holds timeseries data and mutable state for one DC."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


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

    Mutable state (reset each episode):
        backlog: Accumulated unserved work
        current_load: Current CPU utilization assigned to this DC
    """

    name: str
    workload: np.ndarray  # shape: (T,)
    solar: np.ndarray  # shape: (T,)
    price: np.ndarray  # shape: (T,)
    solar_capacity_mw: float = 50.0
    rated_power_mw: float = 100.0
    capacity: float = 1.0

    # Mutable state
    backlog: float = field(default=0.0, init=False, repr=False)
    current_load: float = field(default=0.0, init=False, repr=False)

    def reset(self) -> None:
        """Reset mutable state for a new episode."""
        self.backlog = 0.0
        self.current_load = 0.0

    @property
    def num_timesteps(self) -> int:
        return len(self.workload)

    def get_local_demand(self, t: int) -> float:
        """Return the natural workload demand at timestep t."""
        return float(self.workload[t])

    def get_solar_fraction(self, t: int) -> float:
        """Return solar availability fraction at timestep t."""
        return float(self.solar[t])

    def get_price(self, t: int) -> float:
        """Return electricity price at timestep t."""
        return float(self.price[t])
