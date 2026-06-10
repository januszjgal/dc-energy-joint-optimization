"""Data center site: holds timeseries data and mutable state for one DC.

Each DC is a pure grid-connected load — no on-site solar self-consumption.
Solar irradiance is retained only as a *forecast feature* for the agent,
since high midday solar in a solar-heavy region predicts a steep evening
ramp in regional grid net demand.
"""

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
        solar: Solar capacity factor timeseries [0, 1] (forecast feature only)
        price: Electricity price timeseries ($/kWh)
        net_demand: Regional grid net demand, normalized to [0, 1] by
            region historical peak. Drives the peak-contribution penalty.
        rated_power_mw: Total DC rated power capacity in MW
        capacity: Max normalized CPU utilization (from fleet data or 1.0)
        memory_capacity: Normalized memory capacity (from fleet data or 1.0)
        memory_cpu_ratio: Ratio of memory demand to CPU demand for workloads
        batch_fraction: Fraction of workload that is deferrable batch [0, 1]
            (Borg no-SLO tiers: free <=99 or beb 110-115; Tirmazi et al. 2020 §2)
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
    net_demand: np.ndarray  # shape: (T,), normalized to [0, 1]
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

    # Rolling history of recent batch arrivals (24h window at 5-min resolution
    # = 288 steps). Used by the burst-aware observation augmentation (§7-burst).
    _arrival_history: np.ndarray = field(default=None, init=False, repr=False)
    _arrival_pos: int = field(default=0, init=False, repr=False)
    _arrival_count: int = field(default=0, init=False, repr=False)
    _arrival_sum: float = field(default=0.0, init=False, repr=False)

    def reset(self, seed: int | None = None) -> None:
        """Reset mutable state for a new episode."""
        self.backlog = 0.0
        self.memory_backlog = 0.0
        self.current_load = 0.0
        self.current_memory_load = 0.0
        self.batch_pool.reset()
        if self.batch_generator is not None:
            self.batch_generator.reset(seed=seed)
        # Reset rolling-arrival buffer (24h = 288 5-min steps)
        self._arrival_history = np.zeros(288, dtype=np.float32)
        self._arrival_pos = 0
        self._arrival_count = 0
        self._arrival_sum = 0.0

    @property
    def num_timesteps(self) -> int:
        return min(
            len(self.workload),
            len(self.solar),
            len(self.price),
            len(self.net_demand),
        )

    def get_local_demand(self, t: int) -> float:
        return float(self.workload[t])

    # Service vs batch split. Deferrable "batch" = the Borg NO-SLO tiers: free
    # (priority <= 99) and best-effort batch / beb (110-115) — both run without
    # SLOs (Tirmazi et al. 2020, "Borg: the Next Generation", EuroSys '20, §2).
    # Service is the SLO-bearing remainder (mid 116-119, production 120-359,
    # monitoring >= 360); production in particular requires high availability —
    # Borg evicts lower tiers to protect it — so service must be served
    # immediately (unserved service accrues a backlog penalty; it is never
    # moved into the batch pool).
    def get_service_demand(self, t: int) -> float:
        return float(self.workload[t]) * (1.0 - self.batch_fraction)

    def get_batch_demand(self, t: int) -> float:
        if self.batch_generator is not None:
            return self.batch_generator.get_batch_cpu_demand(t)
        return float(self.workload[t]) * self.batch_fraction

    def get_batch_memory_demand(self, t: int) -> float:
        if self.batch_generator is not None:
            return self.batch_generator.get_batch_memory_demand(t)
        return self.get_batch_demand(t) * self.memory_cpu_ratio

    def get_memory_demand(self, t: int) -> float:
        return self.get_service_demand(t) * self.memory_cpu_ratio

    def get_solar_fraction(self, t: int) -> float:
        """Solar capacity factor at t. A forecast feature, not a supply."""
        return float(self.solar[t])

    def get_price(self, t: int) -> float:
        return float(self.price[t])

    def get_net_demand(self, t: int) -> float:
        """Regional grid net demand at t, normalized to [0, 1] by region peak."""
        return float(self.net_demand[t])

    def record_arrival(self, t: int) -> None:
        """Append the current timestep's batch arrival into the rolling buffer.

        Called once per env step (immediately after `get_batch_demand(t)` is
        injected into the pool). The buffer is a 288-step (24h) circular
        buffer used by `get_burst_severity` to express the agent's current
        arrival as a multiple of recent average load.
        """
        if self._arrival_history is None:
            return  # reset() not yet called; should never happen
        arrival = self.get_batch_demand(t)
        window = self._arrival_history.shape[0]
        if self._arrival_count >= window:
            self._arrival_sum -= float(self._arrival_history[self._arrival_pos])
        self._arrival_history[self._arrival_pos] = arrival
        self._arrival_sum += arrival
        self._arrival_pos = (self._arrival_pos + 1) % window
        self._arrival_count = min(self._arrival_count + 1, window)

    def get_burst_severity(self, t: int) -> float:
        """Return current arrival / rolling 24h mean arrival.

        Semantics: 1.0 = average arrival, 2.0 = twice the recent mean
        (a moderate burst), 5+ = extreme burst. Clipped to [0, 10] for
        numerical stability in observations.

        Falls back to 1.0 (neutral signal) when the rolling buffer has not
        accumulated enough history or recent mean is effectively zero.
        """
        if self._arrival_history is None or self._arrival_count == 0:
            return 1.0
        mean = self._arrival_sum / self._arrival_count
        if mean < 1e-9:
            return 1.0
        current = self.get_batch_demand(t)
        return float(min(10.0, current / mean))
