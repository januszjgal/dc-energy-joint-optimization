"""Workload generator: splits aggregate demand into service/batch components
and manages deferrable batch pools with deadline tracking.

Can be used standalone for analysis or integrated into MultiDCEnv.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np


class BatchEntry(NamedTuple):
    """A chunk of deferrable batch work in the pool."""

    cpu_demand: float  # normalized CPU demand for this chunk
    deadline_step: int  # absolute timestep by which this must be served


@dataclass
class BatchPool:
    """Efficient batch pool for one DC using a deque of (demand, deadline) tuples.

    Entries are appended at the right (newest).  Deadline violations are
    scanned from the left (oldest, typically earliest deadlines).
    """

    entries: deque[BatchEntry] = field(default_factory=deque)

    @property
    def total_demand(self) -> float:
        """Total CPU demand in the pool."""
        return sum(e.cpu_demand for e in self.entries)

    def urgency(self, current_step: int, horizon_steps: int) -> float:
        """Fraction of pool demand within *horizon_steps* of its deadline."""
        total = 0.0
        urgent = 0.0
        for e in self.entries:
            total += e.cpu_demand
            if e.deadline_step - current_step <= horizon_steps:
                urgent += e.cpu_demand
        return urgent / total if total > 0 else 0.0

    def add(self, cpu_demand: float, deadline_step: int) -> None:
        """Add a batch work chunk to the pool."""
        if cpu_demand > 0:
            self.entries.append(BatchEntry(cpu_demand, deadline_step))

    def drain(self, fraction: float) -> float:
        """Remove *fraction* of total pool demand, preferring urgent entries.

        Returns the actual CPU demand drained.  Handles partial consumption
        of individual entries so no demand is lost.
        """
        if not self.entries or fraction <= 0:
            return 0.0

        target = self.total_demand * min(fraction, 1.0)
        drained = 0.0

        # Sort indices by deadline (most urgent first)
        sorted_indices = sorted(
            range(len(self.entries)),
            key=lambda i: self.entries[i].deadline_step,
        )

        to_remove: set[int] = set()
        for idx in sorted_indices:
            if drained >= target - 1e-12:
                break
            entry = self.entries[idx]
            remaining_need = target - drained
            if entry.cpu_demand <= remaining_need + 1e-12:
                drained += entry.cpu_demand
                to_remove.add(idx)
            else:
                # Partially drain this entry
                drained += remaining_need
                self.entries[idx] = BatchEntry(
                    entry.cpu_demand - remaining_need,
                    entry.deadline_step,
                )
                break

        # Remove fully drained entries (reverse order to preserve indices)
        for idx in sorted(to_remove, reverse=True):
            del self.entries[idx]

        return drained

    def expire(self, current_step: int) -> float:
        """Remove and return demand of entries past their deadline."""
        expired = 0.0
        remaining: deque[BatchEntry] = deque()
        for e in self.entries:
            if e.deadline_step <= current_step:
                expired += e.cpu_demand
            else:
                remaining.append(e)
        self.entries = remaining
        return expired

    def reset(self) -> None:
        """Clear the pool for a new episode."""
        self.entries.clear()


@dataclass
class CellBatchConfig:
    """Per-cell batch configuration loaded from distribution JSON files."""

    cell_name: str
    batch_fraction: float  # fraction of total demand that is batch
    mean_duration_sec: float  # mean batch job duration in seconds
    memory_cpu_ratio: float = 0.7  # mean(memory_request) / mean(cpu_request)
    dist_config: dict | None = None  # full distribution config for generator


def load_cell_batch_config(dist_json_path: Path) -> CellBatchConfig:
    """Load batch config from a per-cell ``batch_distributions_*.json`` file."""
    with open(dist_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Compute memory-to-CPU ratio from distribution means
    cpu_mean = data.get("cpu_request", {}).get("mean", 0.01)
    mem_mean = data.get("memory_request", {}).get("mean", 0.007)
    mem_cpu_ratio = mem_mean / cpu_mean if cpu_mean > 0 else 0.7

    return CellBatchConfig(
        cell_name=dist_json_path.stem,
        batch_fraction=data["workload_mix"]["batch_fraction"],
        mean_duration_sec=data.get("duration", {}).get("mean", 2100.0),
        memory_cpu_ratio=mem_cpu_ratio,
        dist_config=data,
    )


# ------------------------------------------------------------------
# Distribution sampling helpers
# ------------------------------------------------------------------


def _sample_from_dist(
    dist_config: dict, rng: np.random.Generator, size: int = 1
) -> np.ndarray:
    """Sample from a distribution specification (as stored in JSON).

    Supports: lognorm, weibull_min, gamma, expon (scipy parameterizations).
    Falls back to exponential with the distribution's mean on unknown types.
    """
    from scipy import stats as sp_stats

    dist_name = dist_config.get("distribution", "expon")
    params = dist_config.get("params", [])

    if dist_name == "mixture":
        return _sample_mixture(dist_config, rng, size)

    dist_map = {
        "lognorm": sp_stats.lognorm,
        "weibull_min": sp_stats.weibull_min,
        "gamma": sp_stats.gamma,
        "expon": sp_stats.expon,
        # discrete (count) distributions — used for tasks_per_job
        "nbinom": sp_stats.nbinom,
        "poisson": sp_stats.poisson,
        "geom": sp_stats.geom,
    }

    dist_cls = dist_map.get(dist_name)
    if dist_cls is not None and params:
        return dist_cls.rvs(*params, size=size, random_state=rng)

    # Fallback: exponential with the mean
    mean = dist_config.get("mean", 1.0)
    return rng.exponential(max(mean, 1e-6), size=size)


def _sample_mixture(
    dist_config: dict, rng: np.random.Generator, size: int = 1
) -> np.ndarray:
    """Sample from a point-mass + continuous tail mixture."""
    pm = dist_config.get("point_mass", {})
    pm_value = pm.get("value", 1)
    pm_weight = pm.get("weight", 0.5)
    tail = dist_config.get("tail", {})

    result = np.full(size, float(pm_value))
    # Determine which samples come from the tail
    tail_mask = rng.random(size) > pm_weight
    n_tail = int(tail_mask.sum())
    if n_tail > 0 and tail:
        tail_samples = _sample_from_dist(tail, rng, n_tail)
        result[tail_mask] = np.maximum(tail_samples, pm_value + 1)
    return result


# ------------------------------------------------------------------
# BatchArrivalGenerator
# ------------------------------------------------------------------


@dataclass
class BatchArrivalGenerator:
    """Generates synthetic batch arrival timeseries from fitted distributions.

    At reset(), pre-generates a full episode of batch arrivals by:
    1. Sampling inter-arrival times to determine arrival timesteps
    2. For each arrival, sampling cpu_demand and memory_demand
    3. Aggregating into per-timestep batch CPU and memory demand arrays
    4. Normalizing so total demand matches ``target_total_demand``

    The normalization preserves the temporal *shape* (burstiness) of
    arrivals while matching the aggregate volume expected from the
    static ``workload[t] * batch_fraction`` split.  This follows the
    Grange et al. methodology of using distribution-generated arrivals
    for realistic temporal variation.
    """

    dist_config: dict
    interval_seconds: int = 300
    num_timesteps: int = 8917
    target_total_demand: float = 0.0  # expected total batch CPU demand over episode
    seed: int = 42

    # Generated timeseries (populated on reset)
    _batch_cpu: np.ndarray = field(init=False, repr=False, default=None)
    _batch_mem: np.ndarray = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self.reset()

    def reset(self, seed: int | None = None) -> None:
        """Re-generate the batch arrival timeseries."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._batch_cpu = np.zeros(self.num_timesteps, dtype=np.float64)
        self._batch_mem = np.zeros(self.num_timesteps, dtype=np.float64)

        ia_config = self.dist_config.get("inter_arrival", {})
        cpu_config = self.dist_config.get("cpu_request", {})
        mem_config = self.dist_config.get("memory_request", {})
        tasks_config = self.dist_config.get("tasks_per_job", {})

        ia_mean = ia_config.get("mean", 30.0)
        total_time_sec = self.num_timesteps * self.interval_seconds

        # Estimate number of arrivals and generate in bulk
        est_arrivals = int(total_time_sec / max(ia_mean, 0.1) * 1.2)
        est_arrivals = max(est_arrivals, 100)

        # Sample inter-arrival times
        inter_arrivals = _sample_from_dist(ia_config, self._rng, est_arrivals)
        inter_arrivals = np.abs(inter_arrivals)  # ensure positive
        arrival_times = np.cumsum(inter_arrivals)

        # Truncate to episode duration
        mask = arrival_times < total_time_sec
        arrival_times = arrival_times[mask]
        n_jobs = len(arrival_times)

        if n_jobs == 0:
            return

        # Convert to timestep indices
        timesteps = (arrival_times / self.interval_seconds).astype(int)
        timesteps = np.clip(timesteps, 0, self.num_timesteps - 1)

        # Sample per-job characteristics
        cpu_per_task = _sample_from_dist(cpu_config, self._rng, n_jobs)
        cpu_per_task = np.abs(cpu_per_task)
        mem_per_task = _sample_from_dist(mem_config, self._rng, n_jobs)
        mem_per_task = np.abs(mem_per_task)

        if tasks_config:
            n_tasks = _sample_from_dist(tasks_config, self._rng, n_jobs)
            n_tasks = np.maximum(np.nan_to_num(n_tasks, nan=1.0), 1.0)
        else:
            n_tasks = np.ones(n_jobs)

        # Cap extreme outliers at 99.9th percentile to prevent single-job
        # domination of entire timestep demand
        for arr in (cpu_per_task, n_tasks):
            cap = np.percentile(arr, 99.9)
            np.clip(arr, None, cap, out=arr)

        raw_cpu = cpu_per_task * n_tasks
        raw_mem = mem_per_task * n_tasks

        # Aggregate into per-timestep arrays
        np.add.at(self._batch_cpu, timesteps, raw_cpu)
        np.add.at(self._batch_mem, timesteps, raw_mem)

        # Normalize to match expected aggregate demand level
        # This preserves temporal burstiness while ensuring demand scale
        # is consistent with the workload trace
        raw_total = self._batch_cpu.sum()
        if raw_total > 0 and self.target_total_demand > 0:
            scale = self.target_total_demand / raw_total
            self._batch_cpu *= scale
            self._batch_mem *= scale

    def get_batch_cpu_demand(self, t: int) -> float:
        """Return batch CPU demand at timestep t."""
        if self._batch_cpu is None or t >= len(self._batch_cpu):
            return 0.0
        return float(self._batch_cpu[t])

    def get_batch_memory_demand(self, t: int) -> float:
        """Return batch memory demand at timestep t."""
        if self._batch_mem is None or t >= len(self._batch_mem):
            return 0.0
        return float(self._batch_mem[t])
