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


def load_cell_batch_config(dist_json_path: Path) -> CellBatchConfig:
    """Load batch config from a per-cell ``batch_distributions_*.json`` file."""
    with open(dist_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return CellBatchConfig(
        cell_name=dist_json_path.stem,
        batch_fraction=data["workload_mix"]["batch_fraction"],
        mean_duration_sec=data["duration"]["mean"],
    )
