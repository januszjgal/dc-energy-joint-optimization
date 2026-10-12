"""The simulation's month clock: the clock of the Google workload trace.

The Google trace starts at midnight Pacific daylight time on 1 May 2019, which
is 07:00 UTC. Simulated months follow that clock with a fixed UTC-7 offset all
year, so every month starts at 07:00 UTC on its first day and trace hour 0
lands on the month's first hour, with no daylight-saving switch. Grid data
stays in UTC; only the month boundaries move.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The trace's clock is UTC-7 (Pacific daylight time).
TRACE_CLOCK_OFFSET = pd.Timedelta(hours=7)


def month_start(month_id: str) -> pd.Timestamp:
    """First hour, in UTC, of a simulated month such as ``2025-05``."""
    return pd.Timestamp(f"{month_id}-01T00:00:00Z") + TRACE_CLOCK_OFFSET


def month_hours(month_id: str) -> int:
    """Number of hourly decision slots in a simulated month."""
    start = pd.Timestamp(f"{month_id}-01T00:00:00Z")
    return int((start + pd.offsets.MonthBegin(1) - start) / pd.Timedelta(hours=1))


def month_labels(index: pd.DatetimeIndex) -> pd.Index:
    """Simulated month (``YYYY-MM``) of each UTC timestamp, on the trace clock."""
    return pd.Index((pd.DatetimeIndex(index) - TRACE_CLOCK_OFFSET).strftime("%Y-%m"))


def trace_hours(index: pd.DatetimeIndex) -> np.ndarray:
    """Hour of the month on the trace clock, i.e. the trace hour replayed then.

    Day D of a simulated month draws on trace day D at the same trace-clock
    hour, so the month's first hour (07:00 UTC on the 1st) is trace hour 0.
    """
    local = pd.DatetimeIndex(index) - TRACE_CLOCK_OFFSET
    return ((local.day - 1) * 24 + local.hour).to_numpy(dtype=np.int64)
