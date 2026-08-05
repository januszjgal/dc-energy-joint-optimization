"""Derive REAL per-tier demand curves (service vs deferrable batch) from local data.

Replaces the synthetic BatchArrivalGenerator pulses as the primary batch-demand
input, following the data organization of Radovanovic et al. (2023), "Carbon-Aware
Computing for Datacenters" (CICS), which consumes real aggregate flexible vs
inflexible demand curves per cluster rather than stylized job-level models.

Method (no BigQuery needed — uses the local full jobs_{cell}.csv extracts):
  1. For every job, spread its total_cpu_request (a rate, NCUs) uniformly over its
     [submit, end) window (unfinished jobs charged to trace end). Do this separately
     for the deferrable no-SLO tiers (priority <= 115: free + beb per trace docs v3,
     which corrects Tirmazi et al. 2020
     §2) and for ALL jobs. Interval sums are computed with a diff-array + cumsum.
  2. Align the job-derived total curve to the measured cell curve (cells/cell_X.csv)
     by Pearson cross-correlation over candidate bucket offsets.
  3. Per-bucket deferrable share = batch_spread / all_spread (clamped to [0, 0.95]).
  4. Real curves: batch[t] = cpu_demand_norm[t] * share[t];
                  service[t] = cpu_demand_norm[t] * (1 - share[t]).
     Magnitude is the *measured* aggregate; only the tier split uses request windows.

Known approximation: the [submit, end) window includes queueing delay (we lack
schedule times locally), which slightly smooths the batch share. The ground-truth
version splits instance_usage by tier in BigQuery — see extract_tier_curves.ipynb.

Output: data/cells/cell_{x}_tiers.csv with columns
  timestep, cpu_demand_norm, batch_share, service_demand_norm, batch_demand_norm

Sanity property: service + batch == measured aggregate at every timestep, so a
serve-everything-now policy reproduces the original aggregate demand exactly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
CELLS = ["a", "b", "c", "d"]
BUCKET_US = 300e6  # 5 minutes in microseconds
MAX_SHARE = 0.95   # clamp: never attribute >95% of a bucket to deferrable work


def spread_jobs(submit_us: np.ndarray, end_us: np.ndarray, rate: np.ndarray,
                n_buckets: int) -> np.ndarray:
    """Sum of job rates active in each bucket via diff-array (O(n + buckets))."""
    b0 = np.clip((submit_us // BUCKET_US).astype(np.int64), 0, n_buckets - 1)
    b1 = np.clip(np.ceil(end_us / BUCKET_US).astype(np.int64), b0 + 1, n_buckets)
    diff = np.zeros(n_buckets + 1, dtype=np.float64)
    np.add.at(diff, b0, rate)
    np.add.at(diff, b1, -rate)
    return np.cumsum(diff)[:n_buckets]


def best_offset(job_total: np.ndarray, measured: np.ndarray, max_off: int = 48) -> tuple[int, float]:
    """Bucket offset aligning the job-derived curve to the measured cell curve."""
    T = len(measured)
    best, best_r = 0, -2.0
    for off in range(0, max_off + 1):
        seg = job_total[off:off + T]
        if len(seg) < T:
            break
        r = float(np.corrcoef(seg, measured)[0, 1])
        if r > best_r:
            best, best_r = off, r
    return best, best_r


def process_cell(cell: str) -> None:
    measured = pd.read_csv(DATA / "cells" / f"cell_{cell}.csv")
    wl = measured["cpu_demand_norm"].values.astype(np.float64)
    T = len(wl)

    df = pd.read_csv(
        DATA / "jobs" / f"jobs_{cell}.csv",
        usecols=["submit_time", "end_time", "priority", "total_cpu_request"],
    )
    df = df[(df["submit_time"] > 0) & df["total_cpu_request"].notna()]
    trace_end = float(df["submit_time"].max())
    end = df["end_time"].fillna(trace_end).clip(lower=df["submit_time"]).values
    sub = df["submit_time"].values.astype(np.float64)
    rate = df["total_cpu_request"].values.astype(np.float64)
    pr = df["priority"].values

    n_buckets = int(np.ceil(trace_end / BUCKET_US)) + 2
    deferrable = pr <= 115  # free (<=99) + beb (100-115); trace docs v3

    all_spread = spread_jobs(sub, end, rate, n_buckets)
    bat_spread = spread_jobs(sub[deferrable], end[deferrable], rate[deferrable], n_buckets)

    off, r = best_offset(all_spread, wl)
    a = all_spread[off:off + T]
    b = bat_spread[off:off + T]

    share = np.where(a > 1e-9, b / np.maximum(a, 1e-9), 0.0)
    share = np.clip(share, 0.0, MAX_SHARE)

    batch = wl * share
    service = wl - batch  # exact decomposition of the measured aggregate

    out = pd.DataFrame({
        "timestep": measured["timestep"].values,
        "cpu_demand_norm": wl,
        "batch_share": share,
        "service_demand_norm": service,
        "batch_demand_norm": batch,
    })
    path = DATA / "cells" / f"cell_{cell}_tiers.csv"
    out.to_csv(path, index=False)

    bf = float(batch.sum() / wl.sum())
    pm = batch.max() / batch.mean() if batch.mean() > 0 else 0.0
    print(f"cell {cell}: offset={off} (r={r:.3f})  batch_fraction={bf:.3f}  "
          f"batch peak/mean={pm:.2f}  share p50={np.median(share):.3f}  -> {path.name}")


def main() -> None:
    for c in CELLS:
        process_cell(c)
    print("\nDone. Real per-tier curves written (service + batch = measured aggregate).")


if __name__ == "__main__":
    main()
