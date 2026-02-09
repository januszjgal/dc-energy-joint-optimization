"""Extract per-cell CPU utilization from Google ClusterData2019 via BigQuery.

Produces one CSV per cell with columns: timestep, cpu_demand_norm
where timestep is a 5-minute interval index and cpu_demand_norm is the
aggregate CPU utilization as a fraction of cell capacity [0, 1].

Usage:
    python preprocess/google_extractor.py --project YOUR_GCP_PROJECT --cells a b c d
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from google.cloud import bigquery

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "cells"


def get_cell_capacity(client: bigquery.Client, cell: str) -> float:
    """Return total CPU capacity for a cell (sum of max per-machine cpus)."""
    query = f"""
    SELECT SUM(cpu_cap) AS cpu_capacity
    FROM (
        SELECT machine_id, MAX(capacity.cpus) AS cpu_cap
        FROM `google.com:google-cluster-data`.clusterdata_2019_{cell}.machine_events
        GROUP BY 1
    )
    """
    result = client.query(query).to_dataframe()
    return float(result["cpu_capacity"].iloc[0])


def extract_cell_utilization(
    client: bigquery.Client, cell: str, cpu_capacity: float
) -> pd.DataFrame:
    """Extract 5-minute aggregate CPU utilization for a cell.

    Timestamps in the trace are microseconds since 600 seconds before
    May 1 2019 00:00 PT.  We bucket into 5-minute windows (300 seconds)
    and normalize by cell capacity.

    Each 5-min window may contain multiple instance_usage rows (one per
    instance per 5-min measurement period).  We sum cpu_usage across all
    instances and divide by capacity to get utilization in [0, 1].

    The divisor accounts for the fact that within each 5-min bucket there
    is exactly one measurement row per instance, so the sum of cpu_usage
    already represents that bucket's demand (no need for a "12" factor
    like the hourly queries in the Colab which aggregate 12 five-min
    windows into one hour).
    """
    query = f"""
    SELECT
        CAST(FLOOR(start_time / (1e6 * 300)) AS INT64) AS time_bucket,
        SUM(average_usage.cpus) / {cpu_capacity} AS cpu_demand_norm
    FROM `google.com:google-cluster-data`.clusterdata_2019_{cell}.instance_usage
    WHERE (alloc_collection_id IS NULL OR alloc_collection_id = 0)
        AND (end_time - start_time) >= (5 * 60 * 1e6)
    GROUP BY 1
    ORDER BY 1
    """
    df = client.query(query).to_dataframe()

    # Convert time_bucket to a zero-based timestep index
    df["timestep"] = df["time_bucket"] - df["time_bucket"].min()
    df = df[["timestep", "cpu_demand_norm"]].copy()

    # Clip to [0, 1] (over-committed instances can exceed capacity)
    df["cpu_demand_norm"] = df["cpu_demand_norm"].clip(0.0, 1.0)

    return df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Extract per-cell CPU utilization from Google ClusterData2019"
    )
    parser.add_argument(
        "--project",
        required=True,
        help="GCP project ID with BigQuery billing enabled",
    )
    parser.add_argument(
        "--cells",
        nargs="+",
        default=["a", "b", "c", "d"],
        help="Cell letters to extract (default: a b c d)",
    )
    args = parser.parse_args(argv)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    client = bigquery.Client(project=args.project)

    for cell in args.cells:
        print(f"[cell {cell}] Querying cell capacity...")
        capacity = get_cell_capacity(client, cell)
        print(f"[cell {cell}] CPU capacity = {capacity:.2f}")

        print(f"[cell {cell}] Extracting 5-min CPU utilization...")
        df = extract_cell_utilization(client, cell, capacity)

        out_path = DATA_DIR / f"cell_{cell}.csv"
        df.to_csv(out_path, index=False)
        print(
            f"[cell {cell}] Saved {len(df)} rows to {out_path} "
            f"(mean={df['cpu_demand_norm'].mean():.4f}, "
            f"max={df['cpu_demand_norm'].max():.4f})"
        )

    print("Done.")


if __name__ == "__main__":
    main()
