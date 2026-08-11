"""Calibrate a linear power model from Google PowerData2019.

Extracts hourly (CPU utilization, power utilization) pairs for cells a-h,
fits a linear model  P = P_idle + (P_peak - P_idle) * cpu_util,
and saves the parameters to data/power_model_params.json.

Usage:
    python preprocess/power_calibrator.py --project YOUR_GCP_PROJECT
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from google.cloud import bigquery

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def extract_cpu_vs_power(client: bigquery.Client) -> pd.DataFrame:
    """Extract hourly CPU utilization vs measured power utilization per cell.

    Joins cluster CPU data with power data at the cell level (not PDU level)
    for a simpler, more robust fit.  We aggregate all PDUs per cell into a
    single measured_power_util average per hour, and compute total CPU
    utilization per hour across the whole cell.
    """
    query = """
    WITH cell_capacities AS (
        SELECT cell, SUM(cpu_cap) AS cpu_capacity
        FROM (
            SELECT m.cell, me.machine_id, MAX(capacity.cpus) AS cpu_cap
            FROM `google.com:google-cluster-data`.powerdata_2019.machine_to_pdu_mapping AS m
            JOIN `google.com:google-cluster-data`.clusterdata_2019_a.machine_events AS me
                ON m.machine_id = me.machine_id
            WHERE m.cell = 'a'
            GROUP BY 1, 2
            UNION ALL
            SELECT m.cell, me.machine_id, MAX(capacity.cpus) AS cpu_cap
            FROM `google.com:google-cluster-data`.powerdata_2019.machine_to_pdu_mapping AS m
            JOIN `google.com:google-cluster-data`.clusterdata_2019_b.machine_events AS me
                ON m.machine_id = me.machine_id
            WHERE m.cell = 'b'
            GROUP BY 1, 2
            UNION ALL
            SELECT m.cell, me.machine_id, MAX(capacity.cpus) AS cpu_cap
            FROM `google.com:google-cluster-data`.powerdata_2019.machine_to_pdu_mapping AS m
            JOIN `google.com:google-cluster-data`.clusterdata_2019_c.machine_events AS me
                ON m.machine_id = me.machine_id
            WHERE m.cell = 'c'
            GROUP BY 1, 2
            UNION ALL
            SELECT m.cell, me.machine_id, MAX(capacity.cpus) AS cpu_cap
            FROM `google.com:google-cluster-data`.powerdata_2019.machine_to_pdu_mapping AS m
            JOIN `google.com:google-cluster-data`.clusterdata_2019_d.machine_events AS me
                ON m.machine_id = me.machine_id
            WHERE m.cell = 'd'
            GROUP BY 1, 2
        )
        GROUP BY 1
    ),
    power_hourly AS (
        SELECT
            cell,
            CAST(FLOOR(time / (1e6 * 60 * 60)) AS INT64) AS hour_index,
            AVG(measured_power_util) AS avg_power_util
        FROM `google.com:google-cluster-data`.`powerdata_2019.cell*`
        WHERE NOT bad_measurement_data
            AND cell IN ('a', 'b', 'c', 'd')
        GROUP BY 1, 2
    )
    SELECT
        p.cell,
        p.hour_index,
        p.avg_power_util
    FROM power_hourly AS p
    ORDER BY p.cell, p.hour_index
    """
    return client.query(query).to_dataframe()


def extract_cpu_utilization_hourly(client: bigquery.Client, cell: str) -> pd.DataFrame:
    """Extract hourly aggregate CPU utilization for a cell."""
    cap_query = f"""
    SELECT SUM(cpu_cap) AS cpu_capacity
    FROM (
        SELECT machine_id, MAX(capacity.cpus) AS cpu_cap
        FROM `google.com:google-cluster-data`.clusterdata_2019_{cell}.machine_events
        GROUP BY 1
    )
    """
    cap_df = client.query(cap_query).to_dataframe()
    cpu_capacity = float(cap_df["cpu_capacity"].iloc[0])

    usage_query = f"""
    SELECT
        CAST(FLOOR(start_time / (1e6 * 60 * 60)) AS INT64) AS hour_index,
        SUM(average_usage.cpus) / (12 * {cpu_capacity}) AS avg_cpu_util
    FROM `google.com:google-cluster-data`.clusterdata_2019_{cell}.instance_usage
    WHERE (alloc_collection_id IS NULL OR alloc_collection_id = 0)
        AND (end_time - start_time) >= (5 * 60 * 1e6)
    GROUP BY 1
    ORDER BY 1
    """
    df = client.query(usage_query).to_dataframe()
    df["cell"] = cell
    return df


def fit_linear_power_model(
    cpu_util: np.ndarray, power_util: np.ndarray
) -> dict:
    """Fit P = idle + slope * cpu_util via least squares.

    Returns dict with idle_power, peak_power, slope, r_squared.
    All values are in the normalized utilization space [0, 1].
    """
    # Simple linear regression: power = a + b * cpu
    A = np.vstack([np.ones_like(cpu_util), cpu_util]).T
    result = np.linalg.lstsq(A, power_util, rcond=None)
    intercept, slope = result[0]

    predicted = intercept + slope * cpu_util
    ss_res = np.sum((power_util - predicted) ** 2)
    ss_tot = np.sum((power_util - power_util.mean()) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "idle_power": float(intercept),
        "peak_power": float(intercept + slope),
        "slope": float(slope),
        "r_squared": float(r_squared),
        "description": "Linear power model: P = idle_power + slope * cpu_utilization. "
        "All values are normalized utilization fractions [0, 1].",
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate linear power model from Google PowerData2019"
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
        help="Cells to use for calibration (default: a b c d)",
    )
    args = parser.parse_args(argv)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    client = bigquery.Client(project=args.project)

    # Extract power data (hourly averages per cell)
    print("Extracting hourly power utilization...")
    power_df = extract_cpu_vs_power(client)
    print(f"  Got {len(power_df)} power rows")

    # Extract CPU utilization per cell and merge
    all_cpu = []
    for cell in args.cells:
        print(f"Extracting hourly CPU utilization for cell {cell}...")
        cpu_df = extract_cpu_utilization_hourly(client, cell)
        all_cpu.append(cpu_df)
        print(f"  Got {len(cpu_df)} CPU rows for cell {cell}")

    cpu_combined = pd.concat(all_cpu, ignore_index=True)

    # Merge on (cell, hour_index)
    merged = pd.merge(
        cpu_combined, power_df, on=["cell", "hour_index"], how="inner"
    )
    print(f"Merged dataset: {len(merged)} rows")

    # Fit model
    cpu_util = merged["avg_cpu_util"].values
    power_util = merged["avg_power_util"].values

    # Filter out any NaN/invalid rows
    valid = np.isfinite(cpu_util) & np.isfinite(power_util)
    cpu_util = cpu_util[valid]
    power_util = power_util[valid]

    params = fit_linear_power_model(cpu_util, power_util)
    print(f"\nFitted power model:")
    print(f"  P_idle  = {params['idle_power']:.4f}")
    print(f"  P_peak  = {params['peak_power']:.4f}")
    print(f"  slope   = {params['slope']:.4f}")
    print(f"  R^2     = {params['r_squared']:.4f}")

    # Save parameters
    out_path = DATA_DIR / "power_model_params.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Also save the raw scatter data for plotting
    scatter_path = DATA_DIR / "power_model_scatter.csv"
    scatter_df = pd.DataFrame(
        {"cpu_util": cpu_util, "power_util": power_util}
    )
    scatter_df.to_csv(scatter_path, index=False)
    print(f"Saved scatter data to {scatter_path}")


if __name__ == "__main__":
    main()
