"""Generate synthetic workload data for cells a-d for local testing.

This stands in for google_extractor.py when BigQuery is not available.
Produces realistic diurnal CPU utilization patterns with per-cell variation.
Replace with real data from BigQuery when running on Colab.

Usage:
    python preprocess/generate_synthetic_cells.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "cells"

# 31 days of May, 288 five-minute steps per day
TOTAL_STEPS = 31 * 288

# Per-cell parameters: (base_util, amplitude, phase_shift_hours, noise_std)
# Designed to mimic real variation across Google's 8 Borg cells
CELL_PARAMS = {
    "a": {"base": 0.45, "amplitude": 0.15, "phase_h": 0.0, "noise": 0.04},
    "b": {"base": 0.55, "amplitude": 0.12, "phase_h": 2.0, "noise": 0.05},
    "c": {"base": 0.50, "amplitude": 0.18, "phase_h": -1.0, "noise": 0.03},
    "d": {"base": 0.40, "amplitude": 0.10, "phase_h": 3.0, "noise": 0.06},
}


def generate_cell(params: dict, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(TOTAL_STEPS)

    # Diurnal pattern (peaks during business hours)
    hours = t / 12.0  # convert 5-min steps to hours
    phase = params["phase_h"]
    diurnal = params["amplitude"] * np.sin(
        2 * np.pi * (hours - 10.0 - phase) / 24.0
    )

    # Weekly pattern (slight dip on weekends)
    # May 1 2019 was Wednesday, so day 0 = Wed
    day_index = t // 288
    day_of_week = (day_index + 2) % 7  # 0=Mon, 5=Sat, 6=Sun
    weekend = np.isin(day_of_week, [5, 6]).astype(float)
    weekly = -0.08 * weekend

    # Base + patterns + noise
    cpu = params["base"] + diurnal + weekly + rng.normal(0, params["noise"], TOTAL_STEPS)
    cpu = np.clip(cpu, 0.05, 0.95)

    return pd.DataFrame({"timestep": t, "cpu_demand_norm": cpu})


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for i, (cell, params) in enumerate(CELL_PARAMS.items()):
        df = generate_cell(params, seed=42 + i)
        path = DATA_DIR / f"cell_{cell}.csv"
        df.to_csv(path, index=False)
        print(
            f"cell_{cell}.csv: {len(df)} rows, "
            f"mean={df['cpu_demand_norm'].mean():.4f}, "
            f"std={df['cpu_demand_norm'].std():.4f}"
        )

    print("Done. Replace with real BigQuery data when available.")


if __name__ == "__main__":
    main()
