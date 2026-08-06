"""Plot workload patterns from Google cells side by side.

Usage:
    python analysis/plot_workloads.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cell_files = sorted((DATA_DIR / "cells").glob("cell_*.csv"))
    if not cell_files:
        print("No cell files found. Run google_extractor.py first.")
        return

    fig, axes = plt.subplots(len(cell_files), 1, figsize=(14, 3 * len(cell_files)), sharex=True)
    if len(cell_files) == 1:
        axes = [axes]

    for ax, path in zip(axes, cell_files):
        df = pd.read_csv(path)
        name = path.stem.replace("cell_", "Cell ").upper()

        # Smooth for readability (1-hour rolling average)
        smoothed = df["cpu_demand_norm"].rolling(12, min_periods=1).mean()

        ax.plot(df["timestep"], df["cpu_demand_norm"], alpha=0.2, color="steelblue", linewidth=0.5)
        ax.plot(df["timestep"], smoothed, color="steelblue", linewidth=1.2, label=name)
        ax.set_ylabel("CPU Util")
        ax.set_ylim(0, 1)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Timestep (5-min intervals)")
    fig.suptitle("Google ClusterData2019: CPU Utilization per Cell", fontsize=14)
    fig.tight_layout()

    out_path = OUTPUT_DIR / "workload_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
