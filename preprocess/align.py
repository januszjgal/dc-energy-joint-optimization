"""Align all data traces to a common 5-minute timestep index.

Reads cell workload CSVs, solar CSVs, and price CSVs from data/,
trims them to the same length (shortest common window), and rewrites
them with a consistent 0-based timestep index.

Usage:
    python preprocess/align.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CELLS_DIR = DATA_DIR / "cells"
SOLAR_DIR = DATA_DIR / "solar"
PRICES_DIR = DATA_DIR / "prices"


def load_and_report(path: Path, value_col: str) -> pd.DataFrame:
    """Load a CSV and report its length."""
    df = pd.read_csv(path)
    print(f"  {path.name}: {len(df)} rows, "
          f"{value_col} range [{df[value_col].min():.4f}, {df[value_col].max():.4f}]")
    return df


def align_to_length(df: pd.DataFrame, target_len: int) -> pd.DataFrame:
    """Trim or report mismatch for a dataframe to target length."""
    if len(df) >= target_len:
        df = df.iloc[:target_len].copy()
    else:
        print(f"    Warning: only {len(df)} rows, padding with last value to {target_len}")
        pad_count = target_len - len(df)
        last_row = df.iloc[-1:].copy()
        padding = pd.concat([last_row] * pad_count, ignore_index=True)
        df = pd.concat([df, padding], ignore_index=True)

    df["timestep"] = range(target_len)
    return df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Align all data traces to common timestep index"
    )
    parser.add_argument(
        "--target-days",
        type=int,
        default=None,
        help="Target number of days (default: auto-detect from shortest trace)",
    )
    args = parser.parse_args(argv)

    print("=== Loading cell workloads ===")
    cell_files = sorted(CELLS_DIR.glob("cell_*.csv"))
    if not cell_files:
        print("No cell files found. Run google_extractor.py first.")
        return

    cells = {}
    for path in cell_files:
        cells[path.stem] = load_and_report(path, "cpu_demand_norm")

    print("\n=== Loading solar data ===")
    solar_files = sorted(SOLAR_DIR.glob("*.csv"))
    solars = {}
    for path in solar_files:
        solars[path.stem] = load_and_report(path, "solar_fraction")

    print("\n=== Loading price data ===")
    price_files = sorted(PRICES_DIR.glob("*.csv"))
    prices = {}
    for path in price_files:
        prices[path.stem] = load_and_report(path, "price_usd_kwh")

    # Determine target length
    all_lengths = (
        [len(df) for df in cells.values()]
        + [len(df) for df in solars.values()]
        + [len(df) for df in prices.values()]
    )

    if args.target_days:
        target_len = args.target_days * 288
    else:
        target_len = min(all_lengths)

    target_days = target_len / 288
    print(f"\n=== Aligning to {target_len} timesteps ({target_days:.1f} days) ===")

    # Align and save
    for name, df in cells.items():
        df = align_to_length(df, target_len)
        out = df[["timestep", "cpu_demand_norm"]]
        out.to_csv(CELLS_DIR / f"{name}.csv", index=False)
        print(f"  Aligned {name}")

    for name, df in solars.items():
        df = align_to_length(df, target_len)
        out = df[["timestep", "solar_fraction"]]
        out.to_csv(SOLAR_DIR / f"{name}.csv", index=False)
        print(f"  Aligned {name}")

    for name, df in prices.items():
        df = align_to_length(df, target_len)
        out = df[["timestep", "price_usd_kwh"]]
        out.to_csv(PRICES_DIR / f"{name}.csv", index=False)
        print(f"  Aligned {name}")

    # Write a metadata file
    meta = {
        "total_timesteps": target_len,
        "days": target_days,
        "interval_seconds": 300,
        "cells": list(cells.keys()),
        "solar_locations": list(solars.keys()),
        "price_markets": list(prices.keys()),
    }
    import json
    meta_path = DATA_DIR / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata saved to {meta_path}")

    print("Done.")


if __name__ == "__main__":
    main()
