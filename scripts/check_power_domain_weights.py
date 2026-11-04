#!/usr/bin/env python3
"""Check the committed power fits against each cell's power domains.

Each domain's power is a fraction of that domain's own capacity, and the
capacities are not published. ``extract_cells.ipynb`` fits each cell's power
line on a plain average over its domains, which is exact only if the domains
are the same size. This refits every cell with the domains weighted by two
proxies for size, their machines' total CPU capacity and their machine count.

It also compares fits on all hours with fits on complete hours only. The trace's
first and last hours hold fewer than twelve windows, so dividing their CPU sum by
twelve understates their CPU utilization against a power value that is an average.

Input is the CSV written by the ``power-domains`` cell of the notebook. Hourly
CPU utilization comes from the committed five-minute curves. Writes nothing unless
``--write`` is given.

    python scripts/check_power_domain_weights.py data/power_domain_hours.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CELLS = "abcdefgh"
CELL_C_CAPACITY = 7619.8


def hourly_cpu(cell: str, first_bucket: int = 1) -> pd.Series:
    """Hourly CPU utilization on the trace's absolute hour index, summed over twelve.

    The first five-minute bucket in the usage curves is number 1. This alignment
    is the one that reproduces the committed fits.
    """
    curve = pd.read_csv(ROOT / "data" / "cells" / f"cell_{cell}_tiers.csv")
    hour = (curve["timestep"] + first_bucket) // 12
    return curve.groupby(hour)["cpu_demand_norm"].sum() / 12.0


def fit(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Least-squares line with the slope's ordinary standard error."""
    design = np.column_stack([np.ones(len(x)), x])
    (alpha, beta), *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ np.array([alpha, beta])
    se = float(np.sqrt(residual.var(ddof=2) / np.sum((x - x.mean()) ** 2)))
    r2 = 1.0 - float(residual @ residual) / float(np.sum((y - y.mean()) ** 2))
    return {"alpha": float(alpha), "beta": float(beta), "se_beta": se, "r2": r2, "n": len(x)}


def power_series(hours: pd.DataFrame, cell: str) -> dict[str, pd.Series]:
    """Hourly power fraction of a cell under each way of combining its domains."""
    h = hours[hours["cell"] == cell]
    pooled = h.groupby("hour_index").agg(sum_power=("sum_power_util", "sum"), rows=("good_rows", "sum"))
    series = {"plain": pooled["sum_power"] / pooled["rows"]}
    for name, column in (("CPU capacity", "cpu_capacity"), ("machines", "machines")):
        series[name] = h.groupby("hour_index").apply(
            lambda g: np.average(g["power"], weights=g[column]), include_groups=False
        )
    return series


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--write", action="store_true",
                        help="store the complete-hours, CPU-capacity-weighted fits in power_model_params.json")
    args = parser.parse_args()

    hours = pd.read_csv(args.csv)
    hours = hours[hours["good_rows"] > 0].copy()
    hours["power"] = hours["sum_power_util"] / hours["good_rows"]
    committed = json.loads((ROOT / "data" / "power_model_params.json").read_text(encoding="utf-8"))["per_cell_cpu_model"]
    capacity = json.loads((ROOT / "data" / "cell_capacity.json").read_text(encoding="utf-8"))["cells"]

    print("1. Domain sizes (coefficient of variation = standard deviation / mean)")
    print(f"{'cell':5s}{'domains':>8s}{'machines':>10s}{'CPU cap':>10s}{'CV mach':>9s}{'CV cap':>8s}"
          f"{'small/large cap':>17s}{'max machine':>13s}{'bad rows %':>12s}")
    sizes = hours.drop_duplicates(["cell", "pdu"])
    for cell in CELLS:
        s = sizes[sizes["cell"] == cell]
        h = hours[hours["cell"] == cell]
        print(f"{cell:5s}{len(s):8d}{s.machines.sum():10.0f}{s.cpu_capacity.sum():10.1f}"
              f"{s.machines.std(ddof=0) / s.machines.mean():9.3f}{s.cpu_capacity.std(ddof=0) / s.cpu_capacity.mean():8.3f}"
              f"{s.cpu_capacity.min() / s.cpu_capacity.max():17.2f}{s.largest_machine.max():13.3f}"
              f"{100 * h.bad_rows.sum() / (h.bad_rows.sum() + h.good_rows.sum()):12.3f}")

    print("\n2. Plain average on all hours, beside the stored fit")
    print(f"{'cell':5s}{'alpha':>10s}{'committed':>11s}{'beta':>10s}{'committed':>11s}{'R2':>9s}{'committed':>11s}{'lowest u':>10s}")
    for cell in CELLS:
        joined = pd.concat([hourly_cpu(cell).rename("cpu"), power_series(hours, cell)["plain"].rename("power")],
                           axis=1, join="inner").dropna()
        r = fit(joined["cpu"].to_numpy(), joined["power"].to_numpy())
        c = committed[cell]
        print(f"{cell:5s}{r['alpha']:10.5f}{c['idle_power']:11.5f}{r['beta']:10.5f}{c['slope']:11.5f}"
              f"{r['r2']:9.5f}{c['r_squared']:11.5f}{joined['cpu'].min():10.3f}")

    print("\n3. Complete hours only (drops the first and last hours). % = change from the committed fit.")
    print(f"{'cell':5s}{'weighting':14s}{'alpha':>8s}{'beta':>8s}{'se':>7s}{'R2':>7s}{'beta %':>8s}"
          f"{'SEs vs plain':>13s}{'rated MW':>10s}{'idle MW':>9s}{'full MW':>9s}{'swing MW':>10s}{'was':>6s}")
    largest_move = 0.0
    totals: dict[str, dict[str, float]] = {}
    for cell in CELLS:
        cpu = hourly_cpu(cell).iloc[1:-1]
        rated = 500.0 * capacity[cell]["cpu_capacity"] / CELL_C_CAPACITY
        c = committed[cell]
        plain = None
        for name, power in power_series(hours, cell).items():
            joined = pd.concat([cpu.rename("cpu"), power.rename("power")], axis=1, join="inner").dropna()
            r = fit(joined["cpu"].to_numpy(), joined["power"].to_numpy())
            plain = plain or r
            move = (r["beta"] - plain["beta"]) / plain["se_beta"]
            if name != "plain":
                largest_move = max(largest_move, abs(move))
            fleet = "training" if cell in "abcd" else "test"
            bucket = totals.setdefault(f"{fleet}, {name}", {"swing": 0.0, "was": 0.0})
            bucket["swing"] += rated * r["beta"]
            bucket["was"] += rated * c["slope"]
            print(f"{cell:5s}{name:14s}{r['alpha']:8.4f}{r['beta']:8.4f}{r['se_beta']:7.4f}{r['r2']:7.3f}"
                  f"{100 * (r['beta'] / c['slope'] - 1):8.1f}{move:13.2f}{rated:10.0f}{rated * r['alpha']:9.0f}"
                  f"{rated * (r['alpha'] + r['beta']):9.0f}{rated * r['beta']:10.0f}{rated * c['slope']:6.0f}")
    print("\nFleet swing in MW (idle to full), committed then new")
    for key, value in totals.items():
        print(f"  {key:26s}{value['was']:8.0f}{value['swing']:8.0f}   {100 * (value['swing'] / value['was'] - 1):+.1f}%")
    print(f"\nLargest slope move from domain weighting: {largest_move:.2f} standard errors of the complete-hours fit.")

    if args.write:
        path = ROOT / "data" / "power_model_params.json"
        params = json.loads(path.read_text(encoding="utf-8"))
        for cell in CELLS:
            power = power_series(hours, cell)["CPU capacity"]
            joined = pd.concat([hourly_cpu(cell).iloc[1:-1].rename("cpu"), power.rename("power")],
                               axis=1, join="inner").dropna()
            r = fit(joined["cpu"].to_numpy(), joined["power"].to_numpy())
            params["per_cell_cpu_model"][cell] = {
                "idle_power": r["alpha"], "slope": r["beta"], "peak_power": r["alpha"] + r["beta"],
                "r_squared": r["r2"], "n": r["n"],
            }
        params["per_cell_cpu_model_method"] = (
            "all usage windows, weighted by window length; complete hours only (twelve five-minute "
            "windows), with each cell's power domains weighted by their machines' CPU capacity; "
            "from scripts/check_power_domain_weights.py"
        )
        params["pooled_diagnostics_note"] = (
            "The pooled fields come from an earlier all-hours, unweighted extraction and are not used."
        )
        path.write_text(json.dumps(params, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
