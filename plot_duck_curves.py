"""Plot regional duck curves from EIA-930 net demand data.

Visualizes the actual grid duck curve in each region: the daily profile of
net demand (total load minus utility-scale solar + wind), normalized to
[0, 1] by regional peak. The duck-curve neck — when solar drops off and
residential load ramps up in the evening — is the period when DC load
contributes most to grid stress.

Usage:
    python plot_duck_curves.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

NET_DEMAND_FILES = {
    "CAISO (US-West / California)": "data/net_demand/caiso.csv",
    "MISO (US-Central / Iowa)": "data/net_demand/miso.csv",
    "Southern Co (US-SE-1 / Georgia)": "data/net_demand/southern_co.csv",
    "Duke Carolinas (US-SE-2 / SC)": "data/net_demand/duke_carolinas.csv",
}

GLOBAL_FILES = {
    "ENTSO-E NL (Global-EU)": "data/net_demand/entso_e_nl.csv",
    "EMA Singapore (Global-Asia)": "data/net_demand/ema_singapore.csv",
}

PRICE_FILES = {
    "CAISO": "data/prices/caiso.csv",
    "MISO": "data/prices/miso.csv",
    "Southern Co": "data/prices/southern_co.csv",
    "Duke Carolinas": "data/prices/duke_carolinas.csv",
}

STEPS_PER_DAY = 288
ROOT = Path(__file__).resolve().parent


def load_daily_profiles(files: dict[str, str], col: str) -> dict[str, dict]:
    """Compute mean / p25 / p75 daily profiles for each region."""
    profiles = {}
    for label, path in files.items():
        df = pd.read_csv(ROOT / path)
        values = df[col].values.astype(np.float64)
        n_days = len(values) // STEPS_PER_DAY
        mat = values[: n_days * STEPS_PER_DAY].reshape(n_days, STEPS_PER_DAY)
        profiles[label] = {
            "mean": mat.mean(axis=0),
            "p25": np.percentile(mat, 25, axis=0),
            "p75": np.percentile(mat, 75, axis=0),
        }
    return profiles


def make_figure(
    nd_us: dict, nd_global: dict, price_us: dict, out_path: Path
) -> None:
    hours = np.linspace(0, 24, STEPS_PER_DAY, endpoint=False)

    us_colors = ["#e6194b", "#3cb44b", "#4363d8", "#f58231"]
    gl_colors = ["#911eb4", "#42d4f4"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Regional Duck Curves: Average Daily Grid Net Demand by Location\n"
        "(EIA-930 hourly data, May 2019, normalized [0,1] by regional peak)",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    # Panel A: US net demand profiles
    ax = axes[0, 0]
    for (label, data), color in zip(nd_us.items(), us_colors):
        short = label.split("(")[0].strip()
        ax.plot(hours, data["mean"], color=color, lw=2, label=short)
        ax.fill_between(
            hours, data["p25"], data["p75"], color=color, alpha=0.15
        )
    ax.axvspan(17, 21, color="red", alpha=0.07)
    ax.set_title("A  –  US Regions: Grid Net Demand (normalized)", fontweight="bold")
    ax.set_ylabel("Net Demand (fraction of peak)")
    ax.set_xlabel("Hour of Day")
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    # Panel B: US price profiles for context
    ax = axes[0, 1]
    for (label, path), color in zip(PRICE_FILES.items(), us_colors):
        df = pd.read_csv(ROOT / path)
        price = df["price_usd_kwh"].values.astype(np.float64)
        n_days = len(price) // STEPS_PER_DAY
        mat = price[: n_days * STEPS_PER_DAY].reshape(n_days, STEPS_PER_DAY)
        ax.plot(hours, mat.mean(axis=0) * 1000, color=color, lw=2, label=label)
    ax.axvspan(17, 21, color="red", alpha=0.07)
    ax.set_title("B  –  US Regions: Wholesale Price for Context", fontweight="bold")
    ax.set_ylabel("Price ($/MWh)")
    ax.set_xlabel("Hour of Day")
    ax.set_xlim(0, 24)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Panel C: All regions net demand
    ax = axes[1, 0]
    all_regions = {**nd_us, **nd_global}
    all_colors = us_colors + gl_colors
    for (label, data), color in zip(all_regions.items(), all_colors):
        short = label.split("(")[0].strip()
        ax.plot(hours, data["mean"], color=color, lw=2, label=short)
    ax.set_title("C  –  All Regions: Net Demand Comparison", fontweight="bold")
    ax.set_ylabel("Net Demand (fraction of peak)")
    ax.set_xlabel("Hour of Day")
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(True, alpha=0.3)

    # Panel D: Heatmap
    ax = axes[1, 1]
    short_labels = [l.split("(")[0].strip() for l in all_regions.keys()]
    nd_matrix = np.array([data["mean"] for data in all_regions.values()])

    im = ax.imshow(
        nd_matrix,
        aspect="auto",
        extent=[0, 24, len(short_labels) - 0.5, -0.5],
        cmap="RdYlGn_r",
        vmin=0.3,
        vmax=1.0,
    )
    ax.set_yticks(range(len(short_labels)))
    ax.set_yticklabels(short_labels, fontsize=8)
    ax.set_xlabel("Hour of Day")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
    ax.set_title("D  –  Net Demand Heatmap by Region & Hour", fontweight="bold")
    fig.colorbar(im, ax=ax, label="Net Demand (norm)", fraction=0.046, pad=0.04)
    ax.axvline(17, color="red", lw=1.2, ls="--", alpha=0.7)
    ax.axvline(21, color="red", lw=1.2, ls="--", alpha=0.7)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    print("Loading net demand data...")
    nd_us = load_daily_profiles(NET_DEMAND_FILES, "net_demand_normalized")
    nd_global = load_daily_profiles(GLOBAL_FILES, "net_demand_normalized")
    price_us = load_daily_profiles(PRICE_FILES, "price_usd_kwh")

    out = ROOT / "output" / "duck_curves.png"
    out.parent.mkdir(exist_ok=True)
    make_figure(nd_us, nd_global, price_us, out)

    print("\nNet demand peak hour per region:")
    all_profiles = {**nd_us, **nd_global}
    for label, data in all_profiles.items():
        peak_step = int(np.argmax(data["mean"]))
        peak_hour = peak_step / STEPS_PER_DAY * 24
        trough_step = int(np.argmin(data["mean"]))
        trough_hour = trough_step / STEPS_PER_DAY * 24
        amp = data["mean"].max() - data["mean"].min()
        print(
            f"  {label.split('(')[0].strip():<32} "
            f"peak={peak_hour:5.1f}h  "
            f"trough={trough_hour:5.1f}h  "
            f"amplitude={amp:.2f}"
        )
