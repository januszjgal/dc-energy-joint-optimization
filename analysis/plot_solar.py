"""Plot solar availability across all DC locations.

Overlays solar profiles for US-only and global scenarios to
illustrate the timezone/solar diversity advantage.

Usage:
    python analysis/plot_solar.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "solar"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

# Location groupings
US_LOCATIONS = {
    "the_dalles_or": "The Dalles, OR (UTC-8)",
    "council_bluffs_ia": "Council Bluffs, IA (UTC-6)",
    "douglas_county_ga": "Douglas County, GA (UTC-5)",
    "berkeley_county_sc": "Berkeley County, SC (UTC-5)",
}

GLOBAL_LOCATIONS = {
    "the_dalles_or": "The Dalles, OR (UTC-8)",
    "council_bluffs_ia": "Council Bluffs, IA (UTC-6)",
    "eemshaven_nl": "Eemshaven, NL (UTC+1)",
    "singapore": "Singapore (UTC+8)",
}


def plot_group(locations: dict, title: str, filename: str, days: int = 3) -> None:
    """Plot solar profiles for a group of locations, zoomed to N days."""
    fig, ax = plt.subplots(figsize=(14, 5))
    max_steps = days * 288  # 288 steps per day

    for file_key, label in locations.items():
        path = DATA_DIR / f"{file_key}.csv"
        if not path.exists():
            print(f"  Skipping {file_key} (file not found)")
            continue
        df = pd.read_csv(path)
        df = df[df["timestep"] < max_steps]
        ax.plot(df["timestep"], df["solar_fraction"], label=label, linewidth=1.0)

    ax.set_xlabel("Timestep (5-min intervals)")
    ax.set_ylabel("Solar Capacity Factor")
    ax.set_title(f"{title} (first {days} days)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Add day markers
    for d in range(days):
        ax.axvline(d * 288, color="gray", linestyle="--", alpha=0.3)

    fig.tight_layout()
    out_path = OUTPUT_DIR / filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved to {out_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Plotting US solar profiles...")
    plot_group(US_LOCATIONS, "Solar Availability: US Model", "solar_us.png")

    print("Plotting global solar profiles...")
    plot_group(GLOBAL_LOCATIONS, "Solar Availability: Global Model", "solar_global.png")


if __name__ == "__main__":
    main()
