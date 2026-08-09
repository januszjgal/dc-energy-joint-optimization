"""Quality plots and compact panel diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_market_diagnostics(panel: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    ts = pd.to_datetime(panel["interval_start_utc"], utc=True)
    figure, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    axes[0].plot(ts, panel["da_lmp_usd_mwh"], label="DA LMP", linewidth=0.7)
    axes[0].legend()
    axes[0].set_ylabel("USD/MWh")
    axes[1].plot(ts, panel["gross_level_norm"], label="gross/S", linewidth=0.7)
    axes[1].plot(ts, panel["net_level_norm"], label="net/S", linewidth=0.7)
    axes[1].legend()
    axes[1].set_ylabel("normalized level")
    axes[2].hist(
        panel["net_ramp_1h_norm_per_h"].dropna(),
        bins=80,
        alpha=0.65,
        label="1h",
    )
    axes[2].hist(
        panel["net_ramp_3h_norm_per_h"].dropna(),
        bins=80,
        alpha=0.65,
        label="3h",
    )
    axes[2].legend()
    axes[2].set_ylabel("ramp count")
    axes[3].plot(
        ts,
        panel["incremental_ramp_weighted"],
        linewidth=0.7,
        label="incremental ramp",
    )
    axes[3].legend()
    axes[3].set_ylabel("incremental score")
    axes[3].set_xlabel("UTC")
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)
