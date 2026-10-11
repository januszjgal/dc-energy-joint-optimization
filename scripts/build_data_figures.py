"""Draw the two Chapter 3 data figures from the committed data.

* ``workload_week``: hourly service and batch utilization of training cells
  a-d over the first week of the May 2019 trace (Section 3.3).
* ``net_load_week``: hourly gross demand and net load of the four markets over
  23-29 July 2025 UTC, the week of the worked example (Section 3.5.2).

Writes vector PDFs (used by the thesis) and PNG previews to
``docs/figures/data/``. Reads only committed files; nothing is fitted here.

    python scripts/build_data_figures.py
"""

from __future__ import annotations

import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "figures" / "data"

# Validated categorical slots 1-2 (blue, orange) and recessive ink, from the
# dataviz reference palette; checked with its validator on the light surface.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

TRAIN_CELLS = ("a", "b", "c", "d")
MARKETS = (("CAISO_NP15", "CAISO"), ("MISO_MINN_HUB", "MISO"),
           ("SPP_NORTH_HUB", "SPP"), ("ISONE_NEMA", "ISO-NE"))
WEEK_START = pd.Timestamp("2025-07-23T00:00:00Z")
WEEK_END = pd.Timestamp("2025-07-30T00:00:00Z")
WORKED_EXAMPLE_HOUR = pd.Timestamp("2025-07-29T22:00:00Z")


def _style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "text.color": INK,
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.6,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.facecolor": "white",
    })


def _hourly(cell: str, column: str) -> np.ndarray:
    """Hourly means of a five-minute curve, as the simulation uses them."""
    x = pd.read_csv(ROOT / "data" / "cells" / f"cell_{cell}_tiers.csv")[column].to_numpy()
    x = x[: (len(x) // 12) * 12]
    return x.reshape(-1, 12).mean(axis=1)


def workload_week() -> None:
    hours = np.arange(7 * 24)
    fig, axes = plt.subplots(4, 1, figsize=(6.0, 6.2), sharex=True, sharey=True)
    for ax, cell in zip(axes, TRAIN_CELLS):
        service = _hourly(cell, "service_demand_norm")[hours]
        batch = _hourly(cell, "batch_demand_norm")[hours]
        ax.fill_between(hours, 0, service, color=BLUE, linewidth=0)
        ax.fill_between(hours, service, service + batch, color=ORANGE, linewidth=0)
        # Thin surface-colored gap between the two stacked fills.
        ax.plot(hours, service, color="white", linewidth=0.8)
        ax.set_ylim(0, 0.8)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
        ax.set_title(f"Cell {cell}", loc="left", pad=3)
        ax.set_axisbelow(True)
    first = axes[0]
    first.text(30, 0.16, "service", color="white", fontsize=8, va="center")
    first.text(30, _hourly("a", "service_demand_norm")[30] + 0.06, "batch",
               color=INK, fontsize=8, va="center")
    axes[-1].set_xticks(np.arange(0, 7 * 24 + 1, 24))
    axes[-1].set_xticklabels([f"day {d}" for d in range(1, 9)])
    axes[-1].set_xlim(0, 7 * 24)
    axes[-1].set_xlabel("Day of the May 2019 trace")
    fig.supylabel("Fraction of cell capacity", fontsize=9, color=INK_SECONDARY, x=0.015)
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)]
    fig.legend(handles, ["service", "batch"], loc="upper right", ncol=2, frameon=False,
               bbox_to_anchor=(0.98, 1.0))
    fig.tight_layout(rect=(0.02, 0, 1, 0.97))
    _save(fig, "workload_week")


def net_load_week() -> None:
    frames = [pd.read_csv(p) for p in sorted(glob.glob(str(
        ROOT / "data" / "four_market_2025" / "months" / "2025-07" / "canonical_panel.csv")))]
    panel = pd.concat(frames)
    panel["ts"] = pd.to_datetime(panel["timestamp_utc"], utc=True)
    week = panel[(panel.ts >= WEEK_START) & (panel.ts < WEEK_END)]
    fig, axes = plt.subplots(4, 1, figsize=(6.0, 6.6), sharex=True)
    thousands = FuncFormatter(lambda v, _: f"{v:,.0f}")
    for ax, (market_id, name) in zip(axes, MARKETS):
        g = week[week.market_id == market_id].sort_values("ts")
        ax.plot(g.ts, g.gross_demand_mw, color=MUTED, linewidth=1.0, linestyle=(0, (3, 2)))
        ax.plot(g.ts, g.net_load_mw, color=BLUE, linewidth=1.4)
        ax.set_title(name, loc="left", pad=3)
        ax.yaxis.set_major_formatter(thousands)
        ax.set_axisbelow(True)
        if name == "ISO-NE":
            ax.axvline(WORKED_EXAMPLE_HOUR, color=INK_SECONDARY, linewidth=0.6, linestyle=":")
            ax.annotate("worked example", xy=(WORKED_EXAMPLE_HOUR, ax.get_ylim()[0]),
                        xytext=(-4, 3), textcoords="offset points", ha="right", va="bottom",
                        fontsize=7.5, color=INK_SECONDARY)
    caiso = week[week.market_id == "CAISO_NP15"].sort_values("ts").reset_index(drop=True)
    i = 66
    axes[0].text(caiso.ts[i], caiso.gross_demand_mw[i] + 2600, "gross demand",
                 color=INK_SECONDARY, fontsize=8, ha="center")
    axes[0].set_ylim(top=axes[0].get_ylim()[1] + 2500)
    j = 20
    axes[0].text(caiso.ts[j], caiso.net_load_mw[j] - 4200, "net load",
                 color=INK, fontsize=8, ha="center")
    axes[-1].xaxis.set_major_locator(mdates.DayLocator(tz="UTC"))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d", tz="UTC"))
    axes[-1].set_xlim(WEEK_START, WEEK_END)
    axes[-1].set_xlabel("July 2025, UTC")
    fig.supylabel("MW", fontsize=9, color=INK_SECONDARY, x=0.015)
    handles = [plt.Line2D([], [], color=MUTED, linewidth=1.0, linestyle=(0, (3, 2))),
               plt.Line2D([], [], color=BLUE, linewidth=1.4)]
    fig.legend(handles, ["gross demand", "net load"], loc="upper right", ncol=2,
               frameon=False, bbox_to_anchor=(0.98, 1.0))
    fig.tight_layout(rect=(0.02, 0, 1, 0.97))
    _save(fig, "net_load_week")


def _save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=150)
    plt.close(fig)
    print(f"wrote {OUT / name}.pdf and .png")


def main() -> int:
    _style()
    workload_week()
    net_load_week()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
