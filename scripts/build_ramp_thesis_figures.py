"""Generate ramp-thesis figures from committed v1/v3 non-test evidence."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "figures" / "ramp_v6"
V1_RESULTS = ROOT / "output" / "ramp_rl_v6" / "live" / "final_results.json"
SOURCE_CONTRACT = (
    ROOT / "data" / "energy_model_v3" / "provenance" / "source_contract.json"
)


def build_v1_closeout() -> None:
    payload = json.loads(V1_RESULTS.read_text(encoding="utf-8"))
    stages = ["100k screen", "500k confirmation"]
    impacts = [
        payload["screen"]["aggregate"]["mean_incremental_ramp_impact"],
        payload["confirmation"]["aggregate"]["mean_incremental_ramp_impact"],
    ]
    costs = [
        payload["screen"]["aggregate"]["mean_energy_cost_ratio"],
        payload["confirmation"]["aggregate"]["mean_energy_cost_ratio"],
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    colors = ["#2f6f9f", "#b44d4d"]
    axes[0].bar(stages, impacts, color=colors)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Mean incremental ramp impact")
    axes[0].set_title("Validation ramp metric (lower is better)")
    axes[0].tick_params(axis="x", rotation=12)
    axes[1].bar(stages, costs, color=colors)
    axes[1].axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_ylabel("DA energy cost / status quo")
    axes[1].set_title("Validation cost ratio")
    axes[1].tick_params(axis="x", rotation=12)
    fig.suptitle("Immutable v1 validation-only campaign closeout")
    fig.text(
        0.5,
        0.01,
        "Confirmation failed strict per-seed/per-market gates; sealed test unopened.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(OUTPUT / "v1_validation_closeout.png", dpi=180)
    plt.close(fig)


def build_six_market_design() -> None:
    contract = json.loads(SOURCE_CONTRACT.read_text(encoding="utf-8"))
    ordered = [
        ("CAISO_NP15", "CAISO NP15", "cell a"),
        ("ERCOT_LZ_NORTH", "ERCOT North", "cell b"),
        ("NYISO_NYC_J", "NYISO Zone J", "cell c"),
        ("MISO_MINN_HUB", "MISO Minnesota", "cell d"),
        ("SPP_NORTH_HUB", "SPP North", "cell e"),
        ("ISONE_NEMA", "ISO-NE NEMA", "cell f"),
    ]
    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")
    for index, (market, label, cell) in enumerate(ordered):
        y = 6.25 - index * 0.85
        descriptors = contract["sources"][market]
        price = descriptors[0]["source"]
        physical = descriptors[-1]["source"]
        ax.add_patch(
            plt.Rectangle((0.2, y - 0.28), 2.4, 0.56, color="#d9eaf7", ec="#376996")
        )
        ax.text(1.4, y, f"{label}\n{cell}", ha="center", va="center", fontsize=9)
        ax.annotate("", xy=(3.1, y), xytext=(2.6, y), arrowprops={"arrowstyle": "->"})
        ax.text(3.2, y + 0.12, f"Price: {price}", fontsize=7.5, va="center")
        ax.text(3.2, y - 0.12, f"Physical: {physical}", fontsize=7.5, va="center")
    ax.add_patch(
        plt.Rectangle((8.7, 1.2), 2.8, 4.9, color="#f2f2f2", ec="#555555")
    )
    ax.text(10.1, 5.65, "Common UTC hourly panel", ha="center", weight="bold")
    ax.text(10.1, 4.85, "Sep 2025-Jan 2026\nTRAIN", ha="center", va="center")
    ax.text(10.1, 3.55, "Feb 2026\nVALIDATION", ha="center", va="center")
    ax.text(10.1, 2.25, "Mar-Apr 2026\nSEALED TEST", ha="center", va="center")
    ax.text(
        6.0,
        0.45,
        "PJM DOM / Northern Virginia: credential-blocked, excluded, not evaluated",
        ha="center",
        color="#9b2c2c",
        weight="bold",
    )
    ax.set_title("Energy model v3: six independent evaluated markets and fixed split")
    fig.tight_layout()
    fig.savefig(OUTPUT / "six_market_study_design.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    build_v1_closeout()
    build_six_market_design()
    print(f"Wrote figures under {OUTPUT}")


if __name__ == "__main__":
    main()
