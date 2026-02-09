"""Compare US model vs Global model optimization results.

This is the key thesis figure: shows how much additional optimization
potential exists when routing is unconstrained (global) vs. constrained
to a single country (US).

Usage:
    python analysis/plot_us_vs_global.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def load_results(scenario_name: str) -> dict | None:
    """Load evaluation results JSON for a scenario."""
    path = OUTPUT_DIR / f"{scenario_name}_results.json"
    if not path.exists():
        print(f"  Results not found: {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    us_results = load_results("us_model")
    global_results = load_results("global_model")

    if us_results is None or global_results is None:
        print("Run evaluate.py for both scenarios first.")
        return

    # Extract total costs for each policy
    policies = list(us_results.keys())
    us_costs = [us_results[p]["summary"]["total_cost"] for p in policies]
    global_costs = [global_results[p]["summary"]["total_cost"] for p in policies]

    # Bar chart comparison
    x = np.arange(len(policies))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars_us = ax.bar(x - width / 2, us_costs, width, label="US Model", color="steelblue")
    bars_global = ax.bar(x + width / 2, global_costs, width, label="Global Model", color="coral")

    ax.set_ylabel("Total Energy Cost ($)")
    ax.set_title("US Model vs Global Model: Total Cost by Policy")
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=15, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Add percentage difference labels
    for i, (u, g) in enumerate(zip(us_costs, global_costs)):
        if u > 0:
            pct = (u - g) / u * 100
            ax.annotate(
                f"{pct:+.1f}%",
                xy=(x[i] + width / 2, g),
                ha="center",
                va="bottom",
                fontsize=8,
                color="darkred",
            )

    fig.tight_layout()
    out_path = OUTPUT_DIR / "us_vs_global_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved to {out_path}")

    # Renewable fraction comparison
    us_renew = [us_results[p]["summary"]["avg_renewable_frac"] * 100 for p in policies]
    global_renew = [global_results[p]["summary"]["avg_renewable_frac"] * 100 for p in policies]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, us_renew, width, label="US Model", color="steelblue")
    ax.bar(x + width / 2, global_renew, width, label="Global Model", color="coral")
    ax.set_ylabel("Average Renewable Utilization (%)")
    ax.set_title("US Model vs Global Model: Renewable Utilization by Policy")
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=15, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    renew_path = OUTPUT_DIR / "us_vs_global_renewable.png"
    fig.savefig(renew_path, dpi=150)
    plt.close(fig)
    print(f"Saved to {renew_path}")


if __name__ == "__main__":
    main()
