"""No-training objective-coefficient review for energy model v2."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.compute_qp_optimum import build_inputs, solve_clarabel  # noqa: E402
from scripts.review_energy_model_v2 import (  # noqa: E402
    COMPLETION_PENALTY,
    evaluate_baselines,
)

YEAR = 2025
ALPHAS = (0.0, 0.005, 0.015, 0.03)
CALIBRATION_SCENARIOS = (
    ("US a-d", "us_model_v2_2025.yaml"),
    ("Global a-d", "global_model_v2_2025.yaml"),
)
STATUS_QUO = "Status Quo (local, no deferral)"


def build() -> dict[str, Any]:
    output = ROOT / "output" / "energy_model_v2" / str(YEAR)
    output.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "year": YEAR,
        "calibration_scope": (
            "a-d reference cells only; held-out e-h cells are not used to "
            "select the objective coefficient"
        ),
        "alphas": list(ALPHAS),
        "demand_charge_treatment": (
            "excluded from primary objective; reported at the standardized "
            "$15/kW-cycle reference as a secondary sensitivity"
        ),
        "scenarios": {},
    }

    for label, filename in CALIBRATION_SCENARIOS:
        scenario = ROOT / "env" / "scenarios" / filename
        entries = {}
        for alpha in ALPHAS:
            spatial_baselines = evaluate_baselines(
                scenario,
                False,
                peak_penalty_weight=alpha,
            )
            batch_baselines = evaluate_baselines(
                scenario,
                True,
                peak_penalty_weight=alpha,
            )
            spatial_sq = spatial_baselines[STATUS_QUO]
            batch_sq = batch_baselines[STATUS_QUO]
            spatial_qp = solve_clarabel(
                build_inputs(
                    str(scenario),
                    False,
                    peak_penalty_weight=alpha,
                ),
                False,
            )
            batch_qp = solve_clarabel(
                build_inputs(
                    str(scenario),
                    True,
                    enforce_batch_completion=True,
                    completion_penalty_weight=COMPLETION_PENALTY,
                    peak_penalty_weight=alpha,
                ),
                True,
            )
            entries[str(alpha)] = {
                "spatial_status_quo": spatial_sq,
                "batch_status_quo": batch_sq,
                "spatial_qp": spatial_qp,
                "batch_qp": batch_qp,
                "spatial_headroom_pct": (
                    100.0
                    * (spatial_sq["total_cost"] - spatial_qp["optimum_total"])
                    / spatial_sq["total_cost"]
                ),
                "joint_headroom_pct": (
                    100.0
                    * (batch_sq["total_cost"] - batch_qp["optimum_total"])
                    / batch_sq["total_cost"]
                ),
                "incremental_temporal_headroom_pct": (
                    100.0
                    * (
                        spatial_qp["optimum_total"]
                        - batch_qp["optimum_total"]
                    )
                    / spatial_qp["optimum_total"]
                ),
            }
        result["scenarios"][label] = entries

    (output / "objective_sensitivity.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    plot(result, output / "objective_sensitivity.png")
    write_report(result, output / "objective_sensitivity.md")
    return result


def plot(result: dict[str, Any], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = {"US a-d": "#35637b", "Global a-d": "#b94d2f"}

    for label, entries in result["scenarios"].items():
        x = list(ALPHAS)
        component_share = [
            100.0
            * entries[str(alpha)]["spatial_status_quo"][
                "total_peak_penalty"
            ]
            / entries[str(alpha)]["spatial_status_quo"][
                "total_energy_cost"
            ]
            for alpha in x
        ]
        axes[0].plot(
            x,
            component_share,
            marker="o",
            label=label,
            color=colors[label],
        )
        spatial = [
            entries[str(alpha)]["spatial_headroom_pct"] for alpha in x
        ]
        joint = [entries[str(alpha)]["joint_headroom_pct"] for alpha in x]
        axes[1].plot(
            x,
            spatial,
            marker="o",
            linestyle="--",
            color=colors[label],
            label=f"{label} spatial",
        )
        axes[1].plot(
            x,
            joint,
            marker="s",
            color=colors[label],
            label=f"{label} joint",
        )

    axes[0].set_title("Status Quo grid-stress / energy component")
    axes[0].set_xlabel("alpha")
    axes[0].set_ylabel("Grid-stress cost as % of energy cost")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].set_title("Clairvoyant headroom vs Status Quo")
    axes[1].set_xlabel("alpha")
    axes[1].set_ylabel("Headroom (%)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)

    fig.suptitle(
        "Energy model v2 objective sensitivity (a-d calibration cells only)"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Energy Model v2 Objective Sensitivity",
        "",
        "> No PPO training is involved. Alpha is calibrated on a-d only; e-h "
        "remains held out.",
        "",
        "The primary objective includes real energy cost plus the convex "
        "`alpha * grid_mw^2 * max(net_demand_signed, 0)` grid-stress term. "
        "The standardized $15/kW-cycle demand charge is excluded from the "
        "primary reward and reported as a secondary sensitivity.",
        "",
        "| Scenario | Alpha | Stress / energy | Spatial headroom | "
        "Joint headroom | Temporal | Ref demand charge |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, entries in result["scenarios"].items():
        for alpha in ALPHAS:
            entry = entries[str(alpha)]
            sq = entry["spatial_status_quo"]
            share = (
                100.0
                * sq["total_peak_penalty"]
                / sq["total_energy_cost"]
            )
            lines.append(
                f"| {label} | {alpha:.3f} | {share:.1f}% | "
                f"{entry['spatial_headroom_pct']:.2f}% | "
                f"{entry['joint_headroom_pct']:.2f}% | "
                f"{entry['incremental_temporal_headroom_pct']:.2f}% | "
                f"${sq['demand_charge_ref'] / 1e6:.3f}M |"
            )
    lines += [
        "",
        "![Objective sensitivity](objective_sensitivity.png)",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()
