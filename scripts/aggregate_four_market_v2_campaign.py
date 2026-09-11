"""Aggregate the locked ten-seed one-factory PPO campaign."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ramp_rl.campaign_statistics import paired_seed_summary, slope_diagnostic  # noqa: E402
from energy_model_v3.four_market_v2 import ARTIFACT_NAMESPACE  # noqa: E402
from ramp_rl.runner import LEARNING_CURVE_COLUMNS  # noqa: E402
from scripts.run_four_market_v2_campaign import (  # noqa: E402
    REQUESTED_TIMESTEPS,
    _campaign_training_geometry,
    _load_completed_seed_summary,
    _validate_campaign_geometry,
)


OUTPUT_ROOT = ROOT / "output" / ARTIFACT_NAMESPACE / "campaign"
SEEDS = tuple(range(4101, 4111))

def _require_complete_summaries(output_root: Path) -> list[dict[str, Any]]:
    missing = [
        seed for seed in SEEDS
        if not (output_root / f"seed-{seed}" / "summary.json").is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "campaign aggregation requires completed summaries for all 10 seeds; "
            f"missing: {', '.join(map(str, missing))}"
        )
    geometry = _validate_campaign_geometry()
    training_geometry = _campaign_training_geometry()
    return [
        _load_completed_seed_summary(
            seed, geometry, training_geometry, output_root=output_root
        )
        for seed in SEEDS
    ]


def _read_curve(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != LEARNING_CURVE_COLUMNS:
            raise ValueError(f"unexpected learning curve columns in {path}")
        return [{key: float(row[key]) for key in LEARNING_CURVE_COLUMNS} for row in reader]


def aggregate_learning_curves(output_root: Path) -> dict[str, Any]:
    curves: dict[int, dict[int, dict[str, float]]] = {}
    for seed in SEEDS:
        path = output_root / f"seed-{seed}" / "learning_curve.csv"
        if not path.is_file():
            raise FileNotFoundError(f"campaign aggregation requires learning curve: {path}")
        rows = _read_curve(path)
        curves[seed] = {int(row["interaction_count"]): row for row in rows}
    aligned_counts = sorted(set.intersection(*(set(curve) for curve in curves.values())))
    if not aligned_counts:
        raise ValueError("learning curves have no common rollout interaction counts")
    aggregate_path = output_root / "learning_curve_aggregate.csv"
    fields = (
        "interaction_count", "seed_count", "mean_raw_joint_reward",
        "median_raw_joint_reward", "mean_raw_peak_reward", "mean_raw_ramp_squared",
        "mean_raw_peak_increment", "mean_raw_ramp_reward",
        "median_raw_ramp_reward", "mean_raw_incremental_ramp_impact",
        "median_raw_incremental_ramp_impact", "mean_episode_count",
        "mean_elapsed_seconds", "mean_completed_episode_joint_J",
    )
    rows: list[dict[str, float | int]] = []
    for interaction_count in aligned_counts:
        values = [curves[seed][interaction_count] for seed in SEEDS]
        rows.append({
            "interaction_count": interaction_count,
            "seed_count": len(values),
            "mean_raw_joint_reward": float(mean(row["mean_raw_joint_reward"] for row in values)),
            "median_raw_joint_reward": float(np.median([row["mean_raw_joint_reward"] for row in values])),
            "mean_raw_peak_reward": float(mean(row["mean_raw_peak_reward"] for row in values)),
            "mean_raw_ramp_squared": float(mean(row["mean_raw_ramp_squared"] for row in values)),
            "mean_raw_peak_increment": float(mean(row["mean_raw_peak_increment"] for row in values)),
            "mean_raw_ramp_reward": float(mean(row["mean_raw_ramp_reward"] for row in values)),
            "median_raw_ramp_reward": float(np.median([row["mean_raw_ramp_reward"] for row in values])),
            "mean_raw_incremental_ramp_impact": float(mean(row["mean_raw_incremental_ramp_impact"] for row in values)),
            "median_raw_incremental_ramp_impact": float(np.median([row["mean_raw_incremental_ramp_impact"] for row in values])),
            "mean_episode_count": float(mean(row["episode_count"] for row in values)),
            "mean_elapsed_seconds": float(mean(row["elapsed_seconds"] for row in values)),
            "mean_completed_episode_joint_J": float(mean(
                -row["mean_completed_episode_return"] for row in values
                if np.isfinite(row["mean_completed_episode_return"])
            )) if any(np.isfinite(row["mean_completed_episode_return"]) for row in values) else float("nan"),
        })
    with aggregate_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    figure_path = output_root / "learning_curve_aggregate.png"
    x = [int(row["interaction_count"]) for row in rows]
    y = [-float(row["mean_raw_joint_reward"]) for row in rows]
    fig, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    axis.plot(x, y, color="#0078d4", linewidth=1.8)
    axis.axvline(110_592, color="#666666", linestyle="--", linewidth=1, label="110,592 comparison point")
    axis.set(xlabel="environment interactions", ylabel="mean hourly joint-objective contribution")
    axis.set_title("Ten-seed joint peak/ramp PPO learning curve")
    axis.legend()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    x_array = np.asarray(x)
    y_array = np.asarray(y)
    return {
        "aligned_rollout_count": len(rows),
        "aggregate_csv": str(aggregate_path.relative_to(ROOT)),
        "aggregate_png": str(figure_path.relative_to(ROOT)),
        "convergence_diagnostic": {
            "around_110592": slope_diagnostic(
                x_array, y_array,
                selector=lambda counts: (counts >= 90_112) & (counts <= 131_072),
                label="90,112 to 131,072 interactions around the prespecified 110,592 comparison point",
            ),
            "final_20_percent": slope_diagnostic(
                x_array, y_array,
                selector=lambda counts: counts >= 0.8 * counts.max(),
                label="final 20 percent of aligned interactions",
            ),
        },
    }


def aggregate(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    rows = _require_complete_summaries(output_root)
    differences = [
        float(row["validation"]["status_quo_comparison"]["policy_minus_status_quo_joint_J"])
        for row in rows
    ]
    markets = sorted(
        rows[0]["validation"]["status_quo_comparison"]["per_market"]
    )
    per_market = {}
    for market in markets:
        market_rows = [
            row["validation"]["status_quo_comparison"]["per_market"][market]
            for row in rows
        ]
        per_market[market] = {
            "mean_policy_monthly_ramp_impact": float(mean(item["policy_ramp_impact_sum"] for item in market_rows)),
            "mean_status_quo_monthly_ramp_impact": float(mean(item["status_quo_ramp_impact_sum"] for item in market_rows)),
            "mean_policy_joint_J": float(mean(item["policy_joint_J"] for item in market_rows)),
            "mean_status_quo_joint_J": float(mean(item["status_quo_joint_J"] for item in market_rows)),
            "mean_joint_improvement": float(mean(item["improvement"] for item in market_rows)),
            "mean_policy_peak_mw": float(mean(item["policy_peak_mw"] for item in market_rows)),
            "mean_status_quo_peak_mw": float(mean(item["status_quo_peak_mw"] for item in market_rows)),
            "mean_peak_reduction_mw": float(mean(item["peak_reduction_mw"] for item in market_rows)),
            "mean_policy_native_relative_incremental_ramp_impact": float(
                mean(
                    item["policy_native_relative_incremental_ramp_impact"]
                    for item in market_rows
                )
            ),
            "mean_status_quo_native_relative_incremental_ramp_impact": float(
                mean(
                    item["status_quo_native_relative_incremental_ramp_impact"]
                    for item in market_rows
                )
            ),
            "mean_policy_minus_status_quo_incremental_ramp_impact": float(
                mean(
                    item["policy_minus_status_quo_incremental_ramp_impact"]
                    for item in market_rows
                )
            ),
            "policy_seed_win_count": int(
                sum(item["policy_outperforms_status_quo"] for item in market_rows)
            ),
        }
    safety_fields = (
        "service_unserved",
        "batch_unfinished",
        "batch_expired",
        "terminal_work",
        "certificate_violations",
    )
    seed_summaries = [
        {
            "seed": int(row["seed"]),
            "effective_interactions": int(row["effective_interactions"]),
            "resumed_from_interactions": int(row["resumed_from_interactions"]),
            "training_elapsed_seconds": float(row["training_elapsed_seconds"]),
            "policy_joint_J": float(row["validation"]["status_quo_comparison"]["policy_J"]),
            "status_quo_joint_J": float(row["validation"]["status_quo_comparison"]["status_quo_J"]),
            "components": row["validation"]["status_quo_comparison"]["components"],
            "policy_native_relative_incremental_ramp_impact": float(
                row["validation"]["mean_incremental_ramp_impact"]
            ),
            "status_quo_native_relative_incremental_ramp_impact": float(
                row["validation"]["status_quo_comparison"][
                    "status_quo_native_relative_mean_incremental_ramp_impact"
                ]
            ),
            "policy_minus_status_quo_incremental_ramp_impact": float(
                row["validation"]["status_quo_comparison"][
                    "policy_minus_status_quo_mean_incremental_ramp_impact"
                ]
            ),
            "improvement": float(
                row["validation"]["status_quo_comparison"]["improvement"]
            ),
        }
        for row in rows
    ]
    result = {
        "objective": rows[0]["validation"]["objective"],
        "seed_count": len(rows),
        "requested_interactions_per_seed": REQUESTED_TIMESTEPS,
        "validation_month_ids": list(rows[0]["validation_window_ids"]),
        "validation_days_per_seed": int(rows[0]["validation"]["day_count"]),
        "improvement_definition": (
            "status-quo minus policy mean monthly normalized joint score; positive favors the policy"
        ),
        "mean_improvement": float(-mean(differences)),
        "statistical_unit": "optimizer seed",
        "mean_monthly_ramp_impact": float(mean(row["validation"]["mean_monthly_ramp_impact"] for row in rows)),
        "mean_joint_J": float(mean(row["validation"]["mean_joint_J"] for row in rows)),
        "sample_standard_deviation_joint_J": float(stdev(row["validation"]["mean_joint_J"] for row in rows)),
        "mean_incremental_ramp_impact": float(mean(
            float(row["validation"]["mean_incremental_ramp_impact"]) for row in rows
        )),
        "sample_standard_deviation_incremental_ramp_impact": float(stdev(
            float(row["validation"]["mean_incremental_ramp_impact"]) for row in rows
        )),
        "per_market": per_market,
        "safety_totals": {
            field: float(
                sum(float(row["validation"][field]) for row in rows)
            )
            for field in safety_fields
        },
        "mean_emergency_feasibility_rate": float(
            mean(
                float(row["validation"]["emergency_feasibility_rate"])
                for row in rows
            )
        ),
        "paired_policy_minus_status_quo": paired_seed_summary(differences, metric="monthly joint J"),
        "component_comparisons": {
            component: paired_seed_summary(
                [
                    -float(row["validation"]["status_quo_comparison"]["components"][component]["improvement"])
                    for row in rows
                ],
                metric=(
                    "monthly sum of incremental squared ramp impacts"
                    if component == "ramp" else "summed normalized regional monthly peaks"
                ),
            )
            for component in ("ramp", "net_load_peak")
        },
        "learning_curves": aggregate_learning_curves(output_root),
        "seeds": seed_summaries,
    }
    path = output_root / "aggregation.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    argparse.ArgumentParser(
        description="Aggregate completed summaries from the locked ten-seed PPO campaign."
    ).parse_args()
    result = aggregate()
    print(json.dumps({
        "seed_count": result["seed_count"],
        "wilcoxon_p_value": result["paired_policy_minus_status_quo"]["wilcoxon_signed_rank"]["p_value"],
    }, indent=2))


if __name__ == "__main__":
    main()
