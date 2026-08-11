"""Aggregate paired seed-level four-market v2 validation outcomes."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = ROOT / "output" / "four_market_v2" / "campaign"
SEEDS = (4101, 4102, 4103)
VARIANTS = ("envelope_on", "envelope_off")


def _summary(variant: str, seed: int) -> dict[str, Any]:
    return json.loads(
        (OUTPUT_ROOT / variant / f"seed-{seed}" / "summary.json").read_text()
    )


def _seed_row(summary: dict[str, Any]) -> dict[str, Any]:
    validation = summary["validation"]
    comparison = validation["status_quo_comparison"]
    return {
        "seed": summary["seed"],
        "native_relative_incremental_ramp_impact": validation[
            "mean_policy_native_relative_incremental_ramp_impact"
        ],
        "policy_minus_status_quo_impact": comparison[
            "policy_minus_status_quo_mean_incremental_ramp_impact"
        ],
        "day_ahead_cost_ratio": validation["energy_cost_ratio"],
        "service_unserved": validation["service_unserved"],
        "batch_unfinished": validation["batch_unfinished"],
        "batch_expired": validation["batch_expired"],
        "terminal_work": validation["terminal_work"],
        "certificate_violations": validation["certificate_violations"],
        "emergency_fallback_rate": validation["emergency_feasibility_rate"],
        "behavior_audit": validation["behavior_audit"],
        "per_market": comparison["per_market"],
        "success_gate": validation["success_gate"],
        "elapsed_seconds": summary["training_elapsed_seconds"],
        "throughput_interactions_per_second": summary[
            "training_throughput_interactions_per_second"
        ],
    }


def _mean(rows: list[dict[str, Any]], field: str) -> float:
    return float(mean(float(row[field]) for row in rows))


def _per_market_mean(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    fields = (
        "policy_native_relative_incremental_ramp_impact",
        "status_quo_native_relative_incremental_ramp_impact",
        "policy_minus_status_quo_incremental_ramp_impact",
    )
    return {
        market: {
            field: float(mean(row["per_market"][market][field] for row in rows))
            for field in fields
        }
        for market in rows[0]["per_market"]
    }


def main() -> None:
    seed_results = {
        variant: [_seed_row(_summary(variant, seed)) for seed in SEEDS]
        for variant in VARIANTS
    }
    aggregates = {
        variant: {
            "seed_count": len(rows),
            "mean_native_relative_incremental_ramp_impact": _mean(
                rows, "native_relative_incremental_ramp_impact"
            ),
            "sample_standard_deviation_native_relative_incremental_ramp_impact": stdev(
                float(row["native_relative_incremental_ramp_impact"]) for row in rows
            ),
            "minimum_native_relative_incremental_ramp_impact": min(
                float(row["native_relative_incremental_ramp_impact"]) for row in rows
            ),
            "maximum_native_relative_incremental_ramp_impact": max(
                float(row["native_relative_incremental_ramp_impact"]) for row in rows
            ),
            "mean_policy_minus_status_quo_impact": _mean(
                rows, "policy_minus_status_quo_impact"
            ),
            "mean_day_ahead_cost_ratio": _mean(rows, "day_ahead_cost_ratio"),
            "mean_emergency_fallback_rate": _mean(rows, "emergency_fallback_rate"),
            "mean_training_elapsed_seconds": _mean(rows, "elapsed_seconds"),
            "mean_throughput_interactions_per_second": _mean(
                rows, "throughput_interactions_per_second"
            ),
            "per_market_mean": _per_market_mean(rows),
            "all_feasible": all(
                row["service_unserved"] == 0.0
                and row["batch_unfinished"] == 0.0
                and row["batch_expired"] == 0.0
                and row["terminal_work"] == 0.0
                and row["certificate_violations"] == 0
                for row in rows
            ),
        }
        for variant, rows in seed_results.items()
    }
    pairs = []
    for on, off in zip(seed_results["envelope_on"], seed_results["envelope_off"]):
        difference = (
            off["native_relative_incremental_ramp_impact"]
            - on["native_relative_incremental_ramp_impact"]
        )
        pairs.append(
            {
                "seed": on["seed"],
                "envelope_off_minus_envelope_on_native_relative_impact": difference,
                "envelope_off_outperforms_envelope_on": difference < 0.0,
            }
        )
    paired_differences = [
        pair["envelope_off_minus_envelope_on_native_relative_impact"]
        for pair in pairs
    ]
    result = {
        "schema_version": "four-market-v2-paired-aggregation-v1",
        "classification": "large_non_final_validation_only",
        "seed_results": seed_results,
        "variant_aggregates": aggregates,
        "paired_variant_comparison": {
            "pairs": pairs,
            "envelope_off_outperforms_envelope_on_seed_count": sum(
                pair["envelope_off_outperforms_envelope_on"] for pair in pairs
            ),
            "envelope_off_outperforms_envelope_on_consistently": all(
                pair["envelope_off_outperforms_envelope_on"] for pair in pairs
            ),
            "mean_envelope_off_minus_envelope_on_native_relative_impact": float(
                mean(paired_differences)
            ),
            "sample_standard_deviation_of_paired_difference": float(
                stdev(paired_differences)
            ),
        },
        "sealed_test_accessed": False,
    }
    path = OUTPUT_ROOT / "paired_aggregation.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["paired_variant_comparison"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
