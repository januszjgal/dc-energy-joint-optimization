"""Build a validation-only report for a frozen ramp-RL v2 candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--evidence-index", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    summary = _read(args.summary)
    evidence_index = _read(args.evidence_index)
    baseline = _read(args.baseline)
    if summary["sealed_test_used"] or not evidence_index["passed"]:
        raise RuntimeError("candidate evidence is invalid or used sealed test")
    rows = summary["rows"]
    market_values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for market, value in row["per_market_macro"].items():
            market_values[market].append(float(value))
    current = summary["aggregates"]["ppo"]
    previous = baseline["aggregates"]["ppo"]
    result = {
        "schema_version": "ramp-pure-rl-v2-candidate-result-v1",
        "candidate": args.candidate,
        "selection_split": "validation",
        "sealed_test_opened": False,
        "confirmation_launched": False,
        "protocol_id": evidence_index["protocol_id"],
        "protocol_sha256": evidence_index["protocol_sha256"],
        "source_commit": "4d47a9a",
        "training": {
            "random_initialization_only": True,
            "seeds": [int(row["seed"]) for row in rows],
            "requested_timesteps": 100000,
            "effective_boundary_timesteps": 110592,
            "reward_normalization": False,
        },
        "aggregate": current,
        "baseline_v1_screen": previous,
        "effect_vs_v1_screen": {
            "mean_incremental_ramp_impact_delta": (
                current["mean_incremental_ramp_impact"]
                - previous["mean_incremental_ramp_impact"]
            ),
            "ramp_improvement_magnitude_change_pct": 100.0
            * (
                abs(current["mean_incremental_ramp_impact"])
                - abs(previous["mean_incremental_ramp_impact"])
            )
            / abs(previous["mean_incremental_ramp_impact"]),
            "mean_energy_cost_ratio_delta": (
                current["mean_energy_cost_ratio"]
                - previous["mean_energy_cost_ratio"]
            ),
        },
        "rows": rows,
        "per_market_mean_incremental_ramp_impact": {
            market: mean(values)
            for market, values in sorted(market_values.items())
        },
        "promotion_decision": summary["promotion_decision"],
        "artifacts": {
            "summary": {
                "path": str(args.summary),
                "sha256": _sha256(args.summary),
            },
            "evidence_index": {
                "path": str(args.evidence_index),
                "sha256": _sha256(args.evidence_index),
            },
        },
    }
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        f"# Ramp-RL {args.candidate} validation report",
        "",
        f"Protocol `{result['protocol_id']}` (`{result['protocol_sha256']}`) was "
        "trained from random initialization for a nominal 100k steps "
        "(110,592 complete-boundary interactions) on seeds 2701-2703.",
        "",
        "| Seed | Ramp impact | Cost ratio | Strict gate | Failed gates |",
        "|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['mean_incremental_ramp_impact']:.12g} | "
            f"{row['energy_cost_ratio']:.10f} | "
            f"{row['success_gate_pass']} | "
            f"{', '.join(row['failed_gates']) or '-'} |"
        )
    effect = result["effect_vs_v1_screen"]
    lines.extend(
        [
            "",
            f"Three-seed mean ramp impact was "
            f"`{current['mean_incremental_ramp_impact']:.12g}` and mean cost ratio "
            f"was `{current['mean_energy_cost_ratio']:.10f}`. Ramp-improvement "
            f"magnitude changed by "
            f"`{effect['ramp_improvement_magnitude_change_pct']:.4f}%` versus the "
            "frozen v1 100k PPO screen.",
            "",
            "All three seeds failed at least one strict validation gate. Candidate "
            "A therefore stops at validation; seeds 2704-2705 were not launched and "
            "the sealed March-April test remains unopened.",
            "",
        ]
    )
    args.report.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
