"""Seal validation-only ramp-RL stage metrics and promotion decisions."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ramp_rl.campaign import extension_allowed, promotion_decision  # noqa: E402
from ramp_rl.evidence import sha256_file  # noqa: E402
from ramp_rl.schema import DEFAULT_PROTOCOL_PATH, load_protocol  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--expected-seeds", type=int, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--previous-summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    rows = []
    for path in sorted(args.validation_root.glob("*_validation.json")):
        match = re.fullmatch(r"(ppo|sac)_(\d+)_validation\.json", path.name)
        if match is None:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        safety = (
            float(payload["service_unserved"]) == 0.0
            and float(payload["batch_unfinished"]) == 0.0
            and float(payload["batch_expired"]) == 0.0
            and float(payload["terminal_work"]) == 0.0
            and int(payload["certificate_violations"]) == 0
        )
        rows.append(
            {
                "algorithm": match.group(1),
                "seed": int(match.group(2)),
                "split": "validation",
                "episode_count": int(payload["episode_count"]),
                "mean_incremental_ramp_impact": float(
                    payload["mean_incremental_ramp_impact"]
                ),
                "per_market_macro": payload["per_market_macro"],
                "energy_cost_ratio": float(payload["energy_cost_ratio"]),
                "energy_budget_pass": float(payload["energy_cost_ratio"]) <= 1.02,
                "safety_pass": safety,
                "emergency_feasibility_rate": float(
                    payload["emergency_feasibility_rate"]
                ),
                "behavior_audit": payload["behavior_audit"],
                "success_gate_pass": bool(payload["success_gate"]["passed"]),
                "failed_gates": list(payload["success_gate"]["failed_gates"]),
                "validation_artifact": str(path),
                "validation_sha256": sha256_file(path),
            }
        )
    stage_config = protocol["campaign"]["stages"].get(args.stage)
    if stage_config is not None:
        expected_seed_values = {
            int(value) for value in stage_config["seed_values"]
        }
        actual_seed_values = {int(row["seed"]) for row in rows}
        if actual_seed_values != expected_seed_values:
            raise ValueError(
                "validation evidence does not match the frozen stage seed set"
            )
        expected_algorithms = stage_config.get("algorithms")
        if expected_algorithms is not None and {
            str(row["algorithm"]) for row in rows
        } != {str(value) for value in expected_algorithms}:
            raise ValueError(
                "validation evidence does not match the frozen stage algorithms"
            )
    decision = promotion_decision(
        rows, expected_seed_count=args.expected_seeds
    )
    for algorithm in list(decision["promoted_algorithms"]):
        selected = [row for row in rows if row["algorithm"] == algorithm]
        if not all(row["success_gate_pass"] for row in selected):
            decision["promoted_algorithms"].remove(algorithm)
            decision["failed_gates"].setdefault(algorithm, []).append(
                "validation_success_gate"
            )
    decision["failure_is_publishable"] = not decision["promoted_algorithms"]
    aggregates = {}
    for algorithm in sorted({row["algorithm"] for row in rows}):
        selected = [row for row in rows if row["algorithm"] == algorithm]
        aggregates[algorithm] = {
            "seed_count": len(selected),
            "mean_incremental_ramp_impact": mean(
                row["mean_incremental_ramp_impact"] for row in selected
            ),
            "mean_energy_cost_ratio": mean(
                row["energy_cost_ratio"] for row in selected
            ),
            "all_safety_pass": all(row["safety_pass"] for row in selected),
            "all_success_gates_pass": all(
                row["success_gate_pass"] for row in selected
            ),
        }
    evidence = {
        "schema_version": "ramp-pure-rl-stage-summary-v1",
        "stage": args.stage,
        "selection_split": "validation",
        "sealed_test_used": False,
        "rows": rows,
        "aggregates": aggregates,
        "promotion_decision": decision,
    }
    if args.previous_summary is not None:
        previous = json.loads(args.previous_summary.read_text(encoding="utf-8"))
        algorithm = next(iter(aggregates))
        validation_curve = [
            {
                "stage": previous["stage"],
                "split": "validation",
                "mean_incremental_ramp_impact": previous["aggregates"][algorithm][
                    "mean_incremental_ramp_impact"
                ],
            },
            {
                "stage": args.stage,
                "split": "validation",
                "mean_incremental_ramp_impact": aggregates[algorithm][
                    "mean_incremental_ramp_impact"
                ],
            },
        ]
        evidence["validation_curve"] = validation_curve
        evidence["extension_decision"] = extension_allowed(
            protocol, validation_curve
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
