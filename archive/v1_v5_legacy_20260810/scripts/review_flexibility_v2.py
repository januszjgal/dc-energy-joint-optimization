"""Run the no-training v2 deadline-flexibility QP sensitivity."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.compute_qp_optimum import build_inputs, solve_clarabel  # noqa: E402
from scripts.review_energy_model_v2 import COMPLETION_PENALTY  # noqa: E402

YEAR = 2025
FLEXIBILITY_FACTORS = (0.0, 1.0, 2.0)
SCENARIOS = (
    ("US a-d", "us_model_v2_2025.yaml"),
    ("US e-h", "us_model_eh_v2_2025.yaml"),
    ("Global a-d", "global_model_v2_2025.yaml"),
    ("Global e-h", "global_model_eh_v2_2025.yaml"),
)


def build() -> dict[str, Any]:
    output = ROOT / "output" / "energy_model_v2" / str(YEAR)
    output.mkdir(parents=True, exist_ok=True)
    gate = json.loads(
        (output / "gate_summary.json").read_text(encoding="utf-8")
    )

    result: dict[str, Any] = {
        "year": YEAR,
        "primary_flexibility_factor": 1.0,
        "semantics": (
            "H = ceil(mean_duration * (1 + flexibility_factor) / "
            "interval_seconds); deadlines are experimental controls, not "
            "trace-provided SLOs"
        ),
        "scenarios": {},
    }
    rows = []

    for label, filename in SCENARIOS:
        scenario = ROOT / "env" / "scenarios" / filename
        spatial_optimum = gate["scenarios"][label]["spatial_qp"][
            "optimum_total"
        ]
        values = {}
        for factor in FLEXIBILITY_FACTORS:
            solved = solve_clarabel(
                build_inputs(
                    str(scenario),
                    True,
                    enforce_batch_completion=True,
                    completion_penalty_weight=COMPLETION_PENALTY,
                    flexibility_factor=factor,
                ),
                True,
            )
            temporal_headroom = (
                100.0
                * (spatial_optimum - solved["optimum_total"])
                / spatial_optimum
            )
            values[str(factor)] = {
                "deadline_horizon_multiplier": 1.0 + factor,
                "joint_qp": solved,
                "incremental_temporal_headroom_pct": temporal_headroom,
            }
            rows.append(
                (
                    label,
                    factor,
                    1.0 + factor,
                    solved["optimum_total"],
                    temporal_headroom,
                )
            )
        result["scenarios"][label] = values

    (output / "flexibility_sweep.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    report = [
        "# Energy Model v2 Deadline-Flexibility Sensitivity",
        "",
        "> No PPO training is involved. These are clairvoyant QP diagnostics.",
        "",
        "ClusterData 2019 does not publish workload deadlines. The model uses "
        "fitted mean job duration and an experimental flexibility factor:",
        "",
        "`H = ceil(mean_duration * (1 + flexibility_factor) / 300s)`",
        "",
        "The primary experiment freezes `flexibility_factor = 1` (`H = 2x "
        "mean duration`). Factors 0 and 2 provide tight (`H = 1x`) and loose "
        "(`H = 3x`) robustness cases.",
        "",
        "| Scenario | Factor | Horizon | Joint QP | Incremental temporal |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, factor, multiplier, optimum, temporal in rows:
        report.append(
            f"| {label} | {factor:.0f} | {multiplier:.0f}x mean duration | "
            f"${optimum / 1e6:.3f}M | {temporal:.2f}% |"
        )
    (output / "flexibility_sweep.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    build()
