"""Predeclared staged job planning and validation-only promotion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class CampaignJob:
    stage: str
    algorithm: str
    seed: int
    timesteps: int
    epsilon_pct: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_stage(
    protocol: dict[str, Any],
    stage: str,
    *,
    algorithms: Sequence[str] | None = None,
    epsilon_pct: float | None = None,
) -> list[CampaignJob]:
    if stage not in {"screen", "confirmation", "extension"}:
        raise ValueError("stage must be screen, confirmation, or extension")
    config = protocol["campaign"]["stages"][stage]
    selected_algorithms = list(algorithms or config.get("algorithms", ("ppo", "sac")))
    if not selected_algorithms or set(selected_algorithms) - {"ppo", "sac"}:
        raise ValueError("campaign algorithms must be pure PPO and/or pure SAC")
    epsilon = (
        float(protocol["multiobjective"]["primary_energy_budget_pct"])
        if epsilon_pct is None
        else float(epsilon_pct)
    )
    if epsilon not in {
        float(value) for value in protocol["multiobjective"]["epsilon_sensitivity_pct"]
    }:
        raise ValueError("campaign epsilon is outside the preregistered sensitivities")
    timesteps = int(config.get("timesteps", config.get("max_timesteps")))
    seeds = [int(value) for value in config["seed_values"]]
    return [
        CampaignJob(
            stage=stage,
            algorithm=algorithm,
            seed=seed,
            timesteps=timesteps,
            epsilon_pct=epsilon,
        )
        for algorithm in selected_algorithms
        for seed in seeds
    ]


def promotion_decision(
    validation_rows: Sequence[dict[str, Any]],
    *,
    expected_seed_count: int,
) -> dict[str, Any]:
    if not validation_rows:
        raise ValueError("promotion requires validation evidence")
    if any(row.get("split") != "validation" for row in validation_rows):
        raise ValueError("promotion may not inspect train or sealed-test metrics")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in validation_rows:
        grouped.setdefault(str(row["algorithm"]), []).append(row)
    promoted: list[str] = []
    failures: dict[str, list[str]] = {}
    scores: dict[str, float] = {}
    for algorithm, rows in grouped.items():
        reasons: list[str] = []
        if len({int(row["seed"]) for row in rows}) != expected_seed_count:
            reasons.append("incomplete_optimizer_seed_set")
        if not all(bool(row["safety_pass"]) for row in rows):
            reasons.append("hard_workload_safety")
        if not all(bool(row["energy_budget_pass"]) for row in rows):
            reasons.append("primary_energy_budget")
        mean_impact = sum(float(row["mean_incremental_ramp_impact"]) for row in rows) / len(rows)
        scores[algorithm] = mean_impact
        if mean_impact >= 0.0:
            reasons.append("validation_ramp_improvement")
        if reasons:
            failures[algorithm] = reasons
        else:
            promoted.append(algorithm)
    return {
        "selection_split": "validation",
        "promoted_algorithms": sorted(promoted, key=scores.get),
        "mean_validation_incremental_ramp_impact": scores,
        "failed_gates": failures,
        "sealed_test_used": False,
        "failure_is_publishable": not promoted,
    }


def extension_allowed(
    protocol: dict[str, Any],
    validation_curve: Sequence[dict[str, float]],
) -> dict[str, Any]:
    if len(validation_curve) < 2:
        return {
            "allowed": False,
            "failed_gate": "insufficient_preregistered_validation_curve",
        }
    if any(point.get("split") != "validation" for point in validation_curve):
        raise ValueError("extension decisions are validation-only")
    previous = float(validation_curve[-2]["mean_incremental_ramp_impact"])
    current = float(validation_curve[-1]["mean_incremental_ramp_impact"])
    denominator = max(abs(previous), 1e-12)
    improvement_pct = 100.0 * (previous - current) / denominator
    threshold = float(
        protocol["campaign"]["stages"]["extension"][
            "preregistered_material_improvement_pct"
        ]
    )
    return {
        "allowed": improvement_pct >= threshold,
        "validation_improvement_pct": improvement_pct,
        "required_pct": threshold,
        "failed_gate": None if improvement_pct >= threshold else "validation_curve_not_material",
        "sealed_test_used": False,
    }


def assert_long_campaign_ready(integration: dict[str, bool]) -> None:
    required = ("integrated_v6_ramp_environment", "integrated_energy_model_v3")
    missing = [name for name in required if integration.get(name) is not True]
    if missing:
        raise RuntimeError("final long campaign remains blocked: " + ", ".join(missing))
