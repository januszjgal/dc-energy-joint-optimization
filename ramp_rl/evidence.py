"""Attribution, statistical evidence, selection, and success-gate helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

import numpy as np

from ramp_rl.schema import FORBIDDEN_TRAINING_INPUTS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_pure_rl_manifest(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    valid_sha256 = lambda value: isinstance(value, str) and len(value) == 64 and all(  # noqa: E731
        character in "0123456789abcdef" for character in value
    )
    attribution = manifest.get("pure_rl_assertions", {})
    if attribution.get("random_initialization_only") is not True:
        errors.append("random initialization assertion missing")
    for key in FORBIDDEN_TRAINING_INPUTS:
        if attribution.get(key) is not False:
            errors.append(f"forbidden training input enabled: {key}")
    if manifest.get("semantic_feasible_action") is not True:
        errors.append("semantic feasible action assertion missing")
    if manifest.get("raw_redundant_projected_logits") is not False:
        errors.append("raw redundant logits were not rejected")
    if manifest.get("initial_policy_sha256") == manifest.get("final_policy_sha256"):
        errors.append("policy parameters did not update")
    if manifest.get("initial_critic_sha256") == manifest.get("final_critic_sha256"):
        errors.append("critic parameters did not update")
    for key in (
        "protocol_sha256",
        "source_bundle_sha256",
        "initial_policy_sha256",
        "initial_critic_sha256",
        "final_policy_sha256",
        "final_critic_sha256",
    ):
        if not valid_sha256(manifest.get(key)):
            errors.append(f"{key} is not a SHA-256 digest")
    if int(manifest.get("interaction_count", 0)) <= 0:
        errors.append("interaction count is not positive")
    if int(manifest.get("update_count", 0)) <= 0:
        errors.append("update count is not positive")
    if manifest.get("normalization", {}).get("fit_split") != "train":
        errors.append("normalization was not fit exclusively on train")
    provenance_rows = manifest.get("training_data_provenance", {}).get("episodes", [])
    if not provenance_rows:
        errors.append("per-window source/data provenance is empty")
    for row in provenance_rows:
        if row.get("split") != "train":
            errors.append("non-training split appears in training provenance")
        if not row.get("source_hashes"):
            errors.append("training window is missing source/data hashes")
        if not row.get("forecast_model") or not row.get("forecast_vintage"):
            errors.append("training window is missing forecast identity")
        if row.get("future_realized_features_exposed") is not False:
            errors.append("training window exposes realized future features")
    if manifest.get("algorithm") == "sac":
        provenance = manifest.get("replay_provenance", {})
        allowed = {"safe_random_feasible_warmup", "randomly_initialized_policy"}
        if set(provenance.get("sources", [])) - allowed:
            errors.append("SAC replay contains a prohibited source")
        if provenance.get("external_rows", -1) != 0:
            errors.append("SAC replay contains external rows")
    return errors


@dataclass
class FrozenLagrangian:
    """Predeclared scalar update; callers may feed train/validation data only."""

    multiplier: float
    learning_rate: float
    maximum: float

    def update(self, energy_cost: float, status_quo_cost: float, epsilon_pct: float, *, split: str) -> float:
        if split == "test":
            raise ValueError("sealed test data may not update the Lagrangian")
        budget = status_quo_cost * (1.0 + epsilon_pct / 100.0)
        self.multiplier = float(np.clip(self.multiplier + self.learning_rate * (energy_cost - budget), 0.0, self.maximum))
        return self.multiplier


def select_on_validation(candidates: Sequence[dict[str, Any]], *, split: str = "validation") -> dict[str, Any]:
    if split != "validation":
        raise ValueError("model selection is validation-only")
    eligible = [
        candidate
        for candidate in candidates
        if candidate["safety_pass"] and candidate["energy_budget_pass"]
    ]
    if not eligible:
        raise ValueError("no candidate satisfies safety and energy budget on validation")
    return min(eligible, key=lambda candidate: float(candidate["mean_incremental_ramp_impact"]))


def bootstrap_interval(
    values: Sequence[float],
    groups: Sequence[str],
    *,
    seed: int = 20260808,
    draws: int = 1000,
) -> dict[str, float]:
    if len(values) != len(groups) or not values:
        raise ValueError("bootstrap values/groups must be non-empty and aligned")
    grouped: dict[str, list[float]] = {}
    for value, group in zip(values, groups):
        grouped.setdefault(str(group), []).append(float(value))
    keys = sorted(grouped)
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        selected = rng.choice(keys, size=len(keys), replace=True)
        samples[index] = mean(value for key in selected for value in grouped[str(key)])
    return {
        "mean": float(mean(values)),
        "lower_95": float(np.quantile(samples, 0.025)),
        "upper_95": float(np.quantile(samples, 0.975)),
        "unit": "day_or_month",
        "draws": int(draws),
    }


def optimizer_seed_interval(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("optimizer seed interval needs values")
    return {
        "mean": float(array.mean()),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "seed_count": int(array.size),
    }


def record_pareto_sensitivities(
    candidates: Sequence[dict[str, Any]],
    *,
    split: str,
    epsilons: Sequence[float] = (0.0, 2.0, 5.0),
) -> list[dict[str, Any]]:
    if split not in {"validation", "test"}:
        raise ValueError("Pareto evidence is defined on validation or sealed test")
    if {float(value) for value in epsilons} != {0.0, 2.0, 5.0}:
        raise ValueError("predeclared epsilon sensitivities are 0, 2, and 5 percent")
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        status_quo_cost = float(candidate["status_quo_energy_cost"])
        energy_cost = float(candidate["energy_cost"])
        for epsilon in epsilons:
            rows.append(
                {
                    "candidate": str(candidate["name"]),
                    "split": split,
                    "epsilon_pct": float(epsilon),
                    "mean_incremental_ramp_impact": float(candidate["mean_incremental_ramp_impact"]),
                    "energy_cost_ratio": energy_cost / status_quo_cost,
                    "energy_budget_pass": energy_cost <= status_quo_cost * (1.0 + float(epsilon) / 100.0),
                    "safety_pass": bool(candidate["safety_pass"]),
                    "selected_on_test": False,
                }
            )
    return rows


def evaluate_success_gate(summary: dict[str, Any], *, split: str) -> dict[str, Any]:
    if split not in {"validation", "test"}:
        raise ValueError("success gates apply only to validation or sealed test")
    checks = {
        "exact_service": float(summary["service_unserved"]) == 0.0,
        "exact_batch_completion": float(summary["batch_unfinished"]) == 0.0,
        "zero_expiry": float(summary["batch_expired"]) == 0.0,
        "zero_terminal_work": float(summary["terminal_work"]) == 0.0,
        "zero_certificate_violations": int(summary["certificate_violations"]) == 0,
        "mean_ramp_improves": float(summary["mean_incremental_ramp_impact"]) < 0.0,
        "every_market_ramp_improves": all(float(value) < 0.0 for value in summary["per_market_macro"].values()),
        "primary_energy_budget": float(summary["energy_cost_ratio"]) <= 1.02,
        "emergency_path_below_one_percent": float(summary["emergency_feasibility_rate"]) < 0.01,
        "no_future_leakage": summary["future_leakage_detected"] is False,
        "behavior_pre_service": float(summary["behavior_audit"]["deferrable_pre_service"]) > 0.0,
        "behavior_lower_ramp_power": float(summary["behavior_audit"]["policy_ramp_power"]) < float(summary["behavior_audit"]["status_quo_ramp_power"]),
    }
    return {
        "split": split,
        "passed": all(checks.values()),
        "checks": checks,
        "failed_gates": [name for name, passed in checks.items() if not passed],
        "failure_is_publishable": True,
        "iterations_may_use": ["train", "validation"],
        "test_tuning_prohibited": True,
    }
