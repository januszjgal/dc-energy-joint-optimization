"""Fail-closed aggregation and claim auditing for ramp-v6 thesis results."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402

from ramp_rl.evidence import verify_pure_rl_manifest
from ramp_rl.schema import validate_protocol

SCHEMA_VERSION = "ramp-v6-thesis-evidence-manifest-v1"
MARKET_RAMP_CONTROLLERS = {"native", "status_quo", "policy"}
REQUIRED_ANALYSIS_SECTIONS = {
    "market_ramps",
    "forecast_error_strata",
    "dc_ramp_behavior",
    "learning_curves",
    "sensitivities",
}
SHA_KEYS = (
    "source_panel_sha256",
    "raw_acquisition_sha256",
    "factory_input_hashes_sha256",
    "protocol_bundle_sha256",
    "source_bundle_sha256",
)
V1_CONFIRMATION_COMMIT = "3391440d168ade7127899a8a2c2d84e3a47a9ad3"
V1_CANONICAL_SHA256 = "ffe94a7c00d8d721940060d06d4aff0880b63ec70a7028b9464a8b188cf6e656"
V2_PROTOCOL_ID = "v6-ramp-pure-rl-preregistered-v2"
V2_PROTOCOL_FILE_SHA256 = "bffffe507d40c39813544055926922ced92b7e5b3f46a8b97fd0a5693ab74add"
V2_PROTOCOL_BUNDLE_SHA256 = "d7dab14572eb3c2465f7cba3a863d532c4cca97bbc81b8768529b69056acda5d"
V2_SOURCE_BUNDLE_SHA256 = "6c463cbb894d7a176b931992004d3224af6d5a89ab30701b9a2eac4f782a6a1f"


class EvidenceError(ValueError):
    """Raised when an evidence package cannot support fail-closed reporting."""


@dataclass(frozen=True)
class Artifact:
    label: str
    path: Path
    relative_path: str
    sha256: str
    reference: dict[str, Any]
    payload: dict[str, Any]


@dataclass(frozen=True)
class AuditedEvidence:
    artifact_root: Path
    manifest_path: Path
    manifest: dict[str, Any]
    protocol: Artifact
    factory: Artifact
    canonical: Artifact
    training: tuple[Artifact, ...]
    validation: tuple[Artifact, ...]
    analysis: Artifact | None
    opening_record: Artifact | None
    sealed_test: Artifact | None
    negative_protocol: Artifact


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def _require_keys(payload: dict[str, Any], keys: Iterable[str], context: str) -> None:
    missing = [key for key in keys if key not in payload]
    _require(not missing, f"{context} missing required fields: {', '.join(missing)}")


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _resolve_inside(root: Path, relative: str, label: str) -> Path:
    _require(relative and not Path(relative).is_absolute(), f"{label} path must be relative")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise EvidenceError(f"{label} path escapes artifact root") from error
    _require(path.is_file(), f"{label} artifact does not exist: {relative}")
    return path


def _artifact(root: Path, reference: dict[str, Any], label: str) -> Artifact:
    _require(isinstance(reference, dict), f"{label} reference is missing")
    _require_keys(reference, ("path", "sha256"), f"{label} reference")
    expected = reference["sha256"]
    _require(_valid_sha256(expected), f"{label} sha256 is invalid")
    path = _resolve_inside(root, str(reference["path"]), label)
    actual = sha256_file(path)
    _require(actual == expected, f"{label} hash mismatch: expected {expected}, found {actual}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise EvidenceError(f"{label} is not valid JSON") from error
    _require(isinstance(payload, dict), f"{label} JSON root must be an object")
    return Artifact(label, path, str(reference["path"]), actual, dict(reference), payload)


def _protocol_artifact(root: Path, reference: dict[str, Any]) -> Artifact:
    _require(isinstance(reference, dict), "protocol reference is missing")
    _require_keys(reference, ("path", "sha256"), "protocol reference")
    path = _resolve_inside(root, str(reference["path"]), "protocol")
    actual = sha256_file(path)
    _require(actual == reference["sha256"], "protocol artifact hash mismatch")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise EvidenceError("protocol is not valid YAML") from error
    _require(isinstance(payload, dict), "protocol YAML root must be an object")
    return Artifact(
        "protocol", path, str(reference["path"]), actual, dict(reference), payload
    )


def _split_months(payload: dict[str, Any]) -> dict[str, list[str]]:
    return {
        name: [str(value) for value in payload[name]]
        for name in ("train", "validation", "test")
    }


def _validate_splits(splits: dict[str, Any]) -> dict[str, list[str]]:
    _require_keys(splits, ("train", "validation", "test"), "split identities")
    normalized = _split_months(splits)
    for name, months in normalized.items():
        _require(months and len(months) == len(set(months)), f"{name} split is empty or duplicated")
    sets = [set(normalized[name]) for name in ("train", "validation", "test")]
    _require(
        not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]),
        "train, validation, and test split identities overlap",
    )
    return normalized


def _factory_splits(factory: dict[str, Any]) -> dict[str, list[str]]:
    return {
        name: [str(value) for value in factory["split_periods"][name]]
        for name in ("train", "validation", "test")
    }


def _training_splits(training: dict[str, Any]) -> dict[str, list[str]]:
    return {
        name: [str(value) for value in training["data_split"][name]["months"]]
        for name in ("train", "validation", "test")
    }


def _canonical_rows(canonical: dict[str, Any]) -> list[dict[str, Any]]:
    confirmation = canonical.get("confirmation", {})
    rows = confirmation.get("rows")
    _require(isinstance(rows, list) and rows, "canonical result has no confirmation rows")
    return rows


def _evaluation_seed(payload: dict[str, Any]) -> int:
    if "evaluation_seed" in payload:
        return int(payload["evaluation_seed"])
    episodes = payload.get("policy_episodes", [])
    _require(episodes and "evaluation_seed" in episodes[0], "evaluation seed is missing")
    seed = int(episodes[0]["evaluation_seed"])
    _require(
        all(int(row.get("evaluation_seed", seed)) == seed for row in episodes),
        "validation artifact mixes evaluation seeds",
    )
    return seed


def _validate_market_and_seed_sets(
    manifest: dict[str, Any],
    canonical: dict[str, Any],
    validations: tuple[Artifact, ...],
) -> None:
    seeds = [int(value) for value in manifest["selection"]["seeds"]]
    markets = [str(value) for value in manifest["selection"]["markets"]]
    _require(len(seeds) == len(set(seeds)) and seeds, "selected seed set is empty or duplicated")
    _require(len(markets) == len(set(markets)) and markets, "market set is empty or duplicated")
    rows = _canonical_rows(canonical)
    _require(
        len(rows) == len(seeds)
        and len({int(row["seed"]) for row in rows}) == len(rows),
        "canonical result must contain exactly one row per selected seed",
    )
    _require({int(row["seed"]) for row in rows} == set(seeds), "canonical seed set mismatch")
    _require(
        all(set(row["per_market_macro"]) == set(markets) for row in rows),
        "canonical market set mismatch",
    )
    _require(
        {_evaluation_seed(artifact.payload) for artifact in validations} == set(seeds),
        "validation evidence seed set mismatch",
    )
    for artifact in validations:
        payload = artifact.payload
        _require(payload.get("split") == "validation", "non-validation evidence used for promotion")
        _require(payload.get("future_leakage_detected") is False, "future leakage detected")
        _require(
            set(payload.get("per_market_macro", {})) == set(markets),
            f"{artifact.label} market set mismatch",
        )
        episodes = payload.get("policy_episodes", [])
        _require(episodes, f"{artifact.label} has no policy episodes")
        _require(
            all(row.get("future_realized_features_exposed") is False for row in episodes),
            f"{artifact.label} exposes realized future features",
        )


def _raw_gate_outcome(payload: dict[str, Any], context: str) -> dict[str, bool]:
    gate = payload.get("success_gate")
    _require(isinstance(gate, dict), f"{context} success gate is missing")
    checks = gate.get("checks")
    _require(isinstance(checks, dict), f"{context} success gate checks are missing")
    _require_keys(
        checks,
        (
            "exact_service",
            "exact_batch_completion",
            "zero_expiry",
            "zero_terminal_work",
            "zero_certificate_violations",
            "mean_ramp_improves",
            "every_market_ramp_improves",
            "primary_energy_budget",
            "emergency_path_below_one_percent",
            "no_future_leakage",
            "behavior_pre_service",
            "behavior_lower_ramp_power",
        ),
        f"{context} success gate checks",
    )
    behavior = payload.get("behavior_audit")
    _require(isinstance(behavior, dict), f"{context} behavior audit is missing")
    numeric_values = {
        "service_unserved": payload["service_unserved"],
        "batch_unfinished": payload["batch_unfinished"],
        "batch_expired": payload["batch_expired"],
        "terminal_work": payload["terminal_work"],
        "certificate_violations": payload["certificate_violations"],
        "mean_incremental_ramp_impact": payload["mean_incremental_ramp_impact"],
        "energy_cost_ratio": payload["energy_cost_ratio"],
        "emergency_feasibility_rate": payload["emergency_feasibility_rate"],
        "deferrable_pre_service": behavior["deferrable_pre_service"],
        "policy_ramp_power": behavior["policy_ramp_power"],
        "status_quo_ramp_power": behavior["status_quo_ramp_power"],
        **{
            f"per_market_macro.{market}": value
            for market, value in payload["per_market_macro"].items()
        },
    }
    _require(
        all(math.isfinite(float(value)) for value in numeric_values.values()),
        f"{context} contains non-finite telemetry",
    )
    expected_checks = {
        "exact_service": float(payload["service_unserved"]) == 0.0,
        "exact_batch_completion": float(payload["batch_unfinished"]) == 0.0,
        "zero_expiry": float(payload["batch_expired"]) == 0.0,
        "zero_terminal_work": float(payload["terminal_work"]) == 0.0,
        "zero_certificate_violations": int(payload["certificate_violations"]) == 0,
        "mean_ramp_improves": float(payload["mean_incremental_ramp_impact"]) < 0.0,
        "every_market_ramp_improves": all(
            float(value) < 0.0 for value in payload["per_market_macro"].values()
        ),
        "primary_energy_budget": float(payload["energy_cost_ratio"]) <= 1.02,
        "emergency_path_below_one_percent": (
            float(payload["emergency_feasibility_rate"]) < 0.01
        ),
        "no_future_leakage": payload["future_leakage_detected"] is False,
        "behavior_pre_service": float(behavior["deferrable_pre_service"]) > 0.0,
        "behavior_lower_ramp_power": (
            float(behavior["policy_ramp_power"])
            < float(behavior["status_quo_ramp_power"])
        ),
    }
    _require(
        all(bool(checks[key]) == value for key, value in expected_checks.items()),
        f"{context} success gate checks do not match raw telemetry",
    )
    _require(
        bool(gate["passed"]) == all(bool(value) for value in checks.values()),
        f"{context} success gate aggregate is internally inconsistent",
    )
    return {
        "safety": all(
            expected_checks[key]
            for key in (
                "exact_service",
                "exact_batch_completion",
                "zero_expiry",
                "zero_terminal_work",
                "zero_certificate_violations",
                "emergency_path_below_one_percent",
                "no_future_leakage",
            )
        ),
        "cost": expected_checks["primary_energy_budget"],
        "ramp": expected_checks["mean_ramp_improves"]
        and expected_checks["every_market_ramp_improves"],
        "behavior": expected_checks["behavior_pre_service"]
        and expected_checks["behavior_lower_ramp_power"],
        "passed": bool(gate["passed"]),
    }


def _validate_gates(
    manifest: dict[str, Any],
    canonical: dict[str, Any],
    validations: tuple[Artifact, ...],
) -> None:
    selection = manifest["selection"]
    _require_keys(
        selection,
        ("algorithm", "seeds", "markets", "split", "passed", "gate_assertions"),
        "selection",
    )
    _require(selection["split"] == "validation", "selection split must be validation")
    assertions = selection["gate_assertions"]
    _require_keys(
        assertions, ("safety", "cost", "ramp", "behavior"), "selection gate assertions"
    )
    rows = _canonical_rows(canonical)
    by_seed = {int(row["seed"]): row for row in rows}
    all_safety = True
    all_cost = True
    all_ramp = True
    all_behavior = True
    for artifact in validations:
        payload = artifact.payload
        seed = _evaluation_seed(payload)
        outcome = _raw_gate_outcome(payload, f"seed {seed}")
        row = by_seed[seed]
        _require(
            bool(row["safety_pass"]) == outcome["safety"],
            f"seed {seed} safety gate mismatch",
        )
        _require(
            bool(row["energy_budget_pass"]) == outcome["cost"],
            f"seed {seed} cost gate mismatch",
        )
        _require(
            float(row["mean_incremental_ramp_impact"])
            == float(payload["mean_incremental_ramp_impact"]),
            f"seed {seed} ramp metric mismatch",
        )
        _require(
            float(row["energy_cost_ratio"]) == float(payload["energy_cost_ratio"]),
            f"seed {seed} cost metric mismatch",
        )
        _require(
            {
                str(market): float(value)
                for market, value in row["per_market_macro"].items()
            }
            == {
                str(market): float(value)
                for market, value in payload["per_market_macro"].items()
            },
            f"seed {seed} per-market telemetry mismatch",
        )
        _require(
            {
                key: float(value)
                for key, value in row["behavior_audit"].items()
            }
            == {
                key: float(value)
                for key, value in payload["behavior_audit"].items()
            },
            f"seed {seed} behavior telemetry mismatch",
        )
        _require(
            bool(row["success_gate_pass"]) == outcome["passed"],
            f"seed {seed} aggregate gate mismatch",
        )
        all_safety &= outcome["safety"]
        all_cost &= outcome["cost"]
        all_ramp &= outcome["ramp"]
        all_behavior &= outcome["behavior"]
    observed = {
        "safety": all_safety,
        "cost": all_cost,
        "ramp": all_ramp,
        "behavior": all_behavior,
    }
    _require(
        {key: bool(assertions[key]) for key in observed} == observed,
        "selection gate assertions do not match evidence",
    )
    decision = canonical["confirmation"]["promotion_decision"]
    _require(
        decision.get("selection_split") == "validation"
        and decision.get("sealed_test_used") is False,
        "canonical promotion is not validation-only",
    )
    promoted = decision["promoted_algorithms"]
    passed = bool(selection["passed"])
    if manifest["result_status"] == "selected":
        _require(passed and all(observed.values()), "selected result failed a strict gate")
        _require(
            promoted == [selection["algorithm"]],
            "selected result was not the sole validation promotion",
        )
    else:
        _require(not passed, "failed-validation result cannot claim promotion")
        _require(not promoted, "failed-validation result contains a promoted algorithm")
        _require(
            not all(observed.values()),
            "failed-validation result has no failed strict gate",
        )


def _validate_evidence_bindings(
    manifest: dict[str, Any],
    canonical: Artifact,
    training: tuple[Artifact, ...],
    validation: tuple[Artifact, ...],
) -> None:
    algorithm = str(manifest["selection"]["algorithm"])
    training_by_seed = {int(item.payload["seed"]): item for item in training}
    canonical_by_seed = {
        int(row["seed"]): row for row in _canonical_rows(canonical.payload)
    }
    for artifact in validation:
        seed = _evaluation_seed(artifact.payload)
        reference = artifact.reference
        _require_keys(
            reference,
            (
                "seed",
                "algorithm",
                "training_manifest_sha256",
                "model_sha256",
                "final_policy_sha256",
            ),
            f"validation result {seed} model binding",
        )
        _require(int(reference["seed"]) == seed, f"validation result {seed} seed binding mismatch")
        _require(reference["algorithm"] == algorithm, f"validation result {seed} algorithm mismatch")
        training_artifact = training_by_seed[seed]
        training_payload = training_artifact.payload
        _require(
            training_payload.get("algorithm") == algorithm,
            f"training manifest {seed} algorithm mismatch",
        )
        _require(
            reference["training_manifest_sha256"] == training_artifact.sha256,
            f"validation result {seed} training-manifest binding mismatch",
        )
        _require(
            reference["model_sha256"]
            == training_payload["artifacts"]["model"]["sha256"],
            f"validation result {seed} model hash mismatch",
        )
        _require(
            reference["final_policy_sha256"] == training_payload["final_policy_sha256"],
            f"validation result {seed} policy hash mismatch",
        )
        if manifest["result_status"] == "selected":
            identity = artifact.payload.get("model_identity")
            _require(
                isinstance(identity, dict),
                f"selected validation result {seed} lacks embedded model identity",
            )
            expected_identity = {
                "algorithm": algorithm,
                "training_manifest_sha256": training_artifact.sha256,
                "model_sha256": training_payload["artifacts"]["model"]["sha256"],
                "final_policy_sha256": training_payload["final_policy_sha256"],
                "protocol_sha256": manifest["identities"]["protocol_bundle_sha256"],
                "source_bundle_sha256": manifest["identities"]["source_bundle_sha256"],
            }
            _require(
                all(identity.get(key) == value for key, value in expected_identity.items()),
                f"selected validation result {seed} embedded model identity mismatch",
            )
        row = canonical_by_seed[seed]
        _require(row.get("algorithm") == algorithm, f"canonical seed {seed} algorithm mismatch")
        _require(
            row.get("validation_sha256") == artifact.sha256,
            f"canonical seed {seed} validation hash mismatch",
        )
        _require(
            str(row.get("validation_artifact")) == artifact.relative_path,
            f"canonical seed {seed} validation path mismatch",
        )


def _validate_analysis(
    manifest: dict[str, Any],
    analysis: Artifact | None,
    status: str,
    canonical: Artifact,
) -> None:
    omissions = manifest.get("figure_omissions", {})
    _require(isinstance(omissions, dict), "figure_omissions must be an object")
    if status == "selected":
        _require(analysis is not None, "selected result is missing thesis analysis evidence")
        missing = REQUIRED_ANALYSIS_SECTIONS - set(analysis.payload)
        _require(not missing, f"selected result missing analysis sections: {sorted(missing)}")
        _require(not omissions, "selected result may not omit thesis figures")
    for section, reason in omissions.items():
        _require(section in REQUIRED_ANALYSIS_SECTIONS, f"unknown figure omission: {section}")
        _require(isinstance(reason, str) and reason.strip(), f"{section} omission needs a reason")
    if analysis is None:
        _require(
            set(omissions) == REQUIRED_ANALYSIS_SECTIONS,
            "missing analysis artifact requires explicit reasons for every analysis section",
        )
        return
    missing_sections = REQUIRED_ANALYSIS_SECTIONS - set(analysis.payload)
    _require(
        missing_sections <= set(omissions),
        "every missing analysis section requires an explicit omission reason",
    )
    identity = analysis.payload.get("identity")
    _require(isinstance(identity, dict), "analysis artifact identity is missing")
    _require_keys(
        identity,
        (
            "canonical_result_sha256",
            "protocol_id",
            "protocol_sha256",
            "source_bundle_sha256",
            "algorithm",
            "seeds",
            "validation_bindings",
        ),
        "analysis identity",
    )
    _require(
        identity["canonical_result_sha256"] == canonical.sha256
        and identity["protocol_id"] == manifest["protocol"]["id"]
        and identity["protocol_sha256"]
        == manifest["identities"]["protocol_bundle_sha256"]
        and identity["source_bundle_sha256"]
        == manifest["identities"]["source_bundle_sha256"]
        and identity["algorithm"] == manifest["selection"]["algorithm"],
        "analysis identity does not match the selected result",
    )
    _require(
        {int(value) for value in identity["seeds"]}
        == {int(value) for value in manifest["selection"]["seeds"]},
        "analysis identity seed set mismatch",
    )
    expected_bindings = {
        (
            int(reference["seed"]),
            reference["sha256"],
            reference["model_sha256"],
            reference["final_policy_sha256"],
        )
        for reference in manifest["validation_results"]
    }
    actual_bindings = {
        (
            int(row["seed"]),
            row["validation_sha256"],
            row["model_sha256"],
            row["final_policy_sha256"],
        )
        for row in identity["validation_bindings"]
    }
    _require(
        actual_bindings == expected_bindings,
        "analysis validation/model bindings mismatch",
    )
    for section in REQUIRED_ANALYSIS_SECTIONS:
        if section in analysis.payload:
            _require(analysis.payload[section], f"{section} analysis is empty")
            _require(
                all(row.get("split") == "validation" for row in analysis.payload[section]),
                f"{section} contains non-validation rows",
            )
            _require(
                all(
                    math.isfinite(float(value))
                    for row in analysis.payload[section]
                    for value in row.values()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                ),
                f"{section} contains non-finite telemetry",
            )
    overlap = set(analysis.payload) & set(omissions)
    _require(not overlap, f"analysis sections cannot also be omitted: {sorted(overlap)}")
    if "market_ramps" in analysis.payload:
        rows = analysis.payload["market_ramps"]
        observed = {(str(row["market"]), int(row["horizon_hours"]), str(row["controller"])) for row in rows}
        expected = {
            (market, horizon, controller)
            for market in manifest["selection"]["markets"]
            for horizon in (1, 3)
            for controller in MARKET_RAMP_CONTROLLERS
        }
        _require(
            observed == expected and len(rows) == len(expected),
            "market-ramp analysis is incomplete or duplicated",
        )
    if "learning_curves" in analysis.payload:
        _require(
            all(row.get("split") in {"train", "validation"} for row in analysis.payload["learning_curves"]),
            "learning curves contain sealed-test observations",
        )
        _require(
            {int(row["seed"]) for row in analysis.payload["learning_curves"]}
            == {int(value) for value in manifest["selection"]["seeds"]},
            "learning curves do not cover the selected seed set",
        )
        learning_keys = {
            (int(row["seed"]), int(row["timesteps"]), str(row["split"]))
            for row in analysis.payload["learning_curves"]
        }
        _require(
            len(learning_keys) == len(analysis.payload["learning_curves"]),
            "learning curves contain duplicate observations",
        )
    if "forecast_error_strata" in analysis.payload:
        _require(
            {str(row["market"]) for row in analysis.payload["forecast_error_strata"]}
            == set(manifest["selection"]["markets"]),
            "forecast-error strata do not cover the market set",
        )
        forecast_keys = {
            (str(row["market"]), str(row["stratum"]))
            for row in analysis.payload["forecast_error_strata"]
        }
        _require(
            len(forecast_keys) == len(analysis.payload["forecast_error_strata"]),
            "forecast-error strata contain duplicate observations",
        )
    if "dc_ramp_behavior" in analysis.payload:
        observed = {
            (str(row["market"]), str(row["phase"]), str(row["controller"]))
            for row in analysis.payload["dc_ramp_behavior"]
        }
        expected = {
            (market, phase, controller)
            for market in manifest["selection"]["markets"]
            for phase in ("before_ramp", "during_ramp")
            for controller in ("status_quo", "policy")
        }
        _require(
            observed == expected
            and len(analysis.payload["dc_ramp_behavior"]) == len(expected),
            "DC ramp behavior analysis is incomplete or duplicated",
        )
    if "sensitivities" in analysis.payload:
        phases = {str(row.get("phase")) for row in analysis.payload["sensitivities"]}
        _require(
            {"base_100mw", "post_selection"} <= phases,
            "sensitivity evidence needs 100MW base and post-selection phases",
        )
        sensitivity_keys = {
            (
                str(row["scenario"]),
                str(row["phase"]),
                float(row["scale_mw"]),
                float(row["workload_multiplier"]),
            )
            for row in analysis.payload["sensitivities"]
        }
        _require(
            len(sensitivity_keys) == len(analysis.payload["sensitivities"]),
            "sensitivities contain duplicate observations",
        )
        post_selection = [
            row
            for row in analysis.payload["sensitivities"]
            if row.get("phase") == "post_selection"
        ]
        _require(
            any(float(row["scale_mw"]) != 100.0 for row in post_selection),
            "post-selection scale sensitivity is missing",
        )
        _require(
            any(float(row["workload_multiplier"]) != 1.0 for row in post_selection),
            "post-selection workload sensitivity is missing",
        )


def _validate_selected_protocol(
    root: Path,
    manifest: dict[str, Any],
    protocol: Artifact,
    splits: dict[str, list[str]],
) -> None:
    _require(
        manifest["protocol"]["id"] == V2_PROTOCOL_ID,
        "selected result must use the frozen v2 protocol",
    )
    _require(
        protocol.sha256 == V2_PROTOCOL_FILE_SHA256,
        "selected result protocol file does not match the frozen v2 trust anchor",
    )
    validate_protocol(protocol.payload)
    environment = _resolve_inside(
        root,
        str(protocol.payload["environment_protocol"]["path"]),
        "selected environment protocol",
    )
    panel_schema = _resolve_inside(
        root,
        str(protocol.payload["environment_protocol"]["panel_schema"]),
        "selected panel schema",
    )
    component_hashes = {
        "campaign": _normalized_sha256(protocol.path),
        "environment": _normalized_sha256(environment),
        "panel_schema": _normalized_sha256(panel_schema),
    }
    bundle_sha256 = hashlib.sha256(
        json.dumps(component_hashes, sort_keys=True).encode("utf-8")
    ).hexdigest()
    _require(
        bundle_sha256 == V2_PROTOCOL_BUNDLE_SHA256
        and bundle_sha256 == manifest["identities"]["protocol_bundle_sha256"],
        "selected protocol bundle hash mismatch",
    )
    protocol_with_paths = dict(protocol.payload)
    protocol_with_paths["_path"] = str(protocol.path)
    protocol_with_paths["_environment_protocol_path"] = str(environment)
    protocol_with_paths["_panel_schema_path"] = str(panel_schema)
    from ramp_rl.runner import source_bundle_hash

    selected_source_bundle = source_bundle_hash(protocol_with_paths)
    _require(
        selected_source_bundle == V2_SOURCE_BUNDLE_SHA256
        and selected_source_bundle == manifest["identities"]["source_bundle_sha256"],
        "selected source bundle hash mismatch",
    )
    protocol_splits = {
        name: [
            str(value)
            for value in protocol.payload["data"]["split"][name]["months"]
        ]
        for name in ("train", "validation", "test")
    }
    _require(protocol_splits == splits, "selected splits do not match frozen v2")
    _require(
        manifest["selection"]["algorithm"]
        in protocol.payload["campaign"]["stages"]["confirmation"]["algorithms"],
        "selected algorithm is outside the frozen v2 confirmation stage",
    )
    _require(
        [int(value) for value in manifest["selection"]["seeds"]]
        == [
            int(value)
            for value in protocol.payload["campaign"]["stages"]["confirmation"][
                "seed_values"
            ]
        ],
        "selected seed set does not match frozen v2",
    )
def _validate_sealed_test(
    root: Path, manifest: dict[str, Any]
) -> tuple[Artifact | None, Artifact | None]:
    sealed = manifest["sealed_test"]
    _require_keys(
        sealed,
        ("opened", "opening_record", "result", "used_for_selection", "reason"),
        "sealed_test",
    )
    _require(sealed["used_for_selection"] is False, "sealed test was used for selection")
    opened = bool(sealed["opened"])
    if not opened:
        _require(sealed["opening_record"] is None, "unopened sealed test has an opening record")
        _require(sealed["result"] is None, "unopened sealed test has result evidence")
        _require(isinstance(sealed["reason"], str) and sealed["reason"], "unopened test needs a reason")
        return None, None
    _require(manifest["result_status"] == "selected", "sealed test opened before valid promotion")
    opening = _artifact(root, sealed["opening_record"], "sealed-test opening record")
    result = _artifact(root, sealed["result"], "sealed-test result")
    record = opening.payload
    _require_keys(
        record,
        (
            "opened",
            "opened_at_utc",
            "validation_finalized_at_utc",
            "authorized_after_validation",
            "selected_protocol_id",
            "validation_result_sha256",
            "test_result_sha256",
        ),
        "sealed-test opening record",
    )
    _require(record["opened"] is True, "sealed-test opening record is not affirmative")
    _require(
        record["authorized_after_validation"] is True,
        "sealed-test opening was not authorized after validation",
    )
    _require(
        record["selected_protocol_id"] == manifest["protocol"]["id"],
        "sealed-test opening protocol mismatch",
    )
    _require(
        record["validation_result_sha256"] == manifest["canonical_result"]["sha256"],
        "sealed-test opening validation hash mismatch",
    )
    _require(record["test_result_sha256"] == result.sha256, "sealed-test result hash mismatch")
    try:
        opened_at = datetime.fromisoformat(str(record["opened_at_utc"]).replace("Z", "+00:00"))
        finalized_at = datetime.fromisoformat(
            str(record["validation_finalized_at_utc"]).replace("Z", "+00:00")
        )
    except ValueError as error:
        raise EvidenceError("sealed-test opening chronology is invalid") from error
    _require(opened_at > finalized_at, "sealed test opened before validation finalized")
    _require(result.payload.get("split") == "test", "sealed-test result split is not test")
    _require(
        result.payload.get("used_for_selection") is False,
        "sealed-test result claims selection use",
    )
    _require_keys(
        result.payload,
        (
            "protocol_id",
            "protocol_sha256",
            "source_bundle_sha256",
            "algorithm",
            "seeds",
            "markets",
            "future_leakage_detected",
            "success_gate",
            "model_bindings",
            "rows",
        ),
        "sealed-test result",
    )
    _require(
        result.payload["protocol_id"] == manifest["protocol"]["id"]
        and result.payload["protocol_sha256"]
        == manifest["identities"]["protocol_bundle_sha256"]
        and result.payload["source_bundle_sha256"]
        == manifest["identities"]["source_bundle_sha256"],
        "sealed-test protocol/source identity mismatch",
    )
    _require(
        result.payload["algorithm"] == manifest["selection"]["algorithm"],
        "sealed-test algorithm mismatch",
    )
    _require(
        {int(value) for value in result.payload["seeds"]}
        == {int(value) for value in manifest["selection"]["seeds"]},
        "sealed-test seed set mismatch",
    )
    _require(
        set(result.payload["markets"]) == set(manifest["selection"]["markets"]),
        "sealed-test market set mismatch",
    )
    _require(
        result.payload["future_leakage_detected"] is False,
        "sealed-test result reports future leakage",
    )
    _require(
        isinstance(result.payload["success_gate"], dict)
        and "passed" in result.payload["success_gate"],
        "sealed-test success gate is missing",
    )
    test_rows = result.payload["rows"]
    _require(
        isinstance(test_rows, list)
        and {int(row["seed"]) for row in test_rows}
        == {int(value) for value in manifest["selection"]["seeds"]},
        "sealed-test result rows do not cover the seed set",
    )
    test_outcomes = []
    for row in test_rows:
        seed = int(row["seed"])
        _require(row.get("split") == "test", f"sealed-test seed {seed} split mismatch")
        _require(
            set(row.get("per_market_macro", {}))
            == set(manifest["selection"]["markets"]),
            f"sealed-test seed {seed} market set mismatch",
        )
        test_outcomes.append(_raw_gate_outcome(row, f"sealed-test seed {seed}"))
    _require(
        bool(result.payload["success_gate"]["passed"])
        == all(outcome["passed"] for outcome in test_outcomes),
        "sealed-test aggregate gate does not match seed rows",
    )
    expected_model_bindings = {
        (
            int(reference["seed"]),
            reference["training_manifest_sha256"],
            reference["model_sha256"],
            reference["final_policy_sha256"],
        )
        for reference in manifest["validation_results"]
    }
    actual_model_bindings = {
        (
            int(row["seed"]),
            row["training_manifest_sha256"],
            row["model_sha256"],
            row["final_policy_sha256"],
        )
        for row in result.payload["model_bindings"]
    }
    _require(
        actual_model_bindings == expected_model_bindings,
        "sealed-test model bindings do not match validation-selected models",
    )
    return opening, result


def audit_manifest(
    manifest_path: Path,
    artifact_root: Path | None = None,
) -> AuditedEvidence:
    manifest_path = manifest_path.resolve()
    root = (artifact_root or manifest_path.parent).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(isinstance(manifest, dict), "manifest root must be an object")
    _require_keys(
        manifest,
        (
            "schema_version",
            "label",
            "finality",
            "result_status",
            "protocol",
            "identities",
            "splits",
            "factory_manifest",
            "canonical_result",
            "training_manifests",
            "validation_results",
            "analysis_artifact",
            "selection",
            "sealed_test",
            "negative_protocol",
        ),
        "evidence manifest",
    )
    _require(manifest["schema_version"] == SCHEMA_VERSION, "unsupported evidence manifest schema")
    status = str(manifest["result_status"])
    _require(status in {"selected", "failed_validation"}, "invalid result_status")
    _require(
        manifest["finality"] == ("final" if status == "selected" else "non-final"),
        "result status and finality disagree",
    )
    identities = manifest["identities"]
    _require_keys(identities, SHA_KEYS + ("forecast_model_id",), "identities")
    for key in SHA_KEYS:
        _require(_valid_sha256(identities[key]), f"identities.{key} is not SHA-256")
    splits = _validate_splits(manifest["splits"])

    protocol = _protocol_artifact(root, manifest["protocol"])
    factory = _artifact(root, manifest["factory_manifest"], "factory manifest")
    canonical = _artifact(root, manifest["canonical_result"], "canonical result")
    training = tuple(
        _artifact(root, reference, f"training manifest {index}")
        for index, reference in enumerate(manifest["training_manifests"])
    )
    validation = tuple(
        _artifact(root, reference, f"validation result {index}")
        for index, reference in enumerate(manifest["validation_results"])
    )
    analysis = (
        None
        if manifest["analysis_artifact"] is None
        else _artifact(root, manifest["analysis_artifact"], "analysis artifact")
    )
    negative = _artifact(root, manifest["negative_protocol"], "v1 negative protocol")

    _require(
        protocol.payload["protocol"]["id"] == manifest["protocol"]["id"],
        "protocol identity mismatch",
    )
    if status == "selected":
        _validate_selected_protocol(root, manifest, protocol, splits)
    _require(
        factory.payload["source_panel_manifest_sha256"] == identities["source_panel_sha256"],
        "source panel hash mismatch",
    )
    _require(
        factory.payload["raw_acquisition_manifest_sha256"] == identities["raw_acquisition_sha256"],
        "raw acquisition hash mismatch",
    )
    _require(
        factory.payload["frozen_stats_sha256"] == identities["factory_input_hashes_sha256"],
        "factory input hash mismatch",
    )
    _require(
        factory.payload["forecast_model"] == identities["forecast_model_id"],
        "forecast model identity mismatch",
    )
    _require(_factory_splits(factory.payload) == splits, "factory split identity mismatch")
    _require(
        canonical.payload["frozen_inputs"]["train_months"] == splits["train"]
        and canonical.payload["frozen_inputs"]["validation_months"] == splits["validation"]
        and canonical.payload["frozen_inputs"]["sealed_test_months"] == splits["test"],
        "canonical split identity mismatch",
    )
    _require(
        canonical.payload["frozen_inputs"]["live_panel_manifest_sha256"]
        == identities["source_panel_sha256"]
        and canonical.payload["frozen_inputs"]["raw_acquisition_manifest_sha256"]
        == identities["raw_acquisition_sha256"],
        "canonical source/data hash mismatch",
    )

    seeds = {int(value) for value in manifest["selection"]["seeds"]}
    _require(len(training) == len(seeds), "training manifest count does not match seed set")
    _require({int(item.payload["seed"]) for item in training} == seeds, "training seed set mismatch")
    for artifact in training:
        payload = artifact.payload
        errors = verify_pure_rl_manifest(payload)
        _require(not errors, f"{artifact.label} pure-RL audit failed: {'; '.join(errors)}")
        _require(payload["protocol_id"] == manifest["protocol"]["id"], "training protocol ID mismatch")
        _require(
            payload["protocol_sha256"] == identities["protocol_bundle_sha256"],
            "training protocol bundle hash mismatch",
        )
        _require(
            payload["source_bundle_sha256"] == identities["source_bundle_sha256"],
            "training source bundle hash mismatch",
        )
        _require(_training_splits(payload) == splits, "training split identity mismatch")
        model = payload.get("artifacts", {}).get("model", {})
        _require(_valid_sha256(model.get("sha256")), "training model hash is missing")

    _validate_evidence_bindings(manifest, canonical, training, validation)
    _validate_market_and_seed_sets(manifest, canonical.payload, validation)
    _validate_gates(manifest, canonical.payload, validation)
    _validate_analysis(manifest, analysis, status, canonical)
    opening, sealed_result = _validate_sealed_test(root, manifest)
    _require(
        negative.payload.get("campaign_status") == "validation_blocker_no_selected_protocol",
        "v1 negative protocol artifact is not the failed canonical result",
    )
    _require(
        negative.sha256 == V1_CANONICAL_SHA256,
        "v1 negative protocol artifact hash does not match the canonical trust anchor",
    )
    _require(
        negative.reference.get("protocol_id") == "v6-ramp-pure-rl-preregistered-v1",
        "negative protocol reference is not v1",
    )
    _require(
        negative.reference.get("confirmation_commit") == V1_CONFIRMATION_COMMIT
        and negative.payload.get("commits", {}).get("confirmation")
        == V1_CONFIRMATION_COMMIT,
        "negative protocol confirmation commit mismatch",
    )
    negative_decision = negative.payload.get("confirmation", {}).get(
        "promotion_decision", {}
    )
    _require(
        negative_decision.get("selection_split") == "validation"
        and negative_decision.get("sealed_test_used") is False
        and not negative_decision.get("promoted_algorithms"),
        "v1 negative protocol is not validation-only failed evidence",
    )
    _require(
        negative.payload.get("sealed_test", {}).get("opened") is False,
        "v1 negative protocol improperly contains sealed-test evidence",
    )
    return AuditedEvidence(
        root,
        manifest_path,
        manifest,
        protocol,
        factory,
        canonical,
        training,
        validation,
        analysis,
        opening,
        sealed_result,
        negative,
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _save_unavailable(path: Path, title: str, reason: str) -> None:
    figure, axis = plt.subplots(figsize=(9, 4.8))
    axis.axis("off")
    axis.text(0.5, 0.58, title, ha="center", va="center", fontsize=16, weight="bold")
    axis.text(0.5, 0.38, f"Not available in this evidence package\n{reason}", ha="center", va="center", wrap=True)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _bar_figure(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    category: str,
    value: str,
    title: str,
    ylabel: str,
    group: str | None = None,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 5.5))
    categories = list(dict.fromkeys(str(row[category]) for row in rows))
    if group is None:
        values = [float(next(row[value] for row in rows if str(row[category]) == item)) for item in categories]
        axis.bar(categories, values, color="#3569b7")
    else:
        groups = list(dict.fromkeys(str(row[group]) for row in rows))
        width = 0.8 / max(len(groups), 1)
        for index, group_name in enumerate(groups):
            values = [
                float(
                    next(
                        (
                            row[value]
                            for row in rows
                            if str(row[category]) == item
                            and str(row[group]) == group_name
                        ),
                        math.nan,
                    )
                )
                for item in categories
            ]
            positions = [position + (index - (len(groups) - 1) / 2) * width for position in range(len(categories))]
            axis.bar(positions, values, width=width, label=group_name)
        axis.set_xticks(range(len(categories)), categories)
        axis.legend(frameon=False)
    axis.axhline(0.0, color="#333333", linewidth=0.8)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _derived_rows(evidence: AuditedEvidence) -> dict[str, list[dict[str, Any]]]:
    canonical_rows = _canonical_rows(evidence.canonical.payload)
    seed_market = [
        {
            "seed": int(row["seed"]),
            "market": market,
            "split": "validation",
            "mean_incremental_ramp_impact": float(value),
        }
        for row in canonical_rows
        for market, value in sorted(row["per_market_macro"].items())
    ]
    pareto = [
        {
            "seed": int(row["seed"]),
            "split": "validation",
            "energy_cost_ratio": float(row["energy_cost_ratio"]),
            "mean_incremental_ramp_impact": float(row["mean_incremental_ramp_impact"]),
            "safety_pass": bool(row["safety_pass"]),
            "cost_gate_pass": bool(row["energy_budget_pass"]),
            "ramp_gate_pass": all(float(value) < 0.0 for value in row["per_market_macro"].values()),
        }
        for row in canonical_rows
    ]
    behavior = [
        {
            "seed": int(row["seed"]),
            "split": "validation",
            "deferrable_pre_service": float(row["behavior_audit"]["deferrable_pre_service"]),
            "status_quo_ramp_power": float(row["behavior_audit"]["status_quo_ramp_power"]),
            "policy_ramp_power": float(row["behavior_audit"]["policy_ramp_power"]),
        }
        for row in canonical_rows
    ]
    safety = [
        {
            "seed": _evaluation_seed(artifact.payload),
            "split": "validation",
            "emergency_feasibility_rate": float(artifact.payload["emergency_feasibility_rate"]),
            "semantic_adjustment_l2": float(artifact.payload["semantic_adjustment_l2"]),
            "certificate_violations": int(artifact.payload["certificate_violations"]),
            "service_unserved": float(artifact.payload["service_unserved"]),
            "batch_unfinished": float(artifact.payload["batch_unfinished"]),
            "terminal_work": float(artifact.payload["terminal_work"]),
        }
        for artifact in evidence.validation
    ]
    negative_row = {
        "protocol": "v1",
        "evidence_split": "validation",
        "status": "failed_validation",
        "mean_incremental_ramp_impact": float(
            evidence.negative_protocol.payload["confirmation"]["aggregate"][
                "mean_incremental_ramp_impact"
            ]
        ),
        "energy_cost_ratio": float(
            evidence.negative_protocol.payload["confirmation"]["aggregate"][
                "mean_energy_cost_ratio"
            ]
        ),
        "sealed_test_opened": False,
    }
    current_row = {
        "protocol": evidence.manifest["protocol"]["id"],
        "evidence_split": "validation",
        "status": evidence.manifest["result_status"],
        "mean_incremental_ramp_impact": mean(
            float(row["mean_incremental_ramp_impact"]) for row in canonical_rows
        ),
        "energy_cost_ratio": mean(float(row["energy_cost_ratio"]) for row in canonical_rows),
        "sealed_test_opened": bool(evidence.manifest["sealed_test"]["opened"]),
    }
    protocol_comparison = (
        [negative_row]
        if evidence.negative_protocol.sha256 == evidence.canonical.sha256
        else [negative_row, current_row]
    )
    return {
        "seed_market_impact": seed_market,
        "cost_ramp_pareto": pareto,
        "aggregate_dc_behavior": behavior,
        "aggregate_safety_decoder": safety,
        "protocol_comparison": protocol_comparison,
    }


def build_claim_ledger(evidence: AuditedEvidence) -> dict[str, Any]:
    manifest = evidence.manifest
    status = manifest["result_status"]
    assertions = manifest["selection"]["gate_assertions"]
    analysis = evidence.analysis.payload if evidence.analysis is not None else {}

    def source(artifact: Artifact, fields: list[str]) -> dict[str, Any]:
        return {
            "artifact": artifact.relative_path,
            "sha256": artifact.sha256,
            "fields": fields,
        }

    claims = [
        {
            "claim_id": "pure-rl-attribution",
            "claim": "The evaluated policies were trained by pure RL under the frozen protocol.",
            "status": "confirmed",
            "evidence": [
                source(
                    artifact,
                    [
                        "/pure_rl_assertions",
                        "/protocol_id",
                        "/protocol_sha256",
                        "/source_bundle_sha256",
                        "/artifacts/model/sha256",
                    ],
                )
                for artifact in evidence.training
            ],
        },
        {
            "claim_id": "market-ramp-decomposition",
            "claim": "Policy 1h and 3h ramps are lower than native and status quo in every market.",
            "status": (
                "confirmed"
                if analysis.get("market_ramps")
                and all(
                    next(
                        float(row["mean_ramp"])
                        for row in analysis["market_ramps"]
                        if row["market"] == market
                        and int(row["horizon_hours"]) == horizon
                        and row["controller"] == "policy"
                    )
                    < min(
                        next(
                            float(row["mean_ramp"])
                            for row in analysis["market_ramps"]
                            if row["market"] == market
                            and int(row["horizon_hours"]) == horizon
                            and row["controller"] == baseline
                        )
                        for baseline in ("native", "status_quo")
                    )
                    for market in manifest["selection"]["markets"]
                    for horizon in (1, 3)
                )
                else "failed"
                if analysis.get("market_ramps")
                else "pending"
            ),
            "evidence": (
                [source(evidence.analysis, ["/market_ramps"])]
                if evidence.analysis is not None and analysis.get("market_ramps")
                else [source(evidence.canonical, ["/metric_scope"])]
            ),
        },
        {
            "claim_id": "forecast-error-robustness",
            "claim": "Ramp improvement remains negative across reported forecast-error strata.",
            "status": (
                "confirmed"
                if analysis.get("forecast_error_strata")
                and all(
                    float(row["mean_incremental_ramp_impact"]) < 0.0
                    for row in analysis["forecast_error_strata"]
                )
                else "failed"
                if analysis.get("forecast_error_strata")
                else "pending"
            ),
            "evidence": (
                [source(evidence.analysis, ["/forecast_error_strata"])]
                if evidence.analysis is not None and analysis.get("forecast_error_strata")
                else [
                    source(
                        evidence.factory,
                        ["/forecast_model", "/frozen_stats_sha256"],
                    )
                ]
            ),
        },
        {
            "claim_id": "dc-ramp-behavior",
            "claim": "The policy pre-services deferrable work and lowers DC power during ramps in every market.",
            "status": (
                "confirmed"
                if analysis.get("dc_ramp_behavior")
                and all(
                    float(row["behavior_audit"]["deferrable_pre_service"]) > 0.0
                    for row in _canonical_rows(evidence.canonical.payload)
                )
                and all(
                    next(
                        float(row["mean_dc_power_mw"])
                        for row in analysis["dc_ramp_behavior"]
                        if row["market"] == market
                        and row["phase"] == "during_ramp"
                        and row["controller"] == "policy"
                    )
                    < next(
                        float(row["mean_dc_power_mw"])
                        for row in analysis["dc_ramp_behavior"]
                        if row["market"] == market
                        and row["phase"] == "during_ramp"
                        and row["controller"] == "status_quo"
                    )
                    for market in manifest["selection"]["markets"]
                )
                else "failed"
                if analysis.get("dc_ramp_behavior")
                else "pending"
            ),
            "evidence": [
                source(evidence.canonical, ["/confirmation/rows/*/behavior_audit"]),
                *(
                    [source(evidence.analysis, ["/dc_ramp_behavior"])]
                    if evidence.analysis is not None and analysis.get("dc_ramp_behavior")
                    else []
                ),
            ],
        },
        {
            "claim_id": "learning-curve",
            "claim": "Each selected seed improves its reported validation ramp metric over training.",
            "status": (
                "confirmed"
                if analysis.get("learning_curves")
                and all(
                    sorted(
                        (
                            int(row["timesteps"]),
                            float(row["mean_incremental_ramp_impact"]),
                        )
                        for row in analysis["learning_curves"]
                        if int(row["seed"]) == seed
                    )[-1][1]
                    < sorted(
                        (
                            int(row["timesteps"]),
                            float(row["mean_incremental_ramp_impact"]),
                        )
                        for row in analysis["learning_curves"]
                        if int(row["seed"]) == seed
                    )[0][1]
                    for seed in manifest["selection"]["seeds"]
                )
                else "failed"
                if analysis.get("learning_curves")
                else "pending"
            ),
            "evidence": (
                [source(evidence.analysis, ["/learning_curves"])]
                if evidence.analysis is not None and analysis.get("learning_curves")
                else [
                    source(
                        evidence.canonical,
                        ["/confirmation/validation_curve"],
                    )
                ]
            ),
        },
        {
            "claim_id": "scale-workload-sensitivity",
            "claim": "Post-selection scale and workload sensitivities retain safety, cost, and ramp gates.",
            "status": (
                "confirmed"
                if analysis.get("sensitivities")
                and all(
                    bool(row["safety_pass"])
                    and float(row["energy_cost_ratio"]) <= 1.02
                    and float(row["mean_incremental_ramp_impact"]) < 0.0
                    for row in analysis["sensitivities"]
                    if row["phase"] == "post_selection"
                )
                else "failed"
                if analysis.get("sensitivities")
                else "pending"
            ),
            "evidence": (
                [source(evidence.analysis, ["/sensitivities"])]
                if evidence.analysis is not None and analysis.get("sensitivities")
                else [source(evidence.canonical, ["/robustness"])]
            ),
        },
        {
            "claim_id": "strict-validation-promotion",
            "claim": "The protocol passed all preregistered validation promotion gates.",
            "status": "confirmed" if status == "selected" else "failed",
            "evidence": [
                source(
                    evidence.canonical,
                    [
                        "/confirmation/promotion_decision",
                        "/confirmation/rows/*/safety_pass",
                        "/confirmation/rows/*/energy_budget_pass",
                        "/confirmation/rows/*/success_gate_pass",
                    ],
                )
            ],
        },
        {
            "claim_id": "validation-safety",
            "claim": "All selected-seed validation evaluations passed hard safety gates.",
            "status": "confirmed" if assertions["safety"] else "failed",
            "evidence": [
                source(
                    artifact,
                    [
                        "/success_gate/checks/exact_service",
                        "/success_gate/checks/exact_batch_completion",
                        "/success_gate/checks/zero_expiry",
                        "/success_gate/checks/zero_terminal_work",
                        "/success_gate/checks/zero_certificate_violations",
                    ],
                )
                for artifact in evidence.validation
            ],
        },
        {
            "claim_id": "validation-cost",
            "claim": "All selected-seed validation evaluations met the DA cost gate.",
            "status": "confirmed" if assertions["cost"] else "failed",
            "evidence": [
                source(artifact, ["/energy_cost_ratio", "/success_gate/checks/primary_energy_budget"])
                for artifact in evidence.validation
            ],
        },
        {
            "claim_id": "validation-ramp",
            "claim": "All selected-seed validation evaluations improved ramp impact in every market.",
            "status": "confirmed" if assertions["ramp"] else "failed",
            "evidence": [
                source(
                    artifact,
                    [
                        "/mean_incremental_ramp_impact",
                        "/per_market_macro",
                        "/success_gate/checks/every_market_ramp_improves",
                    ],
                )
                for artifact in evidence.validation
            ],
        },
        {
            "claim_id": "sealed-test-confirmation",
            "claim": "The validation-selected protocol was confirmed once on the sealed test.",
            "status": (
                (
                    "confirmed"
                    if evidence.sealed_test.payload["success_gate"]["passed"] is True
                    else "failed"
                )
                if evidence.sealed_test is not None
                else "failed"
                if status == "failed_validation"
                else "pending"
            ),
            "evidence": (
                [
                    source(
                        evidence.opening_record,
                        [
                            "/authorized_after_validation",
                            "/validation_result_sha256",
                            "/test_result_sha256",
                        ],
                    ),
                    source(evidence.sealed_test, ["/split", "/used_for_selection", "/success_gate"]),
                ]
                if evidence.sealed_test is not None and evidence.opening_record is not None
                else [
                    source(
                        evidence.canonical,
                        ["/sealed_test/opened", "/sealed_test/reason"],
                    )
                ]
            ),
        },
        {
            "claim_id": "v1-negative-protocol",
            "claim": "The v1 confirmation failed validation and did not open the sealed test.",
            "status": "confirmed",
            "evidence": [
                source(
                    evidence.negative_protocol,
                    [
                        "/campaign_status",
                        "/confirmation/promotion_decision/promoted_algorithms",
                        "/sealed_test/opened",
                    ],
                )
            ],
        },
    ]
    return {
        "schema_version": "ramp-v6-claim-ledger-v1",
        "result_label": manifest["label"],
        "result_status": status,
        "finality": manifest["finality"],
        "claims": claims,
    }


def _report(
    evidence: AuditedEvidence,
    derived: dict[str, list[dict[str, Any]]],
    ledger: dict[str, Any],
) -> str:
    manifest = evidence.manifest
    rows = derived["cost_ramp_pareto"]
    label = "FINAL" if manifest["finality"] == "final" else "NON-FINAL VALIDATION EVIDENCE"
    lines = [
        f"# Ramp-v6 thesis result package: {manifest['label']}",
        "",
        f"**{label}.** Result status: `{manifest['result_status']}`. Selection used validation only; "
        f"sealed test opened: **{str(manifest['sealed_test']['opened']).lower()}**.",
        "",
        "## Audited result",
        "",
        "| Seed | Validation ramp impact | DA cost ratio | Safety | Cost gate | Every-market ramp gate |",
        "|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['mean_incremental_ramp_impact']:.12g} | "
            f"{row['energy_cost_ratio']:.8f} | {row['safety_pass']} | "
            f"{row['cost_gate_pass']} | {row['ramp_gate_pass']} |"
        )
    lines.extend(
        [
            "",
            "## Evidence boundaries",
            "",
            f"- Protocol: `{manifest['protocol']['id']}` at `{evidence.protocol.sha256}`.",
            f"- Seeds: `{', '.join(str(value) for value in manifest['selection']['seeds'])}`.",
            f"- Markets: `{', '.join(manifest['selection']['markets'])}`.",
            "- Selection split: `validation`; test metrics are never mixed into validation tables.",
            f"- Sealed-test opening record: `{'present' if evidence.opening_record else 'not opened'}`.",
            "",
            "## Figure availability",
            "",
        ]
    )
    omissions = manifest.get("figure_omissions", {})
    for section in sorted(REQUIRED_ANALYSIS_SECTIONS):
        if section in omissions:
            lines.append(f"- `{section}`: unavailable - {omissions[section]}")
        else:
            lines.append(f"- `{section}`: available and hash-bound.")
    lines.append("- `safety_decoder`: available from hash-bound per-seed validation evidence.")
    lines.extend(
        [
            "",
            "## Claim audit",
            "",
            "| Claim | Status |",
            "|---|---|",
        ]
    )
    for claim in ledger["claims"]:
        lines.append(f"| {claim['claim']} | **{claim['status']}** |")
    lines.extend(
        [
            "",
            "The v1 row is retained as a negative validation protocol comparison. It is not "
            "pooled with a test result, and no sealed-test metric is inferred.",
            "",
        ]
    )
    return "\n".join(lines)


def generate_package(evidence: AuditedEvidence, output_dir: Path) -> dict[str, Any]:
    requested_output = output_dir.absolute()
    link_candidates = (
        requested_output,
        requested_output / "tables",
        requested_output / "figures",
    )
    _require(
        not any(
            path.exists()
            and (
                path.is_symlink()
                or bool(getattr(path, "is_junction", lambda: False)())
            )
            for path in link_candidates
        ),
        "output directory may not contain symlink or junction paths",
    )
    output_dir = requested_output.resolve()
    tables = output_dir / "tables"
    figures = output_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    generated_files = [
        output_dir / "claim_ledger.json",
        output_dir / "report.md",
        output_dir / "package_manifest.json",
        *[
            tables / f"{name}.csv"
            for name in (
                "seed_market_impact",
                "cost_ramp_pareto",
                "aggregate_dc_behavior",
                "aggregate_safety_decoder",
                "protocol_comparison",
                "market_ramps",
                "forecast_error_strata",
                "dc_ramp_behavior",
                "learning_curves",
                "safety_decoder",
                "sensitivities",
            )
        ],
        *[
            figures / f"{name}.png"
            for name in (
                "incremental_ramp_by_seed_market",
                "da_cost_vs_ramp_pareto",
                "market_ramps",
                "forecast_error_strata",
                "dc_ramp_behavior",
                "learning_curves",
                "safety_decoder",
                "sensitivities",
            )
        ],
    ]
    allowed_paths = {path.resolve() for path in generated_files}
    allowed_directories = {tables.resolve(), figures.resolve()}
    unexpected = [
        path
        for path in output_dir.rglob("*")
        if (
            (path.is_file() and path.resolve() not in allowed_paths)
            or (path.is_dir() and path.resolve() not in allowed_directories)
        )
    ]
    _require(
        not unexpected,
        "output directory contains unexpected stale paths: "
        + ", ".join(str(path.relative_to(output_dir)) for path in unexpected),
    )
    for path in generated_files:
        try:
            path.resolve().relative_to(output_dir)
        except ValueError as error:
            raise EvidenceError("generated output path escapes output directory") from error
        if path.is_file():
            path.unlink()
    derived = _derived_rows(evidence)
    analysis = evidence.analysis.payload if evidence.analysis is not None else {}
    omissions = evidence.manifest.get("figure_omissions", {})

    table_specs = {
        "seed_market_impact": ["seed", "market", "split", "mean_incremental_ramp_impact"],
        "cost_ramp_pareto": [
            "seed",
            "split",
            "energy_cost_ratio",
            "mean_incremental_ramp_impact",
            "safety_pass",
            "cost_gate_pass",
            "ramp_gate_pass",
        ],
        "aggregate_dc_behavior": [
            "seed",
            "split",
            "deferrable_pre_service",
            "status_quo_ramp_power",
            "policy_ramp_power",
        ],
        "aggregate_safety_decoder": [
            "seed",
            "split",
            "emergency_feasibility_rate",
            "semantic_adjustment_l2",
            "certificate_violations",
            "service_unserved",
            "batch_unfinished",
            "terminal_work",
        ],
        "protocol_comparison": [
            "protocol",
            "evidence_split",
            "status",
            "mean_incremental_ramp_impact",
            "energy_cost_ratio",
            "sealed_test_opened",
        ],
    }
    for name, fields in table_specs.items():
        _write_csv(tables / f"{name}.csv", derived[name], fields)

    analysis_fields = {
        "market_ramps": ["market", "horizon_hours", "controller", "mean_ramp", "p95_ramp", "max_ramp", "split"],
        "forecast_error_strata": ["market", "stratum", "mean_abs_error", "mean_incremental_ramp_impact", "sample_count", "split"],
        "dc_ramp_behavior": ["market", "phase", "controller", "mean_dc_power_mw", "split"],
        "learning_curves": ["seed", "timesteps", "split", "mean_incremental_ramp_impact"],
        "sensitivities": ["scenario", "phase", "scale_mw", "workload_multiplier", "mean_incremental_ramp_impact", "energy_cost_ratio", "safety_pass", "split"],
    }
    for section, fields in analysis_fields.items():
        _write_csv(tables / f"{section}.csv", list(analysis.get(section, [])), fields)

    _bar_figure(
        figures / "incremental_ramp_by_seed_market.png",
        derived["seed_market_impact"],
        category="market",
        value="mean_incremental_ramp_impact",
        group="seed",
        title="Validation incremental ramp impact by seed and market",
        ylabel="Incremental squared ramp impact",
    )
    figure, axis = plt.subplots(figsize=(8, 5.5))
    for row in derived["cost_ramp_pareto"]:
        color = "#25855a" if row["safety_pass"] else "#b63a3a"
        axis.scatter(row["energy_cost_ratio"], row["mean_incremental_ramp_impact"], color=color)
        axis.annotate(str(row["seed"]), (row["energy_cost_ratio"], row["mean_incremental_ramp_impact"]))
    axis.axvline(1.02, color="#b63a3a", linestyle="--", label="2% cost gate")
    axis.axhline(0.0, color="#333333", linewidth=0.8)
    axis.set_title("Validation DA cost versus ramp impact")
    axis.set_xlabel("DA energy cost ratio")
    axis.set_ylabel("Mean incremental ramp impact")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(figures / "da_cost_vs_ramp_pareto.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure_specs = {
        "market_ramps": ("Per-market 1h/3h native, status-quo, and policy ramps", "market", "mean_ramp", "controller"),
        "forecast_error_strata": ("Forecast-error strata", "stratum", "mean_incremental_ramp_impact", "market"),
        "dc_ramp_behavior": ("DC power before and during ramps", "market", "mean_dc_power_mw", "phase"),
        "learning_curves": ("Learning curves", "timesteps", "mean_incremental_ramp_impact", "seed"),
        "sensitivities": ("100MW base and post-selection sensitivities", "scenario", "mean_incremental_ramp_impact", "phase"),
    }
    for section, (title, category, value, group) in figure_specs.items():
        path = figures / f"{section}.png"
        rows = list(analysis.get(section, []))
        if not rows:
            _save_unavailable(path, title, omissions[section])
        else:
            category_key = category
            if section == "market_ramps":
                rows = [
                    {
                        **row,
                        "_market_horizon": (
                            f"{row['market']}\n{int(row['horizon_hours'])}h"
                        ),
                    }
                    for row in rows
                ]
                category_key = "_market_horizon"
            _bar_figure(
                path,
                rows,
                category=category_key,
                value=value,
                group=group,
                title=title,
                ylabel=value.replace("_", " "),
            )
    _bar_figure(
        figures / "safety_decoder.png",
        derived["aggregate_safety_decoder"],
        category="seed",
        value="semantic_adjustment_l2",
        title="Validation safety and decoder adjustments",
        ylabel="Semantic adjustment L2",
    )

    ledger = build_claim_ledger(evidence)
    ledger_path = output_dir / "claim_ledger.json"
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path = output_dir / "report.md"
    report_path.write_text(_report(evidence, derived, ledger), encoding="utf-8")
    inventory = {
        "schema_version": "ramp-v6-thesis-package-v1",
        "label": evidence.manifest["label"],
        "result_status": evidence.manifest["result_status"],
        "finality": evidence.manifest["finality"],
        "manifest": {
            "path": str(evidence.manifest_path.relative_to(evidence.artifact_root)),
            "sha256": sha256_file(evidence.manifest_path),
        },
        "report_sha256": sha256_file(report_path),
        "claim_ledger_sha256": sha256_file(ledger_path),
        "tables": {
            path.name: sha256_file(path)
            for path in generated_files
            if path.parent == tables and path.is_file()
        },
        "figures": {
            path.name: sha256_file(path)
            for path in generated_files
            if path.parent == figures and path.is_file()
        },
    }
    inventory_path = output_dir / "package_manifest.json"
    inventory_path.write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return inventory
