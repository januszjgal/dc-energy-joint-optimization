"""Hash-gated thesis view over sealed V4R evidence and post-hoc telemetry."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from ramp_rl.provenance import canonical_json_file_sha256
from ramp_rl.v4r_thesis import (
    EXPECTED_CANONICAL_SHA256 as EXPECTED_SEALED_CANONICAL_SHA256,
    EXPECTED_PROTOCOL_ID,
    EXPECTED_PROTOCOL_SHA256,
    EXPECTED_SOURCE_COMMIT,
    load_verified_evidence as load_verified_sealed_evidence,
)
from scripts.recompute_v4r_posthoc_metrics import verify as verify_posthoc


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANONICAL = (
    ROOT
    / "output"
    / "ramp_rl_v6"
    / "recovered_v4r_posthoc_metrics_v1"
    / "canonical_posthoc_metrics.json"
)
DEFAULT_OUTPUT = DEFAULT_CANONICAL.parent / "thesis"
EXPECTED_CANONICAL_SHA256 = (
    "bd1e8a242ec93e389c9e7b9e13ae7f19b60a9445716206b6527938ffe639f842"
)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def load_verified_evidence(
    path: Path = DEFAULT_CANONICAL,
) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"missing corrected canonical evidence: {path}")
    actual_hash = canonical_json_file_sha256(path)
    if actual_hash != EXPECTED_CANONICAL_SHA256:
        raise ValueError(
            "wrong corrected canonical evidence canonical-JSON SHA-256: "
            f"expected {EXPECTED_CANONICAL_SHA256}, got {actual_hash}"
        )
    wrapper = verify_posthoc(path)
    sealed = load_verified_sealed_evidence()
    corrected: dict[str, dict[str, Any]] = {}
    sealed_results: dict[str, dict[str, Any]] = {}
    for split in ("validation", "test"):
        reference = wrapper["corrected_results"][split]
        result_path = ROOT / str(reference["path"])
        payload = _load(result_path)
        result = payload.get("result")
        if not isinstance(result, dict):
            raise ValueError(f"{split} corrected result is missing")
        if result.get("split") != split:
            raise ValueError(f"{split} corrected result uses the wrong split")
        if result.get("success_gate", {}).get("passed") is not True:
            raise ValueError(f"{split} corrected result changed the gate outcome")
        sealed_results[split] = sealed[split]["result"]
        corrected[split] = {
            **sealed[split],
            "corrected_metrics_path": reference["path"],
            "corrected_metrics_sha256": reference["sha256"],
            "declaration": payload["declaration"],
            "invariant_audit": payload["invariant_audit"],
            "result": result,
        }
    evidence = {
        **sealed,
        "schema_version": wrapper["schema_version"],
        "status": wrapper["status"],
        "supersession_scope": wrapper["supersession_scope"],
        "validation": corrected["validation"],
        "test": corrected["test"],
        "sealed_results": sealed_results,
        "posthoc_correction": wrapper,
    }
    evidence["_verified"] = {
        **sealed["_verified"],
        "sealed_canonical_sha256": EXPECTED_SEALED_CANONICAL_SHA256,
        "canonical_sha256": EXPECTED_CANONICAL_SHA256,
        "corrected_canonical_path": path.relative_to(ROOT).as_posix(),
    }
    return evidence


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_publication_package(
    evidence: dict[str, Any],
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    validation = evidence["validation"]["result"]
    test = evidence["test"]["result"]
    _write_csv(
        output_dir / "split_summary.csv",
        [
            {
                "split": label,
                "episode_count": result["episode_count"],
                "mean_policy_native_relative_incremental_ramp_impact": result[
                    "mean_policy_native_relative_incremental_ramp_impact"
                ],
                "energy_cost_ratio": result["energy_cost_ratio"],
                "historical_gate_passed": result["success_gate"]["passed"],
            }
            for label, result in (
                ("validation", validation),
                ("sealed_test", test),
            )
        ],
    )
    comparison = test["status_quo_comparison"]
    _write_csv(
        output_dir / "sealed_test_market_comparison.csv",
        [
            {
                "market": market,
                **comparison["per_market"][market],
            }
            for market in sorted(comparison["per_market"])
        ],
    )
    _write_csv(
        output_dir / "physical_ramps.csv",
        [
            {
                "split": label,
                "abs_adjusted_h1_fraction_s_per_hour_p95": result[
                    "abs_adjusted_ramp_h1_fraction_s_per_hour_p95"
                ],
                "abs_adjusted_h1_fraction_s_per_hour_max": result[
                    "abs_adjusted_ramp_h1_fraction_s_per_hour_max"
                ],
                "abs_adjusted_h3_fraction_s_per_hour_p95": result[
                    "abs_adjusted_ramp_h3_fraction_s_per_hour_p95"
                ],
                "abs_adjusted_h3_fraction_s_per_hour_max": result[
                    "abs_adjusted_ramp_h3_fraction_s_per_hour_max"
                ],
                "pooling": "equal_weight_market_timestep_absolute_magnitudes",
            }
            for label, result in (
                ("validation", validation),
                ("sealed_test", test),
            )
        ],
    )
    _write_csv(
        output_dir / "decoder_adjustment.csv",
        [
            {
                "split": label,
                **result["semantic_adjustment"],
            }
            for label, result in (
                ("validation", validation),
                ("sealed_test", test),
            )
        ],
    )
    _write_csv(
        output_dir / "behavior.csv",
        [
            {
                "split": label,
                **result["behavior_audit"],
                "ramp_power_reduction_percent": 100
                * (
                    1
                    - result["behavior_audit"]["policy_ramp_power"]
                    / result["behavior_audit"]["status_quo_ramp_power"]
                ),
            }
            for label, result in (
                ("validation", validation),
                ("sealed_test", test),
            )
        ],
    )
    cost_rows = []
    for label, result in (
        ("validation", validation),
        ("sealed_test", test),
    ):
        policy_cost = sum(
            float(episode["energy_cost"])
            for episode in result["policy_episodes"]
        )
        status_cost = sum(
            float(episode["energy_cost"])
            for episode in result["status_quo_episodes"]
        )
        cost_rows.append(
            {
                "split": label,
                "status_quo_energy_cost_usd": status_cost,
                "policy_energy_cost_usd": policy_cost,
                "policy_saving_usd": status_cost - policy_cost,
                "policy_saving_percent": 100
                * (status_cost - policy_cost)
                / status_cost,
                "policy_saving_usd_per_episode": (
                    status_cost - policy_cost
                )
                / result["episode_count"],
            }
        )
    _write_csv(output_dir / "cost_comparison.csv", cost_rows)
    claim_ledger = {
        "schema_version": "v4r-corrected-thesis-claim-ledger-v1",
        "corrected_canonical_sha256": EXPECTED_CANONICAL_SHA256,
        "sealed_canonical_sha256": EXPECTED_SEALED_CANONICAL_SHA256,
        "protocol_id": EXPECTED_PROTOCOL_ID,
        "classification": (
            "post_hoc_frozen_policy_metric_recomputation"
        ),
        "not_a_second_sealed_generalization_test": True,
        "claims": {
            "test_primary_mean": test[
                "mean_incremental_ramp_impact"
            ],
            "test_day_bootstrap": test["bootstrap_by_day"],
            "test_status_quo_comparison": comparison,
            "test_physical_ramp_contract": test[
                "physical_ramp_metric_contract"
            ],
            "test_semantic_adjustment": test[
                "semantic_adjustment"
            ],
            "test_behavior": test["behavior_audit"],
            "test_cost": cost_rows[-1],
        },
    }
    ledger_path = output_dir / "claim_ledger.json"
    ledger_path.write_text(
        json.dumps(claim_ledger, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    manifest = {
        "schema_version": "v4r-corrected-thesis-package-v1",
        "corrected_canonical_sha256": EXPECTED_CANONICAL_SHA256,
        "sealed_canonical_sha256": EXPECTED_SEALED_CANONICAL_SHA256,
        "files": {
            path.name: _sha256(path)
            for path in sorted(output_dir.glob("*"))
            if path.is_file() and path.name != "package_manifest.json"
        },
    }
    manifest_path = output_dir / "package_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest
