"""Build and verify the append-only V4R provenance reseal without evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ramp_rl.ensemble import _SpaceOnlyEnv  # noqa: E402
from ramp_rl.evaluation import _aggregate  # noqa: E402
from ramp_rl.provenance import (  # noqa: E402
    CANONICAL_JSON_REPRESENTATION,
    GIT_BLOB_REPRESENTATION,
    HASH_CONTRACT_ID,
    RAW_BYTES_REPRESENTATION,
    canonical_json_bytes,
    canonical_json_file_sha256,
    canonical_json_sha256,
    git_blob_bytes,
    git_blob_oid,
    git_blob_reference,
    git_blob_sha256,
    json_file_reference,
    load_json,
    raw_file_reference,
    resolve_commit,
    sha256_bytes,
    sha256_file,
)
from ramp_rl.runner import model_hashes  # noqa: E402


ORIGINAL_ROOT = ROOT / "output" / "ramp_rl_v6" / "recovered_v4r"
RESEAL_ROOT = ROOT / "output" / "ramp_rl_v6" / "recovered_v4r_resealed_v2"
PROTOCOL_PATH = ROOT / "env" / "protocols" / "v6_pure_ramp_rl_v4r.yaml"
RECOVERY_COMMIT = "d1d4828c32bada1ba1e852d0479c37bb71164561"
SOURCE_COMMIT = "46329fe765f596f84eeac71f061dcbd191a90583"
BASE_COMMIT = "7ebd9b2"
PROTOCOL_ID = "v6-ramp-pure-rl-recovered-equal-action-ensemble-v4r"
ORIGINAL_PATHS = {
    "recovery_source": ORIGINAL_ROOT / "recovery_source_manifest.json",
    "recovery_binding": ORIGINAL_ROOT / "recovery_binding.json",
    "source_freeze": ORIGINAL_ROOT / "source_freeze.json",
    "validation_opening": ORIGINAL_ROOT / "validation_evaluation_opening.json",
    "validation_result": ORIGINAL_ROOT / "validation" / "ensemble_validation.json",
    "validation_decision": ORIGINAL_ROOT / "validation_decision.json",
    "sealed_test_opening": ORIGINAL_ROOT / "sealed_test_opening.json",
    "test_opening": ORIGINAL_ROOT / "sealed_test_evaluation_opening.json",
    "test_result": ORIGINAL_ROOT / "test" / "ensemble_test.json",
    "test_decision": ORIGINAL_ROOT / "test_decision.json",
    "one_gw_total": ORIGINAL_ROOT / "robustness" / "one_gw_total_test.json",
    "c_h_overlapping": ORIGINAL_ROOT / "robustness" / "c_h_overlapping_test.json",
    "canonical": ORIGINAL_ROOT / "canonical_evidence.json",
}
INTRODUCTION_COMMITS = {
    "recovery_source": SOURCE_COMMIT,
    "recovery_binding": SOURCE_COMMIT,
    "source_freeze": "fd15294",
    "validation_opening": "35969b7",
    "validation_result": "35969b7",
    "validation_decision": "35969b7",
    "sealed_test_opening": "9649674",
    "test_opening": "864ea82",
    "test_result": "864ea82",
    "test_decision": "864ea82",
    "one_gw_total": BASE_COMMIT,
    "c_h_overlapping": BASE_COMMIT,
    "canonical": BASE_COMMIT,
}
OUTPUT_FILES = {
    "hash_contract": RESEAL_ROOT / "hash_contract.json",
    "recovery_binding": RESEAL_ROOT / "recovery_binding.json",
    "source_freeze": RESEAL_ROOT / "source_freeze.json",
    "validation_chain": RESEAL_ROOT / "validation_chain.json",
    "sealed_test_chain": RESEAL_ROOT / "sealed_test_chain.json",
    "robustness_chain": RESEAL_ROOT / "robustness_chain.json",
    "migration_map": RESEAL_ROOT / "migration_map.json",
    "canonical": RESEAL_ROOT / "canonical_evidence.json",
}
AGGREGATE_KEYS = (
    "split",
    "episode_count",
    "service_unserved",
    "batch_unfinished",
    "batch_expired",
    "certificate_violations",
    "terminal_work",
    "mean_incremental_ramp_impact",
    "per_market_macro",
    "ramp_h1_adjusted_p95",
    "ramp_h1_adjusted_max",
    "ramp_h3_adjusted_p95",
    "ramp_h3_adjusted_max",
    "energy_cost_ratio",
    "emergency_feasibility_rate",
    "semantic_adjustment_l2",
    "future_leakage_detected",
    "behavior_audit",
    "bootstrap_by_month",
    "bootstrap_by_day",
    "evaluation_seed_interval",
    "policy_episodes",
    "status_quo_episodes",
    "success_gate",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _json_blob_reference(commit: str, path: str) -> dict[str, str]:
    content = git_blob_bytes(ROOT, commit, path)
    payload = json.loads(content)
    return {
        "algorithm": "sha256",
        "commit": resolve_commit(ROOT, commit),
        "git_blob_oid": git_blob_oid(ROOT, commit, path),
        "git_blob_sha256": sha256_bytes(content),
        "path": path,
        "representation": CANONICAL_JSON_REPRESENTATION,
        "sha256": canonical_json_sha256(payload),
    }


def _raw_git_blob_reference(commit: str, path: str) -> dict[str, str]:
    content = git_blob_bytes(ROOT, commit, path)
    return {
        "algorithm": "sha256",
        "commit": resolve_commit(ROOT, commit),
        "git_blob_oid": git_blob_oid(ROOT, commit, path),
        "path": path,
        "representation": RAW_BYTES_REPRESENTATION,
        "sha256": sha256_bytes(content),
    }


def _payload_reference(name: str, payload: Any) -> dict[str, str]:
    return {
        "algorithm": "sha256",
        "path": _relative(OUTPUT_FILES[name]),
        "representation": CANONICAL_JSON_REPRESENTATION,
        "sha256": canonical_json_sha256(payload),
    }


def _legacy_representations(commit: str, path: str) -> dict[str, str]:
    content = git_blob_bytes(ROOT, commit, path)
    normalized = content.replace(b"\r\n", b"\n")
    representations = {
        GIT_BLOB_REPRESENTATION: sha256_bytes(content),
        "legacy-working-tree-crlf": sha256_bytes(
            normalized.replace(b"\n", b"\r\n")
        ),
        "legacy-text-lf": sha256_bytes(normalized),
    }
    if path.endswith(".json"):
        representations[CANONICAL_JSON_REPRESENTATION] = canonical_json_sha256(
            json.loads(content)
        )
    return representations


def _classify_legacy_hash(commit: str, path: str, digest: str) -> list[str]:
    matches = [
        name
        for name, value in _legacy_representations(commit, path).items()
        if value == digest
    ]
    _require(matches, f"unclassified legacy digest for {commit}:{path}: {digest}")
    return matches


def _legacy_json_migration(
    *,
    name: str,
    legacy_fields: list[str],
    legacy_hashes: list[str],
) -> dict[str, Any]:
    path = ORIGINAL_PATHS[name]
    relative = _relative(path)
    commit = INTRODUCTION_COMMITS[name]
    return {
        "artifact": relative,
        "legacy_fields": legacy_fields,
        "legacy_hashes": [
            {
                "sha256": digest,
                "matching_representations": _classify_legacy_hash(
                    commit, relative, digest
                ),
            }
            for digest in sorted(set(legacy_hashes))
        ],
        "new_hash": json_file_reference(ROOT, path),
        "source_commit": resolve_commit(ROOT, commit),
    }


def _load_originals() -> dict[str, Any]:
    return {name: load_json(path) for name, path in ORIGINAL_PATHS.items()}


def _numeric_summary(artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        key: artifact[key]
        for key in AGGREGATE_KEYS
        if key not in {"policy_episodes", "status_quo_episodes"}
    }


def _hash_contract() -> dict[str, Any]:
    return {
        "schema_version": "dc-energy-provenance-hash-contract-v2",
        "contract_id": HASH_CONTRACT_ID,
        "algorithm": "sha256",
        "binary_and_data": {
            "representation": RAW_BYTES_REPRESENTATION,
            "normalization": "none",
        },
        "internal_model_state": {
            "representation": "ordered-tensor-state-bytes-v1",
            "implementation": "ramp_rl.runner.model_hashes",
        },
        "json_evidence": {
            "representation": CANONICAL_JSON_REPRESENTATION,
            "encoding": "UTF-8",
            "sort_keys": True,
            "separators": [",", ":"],
            "ensure_ascii": False,
            "allow_nan": False,
            "trailing_newline": False,
        },
        "tracked_non_json_text": {
            "representation": GIT_BLOB_REPRESENTATION,
            "commit_and_path_required": True,
            "working_tree_used": False,
        },
    }


def _source_freeze_payload(
    originals: dict[str, Any],
    protocol: dict[str, Any],
    recovery_binding: dict[str, Any],
) -> dict[str, Any]:
    legacy = originals["source_freeze"]
    source_files = {
        path: git_blob_reference(ROOT, SOURCE_COMMIT, path)
        for path in sorted(legacy["source_files"])
    }
    bundle_identity = {
        "source_commit": resolve_commit(ROOT, SOURCE_COMMIT),
        "source_files": {
            path: reference["sha256"] for path, reference in source_files.items()
        },
    }
    bindings = protocol["frozen_bindings"]
    factory_manifest = ROOT / bindings["factory"]["manifest_path"]
    inherited = {
        "preserved_protocol_bindings_sha256": canonical_json_sha256(bindings),
        "v3_protocol": git_blob_reference(
            ROOT, SOURCE_COMMIT, bindings["v3_protocol"]["path"]
        ),
        "v3_source_freeze": json_file_reference(
            ROOT, ROOT / bindings["v3_source_freeze"]["path"]
        ),
        "factory_source": git_blob_reference(
            ROOT, SOURCE_COMMIT, "env/ramp_v6/factory.py"
        ),
        "factory_manifest": json_file_reference(ROOT, factory_manifest),
        "factory_identity": {
            key: value
            for key, value in bindings["factory"].items()
            if key
            not in {
                "manifest_path",
                "manifest_sha256",
                "source_sha256",
            }
        },
    }
    robustness_inputs: dict[str, dict[str, str]] = {}
    for key, legacy_hash in protocol["post_selection"]["c_h_overlapping"][
        "input_bindings"
    ].items():
        if key.startswith("cell_"):
            cell = key.split("_")[1]
            relative = f"data/cells/cell_{cell}_tiers.csv"
            reference = _raw_git_blob_reference(SOURCE_COMMIT, relative)
        else:
            cell = key.rsplit("_", 1)[1]
            relative = f"data/jobs/batch_distributions_{cell}.json"
            reference = _json_blob_reference(SOURCE_COMMIT, relative)
        robustness_inputs[key] = {
            **reference,
            "legacy_declared_sha256": legacy_hash,
        }
    return {
        "schema_version": "ramp-pure-rl-recovered-v4r-source-freeze-v2",
        "status": "corrected-provenance-superseding-v1",
        "protocol_id": PROTOCOL_ID,
        "experimental_source_commit": resolve_commit(ROOT, SOURCE_COMMIT),
        "protocol": git_blob_reference(
            ROOT, SOURCE_COMMIT, _relative(PROTOCOL_PATH)
        ),
        "source_files": source_files,
        "source_bundle": {
            "algorithm": "sha256",
            "representation": CANONICAL_JSON_REPRESENTATION,
            "sha256": canonical_json_sha256(bundle_identity),
        },
        "recovery_binding": _payload_reference(
            "recovery_binding", recovery_binding
        ),
        "inherited_bindings": inherited,
        "robustness_inputs": robustness_inputs,
        "preserved_identity": {
            "blocked_original_v4_commit": legacy["blocked_original_v4_commit"],
            "blocked_original_v4_unchanged": True,
            "v3_predecessor_commit": legacy["v3_predecessor_commit"],
            "v3_artifacts_unchanged": True,
            "new_binary_identity": True,
            "test_opened_at_freeze": False,
        },
        "supersedes": json_file_reference(ROOT, ORIGINAL_PATHS["source_freeze"]),
        "results_changed": False,
    }


def _recovery_binding_payload(
    originals: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, Any]:
    evidence = protocol["recovery_evidence"]
    historical_manifest = _json_blob_reference(
        RECOVERY_COMMIT, evidence["recovery_manifest_git_path"]
    )
    copied_manifest = json_file_reference(
        ROOT, ORIGINAL_PATHS["recovery_source"]
    )
    _require(
        historical_manifest["sha256"] == copied_manifest["sha256"],
        "copied recovery manifest differs semantically from d1d4828",
    )
    members = []
    original_members = {
        int(row["seed"]): row for row in originals["recovery_binding"]["members"]
    }
    for member in protocol["frozen_bindings"]["members"]:
        seed = int(member["seed"])
        training_manifest = ROOT / member["original_training_manifest_path"]
        original = original_members[seed]
        members.append(
            {
                "seed": seed,
                "original_training_manifest": json_file_reference(
                    ROOT, training_manifest
                ),
                "original_model_sha256": member["original_model_sha256"],
                "recovered_model": {
                    "algorithm": "sha256",
                    "path": member["recovered_model_path"],
                    "representation": RAW_BYTES_REPRESENTATION,
                    "sha256": member["recovered_model_sha256"],
                },
                "recovered_vecnormalize": {
                    "algorithm": "sha256",
                    "path": member["recovered_vecnormalize_path"],
                    "representation": RAW_BYTES_REPRESENTATION,
                    "sha256": member["recovered_vecnormalize_sha256"],
                },
                "internal_state": {
                    "representation": "ordered-tensor-state-bytes-v1",
                    "policy_sha256": member["final_policy_sha256"],
                    "critic_sha256": member["final_critic_sha256"],
                    "loaded_policy_sha256": original["loaded_weights"]["policy"],
                    "loaded_critic_sha256": original["loaded_weights"]["critic"],
                },
                "interaction_count": member["interaction_count"],
                "update_count": member["update_count"],
                "weight_equivalent_to_v3": True,
                "normalization_byte_equivalent_to_v3": True,
                "new_model_container_identity": True,
            }
        )
    return {
        "schema_version": "ramp-pure-rl-recovered-v4r-binding-v2",
        "status": "corrected-provenance-superseding-v1",
        "protocol_id": PROTOCOL_ID,
        "recovery_commit": resolve_commit(ROOT, RECOVERY_COMMIT),
        "recovery_commit_is_ancestor": True,
        "recovery_manifest": historical_manifest,
        "copied_recovery_manifest": copied_manifest,
        "recovery_source": git_blob_reference(
            ROOT, RECOVERY_COMMIT, evidence["recovery_source_git_path"]
        ),
        "members": members,
        "allowed_container_divergences": originals["recovery_binding"][
            "allowed_container_divergences"
        ],
        "all_non_container_recovery_checks_exact": True,
        "classification": originals["recovery_binding"]["classification"],
        "test_opened": False,
        "supersedes": json_file_reference(
            ROOT, ORIGINAL_PATHS["recovery_binding"]
        ),
        "results_changed": False,
    }


def _chain_payloads(
    originals: dict[str, Any], source_freeze: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source_reference = _payload_reference("source_freeze", source_freeze)
    validation = {
        "schema_version": "ramp-pure-rl-recovered-v4r-validation-chain-v2",
        "status": "corrected-provenance-superseding-v1",
        "protocol_id": PROTOCOL_ID,
        "source_freeze": source_reference,
        "evaluation_opening": json_file_reference(
            ROOT, ORIGINAL_PATHS["validation_opening"]
        ),
        "result": json_file_reference(ROOT, ORIGINAL_PATHS["validation_result"]),
        "decision": json_file_reference(
            ROOT, ORIGINAL_PATHS["validation_decision"]
        ),
        "chronology": {
            "opened_at_utc": originals["validation_opening"]["opened_at_utc"],
            "decided_at_utc": originals["validation_decision"]["decided_at_utc"],
            "run_count": originals["validation_opening"]["run_count"],
            "test_opened": False,
            "test_open_count": 0,
        },
        "selected": originals["validation_decision"]["selected"],
        "strict_gate": originals["validation_decision"]["strict_gate"],
        "numeric_summary": _numeric_summary(originals["validation_result"]),
        "evaluation_rerun": False,
        "results_changed": False,
    }
    sealed_test = {
        "schema_version": "ramp-pure-rl-recovered-v4r-sealed-test-chain-v2",
        "status": "corrected-provenance-superseding-v1",
        "protocol_id": PROTOCOL_ID,
        "source_freeze": source_reference,
        "validation_chain": _payload_reference("validation_chain", validation),
        "sealed_test_opening": json_file_reference(
            ROOT, ORIGINAL_PATHS["sealed_test_opening"]
        ),
        "evaluation_opening": json_file_reference(
            ROOT, ORIGINAL_PATHS["test_opening"]
        ),
        "result": json_file_reference(ROOT, ORIGINAL_PATHS["test_result"]),
        "decision": json_file_reference(ROOT, ORIGINAL_PATHS["test_decision"]),
        "chronology": {
            "authorization_opened_at_utc": originals["sealed_test_opening"][
                "opened_at_utc"
            ],
            "evaluation_opened_at_utc": originals["test_opening"]["opened_at_utc"],
            "decided_at_utc": originals["test_decision"]["decided_at_utc"],
            "open_count": originals["sealed_test_opening"]["open_count"],
            "run_count": originals["test_opening"]["run_count"],
            "tuning_or_selection_permitted": False,
        },
        "sealed_test_passed": originals["test_decision"]["sealed_test_passed"],
        "strict_gate": originals["test_decision"]["strict_gate"],
        "numeric_summary": _numeric_summary(originals["test_result"]),
        "evaluation_rerun": False,
        "results_changed": False,
    }
    robustness = {
        "schema_version": "ramp-pure-rl-recovered-v4r-robustness-chain-v2",
        "status": "corrected-provenance-superseding-v1",
        "protocol_id": PROTOCOL_ID,
        "source_freeze": source_reference,
        "sealed_test_chain": _payload_reference(
            "sealed_test_chain", sealed_test
        ),
        "one_gw_total": {
            "artifact": json_file_reference(
                ROOT, ORIGINAL_PATHS["one_gw_total"]
            ),
            "numeric_summary": _numeric_summary(originals["one_gw_total"]),
        },
        "c_h_overlapping": {
            "artifact": json_file_reference(
                ROOT, ORIGINAL_PATHS["c_h_overlapping"]
            ),
            "numeric_summary": _numeric_summary(originals["c_h_overlapping"]),
        },
        "post_selection_only": True,
        "selection_or_tuning": False,
        "evaluation_rerun": False,
        "results_changed": False,
    }
    return validation, sealed_test, robustness


def _migration_map(
    originals: dict[str, Any],
    protocol: dict[str, Any],
    payloads: dict[str, Any],
) -> dict[str, Any]:
    validation = originals["validation_result"]
    test = originals["test_result"]
    robustness = originals["one_gw_total"]
    migrations = [
        _legacy_json_migration(
            name="recovery_source",
            legacy_fields=[
                "protocol.recovery_evidence.recovery_manifest_sha256",
                "recovery_binding.recovery_source_report_sha256",
                "source_freeze.recovery_source_report_sha256",
            ],
            legacy_hashes=[
                protocol["recovery_evidence"]["recovery_manifest_sha256"],
                originals["recovery_binding"]["recovery_source_report_sha256"],
                originals["source_freeze"]["recovery_source_report_sha256"],
            ],
        ),
        _legacy_json_migration(
            name="recovery_binding",
            legacy_fields=[
                "source_freeze.recovery_binding_sha256",
                "validation_result.recovery_binding_sha256",
                "validation_opening.recovery_binding_sha256",
                "test_result.recovery_binding_sha256",
                "test_opening.recovery_binding_sha256",
                "canonical_evidence.recovery_binding_sha256",
            ],
            legacy_hashes=[
                originals["source_freeze"]["recovery_binding_sha256"],
                validation["recovery_binding_sha256"],
                originals["validation_opening"]["recovery_binding_sha256"],
                test["recovery_binding_sha256"],
                originals["test_opening"]["recovery_binding_sha256"],
                originals["canonical"]["recovery_binding_sha256"],
            ],
        ),
        _legacy_json_migration(
            name="source_freeze",
            legacy_fields=[
                "validation_result.source_freeze_sha256",
                "test_result.source_freeze_sha256",
                "canonical_evidence.source_freeze_sha256",
            ],
            legacy_hashes=[
                validation["source_freeze_sha256"],
                test["source_freeze_sha256"],
                originals["canonical"]["source_freeze_sha256"],
            ],
        ),
        _legacy_json_migration(
            name="validation_opening",
            legacy_fields=["validation_result.evaluation_opening_sha256"],
            legacy_hashes=[validation["evaluation_opening_sha256"]],
        ),
        _legacy_json_migration(
            name="validation_result",
            legacy_fields=[
                "validation_decision.validation_sha256",
                "sealed_test_opening.validation_result_sha256",
                "test_result.validation_result_sha256",
                "canonical_evidence.validation.sha256",
            ],
            legacy_hashes=[
                originals["validation_decision"]["validation_sha256"],
                originals["sealed_test_opening"]["validation_result_sha256"],
                test["validation_result_sha256"],
                originals["canonical"]["validation"]["sha256"],
            ],
        ),
        _legacy_json_migration(
            name="validation_decision",
            legacy_fields=[
                "sealed_test_opening.validation_decision_sha256",
                "test_result.validation_decision_sha256",
                "canonical_evidence.validation.decision_sha256",
            ],
            legacy_hashes=[
                originals["sealed_test_opening"]["validation_decision_sha256"],
                test["validation_decision_sha256"],
                originals["canonical"]["validation"]["decision_sha256"],
            ],
        ),
        _legacy_json_migration(
            name="sealed_test_opening",
            legacy_fields=["test_result.sealed_test_opening_sha256"],
            legacy_hashes=[test["sealed_test_opening_sha256"]],
        ),
        _legacy_json_migration(
            name="test_opening",
            legacy_fields=["test_result.evaluation_opening_sha256"],
            legacy_hashes=[test["evaluation_opening_sha256"]],
        ),
        _legacy_json_migration(
            name="test_result",
            legacy_fields=[
                "test_decision.test_sha256",
                "robustness.*.test_result_sha256",
                "canonical_evidence.test.sha256",
            ],
            legacy_hashes=[
                originals["test_decision"]["test_sha256"],
                robustness["test_result_sha256"],
                originals["c_h_overlapping"]["test_result_sha256"],
                originals["canonical"]["test"]["sha256"],
            ],
        ),
        _legacy_json_migration(
            name="test_decision",
            legacy_fields=[
                "robustness.*.test_decision_sha256",
                "canonical_evidence.test.decision_sha256",
            ],
            legacy_hashes=[
                robustness["test_decision_sha256"],
                originals["c_h_overlapping"]["test_decision_sha256"],
                originals["canonical"]["test"]["decision_sha256"],
            ],
        ),
        _legacy_json_migration(
            name="one_gw_total",
            legacy_fields=["canonical_evidence.robustness.one_gw_total.sha256"],
            legacy_hashes=[
                originals["canonical"]["robustness"]["one_gw_total"]["sha256"]
            ],
        ),
        _legacy_json_migration(
            name="c_h_overlapping",
            legacy_fields=["canonical_evidence.robustness.c_h_overlapping.sha256"],
            legacy_hashes=[
                originals["canonical"]["robustness"]["c_h_overlapping"]["sha256"]
            ],
        ),
    ]
    source_migrations = []
    for path, digest in sorted(originals["source_freeze"]["source_files"].items()):
        reference = git_blob_reference(ROOT, SOURCE_COMMIT, path)
        source_migrations.append(
            {
                "artifact": path,
                "legacy_field": f"source_freeze.source_files.{path}",
                "legacy_sha256": digest,
                "legacy_matching_representations": _classify_legacy_hash(
                    SOURCE_COMMIT, path, digest
                ),
                "new_hash": reference,
            }
        )
    inherited_migrations = [
        {
            "artifact": _relative(PROTOCOL_PATH),
            "legacy_fields": ["*.protocol_sha256"],
            "legacy_sha256": originals["source_freeze"]["protocol_sha256"],
            "legacy_matching_representations": _classify_legacy_hash(
                SOURCE_COMMIT,
                _relative(PROTOCOL_PATH),
                originals["source_freeze"]["protocol_sha256"],
            ),
            "new_hash": git_blob_reference(
                ROOT, SOURCE_COMMIT, _relative(PROTOCOL_PATH)
            ),
        },
        {
            "artifact": protocol["frozen_bindings"]["v3_protocol"]["path"],
            "legacy_fields": ["protocol.frozen_bindings.v3_protocol.file_sha256"],
            "legacy_sha256": protocol["frozen_bindings"]["v3_protocol"][
                "file_sha256"
            ],
            "legacy_matching_representations": _classify_legacy_hash(
                SOURCE_COMMIT,
                protocol["frozen_bindings"]["v3_protocol"]["path"],
                protocol["frozen_bindings"]["v3_protocol"]["file_sha256"],
            ),
            "new_hash": git_blob_reference(
                ROOT,
                SOURCE_COMMIT,
                protocol["frozen_bindings"]["v3_protocol"]["path"],
            ),
        },
        {
            "artifact": protocol["frozen_bindings"]["v3_source_freeze"]["path"],
            "legacy_fields": ["protocol.frozen_bindings.v3_source_freeze.sha256"],
            "legacy_sha256": protocol["frozen_bindings"]["v3_source_freeze"][
                "sha256"
            ],
            "legacy_matching_representations": _classify_legacy_hash(
                SOURCE_COMMIT,
                protocol["frozen_bindings"]["v3_source_freeze"]["path"],
                protocol["frozen_bindings"]["v3_source_freeze"]["sha256"],
            ),
            "new_hash": _json_blob_reference(
                SOURCE_COMMIT,
                protocol["frozen_bindings"]["v3_source_freeze"]["path"],
            ),
        },
        {
            "artifact": protocol["frozen_bindings"]["factory"]["manifest_path"],
            "legacy_fields": [
                "protocol.frozen_bindings.factory.manifest_sha256",
                "*.factory_manifest_sha256",
            ],
            "legacy_sha256": protocol["frozen_bindings"]["factory"][
                "manifest_sha256"
            ],
            "legacy_matching_representations": _classify_legacy_hash(
                SOURCE_COMMIT,
                protocol["frozen_bindings"]["factory"]["manifest_path"],
                protocol["frozen_bindings"]["factory"]["manifest_sha256"],
            ),
            "new_hash": _json_blob_reference(
                SOURCE_COMMIT,
                protocol["frozen_bindings"]["factory"]["manifest_path"],
            ),
        },
    ]
    for member in protocol["frozen_bindings"]["members"]:
        path = member["original_training_manifest_path"]
        digest = member["original_training_manifest_sha256"]
        inherited_migrations.append(
            {
                "artifact": path,
                "legacy_fields": [
                    (
                        "protocol.frozen_bindings.members."
                        f"{member['seed']}.original_training_manifest_sha256"
                    )
                ],
                "legacy_sha256": digest,
                "legacy_matching_representations": _classify_legacy_hash(
                    SOURCE_COMMIT, path, digest
                ),
                "new_hash": _json_blob_reference(SOURCE_COMMIT, path),
            }
        )
    for key, legacy_hash in protocol["post_selection"]["c_h_overlapping"][
        "input_bindings"
    ].items():
        if key.startswith("cell_"):
            cell = key.split("_")[1]
            path = f"data/cells/cell_{cell}_tiers.csv"
            new_hash = _raw_git_blob_reference(SOURCE_COMMIT, path)
        else:
            cell = key.rsplit("_", 1)[1]
            path = f"data/jobs/batch_distributions_{cell}.json"
            new_hash = _json_blob_reference(SOURCE_COMMIT, path)
        inherited_migrations.append(
            {
                "artifact": path,
                "legacy_fields": [
                    f"protocol.post_selection.c_h_overlapping.input_bindings.{key}"
                ],
                "legacy_sha256": legacy_hash,
                "legacy_matching_representations": _classify_legacy_hash(
                    SOURCE_COMMIT, path, legacy_hash
                ),
                "new_hash": new_hash,
            }
        )
    binary_identities = []
    for member in protocol["frozen_bindings"]["members"]:
        binary_identities.extend(
            [
                {
                    "field": f"members.{member['seed']}.recovered_model_sha256",
                    "representation": RAW_BYTES_REPRESENTATION,
                    "sha256": member["recovered_model_sha256"],
                    "changed": False,
                },
                {
                    "field": (
                        f"members.{member['seed']}.recovered_vecnormalize_sha256"
                    ),
                    "representation": RAW_BYTES_REPRESENTATION,
                    "sha256": member["recovered_vecnormalize_sha256"],
                    "changed": False,
                },
                {
                    "field": f"members.{member['seed']}.final_policy_sha256",
                    "representation": "ordered-tensor-state-bytes-v1",
                    "sha256": member["final_policy_sha256"],
                    "changed": False,
                },
                {
                    "field": f"members.{member['seed']}.final_critic_sha256",
                    "representation": "ordered-tensor-state-bytes-v1",
                    "sha256": member["final_critic_sha256"],
                    "changed": False,
                },
            ]
        )
    return {
        "schema_version": "ramp-pure-rl-recovered-v4r-hash-migration-v2",
        "status": "complete",
        "contract_id": HASH_CONTRACT_ID,
        "legacy_problem": (
            "raw checkout bytes mixed LF Git blobs and CRLF Windows worktree bytes"
        ),
        "json_artifacts": migrations,
        "source_files": source_migrations,
        "inherited_protocol_data_and_factory_bindings": inherited_migrations,
        "legacy_source_bundle": {
            "legacy_fields": ["*.source_bundle_sha256"],
            "legacy_sha256": originals["source_freeze"]["source_bundle_sha256"],
            "new_hash": payloads["source_freeze"]["source_bundle"],
        },
        "legacy_terminal_evidence": {
            "artifact": _relative(ORIGINAL_PATHS["canonical"]),
            "legacy_git_blob_sha256": git_blob_sha256(
                ROOT, BASE_COMMIT, _relative(ORIGINAL_PATHS["canonical"])
            ),
            "legacy_windows_checkout_sha256": sha256_bytes(
                git_blob_bytes(
                    ROOT, BASE_COMMIT, _relative(ORIGINAL_PATHS["canonical"])
                ).replace(b"\n", b"\r\n")
            ),
            "new_target": _relative(OUTPUT_FILES["canonical"]),
        },
        "unchanged_binary_and_internal_identities": binary_identities,
        "new_chain": {
            name: _payload_reference(name, payloads[name])
            for name in (
                "recovery_binding",
                "source_freeze",
                "validation_chain",
                "sealed_test_chain",
                "robustness_chain",
            )
        },
        "experimental_results_changed": False,
    }


def build_payloads() -> tuple[dict[str, Any], str]:
    _require(
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", RECOVERY_COMMIT, "HEAD"],
            cwd=ROOT,
        ).returncode
        == 0,
        "historical recovery commit is not reachable from HEAD",
    )
    originals = _load_originals()
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    _require(protocol["protocol"]["id"] == PROTOCOL_ID, "protocol ID changed")
    payloads: dict[str, Any] = {"hash_contract": _hash_contract()}
    payloads["recovery_binding"] = _recovery_binding_payload(originals, protocol)
    payloads["source_freeze"] = _source_freeze_payload(
        originals, protocol, payloads["recovery_binding"]
    )
    (
        payloads["validation_chain"],
        payloads["sealed_test_chain"],
        payloads["robustness_chain"],
    ) = _chain_payloads(originals, payloads["source_freeze"])
    payloads["migration_map"] = _migration_map(originals, protocol, payloads)
    payloads["canonical"] = {
        "schema_version": "ramp-pure-rl-recovered-v4r-canonical-evidence-v2",
        "status": "canonical-corrected-provenance",
        "protocol_id": PROTOCOL_ID,
        "hash_contract": _payload_reference(
            "hash_contract", payloads["hash_contract"]
        ),
        "recovery_binding": _payload_reference(
            "recovery_binding", payloads["recovery_binding"]
        ),
        "source_freeze": _payload_reference(
            "source_freeze", payloads["source_freeze"]
        ),
        "validation_chain": _payload_reference(
            "validation_chain", payloads["validation_chain"]
        ),
        "sealed_test_chain": _payload_reference(
            "sealed_test_chain", payloads["sealed_test_chain"]
        ),
        "robustness_chain": _payload_reference(
            "robustness_chain", payloads["robustness_chain"]
        ),
        "migration_map": _payload_reference(
            "migration_map", payloads["migration_map"]
        ),
        "supersedes": json_file_reference(ROOT, ORIGINAL_PATHS["canonical"]),
        "supersession_scope": "provenance-hash-chain-only",
        "protocol_controller_model_data_factory_forecast_identities_changed": False,
        "training_or_retuning_performed": False,
        "validation_or_test_evaluation_rerun": False,
        "sealed_test_open_count": originals["sealed_test_opening"]["open_count"],
        "validation_passed": originals["validation_decision"]["selected"],
        "sealed_test_passed": originals["test_decision"]["sealed_test_passed"],
        "numerical_results_changed": False,
    }
    validation = originals["validation_result"]
    test = originals["test_result"]
    report = "\n".join(
        [
            "# Recovered-policy V4R corrected provenance",
            "",
            "Status: **canonical corrected provenance**.",
            "",
            (
                "This package supersedes only the checkout-byte hash chain under "
                "`output/ramp_rl_v6/recovered_v4r/`."
            ),
            "The protocol, controller, model, data, factory, forecast identities, chronology, and numerical results are unchanged.",
            "",
            f"- Hash contract: `{HASH_CONTRACT_ID}`",
            f"- Recovery commit: `{resolve_commit(ROOT, RECOVERY_COMMIT)}`",
            f"- Experimental source commit: `{resolve_commit(ROOT, SOURCE_COMMIT)}`",
            f"- Validation strict pass: **{originals['validation_decision']['selected']}**",
            f"- Validation mean incremental ramp impact: `{validation['mean_incremental_ramp_impact']!r}`",
            f"- Validation energy cost ratio: `{validation['energy_cost_ratio']!r}`",
            f"- Sealed test opened exactly once: **{originals['sealed_test_opening']['open_count'] == 1}**",
            f"- Sealed test strict pass: **{originals['test_decision']['sealed_test_passed']}**",
            f"- Test mean incremental ramp impact: `{test['mean_incremental_ramp_impact']!r}`",
            f"- Test energy cost ratio: `{test['energy_cost_ratio']!r}`",
            f"- One-GW robustness mean incremental ramp impact: `{originals['one_gw_total']['mean_incremental_ramp_impact']!r}`",
            f"- C-H robustness mean incremental ramp impact: `{originals['c_h_overlapping']['mean_incremental_ramp_impact']!r}`",
            "- Training, retraining, retuning, validation evaluation, and sealed-test evaluation performed by this reseal: **false**",
            "",
            "Verify with:",
            "",
            "```text",
            "python scripts/reseal_ramp_rl_v4r_provenance.py verify",
            "```",
            "",
        ]
    )
    return payloads, report


def write_package() -> None:
    payloads, report = build_payloads()
    RESEAL_ROOT.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        path = OUTPUT_FILES[name]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    (RESEAL_ROOT / "final_report.md").write_text(
        report, encoding="utf-8", newline="\n"
    )


def _assert_exact(value: Any, expected: Any, path: str = "") -> None:
    if isinstance(expected, float):
        _require(
            isinstance(value, (float, int))
            and math.isclose(float(value), expected, rel_tol=0.0, abs_tol=0.0),
            f"numeric mismatch at {path}: {value!r} != {expected!r}",
        )
        return
    if isinstance(expected, dict):
        _require(isinstance(value, dict), f"type mismatch at {path}")
        _require(set(value) == set(expected), f"keys mismatch at {path}")
        for key in expected:
            _assert_exact(value[key], expected[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        _require(isinstance(value, list), f"type mismatch at {path}")
        _require(len(value) == len(expected), f"length mismatch at {path}")
        for index, item in enumerate(expected):
            _assert_exact(value[index], item, f"{path}[{index}]")
        return
    _require(value == expected, f"value mismatch at {path}: {value!r} != {expected!r}")


def _verify_numerical_summaries(originals: dict[str, Any]) -> None:
    for name in (
        "validation_result",
        "test_result",
        "one_gw_total",
        "c_h_overlapping",
    ):
        artifact = originals[name]
        recomputed = _aggregate(
            artifact["policy_episodes"],
            artifact["status_quo_episodes"],
            artifact["split"],
        )
        for key in AGGREGATE_KEYS:
            _assert_exact(artifact[key], recomputed[key], f"{name}.{key}")


def _verify_original_canonical(originals: dict[str, Any]) -> None:
    canonical = originals["canonical"]
    _assert_exact(
        canonical["validation"]["result"],
        originals["validation_result"],
        "canonical.validation.result",
    )
    _assert_exact(
        canonical["test"]["result"],
        originals["test_result"],
        "canonical.test.result",
    )
    _assert_exact(
        canonical["robustness"]["one_gw_total"]["result"],
        originals["one_gw_total"],
        "canonical.robustness.one_gw_total.result",
    )
    _assert_exact(
        canonical["robustness"]["c_h_overlapping"]["result"],
        originals["c_h_overlapping"],
        "canonical.robustness.c_h_overlapping.result",
    )


def _verify_binaries(binary_root: Path) -> None:
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    for member in protocol["frozen_bindings"]["members"]:
        seed = int(member["seed"])
        member_root = binary_root / "members" / str(seed)
        model_path = member_root / "model.zip"
        normalization_path = member_root / "vecnormalize.pkl"
        _require(model_path.is_file(), f"missing local V4R model: {model_path}")
        _require(
            normalization_path.is_file(),
            f"missing local V4R normalizer: {normalization_path}",
        )
        _require(
            sha256_file(model_path) == member["recovered_model_sha256"],
            f"seed {seed} model raw hash mismatch",
        )
        _require(
            sha256_file(normalization_path)
            == member["recovered_vecnormalize_sha256"],
            f"seed {seed} normalizer raw hash mismatch",
        )
        manifest = load_json(ROOT / member["original_training_manifest_path"])
        model = PPO.load(model_path, device="cpu")
        hashes = model_hashes(model)
        _require(
            hashes["policy"] == member["final_policy_sha256"],
            f"seed {seed} internal policy hash mismatch",
        )
        _require(
            hashes["critic"] == member["final_critic_sha256"],
            f"seed {seed} internal critic hash mismatch",
        )
        holder = DummyVecEnv(
            [
                lambda model=model: _SpaceOnlyEnv(
                    observation_space=model.observation_space,
                    action_space=model.action_space,
                )
            ]
        )
        normalizer = VecNormalize.load(normalization_path, holder)
        try:
            _require(
                np.array_equal(
                    np.asarray(normalizer.obs_rms.mean),
                    np.asarray(manifest["normalization"]["observation_mean"]),
                ),
                f"seed {seed} normalizer mean mismatch",
            )
            _require(
                np.array_equal(
                    np.asarray(normalizer.obs_rms.var),
                    np.asarray(manifest["normalization"]["observation_variance"]),
                ),
                f"seed {seed} normalizer variance mismatch",
            )
            _require(
                float(normalizer.obs_rms.count)
                == float(manifest["normalization"]["sample_count"]),
                f"seed {seed} normalizer count mismatch",
            )
        finally:
            normalizer.close()


def verify_package(binary_root: Path, skip_binaries: bool) -> list[str]:
    expected_payloads, expected_report = build_payloads()
    for name, expected in expected_payloads.items():
        path = OUTPUT_FILES[name]
        _require(path.is_file(), f"missing resealed evidence: {path}")
        _assert_exact(load_json(path), expected, f"reseal.{name}")
        _require(
            canonical_json_file_sha256(path) == canonical_json_sha256(expected),
            f"canonical JSON hash mismatch: {path}",
        )
    report_path = RESEAL_ROOT / "final_report.md"
    _require(
        report_path.read_text(encoding="utf-8") == expected_report,
        "corrected report changed",
    )
    originals = _load_originals()
    _require(
        originals["sealed_test_opening"]["open_count"] == 1,
        "sealed test open count changed",
    )
    _require(
        originals["validation_opening"]["run_count"] == 1
        and originals["test_opening"]["run_count"] == 1,
        "evaluation run count changed",
    )
    _verify_original_canonical(originals)
    _verify_numerical_summaries(originals)
    if not skip_binaries:
        _verify_binaries(binary_root.resolve())
    return [
        f"hash_contract={HASH_CONTRACT_ID}",
        f"canonical_evidence={_relative(OUTPUT_FILES['canonical'])}",
        (
            "canonical_evidence_sha256="
            f"{canonical_json_file_sha256(OUTPUT_FILES['canonical'])}"
        ),
        f"recovery_commit={resolve_commit(ROOT, RECOVERY_COMMIT)}",
        f"recovery_commit_ancestor=true",
        f"source_commit={resolve_commit(ROOT, SOURCE_COMMIT)}",
        "sealed_test_open_count=1",
        "numerical_summaries_exact=true",
        f"local_binary_and_internal_hashes_verified={str(not skip_binaries).lower()}",
        "training_or_evaluation_executed=false",
        "status=PASS",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("build")
    verify = subparsers.add_parser("verify")
    verify.add_argument(
        "--binary-root",
        type=Path,
        default=ROOT / "models" / "ramp_rl_v6" / "recovered_v4r",
    )
    verify.add_argument("--skip-binaries", action="store_true")
    verify.add_argument("--log", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "build":
        write_package()
        print(_relative(RESEAL_ROOT))
        return 0
    lines = verify_package(args.binary_root, args.skip_binaries)
    output = "\n".join(lines) + "\n"
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(output, encoding="utf-8", newline="\n")
    print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
