"""Bind deterministic V3 checkpoint reconstruction to V4R metric replay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from stable_baselines3 import PPO

from ramp_rl.evidence import verify_pure_rl_manifest
from ramp_rl.provenance import (
    CANONICAL_JSON_REPRESENTATION,
    RAW_BYTES_REPRESENTATION,
    canonical_json_file_sha256,
    canonical_json_sha256,
    git_blob_reference,
    historical_text_sha256_matches,
    json_file_reference,
    raw_file_reference,
    sha256_file,
)
from ramp_rl.runner import model_hashes
from ramp_rl.v4r_thesis import (
    DEFAULT_CANONICAL as SEALED_CANONICAL,
    EXPECTED_CANONICAL_SHA256 as SEALED_CANONICAL_SHA256,
)


ROOT = Path(__file__).resolve().parents[1]
STABLE_RECOVERY_COMMIT = "d1d4828c32bada1ba1e852d0479c37bb71164561"
PROTOCOL_PATH = (
    ROOT / "env" / "protocols" / "v6_pure_ramp_rl_v4r.yaml"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "output"
    / "ramp_rl_v6"
    / "recovered_v4r_posthoc_metrics_v1"
)
DEFAULT_RECOVERY_ROOT = (
    ROOT / "models" / "ramp_rl_v6" / "recovery_v3_metric_replay"
)
DEFAULT_RAW_VERIFICATION = (
    DEFAULT_OUTPUT_ROOT / "metric_replay_recovery_verification_raw.json"
)
DEFAULT_MANIFEST = (
    DEFAULT_OUTPUT_ROOT / "metric_replay_recovery_manifest.json"
)
COMPARISON_FIELDS = (
    "initial_policy_sha256",
    "initial_critic_sha256",
    "final_policy_sha256",
    "final_critic_sha256",
    "interaction_count",
    "interaction_count_this_invocation",
    "update_count",
    "actual_terminal_count_this_invocation",
    "requested_target_timesteps",
    "effective_boundary_target_timesteps",
    "checkpoint_boundary_quantum",
    "normalization",
    "pure_rl_assertions",
    "pure_rl_verification",
    "replay_provenance",
    "training_data_provenance",
    "forecast_identity",
    "job_identity",
)
STABLE_TOOL_PATHS = (
    "scripts/recover_ramp_rl_v3.py",
    "output/ramp_rl_v6/live_v3/recovery_transfer_manifest.json",
)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_raw_verification(
    raw: dict[str, Any],
) -> None:
    _require(
        raw.get("schema_version")
        == "ramp-pure-rl-v3-recovery-transfer-v1",
        "unsupported raw checkpoint recovery verification",
    )
    _require(
        raw.get("status") == "exact_match_not_published",
        "checkpoint recovery was not an exact unpublished reconstruction",
    )
    _require(raw.get("all_exact") is True, "checkpoint recovery diverged")
    _require(raw.get("published") is False, "checkpoint recovery was published")
    _require(raw.get("errors") == [], "checkpoint recovery contains errors")
    source = raw.get("source_identity", {})
    _require(
        source.get("source_commits") == ["f6cd9c9", "dede685"],
        "checkpoint recovery used the wrong source commits",
    )
    _require(
        source.get("training_evidence_commits")
        == ["fc8954b", "1f2a7dc"],
        "checkpoint recovery used the wrong training evidence",
    )
    _require(
        source.get("test_opened") is False,
        "checkpoint recovery opened validation or test",
    )
    contract = raw.get("equivalence_contract", {})
    expected_contract = {
        "model_container_sha256_required": False,
        "vecnormalize_sha256_required": True,
        "initial_policy_sha256_required": True,
        "initial_critic_sha256_required": True,
        "final_policy_sha256_required": True,
        "final_critic_sha256_required": True,
        "loaded_policy_sha256_required": True,
        "loaded_critic_sha256_required": True,
        "training_manifest_fields_required": list(COMPARISON_FIELDS),
    }
    _require(
        contract == expected_contract,
        "checkpoint recovery equivalence contract changed",
    )
    for label, record in raw.get("checks", {}).items():
        required = bool(record.get("required", True))
        if required:
            _require(
                record.get("matched") is True,
                f"required checkpoint recovery check failed: {label}",
            )
    seeds = raw.get("seeds")
    _require(
        isinstance(seeds, list)
        and [int(row.get("seed", -1)) for row in seeds]
        == [2801, 2802, 2803, 2804, 2805],
        "checkpoint recovery seed set changed",
    )


def build_metric_replay_recovery_manifest(
    *,
    raw_verification_path: Path,
    recovery_root: Path,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    raw_verification_path = raw_verification_path.resolve()
    recovery_root = recovery_root.resolve()
    output_root = output_root.resolve()
    _require(raw_verification_path.is_file(), "raw recovery verification is missing")
    try:
        relative_recovery_root = recovery_root.relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise ValueError("metric-replay recovery root must be inside the repository") from exc
    _require(
        relative_recovery_root
        == "models/ramp_rl_v6/recovery_v3_metric_replay",
        "metric-replay recovery used the wrong ignored root",
    )
    ignore_text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    _require(
        "models/ramp_rl_v6/recovery_v3_metric_replay/" in ignore_text,
        "metric-replay checkpoint root is not ignored",
    )
    raw = _load(raw_verification_path)
    _validate_raw_verification(raw)
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    bindings = {
        int(binding["seed"]): binding
        for binding in protocol["frozen_bindings"]["members"]
    }
    raw_seeds = {int(row["seed"]): row for row in raw["seeds"]}
    members: list[dict[str, Any]] = []
    for seed in sorted(bindings):
        binding = bindings[seed]
        raw_seed = raw_seeds[seed]
        member_root = recovery_root / str(seed)
        model_path = member_root / "model.zip"
        normalization_path = member_root / "vecnormalize.pkl"
        recovered_manifest_path = member_root / "training_manifest.json"
        original_manifest_path = ROOT / binding["original_training_manifest_path"]
        for path in (
            model_path,
            normalization_path,
            recovered_manifest_path,
            original_manifest_path,
        ):
            _require(path.is_file(), f"missing recovery artifact: {path}")
        original = _load(original_manifest_path)
        recovered = _load(recovered_manifest_path)
        pure_rl_errors = verify_pure_rl_manifest(recovered)
        _require(
            pure_rl_errors == [],
            f"seed {seed} reconstructed manifest is not pure RL: {pure_rl_errors}",
        )
        comparison = {
            field: recovered[field]
            for field in COMPARISON_FIELDS
        }
        _require(
            all(recovered[field] == original[field] for field in COMPARISON_FIELDS),
            f"seed {seed} training provenance differs from frozen V3",
        )
        _require(
            historical_text_sha256_matches(
                original_manifest_path,
                binding["original_training_manifest_sha256"],
            ),
            f"seed {seed} original training manifest identity changed",
        )
        model_sha256 = sha256_file(model_path)
        normalization_sha256 = sha256_file(normalization_path)
        recovered_manifest_sha256 = sha256_file(recovered_manifest_path)
        raw_artifacts = raw_seed["artifacts"]
        _require(
            model_sha256
            == raw_artifacts["model.zip"]["reproduced_sha256"],
            f"seed {seed} model differs from the full recovery verifier",
        )
        _require(
            normalization_sha256
            == raw_artifacts["vecnormalize.pkl"]["reproduced_sha256"]
            == binding["recovered_vecnormalize_sha256"],
            f"seed {seed} normalizer is not byte-identical to frozen V3",
        )
        _require(
            recovered_manifest_sha256
            == raw_seed["recovered_manifest_sha256"],
            f"seed {seed} reconstructed manifest differs from verifier input",
        )
        _require(
            historical_text_sha256_matches(
                original_manifest_path,
                raw_seed["original_manifest_sha256"],
            ),
            f"seed {seed} raw verifier used a different original manifest",
        )
        loaded = model_hashes(PPO.load(model_path, device="cpu"))
        _require(
            loaded["policy"] == binding["final_policy_sha256"],
            f"seed {seed} loaded policy differs from frozen V3",
        )
        _require(
            loaded["critic"] == binding["final_critic_sha256"],
            f"seed {seed} loaded critic differs from frozen V3",
        )
        observed_splits = sorted(
            {
                str(row["split"])
                for row in recovered["training_data_provenance"]["episodes"]
            }
        )
        _require(
            observed_splits == ["train"],
            f"seed {seed} reconstruction observed non-training data",
        )
        members.append(
            {
                "seed": seed,
                "checkpoint_root": member_root.relative_to(ROOT).as_posix(),
                "binaries_tracked": False,
                "binaries_ignored": True,
                "model_container": {
                    **raw_file_reference(ROOT, model_path),
                    "original_v4r_container_sha256": binding[
                        "recovered_model_sha256"
                    ],
                    "byte_identical_to_original_v4r_container": (
                        model_sha256 == binding["recovered_model_sha256"]
                    ),
                    "container_difference_scope": (
                        "outer SB3 container byte identity is not claimed; "
                        "exact policy and critic tensors are verified"
                    ),
                },
                "vecnormalize": {
                    **raw_file_reference(ROOT, normalization_path),
                    "byte_identical_to_frozen_v3": True,
                },
                "reconstructed_training_manifest": {
                    **raw_file_reference(ROOT, recovered_manifest_path),
                    "canonical_json_sha256": canonical_json_file_sha256(
                        recovered_manifest_path
                    ),
                },
                "original_training_manifest": {
                    "path": binding["original_training_manifest_path"],
                    "historical_raw_sha256": binding[
                        "original_training_manifest_sha256"
                    ],
                    "canonical_json_sha256": canonical_json_file_sha256(
                        original_manifest_path
                    ),
                    "unchanged": True,
                },
                "initial_policy_sha256": recovered["initial_policy_sha256"],
                "initial_critic_sha256": recovered["initial_critic_sha256"],
                "final_policy_sha256": loaded["policy"],
                "final_critic_sha256": loaded["critic"],
                "interaction_count": recovered["interaction_count"],
                "update_count": recovered["update_count"],
                "comparison_fields": list(COMPARISON_FIELDS),
                "comparison_fields_exact": True,
                "comparison_fields_canonical_sha256": canonical_json_sha256(
                    comparison
                ),
                "source_bundle_sha256": recovered["source_bundle_sha256"],
                "protocol_sha256": recovered["protocol_sha256"],
                "job_identity": recovered["job_identity"],
                "training_data_splits_observed": observed_splits,
                "validation_or_test_rows": recovered[
                    "training_data_provenance"
                ]["validation_or_test_rows"],
            }
        )
    data_access_audit = {
        "raw_source_test_opened": raw["source_identity"]["test_opened"],
        "member_training_splits": {
            str(member["seed"]): member["training_data_splits_observed"]
            for member in members
        },
        "member_validation_or_test_rows": {
            str(member["seed"]): member["validation_or_test_rows"]
            for member in members
        },
    }
    validation_or_test_opened = not (
        data_access_audit["raw_source_test_opened"] is False
        and all(
            splits == ["train"]
            for splits in data_access_audit["member_training_splits"].values()
        )
        and all(
            rows == 0
            for rows in data_access_audit[
                "member_validation_or_test_rows"
            ].values()
        )
    )
    _require(
        validation_or_test_opened is False,
        "checkpoint recovery observed validation or test data",
    )
    raw_output = output_root / DEFAULT_RAW_VERIFICATION.name
    _write(raw_output, raw)
    manifest = {
        "schema_version": "v4r-metric-replay-checkpoint-recovery-v1",
        "status": "exact_internal_equivalence_for_posthoc_metric_replay",
        "classification": (
            "deterministic_checkpoint_reconstruction_after_binary_archival"
        ),
        "completed_at_utc": raw["completed_at_utc"],
        "performed_after_sealed_test_unblinding": True,
        "fresh_generalization_evidence": False,
        "validation_or_test_opened_during_recovery": (
            validation_or_test_opened
        ),
        "data_access_audit": data_access_audit,
        "training_computation_performed": True,
        "training_purpose": (
            "reconstruct frozen V3 checkpoint state for post-hoc telemetry replay"
        ),
        "training_or_retuning_for_policy_change": False,
        "model_selection_performed": False,
        "member_selection_or_exclusion_performed": False,
        "member_weighting_changed": False,
        "policy_critic_or_normalizer_identity_changed": False,
        "original_sealed_result_immutable": True,
        "march_april_replay_is_fresh_sealed_test": False,
        "corrective_replay_scope": (
            "physical-ramp-status-quo-labeling-and-decoder-telemetry-only"
        ),
        "checkpoint_root": relative_recovery_root,
        "checkpoint_binaries_tracked": False,
        "checkpoint_binaries_ignored": True,
        "stable_recovery_commit": STABLE_RECOVERY_COMMIT,
        "stable_recovery_tooling": {
            path: git_blob_reference(ROOT, STABLE_RECOVERY_COMMIT, path)
            for path in STABLE_TOOL_PATHS
        },
        "raw_equivalence_verification": json_file_reference(
            ROOT,
            raw_output,
        ),
        "raw_equivalence_contract": raw["equivalence_contract"],
        "raw_equivalence_all_exact": raw["all_exact"],
        "source_identity": raw["source_identity"],
        "sealed_evidence": {
            **json_file_reference(ROOT, SEALED_CANONICAL),
            "sha256": SEALED_CANONICAL_SHA256,
            "immutable": True,
            "sealed_test_open_count": 1,
        },
        "members": members,
    }
    output_path = output_root / DEFAULT_MANIFEST.name
    _write(output_path, manifest)
    verify_metric_replay_recovery_manifest(
        output_path,
        require_binaries=True,
    )
    return manifest


def verify_metric_replay_recovery_manifest(
    path: Path = DEFAULT_MANIFEST,
    *,
    require_binaries: bool = False,
) -> dict[str, Any]:
    path = path.resolve()
    manifest = _load(path)
    _require(
        manifest.get("schema_version")
        == "v4r-metric-replay-checkpoint-recovery-v1",
        "unsupported metric-replay recovery manifest",
    )
    _require(
        manifest.get("status")
        == "exact_internal_equivalence_for_posthoc_metric_replay",
        "metric-replay checkpoint recovery is not exact",
    )
    expected_false = (
        "fresh_generalization_evidence",
        "validation_or_test_opened_during_recovery",
        "training_or_retuning_for_policy_change",
        "model_selection_performed",
        "member_selection_or_exclusion_performed",
        "member_weighting_changed",
        "policy_critic_or_normalizer_identity_changed",
        "march_april_replay_is_fresh_sealed_test",
        "checkpoint_binaries_tracked",
    )
    for field in expected_false:
        _require(
            manifest.get(field) is False,
            f"invalid metric-replay recovery declaration: {field}",
        )
    for field in (
        "performed_after_sealed_test_unblinding",
        "training_computation_performed",
        "original_sealed_result_immutable",
        "checkpoint_binaries_ignored",
        "raw_equivalence_all_exact",
    ):
        _require(
            manifest.get(field) is True,
            f"missing metric-replay recovery declaration: {field}",
        )
    _require(
        manifest.get("stable_recovery_commit") == STABLE_RECOVERY_COMMIT,
        "metric replay used the wrong recovery commit",
    )
    _require(
        manifest.get("checkpoint_root")
        == "models/ramp_rl_v6/recovery_v3_metric_replay"
        and "models/ramp_rl_v6/recovery_v3_metric_replay/"
        in (ROOT / ".gitignore").read_text(encoding="utf-8"),
        "metric-replay checkpoint root is not the declared ignored path",
    )
    for tool_path in STABLE_TOOL_PATHS:
        _require(
            manifest["stable_recovery_tooling"][tool_path]
            == git_blob_reference(ROOT, STABLE_RECOVERY_COMMIT, tool_path),
            f"stable recovery tooling reference changed: {tool_path}",
        )
    raw_reference = manifest["raw_equivalence_verification"]
    raw_path = ROOT / raw_reference["path"]
    _require(raw_path.is_file(), "raw recovery verification is missing")
    _require(
        raw_reference["representation"] == CANONICAL_JSON_REPRESENTATION
        and canonical_json_file_sha256(raw_path) == raw_reference["sha256"],
        "raw recovery verification hash mismatch",
    )
    raw = _load(raw_path)
    _validate_raw_verification(raw)
    raw_members = {
        int(member["seed"]): member
        for member in raw["seeds"]
    }
    sealed = manifest["sealed_evidence"]
    _require(
        sealed["sha256"] == SEALED_CANONICAL_SHA256
        and sealed["representation"] == CANONICAL_JSON_REPRESENTATION
        and canonical_json_file_sha256(ROOT / sealed["path"])
        == SEALED_CANONICAL_SHA256,
        "metric-replay recovery points to the wrong sealed evidence",
    )
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    bindings = {
        int(binding["seed"]): binding
        for binding in protocol["frozen_bindings"]["members"]
    }
    members = manifest.get("members")
    _require(
        isinstance(members, list)
        and [int(member["seed"]) for member in members]
        == [2801, 2802, 2803, 2804, 2805],
        "metric-replay recovery member set changed",
    )
    for member in members:
        seed = int(member["seed"])
        binding = bindings[seed]
        _require(
            member["initial_policy_sha256"]
            == binding["initial_policy_sha256"]
            and member["initial_critic_sha256"]
            == binding["initial_critic_sha256"]
            and member["final_policy_sha256"]
            == binding["final_policy_sha256"]
            and member["final_critic_sha256"]
            == binding["final_critic_sha256"],
            f"seed {seed} policy or critic identity changed",
        )
        _require(
            member["interaction_count"] == binding["interaction_count"]
            and member["update_count"] == binding["update_count"],
            f"seed {seed} interaction or update count changed",
        )
        _require(
            member["vecnormalize"]["sha256"]
            == binding["recovered_vecnormalize_sha256"]
            and member["vecnormalize"]["byte_identical_to_frozen_v3"] is True,
            f"seed {seed} normalizer identity changed",
        )
        _require(
            member["comparison_fields"] == list(COMPARISON_FIELDS)
            and member["comparison_fields_exact"] is True
            and member["training_data_splits_observed"] == ["train"]
            and member["validation_or_test_rows"] == 0,
            f"seed {seed} recovery provenance contract changed",
        )
        model_record = member["model_container"]
        raw_member = raw_members[seed]
        _require(
            model_record["representation"] == RAW_BYTES_REPRESENTATION
            and model_record["original_v4r_container_sha256"]
            == binding["recovered_model_sha256"],
            f"seed {seed} model container binding changed",
        )
        _require(
            model_record["sha256"]
            == raw_member["artifacts"]["model.zip"]["reproduced_sha256"]
            and member["vecnormalize"]["sha256"]
            == raw_member["artifacts"]["vecnormalize.pkl"]["reproduced_sha256"]
            and member["reconstructed_training_manifest"]["sha256"]
            == raw_member["recovered_manifest_sha256"],
            f"seed {seed} compact recovery binding differs from raw verifier",
        )
        original_manifest_path = (
            ROOT / member["original_training_manifest"]["path"]
        )
        _require(
            historical_text_sha256_matches(
                original_manifest_path,
                raw_member["original_manifest_sha256"],
            ),
            f"seed {seed} raw verifier original manifest changed",
        )
        if require_binaries:
            model_path = ROOT / model_record["path"]
            normalization_path = ROOT / member["vecnormalize"]["path"]
            recovered_manifest_path = (
                ROOT / member["reconstructed_training_manifest"]["path"]
            )
            _require(
                model_path.is_file()
                and sha256_file(model_path) == model_record["sha256"],
                f"seed {seed} local reconstructed model is missing or changed",
            )
            _require(
                normalization_path.is_file()
                and sha256_file(normalization_path)
                == member["vecnormalize"]["sha256"],
                f"seed {seed} local reconstructed normalizer is missing or changed",
            )
            _require(
                recovered_manifest_path.is_file()
                and sha256_file(recovered_manifest_path)
                == member["reconstructed_training_manifest"]["sha256"],
                f"seed {seed} local reconstructed manifest is missing or changed",
            )
            loaded = model_hashes(PPO.load(model_path, device="cpu"))
            _require(
                loaded
                == {
                    "policy": binding["final_policy_sha256"],
                    "critic": binding["final_critic_sha256"],
                },
                f"seed {seed} local reconstructed tensors changed",
            )
    expected_data_access_audit = {
        "raw_source_test_opened": False,
        "member_training_splits": {
            str(member["seed"]): ["train"] for member in members
        },
        "member_validation_or_test_rows": {
            str(member["seed"]): 0 for member in members
        },
    }
    _require(
        manifest.get("data_access_audit") == expected_data_access_audit,
        "metric-replay recovery data-access audit changed",
    )
    return manifest
