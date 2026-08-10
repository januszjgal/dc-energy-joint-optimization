"""Replay frozen V4R policies for post-hoc telemetry correction only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env.ramp_v6.factory import make_energy_model_v3_env  # noqa: E402
from ramp_rl.provenance import (  # noqa: E402
    CANONICAL_JSON_REPRESENTATION,
    canonical_json_file_sha256,
)
from ramp_rl.recovered_ensemble import (  # noqa: E402
    PROTOCOL_ID,
    evaluate_recovered_equal_action_ensemble,
)
from ramp_rl.v4r_metric_replay_recovery import (  # noqa: E402
    DEFAULT_MANIFEST as DEFAULT_RECOVERY_MANIFEST,
    verify_metric_replay_recovery_manifest,
)
from ramp_rl.v4r_thesis import (  # noqa: E402
    DEFAULT_CANONICAL as ORIGINAL_CANONICAL,
    EXPECTED_CANONICAL_SHA256 as ORIGINAL_CANONICAL_SHA256,
    load_verified_evidence,
)


DEFAULT_OUTPUT = (
    ROOT
    / "output"
    / "ramp_rl_v6"
    / "recovered_v4r_posthoc_metrics_v1"
)
PROTOCOL_PATH = (
    ROOT / "env" / "protocols" / "v6_pure_ramp_rl_v4r.yaml"
)
FACTORY_MANIFEST = (
    ROOT / "output" / "energy_model_v3" / "ramp_v6" / "factory_manifest.json"
)
ORIGINAL_RESULTS = {
    "validation": (
        ROOT
        / "output"
        / "ramp_rl_v6"
        / "recovered_v4r"
        / "validation"
        / "ensemble_validation.json"
    ),
    "test": (
        ROOT
        / "output"
        / "ramp_rl_v6"
        / "recovered_v4r"
        / "test"
        / "ensemble_test.json"
    ),
}
CORRECTED_SOURCE_PATHS = (
    ".gitignore",
    "env/ramp_v6/environment.py",
    "env/ramp_v6/factory.py",
    "env/ramp_v6/projection.py",
    "energy_model_v3/ramp_factory.py",
    "ramp_rl/contract.py",
    "ramp_rl/ensemble.py",
    "ramp_rl/evaluation.py",
    "ramp_rl/evidence.py",
    "ramp_rl/provenance.py",
    "ramp_rl/recovered_ensemble.py",
    "ramp_rl/v4r_metric_replay_recovery.py",
    "scripts/bind_v4r_metric_replay_recovery.py",
    "scripts/recompute_v4r_posthoc_metrics.py",
)
INVARIANT_FIELDS = (
    "episode_count",
    "service_unserved",
    "batch_unfinished",
    "batch_expired",
    "certificate_violations",
    "terminal_work",
    "mean_incremental_ramp_impact",
    "per_market_macro",
    "energy_cost_ratio",
    "emergency_feasibility_rate",
    "future_leakage_detected",
    "behavior_audit",
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
    )


def _raw_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalized_text_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _reference(path: Path) -> dict[str, str]:
    return {
        "algorithm": "sha256",
        "representation": CANONICAL_JSON_REPRESENTATION,
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": canonical_json_file_sha256(path),
    }


def _windows(split: str) -> list[str]:
    manifest = _load(FACTORY_MANIFEST)
    windows = sorted(manifest["windows"][split])
    expected_periods = {
        "validation": {"2026-02"},
        "test": {"2026-03", "2026-04"},
    }[split]
    periods = {
        str(manifest["windows"][split][window]["period"])
        for window in windows
    }
    if periods != expected_periods:
        raise ValueError(f"{split} windows do not match the frozen calendar")
    return windows


def _assert_equal(actual: Any, expected: Any, label: str) -> None:
    if isinstance(actual, float) or isinstance(expected, float):
        if not math.isclose(
            float(actual),
            float(expected),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(
                f"post-hoc replay changed {label}: "
                f"{actual!r} != {expected!r}"
            )
        return
    if isinstance(actual, dict) and isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError(f"post-hoc replay changed keys for {label}")
        for key in actual:
            _assert_equal(
                actual[key],
                expected[key],
                f"{label}.{key}",
            )
        return
    if actual != expected:
        raise ValueError(
            f"post-hoc replay changed {label}: "
            f"{actual!r} != {expected!r}"
        )


def _verify_invariants(
    corrected: dict[str, Any],
    original: dict[str, Any],
    *,
    split: str,
) -> dict[str, Any]:
    for field in INVARIANT_FIELDS:
        _assert_equal(
            corrected[field],
            original[field],
            f"{split}.{field}",
        )
    _assert_equal(
        corrected["controller_audit"]["ensemble_action_chain_sha256"],
        original["controller_audit"]["ensemble_action_chain_sha256"],
        f"{split}.controller_audit.ensemble_action_chain_sha256",
    )
    original_policy_cost = sum(
        float(episode["energy_cost"])
        for episode in original["policy_episodes"]
    )
    corrected_policy_cost = sum(
        float(episode["energy_cost"])
        for episode in corrected["policy_episodes"]
    )
    original_status_cost = sum(
        float(episode["energy_cost"])
        for episode in original["status_quo_episodes"]
    )
    corrected_status_cost = sum(
        float(episode["energy_cost"])
        for episode in corrected["status_quo_episodes"]
    )
    _assert_equal(
        corrected_policy_cost,
        original_policy_cost,
        f"{split}.policy_energy_cost",
    )
    _assert_equal(
        corrected_status_cost,
        original_status_cost,
        f"{split}.status_quo_energy_cost",
    )
    return {
        "decision_relevant_metrics_unchanged": True,
        "ensemble_action_chain_unchanged": True,
        "policy_energy_cost_unchanged": True,
        "status_quo_energy_cost_unchanged": True,
        "original_result": _reference(ORIGINAL_RESULTS[split]),
    }


def _correction_declaration(
    split: str,
    *,
    checkpoint_reconstruction_performed: bool,
) -> dict[str, Any]:
    return {
        "classification": "post_hoc_frozen_policy_metric_recomputation",
        "split": split,
        "performed_after_sealed_test_unblinding": True,
        "second_sealed_generalization_test": False,
        "training_performed": False,
        "retuning_performed": False,
        "model_selection_performed": False,
        "member_selection_or_exclusion_performed": False,
        "member_weighting_changed": False,
        "policy_or_critic_parameters_changed": False,
        "deterministic_checkpoint_reconstruction_performed_before_replay": (
            checkpoint_reconstruction_performed
        ),
        "normalization_changed": False,
        "environment_dynamics_changed": False,
        "reward_or_primary_objective_changed": False,
        "corrected_telemetry": [
            "absolute adjusted 1h/3h ramp magnitudes pooled over market-timestep values",
            "decoded semantic allocation adjustment L2 distribution",
            "explicit policy-vs-status-quo incremental ramp comparison",
            "native-grid-relative gate labels",
        ],
    }


def _replay_bindings(
    bindings: list[dict[str, Any]],
    recovered_binary_root: Path | None,
    recovery_manifest: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if recovered_binary_root is None:
        if recovery_manifest is not None:
            raise ValueError(
                "checkpoint recovery manifest requires reconstructed binaries"
            )
        return bindings, {
            "mode": "protocol_bound_v4r_containers",
            "deterministic_checkpoint_reconstruction_performed": False,
            "members": [
                {
                    "seed": int(binding["seed"]),
                    "protocol_model_sha256": binding[
                        "recovered_model_sha256"
                    ],
                    "replay_model_sha256": binding[
                        "recovered_model_sha256"
                    ],
                    "container_byte_identical_to_protocol": True,
                    "normalizer_sha256": binding[
                        "recovered_vecnormalize_sha256"
                    ],
                }
                for binding in bindings
            ],
        }
    if recovery_manifest is None:
        raise ValueError(
            "reconstructed binaries require a verified checkpoint recovery manifest"
        )
    recovery_members = {
        int(member["seed"]): member
        for member in recovery_manifest["members"]
    }
    replay: list[dict[str, Any]] = []
    members: list[dict[str, Any]] = []
    for binding in bindings:
        seed = int(binding["seed"])
        member_root = recovered_binary_root / str(seed)
        model_path = member_root / "model.zip"
        normalization_path = member_root / "vecnormalize.pkl"
        if not model_path.is_file() or not normalization_path.is_file():
            raise FileNotFoundError(
                f"missing reconstructed checkpoint for seed {seed}: "
                f"{member_root}"
            )
        model_sha256 = _raw_sha256(model_path)
        normalization_sha256 = _raw_sha256(normalization_path)
        recovery_member = recovery_members[seed]
        if (
            model_sha256
            != recovery_member["model_container"]["sha256"]
            or normalization_sha256
            != recovery_member["vecnormalize"]["sha256"]
        ):
            raise ValueError(
                f"seed {seed} reconstructed checkpoint differs from recovery manifest"
            )
        if (
            normalization_sha256
            != binding["recovered_vecnormalize_sha256"]
        ):
            raise ValueError(
                f"seed {seed} reconstructed normalizer differs from V4R"
            )
        replay.append(
            {
                **binding,
                "recovered_model_path": str(model_path.resolve()),
                "recovered_model_sha256": model_sha256,
                "recovered_vecnormalize_path": str(
                    normalization_path.resolve()
                ),
                "recovered_vecnormalize_sha256": normalization_sha256,
            }
        )
        members.append(
            {
                "seed": seed,
                "protocol_model_sha256": binding[
                    "recovered_model_sha256"
                ],
                "replay_model_sha256": model_sha256,
                "container_byte_identical_to_protocol": (
                    model_sha256 == binding["recovered_model_sha256"]
                ),
                "normalizer_sha256": normalization_sha256,
                "final_policy_sha256": binding["final_policy_sha256"],
                "final_critic_sha256": binding["final_critic_sha256"],
            }
        )
    return replay, {
        "mode": "deterministically_reconstructed_v3_checkpoints",
        "deterministic_checkpoint_reconstruction_performed": True,
        "purpose": (
            "restore exact frozen policy/critic/normalizer state for "
            "post-hoc metric replay after ignored containers were unavailable"
        ),
        "fresh_generalization_evidence": False,
        "selection_or_tuning": False,
        "members": members,
    }


def recompute(
    *,
    output: Path,
    recomputed_at_utc: str,
    recovered_binary_root: Path | None = None,
    recovery_manifest_path: Path | None = None,
) -> dict[str, Any]:
    load_verified_evidence()
    if recovered_binary_root is None:
        raise ValueError(
            "canonical v2 metric correction requires reconstructed binaries "
            "and a tracked checkpoint recovery manifest"
        )
    if canonical_json_file_sha256(
        ORIGINAL_CANONICAL
    ) != ORIGINAL_CANONICAL_SHA256:
        raise ValueError("original canonical sealed evidence changed")
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol["protocol"]["id"] != PROTOCOL_ID:
        raise ValueError("unexpected V4R protocol identity")
    recovery_manifest = None
    resolved_recovery_manifest = None
    if recovered_binary_root is not None:
        resolved_recovery_manifest = (
            recovery_manifest_path or DEFAULT_RECOVERY_MANIFEST
        ).resolve()
        recovery_manifest = verify_metric_replay_recovery_manifest(
            resolved_recovery_manifest,
            require_binaries=True,
        )
    bindings, binary_recovery = _replay_bindings(
        protocol["frozen_bindings"]["members"],
        recovered_binary_root,
        recovery_manifest,
    )
    reconstructed = bool(
        binary_recovery[
            "deterministic_checkpoint_reconstruction_performed"
        ]
    )
    corrected_paths: dict[str, Path] = {}
    invariant_audits: dict[str, dict[str, Any]] = {}
    for split in ("validation", "test"):
        corrected = evaluate_recovered_equal_action_ensemble(
            factory=make_energy_model_v3_env,
            bindings=bindings,
            root=ROOT,
            split=split,
            seed=int(protocol["evaluation"]["environment_seed"]),
            windows=_windows(split),
        )
        original = _load(ORIGINAL_RESULTS[split])
        invariant_audits[split] = _verify_invariants(
            corrected,
            original,
            split=split,
        )
        corrected_payload = {
            "schema_version": "v4r-posthoc-corrected-metrics-result-v1",
            "protocol_id": PROTOCOL_ID,
            "recomputed_at_utc": recomputed_at_utc,
            "declaration": _correction_declaration(
                split,
                checkpoint_reconstruction_performed=reconstructed,
            ),
            "invariant_audit": invariant_audits[split],
            "result": corrected,
        }
        path = output / f"{split}_corrected_metrics.json"
        _write(path, corrected_payload)
        corrected_paths[split] = path

    source_hashes = {
        path: {
            "representation": "utf8-lf-text-v1",
            "sha256": _normalized_text_sha256(ROOT / path),
        }
        for path in CORRECTED_SOURCE_PATHS
    }
    wrapper = {
        "schema_version": "v4r-posthoc-corrected-metrics-canonical-v2",
        "status": (
            "authoritative_posthoc_metric_correction_with_bound_checkpoint_recovery"
        ),
        "protocol_id": PROTOCOL_ID,
        "recomputed_at_utc": recomputed_at_utc,
        "original_sealed_evidence": {
            **_reference(ORIGINAL_CANONICAL),
            "immutable": True,
            "sealed_test_open_count": 1,
            "superseded": False,
        },
        "supersession_scope": (
            "physical-ramp-status-quo-labeling-and-decoder-telemetry-only"
        ),
        "original_sealed_result_and_chronology_preserved": True,
        "second_sealed_generalization_test": False,
        "metric_replay_training_or_retuning_performed": False,
        "deterministic_checkpoint_reconstruction_performed": (
            reconstructed
        ),
        "model_selection_or_weighting_performed": False,
        "policy_critic_normalization_controller_data_and_forecast_identities_changed": False,
        "factory_manifest_or_window_content_changed": False,
        "environment_transition_or_reward_changed": False,
        "telemetry_and_portable_verification_source_changed": True,
        "primary_incremental_squared_ramp_objective_changed": False,
        "binary_recovery": binary_recovery,
        "checkpoint_recovery": (
            _reference(resolved_recovery_manifest)
            if resolved_recovery_manifest is not None
            else None
        ),
        "corrected_results": {
            split: _reference(path)
            for split, path in corrected_paths.items()
        },
        "invariant_audits": invariant_audits,
        "corrected_source_files": source_hashes,
        "factory_manifest": {
            "algorithm": "sha256",
            "representation": CANONICAL_JSON_REPRESENTATION,
            "path": FACTORY_MANIFEST.relative_to(ROOT).as_posix(),
            "sha256": canonical_json_file_sha256(FACTORY_MANIFEST),
        },
        "protocol": {
            "algorithm": "sha256",
            "representation": "utf8-lf-text-v1",
            "path": PROTOCOL_PATH.relative_to(ROOT).as_posix(),
            "sha256": _normalized_text_sha256(PROTOCOL_PATH),
        },
    }
    canonical = output / "canonical_posthoc_metrics.json"
    _write(canonical, wrapper)
    verify(canonical)
    return wrapper


def _verify_reference(reference: dict[str, Any], label: str) -> Path:
    if reference.get("algorithm") != "sha256":
        raise ValueError(f"{label} uses the wrong hash algorithm")
    if (
        reference.get("representation")
        != CANONICAL_JSON_REPRESENTATION
    ):
        raise ValueError(f"{label} uses the wrong representation")
    path = ROOT / str(reference.get("path", ""))
    if not path.is_file():
        raise ValueError(f"missing {label}: {path}")
    if canonical_json_file_sha256(path) != reference.get("sha256"):
        raise ValueError(f"{label} hash mismatch")
    return path


def verify(canonical: Path) -> dict[str, Any]:
    wrapper = _load(canonical)
    if (
        wrapper.get("schema_version")
        != "v4r-posthoc-corrected-metrics-canonical-v2"
    ):
        raise ValueError("unsupported post-hoc metric package")
    if (
        wrapper.get("status")
        != "authoritative_posthoc_metric_correction_with_bound_checkpoint_recovery"
    ):
        raise ValueError("post-hoc metric package is not authoritative")
    if wrapper.get("second_sealed_generalization_test") is not False:
        raise ValueError("post-hoc replay is mislabeled as a new sealed test")
    for field in (
        "metric_replay_training_or_retuning_performed",
        "model_selection_or_weighting_performed",
        "policy_critic_normalization_controller_data_and_forecast_identities_changed",
        "factory_manifest_or_window_content_changed",
        "environment_transition_or_reward_changed",
        "primary_incremental_squared_ramp_objective_changed",
    ):
        if wrapper.get(field) is not False:
            raise ValueError(f"invalid post-hoc declaration: {field}")
    original = _verify_reference(
        wrapper["original_sealed_evidence"],
        "original sealed evidence",
    )
    if canonical_json_file_sha256(original) != ORIGINAL_CANONICAL_SHA256:
        raise ValueError("post-hoc package points to the wrong sealed evidence")
    for relative, reference in wrapper.get(
        "corrected_source_files", {}
    ).items():
        if reference.get("representation") != "utf8-lf-text-v1":
            raise ValueError(
                f"corrected source uses the wrong representation: {relative}"
            )
        source_path = ROOT / relative
        if (
            not source_path.is_file()
            or _normalized_text_sha256(source_path)
            != reference.get("sha256")
        ):
            raise ValueError(f"corrected source hash mismatch: {relative}")
    factory_reference = wrapper.get("factory_manifest", {})
    if (
        factory_reference.get("representation")
        != CANONICAL_JSON_REPRESENTATION
        or canonical_json_file_sha256(FACTORY_MANIFEST)
        != factory_reference.get("sha256")
    ):
        raise ValueError("factory manifest hash mismatch")
    protocol_reference = wrapper.get("protocol", {})
    if (
        protocol_reference.get("representation") != "utf8-lf-text-v1"
        or _normalized_text_sha256(PROTOCOL_PATH)
        != protocol_reference.get("sha256")
    ):
        raise ValueError("protocol hash mismatch")
    checkpoint_recovery = wrapper.get("checkpoint_recovery")
    if wrapper.get("deterministic_checkpoint_reconstruction_performed"):
        if not isinstance(checkpoint_recovery, dict):
            raise ValueError("checkpoint recovery reference is missing")
        recovery_path = _verify_reference(
            checkpoint_recovery,
            "checkpoint recovery",
        )
        recovery = verify_metric_replay_recovery_manifest(recovery_path)
        recovery_members = {
            int(member["seed"]): member
            for member in recovery["members"]
        }
        for member in wrapper.get("binary_recovery", {}).get("members", []):
            seed = int(member["seed"])
            if (
                member["replay_model_sha256"]
                != recovery_members[seed]["model_container"]["sha256"]
                or member["normalizer_sha256"]
                != recovery_members[seed]["vecnormalize"]["sha256"]
                or member["final_policy_sha256"]
                != recovery_members[seed]["final_policy_sha256"]
                or member["final_critic_sha256"]
                != recovery_members[seed]["final_critic_sha256"]
            ):
                raise ValueError(
                    f"seed {seed} binary recovery differs from tracked manifest"
                )
    else:
        raise ValueError(
            "canonical v2 metric correction is missing checkpoint recovery"
        )
    for split in ("validation", "test"):
        path = _verify_reference(
            wrapper["corrected_results"][split],
            f"{split} corrected result",
        )
        payload = _load(path)
        declaration = payload.get("declaration", {})
        if declaration.get("second_sealed_generalization_test") is not False:
            raise ValueError(f"{split} correction has the wrong test label")
        if declaration.get("training_performed") is not False:
            raise ValueError(f"{split} correction performed training")
        result = payload.get("result", {})
        if (
            result.get("physical_ramp_metric_contract", {}).get(
                "cross_market_signed_averaging"
            )
            is not False
        ):
            raise ValueError(f"{split} physical ramp contract is ambiguous")
        if "status_quo_comparison" not in result:
            raise ValueError(f"{split} status-quo comparison is missing")
        if "semantic_adjustment" not in result:
            raise ValueError(f"{split} decoder distribution is missing")
    return wrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    replay = subparsers.add_parser("recompute")
    replay.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    replay.add_argument(
        "--recomputed-at-utc",
        required=True,
        help="Explicit ISO-8601 timestamp recorded in the immutable wrapper.",
    )
    replay.add_argument(
        "--recovered-binary-root",
        type=Path,
        help=(
            "Optional root containing deterministically reconstructed "
            "<seed>/model.zip and vecnormalize.pkl files. Internal policy, "
            "critic, and normalizer identities must match frozen V4R."
        ),
    )
    replay.add_argument(
        "--recovery-manifest",
        type=Path,
        help=(
            "Tracked metric-replay checkpoint recovery manifest. Required "
            "when reconstructed binaries are used."
        ),
    )
    check = subparsers.add_parser("verify")
    check.add_argument(
        "--canonical",
        type=Path,
        default=DEFAULT_OUTPUT / "canonical_posthoc_metrics.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "recompute":
        payload = recompute(
            output=args.output.resolve(),
            recomputed_at_utc=str(args.recomputed_at_utc),
            recovered_binary_root=(
                args.recovered_binary_root.resolve()
                if args.recovered_binary_root
                else None
            ),
            recovery_manifest_path=(
                args.recovery_manifest.resolve()
                if args.recovery_manifest
                else None
            ),
        )
        canonical = args.output.resolve() / "canonical_posthoc_metrics.json"
        print(
            json.dumps(
                {
                    "status": payload["status"],
                    "canonical": canonical.relative_to(ROOT).as_posix(),
                    "canonical_json_sha256": (
                        canonical_json_file_sha256(canonical)
                    ),
                },
                indent=2,
            )
        )
        return 0
    payload = verify(args.canonical.resolve())
    print(
        json.dumps(
            {
                "status": payload["status"],
                "canonical_json_sha256": canonical_json_file_sha256(
                    args.canonical.resolve()
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
