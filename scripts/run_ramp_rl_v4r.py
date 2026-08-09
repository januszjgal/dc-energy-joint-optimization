"""Bind, freeze, and evaluate the distinct recovered-policy V4R ensemble."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from env.ramp_v6.factory import make_energy_model_v3_env  # noqa: E402
from ramp_rl.ensemble import EXPECTED_MEMBER_SEEDS  # noqa: E402
from ramp_rl.ensemble_robustness import (  # noqa: E402
    make_c_h_overlapping_env,
    make_one_gw_total_env,
)
from ramp_rl.evidence import sha256_file  # noqa: E402
from ramp_rl.recovered_ensemble import (  # noqa: E402
    PROTOCOL_ID,
    evaluate_recovered_equal_action_ensemble,
)
from ramp_rl.runner import model_hashes  # noqa: E402


PROTOCOL_PATH = ROOT / "env" / "protocols" / "v6_pure_ramp_rl_v4r.yaml"
FACTORY_MANIFEST = (
    ROOT / "output" / "energy_model_v3" / "ramp_v6" / "factory_manifest.json"
)
V4R_ROOT = ROOT / "output" / "ramp_rl_v6" / "recovered_v4r"
RECOVERY_SOURCE_REPORT = V4R_ROOT / "recovery_source_manifest.json"
RECOVERY_BINDING = V4R_ROOT / "recovery_binding.json"
SOURCE_FREEZE = V4R_ROOT / "source_freeze.json"
VALIDATION_RUN_RECORD = V4R_ROOT / "validation_evaluation_opening.json"
VALIDATION_RESULT = V4R_ROOT / "validation" / "ensemble_validation.json"
VALIDATION_DECISION = V4R_ROOT / "validation_decision.json"
VALIDATION_REPORT = V4R_ROOT / "validation_report.md"
TEST_OPENING = V4R_ROOT / "sealed_test_opening.json"
TEST_RUN_RECORD = V4R_ROOT / "sealed_test_evaluation_opening.json"
TEST_RESULT = V4R_ROOT / "test" / "ensemble_test.json"
TEST_DECISION = V4R_ROOT / "test_decision.json"
ROBUSTNESS_ONE_GW = V4R_ROOT / "robustness" / "one_gw_total_test.json"
ROBUSTNESS_C_H = V4R_ROOT / "robustness" / "c_h_overlapping_test.json"
CANONICAL_EVIDENCE = V4R_ROOT / "canonical_evidence.json"
CANONICAL_REPORT = V4R_ROOT / "final_report.md"

SOURCE_PATHS = (
    "env/protocols/v6_pure_ramp_rl_v4r.yaml",
    "ramp_rl/recovered_ensemble.py",
    "scripts/run_ramp_rl_v4r.py",
    "tests/ramp_v6/test_v4r_recovered_ensemble.py",
    "ramp_rl/ensemble.py",
    "ramp_rl/ensemble_robustness.py",
    "ramp_rl/campaign.py",
    "ramp_rl/contract.py",
    "ramp_rl/evaluation.py",
    "ramp_rl/evidence.py",
    "ramp_rl/runner.py",
    "ramp_rl/schema.py",
    "env/ramp_v6/environment.py",
    "env/ramp_v6/factory.py",
    "env/ramp_v6/fixture.py",
    "env/ramp_v6/models.py",
    "env/ramp_v6/panel.py",
    "env/ramp_v6/projection.py",
    "env/ramp_v6/protocol.py",
    "env/ramp_v6/reward.py",
    "energy_model_v3/ramp_factory.py",
    "env/protocols/v6_ramp_pure_rl.yaml",
    "env/protocols/v6_pure_ramp_rl.yaml",
    "env/protocols/v6_pure_ramp_rl.schema.json",
    "env/protocols/v6_ramp_panel.schema.json",
)
BLOCKED_V4_PATHS = (
    "env/protocols/v6_pure_ramp_rl_v4.yaml",
    "ramp_rl/ensemble.py",
    "ramp_rl/ensemble_robustness.py",
    "scripts/run_ramp_rl_v4.py",
    "tests/ramp_v6/test_v4_ensemble.py",
)
V3_IMMUTABLE_PATHS = (
    "env/protocols/v6_pure_ramp_rl_v3.yaml",
    "output/ramp_rl_v6/live_v3/source_freeze.json",
    "output/ramp_rl_v6/live_v3/command_manifest.json",
    "output/ramp_rl_v6/live_v3/validation_decision.json",
    "output/ramp_rl_v6/live_v3/validation_report.md",
    *(
        f"output/ramp_rl_v6/live_v3/validation/ppo_{seed}_validation.json"
        for seed in EXPECTED_MEMBER_SEEDS
    ),
    *(
        f"models/ramp_rl_v6/live_v3/confirmation/ppo/{seed}/training_manifest.json"
        for seed in EXPECTED_MEMBER_SEEDS
    ),
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _git(*arguments: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=not binary,
    )
    return result.stdout if binary else result.stdout.strip()


def _normalized_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    except FileExistsError as error:
        raise RuntimeError(f"immutable output already exists: {path}") from error


def _require_canonical(path: Path, expected: Path) -> None:
    _require(
        path.resolve() == expected.resolve(),
        f"phase output must use canonical path: {expected}",
    )


def _require_committed(path: Path, label: str) -> None:
    relative = str(path.relative_to(ROOT)).replace("\\", "/")
    _git("ls-files", "--error-unmatch", relative)
    _require(
        not _git("status", "--porcelain", "--", relative),
        f"{label} must be committed and clean: {relative}",
    )
    _require(
        _git("hash-object", relative) == _git("rev-parse", f"HEAD:{relative}"),
        f"{label} differs from the committed blob: {relative}",
    )


def load_protocol() -> dict[str, Any]:
    payload = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    _require(payload["protocol"]["id"] == PROTOCOL_ID, "unexpected V4R protocol ID")
    _require(
        payload["protocol"]["blocked_original_v4_identity_reused"] is False,
        "blocked V4 identity cannot be reused",
    )
    identity = payload["artifact_identity"]
    _require(identity["new_binary_identity"] is True, "V4R must use a new binary ID")
    _require(
        identity["original_v3_model_container_reuse"] is False,
        "V4R cannot claim original V3 containers",
    )
    _require(
        identity["performance_claim_before_evaluation"] is False,
        "V4R cannot make a pre-evaluation performance claim",
    )
    controller = payload["controller"]
    _require(
        controller["member_seeds"] == list(EXPECTED_MEMBER_SEEDS),
        "V4R member seeds changed",
    )
    _require(controller["weights"] == [0.2] * 5, "V4R weights are not equal")
    _require(controller["action_space"] == "environment", "wrong action space")
    _require(controller["action_dimensions"] == 13, "wrong action dimension")
    _require(controller["action_bounds"] == [-6.0, 6.0], "wrong action bounds")
    for key in (
        "trainable_combiner",
        "member_selection",
        "member_exclusion",
        "member_weighting_from_validation",
        "teacher",
        "behavior_cloning",
        "demonstrations",
        "mpc_actions",
        "optimizer_actions",
        "analytic_actions",
        "evaluation_actions",
    ):
        _require(controller[key] is False, f"forbidden V4R controller role: {key}")
    gate = payload["success_gate"]
    _require(
        gate["macro_raw_incremental_ramp_impact_lt"] == 0.0
        and gate["every_market_raw_incremental_ramp_impact_lt"] == 0.0
        and gate["primary_da_cost_ratio_max"] == 1.02
        and gate["emergency_feasibility_rate_lt"] == 0.01
        and gate["post_hoc_tolerance"] is False,
        "V4R strict gates changed",
    )
    payload["_path"] = str(PROTOCOL_PATH.relative_to(ROOT))
    payload["_sha256"] = _normalized_sha256(PROTOCOL_PATH)
    return payload


def _verify_file(path: Path, expected: str, label: str) -> None:
    _require(path.is_file(), f"missing {label}: {path}")
    _require(sha256_file(path) == expected, f"{label} hash mismatch: {path}")


def _verify_member_files(
    protocol: dict[str, Any], *, load_weights: bool
) -> list[dict[str, Any]]:
    verified: list[dict[str, Any]] = []
    for member in protocol["frozen_bindings"]["members"]:
        seed = int(member["seed"])
        manifest_path = ROOT / member["original_training_manifest_path"]
        model_path = ROOT / member["recovered_model_path"]
        normalization_path = ROOT / member["recovered_vecnormalize_path"]
        _verify_file(
            manifest_path,
            member["original_training_manifest_sha256"],
            f"seed {seed} original training manifest",
        )
        _verify_file(
            model_path,
            member["recovered_model_sha256"],
            f"seed {seed} recovered model",
        )
        _verify_file(
            normalization_path,
            member["recovered_vecnormalize_sha256"],
            f"seed {seed} recovered normalization",
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _require(
            manifest["artifacts"]["model"]["sha256"]
            == member["original_model_sha256"],
            f"seed {seed} original container binding changed",
        )
        _require(
            manifest["artifacts"]["normalization"]["sha256"]
            == member["recovered_vecnormalize_sha256"],
            f"seed {seed} normalization is not V3-byte-identical",
        )
        for field in (
            "initial_policy_sha256",
            "initial_critic_sha256",
            "final_policy_sha256",
            "final_critic_sha256",
            "interaction_count",
            "update_count",
        ):
            _require(
                manifest[field] == member[field],
                f"seed {seed} {field} differs from original V3 manifest",
            )
        loaded = None
        if load_weights:
            loaded = model_hashes(PPO.load(model_path, device="cpu"))
            _require(
                loaded["policy"] == member["final_policy_sha256"],
                f"seed {seed} recovered policy hash mismatch",
            )
            _require(
                loaded["critic"] == member["final_critic_sha256"],
                f"seed {seed} recovered critic hash mismatch",
            )
        verified.append(
            {
                "seed": seed,
                "original_training_manifest_sha256": sha256_file(manifest_path),
                "original_model_sha256": member["original_model_sha256"],
                "recovered_model_sha256": sha256_file(model_path),
                "recovered_vecnormalize_sha256": sha256_file(normalization_path),
                "loaded_weights": loaded,
                "weight_equivalent_to_v3": bool(loaded),
                "normalization_byte_equivalent_to_v3": True,
                "new_model_container_identity": (
                    member["recovered_model_sha256"]
                    != member["original_model_sha256"]
                ),
            }
        )
    return verified


def _verify_static_bindings(protocol: dict[str, Any]) -> None:
    bindings = protocol["frozen_bindings"]
    _verify_file(
        ROOT / bindings["v3_protocol"]["path"],
        bindings["v3_protocol"]["file_sha256"],
        "V3 protocol",
    )
    _verify_file(
        ROOT / bindings["v3_source_freeze"]["path"],
        bindings["v3_source_freeze"]["sha256"],
        "V3 source freeze",
    )
    factory = bindings["factory"]
    _verify_file(
        ROOT / factory["manifest_path"],
        factory["manifest_sha256"],
        "factory manifest",
    )
    _verify_file(
        ROOT / "env" / "ramp_v6" / "factory.py",
        factory["source_sha256"],
        "factory source",
    )
    robustness = protocol["post_selection"]["c_h_overlapping"]["input_bindings"]
    for cell in ("c", "d", "e", "f", "g", "h"):
        _verify_file(
            ROOT / "data" / "cells" / f"cell_{cell}_tiers.csv",
            robustness[f"cell_{cell}_tiers"],
            f"cell {cell} tier robustness input",
        )
        _verify_file(
            ROOT / "data" / "jobs" / f"batch_distributions_{cell}.json",
            robustness[f"batch_distributions_{cell}"],
            f"cell {cell} deadline robustness input",
        )


def bind_recovery(source_output: Path, binding_output: Path) -> dict[str, Any]:
    _require_canonical(source_output, RECOVERY_SOURCE_REPORT)
    _require_canonical(binding_output, RECOVERY_BINDING)
    _require(not source_output.exists(), f"immutable output exists: {source_output}")
    protocol = load_protocol()
    evidence = protocol["recovery_evidence"]
    recovery_commit = evidence["recovery_commit"]
    raw = _git(
        "show",
        f"{recovery_commit}:{evidence['recovery_manifest_git_path']}",
        binary=True,
    )
    assert isinstance(raw, bytes)
    _require(
        hashlib.sha256(raw).hexdigest() == evidence["recovery_manifest_sha256"],
        "committed recovery report hash mismatch",
    )
    recovery_source = _git(
        "show",
        f"{recovery_commit}:{evidence['recovery_source_git_path']}",
        binary=True,
    )
    assert isinstance(recovery_source, bytes)
    _require(
        hashlib.sha256(recovery_source).hexdigest()
        == evidence["recovery_source_sha256"],
        "committed recovery source hash mismatch",
    )
    report = json.loads(raw)
    mismatches = {
        label for label, row in report["checks"].items() if not row["matched"]
    }
    _require(
        mismatches == set(evidence["expected_mismatches"]),
        f"unexpected recovery divergences: {sorted(mismatches)}",
    )
    _require(
        report["source_identity"]["test_opened"] is False,
        "recovery process opened sealed test",
    )
    report_seeds = {int(row["seed"]): row for row in report["seeds"]}
    for member in protocol["frozen_bindings"]["members"]:
        seed = int(member["seed"])
        row = report_seeds[seed]
        _require(
            row["artifacts"]["model.zip"]["reproduced_sha256"]
            == member["recovered_model_sha256"],
            f"seed {seed} recovery report model hash mismatch",
        )
        _require(
            row["artifacts"]["model.zip"]["original_sha256"]
            == member["original_model_sha256"],
            f"seed {seed} recovery report original model hash mismatch",
        )
        _require(
            row["artifacts"]["vecnormalize.pkl"]["reproduced_sha256"]
            == member["recovered_vecnormalize_sha256"],
            f"seed {seed} recovery report normalization hash mismatch",
        )
        _require(
            row["original_manifest_sha256"]
            == member["original_training_manifest_sha256"],
            f"seed {seed} recovery report original manifest mismatch",
        )
        _require(
            row["recovered_manifest_sha256"]
            == member["recovered_training_manifest_sha256"],
            f"seed {seed} recovered manifest hash mismatch",
        )
        for field in (
            "initial_policy_sha256",
            "initial_critic_sha256",
            "final_policy_sha256",
            "final_critic_sha256",
            "interaction_count",
            "update_count",
            "normalization",
            "pure_rl_assertions",
            "pure_rl_verification",
            "replay_provenance",
            "training_data_provenance",
        ):
            _require(
                report["checks"][f"seed.{seed}.{field}"]["matched"],
                f"seed {seed} recovery equivalence failed for {field}",
            )

    verified_members = _verify_member_files(protocol, load_weights=True)
    source_output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with source_output.open("xb") as handle:
            handle.write(raw)
    except FileExistsError as error:
        raise RuntimeError(
            f"immutable output already exists: {source_output}"
        ) from error
    result = {
        "schema_version": "ramp-pure-rl-recovered-v4r-binding-v1",
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": protocol["_sha256"],
        "recovery_commit": recovery_commit,
        "recovery_source_report_path": str(source_output.relative_to(ROOT)),
        "recovery_source_report_sha256": sha256_file(source_output),
        "recovery_source_sha256": evidence["recovery_source_sha256"],
        "allowed_container_divergences": sorted(mismatches),
        "all_non_container_recovery_checks_exact": True,
        "members": verified_members,
        "classification": "new-binary-identity-weight-equivalent-recovered-pure-rl",
        "original_v3_artifact_reuse": False,
        "blocked_original_v4_identity_reused": False,
        "performance_claim": None,
        "test_opened": False,
        "created_at_utc": _now(),
    }
    _write_json(binding_output, result)
    return result


def _source_hashes() -> dict[str, str]:
    return {relative: sha256_file(ROOT / relative) for relative in SOURCE_PATHS}


def _source_bundle_sha256(hashes: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for relative in SOURCE_PATHS:
        digest.update(relative.encode("utf-8"))
        digest.update(hashes[relative].encode("ascii"))
    return digest.hexdigest()


def freeze(output: Path) -> dict[str, Any]:
    _require_canonical(output, SOURCE_FREEZE)
    protocol = load_protocol()
    _verify_static_bindings(protocol)
    _verify_member_files(protocol, load_weights=True)
    for path, label in (
        (RECOVERY_SOURCE_REPORT, "recovery source report"),
        (RECOVERY_BINDING, "recovery binding"),
    ):
        _require_committed(path, label)
    _require(
        not _git("status", "--porcelain", "--", *SOURCE_PATHS),
        "V4R source and protocol must be committed before freezing",
    )
    for relative in SOURCE_PATHS:
        _git("ls-files", "--error-unmatch", relative)
    for relative in BLOCKED_V4_PATHS:
        _require(
            _git("rev-parse", f"HEAD:{relative}")
            == _git("rev-parse", f"1ddd4c8:{relative}"),
            f"blocked original V4 source changed: {relative}",
        )
    for relative in V3_IMMUTABLE_PATHS:
        _require(
            _git("rev-parse", f"HEAD:{relative}")
            == _git("rev-parse", f"1f2a7dc:{relative}"),
            f"immutable V3 artifact changed: {relative}",
        )
    _require(
        not VALIDATION_RESULT.exists()
        and not VALIDATION_RUN_RECORD.exists()
        and not VALIDATION_DECISION.exists()
        and not TEST_OPENING.exists()
        and not TEST_RUN_RECORD.exists()
        and not TEST_RESULT.exists(),
        "evaluation evidence exists before V4R source freeze",
    )
    source_hashes = _source_hashes()
    result = {
        "schema_version": "ramp-pure-rl-recovered-v4r-source-freeze-v1",
        "protocol_id": PROTOCOL_ID,
        "protocol_path": protocol["_path"],
        "protocol_sha256": protocol["_sha256"],
        "source_commit": _git("rev-parse", "HEAD"),
        "source_files": source_hashes,
        "source_bundle_sha256": _source_bundle_sha256(source_hashes),
        "recovery_binding_path": str(RECOVERY_BINDING.relative_to(ROOT)),
        "recovery_binding_sha256": sha256_file(RECOVERY_BINDING),
        "recovery_source_report_sha256": sha256_file(RECOVERY_SOURCE_REPORT),
        "blocked_original_v4_commit": "1ddd4c8a120e12da37bbf75b8ac09efae8a534a9",
        "blocked_original_v4_unchanged": True,
        "v3_predecessor_commit": "1f2a7dc",
        "v3_artifacts_unchanged": True,
        "new_binary_identity": True,
        "test_opened": False,
        "created_at_utc": _now(),
    }
    _write_json(output, result)
    return result


def verify_source_freeze(protocol: dict[str, Any]) -> dict[str, Any]:
    _require_committed(SOURCE_FREEZE, "V4R source freeze")
    _require_committed(RECOVERY_BINDING, "V4R recovery binding")
    frozen = json.loads(SOURCE_FREEZE.read_text(encoding="utf-8"))
    _require(frozen["protocol_sha256"] == protocol["_sha256"], "protocol changed")
    hashes = _source_hashes()
    _require(hashes == frozen["source_files"], "V4R source changed after freeze")
    _require(
        _source_bundle_sha256(hashes) == frozen["source_bundle_sha256"],
        "V4R source bundle changed after freeze",
    )
    _require(
        frozen["recovery_binding_sha256"] == sha256_file(RECOVERY_BINDING),
        "V4R recovery binding changed after freeze",
    )
    _verify_static_bindings(protocol)
    _verify_member_files(protocol, load_weights=True)
    return frozen


def _factory_windows(split: str) -> list[str]:
    factory = json.loads(FACTORY_MANIFEST.read_text(encoding="utf-8"))
    windows = sorted(factory["windows"][split])
    expected = {"validation": {"2026-02"}, "test": {"2026-03", "2026-04"}}[
        split
    ]
    periods = {
        str(factory["windows"][split][window]["period"]) for window in windows
    }
    _require(periods == expected, f"{split} windows do not match frozen months")
    return windows


def evaluate(split: str, output: Path) -> dict[str, Any]:
    expected_output = VALIDATION_RESULT if split == "validation" else TEST_RESULT
    run_record = VALIDATION_RUN_RECORD if split == "validation" else TEST_RUN_RECORD
    _require_canonical(output, expected_output)
    _require(
        not output.exists() and not run_record.exists(),
        f"{split} evaluation was already opened",
    )
    protocol = load_protocol()
    frozen = verify_source_freeze(protocol)
    if split == "validation":
        _require(
            not TEST_OPENING.exists() and not TEST_RESULT.exists(),
            "sealed test was opened before V4R validation",
        )
    else:
        _require_committed(TEST_OPENING, "V4R sealed-test opening")
        opening = json.loads(TEST_OPENING.read_text(encoding="utf-8"))
        decision = json.loads(VALIDATION_DECISION.read_text(encoding="utf-8"))
        _require(opening["open_count"] == 1, "sealed-test opening is invalid")
        _require(decision["selected"] is True, "V4R validation did not pass")
        _require(
            opening["validation_decision_sha256"]
            == sha256_file(VALIDATION_DECISION)
            and opening["validation_result_sha256"] == sha256_file(VALIDATION_RESULT),
            "V4R sealed-test opening hash chain is invalid",
        )
        _require(
            opening["protocol_sha256"] == protocol["_sha256"]
            and opening["source_bundle_sha256"] == frozen["source_bundle_sha256"],
            "V4R sealed-test opening source binding is invalid",
        )
    _write_json(
        run_record,
        {
            "schema_version": "ramp-pure-rl-recovered-v4r-evaluation-opening-v1",
            "protocol_id": PROTOCOL_ID,
            "protocol_sha256": protocol["_sha256"],
            "source_bundle_sha256": frozen["source_bundle_sha256"],
            "recovery_binding_sha256": sha256_file(RECOVERY_BINDING),
            "split": split,
            "run_count": 1,
            "opened_at_utc": _now(),
        },
    )
    summary = evaluate_recovered_equal_action_ensemble(
        factory=make_energy_model_v3_env,
        bindings=protocol["frozen_bindings"]["members"],
        root=ROOT,
        split=split,
        seed=int(protocol["evaluation"]["environment_seed"]),
        windows=_factory_windows(split),
    )
    result = {
        "schema_version": "ramp-pure-rl-recovered-v4r-evaluation-v1",
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": protocol["_sha256"],
        "source_freeze_sha256": sha256_file(SOURCE_FREEZE),
        "source_bundle_sha256": frozen["source_bundle_sha256"],
        "recovery_binding_sha256": sha256_file(RECOVERY_BINDING),
        "evaluation_opening_path": str(run_record.relative_to(ROOT)),
        "evaluation_opening_sha256": sha256_file(run_record),
        "factory_manifest_sha256": sha256_file(FACTORY_MANIFEST),
        "split": split,
        "variant": "primary",
        "run_count": 1,
        "environment_seed": int(protocol["evaluation"]["environment_seed"]),
        "windows": _factory_windows(split),
        "evaluated_at_utc": _now(),
        **summary,
    }
    if split == "test":
        result["sealed_test_opening_sha256"] = sha256_file(TEST_OPENING)
        result["validation_result_sha256"] = sha256_file(VALIDATION_RESULT)
        result["validation_decision_sha256"] = sha256_file(VALIDATION_DECISION)
    _write_json(output, result)
    return result


def decide_validation(output: Path, report: Path) -> dict[str, Any]:
    _require_canonical(output, VALIDATION_DECISION)
    _require_canonical(report, VALIDATION_REPORT)
    protocol = load_protocol()
    frozen = verify_source_freeze(protocol)
    validation = json.loads(VALIDATION_RESULT.read_text(encoding="utf-8"))
    _require(validation["split"] == "validation", "wrong validation artifact")
    _require(
        validation["schema_version"]
        == "ramp-pure-rl-recovered-v4r-evaluation-v1",
        "wrong validation evidence schema",
    )
    _require(
        validation["protocol_id"] == PROTOCOL_ID
        and validation["protocol_sha256"] == protocol["_sha256"],
        "validation protocol binding is invalid",
    )
    _require(
        validation["source_freeze_sha256"] == sha256_file(SOURCE_FREEZE)
        and validation["source_bundle_sha256"] == frozen["source_bundle_sha256"],
        "validation source binding is invalid",
    )
    _require(
        validation["recovery_binding_sha256"] == sha256_file(RECOVERY_BINDING),
        "validation recovery binding is invalid",
    )
    _require(
        validation["evaluation_opening_sha256"]
        == sha256_file(VALIDATION_RUN_RECORD),
        "validation evaluation-opening binding is invalid",
    )
    selected = bool(validation["success_gate"]["passed"])
    result = {
        "schema_version": "ramp-pure-rl-recovered-v4r-validation-decision-v1",
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": protocol["_sha256"],
        "validation_path": str(VALIDATION_RESULT.relative_to(ROOT)),
        "validation_sha256": sha256_file(VALIDATION_RESULT),
        "selected": selected,
        "strict_gate": validation["success_gate"],
        "test_opened": False,
        "test_open_count": 0,
        "decision": (
            "select-v4r-and-open-sealed-test-once"
            if selected
            else "publish-immutable-v4r-failure-and-keep-test-sealed"
        ),
        "decided_at_utc": _now(),
    }
    _write_json(output, result)
    _require(not report.exists(), f"immutable output already exists: {report}")
    report.write_text(
        "\n".join(
            [
                "# Immutable recovered-policy V4R validation result",
                "",
                f"Protocol: `{PROTOCOL_ID}`",
                "",
                (
                    "- Raw mean incremental ramp impact: "
                    f"{validation['mean_incremental_ramp_impact']:.15g}"
                ),
                f"- DA energy cost ratio: {validation['energy_cost_ratio']:.12g}",
                f"- Every strict gate passed: **{selected}**",
                (
                    "- Failed gates: "
                    f"{', '.join(validation['success_gate']['failed_gates']) or '-'}"
                ),
                "- Sealed March-April test opened: **false**",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return result


def open_test(output: Path) -> dict[str, Any]:
    _require_canonical(output, TEST_OPENING)
    protocol = load_protocol()
    frozen = verify_source_freeze(protocol)
    decision = json.loads(VALIDATION_DECISION.read_text(encoding="utf-8"))
    _require(decision["selected"] is True, "V4R validation did not pass")
    _require(
        decision["validation_sha256"] == sha256_file(VALIDATION_RESULT),
        "validation decision does not bind the current result",
    )
    _require(not TEST_RESULT.exists(), "test result exists before opening record")
    for path in (
        VALIDATION_RUN_RECORD,
        VALIDATION_RESULT,
        VALIDATION_DECISION,
        VALIDATION_REPORT,
    ):
        _require_committed(path, "V4R validation artifact")
    result = {
        "schema_version": "ramp-pure-rl-recovered-v4r-sealed-test-opening-v1",
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": protocol["_sha256"],
        "source_bundle_sha256": frozen["source_bundle_sha256"],
        "validation_commit": _git("rev-parse", "HEAD"),
        "validation_decision_sha256": sha256_file(VALIDATION_DECISION),
        "validation_result_sha256": sha256_file(VALIDATION_RESULT),
        "condition_satisfied": True,
        "months_opened": ["2026-03", "2026-04"],
        "open_count": 1,
        "tuning_or_selection_permitted": False,
        "opened_at_utc": _now(),
    }
    _write_json(output, result)
    return result


def decide_test(output: Path) -> dict[str, Any]:
    _require_canonical(output, TEST_DECISION)
    protocol = load_protocol()
    verify_source_freeze(protocol)
    _require_committed(TEST_OPENING, "V4R sealed-test opening")
    test = json.loads(TEST_RESULT.read_text(encoding="utf-8"))
    _require(test["split"] == "test", "wrong sealed-test artifact")
    _require(
        test["protocol_id"] == PROTOCOL_ID
        and test["protocol_sha256"] == protocol["_sha256"],
        "sealed-test protocol binding is invalid",
    )
    _require(
        test["source_freeze_sha256"] == sha256_file(SOURCE_FREEZE)
        and test["recovery_binding_sha256"] == sha256_file(RECOVERY_BINDING),
        "sealed-test source or recovery binding is invalid",
    )
    _require(
        test["sealed_test_opening_sha256"] == sha256_file(TEST_OPENING),
        "sealed-test result does not bind the opening record",
    )
    _require(
        test["evaluation_opening_sha256"] == sha256_file(TEST_RUN_RECORD),
        "sealed-test evaluation-opening binding is invalid",
    )
    passed = bool(test["success_gate"]["passed"])
    result = {
        "schema_version": "ramp-pure-rl-recovered-v4r-test-decision-v1",
        "protocol_id": PROTOCOL_ID,
        "test_path": str(TEST_RESULT.relative_to(ROOT)),
        "test_sha256": sha256_file(TEST_RESULT),
        "sealed_test_passed": passed,
        "strict_gate": test["success_gate"],
        "post_selection_robustness_authorized": passed,
        "retuning_permitted": False,
        "decision": (
            "run-predeclared-post-selection-robustness"
            if passed
            else "publish-v4r-test-failure-without-retuning"
        ),
        "decided_at_utc": _now(),
    }
    _write_json(output, result)
    return result


def evaluate_robustness(variant: str, output: Path) -> dict[str, Any]:
    expected_output = {
        "one-gw-total": ROBUSTNESS_ONE_GW,
        "c-h-overlapping": ROBUSTNESS_C_H,
    }[variant]
    _require_canonical(output, expected_output)
    protocol = load_protocol()
    frozen = verify_source_freeze(protocol)
    test_decision = json.loads(TEST_DECISION.read_text(encoding="utf-8"))
    _require(
        test_decision["sealed_test_passed"] is True,
        "robustness requires passing V4R sealed test",
    )
    _require_committed(TEST_RESULT, "V4R sealed-test result")
    _require_committed(TEST_DECISION, "V4R sealed-test decision")
    _require_committed(TEST_RUN_RECORD, "V4R sealed-test evaluation opening")
    _require(
        test_decision["test_sha256"] == sha256_file(TEST_RESULT),
        "sealed-test decision does not bind the current test result",
    )
    factories = {
        "one-gw-total": make_one_gw_total_env,
        "c-h-overlapping": make_c_h_overlapping_env,
    }
    summary = evaluate_recovered_equal_action_ensemble(
        factory=factories[variant],
        bindings=protocol["frozen_bindings"]["members"],
        root=ROOT,
        split="test",
        seed=int(protocol["evaluation"]["environment_seed"]),
        windows=_factory_windows("test"),
    )
    result = {
        "schema_version": "ramp-pure-rl-recovered-v4r-robustness-v1",
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": protocol["_sha256"],
        "source_bundle_sha256": frozen["source_bundle_sha256"],
        "split": "test",
        "variant": variant,
        "post_selection_only": True,
        "selection_or_tuning": False,
        "run_count": 1,
        "test_result_sha256": sha256_file(TEST_RESULT),
        "test_decision_sha256": sha256_file(TEST_DECISION),
        "evaluated_at_utc": _now(),
        **summary,
    }
    _write_json(output, result)
    return result


def canonical(output: Path, report: Path) -> dict[str, Any]:
    _require_canonical(output, CANONICAL_EVIDENCE)
    _require_canonical(report, CANONICAL_REPORT)
    protocol = load_protocol()
    frozen = verify_source_freeze(protocol)
    validation = json.loads(VALIDATION_RESULT.read_text(encoding="utf-8"))
    validation_decision = json.loads(
        VALIDATION_DECISION.read_text(encoding="utf-8")
    )
    test_opened = TEST_OPENING.exists()
    test = (
        json.loads(TEST_RESULT.read_text(encoding="utf-8"))
        if TEST_RESULT.exists()
        else None
    )
    test_decision = (
        json.loads(TEST_DECISION.read_text(encoding="utf-8"))
        if TEST_DECISION.exists()
        else None
    )
    if validation_decision["selected"]:
        _require(
            test is not None and test_decision is not None,
            "selected V4R validation requires a sealed-test decision",
        )
    robustness = {}
    for name, path in (
        ("one_gw_total", ROBUSTNESS_ONE_GW),
        ("c_h_overlapping", ROBUSTNESS_C_H),
    ):
        if path.exists():
            robustness_result = json.loads(path.read_text(encoding="utf-8"))
            _require(
                robustness_result["test_result_sha256"]
                == sha256_file(TEST_RESULT)
                and robustness_result["test_decision_sha256"]
                == sha256_file(TEST_DECISION),
                f"{name} robustness hash chain is invalid",
            )
            _require(
                robustness_result["protocol_sha256"] == protocol["_sha256"]
                and robustness_result["source_bundle_sha256"]
                == frozen["source_bundle_sha256"],
                f"{name} robustness source binding is invalid",
            )
            robustness[name] = {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
                "result": robustness_result,
            }
    if test_decision and test_decision["sealed_test_passed"]:
        _require(
            set(robustness) == {"one_gw_total", "c_h_overlapping"},
            "passing sealed test requires both robustness results",
        )
    sealed_test_passed = bool(
        test_decision and test_decision["sealed_test_passed"]
    )
    result = {
        "schema_version": "ramp-pure-rl-recovered-v4r-canonical-evidence-v1",
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": _normalized_sha256(PROTOCOL_PATH),
        "source_freeze_sha256": sha256_file(SOURCE_FREEZE),
        "recovery_binding_sha256": sha256_file(RECOVERY_BINDING),
        "validation": {
            "path": str(VALIDATION_RESULT.relative_to(ROOT)),
            "sha256": sha256_file(VALIDATION_RESULT),
            "decision_path": str(VALIDATION_DECISION.relative_to(ROOT)),
            "decision_sha256": sha256_file(VALIDATION_DECISION),
            "result": validation,
        },
        "sealed_test_opened": test_opened,
        "sealed_test_open_count": 1 if test_opened else 0,
        "test": (
            {
                "path": str(TEST_RESULT.relative_to(ROOT)),
                "sha256": sha256_file(TEST_RESULT),
                "decision_path": str(TEST_DECISION.relative_to(ROOT)),
                "decision_sha256": sha256_file(TEST_DECISION),
                "result": test,
            }
            if test is not None and test_decision is not None
            else None
        ),
        "robustness": robustness,
        "blocked_original_v4_evaluated": False,
        "created_at_utc": _now(),
    }
    _write_json(output, result)
    _require(not report.exists(), f"immutable output already exists: {report}")
    report.write_text(
        "\n".join(
            [
                "# Recovered-policy equal-action ensemble V4R",
                "",
                f"Validation strict pass: **{validation_decision['selected']}**.",
                f"Sealed test opened exactly once: **{test_opened}**.",
                f"Sealed test strict pass: **{sealed_test_passed}**.",
                "",
                (
                    "V4R is a new binary identity. It does not claim original V3 "
                    "containers or the blocked V4 identity."
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    binding = subparsers.add_parser("bind-recovery")
    binding.add_argument("--source-output", type=Path, default=RECOVERY_SOURCE_REPORT)
    binding.add_argument("--output", type=Path, default=RECOVERY_BINDING)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--output", type=Path, default=SOURCE_FREEZE)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument(
        "--split", choices=("validation", "test"), required=True
    )
    evaluate_parser.add_argument("--output", type=Path, required=True)
    validation_parser = subparsers.add_parser("decide-validation")
    validation_parser.add_argument("--output", type=Path, default=VALIDATION_DECISION)
    validation_parser.add_argument("--report", type=Path, default=VALIDATION_REPORT)
    opening_parser = subparsers.add_parser("open-test")
    opening_parser.add_argument("--output", type=Path, default=TEST_OPENING)
    test_parser = subparsers.add_parser("decide-test")
    test_parser.add_argument("--output", type=Path, default=TEST_DECISION)
    robustness_parser = subparsers.add_parser("robustness")
    robustness_parser.add_argument(
        "--variant", choices=("one-gw-total", "c-h-overlapping"), required=True
    )
    robustness_parser.add_argument("--output", type=Path, required=True)
    canonical_parser = subparsers.add_parser("canonical")
    canonical_parser.add_argument("--output", type=Path, default=CANONICAL_EVIDENCE)
    canonical_parser.add_argument("--report", type=Path, default=CANONICAL_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "bind-recovery":
        result = bind_recovery(args.source_output, args.output)
    elif args.command == "freeze":
        result = freeze(args.output)
    elif args.command == "evaluate":
        result = evaluate(args.split, args.output)
    elif args.command == "decide-validation":
        result = decide_validation(args.output, args.report)
    elif args.command == "open-test":
        result = open_test(args.output)
    elif args.command == "decide-test":
        result = decide_test(args.output)
    elif args.command == "robustness":
        result = evaluate_robustness(args.variant, args.output)
    else:
        result = canonical(args.output, args.report)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
