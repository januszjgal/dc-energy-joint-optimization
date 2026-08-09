"""Freeze and execute the immutable pure-RL equal-action ensemble v4."""

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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from env.ramp_v6.factory import make_energy_model_v3_env  # noqa: E402
from ramp_rl.ensemble import (  # noqa: E402
    EXPECTED_MEMBER_SEEDS,
    evaluate_equal_action_ensemble,
)
from ramp_rl.ensemble_robustness import (  # noqa: E402
    make_c_h_overlapping_env,
    make_one_gw_total_env,
)
from ramp_rl.evidence import sha256_file  # noqa: E402


PROTOCOL_PATH = ROOT / "env" / "protocols" / "v6_pure_ramp_rl_v4.yaml"
FACTORY_MANIFEST = (
    ROOT / "output" / "energy_model_v3" / "ramp_v6" / "factory_manifest.json"
)
V4_ROOT = ROOT / "output" / "ramp_rl_v6" / "live_v4"
SOURCE_FREEZE = V4_ROOT / "source_freeze.json"
VALIDATION_RESULT = V4_ROOT / "validation" / "ensemble_validation.json"
VALIDATION_DECISION = V4_ROOT / "validation_decision.json"
VALIDATION_REPORT = V4_ROOT / "validation_report.md"
TEST_OPENING = V4_ROOT / "sealed_test_opening.json"
TEST_RESULT = V4_ROOT / "test" / "ensemble_test.json"
TEST_DECISION = V4_ROOT / "test_decision.json"
ROBUSTNESS_ONE_GW = V4_ROOT / "robustness" / "one_gw_total_test.json"
ROBUSTNESS_C_H = V4_ROOT / "robustness" / "c_h_overlapping_test.json"
CANONICAL_EVIDENCE = V4_ROOT / "canonical_evidence.json"
CANONICAL_REPORT = V4_ROOT / "final_report.md"
SOURCE_PATHS = (
    "env/protocols/v6_pure_ramp_rl_v4.yaml",
    "ramp_rl/ensemble.py",
    "ramp_rl/ensemble_robustness.py",
    "scripts/run_ramp_rl_v4.py",
    "tests/ramp_v6/test_v4_ensemble.py",
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


def _normalized_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _require(not path.exists(), f"immutable output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _require_canonical_output(path: Path, expected: Path) -> None:
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
        f"{label} does not match the committed blob: {relative}",
    )


def load_protocol() -> dict[str, Any]:
    payload = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    _require(
        payload["protocol"]["id"]
        == "v6-ramp-pure-rl-equal-action-ensemble-v4",
        "unexpected v4 protocol ID",
    )
    controller = payload["controller"]
    _require(
        controller["member_seeds"] == list(EXPECTED_MEMBER_SEEDS),
        "v4 member seeds changed",
    )
    _require(controller["weights"] == [0.2] * 5, "v4 weights are not equal")
    _require(controller["action_space"] == "environment", "wrong action space")
    _require(controller["action_dimensions"] == 13, "wrong action dimension")
    _require(controller["action_bounds"] == [-6.0, 6.0], "wrong action bounds")
    for key in (
        "trainable_combiner",
        "member_selection",
        "member_exclusion",
        "teacher",
        "behavior_cloning",
        "demonstrations",
        "mpc_actions",
        "optimizer_actions",
        "analytic_actions",
        "evaluation_actions",
    ):
        _require(controller[key] is False, f"forbidden controller role: {key}")
    _require(
        payload["selection_basis"]["split"] == "validation"
        and payload["selection_basis"]["test_used_for_selection"] is False,
        "selection is not validation-only",
    )
    _require(
        payload["success_gate"]["macro_raw_incremental_ramp_impact_lt"] == 0.0
        and payload["success_gate"][
            "every_market_raw_incremental_ramp_impact_lt"
        ]
        == 0.0
        and payload["success_gate"]["post_hoc_tolerance"] is False,
        "strict ramp gates changed",
    )
    payload["_path"] = str(PROTOCOL_PATH.relative_to(ROOT))
    payload["_sha256"] = _normalized_sha256(PROTOCOL_PATH)
    return payload


def _verify_file(path: Path, expected: str, label: str) -> None:
    _require(path.is_file(), f"missing {label}: {path}")
    _require(sha256_file(path) == expected, f"{label} hash mismatch: {path}")


def verify_frozen_bindings(
    protocol: dict[str, Any], *, require_binaries: bool = True
) -> None:
    bindings = protocol["frozen_bindings"]
    _verify_file(
        ROOT / bindings["v3_protocol"]["path"],
        bindings["v3_protocol"]["file_sha256"],
        "v3 protocol",
    )
    _verify_file(
        ROOT / bindings["v3_source_freeze"]["path"],
        bindings["v3_source_freeze"]["sha256"],
        "v3 source freeze",
    )
    _verify_file(
        ROOT / bindings["v3_validation"]["decision_path"],
        bindings["v3_validation"]["decision_sha256"],
        "v3 validation decision",
    )
    _verify_file(
        ROOT / bindings["v3_validation"]["command_manifest_path"],
        bindings["v3_validation"]["command_manifest_sha256"],
        "v3 command manifest",
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
    for member in bindings["members"]:
        _verify_file(
            ROOT / member["training_manifest_path"],
            member["training_manifest_sha256"],
            f"seed {member['seed']} training manifest",
        )
        if require_binaries:
            _verify_file(
                ROOT / member["model_path"],
                member["model_sha256"],
                f"seed {member['seed']} model",
            )
            _verify_file(
                ROOT / member["vecnormalize_path"],
                member["vecnormalize_sha256"],
                f"seed {member['seed']} normalization",
            )
    robustness_bindings = protocol["post_selection"]["c_h_overlapping"][
        "input_bindings"
    ]
    for cell in ("c", "d", "e", "f", "g", "h"):
        _verify_file(
            ROOT / "data" / "cells" / f"cell_{cell}_tiers.csv",
            robustness_bindings[f"cell_{cell}_tiers"],
            f"cell {cell} tier robustness input",
        )
        _verify_file(
            ROOT / "data" / "jobs" / f"batch_distributions_{cell}.json",
            robustness_bindings[f"batch_distributions_{cell}"],
            f"cell {cell} deadline robustness input",
        )


def _source_hashes() -> dict[str, str]:
    return {relative: sha256_file(ROOT / relative) for relative in SOURCE_PATHS}


def _source_bundle_sha256(hashes: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for relative in SOURCE_PATHS:
        digest.update(relative.encode("utf-8"))
        digest.update(hashes[relative].encode("ascii"))
    return digest.hexdigest()


def freeze(output: Path) -> dict[str, Any]:
    _require_canonical_output(output, SOURCE_FREEZE)
    protocol = load_protocol()
    verify_frozen_bindings(protocol)
    _require(
        not _git("status", "--porcelain", "--", *SOURCE_PATHS),
        "v4 source/protocol must be committed before freezing",
    )
    for relative in SOURCE_PATHS:
        _git("ls-files", "--error-unmatch", relative)
    for relative in V3_IMMUTABLE_PATHS:
        current = _git("rev-parse", f"HEAD:{relative}")
        predecessor = _git("rev-parse", f"1f2a7dc:{relative}")
        _require(current == predecessor, f"immutable v3 artifact changed: {relative}")
    _require(
        not VALIDATION_RESULT.exists()
        and not VALIDATION_DECISION.exists()
        and not TEST_OPENING.exists()
        and not TEST_RESULT.exists(),
        "evaluation evidence exists before v4 source freeze",
    )
    source_hashes = _source_hashes()
    result = {
        "schema_version": "ramp-pure-rl-v4-source-freeze-v1",
        "protocol_id": protocol["protocol"]["id"],
        "protocol_path": protocol["_path"],
        "protocol_sha256": protocol["_sha256"],
        "source_commit": _git("rev-parse", "HEAD"),
        "source_files": source_hashes,
        "source_bundle_sha256": _source_bundle_sha256(source_hashes),
        "v3_predecessor_commit": "1f2a7dc",
        "v3_artifacts_unchanged": True,
        "frozen_bindings": protocol["frozen_bindings"],
        "test_opened": False,
        "created_at_utc": _now(),
    }
    _write_json(output, result)
    return result


def verify_source_freeze(protocol: dict[str, Any]) -> dict[str, Any]:
    _require_committed(SOURCE_FREEZE, "v4 source freeze")
    frozen = json.loads(SOURCE_FREEZE.read_text(encoding="utf-8"))
    _require(frozen["protocol_sha256"] == protocol["_sha256"], "protocol changed")
    hashes = _source_hashes()
    _require(hashes == frozen["source_files"], "v4 source changed after freeze")
    _require(
        _source_bundle_sha256(hashes) == frozen["source_bundle_sha256"],
        "v4 source bundle changed after freeze",
    )
    verify_frozen_bindings(protocol)
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
    _require_canonical_output(output, expected_output)
    protocol = load_protocol()
    frozen = verify_source_freeze(protocol)
    if split == "validation":
        _require(
            not TEST_OPENING.exists() and not TEST_RESULT.exists(),
            "sealed test was opened before validation",
        )
        factory = make_energy_model_v3_env
        variant = "primary"
    else:
        opening = json.loads(TEST_OPENING.read_text(encoding="utf-8"))
        _require(opening["open_count"] == 1, "sealed test opening is invalid")
        _require_committed(TEST_OPENING, "sealed-test opening record")
        decision = json.loads(VALIDATION_DECISION.read_text(encoding="utf-8"))
        _require(decision["selected"] is True, "sealed test requires passing frozen validation")
        _require(
            opening["validation_decision_sha256"]
            == sha256_file(VALIDATION_DECISION)
            and opening["validation_result_sha256"] == sha256_file(VALIDATION_RESULT),
            "sealed-test opening hash chain is invalid",
        )
        _require(
            opening["protocol_sha256"] == protocol["_sha256"]
            and opening["source_bundle_sha256"] == frozen["source_bundle_sha256"],
            "sealed-test opening source binding is invalid",
        )
        factory = make_energy_model_v3_env
        variant = "primary"
    windows = _factory_windows(split)
    summary = evaluate_equal_action_ensemble(
        factory=factory,
        bindings=protocol["frozen_bindings"]["members"],
        root=ROOT,
        split=split,
        seed=int(protocol["evaluation"]["environment_seed"]),
        windows=windows,
    )
    result = {
        "schema_version": "ramp-pure-rl-v4-evaluation-v1",
        "protocol_id": protocol["protocol"]["id"],
        "protocol_sha256": protocol["_sha256"],
        "source_freeze_sha256": sha256_file(SOURCE_FREEZE),
        "source_bundle_sha256": frozen["source_bundle_sha256"],
        "factory_manifest_sha256": sha256_file(FACTORY_MANIFEST),
        "split": split,
        "variant": variant,
        "run_count": 1,
        "environment_seed": int(protocol["evaluation"]["environment_seed"]),
        "windows": windows,
        "window_count": len(windows),
        "evaluated_at_utc": _now(),
        **summary,
    }
    if split == "test":
        result["sealed_test_opening_path"] = str(TEST_OPENING.relative_to(ROOT))
        result["sealed_test_opening_sha256"] = sha256_file(TEST_OPENING)
        result["validation_result_sha256"] = sha256_file(VALIDATION_RESULT)
        result["validation_decision_sha256"] = sha256_file(VALIDATION_DECISION)
    _write_json(output, result)
    return result


def decide_validation(output: Path, report: Path) -> dict[str, Any]:
    _require_canonical_output(output, VALIDATION_DECISION)
    _require_canonical_output(report, VALIDATION_REPORT)
    protocol = load_protocol()
    verify_source_freeze(protocol)
    validation = json.loads(VALIDATION_RESULT.read_text(encoding="utf-8"))
    _require(validation["split"] == "validation", "wrong validation artifact")
    selected = bool(validation["success_gate"]["passed"])
    result = {
        "schema_version": "ramp-pure-rl-v4-validation-decision-v1",
        "protocol_id": protocol["protocol"]["id"],
        "protocol_sha256": protocol["_sha256"],
        "validation_path": str(VALIDATION_RESULT.relative_to(ROOT)),
        "validation_sha256": sha256_file(VALIDATION_RESULT),
        "selected": selected,
        "strict_gate": validation["success_gate"],
        "test_opened": False,
        "test_open_count": 0,
        "decision": (
            "select-v4-ensemble-and-open-sealed-test-once"
            if selected
            else "publish-immutable-v4-failure-and-keep-test-sealed"
        ),
        "decided_at_utc": _now(),
    }
    _write_json(output, result)
    _require(not report.exists(), f"immutable output already exists: {report}")
    report.write_text(
        "\n".join(
            [
                "# Immutable pure-RL v4 validation result",
                "",
                f"Protocol: `{protocol['protocol']['id']}`",
                "",
                f"- Raw mean incremental ramp impact: {validation['mean_incremental_ramp_impact']:.15g}",
                f"- DA energy cost ratio: {validation['energy_cost_ratio']:.12g}",
                f"- Every strict gate passed: **{selected}**",
                f"- Failed gates: {', '.join(validation['success_gate']['failed_gates']) or '-'}",
                "- Sealed March-April test opened: **false**",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return result


def open_test(output: Path) -> dict[str, Any]:
    _require_canonical_output(output, TEST_OPENING)
    protocol = load_protocol()
    frozen = verify_source_freeze(protocol)
    decision = json.loads(VALIDATION_DECISION.read_text(encoding="utf-8"))
    _require(decision["selected"] is True, "v4 validation did not pass")
    _require(
        decision["validation_sha256"] == sha256_file(VALIDATION_RESULT),
        "validation decision does not bind the current validation result",
    )
    _require(not TEST_RESULT.exists(), "test result exists before opening record")
    for path in (VALIDATION_RESULT, VALIDATION_DECISION, VALIDATION_REPORT):
        _require_committed(path, "validation artifact")
    result = {
        "schema_version": "ramp-pure-rl-v4-sealed-test-opening-v1",
        "protocol_id": protocol["protocol"]["id"],
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
    _require_canonical_output(output, TEST_DECISION)
    protocol = load_protocol()
    frozen = verify_source_freeze(protocol)
    _require_committed(TEST_OPENING, "sealed-test opening record")
    test = json.loads(TEST_RESULT.read_text(encoding="utf-8"))
    _require(test["split"] == "test", "wrong sealed-test artifact")
    _require(
        test["protocol_sha256"] == protocol["_sha256"]
        and test["source_bundle_sha256"] == frozen["source_bundle_sha256"],
        "sealed-test artifact source binding is invalid",
    )
    _require(
        test["sealed_test_opening_sha256"] == sha256_file(TEST_OPENING),
        "sealed-test result does not bind the committed opening record",
    )
    passed = bool(test["success_gate"]["passed"])
    result = {
        "schema_version": "ramp-pure-rl-v4-test-decision-v1",
        "protocol_id": protocol["protocol"]["id"],
        "test_path": str(TEST_RESULT.relative_to(ROOT)),
        "test_sha256": sha256_file(TEST_RESULT),
        "sealed_test_passed": passed,
        "strict_gate": test["success_gate"],
        "post_selection_robustness_authorized": passed,
        "retuning_permitted": False,
        "decision": (
            "run-predeclared-post-selection-robustness"
            if passed
            else "publish-v4-test-failure-without-retuning"
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
    _require_canonical_output(output, expected_output)
    protocol = load_protocol()
    frozen = verify_source_freeze(protocol)
    test_decision = json.loads(TEST_DECISION.read_text(encoding="utf-8"))
    _require(
        test_decision["sealed_test_passed"] is True,
        "robustness requires passing sealed test",
    )
    _require_committed(TEST_RESULT, "sealed-test result")
    _require_committed(TEST_DECISION, "sealed-test decision")
    _require(
        test_decision["test_sha256"] == sha256_file(TEST_RESULT),
        "sealed-test decision does not bind the current test result",
    )
    factories = {
        "one-gw-total": make_one_gw_total_env,
        "c-h-overlapping": make_c_h_overlapping_env,
    }
    summary = evaluate_equal_action_ensemble(
        factory=factories[variant],
        bindings=protocol["frozen_bindings"]["members"],
        root=ROOT,
        split="test",
        seed=int(protocol["evaluation"]["environment_seed"]),
        windows=_factory_windows("test"),
    )
    result = {
        "schema_version": "ramp-pure-rl-v4-robustness-v1",
        "protocol_id": protocol["protocol"]["id"],
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
    _require_canonical_output(output, CANONICAL_EVIDENCE)
    _require_canonical_output(report, CANONICAL_REPORT)
    protocol = load_protocol()
    validation = json.loads(VALIDATION_RESULT.read_text(encoding="utf-8"))
    validation_decision = json.loads(
        VALIDATION_DECISION.read_text(encoding="utf-8")
    )
    test_opened = TEST_OPENING.exists()
    test = json.loads(TEST_RESULT.read_text(encoding="utf-8")) if TEST_RESULT.exists() else None
    test_decision = (
        json.loads(TEST_DECISION.read_text(encoding="utf-8"))
        if TEST_DECISION.exists()
        else None
    )
    if validation_decision["selected"]:
        _require(
            test is not None and test_decision is not None,
            "selected validation requires a completed sealed-test decision",
        )
    if test_decision and test_decision["sealed_test_passed"]:
        _require(
            ROBUSTNESS_ONE_GW.exists() and ROBUSTNESS_C_H.exists(),
            "passing sealed test requires both robustness artifacts",
        )
    robustness = {}
    for name, path in (
        ("one_gw_total", ROBUSTNESS_ONE_GW),
        ("c_h_overlapping", ROBUSTNESS_C_H),
    ):
        if path.exists():
            robustness_result = json.loads(path.read_text(encoding="utf-8"))
            _require(
                robustness_result["test_result_sha256"] == sha256_file(TEST_RESULT)
                and robustness_result["test_decision_sha256"]
                == sha256_file(TEST_DECISION),
                f"{name} robustness hash chain is invalid",
            )
            robustness[name] = {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
                "result": robustness_result,
            }
    thesis_success = bool(
        validation_decision["selected"]
        and test_decision
        and test_decision["sealed_test_passed"]
        and set(robustness) == {"one_gw_total", "c_h_overlapping"}
        and all(
            row["result"]["success_gate"]["passed"]
            for row in robustness.values()
        )
    )
    result = {
        "schema_version": "ramp-pure-rl-v4-canonical-evidence-v1",
        "protocol_id": protocol["protocol"]["id"],
        "protocol_sha256": protocol["_sha256"],
        "source_freeze": {
            "path": str(SOURCE_FREEZE.relative_to(ROOT)),
            "sha256": sha256_file(SOURCE_FREEZE),
        },
        "validation": {
            "path": str(VALIDATION_RESULT.relative_to(ROOT)),
            "sha256": sha256_file(VALIDATION_RESULT),
            "decision_path": str(VALIDATION_DECISION.relative_to(ROOT)),
            "decision_sha256": sha256_file(VALIDATION_DECISION),
            "metrics": {
                key: validation[key]
                for key in (
                    "mean_incremental_ramp_impact",
                    "per_market_macro",
                    "energy_cost_ratio",
                    "emergency_feasibility_rate",
                    "behavior_audit",
                    "success_gate",
                )
            },
        },
        "sealed_test_opened": test_opened,
        "sealed_test_open_count": 1 if test_opened else 0,
        "test": (
            {
                "path": str(TEST_RESULT.relative_to(ROOT)),
                "sha256": sha256_file(TEST_RESULT),
                "decision_path": str(TEST_DECISION.relative_to(ROOT)),
                "decision_sha256": sha256_file(TEST_DECISION),
                "metrics": {
                    key: test[key]
                    for key in (
                        "mean_incremental_ramp_impact",
                        "per_market_macro",
                        "energy_cost_ratio",
                        "emergency_feasibility_rate",
                        "behavior_audit",
                        "success_gate",
                    )
                },
            }
            if test is not None and test_decision is not None
            else None
        ),
        "robustness": robustness,
        "thesis_success_established": thesis_success,
        "created_at_utc": _now(),
    }
    _write_json(output, result)
    _require(not report.exists(), f"immutable output already exists: {report}")
    lines = [
        "# Pure-RL equal-action ensemble v4",
        "",
        f"Validation strict pass: **{validation_decision['selected']}**.",
        f"Sealed test opened exactly once: **{test_opened}**.",
        f"Sealed test strict pass: **{bool(test_decision and test_decision['sealed_test_passed'])}**.",
        f"Thesis success established: **{thesis_success}**.",
        "",
        "No gate tolerance, member exclusion, post-hoc weighting, retraining, or test tuning was used.",
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--output", type=Path, default=SOURCE_FREEZE)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--split", choices=("validation", "test"), required=True)
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
    if args.command == "freeze":
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
