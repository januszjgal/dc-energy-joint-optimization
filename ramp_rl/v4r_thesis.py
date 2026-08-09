"""Hash-gated V4R thesis evidence, claims, and publication tables."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANONICAL = (
    ROOT / "output" / "ramp_rl_v6" / "recovered_v4r" / "canonical_evidence.json"
)
DEFAULT_OUTPUT = ROOT / "output" / "ramp_rl_v6" / "recovered_v4r" / "thesis"

EXPECTED_CANONICAL_SHA256 = (
    "b1742a2e753d9a899be256667c80679cbfcf2da4cf6056a6b471d42e66ee7b30"
)
EXPECTED_PROTOCOL_ID = "v6-ramp-pure-rl-recovered-equal-action-ensemble-v4r"
EXPECTED_PROTOCOL_SHA256 = (
    "57310edca9e7b1e917be2901010352ad928d124beeaa5a10d21ae5fddd4f78dd"
)
EXPECTED_SOURCE_COMMIT = "46329fe765f596f84eeac71f061dcbd191a90583"
EXPECTED_SOURCE_BUNDLE_SHA256 = (
    "9c660175f345537636816e0278f427ec0f0bea162bf672bd9e8160efe157eb21"
)
EXPECTED_SOURCE_FREEZE_SHA256 = (
    "21e3b036be9e111a4af26ac60697e44993fbc681cd38d2b9ec2ad7c9d9848c02"
)
EXPECTED_RECOVERY_BINDING_SHA256 = (
    "2c125bce0de306aa6606942dba05d85f975153e33dffc41b5b6d8e4d258e98b4"
)
EXPECTED_TEST_OPENING_SHA256 = (
    "fd596881c6f83dc1bc0d77f238727ceb8d2a12b12c42b96a6d4061397c90f837"
)
EXPECTED_VALIDATION_DECISION_SHA256 = (
    "4d07c14122eeb40601e7a573fc9b8026ea9c347e7932c2cd0923ded7e38af5a1"
)
EXPECTED_TEST_DECISION_SHA256 = (
    "ad459855624705dbb0231f1e31595373ba2035659a2e62b2e5bc33aab05ef52e"
)
EXPECTED_MEMBER_SEEDS = (2801, 2802, 2803, 2804, 2805)
EXPECTED_MODEL_SHA256 = {
    2801: "47e68ae430be484da860b6e1520f1868e5869ccbd49d86fe48920a253c861883",
    2802: "93296e54bb61be72b9c41f58175860734208f1785089a11730236891f9a8700f",
    2803: "292a24819bb13dd8372b25c932eb3ac44b06bab891237ab331c7ba224873f507",
    2804: "ee5ee15ce884d7958da253ea7fb6f95297d395cb0ca5cc816d83ac72bb345fbb",
    2805: "ffba64a2b4574e1a10edfd96e91bcd08f78d380c756a4d3231ac5ee1365e07f6",
}
EXPECTED_MARKETS = (
    "CAISO_NP15",
    "ERCOT_LZ_NORTH",
    "ISONE_NEMA",
    "MISO_MINN_HUB",
    "NYISO_NYC_J",
    "SPP_NORTH_HUB",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"{path} must contain a JSON object")
    return payload


def _resolve_evidence_path(raw: str) -> Path:
    return ROOT / Path(raw.replace("\\", "/"))


def _verify_git_identity(path: Path) -> None:
    completed = subprocess.run(
        ["git", "show", f"7ebd9b2:{path.relative_to(ROOT).as_posix()}"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    _require(completed.returncode == 0, "canonical evidence commit 7ebd9b2 is missing")
    committed_hash = hashlib.sha256(completed.stdout).hexdigest()
    normalized_worktree_hash = hashlib.sha256(
        path.read_text(encoding="utf-8").encode("utf-8")
    ).hexdigest()
    _require(
        committed_hash == normalized_worktree_hash,
        "canonical evidence content differs from commit 7ebd9b2",
    )


def _verify_sidecar(
    path: Path,
    expected_sha256: str,
    *,
    label: str,
) -> dict[str, Any]:
    _require(path.is_file(), f"missing {label}: {path}")
    _require(sha256_file(path) == expected_sha256, f"wrong {label} SHA-256")
    return _load(path)


def _verify_controller(result: dict[str, Any], binding: dict[str, Any]) -> None:
    audit = result.get("controller_audit")
    _require(isinstance(audit, dict), "missing controller audit")
    _require(audit.get("member_seeds") == list(EXPECTED_MEMBER_SEEDS), "wrong member seeds")
    _require(audit.get("member_count") == 5, "V4R must invoke five members")
    _require(audit.get("weights") == [0.2] * 5, "V4R weights must be fixed and equal")
    _require(audit.get("member_selection_or_exclusion") is False, "member selection is prohibited")
    _require(audit.get("trainable_combiner") is False, "trainable combiner is prohibited")
    _require(
        audit.get("analytic_or_evaluation_actions_used_by_controller") is False,
        "analytic or evaluation actions are prohibited",
    )
    _require(audit.get("deterministic_member_actions") is True, "members must be deterministic")
    _require(
        audit.get("all_members_invoked_once_per_decision") is True,
        "every member must be invoked once per decision",
    )
    members = audit.get("members")
    _require(isinstance(members, list) and len(members) == 5, "missing member audit rows")
    binding_members = {
        int(member["seed"]): member for member in binding.get("members", [])
    }
    _require(set(binding_members) == set(EXPECTED_MEMBER_SEEDS), "recovery binding seed mismatch")
    for member in members:
        seed = int(member.get("seed", -1))
        _require(seed in EXPECTED_MODEL_SHA256, f"unsupported member seed: {seed}")
        _require(
            member.get("model_sha256") == EXPECTED_MODEL_SHA256[seed],
            f"wrong recovered model hash for seed {seed}",
        )
        _require(
            member.get("model_sha256") == binding_members[seed].get("recovered_model_sha256"),
            f"model hash is not recovery-bound for seed {seed}",
        )
        _require(member.get("weight") == 0.2, f"wrong member weight for seed {seed}")


def _verify_result(
    result: dict[str, Any],
    *,
    split: str,
    episode_count: int,
    binding: dict[str, Any],
    post_selection: bool = False,
) -> None:
    _require(result.get("protocol_id") == EXPECTED_PROTOCOL_ID, "wrong result protocol ID")
    _require(
        result.get("protocol_sha256") == EXPECTED_PROTOCOL_SHA256,
        "wrong result protocol SHA-256",
    )
    _require(result.get("source_bundle_sha256") == EXPECTED_SOURCE_BUNDLE_SHA256, "wrong source bundle")
    _require(result.get("split") == split, f"result split must be {split}")
    _require(result.get("episode_count") == episode_count, "wrong episode count")
    _require(result.get("run_count") == 1, "result must come from exactly one evaluation run")
    _require(result.get("future_leakage_detected") is False, "future leakage detected")
    _require(result.get("success_gate", {}).get("passed") is True, "strict gates did not all pass")
    _require(not result.get("success_gate", {}).get("failed_gates"), "failed gates are present")
    _require(result.get("service_unserved") == 0.0, "service must be exact")
    _require(result.get("batch_unfinished") == 0.0, "batch completion must be exact")
    _require(result.get("batch_expired") == 0.0, "batch expiry must be zero")
    _require(result.get("terminal_work") == 0.0, "terminal work must be zero")
    _require(result.get("certificate_violations") == 0, "certificate violations must be zero")
    _require(result.get("emergency_feasibility_rate") == 0.0, "emergency use must be zero")
    markets = result.get("per_market_macro")
    _require(isinstance(markets, dict), "missing per-market result")
    _require(set(markets) == set(EXPECTED_MARKETS), "missing or unsupported market")
    _require(all(float(markets[market]) < 0.0 for market in EXPECTED_MARKETS), "every market must improve")
    _require(float(result.get("mean_incremental_ramp_impact", 0.0)) < 0.0, "mean ramp must improve")
    _require(float(result.get("energy_cost_ratio", 2.0)) <= 1.02, "cost gate failed")
    if "recovery_binding_sha256" in result:
        _require(
            result["recovery_binding_sha256"] == EXPECTED_RECOVERY_BINDING_SHA256,
            "wrong recovery binding in result",
        )
    if post_selection:
        _require(result.get("post_selection_only") is True, "robustness must be post-selection")
        _require(result.get("selection_or_tuning") is False, "robustness used for selection or tuning")
    _verify_controller(result, binding)


def load_verified_evidence(path: Path = DEFAULT_CANONICAL) -> dict[str, Any]:
    path = path.resolve()
    _require(path.is_file(), f"missing canonical evidence: {path}")
    _require(sha256_file(path) == EXPECTED_CANONICAL_SHA256, "wrong canonical evidence SHA-256")
    _verify_git_identity(path)
    evidence = _load(path)
    _require(
        evidence.get("schema_version")
        == "ramp-pure-rl-recovered-v4r-canonical-evidence-v1",
        "unsupported canonical evidence schema",
    )
    _require(evidence.get("protocol_id") == EXPECTED_PROTOCOL_ID, "wrong canonical protocol ID")
    _require(
        evidence.get("protocol_sha256") == EXPECTED_PROTOCOL_SHA256,
        "wrong canonical protocol SHA-256",
    )
    _require(evidence.get("blocked_original_v4_evaluated") is False, "blocked V4 was evaluated")
    _require(evidence.get("sealed_test_opened") is True, "sealed test was not opened")
    _require(evidence.get("sealed_test_open_count") == 1, "sealed test must open exactly once")

    base = path.parent
    source_freeze = _verify_sidecar(
        base / "source_freeze.json",
        EXPECTED_SOURCE_FREEZE_SHA256,
        label="source freeze",
    )
    binding = _verify_sidecar(
        base / "recovery_binding.json",
        EXPECTED_RECOVERY_BINDING_SHA256,
        label="recovery binding",
    )
    opening = _verify_sidecar(
        base / "sealed_test_opening.json",
        EXPECTED_TEST_OPENING_SHA256,
        label="sealed-test opening",
    )
    _require(source_freeze.get("source_commit") == EXPECTED_SOURCE_COMMIT, "wrong source commit")
    _require(source_freeze.get("new_binary_identity") is True, "V4R must have a new binary identity")
    _require(source_freeze.get("blocked_original_v4_unchanged") is True, "blocked V4 changed")
    _require(source_freeze.get("v3_artifacts_unchanged") is True, "V3 artifacts changed")
    _require(source_freeze.get("test_opened") is False, "source freeze occurred after test opening")
    _require(binding.get("all_non_container_recovery_checks_exact") is True, "recovery is not exact")
    _require(binding.get("blocked_original_v4_identity_reused") is False, "blocked V4 identity reused")
    _require(binding.get("original_v3_artifact_reuse") is False, "original V3 containers were reused")
    _require(binding.get("performance_claim") is None, "recovery binding contains a performance claim")
    _require(opening.get("open_count") == 1, "sealed test opening count is not one")
    _require(opening.get("tuning_or_selection_permitted") is False, "test selection/tuning was permitted")
    _require(opening.get("condition_satisfied") is True, "test opening condition was not satisfied")
    _require(opening.get("months_opened") == ["2026-03", "2026-04"], "wrong test months")

    decisions: dict[str, dict[str, Any]] = {}
    for section, split, count in (
        ("validation", "validation", 28),
        ("test", "test", 60),
    ):
        record = evidence.get(section)
        _require(isinstance(record, dict), f"missing {section} evidence")
        expected_decision_hash = (
            EXPECTED_VALIDATION_DECISION_SHA256
            if section == "validation"
            else EXPECTED_TEST_DECISION_SHA256
        )
        _require(
            record.get("decision_sha256") == expected_decision_hash,
            f"wrong {section} decision hash in canonical evidence",
        )
        decisions[section] = _verify_sidecar(
            _resolve_evidence_path(str(record.get("decision_path", ""))),
            expected_decision_hash,
            label=f"{section} decision",
        )
        result_path = _resolve_evidence_path(str(record.get("path", "")))
        expected_hash = str(record.get("sha256", ""))
        persisted = _verify_sidecar(result_path, expected_hash, label=f"{section} result")
        _require(persisted == record.get("result"), f"{section} canonical embedding differs from source")
        _verify_result(persisted, split=split, episode_count=count, binding=binding)
    validation_decision = decisions["validation"]
    _require(validation_decision.get("selected") is True, "V4R was not selected on validation")
    _require(validation_decision.get("test_opened") is False, "validation decision opened test")
    _require(validation_decision.get("test_open_count") == 0, "test opened before validation freeze")
    _require(
        validation_decision.get("validation_sha256") == evidence["validation"]["sha256"],
        "validation decision is not bound to validation result",
    )
    _require(
        validation_decision.get("strict_gate", {}).get("test_tuning_prohibited") is True,
        "validation decision permits test tuning",
    )
    test_decision = decisions["test"]
    _require(test_decision.get("sealed_test_passed") is True, "test decision is not a strict pass")
    _require(test_decision.get("retuning_permitted") is False, "test decision permits retuning")
    _require(
        test_decision.get("post_selection_robustness_authorized") is True,
        "post-selection robustness was not authorized",
    )
    _require(
        test_decision.get("test_sha256") == evidence["test"]["sha256"],
        "test decision is not bound to test result",
    )

    for key in ("one_gw_total", "c_h_overlapping"):
        record = evidence.get("robustness", {}).get(key)
        _require(isinstance(record, dict), f"missing robustness result: {key}")
        persisted = _verify_sidecar(
            _resolve_evidence_path(str(record.get("path", ""))),
            str(record.get("sha256", "")),
            label=f"{key} robustness",
        )
        _require(persisted == record.get("result"), f"{key} canonical embedding differs from source")
        _verify_result(
            persisted,
            split="test",
            episode_count=60,
            binding=binding,
            post_selection=True,
        )
        _require(
            persisted.get("test_decision_sha256") == EXPECTED_TEST_DECISION_SHA256,
            f"{key} robustness is not bound to the test decision",
        )
        _require(
            persisted.get("test_result_sha256") == evidence["test"]["sha256"],
            f"{key} robustness is not bound to the sealed-test result",
        )
    _require(
        evidence["robustness"]["c_h_overlapping"]["result"].get("variant")
        == "c-h-overlapping",
        "c-h robustness must retain its overlapping identity",
    )
    evidence["_verified"] = {
        "canonical_sha256": EXPECTED_CANONICAL_SHA256,
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "source_freeze_sha256": EXPECTED_SOURCE_FREEZE_SHA256,
        "recovery_binding_sha256": EXPECTED_RECOVERY_BINDING_SHA256,
        "sealed_test_opening_sha256": EXPECTED_TEST_OPENING_SHA256,
        "validation_decision_sha256": EXPECTED_VALIDATION_DECISION_SHA256,
        "test_decision_sha256": EXPECTED_TEST_DECISION_SHA256,
    }
    return evidence


def build_claim_ledger(evidence: dict[str, Any]) -> dict[str, Any]:
    validation = evidence["validation"]["result"]
    test = evidence["test"]["result"]
    robustness = evidence["robustness"]
    claims = [
        {
            "claim_id": "v4r-recovery-identity",
            "value": "new-binary-identity-weight-equivalent-recovered-pure-rl",
            "evidence_pointer": "/recovery_binding_sha256",
        },
        {
            "claim_id": "validation-mean-ramp-impact",
            "value": validation["mean_incremental_ramp_impact"],
            "unit": "normalized incremental squared-ramp impact",
            "evidence_pointer": "/validation/result/mean_incremental_ramp_impact",
        },
        {
            "claim_id": "validation-cost-ratio",
            "value": validation["energy_cost_ratio"],
            "unit": "policy DA cost / status-quo DA cost",
            "evidence_pointer": "/validation/result/energy_cost_ratio",
        },
        {
            "claim_id": "test-mean-ramp-impact",
            "value": test["mean_incremental_ramp_impact"],
            "unit": "normalized incremental squared-ramp impact",
            "evidence_pointer": "/test/result/mean_incremental_ramp_impact",
        },
        {
            "claim_id": "test-cost-ratio",
            "value": test["energy_cost_ratio"],
            "unit": "policy DA cost / status-quo DA cost",
            "evidence_pointer": "/test/result/energy_cost_ratio",
        },
        {
            "claim_id": "sealed-test-opening-count",
            "value": evidence["sealed_test_open_count"],
            "unit": "openings",
            "evidence_pointer": "/sealed_test_open_count",
        },
        {
            "claim_id": "test-every-market-negative",
            "value": True,
            "evidence_pointer": "/test/result/per_market_macro",
        },
        {
            "claim_id": "test-exact-safety",
            "value": True,
            "evidence_pointer": "/test/result/success_gate/checks",
        },
        {
            "claim_id": "one-gw-total-robustness",
            "value": robustness["one_gw_total"]["result"]["mean_incremental_ramp_impact"],
            "unit": "normalized incremental squared-ramp impact",
            "evidence_pointer": "/robustness/one_gw_total/result",
        },
        {
            "claim_id": "c-h-overlapping-robustness",
            "value": robustness["c_h_overlapping"]["result"]["mean_incremental_ramp_impact"],
            "unit": "normalized incremental squared-ramp impact",
            "scope": "overlapping/non-independent robustness",
            "evidence_pointer": "/robustness/c_h_overlapping/result",
        },
    ]
    for market in EXPECTED_MARKETS:
        claims.append(
            {
                "claim_id": f"test-market-{market.lower()}",
                "value": test["per_market_macro"][market],
                "unit": "normalized incremental squared-ramp impact",
                "evidence_pointer": f"/test/result/per_market_macro/{market}",
            }
        )
    return {
        "schema_version": "ramp-v6-v4r-claim-ledger-v1",
        "canonical_evidence_sha256": EXPECTED_CANONICAL_SHA256,
        "protocol_id": EXPECTED_PROTOCOL_ID,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "recovery_binding_sha256": EXPECTED_RECOVERY_BINDING_SHA256,
        "claims": claims,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    _require(bool(rows), f"cannot write empty table: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_publication_package(
    evidence: dict[str, Any],
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger = build_claim_ledger(evidence)
    ledger_path = output_dir / "claim_ledger.json"
    ledger_path.write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    validation = evidence["validation"]["result"]
    test = evidence["test"]["result"]
    robustness = evidence["robustness"]
    _write_csv(
        output_dir / "split_summary.csv",
        [
            {
                "split": label,
                "episodes": result["episode_count"],
                "mean_incremental_ramp_impact": result["mean_incremental_ramp_impact"],
                "energy_cost_ratio": result["energy_cost_ratio"],
                "all_strict_gates_pass": result["success_gate"]["passed"],
            }
            for label, result in (("validation", validation), ("sealed_test", test))
        ],
    )
    _write_csv(
        output_dir / "per_market_test.csv",
        [
            {
                "market": market,
                "mean_incremental_ramp_impact": test["per_market_macro"][market],
            }
            for market in EXPECTED_MARKETS
        ],
    )
    _write_csv(
        output_dir / "physical_ramps.csv",
        [
            {
                "split": label,
                "h1_adjusted_p95_fraction_scale_per_hour": result["ramp_h1_adjusted_p95"],
                "h1_adjusted_max_fraction_scale_per_hour": result["ramp_h1_adjusted_max"],
                "h3_adjusted_p95_fraction_scale_per_hour": result["ramp_h3_adjusted_p95"],
                "h3_adjusted_max_fraction_scale_per_hour": result["ramp_h3_adjusted_max"],
            }
            for label, result in (("validation", validation), ("sealed_test", test))
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
            for label, result in (("validation", validation), ("sealed_test", test))
        ],
    )
    _write_csv(
        output_dir / "robustness.csv",
        [
            {
                "variant": key,
                "scope": (
                    "post-selection scale sensitivity"
                    if key == "one_gw_total"
                    else "post-selection overlapping/non-independent robustness"
                ),
                "mean_incremental_ramp_impact": record["result"]["mean_incremental_ramp_impact"],
                "energy_cost_ratio": record["result"]["energy_cost_ratio"],
                "every_market_negative": all(
                    value < 0 for value in record["result"]["per_market_macro"].values()
                ),
                "all_strict_gates_pass": record["result"]["success_gate"]["passed"],
            }
            for key, record in robustness.items()
        ],
    )
    manifest = {
        "schema_version": "ramp-v6-v4r-thesis-package-v1",
        "canonical_evidence_sha256": EXPECTED_CANONICAL_SHA256,
        "claim_ledger_sha256": sha256_file(ledger_path),
        "files": {},
    }
    for path in sorted(output_dir.glob("*")):
        if path.name == "package_manifest.json" or not path.is_file():
            continue
        manifest["files"][path.name] = sha256_file(path)
    manifest_path = output_dir / "package_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest
