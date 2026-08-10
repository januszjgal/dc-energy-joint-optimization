"""Hash-gated V4R thesis evidence, claims, and publication tables."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from ramp_rl.provenance import (
    CANONICAL_JSON_REPRESENTATION,
    HASH_CONTRACT_ID,
    canonical_json_file_sha256,
    canonical_json_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANONICAL = (
    ROOT
    / "output"
    / "ramp_rl_v6"
    / "recovered_v4r_resealed_v2"
    / "canonical_evidence.json"
)
DEFAULT_OUTPUT = (
    ROOT / "output" / "ramp_rl_v6" / "recovered_v4r_resealed_v2" / "thesis"
)

EXPECTED_CANONICAL_SHA256 = (
    "f642bd5868abdd9f7cda2a6fffb228250f3570fd0c6d440085da68c976892d9b"
)
EXPECTED_RESEAL_COMMIT = "3097c9aeca248a12e7239b40a895c5cd5fcd037b"
EXPECTED_PROTOCOL_ID = "v6-ramp-pure-rl-recovered-equal-action-ensemble-v4r"
EXPECTED_PROTOCOL_SHA256 = (
    "57310edca9e7b1e917be2901010352ad928d124beeaa5a10d21ae5fddd4f78dd"
)
EXPECTED_SOURCE_COMMIT = "46329fe765f596f84eeac71f061dcbd191a90583"
EXPECTED_SOURCE_BUNDLE_SHA256 = (
    "9c660175f345537636816e0278f427ec0f0bea162bf672bd9e8160efe157eb21"
)
EXPECTED_SOURCE_FREEZE_SHA256 = (
    "04d358ff4b7629e49eec92fd2ca10bc25ca38fa3f1116d7fd66c3763b8a08973"
)
EXPECTED_RECOVERY_BINDING_SHA256 = (
    "ecbeb41753627231a0d601eedb1a5b1906879dd80b5ccb248028cfc8256005e7"
)
EXPECTED_VALIDATION_CHAIN_SHA256 = (
    "30e987b704d7af6aee3a9b7a4b630ed2ccfb6c3b33439fb608be278582fd3849"
)
EXPECTED_SEALED_TEST_CHAIN_SHA256 = (
    "af05566bc43afb50eea3ac0f7f186bb4bb192d8c16a48241ca758c35a23a98db"
)
EXPECTED_ROBUSTNESS_CHAIN_SHA256 = (
    "3ae009ae74ee54b7f5ff41e675c1c4f283066f73b9a02c13bd8e332e03ec29b0"
)
EXPECTED_MIGRATION_MAP_SHA256 = (
    "66e89036a30df393ad5cc050af0093bc2cd12bd6b3d073241731548752c070e8"
)
EXPECTED_SUPERSEDED_CANONICAL_SHA256 = (
    "e6d141ba713f1512e8eaf400fcb0ae59dd5a423c5c917290d7bd4c51b6b8bbc3"
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
    relative = path.relative_to(ROOT).as_posix()
    completed = subprocess.run(
        ["git", "show", f"{EXPECTED_RESEAL_COMMIT}:{relative}"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    _require(
        completed.returncode == 0,
        f"corrected canonical evidence is missing from {EXPECTED_RESEAL_COMMIT}",
    )
    committed_payload = json.loads(completed.stdout.decode("utf-8"))
    committed_hash = canonical_json_sha256(committed_payload)
    worktree_hash = canonical_json_file_sha256(path)
    _require(
        committed_hash == worktree_hash == EXPECTED_CANONICAL_SHA256,
        "corrected canonical evidence differs from its committed canonical-JSON identity",
    )


def _verify_reference(
    reference: dict[str, Any],
    *,
    expected_sha256: str | None = None,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    _require(reference.get("algorithm") == "sha256", f"wrong {label} algorithm")
    _require(
        reference.get("representation") == CANONICAL_JSON_REPRESENTATION,
        f"wrong {label} representation",
    )
    path = _resolve_evidence_path(str(reference.get("path", "")))
    expected = expected_sha256 or str(reference.get("sha256", ""))
    _require(reference.get("sha256") == expected, f"wrong {label} reference SHA-256")
    return path, _verify_sidecar(path, expected, label=label)


def _verify_sidecar(
    path: Path,
    expected_sha256: str,
    *,
    label: str,
) -> dict[str, Any]:
    _require(path.is_file(), f"missing {label}: {path}")
    _require(
        canonical_json_file_sha256(path) == expected_sha256,
        f"wrong {label} canonical-JSON SHA-256",
    )
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
            member.get("model_sha256")
            == binding_members[seed].get("recovered_model", {}).get("sha256"),
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
    if post_selection:
        _require(result.get("post_selection_only") is True, "robustness must be post-selection")
        _require(result.get("selection_or_tuning") is False, "robustness used for selection or tuning")
    _verify_controller(result, binding)


def load_verified_evidence(path: Path = DEFAULT_CANONICAL) -> dict[str, Any]:
    path = path.resolve()
    _require(path.is_file(), f"missing canonical evidence: {path}")
    _require(
        canonical_json_file_sha256(path) == EXPECTED_CANONICAL_SHA256,
        "wrong canonical evidence canonical-JSON SHA-256",
    )
    _verify_git_identity(path)
    canonical = _load(path)
    _require(
        canonical.get("schema_version")
        == "ramp-pure-rl-recovered-v4r-canonical-evidence-v2",
        "unsupported canonical evidence schema",
    )
    _require(canonical.get("protocol_id") == EXPECTED_PROTOCOL_ID, "wrong canonical protocol ID")
    _require(
        canonical.get("status") == "canonical-corrected-provenance",
        "canonical evidence is not the corrected provenance package",
    )
    _require(
        canonical.get("supersession_scope") == "provenance-hash-chain-only",
        "wrong provenance supersession scope",
    )
    _require(canonical.get("numerical_results_changed") is False, "numerical results changed")
    _require(
        canonical.get("protocol_controller_model_data_factory_forecast_identities_changed")
        is False,
        "experimental identities changed",
    )
    _require(canonical.get("validation_or_test_evaluation_rerun") is False, "evaluation reran")
    _require(canonical.get("training_or_retuning_performed") is False, "training or retuning ran")
    _require(canonical.get("sealed_test_open_count") == 1, "sealed test must open exactly once")

    _, hash_contract = _verify_reference(
        canonical["hash_contract"],
        label="hash contract",
    )
    _require(hash_contract.get("contract_id") == HASH_CONTRACT_ID, "wrong hash contract ID")
    _, migration_map = _verify_reference(
        canonical["migration_map"],
        expected_sha256=EXPECTED_MIGRATION_MAP_SHA256,
        label="migration map",
    )
    superseded_path, superseded = _verify_reference(
        canonical["supersedes"],
        expected_sha256=EXPECTED_SUPERSEDED_CANONICAL_SHA256,
        label="superseded canonical evidence",
    )
    _require(
        migration_map.get("schema_version")
        == "ramp-pure-rl-recovered-v4r-hash-migration-v2",
        "wrong migration map schema",
    )
    _require(migration_map.get("contract_id") == HASH_CONTRACT_ID, "wrong migration contract ID")
    _require(migration_map.get("status") == "complete", "migration map is incomplete")
    _require(
        migration_map.get("experimental_results_changed") is False,
        "migration changed experimental results",
    )
    legacy_terminal = migration_map.get("legacy_terminal_evidence", {})
    _require(
        legacy_terminal.get("artifact")
        == superseded_path.relative_to(ROOT).as_posix(),
        "migration map superseded artifact mismatch",
    )
    _require(
        legacy_terminal.get("new_target")
        == DEFAULT_CANONICAL.relative_to(ROOT).as_posix(),
        "migration map corrected target mismatch",
    )
    _require(
        superseded.get("schema_version")
        == "ramp-pure-rl-recovered-v4r-canonical-evidence-v1",
        "wrong superseded evidence schema",
    )
    _require(
        superseded.get("protocol_id") == EXPECTED_PROTOCOL_ID,
        "superseded evidence protocol mismatch",
    )
    _require(
        superseded.get("sealed_test_open_count") == 1,
        "superseded evidence has wrong test opening count",
    )
    _require(
        superseded.get("blocked_original_v4_evaluated") is False,
        "superseded evidence changed blocked V4 status",
    )

    _, source_freeze = _verify_reference(
        canonical["source_freeze"],
        expected_sha256=EXPECTED_SOURCE_FREEZE_SHA256,
        label="source freeze",
    )
    _, binding = _verify_reference(
        canonical["recovery_binding"],
        expected_sha256=EXPECTED_RECOVERY_BINDING_SHA256,
        label="recovery binding",
    )
    _, validation_chain = _verify_reference(
        canonical["validation_chain"],
        expected_sha256=EXPECTED_VALIDATION_CHAIN_SHA256,
        label="validation chain",
    )
    _, sealed_test_chain = _verify_reference(
        canonical["sealed_test_chain"],
        expected_sha256=EXPECTED_SEALED_TEST_CHAIN_SHA256,
        label="sealed-test chain",
    )
    _, robustness_chain = _verify_reference(
        canonical["robustness_chain"],
        expected_sha256=EXPECTED_ROBUSTNESS_CHAIN_SHA256,
        label="robustness chain",
    )
    _require(
        migration_map.get("new_chain")
        == {
            key: canonical[key]
            for key in (
                "source_freeze",
                "recovery_binding",
                "validation_chain",
                "sealed_test_chain",
                "robustness_chain",
            )
        },
        "migration map corrected chain mismatch",
    )

    preserved = source_freeze.get("preserved_identity", {})
    _require(
        source_freeze.get("experimental_source_commit") == EXPECTED_SOURCE_COMMIT,
        "wrong source commit",
    )
    _require(preserved.get("new_binary_identity") is True, "V4R must have a new binary identity")
    _require(preserved.get("blocked_original_v4_unchanged") is True, "blocked V4 changed")
    _require(preserved.get("v3_artifacts_unchanged") is True, "V3 artifacts changed")
    _require(
        preserved.get("test_opened_at_freeze") is False,
        "source freeze occurred after test opening",
    )
    _require(binding.get("all_non_container_recovery_checks_exact") is True, "recovery is not exact")
    _require(
        binding.get("classification")
        == "new-binary-identity-weight-equivalent-recovered-pure-rl",
        "wrong recovery classification",
    )

    records: dict[str, dict[str, Any]] = {}
    for section, chain, split, count in (
        ("validation", validation_chain, "validation", 28),
        ("test", sealed_test_chain, "test", 60),
    ):
        _, result = _verify_reference(chain["result"], label=f"{section} result")
        _require(
            all(result.get(key) == value for key, value in chain["numeric_summary"].items()),
            f"{section} corrected numeric summary differs from immutable result",
        )
        _verify_result(result, split=split, episode_count=count, binding=binding)
        decision_path, decision = _verify_reference(
            chain["decision"],
            label=f"{section} decision",
        )
        records[section] = {
            "path": chain["result"]["path"],
            "sha256": chain["result"]["sha256"],
            "decision_path": decision_path.relative_to(ROOT).as_posix(),
            "decision_sha256": chain["decision"]["sha256"],
            "result": result,
        }
        _require(chain["strict_gate"].get("passed") is True, f"{section} strict gates failed")
        _require(
            chain["strict_gate"].get("test_tuning_prohibited") is True,
            f"{section} chain permits test tuning",
        )
        if section == "validation":
            _require(chain.get("selected") is True, "V4R was not selected on validation")
            _require(chain["chronology"].get("test_opened") is False, "validation opened test")
            _require(chain["chronology"].get("test_open_count") == 0, "test opened before freeze")
            _require(decision.get("selected") is True, "validation decision did not select V4R")
        else:
            _require(chain.get("sealed_test_passed") is True, "sealed test did not pass")
            _require(chain["chronology"].get("open_count") == 1, "wrong test opening count")
            _require(
                chain["chronology"].get("tuning_or_selection_permitted") is False,
                "test selection/tuning was permitted",
            )

    robustness: dict[str, dict[str, Any]] = {}
    for key in ("one_gw_total", "c_h_overlapping"):
        chain_record = robustness_chain.get(key)
        _require(isinstance(chain_record, dict), f"missing robustness result: {key}")
        _, result = _verify_reference(
            chain_record["artifact"],
            label=f"{key} robustness",
        )
        _require(
            all(result.get(field) == value for field, value in chain_record["numeric_summary"].items()),
            f"{key} corrected numeric summary differs from immutable result",
        )
        _verify_result(
            result,
            split="test",
            episode_count=60,
            binding=binding,
            post_selection=True,
        )
        robustness[key] = {
            "path": chain_record["artifact"]["path"],
            "sha256": chain_record["artifact"]["sha256"],
            "result": result,
        }
    _require(
        robustness_chain.get("post_selection_only") is True,
        "robustness is not post-selection",
    )
    _require(
        robustness_chain.get("selection_or_tuning") is False,
        "robustness was used for selection or tuning",
    )
    _require(
        robustness["c_h_overlapping"]["result"].get("variant") == "c-h-overlapping",
        "c-h robustness must retain its overlapping identity",
    )

    evidence = {
        "schema_version": canonical["schema_version"],
        "protocol_id": canonical["protocol_id"],
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "hash_contract": canonical["hash_contract"],
        "supersedes": canonical["supersedes"],
        "supersession_scope": canonical["supersession_scope"],
        "blocked_original_v4_evaluated": False,
        "sealed_test_opened": True,
        "sealed_test_open_count": canonical["sealed_test_open_count"],
        "validation": records["validation"],
        "test": records["test"],
        "robustness": robustness,
    }
    evidence["_verified"] = {
        "canonical_sha256": EXPECTED_CANONICAL_SHA256,
        "canonical_representation": CANONICAL_JSON_REPRESENTATION,
        "hash_contract_id": HASH_CONTRACT_ID,
        "reseal_commit": EXPECTED_RESEAL_COMMIT,
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "source_freeze_sha256": EXPECTED_SOURCE_FREEZE_SHA256,
        "recovery_binding_sha256": EXPECTED_RECOVERY_BINDING_SHA256,
        "validation_chain_sha256": EXPECTED_VALIDATION_CHAIN_SHA256,
        "sealed_test_chain_sha256": EXPECTED_SEALED_TEST_CHAIN_SHA256,
        "robustness_chain_sha256": EXPECTED_ROBUSTNESS_CHAIN_SHA256,
        "migration_map_sha256": EXPECTED_MIGRATION_MAP_SHA256,
        "superseded_canonical_sha256": EXPECTED_SUPERSEDED_CANONICAL_SHA256,
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
            "evidence_pointer": "/recovery_binding",
        },
        {
            "claim_id": "provenance-hash-contract",
            "value": HASH_CONTRACT_ID,
            "representation": CANONICAL_JSON_REPRESENTATION,
            "evidence_pointer": "/hash_contract",
        },
        {
            "claim_id": "provenance-supersession-scope",
            "value": "provenance-hash-chain-only",
            "evidence_pointer": "/supersession_scope",
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
        "schema_version": "ramp-v6-v4r-claim-ledger-v2",
        "canonical_evidence_sha256": EXPECTED_CANONICAL_SHA256,
        "canonical_evidence_representation": CANONICAL_JSON_REPRESENTATION,
        "hash_contract_id": HASH_CONTRACT_ID,
        "reseal_commit": EXPECTED_RESEAL_COMMIT,
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
        "schema_version": "ramp-v6-v4r-thesis-package-v2",
        "canonical_evidence_sha256": EXPECTED_CANONICAL_SHA256,
        "canonical_evidence_representation": CANONICAL_JSON_REPRESENTATION,
        "hash_contract_id": HASH_CONTRACT_ID,
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
