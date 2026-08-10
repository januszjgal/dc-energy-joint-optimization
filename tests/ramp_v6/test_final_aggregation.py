"""Synthetic-only tests for fail-closed final-result aggregation."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ramp_rl.final_aggregation import (
    SCHEMA_VERSION,
    EvidenceError,
    _raw_gate_outcome,
    audit_manifest,
    generate_package,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "ramp_final_results"


class FinalAggregationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        shutil.copytree(FIXTURES, self.root / "evidence")
        self.evidence = self.root / "evidence"
        self.manifest_path = self.evidence / "manifest.json"
        self.write_manifest()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def ref(self, name: str) -> dict[str, str]:
        path = self.evidence / name
        return {"path": name, "sha256": sha256_file(path)}

    def validation_ref(self, name: str, seed: int) -> dict:
        training_path = self.evidence / f"training_{seed}.json"
        training = json.loads(training_path.read_text(encoding="utf-8"))
        return {
            **self.ref(name),
            "seed": seed,
            "algorithm": "ppo",
            "training_manifest_sha256": sha256_file(training_path),
            "model_sha256": training["artifacts"]["model"]["sha256"],
            "final_policy_sha256": training["final_policy_sha256"],
        }

    def bind_canonical_validation(self) -> None:
        path = self.evidence / "canonical.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload["confirmation"]["rows"]:
            name = f"validation_{row['seed']}.json"
            row["validation_artifact"] = name
            row["validation_sha256"] = sha256_file(self.evidence / name)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def bind_validation_model_identity(self) -> None:
        for seed in (11, 12):
            training_path = self.evidence / f"training_{seed}.json"
            validation_path = self.evidence / f"validation_{seed}.json"
            payload = json.loads(validation_path.read_text(encoding="utf-8"))
            payload["model_identity"]["training_manifest_sha256"] = sha256_file(
                training_path
            )
            validation_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    def bind_analysis_identity(self) -> None:
        path = self.evidence / "analysis.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["identity"]["canonical_result_sha256"] = sha256_file(
            self.evidence / "canonical.json"
        )
        payload["identity"]["validation_bindings"] = [
            {
                "seed": seed,
                "validation_sha256": sha256_file(
                    self.evidence / f"validation_{seed}.json"
                ),
                "model_sha256": json.loads(
                    (self.evidence / f"training_{seed}.json").read_text(
                        encoding="utf-8"
                    )
                )["artifacts"]["model"]["sha256"],
                "final_policy_sha256": json.loads(
                    (self.evidence / f"training_{seed}.json").read_text(
                        encoding="utf-8"
                    )
                )["final_policy_sha256"],
            }
            for seed in (11, 12)
        ]
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def write_manifest(self, **updates) -> dict:
        self.bind_validation_model_identity()
        self.bind_canonical_validation()
        self.bind_analysis_identity()
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "label": "synthetic selected v2",
            "finality": "final",
            "result_status": "selected",
            "protocol": {**self.ref("protocol.yaml"), "id": "synthetic-ramp-v6-v2"},
            "identities": {
                "factory_input_hashes_sha256": "c" * 64,
                "forecast_model_id": "synthetic-causal-forecast-v1",
                "protocol_bundle_sha256": "a" * 64,
                "raw_acquisition_sha256": "b" * 64,
                "source_bundle_sha256": "b" * 64,
                "source_panel_sha256": "a" * 64,
            },
            "splits": {
                "train": ["2025-01"],
                "validation": ["2025-02"],
                "test": ["2025-03"],
            },
            "factory_manifest": self.ref("factory.json"),
            "canonical_result": self.ref("canonical.json"),
            "training_manifests": [
                self.ref("training_11.json"),
                self.ref("training_12.json"),
            ],
            "validation_results": [
                self.validation_ref("validation_11.json", 11),
                self.validation_ref("validation_12.json", 12),
            ],
            "analysis_artifact": self.ref("analysis.json"),
            "selection": {
                "algorithm": "ppo",
                "gate_assertions": {
                    "safety": True,
                    "cost": True,
                    "ramp": True,
                    "behavior": True,
                },
                "markets": ["MKT_A", "MKT_B"],
                "passed": True,
                "seeds": [11, 12],
                "split": "validation",
            },
            "sealed_test": {
                "opened": False,
                "opening_record": None,
                "reason": "validation selected; test not yet opened",
                "result": None,
                "used_for_selection": False,
            },
            "negative_protocol": {
                **self.ref("negative_v1.json"),
                "protocol_id": "v6-ramp-pure-rl-preregistered-v1",
                "confirmation_commit": (
                    "3391440d168ade7127899a8a2c2d84e3a47a9ad3"
                ),
            },
            "figure_omissions": {},
        }
        manifest.update(updates)
        if manifest["analysis_artifact"] is not None:
            manifest["analysis_artifact"] = self.ref("analysis.json")
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return manifest

    def mutate_json(self, name: str, mutation) -> None:
        path = self.evidence / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        mutation(payload)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def audit(self):
        with (
            patch("ramp_rl.final_aggregation._validate_selected_protocol"),
            patch(
                "ramp_rl.final_aggregation.V1_CANONICAL_SHA256",
                sha256_file(self.evidence / "negative_v1.json"),
            ),
        ):
            return audit_manifest(self.manifest_path)

    def test_builds_complete_package_from_synthetic_evidence(self) -> None:
        audited = self.audit()
        inventory = generate_package(audited, self.root / "report")
        self.assertEqual(inventory["result_status"], "selected")
        self.assertTrue((self.root / "report" / "claim_ledger.json").is_file())
        self.assertTrue((self.root / "report" / "tables" / "market_ramps.csv").is_file())
        self.assertTrue((self.root / "report" / "figures" / "sensitivities.png").is_file())
        self.assertFalse((self.root / "report" / "tables" / "safety_decoder.csv").exists())

    def test_rejects_truthy_string_gate_boolean(self) -> None:
        payload = json.loads(
            (self.evidence / "validation_11.json").read_text(encoding="utf-8")
        )
        payload["success_gate"]["checks"]["every_market_ramp_improves"] = "false"
        with self.assertRaisesRegex(EvidenceError, "must be a boolean"):
            _raw_gate_outcome(payload, "synthetic validation")

    def test_accepts_explicit_native_relative_market_gate_name(self) -> None:
        payload = json.loads(
            (self.evidence / "validation_11.json").read_text(
                encoding="utf-8"
            )
        )
        checks = payload["success_gate"]["checks"]
        checks[
            "every_market_native_relative_incremental_ramp_improves"
        ] = checks.pop("every_market_ramp_improves")
        outcome = _raw_gate_outcome(payload, "synthetic validation")
        self.assertTrue(outcome["ramp"])

    def test_rejects_missing_semantic_adjustment_telemetry(self) -> None:
        payload = json.loads(
            (self.evidence / "validation_11.json").read_text(encoding="utf-8")
        )
        payload.pop("semantic_adjustment_l2")
        with self.assertRaisesRegex(EvidenceError, "decoder telemetry missing"):
            _raw_gate_outcome(payload, "synthetic validation")

    def test_rejects_test_split_leakage_into_promotion(self) -> None:
        self.mutate_json(
            "validation_11.json",
            lambda payload: payload.update({"split": "test"}),
        )
        self.bind_canonical_validation()
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest["validation_results"][0] = self.validation_ref(
            "validation_11.json", 11
        )
        manifest["canonical_result"] = self.ref("canonical.json")
        self.write_manifest(**manifest)
        with self.assertRaisesRegex(EvidenceError, "non-validation evidence"):
            self.audit()

    def test_rejects_invalid_promotion(self) -> None:
        self.mutate_json(
            "canonical.json",
            lambda payload: payload["confirmation"]["promotion_decision"].update(
                {"promoted_algorithms": []}
            ),
        )
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest["canonical_result"] = self.ref("canonical.json")
        self.write_manifest(**manifest)
        with self.assertRaisesRegex(EvidenceError, "not the sole validation promotion"):
            self.audit()

    def test_rejects_canonical_test_based_promotion(self) -> None:
        self.mutate_json(
            "canonical.json",
            lambda payload: payload["confirmation"]["promotion_decision"].update(
                {"sealed_test_used": True}
            ),
        )
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest["canonical_result"] = self.ref("canonical.json")
        self.write_manifest(**manifest)
        with self.assertRaisesRegex(EvidenceError, "promotion is not validation-only"):
            self.audit()

    def test_rejects_missing_market(self) -> None:
        self.mutate_json(
            "validation_12.json",
            lambda payload: payload["per_market_macro"].pop("MKT_B"),
        )
        self.bind_canonical_validation()
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest["validation_results"][1] = self.validation_ref(
            "validation_12.json", 12
        )
        manifest["canonical_result"] = self.ref("canonical.json")
        self.write_manifest(**manifest)
        with self.assertRaisesRegex(EvidenceError, "market set mismatch"):
            self.audit()

    def test_rejects_hash_mismatch(self) -> None:
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest["canonical_result"]["sha256"] = "0" * 64
        self.write_manifest(**manifest)
        with self.assertRaisesRegex(EvidenceError, "hash mismatch"):
            self.audit()

    def test_rejects_canonical_telemetry_mismatch(self) -> None:
        self.mutate_json(
            "canonical.json",
            lambda payload: payload["confirmation"]["rows"][0].update(
                {"energy_cost_ratio": 0.5}
            ),
        )
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest["canonical_result"] = self.ref("canonical.json")
        self.write_manifest(**manifest)
        with self.assertRaisesRegex(EvidenceError, "cost metric mismatch"):
            self.audit()

    def test_rejects_missing_pure_rl_assertion(self) -> None:
        self.mutate_json(
            "training_11.json",
            lambda payload: payload["pure_rl_assertions"].pop("teacher"),
        )
        self.bind_validation_model_identity()
        self.bind_canonical_validation()
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest["training_manifests"][0] = self.ref("training_11.json")
        manifest["validation_results"][0] = self.validation_ref(
            "validation_11.json", 11
        )
        manifest["canonical_result"] = self.ref("canonical.json")
        self.write_manifest(**manifest)
        with self.assertRaisesRegex(EvidenceError, "pure-RL audit failed"):
            self.audit()

    def test_rejects_missing_gate_assertion(self) -> None:
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest["selection"]["gate_assertions"].pop("cost")
        self.write_manifest(**manifest)
        with self.assertRaisesRegex(EvidenceError, "selection gate assertions missing"):
            self.audit()

    def test_rejects_seed_set_mismatch(self) -> None:
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest["selection"]["seeds"] = [11]
        self.write_manifest(**manifest)
        with self.assertRaisesRegex(EvidenceError, "seed set"):
            self.audit()

    def test_rejects_duplicate_canonical_seed(self) -> None:
        self.mutate_json(
            "canonical.json",
            lambda payload: payload["confirmation"]["rows"].append(
                dict(payload["confirmation"]["rows"][0])
            ),
        )
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest["canonical_result"] = self.ref("canonical.json")
        self.write_manifest(**manifest)
        with self.assertRaisesRegex(EvidenceError, "exactly one row"):
            self.audit()

    def test_rejects_source_hash_mismatch(self) -> None:
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest["identities"]["source_panel_sha256"] = "0" * 64
        self.write_manifest(**manifest)
        with self.assertRaisesRegex(EvidenceError, "source panel hash mismatch"):
            self.audit()

    def test_rejects_missing_selected_analysis(self) -> None:
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest["analysis_artifact"] = None
        self.write_manifest(**manifest)
        with self.assertRaisesRegex(EvidenceError, "missing thesis analysis evidence"):
            self.audit()

    def test_rejects_opened_test_without_record(self) -> None:
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest["sealed_test"]["opened"] = True
        self.write_manifest(**manifest)
        with self.assertRaisesRegex(EvidenceError, "reference is missing"):
            self.audit()

    def test_rejects_validation_model_binding_mismatch(self) -> None:
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest["validation_results"][0]["model_sha256"] = "0" * 64
        self.write_manifest(**manifest)
        with self.assertRaisesRegex(EvidenceError, "model hash mismatch"):
            self.audit()

    def test_rejects_gate_boolean_that_disagrees_with_telemetry(self) -> None:
        self.mutate_json(
            "validation_11.json",
            lambda payload: payload["success_gate"]["checks"].update(
                {"every_market_ramp_improves": False}
            ),
        )
        self.bind_canonical_validation()
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest["validation_results"][0] = self.validation_ref(
            "validation_11.json", 11
        )
        manifest["canonical_result"] = self.ref("canonical.json")
        self.write_manifest(**manifest)
        with self.assertRaisesRegex(EvidenceError, "do not match raw telemetry"):
            self.audit()

    def test_rejects_non_finite_gate_telemetry(self) -> None:
        self.mutate_json(
            "validation_11.json",
            lambda payload: payload.update(
                {"mean_incremental_ramp_impact": float("-inf")}
            ),
        )
        self.bind_canonical_validation()
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest["validation_results"][0] = self.validation_ref(
            "validation_11.json", 11
        )
        manifest["canonical_result"] = self.ref("canonical.json")
        self.write_manifest(**manifest)
        with self.assertRaisesRegex(EvidenceError, "non-finite telemetry"):
            self.audit()

    def test_rejects_test_rows_in_analysis(self) -> None:
        self.mutate_json(
            "analysis.json",
            lambda payload: payload["forecast_error_strata"][0].update(
                {"split": "test"}
            ),
        )
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest["analysis_artifact"] = self.ref("analysis.json")
        self.write_manifest(**manifest)
        with self.assertRaisesRegex(EvidenceError, "contains non-validation rows"):
            self.audit()

    def test_rejects_analysis_binding_mismatch(self) -> None:
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.write_manifest(**manifest)
        self.mutate_json(
            "analysis.json",
            lambda payload: payload["identity"].update(
                {"canonical_result_sha256": "0" * 64}
            ),
        )
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest["analysis_artifact"] = self.ref("analysis.json")
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(EvidenceError, "analysis identity"):
            self.audit()

    def test_rejects_duplicate_market_ramp_analysis(self) -> None:
        self.mutate_json(
            "analysis.json",
            lambda payload: payload["market_ramps"].append(
                dict(payload["market_ramps"][0])
            ),
        )
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest["analysis_artifact"] = self.ref("analysis.json")
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(EvidenceError, "duplicated"):
            self.audit()

    def test_rejects_wrong_v1_negative_commit(self) -> None:
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest["negative_protocol"]["confirmation_commit"] = "0" * 40
        self.write_manifest(**manifest)
        with self.assertRaisesRegex(EvidenceError, "confirmation commit mismatch"):
            self.audit()

    def test_default_protocol_trust_anchor_rejects_synthetic_selection(self) -> None:
        with self.assertRaisesRegex(EvidenceError, "frozen v2 protocol"):
            audit_manifest(self.manifest_path)

    def test_rejects_unexpected_stale_output(self) -> None:
        audited = self.audit()
        output = self.root / "report"
        output.mkdir()
        (output / "stale.txt").write_text("old", encoding="utf-8")
        with self.assertRaisesRegex(EvidenceError, "unexpected stale paths"):
            generate_package(audited, output)


if __name__ == "__main__":
    unittest.main()
