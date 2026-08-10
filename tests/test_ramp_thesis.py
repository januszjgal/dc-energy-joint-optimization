from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from ramp_rl.provenance import (
    canonical_json_file_sha256,
    canonical_json_sha256,
)
from ramp_rl.v4r_corrected_thesis import (
    DEFAULT_CANONICAL,
    EXPECTED_CANONICAL_SHA256,
    EXPECTED_SEALED_CANONICAL_SHA256,
    load_verified_evidence,
    write_publication_package,
)
from ramp_rl.v4r_metric_replay_recovery import (
    COMPARISON_FIELDS,
    DEFAULT_MANIFEST as DEFAULT_RECOVERY_MANIFEST,
    STABLE_RECOVERY_COMMIT,
    verify_metric_replay_recovery_manifest,
)
from scripts.build_final_thesis import validate_source
from scripts.materialize_ramp_thesis import (
    DEFAULT_SOURCE,
    EXPECTED_TEMPLATE_SHA256,
    materialize_text,
    render_tokens,
)
from scripts.recompute_v4r_posthoc_metrics import verify as verify_posthoc
from scripts.validate_ramp_thesis import (
    REQUIRED_PLACEHOLDERS,
    validate_text,
)

ROOT = Path(__file__).resolve().parents[1]


class CorrectedRampThesisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.evidence = load_verified_evidence()

    def test_corrected_and_sealed_evidence_identities(self) -> None:
        self.assertEqual(
            canonical_json_file_sha256(DEFAULT_CANONICAL),
            EXPECTED_CANONICAL_SHA256,
        )
        self.assertEqual(
            self.evidence["_verified"]["sealed_canonical_sha256"],
            EXPECTED_SEALED_CANONICAL_SHA256,
        )
        self.assertEqual(self.evidence["sealed_test_open_count"], 1)
        self.assertFalse(self.evidence["blocked_original_v4_evaluated"])

    def test_posthoc_scope_is_explicit_and_non_confirmatory(self) -> None:
        wrapper = verify_posthoc(DEFAULT_CANONICAL)
        self.assertFalse(wrapper["second_sealed_generalization_test"])
        self.assertFalse(
            wrapper["metric_replay_training_or_retuning_performed"]
        )
        self.assertFalse(
            wrapper["model_selection_or_weighting_performed"]
        )
        self.assertTrue(
            wrapper[
                "original_sealed_result_and_chronology_preserved"
            ]
        )
        self.assertEqual(
            wrapper["supersession_scope"],
            "physical-ramp-status-quo-labeling-and-decoder-telemetry-only",
        )
        self.assertEqual(
            wrapper["checkpoint_recovery"]["sha256"],
            canonical_json_file_sha256(DEFAULT_RECOVERY_MANIFEST),
        )
        self.assertTrue(
            wrapper[
                "deterministic_checkpoint_reconstruction_performed"
            ]
        )

    def test_checkpoint_recovery_is_fully_bound_and_non_confirmatory(
        self,
    ) -> None:
        recovery = verify_metric_replay_recovery_manifest(
            DEFAULT_RECOVERY_MANIFEST
        )
        self.assertEqual(
            recovery["stable_recovery_commit"],
            STABLE_RECOVERY_COMMIT,
        )
        self.assertTrue(recovery["training_computation_performed"])
        self.assertFalse(recovery["fresh_generalization_evidence"])
        self.assertFalse(
            recovery["validation_or_test_opened_during_recovery"]
        )
        self.assertFalse(
            recovery["march_april_replay_is_fresh_sealed_test"]
        )
        self.assertFalse(
            recovery["training_or_retuning_for_policy_change"]
        )
        self.assertFalse(recovery["model_selection_performed"])
        self.assertTrue(recovery["original_sealed_result_immutable"])
        self.assertEqual(
            [member["seed"] for member in recovery["members"]],
            list(range(2801, 2806)),
        )
        for member in recovery["members"]:
            self.assertEqual(
                member["comparison_fields"],
                list(COMPARISON_FIELDS),
            )
            self.assertTrue(member["comparison_fields_exact"])
            self.assertEqual(
                member["training_data_splits_observed"],
                ["train"],
            )
            self.assertEqual(member["validation_or_test_rows"], 0)
            self.assertTrue(
                member["vecnormalize"][
                    "byte_identical_to_frozen_v3"
                ]
            )
        self.assertEqual(
            recovery["data_access_audit"],
            {
                "raw_source_test_opened": False,
                "member_training_splits": {
                    str(seed): ["train"] for seed in range(2801, 2806)
                },
                "member_validation_or_test_rows": {
                    str(seed): 0 for seed in range(2801, 2806)
                },
            },
        )

    def test_primary_result_is_invariant(self) -> None:
        for split in ("validation", "test"):
            sealed = self.evidence["sealed_results"][split]
            corrected = self.evidence[split]["result"]
            self.assertEqual(
                corrected["mean_incremental_ramp_impact"],
                sealed["mean_incremental_ramp_impact"],
            )
            self.assertEqual(
                corrected["per_market_macro"],
                sealed["per_market_macro"],
            )
            self.assertEqual(
                corrected["energy_cost_ratio"],
                sealed["energy_cost_ratio"],
            )
            self.assertEqual(
                corrected["behavior_audit"],
                sealed["behavior_audit"],
            )

    def test_physical_metrics_pool_absolute_market_timestep_values(
        self,
    ) -> None:
        test = self.evidence["test"]["result"]
        contract = test["physical_ramp_metric_contract"]
        self.assertEqual(
            contract["input"],
            "absolute adjusted ramp magnitude",
        )
        self.assertFalse(contract["cross_market_signed_averaging"])
        h1_count = sum(
            len(values)
            for episode in test["policy_episodes"]
            for values in episode[
                "abs_adjusted_ramp_h1_by_market"
            ].values()
        )
        h3_count = sum(
            len(values)
            for episode in test["policy_episodes"]
            for values in episode[
                "abs_adjusted_ramp_h3_by_market"
            ].values()
        )
        self.assertEqual(h1_count, 60 * 27 * 6)
        self.assertEqual(h3_count, 60 * 27 * 6)
        self.assertGreaterEqual(
            test[
                "abs_adjusted_ramp_h1_fraction_s_per_hour_max"
            ],
            test[
                "abs_adjusted_ramp_h1_fraction_s_per_hour_p95"
            ],
        )

    def test_status_quo_comparison_reports_miso_exception(self) -> None:
        comparison = self.evidence["test"]["result"][
            "status_quo_comparison"
        ]
        self.assertEqual(comparison["markets_better_count"], 5)
        self.assertEqual(comparison["market_count"], 6)
        self.assertAlmostEqual(
            comparison["markets_better_share"],
            5 / 6,
        )
        self.assertFalse(
            comparison["every_market_outperforms_status_quo"]
        )
        miso = comparison["per_market"]["MISO_MINN_HUB"]
        self.assertFalse(miso["policy_outperforms_status_quo"])
        self.assertAlmostEqual(
            miso["policy_native_relative_incremental_ramp_impact"],
            -5.652448501033376e-07,
        )
        self.assertAlmostEqual(
            miso["status_quo_native_relative_incremental_ramp_impact"],
            -1.1404305922222146e-06,
        )
        self.assertTrue(
            self.evidence["test"]["result"]["success_gate"]["checks"][
                "every_market_native_relative_incremental_ramp_improves"
            ]
        )

    def test_secondary_cost_and_ramp_power_match_evidence(self) -> None:
        test = self.evidence["test"]["result"]
        policy_cost = sum(
            episode["energy_cost"]
            for episode in test["policy_episodes"]
        )
        status_cost = sum(
            episode["energy_cost"]
            for episode in test["status_quo_episodes"]
        )
        self.assertAlmostEqual(policy_cost, 21098313.786029983)
        self.assertAlmostEqual(status_cost, 21582662.393725865)
        self.assertAlmostEqual(
            status_cost - policy_cost,
            484348.6076958813,
        )
        self.assertAlmostEqual(
            test["behavior_audit"]["policy_ramp_power"],
            303077.9035596151,
        )
        self.assertAlmostEqual(
            test["behavior_audit"]["status_quo_ramp_power"],
            313167.22669593635,
        )

    def test_decoder_adjustment_is_real_and_separate_from_emergency(
        self,
    ) -> None:
        adjustment = self.evidence["test"]["result"][
            "semantic_adjustment"
        ]
        self.assertEqual(
            adjustment["units"],
            "compute_work_units_per_hourly_decision",
        )
        self.assertGreater(adjustment["mean_l2"], 0.0)
        self.assertGreater(
            adjustment["positive_adjustment_count"],
            0,
        )
        self.assertGreaterEqual(
            adjustment["max_l2"],
            adjustment["p95_l2"],
        )
        self.assertEqual(adjustment["emergency_fallback_rate"], 0.0)
        self.assertEqual(
            adjustment["decision_count"],
            60 * 27,
        )

    def test_all_frozen_training_cost_multipliers_stayed_zero(
        self,
    ) -> None:
        for seed in range(2801, 2806):
            manifest = json.loads(
                (
                    ROOT
                    / "models"
                    / "ramp_rl_v6"
                    / "live_v3"
                    / "confirmation"
                    / "ppo"
                    / str(seed)
                    / "training_manifest.json"
                ).read_text(encoding="utf-8")
            )
            multiobjective = manifest["multiobjective"]
            self.assertEqual(
                multiobjective["final_lagrangian_multiplier"],
                0.0,
            )
            self.assertTrue(
                all(
                    update["multiplier"] == 0.0
                    for update in multiobjective[
                        "lagrangian_updates"
                    ]
                )
            )

    def test_template_placeholders_and_hash_are_frozen(self) -> None:
        text = DEFAULT_SOURCE.read_text(encoding="utf-8")
        self.assertEqual(
            validate_text(text, allow_placeholders=True),
            [],
        )
        self.assertEqual(
            hashlib.sha256(text.encode("utf-8")).hexdigest(),
            EXPECTED_TEMPLATE_SHA256,
        )
        for token in REQUIRED_PLACEHOLDERS:
            self.assertIn(
                f"{{{{CANONICAL_V4R:{token}}}}}",
                text,
            )

    def test_materialized_contract_matches_evidence(self) -> None:
        template = DEFAULT_SOURCE.read_text(encoding="utf-8")
        text = materialize_text(template, self.evidence)
        self.assertEqual(
            validate_text(
                text,
                allow_placeholders=False,
                evidence=self.evidence,
            ),
            [],
        )
        for name in render_tokens(self.evidence):
            self.assertNotIn(
                f"{{{{CANONICAL_V4R:{name}}}}}",
                text,
            )
        rendered = render_tokens(self.evidence)
        for token_name in (
            "PHYSICAL_RAMP_TABLE",
            "DECODER_ADJUSTMENT_TABLE",
        ):
            self.assertNotIn(
                "| Sealed test |",
                rendered[token_name],
            )
            self.assertIn(
                "March-April original test windows (not a sealed test)",
                rendered[token_name],
            )
        normalized_text = " ".join(text.split())
        self.assertIn(
            "unavailable from the original sealed traces",
            normalized_text,
        )
        self.assertIn(
            "literal-zero decoder field is invalid placeholder telemetry",
            normalized_text,
        )
        self.assertIn(
            "This comparator erratum requires no policy replay",
            normalized_text,
        )
        for factual_phrase in (
            "34,848 market-hours",
            "114 training daily windows",
            "186 scalar values",
            "1,620 ensemble decisions",
            "2, 1, 2, 3, 4, and 3 hours",
        ):
            self.assertIn(factual_phrase, normalized_text)
        tampered = text.replace(
            "-1.42587105143e-05",
            "0.5",
            1,
        )
        self.assertTrue(
            any(
                "contract differs" in error
                for error in validate_text(
                    tampered,
                    allow_placeholders=False,
                    evidence=self.evidence,
                )
            )
        )

    def test_wrong_corrected_hash_fails_closed(self) -> None:
        payload = json.loads(
            DEFAULT_CANONICAL.read_text(encoding="utf-8")
        )
        payload["second_sealed_generalization_test"] = True
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "canonical.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with (
                patch(
                    "ramp_rl.v4r_corrected_thesis.EXPECTED_CANONICAL_SHA256",
                    canonical_json_sha256(payload),
                ),
                self.assertRaisesRegex(
                    ValueError,
                    "mislabeled as a new sealed test",
                ),
            ):
                load_verified_evidence(path)

    def test_corrected_reference_tampering_fails_closed(self) -> None:
        payload = json.loads(
            DEFAULT_CANONICAL.read_text(encoding="utf-8")
        )
        payload["corrected_results"]["test"]["sha256"] = "0" * 64
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "canonical.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_posthoc(path)

    def test_publication_package_writes_corrected_tables(self) -> None:
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            manifest = write_publication_package(
                self.evidence,
                output,
            )
            for name in (
                "claim_ledger.json",
                "split_summary.csv",
                "sealed_test_market_comparison.csv",
                "physical_ramps.csv",
                "decoder_adjustment.csv",
                "behavior.csv",
                "cost_comparison.csv",
                "package_manifest.json",
            ):
                self.assertTrue((output / name).is_file(), name)
            self.assertEqual(
                manifest["corrected_canonical_sha256"],
                EXPECTED_CANONICAL_SHA256,
            )
            self.assertEqual(
                manifest["checkpoint_recovery_sha256"],
                self.evidence["posthoc_correction"][
                    "checkpoint_recovery"
                ]["sha256"],
            )

    def test_docx_source_validation_rejects_tampering(self) -> None:
        materialized = materialize_text(
            DEFAULT_SOURCE.read_text(encoding="utf-8"),
            self.evidence,
        )
        with TemporaryDirectory() as temporary:
            source = Path(temporary) / "materialized.md"
            source.write_text(materialized, encoding="utf-8")
            validate_source(source)
            source.write_text(
                materialized.replace(
                    "five of six markets",
                    "six of six markets",
                    1,
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "does not exactly match",
            ):
                validate_source(source)


if __name__ == "__main__":
    unittest.main()
