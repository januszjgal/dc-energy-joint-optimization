from __future__ import annotations

import hashlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.build_final_thesis import validate_source
from scripts.materialize_ramp_thesis import EXPECTED_TEMPLATE_SHA256, render_tokens
from scripts.validate_ramp_thesis import (
    DEFAULT_SOURCE,
    REQUIRED_PLACEHOLDERS,
    validate_text,
)

PROTOCOL_COMMIT = "4d47a9a"
SOURCE_COMMIT = "b1bb302"


class RampThesisContractTests(unittest.TestCase):
    def test_draft_has_required_placeholders_and_scope_guards(self) -> None:
        text = DEFAULT_SOURCE.read_text(encoding="utf-8")
        self.assertEqual(validate_text(text, allow_placeholders=True), [])
        self.assertEqual(
            hashlib.sha256(text.encode("utf-8")).hexdigest(),
            EXPECTED_TEMPLATE_SHA256,
        )
        for token in REQUIRED_PLACEHOLDERS:
            self.assertIn(f"{{{{CANONICAL_V2:{token}}}}}", text)

    def test_release_validation_rejects_unresolved_results(self) -> None:
        text = DEFAULT_SOURCE.read_text(encoding="utf-8")
        errors = validate_text(text, allow_placeholders=False)
        self.assertTrue(any("unresolved canonical placeholders" in item for item in errors))
        with self.assertRaisesRegex(
            RuntimeError,
            "unresolved canonical-result placeholders",
        ):
            validate_source(DEFAULT_SOURCE)

    def test_materializer_rejects_noncanonical_payload(self) -> None:
        with self.assertRaisesRegex(ValueError, "canonical must be true"):
            render_tokens(
                {
                    "schema_version": "ramp-v6-thesis-results-v1",
                    "canonical": False,
                    "generated": True,
                    "protocol_commit": PROTOCOL_COMMIT,
                    "source_commit": SOURCE_COMMIT,
                    "campaign_status": "validation_blocker_no_selected_protocol",
                    "selection_split": "validation",
                    "validation_rows": [
                        {
                            "candidate_id": "ppo-v2",
                            "seed_count": 5,
                            "mean_incremental_ramp_impact": -0.00001,
                            "mean_da_cost_ratio": 0.99,
                            "exact_safety_all_seeds": True,
                            "all_strict_gates_pass": False,
                        }
                    ],
                    "selection": {
                        "selected": False,
                        "candidate_id": "",
                        "failed_gates": [],
                    },
                    "sealed_test": {
                        "opened": False,
                        "selection_or_tuning_used": False,
                    },
                    "test_rows": [],
                    "statistics": {
                        "optimizer_seed_count": 5,
                        "day_units": 28,
                        "month_units": 0,
                        "interval_method": "day block bootstrap",
                    },
                    "verdict_code": "pending",
                }
            )

    def test_materializer_rejects_test_without_selection(self) -> None:
        payload = {
            "schema_version": "ramp-v6-thesis-results-v1",
            "canonical": True,
            "generated": True,
            "protocol_commit": PROTOCOL_COMMIT,
            "source_commit": SOURCE_COMMIT,
            "campaign_status": "campaign_complete",
            "selection_split": "validation",
            "validation_rows": [
                {
                    "candidate_id": "ppo-v2",
                    "seed_count": 5,
                    "mean_incremental_ramp_impact": -0.00001,
                    "mean_da_cost_ratio": 0.99,
                    "exact_safety_all_seeds": True,
                    "all_strict_gates_pass": False,
                }
            ],
            "selection": {
                "selected": False,
                "candidate_id": "",
                "failed_gates": ["validation_success_gate"],
            },
            "sealed_test": {"opened": True, "selection_or_tuning_used": False},
            "test_rows": [],
            "statistics": {
                "optimizer_seed_count": 5,
                "day_units": 28,
                "month_units": 1,
                "interval_method": "percentile bootstrap",
            },
            "verdict_code": "invalid",
        }
        with self.assertRaisesRegex(ValueError, "cannot be opened"):
            render_tokens(payload)

    def test_validation_blocker_payload_materializes_without_test_claim(self) -> None:
        payload = {
            "schema_version": "ramp-v6-thesis-results-v1",
            "canonical": True,
            "generated": True,
            "protocol_commit": PROTOCOL_COMMIT,
            "source_commit": SOURCE_COMMIT,
            "campaign_status": "validation_blocker_no_selected_protocol",
            "selection_split": "validation",
            "validation_rows": [
                {
                    "candidate_id": "ppo-v2",
                    "seed_count": 5,
                    "mean_incremental_ramp_impact": -0.00001,
                    "mean_da_cost_ratio": 0.99,
                    "exact_safety_all_seeds": True,
                    "all_strict_gates_pass": False,
                }
            ],
            "selection": {
                "selected": False,
                "candidate_id": "",
                "failed_gates": ["every_market_ramp_improves"],
            },
            "sealed_test": {
                "opened": False,
                "selection_or_tuning_used": False,
            },
            "test_rows": [],
            "statistics": {
                "optimizer_seed_count": 5,
                "day_units": 28,
                "month_units": 0,
                "interval_method": "day block bootstrap",
            },
            "verdict_code": "validation_blocker_no_selected_protocol",
        }
        rendered = render_tokens(payload)
        self.assertIn("unopened", rendered["SEALED_TEST_STATUS"])
        self.assertIn("No sealed test result rows", rendered["TEST_RESULTS_TABLE"])

    def test_selected_candidate_must_pass_validation(self) -> None:
        payload = {
            "schema_version": "ramp-v6-thesis-results-v1",
            "canonical": True,
            "generated": True,
            "protocol_commit": PROTOCOL_COMMIT,
            "source_commit": SOURCE_COMMIT,
            "campaign_status": "validation_selected_test_pending",
            "selection_split": "validation",
            "validation_rows": [
                {
                    "candidate_id": "ppo-v2",
                    "seed_count": 5,
                    "mean_incremental_ramp_impact": -0.00001,
                    "mean_da_cost_ratio": 0.99,
                    "exact_safety_all_seeds": True,
                    "all_strict_gates_pass": False,
                }
            ],
            "selection": {
                "selected": True,
                "candidate_id": "ppo-v2",
                "failed_gates": [],
            },
            "sealed_test": {
                "opened": False,
                "selection_or_tuning_used": False,
            },
            "test_rows": [],
            "statistics": {
                "optimizer_seed_count": 5,
                "day_units": 28,
                "month_units": 0,
                "interval_method": "day block bootstrap",
            },
            "verdict_code": "test_pending",
        }
        with self.assertRaisesRegex(ValueError, "must pass all strict"):
            render_tokens(payload)

    def test_pending_campaign_rejects_test_verdict(self) -> None:
        payload = {
            "schema_version": "ramp-v6-thesis-results-v1",
            "canonical": True,
            "generated": True,
            "protocol_commit": PROTOCOL_COMMIT,
            "source_commit": SOURCE_COMMIT,
            "campaign_status": "validation_selected_test_pending",
            "selection_split": "validation",
            "validation_rows": [
                {
                    "candidate_id": "ppo-v2",
                    "seed_count": 5,
                    "mean_incremental_ramp_impact": -0.00001,
                    "mean_da_cost_ratio": 0.99,
                    "exact_safety_all_seeds": True,
                    "all_strict_gates_pass": True,
                }
            ],
            "selection": {
                "selected": True,
                "candidate_id": "ppo-v2",
                "failed_gates": [],
            },
            "sealed_test": {
                "opened": False,
                "selection_or_tuning_used": False,
            },
            "test_rows": [],
            "statistics": {
                "optimizer_seed_count": 5,
                "day_units": 28,
                "month_units": 0,
                "interval_method": "day block bootstrap",
            },
            "verdict_code": "sealed_test_success",
        }
        with self.assertRaisesRegex(ValueError, "invalid for campaign_status"):
            render_tokens(payload)

    def test_sealed_test_rows_must_match_selected_candidate(self) -> None:
        payload = {
            "schema_version": "ramp-v6-thesis-results-v1",
            "canonical": True,
            "generated": True,
            "protocol_commit": PROTOCOL_COMMIT,
            "source_commit": SOURCE_COMMIT,
            "campaign_status": "campaign_complete",
            "selection_split": "validation",
            "validation_rows": [
                {
                    "candidate_id": "ppo-v2",
                    "seed_count": 5,
                    "mean_incremental_ramp_impact": -0.00001,
                    "mean_da_cost_ratio": 0.99,
                    "exact_safety_all_seeds": True,
                    "all_strict_gates_pass": True,
                }
            ],
            "selection": {
                "selected": True,
                "candidate_id": "ppo-v2",
                "failed_gates": [],
            },
            "sealed_test": {
                "opened": True,
                "selection_or_tuning_used": False,
            },
            "test_rows": [
                {
                    "candidate_id": "other-candidate",
                    "seed_count": 5,
                    "mean_incremental_ramp_impact": -0.00001,
                    "mean_da_cost_ratio": 0.99,
                    "exact_safety_all_seeds": True,
                    "all_strict_gates_pass": True,
                }
            ],
            "statistics": {
                "optimizer_seed_count": 5,
                "day_units": 61,
                "month_units": 2,
                "interval_method": "day and month block bootstrap",
            },
            "verdict_code": "sealed_test_success",
        }
        with self.assertRaisesRegex(ValueError, "selected candidate"):
            render_tokens(payload)

    def test_wrapped_v2_claim_is_rejected(self) -> None:
        text = DEFAULT_SOURCE.read_text(encoding="utf-8")
        text += "\nThe v2\nachieved an improvement.\n"
        errors = validate_text(text, allow_placeholders=True)
        self.assertTrue(any("unbound v2 result claim" in item for item in errors))

    def test_materialized_source_can_resolve_repo_relative_images(self) -> None:
        with TemporaryDirectory() as temporary:
            source = Path(temporary) / "materialized.md"
            source.write_text(
                "![Committed figure](docs/figures/ramp_v6/"
                "six_market_study_design.png)\n",
                encoding="utf-8",
            )
            validate_source(source)


if __name__ == "__main__":
    unittest.main()
