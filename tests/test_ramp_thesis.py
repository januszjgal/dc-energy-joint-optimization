from __future__ import annotations

import copy
import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from ramp_rl.v4r_thesis import (
    DEFAULT_CANONICAL,
    EXPECTED_CANONICAL_SHA256,
    EXPECTED_MEMBER_SEEDS,
    EXPECTED_RECOVERY_BINDING_SHA256,
    _verify_controller,
    _verify_result,
    build_claim_ledger,
    load_verified_evidence,
    write_publication_package,
)
from scripts.build_final_thesis import validate_source
from scripts.materialize_ramp_thesis import (
    DEFAULT_SOURCE,
    EXPECTED_TEMPLATE_SHA256,
    materialize_text,
    render_tokens,
)
from scripts.validate_ramp_thesis import REQUIRED_PLACEHOLDERS, validate_text


class RampThesisContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.evidence = load_verified_evidence()

    def test_canonical_evidence_has_expected_hash_and_identity(self) -> None:
        digest = hashlib.sha256(DEFAULT_CANONICAL.read_bytes()).hexdigest()
        self.assertEqual(digest, EXPECTED_CANONICAL_SHA256)
        self.assertEqual(self.evidence["sealed_test_open_count"], 1)
        self.assertFalse(self.evidence["blocked_original_v4_evaluated"])

    def test_template_has_required_v4r_placeholders_and_scope_guards(self) -> None:
        text = DEFAULT_SOURCE.read_text(encoding="utf-8")
        self.assertEqual(validate_text(text, allow_placeholders=True), [])
        self.assertEqual(
            hashlib.sha256(text.encode("utf-8")).hexdigest(),
            EXPECTED_TEMPLATE_SHA256,
        )
        for token in REQUIRED_PLACEHOLDERS:
            self.assertIn(f"{{{{CANONICAL_V4R:{token}}}}}", text)

    def test_materialized_contract_has_no_unresolved_results(self) -> None:
        text = DEFAULT_SOURCE.read_text(encoding="utf-8")
        begin = text.index("<!-- data-result-contract-begin -->")
        end = text.index("<!-- data-result-contract-end -->")
        replacements = render_tokens(self.evidence)
        for name in replacements:
            token = f"{{{{CANONICAL_V4R:{name}}}}}"
            self.assertTrue(begin < text.index(token) < end)
        for name, rendered in replacements.items():
            token = f"{{{{CANONICAL_V4R:{name}}}}}"
            text = text.replace(token, rendered)
        self.assertEqual(
            validate_text(text, allow_placeholders=False, evidence=self.evidence),
            [],
        )
        tampered = text.replace("-1.42587105143e-05", "0.5", 1)
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

    def test_wrong_canonical_hash_fails_closed(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "canonical_evidence.json"
            path.write_bytes(DEFAULT_CANONICAL.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "wrong canonical evidence SHA-256"):
                load_verified_evidence(path)

    def test_more_than_one_test_opening_fails_closed(self) -> None:
        payload = json.loads(DEFAULT_CANONICAL.read_text(encoding="utf-8"))
        payload["sealed_test_open_count"] = 2
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            path = base / "canonical_evidence.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            for name in ("source_freeze.json", "recovery_binding.json", "sealed_test_opening.json"):
                (base / name).write_bytes((DEFAULT_CANONICAL.parent / name).read_bytes())
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with (
                patch("ramp_rl.v4r_thesis.EXPECTED_CANONICAL_SHA256", digest),
                patch("ramp_rl.v4r_thesis._verify_git_identity"),
                self.assertRaisesRegex(ValueError, "open exactly once"),
            ):
                load_verified_evidence(path)

    def test_member_selection_and_wrong_model_hash_fail_closed(self) -> None:
        result = copy.deepcopy(self.evidence["test"]["result"])
        binding = json.loads(
            (DEFAULT_CANONICAL.parent / "recovery_binding.json").read_text(encoding="utf-8")
        )
        result["controller_audit"]["member_selection_or_exclusion"] = True
        with self.assertRaisesRegex(ValueError, "member selection"):
            _verify_controller(result, binding)
        result = copy.deepcopy(self.evidence["test"]["result"])
        result["controller_audit"]["members"][0]["model_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "wrong recovered model hash"):
            _verify_controller(result, binding)

    def test_missing_market_and_seed_fail_closed(self) -> None:
        binding = json.loads(
            (DEFAULT_CANONICAL.parent / "recovery_binding.json").read_text(encoding="utf-8")
        )
        result = copy.deepcopy(self.evidence["test"]["result"])
        result["per_market_macro"].pop("MISO_MINN_HUB")
        with self.assertRaisesRegex(ValueError, "missing or unsupported market"):
            _verify_result(result, split="test", episode_count=60, binding=binding)
        result = copy.deepcopy(self.evidence["test"]["result"])
        result["controller_audit"]["member_seeds"] = list(EXPECTED_MEMBER_SEEDS[:-1])
        with self.assertRaisesRegex(ValueError, "wrong member seeds"):
            _verify_result(result, split="test", episode_count=60, binding=binding)

    def test_test_selection_and_recovery_binding_fail_closed(self) -> None:
        result = copy.deepcopy(self.evidence["robustness"]["one_gw_total"]["result"])
        binding = json.loads(
            (DEFAULT_CANONICAL.parent / "recovery_binding.json").read_text(encoding="utf-8")
        )
        result["selection_or_tuning"] = True
        with self.assertRaisesRegex(ValueError, "selection or tuning"):
            _verify_result(
                result,
                split="test",
                episode_count=60,
                binding=binding,
                post_selection=True,
            )
        result = copy.deepcopy(self.evidence["test"]["result"])
        result["recovery_binding_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "wrong recovery binding"):
            _verify_result(result, split="test", episode_count=60, binding=binding)

    def test_decision_sidecars_are_hash_verified(self) -> None:
        payload = json.loads(DEFAULT_CANONICAL.read_text(encoding="utf-8"))
        payload["test"]["decision_sha256"] = "0" * 64
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            path = base / "canonical_evidence.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            for name in ("source_freeze.json", "recovery_binding.json", "sealed_test_opening.json"):
                (base / name).write_bytes((DEFAULT_CANONICAL.parent / name).read_bytes())
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with (
                patch("ramp_rl.v4r_thesis.EXPECTED_CANONICAL_SHA256", digest),
                patch("ramp_rl.v4r_thesis._verify_git_identity"),
                self.assertRaisesRegex(ValueError, "wrong test decision hash"),
            ):
                load_verified_evidence(path)

    def test_claim_ledger_is_hash_bound_and_labels_overlap(self) -> None:
        ledger = build_claim_ledger(self.evidence)
        self.assertEqual(
            ledger["canonical_evidence_sha256"],
            EXPECTED_CANONICAL_SHA256,
        )
        self.assertEqual(
            ledger["recovery_binding_sha256"],
            EXPECTED_RECOVERY_BINDING_SHA256,
        )
        overlap = next(
            claim
            for claim in ledger["claims"]
            if claim["claim_id"] == "c-h-overlapping-robustness"
        )
        self.assertIn("overlapping/non-independent", overlap["scope"])

    def test_publication_package_writes_claim_ledger_and_tables(self) -> None:
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            manifest = write_publication_package(self.evidence, output)
            self.assertTrue((output / "claim_ledger.json").is_file())
            self.assertTrue((output / "per_market_test.csv").is_file())
            self.assertIn("claim_ledger_sha256", manifest)

    def test_unsupported_scope_claim_is_rejected(self) -> None:
        text = DEFAULT_SOURCE.read_text(encoding="utf-8")
        text += "\nThe c-h result is an independent holdout.\n"
        errors = validate_text(text, allow_placeholders=True)
        self.assertTrue(any("unsupported claim" in item for item in errors))

    def test_materialized_source_resolves_repo_relative_images(self) -> None:
        with TemporaryDirectory() as temporary:
            source = Path(temporary) / "materialized.md"
            source.write_text(materialize_text(DEFAULT_SOURCE.read_text(encoding="utf-8"), self.evidence), encoding="utf-8")
            validate_source(source)

    def test_docx_build_rejects_tampering_outside_contract(self) -> None:
        text = materialize_text(DEFAULT_SOURCE.read_text(encoding="utf-8"), self.evidence)
        tampered = text.replace(
            "incremental normalized squared-ramp impact -1.4086907198e-05",
            "incremental normalized squared-ramp impact 0.5",
            1,
        )
        self.assertNotEqual(tampered, text)
        with TemporaryDirectory() as temporary:
            source = Path(temporary) / "tampered.md"
            source.write_text(tampered, encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "does not exactly match"):
                validate_source(source)

    def test_docx_build_rejects_missing_contract_marker(self) -> None:
        text = materialize_text(DEFAULT_SOURCE.read_text(encoding="utf-8"), self.evidence)
        with TemporaryDirectory() as temporary:
            source = Path(temporary) / "tampered.md"
            source.write_text(
                text.replace("<!-- data-result-contract-begin -->", "", 1),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "does not exactly match"):
                validate_source(source)

    def test_docx_build_rejects_modified_figure_manifest(self) -> None:
        text = materialize_text(DEFAULT_SOURCE.read_text(encoding="utf-8"), self.evidence)
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / "materialized.md"
            source.write_text(text, encoding="utf-8")
            manifest = base / "v4r_figure_manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "canonical_evidence_sha256": EXPECTED_CANONICAL_SHA256,
                        "files": {},
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch("scripts.build_final_thesis.V4R_FIGURE_MANIFEST", manifest),
                self.assertRaisesRegex(RuntimeError, "manifest hash mismatch"),
            ):
                validate_source(source)


if __name__ == "__main__":
    unittest.main()
