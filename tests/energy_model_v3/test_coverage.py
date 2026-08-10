from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from energy_model_v3.coverage import assess_candidate_coverage

ROOT = Path(__file__).resolve().parents[2]


class CandidateCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.live_evidence = json.loads(
            (
                ROOT
                / "data"
                / "energy_model_v3"
                / "provenance"
                / "coverage_evidence.json"
            ).read_text(encoding="utf-8")
        )
        cls.live_evidence_sha256 = hashlib.sha256(
            json.dumps(
                cls.live_evidence, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        cls.evidence = copy.deepcopy(cls.live_evidence)
        cls.evidence.pop("verified_live_handoff")
        for record in cls.evidence["markets"].values():
            record["candidate_complete"] = False
            record["reason"] = "candidate product proof is not staged"
        cls.evidence["markets"]["ERCOT_LZ_NORTH"]["reason"] = (
            "the 243 daily SCED disclosures are not staged"
        )
        cls.evidence_sha256 = hashlib.sha256(
            json.dumps(
                cls.evidence, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()

    def test_verified_live_handoff_clears_all_six_markets(self) -> None:
        missing = [
            reference["path"]
            for reference in self.live_evidence[
                "verified_live_handoff"
            ].values()
            if not (ROOT / reference["path"]).is_file()
        ]
        if missing:
            self.skipTest(
                "restricted local live handoff is not staged: "
                + ", ".join(missing)
            )
        assessment = assess_candidate_coverage(
            self.live_evidence,
            evidence_sha256=self.live_evidence_sha256,
            repository_root=ROOT,
            eia_api_key_available=False,
            ercot_eia_fallback_activated=False,
        )
        self.assertEqual(assessment["status"], "COVERAGE_VERIFIED")
        self.assertEqual(assessment["blockers"], {})
        self.assertEqual(set(assessment["markets"]), set(self.live_evidence["markets"]))

    def assess(
        self,
        evidence: dict,
        *,
        key: bool,
        activated: bool,
        repository_root: Path = ROOT,
    ) -> dict:
        return assess_candidate_coverage(
            evidence,
            evidence_sha256=self.evidence_sha256,
            repository_root=repository_root,
            eia_api_key_available=key,
            ercot_eia_fallback_activated=activated,
        )

    @staticmethod
    def placeholder_product(name: str) -> dict:
        return {
            "name": name,
            "source_type": "authoritative_native",
            "start_utc": "2025-09-01T00:00:00Z",
            "end_utc": "2026-05-01T00:00:00Z",
            "missing_intervals": 0,
            "duplicate_intervals": 0,
            "raw_hash_manifest": {},
            "canonical_file": {},
        }

    def test_candidate_is_blocked_until_ercot_public_path_is_staged(self) -> None:
        assessment = self.assess(
            self.evidence,
            key=False,
            activated=False,
        )
        self.assertEqual(assessment["status"], "BLOCKED")
        self.assertIn("ERCOT_LZ_NORTH", assessment["blockers"])
        self.assertIn("243", assessment["blockers"]["ERCOT_LZ_NORTH"])

    def test_eia_key_without_explicit_activation_is_not_used(self) -> None:
        assessment = self.assess(
            self.evidence,
            key=True,
            activated=False,
        )
        self.assertIn("ERCOT_LZ_NORTH", assessment["blockers"])
        self.assertFalse(assessment["ercot_eia_fallback_activated"])

    def test_explicit_eia_sensitivity_cannot_clear_primary_gate(
        self,
    ) -> None:
        evidence = copy.deepcopy(self.evidence)
        for market, record in evidence["markets"].items():
            if market != "ERCOT_LZ_NORTH":
                record["candidate_complete"] = True
                record["reason"] = "test-complete"
                record["candidate_complete"] = False
        assessment = self.assess(
            evidence, key=True, activated=True
        )
        self.assertEqual(assessment["status"], "BLOCKED")
        self.assertEqual(
            assessment["markets"]["ERCOT_LZ_NORTH"]["status"],
            "BLOCKED",
        )
        self.assertIn("ERCOT_LZ_NORTH", assessment["blockers"])

    def test_complete_boolean_without_product_proof_fails_closed(self) -> None:
        evidence = copy.deepcopy(self.evidence)
        evidence["markets"]["NYISO_NYC_J"]["candidate_complete"] = True
        with self.assertRaisesRegex(ValueError, "lacks product proof"):
            self.assess(evidence, key=False, activated=False)

    def test_eia_sensitivity_flag_without_key_keeps_primary_reason(self) -> None:
        assessment = self.assess(
            self.evidence,
            key=False,
            activated=True,
        )
        self.assertIn("243", assessment["blockers"]["ERCOT_LZ_NORTH"])

    def test_eia_proof_is_never_a_trusted_primary_product(self) -> None:
        evidence = copy.deepcopy(self.evidence)
        for market, record in evidence["markets"].items():
            record["candidate_complete"] = False
        ercot = evidence["markets"]["ERCOT_LZ_NORTH"]
        ercot["candidate_complete"] = True
        ercot["required_products"] = [
            "ercot_eia930_physical",
            "ercot_lz_north_dam",
        ]
        ercot["coverage_proof"] = {
            "products": [
                {
                    "name": "ercot_eia930_physical",
                    "source_type": "EIA-930",
                    "balancing_authority": "ERCO",
                    "start_utc": evidence["candidate_start_utc"],
                    "end_utc": evidence["candidate_end_utc"],
                    "missing_intervals": 0,
                    "duplicate_intervals": 0,
                    "raw_hash_manifest": {},
                    "canonical_file": {},
                },
                self.placeholder_product("ercot_lz_north_dam"),
            ]
        }
        with self.assertRaisesRegex(ValueError, "not trusted"):
            self.assess(evidence, key=False, activated=False)
        with self.assertRaisesRegex(ValueError, "not trusted"):
            self.assess(evidence, key=False, activated=True)
        with self.assertRaisesRegex(ValueError, "not trusted"):
            self.assess(evidence, key=True, activated=True)

    def test_truthy_fake_hash_manifest_is_rejected(self) -> None:
        evidence = copy.deepcopy(self.evidence)
        nyiso = evidence["markets"]["NYISO_NYC_J"]
        nyiso["candidate_complete"] = True
        nyiso["required_products"] = [
            "nyiso_zone_j_pal",
            "nyiso_zone_j_dam",
            "nyiso_system_fuelmix",
        ]
        nyiso["coverage_proof"] = {
            "products": [
                {
                    "name": "nyiso_zone_j_pal",
                    "source_type": "authoritative_native",
                    "start_utc": evidence["candidate_start_utc"],
                    "end_utc": evidence["candidate_end_utc"],
                    "missing_intervals": 0,
                    "duplicate_intervals": 0,
                    "raw_hash_manifest": "not-a-real-manifest",
                    "canonical_file": {},
                },
                self.placeholder_product("nyiso_zone_j_dam"),
                self.placeholder_product("nyiso_system_fuelmix"),
            ]
        }
        with self.assertRaisesRegex(ValueError, "reference is invalid"):
            self.assess(evidence, key=False, activated=False)

    def test_missing_product_timestamps_fail_closed(self) -> None:
        evidence = copy.deepcopy(self.evidence)
        nyiso = evidence["markets"]["NYISO_NYC_J"]
        nyiso["candidate_complete"] = True
        nyiso["required_products"] = [
            "nyiso_zone_j_pal",
            "nyiso_zone_j_dam",
            "nyiso_system_fuelmix",
        ]
        nyiso["coverage_proof"] = {
            "products": [
                {
                    "name": "nyiso_zone_j_pal",
                    "source_type": "authoritative_native",
                    "start_utc": None,
                    "end_utc": evidence["candidate_end_utc"],
                    "missing_intervals": 0,
                    "duplicate_intervals": 0,
                    "raw_hash_manifest": {},
                    "canonical_file": {},
                },
                self.placeholder_product("nyiso_zone_j_dam"),
                self.placeholder_product("nyiso_system_fuelmix"),
            ]
        }
        with self.assertRaisesRegex(ValueError, "timestamps are missing"):
            self.assess(evidence, key=False, activated=False)

    def test_unrelated_file_cannot_serve_as_canonical_coverage(self) -> None:
        evidence = copy.deepcopy(self.evidence)
        nyiso = evidence["markets"]["NYISO_NYC_J"]
        nyiso["candidate_complete"] = True
        nyiso["required_products"] = [
            "nyiso_zone_j_pal",
            "nyiso_zone_j_dam",
            "nyiso_system_fuelmix",
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            unrelated = root / "README.md"
            unrelated.write_text("not market evidence", encoding="utf-8")
            raw_path = (
                root / "data" / "energy_model_v3" / "raw"
                / "NYISO_NYC_J" / "pal.csv"
            )
            raw_path.parent.mkdir(parents=True)
            raw_path.write_text("raw PAL evidence", encoding="utf-8")
            raw_manifest = (
                root / "data" / "energy_model_v3" / "provenance"
                / "raw_manifests" / "NYISO_NYC_J" / "pal.json"
            )
            raw_manifest.parent.mkdir(parents=True)
            raw_payload = {
                "schema_version": "energy-model-v3-raw-hash-manifest",
                "product": "nyiso_zone_j_pal",
                "source": "NYISO MIS",
                "feed": "pal",
                "location": "Zone J PTID 61761",
                "start_utc": evidence["candidate_start_utc"],
                "end_utc": evidence["candidate_end_utc"],
                "files": {
                    str(raw_path.relative_to(root)).replace("\\", "/"):
                    hashlib.sha256(
                        raw_path.read_bytes()
                    ).hexdigest()
                },
            }
            raw_manifest.write_text(
                json.dumps(raw_payload), encoding="utf-8"
            )
            nyiso["coverage_proof"] = {
                "products": [
                    {
                        "name": "nyiso_zone_j_pal",
                        "source_type": "authoritative_native",
                        "start_utc": evidence["candidate_start_utc"],
                        "end_utc": evidence["candidate_end_utc"],
                        "missing_intervals": 0,
                        "duplicate_intervals": 0,
                        "raw_hash_manifest": {
                            "path": str(
                                raw_manifest.relative_to(root)
                            ).replace("\\", "/"),
                            "sha256": hashlib.sha256(
                                raw_manifest.read_bytes()
                            ).hexdigest(),
                        },
                        "canonical_file": {
                            "path": "README.md",
                            "sha256": hashlib.sha256(
                                unrelated.read_bytes()
                            ).hexdigest(),
                            "cadence_minutes": 60,
                        },
                    },
                    self.placeholder_product("nyiso_zone_j_dam"),
                    self.placeholder_product("nyiso_system_fuelmix"),
                ]
            }
            with self.assertRaisesRegex(
                ValueError, "outside its market"
            ):
                self.assess(
                    evidence,
                    key=False,
                    activated=False,
                    repository_root=root,
                )


if __name__ == "__main__":
    unittest.main()
