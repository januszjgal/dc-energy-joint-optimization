"""Strict checks for the persisted real six-market ramp factory."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from energy_model_v3.ramp_factory import (
    ACQUISITION_MANIFEST_SHA256,
    OUTPUT_ROOT,
    PANEL_MANIFEST_SHA256,
    sha256_file,
    sha256_json,
    validate_factory,
)
from env.ramp_v6.factory import make_energy_model_v3_env
from env.ramp_v6.panel import CanonicalMarketPanel
from ramp_rl.contract import EnvRequest, RampEnvAdapter

ROOT = Path(__file__).resolve().parents[2]


class LiveFactoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest_path = OUTPUT_ROOT / "factory_manifest.json"
        if not cls.manifest_path.is_file():
            raise RuntimeError("real energy-v3 ramp factory handoff is missing")
        cls.payload = json.loads(cls.manifest_path.read_text(encoding="utf-8"))

    def test_verified_source_manifests_and_complete_split_maps(self) -> None:
        self.assertEqual(
            self.payload["source_panel_manifest_sha256"], PANEL_MANIFEST_SHA256
        )
        self.assertEqual(
            self.payload["raw_acquisition_manifest_sha256"],
            ACQUISITION_MANIFEST_SHA256,
        )
        self.assertGreaterEqual(len(self.payload["windows"]["train"]), 100)
        self.assertEqual(len(self.payload["windows"]["validation"]), 28)
        self.assertGreaterEqual(len(self.payload["windows"]["test"]), 60)
        self.assertEqual(
            self.payload["split_periods"]["test"], ["2026-03", "2026-04"]
        )

    def test_all_content_hashes_causality_scales_and_envelopes(self) -> None:
        summary = validate_factory()
        self.assertEqual(
            summary["frozen_stats_sha256"], self.payload["frozen_stats_sha256"]
        )
        for split, windows in self.payload["windows"].items():
            for window in windows.values():
                root = ROOT / window["artifact_root"]
                panel = CanonicalMarketPanel.from_csv(root / "canonical_panel.csv")
                fixture = json.loads(
                    (root / "fixture.json").read_text(encoding="utf-8")
                )
                self.assertEqual(len(panel.markets), 6)
                self.assertTrue(
                    (
                        panel.frame["forecast_issue_time_utc"]
                        <= panel.frame["timestamp_utc"]
                    ).all()
                )
                self.assertEqual(
                    sha256_json(fixture["frozen_stats"]),
                    self.payload["frozen_stats_sha256"],
                )
                sites = fixture["sites"]
                self.assertAlmostEqual(
                    sum(float(site["rated_power_mw"]) for site in sites), 600.0
                )
                service = np.asarray(fixture["workload"]["service_arrivals"])
                batch = np.asarray(fixture["workload"]["batch_arrivals"])
                warm = np.asarray(fixture["workload"]["warm_power_mw"])
                self.assertLessEqual(float(service.sum(axis=1).max()), 4.5 + 1e-12)
                self.assertLessEqual(float(batch.sum(axis=1).max()), 0.6 + 1e-12)
                self.assertEqual(warm.shape, (3, 6))
                self.assertTrue((warm > 0.0).all())
                self.assertTrue((warm <= 100.0).all())

    def test_factory_environment_contract_for_every_split(self) -> None:
        for split in ("train", "validation", "test"):
            window = sorted(self.payload["windows"][split])[0]
            request = EnvRequest(
                split=split,
                seed=2601,
                window_id=window,
                training=split == "train",
            )
            adapter = RampEnvAdapter(make_energy_model_v3_env(request), request)
            observation, info = adapter.reset(seed=2601)
            self.assertEqual(adapter.action_space.shape, (13,))
            self.assertEqual(adapter.contract["decision_steps"], 27)
            self.assertEqual(adapter.contract["active_arrival_steps"], 24)
            self.assertFalse(
                info["episode_context"]["future_realized_features_exposed"]
            )
            self.assertTrue(np.isfinite(observation).all())
            adapter.close()


if __name__ == "__main__":
    unittest.main()
