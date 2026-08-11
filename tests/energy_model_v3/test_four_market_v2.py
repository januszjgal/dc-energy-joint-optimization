from __future__ import annotations

import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from energy_model_v3.four_market_v2 import (
    CELLS,
    COMPUTE_CAPACITY,
    MARKET_TO_CELL,
    MARKETS,
    factory_identity_paths,
    make_envelope_off_env,
    make_envelope_on_env,
)
from ramp_rl.contract import EnvRequest, RampEnvAdapter
from ramp_rl.evaluation import _cluster_bootstrap_interval
from ramp_rl.runner import source_bundle_hash
from scripts.build_four_market_v2_factory import _raw_arrivals, build, validate


ROOT = Path(__file__).resolve().parents[2]
FACTORY_ROOT = ROOT / "output" / "four_market_v2" / "factory"
SOURCE_ROOT = ROOT / "data" / "four_market_v2"


class FourMarketV2DesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        build()
        cls.manifests = {
            variant: json.loads(
                (FACTORY_ROOT / variant / "factory_manifest.json").read_text()
            )
            for variant in ("envelope_on", "envelope_off")
        }
        cls.window = sorted(cls.manifests["envelope_off"]["windows"]["validation"])[0]

    def make_env(self, variant: str):
        factory = make_envelope_on_env if variant == "envelope_on" else make_envelope_off_env
        return factory(
            EnvRequest(
                split="validation",
                seed=4101,
                window_id=self.window,
                training=False,
            )
        )

    def test_mapping_k1_power_and_spaces(self) -> None:
        env = self.make_env("envelope_on")
        try:
            observation, _ = env.reset(seed=4101)
            current = env._current
            self.assertEqual(tuple((site.market_id, site.site_id[-1]) for site in current.sites), MARKET_TO_CELL)
            self.assertEqual(tuple(current.panel.markets), tuple(sorted(MARKETS)))
            self.assertEqual(env.action_space.shape, (9,))
            self.assertEqual(observation.shape, (128,))
            self.assertTrue(all(site.compute_capacity == COMPUTE_CAPACITY for site in current.sites))
            self.assertTrue(all(site.rated_power_mw == 500.0 for site in current.sites))
            self.assertEqual(sum(site.rated_power_mw for site in current.sites), 2000.0)
            site = current.sites[0]
            self.assertAlmostEqual(
                site.power_mw(1.0),
                500.0 * (site.idle_power_fraction + site.dynamic_power_fraction),
            )
        finally:
            env.close()

    def test_variant_workload_semantics_and_deadlines(self) -> None:
        on = self.make_env("envelope_on")
        off = self.make_env("envelope_off")
        try:
            on_workload = on._current.workload
            off_workload = off._current.workload
            times = off._current.panel.timestamps[3:-3]
            raw_service, raw_batch = _raw_arrivals(times)
            np.testing.assert_allclose(off_workload.service_arrivals[:-3], raw_service)
            np.testing.assert_allclose(off_workload.batch_arrivals[:-3], raw_batch)
            self.assertLessEqual(
                float(
                    (
                        off_workload.service_arrivals[:-3]
                        + off_workload.batch_arrivals[:-3]
                    ).sum(axis=1).max()
                ),
                4.0,
            )
            np.testing.assert_allclose(
                on_workload.service_arrivals[:-3] + on_workload.batch_arrivals[:-3],
                raw_service + raw_batch,
            )
            self.assertLessEqual(float(on_workload.service_arrivals[:-3].sum(axis=1).max()), 3.0)
            self.assertLessEqual(float(on_workload.batch_arrivals[:-3].sum(axis=1).max()), 0.4)
            self.assertEqual(off_workload.batch_deadline_hours[0].tolist(), [2, 1, 2, 3])
            np.testing.assert_array_equal(off_workload.service_arrivals[-3:], 0.0)
            np.testing.assert_array_equal(off_workload.batch_arrivals[-3:], 0.0)
        finally:
            on.close()
            off.close()

    def test_raw_off_is_immediately_feasible_and_test_is_blocked(self) -> None:
        env = self.make_env("envelope_off")
        try:
            env.reset(seed=4101)
            while True:
                _, _, terminated, _, info = env.step(env.evaluation_action("status_quo"))
                if terminated:
                    self.assertEqual(info["batch_unfinished"], 0.0)
                    self.assertEqual(info["certificate_violations"], 0)
                    break
        finally:
            env.close()
        with self.assertRaisesRegex(ValueError, "train and validation only"):
            make_envelope_off_env(
                EnvRequest(split="test", seed=4101, window_id="forbidden", training=False)
            )

    def test_active_environment_satisfies_training_contract(self) -> None:
        request = EnvRequest(
            split="validation",
            seed=4101,
            window_id=self.window,
            training=False,
        )
        adapter = RampEnvAdapter(make_envelope_off_env(request), request)
        try:
            observation, info = adapter.reset(seed=request.seed)
            self.assertEqual(observation.shape, (128,))
            self.assertEqual(adapter.action_space.shape, (9,))
            self.assertEqual(info["episode_context"]["workload_cells"], list(CELLS))
            terminal = None
            while True:
                action = adapter.evaluation_action("status_quo")
                _, _, terminated, truncated, terminal = adapter.step(action)
                self.assertFalse(truncated)
                if terminated:
                    break
            self.assertTrue(terminal["tail_complete"])
            self.assertEqual(terminal["service_unserved"], 0.0)
            self.assertEqual(terminal["batch_expired"], 0.0)
            self.assertEqual(terminal["terminal_work"], 0.0)
            self.assertEqual(terminal["certificate_violations"], 0)
        finally:
            adapter.close()

    def test_factory_is_self_contained_and_holdout_free(self) -> None:
        for manifest in self.manifests.values():
            rendered = json.dumps(manifest)
            for forbidden in ("ar" + "chive/", "V" + "4R"):
                self.assertNotIn(forbidden, rendered)
            self.assertEqual(manifest["split_periods"]["test"], [])
            self.assertFalse(manifest["sealed_test_access"])
            self.assertEqual(manifest["workload_cells"], list(CELLS))

    def test_rebuild_uses_only_active_source(self) -> None:
        source_manifest = SOURCE_ROOT / "source_manifest.json"
        source = json.loads(source_manifest.read_text(encoding="utf-8"))
        self.assertEqual(len(source["windows"]["train"]), 114)
        self.assertEqual(len(source["windows"]["validation"]), 28)
        shutil.rmtree(FACTORY_ROOT)
        rebuilt = build()
        self.assertEqual(rebuilt, validate())
        for variant in ("envelope_on", "envelope_off"):
            manifest = json.loads(
                (FACTORY_ROOT / variant / "factory_manifest.json").read_text()
            )
            self.assertEqual(
                manifest["physical_source"]["manifest_path"],
                "data/four_market_v2/source_manifest.json",
            )

    def test_owned_source_and_runtime_are_clean(self) -> None:
        files = (
            ROOT / "energy_model_v3" / "four_market_v2.py",
            ROOT / "scripts" / "build_four_market_v2_factory.py",
            ROOT / "ramp_rl" / "runner.py",
            SOURCE_ROOT / "source_manifest.json",
        )
        forbidden = (
            "four_market_" + "v1",
            "ar" + "chive",
            "V" + "4R",
        )
        for path in files:
            text = path.read_text(encoding="utf-8")
            for value in forbidden:
                self.assertNotIn(value, text, f"{path} contains {value}")

    def test_runner_bundle_uses_selected_v2_factory_and_protocol(self) -> None:
        protocol = {
            "_path": "env/protocols/four_market_v2_envelope_on.yaml",
        }
        bundle = source_bundle_hash(make_envelope_on_env, protocol)
        self.assertEqual(len(bundle), 64)
        self.assertNotEqual(bundle, "0" * 64)
        self.assertEqual(
            factory_identity_paths(make_envelope_on_env),
            (
                SOURCE_ROOT / "source_manifest.json",
                FACTORY_ROOT / "envelope_on" / "factory_manifest.json",
            ),
        )

    def test_single_month_bootstrap_is_marked_non_estimable(self) -> None:
        result = _cluster_bootstrap_interval(
            [1.0, 2.0],
            ["2026-02", "2026-02"],
            unit="month",
            draws=100,
        )
        self.assertFalse(result["estimable"])
        self.assertEqual(result["group_count"], 1)
        self.assertNotIn("lower_95", result)
        self.assertNotIn("upper_95", result)

    def test_v2_import_does_not_eagerly_load_legacy_factory(self) -> None:
        factory_module = "env.ramp_v6." + "factory"
        command = (
            "import sys; import energy_model_v3.four_market_v2; "
            f"assert {factory_module!r} not in sys.modules"
        )
        subprocess.run([sys.executable, "-c", command], check=True, cwd=ROOT)


if __name__ == "__main__":
    unittest.main()
