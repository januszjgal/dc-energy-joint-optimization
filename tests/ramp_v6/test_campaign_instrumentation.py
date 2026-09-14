"""Focused tests for the locked long-run campaign instrumentation."""

from __future__ import annotations

import csv
import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ramp_rl.campaign_statistics import (
    exact_two_sided_wilcoxon,
    paired_bootstrap_intervals,
    paired_seed_summary,
)
from ramp_rl.contract import (
    CONTRACT_VERSION, SEMANTIC_ACTION_ID, EnvRequest, environment_identity,
)
from env.ramp_v6.objective import JointObjective
from ramp_rl.runner import (
    LEARNING_CURVE_COLUMNS,
    LearningCurveWriter,
    milestone_interactions,
    run_training,
    safe_boundary_quantum,
    select_latest_checkpoint,
)
from scripts.aggregate_four_market_v2_campaign import aggregate
from scripts.run_four_market_v2_campaign import (
    DEFAULT_WORKERS,
    REQUESTED_TIMESTEPS,
    SEEDS,
)
import scripts.aggregate_four_market_v2_campaign as aggregation
import scripts.run_four_market_v2_campaign as campaign


ROOT = Path(__file__).resolve().parents[2]


class _TinyRampEnv(gym.Env[np.ndarray, np.ndarray]):
    observation_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
    action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def __init__(self) -> None:
        self.step_count = 0

    def ramp_rl_contract(self) -> dict[str, object]:
        return {
            "version": CONTRACT_VERSION,
            "semantic_feasible_action": True,
            "semantic_action_id": SEMANTIC_ACTION_ID,
            "history_hours": 1,
            "terminal_tail_hours": 0,
            "grid_observation_lag_hours": 1,
            "actual_terminal": True,
            "decision_steps": 3,
            "action_shape": [1],
            "observation_shape": [2],
            "objective": JointObjective().as_dict(),
            "factory_id": "tiny-test",
        }

    def reset(self, *, seed: int | None = None, options: dict[str, object] | None = None):
        super().reset(seed=seed)
        self.step_count = 0
        split = (options or {})["split"]
        return np.zeros(2, dtype=np.float32), {
            "episode_context": {
                "split": split,
                "future_realized_features_exposed": False,
            }
        }

    def step(self, action: np.ndarray):
        self.step_count += 1
        terminal = self.step_count == 3
        info = {
            "ramp_h1_adjusted": 0.0,
            "abs_adjusted_ramp_h1_fraction_s_per_hour_by_market": [0.0],
            "physical_ramp_market_order": ["tiny"],
            "incremental_ramp_impact": 0.0,
            "service_unserved": 0.0,
            "batch_unfinished": 0.0,
            "batch_expired": 0.0,
            "certificate_violations": 0.0,
            "emergency_feasibility": 0.0,
            "semantic_adjustment_l2": 0.0,
            "semantic_adjustment_applied": False,
            "semantic_adjustment_coordinate_id": "tiny",
            "semantic_adjustment_units": 0.0,
            "terminal_work": 0.0,
            "actual_terminal": terminal,
            "scalar_reward": -1.0,
            "ramp_reward": -0.5,
            "peak_reward": -0.5,
            "ramp_squared_score": 1.0,
            "peak_normalized_increment": 1.0,
            "peak_impact_increment": 1.0,
        }
        return np.zeros(2, dtype=np.float32), -1.0, terminal, False, info


def _tiny_factory(request: EnvRequest) -> gym.Env:
    return _TinyRampEnv()


def _tiny_protocol() -> dict[str, object]:
    return {
        "training": {
            "vectorized_environments": 2,
            "checkpoint_rollouts": 3,
            "ppo": {
                "net_arch": [32, 32],
                "learning_rate": 5e-4,
                "batch_size": 16,
                "n_steps": 16,
                "n_epochs": 2,
                "gae_lambda": 0.95,
                "gamma": 0.99,
            },
        }
    }


def _tiny_training_summary(seed: int, target: int = 96) -> dict[str, object]:
    config = dict(_tiny_protocol()["training"]["ppo"])
    return {
        "seed": seed,
        "requested_interactions": target,
        "effective_interactions": 96,
        "n_envs": 2,
        "ppo_config": config,
        "safe_quantum": 96,
        "environment_identity": environment_identity(_TinyRampEnv().ramp_rl_contract()),
    }


def _completed_campaign_summary(
    seed: int,
    windows: list[str],
    training_geometry: dict[str, int],
    training_identity: dict[str, object] | None = None,
) -> dict[str, object]:
    summary = {
        "seed": seed,
        "requested_interactions": REQUESTED_TIMESTEPS,
        "effective_interactions": training_geometry["effective_interactions"],
        "safe_boundary_interactions": training_geometry["safe_quantum"],
        "validation_window_ids": windows,
        "validation": {
            "objective": campaign._environment_identity()["objective"],
            "episode_count": 1,
            "day_count": 31,
            "step_count": 744,
            "mean_joint_J": 1.0,
            "mean_monthly_ramp_impact": 744.0,
            "mean_policy_native_relative_incremental_ramp_impact": 1.0,
            "mean_incremental_ramp_impact": 1.0,
            "status_quo_comparison": {
                "policy_native_relative_mean_incremental_ramp_impact": 1.0,
                "status_quo_native_relative_mean_incremental_ramp_impact": 2.0,
                "policy_minus_status_quo_mean_incremental_ramp_impact": -1.0,
                "improvement": 1.0,
                "policy_J": 1.0,
                "status_quo_J": 2.0,
                "improvement_J": 1.0,
                "policy_minus_status_quo_joint_J": -1.0,
            },
        },
    }
    if training_identity is not None:
        summary["training_identity"] = training_identity
    return summary


def _campaign_training_identity(
    seed: int, training_geometry: dict[str, int], ppo_config: dict[str, object]
) -> dict[str, object]:
    return {
        "seed": seed,
        "requested_interactions": REQUESTED_TIMESTEPS,
        "effective_interactions": training_geometry["effective_interactions"],
        "n_envs": training_geometry["n_envs"],
        "ppo_config": ppo_config,
        "safe_quantum": training_geometry["safe_quantum"],
        "environment_identity": campaign._environment_identity(),
    }


def _write_completed_training(
    model_root: Path, seed: int, training_identity: dict[str, object]
) -> None:
    checkpoint = model_root / "ppo" / f"seed-{seed}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.zip").write_bytes(b"model")
    (checkpoint / "vecnormalize.pkl").write_bytes(b"normalization")
    (checkpoint / "training_summary.json").write_text(
        json.dumps(training_identity), encoding="utf-8"
    )


class CampaignInstrumentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from scripts.build_four_market_v2_factory import build

        build()

    def test_locked_campaign_geometry(self) -> None:
        self.assertEqual(SEEDS, tuple(range(4101, 4111)))
        self.assertEqual(REQUESTED_TIMESTEPS, 2_000_000)
        self.assertEqual(DEFAULT_WORKERS, 5)

    def test_learning_curve_writer_persists_raw_metric_schema(self) -> None:
        directory = ROOT / ".test-learning-curve"
        shutil.rmtree(directory, ignore_errors=True)
        try:
            path = directory / "learning_curve.csv"
            writer = LearningCurveWriter(path)
            writer.write({
                **dict.fromkeys(LEARNING_CURVE_COLUMNS, 0.0),
                "interaction_count": 2048,
                "mean_raw_ramp_reward": -1.5,
                "mean_raw_incremental_ramp_impact": 1.5,
                "episode_count": 4,
                "mean_completed_episode_return": -40.5,
                "elapsed_seconds": 2.0,
            })
            writer.close()
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(tuple(rows[0]), LEARNING_CURVE_COLUMNS)
            self.assertEqual(rows[0]["interaction_count"], "2048")
            self.assertEqual(rows[0]["mean_raw_incremental_ramp_impact"], "1.5")
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_learning_curve_truncates_to_checkpoint_before_appending(self) -> None:
        directory = ROOT / ".test-learning-curve-resume"
        shutil.rmtree(directory, ignore_errors=True)
        try:
            path = directory / "learning_curve.csv"
            path.parent.mkdir()
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=LEARNING_CURVE_COLUMNS)
                writer.writeheader()
                for interaction in (32, 64, 96):
                    writer.writerow({
                        **dict.fromkeys(LEARNING_CURVE_COLUMNS, 0.0),
                        "interaction_count": interaction,
                        "mean_raw_ramp_reward": -1.0,
                        "mean_raw_incremental_ramp_impact": 0.0,
                        "episode_count": 2,
                        "mean_completed_episode_return": -3.0,
                        "elapsed_seconds": 1.0,
                    })
            writer = LearningCurveWriter(path, interaction_limit=64)
            writer.write({
                **dict.fromkeys(LEARNING_CURVE_COLUMNS, 0.0),
                "interaction_count": 96,
                "mean_raw_ramp_reward": -1.0,
                "mean_raw_incremental_ramp_impact": 0.0,
                "episode_count": 2,
                "mean_completed_episode_return": -3.0,
                "elapsed_seconds": 2.0,
            })
            writer.close()
            with path.open(newline="", encoding="utf-8") as handle:
                self.assertEqual(
                    [int(row["interaction_count"]) for row in csv.DictReader(handle)],
                    [32, 64, 96],
                )
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_safe_boundary_and_milestone_mapping_follow_rollout_geometry(self) -> None:
        quantum = safe_boundary_quantum(n_envs=4, n_steps=512, checkpoint_rollouts=25)
        self.assertEqual(quantum, 51_200)
        self.assertEqual(
            milestone_interactions(
                (110_592, 500_000, 1_000_000, 1_500_000, 2_000_000),
                safe_quantum=quantum,
            ),
            {
                110_592: 153_600,
                500_000: 512_000,
                1_000_000: 1_024_000,
                1_500_000: 1_536_000,
                2_000_000: 2_048_000,
            },
        )
        self.assertEqual(safe_boundary_quantum(n_envs=2, n_steps=16, checkpoint_rollouts=3), 96)

    def test_resume_selection_accepts_complete_interrupted_swap(self) -> None:
        directory = ROOT / ".test-latest-selection"
        shutil.rmtree(directory, ignore_errors=True)
        try:
            for name, interactions in (("latest", 96), ("latest.previous", 192)):
                checkpoint = directory / name
                checkpoint.mkdir(parents=True)
                (checkpoint / "model.zip").write_bytes(b"model")
                (checkpoint / "vecnormalize.pkl").write_bytes(b"normalization")
                (checkpoint / "state.json").write_text(
                    json.dumps({"interaction_count": interactions}), encoding="utf-8"
                )
            selected = select_latest_checkpoint(directory)
            self.assertIsNotNone(selected)
            assert selected is not None
            self.assertEqual(selected[0].name, "latest.previous")
            self.assertEqual(selected[1]["interaction_count"], 192)
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_completed_training_is_reused_after_validating_training_identity(self) -> None:
        directory = ROOT / ".test-completed-training"
        shutil.rmtree(directory, ignore_errors=True)
        try:
            directory.mkdir()
            (directory / "model.zip").write_bytes(b"model")
            (directory / "vecnormalize.pkl").write_bytes(b"normalization")
            expected = _tiny_training_summary(7)
            (directory / "training_summary.json").write_text(json.dumps(expected), encoding="utf-8")
            self.assertEqual(
                run_training(
                    factory=_tiny_factory,
                    protocol=_tiny_protocol(),
                    seed=7,
                    target_timesteps=96,
                    output_dir=directory,
                    fixture_profile=True,
                ),
                expected,
            )
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_completed_training_rejects_incompatible_identity(self) -> None:
        directory = ROOT / ".test-incompatible-completed-training"
        shutil.rmtree(directory, ignore_errors=True)
        try:
            directory.mkdir()
            (directory / "model.zip").write_bytes(b"model")
            (directory / "vecnormalize.pkl").write_bytes(b"normalization")
            incompatible = _tiny_training_summary(7)
            incompatible["requested_interactions"] = 192
            (directory / "training_summary.json").write_text(
                json.dumps(incompatible), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "does not match current requested_interactions"):
                run_training(
                    factory=_tiny_factory,
                    protocol=_tiny_protocol(),
                    seed=7,
                    target_timesteps=96,
                    output_dir=directory,
                    fixture_profile=True,
                )
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_completed_and_resumable_training_reject_changed_objective(self) -> None:
        class DifferentObjectiveEnv(_TinyRampEnv):
            def ramp_rl_contract(self):
                contract = super().ramp_rl_contract()
                contract["objective"] = JointObjective(ramp_weight=1.0, peak_weight=0.0).as_dict()
                return contract

        directory = ROOT / ".test-changed-objective"
        shutil.rmtree(directory, ignore_errors=True)
        try:
            run_training(
                factory=_tiny_factory, protocol=_tiny_protocol(), seed=23,
                target_timesteps=96, output_dir=directory, fixture_profile=True,
            )
            with self.assertRaisesRegex(RuntimeError, "current environment_identity"):
                run_training(
                    factory=lambda request: DifferentObjectiveEnv(),
                    protocol=_tiny_protocol(), seed=23,
                    target_timesteps=96, output_dir=directory, fixture_profile=True,
                )
            for name in ("model.zip", "vecnormalize.pkl", "training_summary.json"):
                (directory / name).unlink()
            with self.assertRaisesRegex(RuntimeError, "checkpoint environment_identity"):
                run_training(
                    factory=lambda request: DifferentObjectiveEnv(),
                    protocol=_tiny_protocol(), seed=23,
                    target_timesteps=96, output_dir=directory, fixture_profile=True,
                )
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_full_campaign_restarts_from_a_valid_completed_subset(self) -> None:
        directory = ROOT / ".test-campaign-subset-restart"
        shutil.rmtree(directory, ignore_errors=True)
        windows = ["2025-05"]
        geometry = {"train": [], "validation": windows}
        training_geometry = {
            "n_envs": 4,
            "n_steps": 512,
            "checkpoint_rollouts": 25,
            "safe_quantum": 51_200,
            "effective_interactions": 2_048_000,
        }
        submitted: list[int] = []

        class ImmediateExecutor:
            def __init__(self, *, max_workers: int) -> None:
                self.max_workers = max_workers

            def __enter__(self):
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def map(self, function, values):
                values = list(values)
                submitted.extend(values)
                return [function(value) for value in values]

        try:
            completed_path = directory / f"seed-{SEEDS[0]}" / "summary.json"
            completed_path.parent.mkdir(parents=True)
            completed_path.write_text(
                json.dumps(
                    _completed_campaign_summary(SEEDS[0], windows, training_geometry)
                ),
                encoding="utf-8",
            )
            with (
                patch.object(campaign, "OUTPUT_ROOT", directory),
                patch.object(campaign, "_validate_campaign_geometry", return_value=geometry),
                patch.object(
                    campaign,
                    "_campaign_training_geometry",
                    return_value=training_geometry,
                ),
                patch.object(
                    campaign,
                    "_load_completed_seed_summary",
                    side_effect=lambda seed, *_args, **_kwargs: {"seed": seed},
                ),
                patch.object(campaign, "ProcessPoolExecutor", ImmediateExecutor),
                patch.object(
                    campaign,
                    "_run_seed_worker",
                    side_effect=lambda seed: {"seed": seed},
                ),
            ):
                results = campaign.run_campaign(workers=5)
            self.assertEqual([result["seed"] for result in results], list(SEEDS))
            self.assertEqual(submitted, list(SEEDS[1:]))
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_completed_campaign_rejects_changed_ppo_config(self) -> None:
        directory = ROOT / ".test-campaign-ppo-identity"
        shutil.rmtree(directory, ignore_errors=True)
        seed = SEEDS[0]
        windows = ["2025-05"]
        training_geometry = {
            "n_envs": 2,
            "n_steps": 16,
            "checkpoint_rollouts": 3,
            "safe_quantum": 96,
            "effective_interactions": 2_000_064,
        }
        original_config = dict(_tiny_protocol()["training"]["ppo"])
        identity = _campaign_training_identity(seed, training_geometry, original_config)
        try:
            summary_path = directory / f"seed-{seed}" / "summary.json"
            summary_path.parent.mkdir(parents=True)
            summary_path.write_text(
                json.dumps(
                    _completed_campaign_summary(
                        seed, windows, training_geometry, identity
                    )
                ),
                encoding="utf-8",
            )
            model_root = directory / "models"
            _write_completed_training(model_root, seed, identity)
            changed_protocol = _tiny_protocol()
            changed_protocol["training"]["ppo"]["learning_rate"] = 1e-3
            with patch.object(campaign, "_protocol", return_value=changed_protocol):
                with self.assertRaisesRegex(
                    RuntimeError, "does not match current ppo_config"
                ):
                    campaign._load_completed_seed_summary(
                        seed,
                        {"train": [], "validation": windows},
                        training_geometry,
                        output_root=directory,
                        model_root=model_root,
                    )
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_aggregator_rejects_stale_identity_with_all_summary_paths(self) -> None:
        directory = ROOT / ".test-stale-identity-aggregation"
        shutil.rmtree(directory, ignore_errors=True)
        model_root = directory / "models"
        windows = ["2025-05"]
        geometry = {"train": [], "validation": windows}
        training_geometry = {
            "n_envs": 2,
            "n_steps": 16,
            "checkpoint_rollouts": 3,
            "safe_quantum": 96,
            "effective_interactions": 2_000_064,
        }
        current_config = dict(_tiny_protocol()["training"]["ppo"])
        try:
            for seed in SEEDS:
                training_identity = _campaign_training_identity(
                    seed, training_geometry, current_config
                )
                if seed == SEEDS[-1]:
                    stale_config = dict(current_config)
                    stale_config["net_arch"] = [64, 64]
                    training_identity = _campaign_training_identity(
                        seed, training_geometry, stale_config
                    )
                summary_path = directory / f"seed-{seed}" / "summary.json"
                summary_path.parent.mkdir(parents=True)
                summary_path.write_text(
                    json.dumps(
                        _completed_campaign_summary(
                            seed, windows, training_geometry, training_identity
                        )
                    ),
                    encoding="utf-8",
                )
                _write_completed_training(model_root, seed, training_identity)
            with (
                patch.object(campaign, "MODEL_ROOT", model_root),
                patch.object(campaign, "_protocol", return_value=_tiny_protocol()),
                patch.object(aggregation, "_validate_campaign_geometry", return_value=geometry),
                patch.object(
                    aggregation,
                    "_campaign_training_geometry",
                    return_value=training_geometry,
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "does not match current ppo_config"
                ):
                    aggregation._require_complete_summaries(directory)
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_two_stage_safe_boundary_resume_smoke(self) -> None:
        directory = ROOT / ".test-safe-boundary-smoke"
        shutil.rmtree(directory, ignore_errors=True)
        try:
            first = run_training(
                factory=_tiny_factory,
                protocol=_tiny_protocol(),
                seed=17,
                target_timesteps=96,
                output_dir=directory,
                fixture_profile=True,
            )
            self.assertEqual(first["effective_interactions"], 96)
            for name in ("model.zip", "vecnormalize.pkl", "training_summary.json"):
                (directory / name).unlink()
            second = run_training(
                factory=_tiny_factory,
                protocol=_tiny_protocol(),
                seed=17,
                target_timesteps=192,
                output_dir=directory,
                fixture_profile=True,
            )
            self.assertEqual(second["resumed_from_interactions"], 96)
            self.assertEqual(second["effective_interactions"], 192)
            self.assertTrue((directory / "latest" / "state.json").exists())
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_final_boundary_checkpoint_recovers_without_more_learning(self) -> None:
        directory = ROOT / ".test-final-boundary-recovery"
        shutil.rmtree(directory, ignore_errors=True)
        try:
            run_training(
                factory=_tiny_factory,
                protocol=_tiny_protocol(),
                seed=19,
                target_timesteps=96,
                output_dir=directory,
                fixture_profile=True,
            )
            for name in ("model.zip", "vecnormalize.pkl", "training_summary.json"):
                (directory / name).unlink()
            recovered = run_training(
                factory=_tiny_factory,
                protocol=_tiny_protocol(),
                seed=19,
                target_timesteps=96,
                output_dir=directory,
                fixture_profile=True,
            )
            self.assertEqual(recovered["resumed_from_interactions"], 96)
            self.assertEqual(recovered["effective_interactions"], 96)
            self.assertTrue((directory / "model.zip").exists())
            self.assertTrue((directory / "training_summary.json").exists())
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_exact_wilcoxon_known_small_sample(self) -> None:
        result = exact_two_sided_wilcoxon([1.0, 2.0, 3.0])
        self.assertEqual(result["positive_rank_sum"], 6.0)
        self.assertEqual(result["negative_rank_sum"], 0.0)
        self.assertEqual(result["p_value"], 0.25)

    def test_rank_biserial_positive_means_lower_policy_impact(self) -> None:
        result = paired_seed_summary([-4.0, -3.0, 1.0, 2.0])
        self.assertAlmostEqual(result["rank_biserial_correlation"]["value"], 0.4)
        self.assertEqual(result["seed_win_count"], 2)
        self.assertIn("positive favors policy", result["rank_biserial_correlation"]["definition"])

    def test_paired_bootstrap_is_deterministic(self) -> None:
        first = paired_bootstrap_intervals([-2.0, -1.0, 1.0, 3.0])
        second = paired_bootstrap_intervals([-2.0, -1.0, 1.0, 3.0])
        self.assertEqual(first, second)
        self.assertEqual(first["draws"], 10_000)

    def test_aggregator_requires_all_seed_summaries(self) -> None:
        directory = ROOT / ".test-missing-seed-aggregation"
        shutil.rmtree(directory, ignore_errors=True)
        try:
            directory.mkdir()
            with self.assertRaisesRegex(FileNotFoundError, "all 10 seeds"):
                aggregate(directory)
        finally:
            shutil.rmtree(directory, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
