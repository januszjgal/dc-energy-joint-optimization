"""Smoke tests for the isolated teacher-free native BC/PPO pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from env.reward import RewardConfig  # noqa: E402
from env.safety_layer import SafetyConfig  # noqa: E402
from teacher_free.native_experiment import (  # noqa: E402
    collect_teacher_dataset,
    exact_teacher_action,
    make_native_env,
    train_behavior_cloner,
)


def main() -> None:
    reward = RewardConfig(
        service_backlog_weight=700.0,
        batch_completion_weight=1000.0,
        evaluation_service_backlog_weight=1000.0,
        evaluation_batch_completion_weight=1000.0,
        reward_scale=1e-4,
        subtract_idle_cost=False,
        urgency_potential_weight=0.0,
    )
    safe = SafetyConfig(
        service_envelope_total=2.25,
        batch_arrival_envelope_total=1.0,
        future_fleet_capacity_total=4.0,
        envelope_id="ad-rounded-envelope-v1",
        envelope_scope="a-d-development-only",
    )
    scenario = ROOT / "env" / "scenarios" / "us_model_v2_2025.yaml"
    env = make_native_env(
        scenario,
        seed=7,
        peak_penalty_weight=0.015,
        reward_config=reward,
        safety_config=safe,
        domain_randomization=False,
    )
    obs, _ = env.reset(seed=7)
    del obs
    action, diagnostics = exact_teacher_action(env.unwrapped)
    assert action.shape == (12,)
    assert float(diagnostics.native_roundtrip_l2) < 1e-6
    env.close()

    dataset = collect_teacher_dataset(
        scenario,
        seeds=[7],
        peak_penalty_weight=0.015,
        reward_config=reward,
        safety_config=safe,
        max_steps=64,
    )
    assert dataset.observations.shape[0] == 64
    assert dataset.actions.shape == (64, 12)
    result = train_behavior_cloner(
        dataset,
        seed=123,
        hidden_sizes=(64, 64),
        learning_rate=1e-3,
        epochs=8,
        batch_size=32,
    )
    assert np.isfinite(result.final_loss)


if __name__ == "__main__":
    main()
