"""Smoke coverage for the residual-safe off-policy v5 environment."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from env.data_loader import load_scenario
from env.residual_safe_offpolicy_env import ResidualSafeOffPolicyEnv
from env.reward import RewardConfig
from env.safety_layer import SafetyConfig


def make_env() -> ResidualSafeOffPolicyEnv:
    scenario = ROOT / "env" / "scenarios" / "us_model_v2_2025.yaml"
    sites, power_model, batch = load_scenario(
        scenario,
        batch_enabled=True,
        dynamic_arrivals=True,
        seed=42,
    )
    env = ResidualSafeOffPolicyEnv(
        sites=sites,
        power_model=power_model,
        batch_enabled=True,
        flexibility_factor=float(batch["flexibility_factor"]),
        deadline_penalty_weight=float(batch["deadline_penalty_weight"]),
        urgency_horizon_steps=int(batch["urgency_horizon_steps"]),
        peak_penalty_weight=0.015,
        enforce_batch_completion=True,
        reward_config=RewardConfig(
            service_backlog_weight=700.0,
            batch_completion_weight=1000.0,
            reward_scale=2.5e-4,
        ),
        observe_episode_progress=True,
        deadline_bucket_edges=(1, 3, 6, 12, 24),
        safety_config=SafetyConfig(
            service_envelope_total=2.25,
            batch_arrival_envelope_total=1.0,
            future_fleet_capacity_total=4.0,
        ),
    )
    env.reset(seed=42)
    return env


def test_observation_contains_residual_features() -> None:
    env = make_env()
    obs, _ = env.reset(seed=42)
    expected = env.observation_space.shape[0]
    assert obs.shape == (expected,)
    assert np.isfinite(obs).all()
    context = env.residual_decoder_context()
    assert context.fleet_service_slack >= 0.0
    env.close()


def test_status_quo_action_stays_native_safe() -> None:
    env = make_env()
    env.reset(seed=42)
    action = env.status_quo_action()
    _, _, _, _, info = env.step(action)
    assert info["native_decoder_used"] is True
    assert info["safety_intervened"] is False
    assert info["total_batch_expired"] == 0.0
    env.close()


def test_random_action_preserves_hard_safety() -> None:
    env = make_env()
    obs, _ = env.reset(seed=7)
    rng = np.random.default_rng(7)
    for _ in range(24):
        action = rng.uniform(
            low=-env.decoder_logit_bound,
            high=env.decoder_logit_bound,
            size=env.action_space.shape,
        ).astype(np.float32)
        obs, _, terminated, truncated, info = env.step(action)
        assert np.isfinite(obs).all()
        assert info["total_batch_expired"] == 0.0
        assert info["terminal_batch_pool"] if "terminal_batch_pool" in info else True
        if terminated or truncated:
            break
    env.close()


def main() -> None:
    test_observation_contains_residual_features()
    test_status_quo_action_stays_native_safe()
    test_random_action_preserves_hard_safety()
    print("smoke_test_offpolicy_v5: PASS")


if __name__ == "__main__":
    main()
