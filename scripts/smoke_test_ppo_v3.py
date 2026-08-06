"""Targeted invariants for the exploratory PPO v3 recovery environment."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from env.data_loader import load_scenario
from env.multi_dc_env import MultiDCEnv
from env.reward import RewardConfig
from env.workload_generator import BatchPool


SCENARIO = Path("env/scenarios/us_model_v2_2025.yaml")
EDGES = (1, 3, 6, 12, 24)


def make_env(
    reward: RewardConfig | None,
    *,
    max_steps: int | None = None,
    observe_v3: bool = True,
) -> MultiDCEnv:
    sites, power_model, batch = load_scenario(
        SCENARIO,
        batch_enabled=True,
        seed=42,
    )
    return MultiDCEnv(
        sites,
        power_model,
        max_steps=max_steps,
        batch_enabled=True,
        flexibility_factor=batch["flexibility_factor"],
        deadline_penalty_weight=batch["deadline_penalty_weight"],
        urgency_horizon_steps=batch["urgency_horizon_steps"],
        peak_penalty_weight=0.015,
        enforce_batch_completion=True,
        completion_penalty_weight=(
            1000.0 if reward is None else None
        ),
        reward_config=reward,
        observe_episode_progress=observe_v3,
        deadline_bucket_edges=EDGES if observe_v3 else None,
    )


def test_deadline_queries() -> None:
    pool = BatchPool()
    pool.add(1.0, 10)
    pool.add(2.0, 13)
    pool.add(3.0, 40)
    histogram = pool.deadline_histogram(9, EDGES)
    assert np.allclose(histogram, [1.0, 0.0, 2.0, 0.0, 0.0, 3.0])
    assert np.isclose(sum(histogram), pool.total_demand)
    assert np.isclose(pool.demand_due_by(9, 3), 1.0)


def test_observation_compatibility() -> None:
    legacy = make_env(None, observe_v3=False)
    legacy_obs, _ = legacy.reset(seed=201)
    assert legacy.observation_space.shape == (55,)
    assert legacy_obs.shape == (55,)

    recovery = make_env(RewardConfig(700.0, 1000.0))
    obs, _ = recovery.reset(seed=201)
    assert recovery.observation_space.shape == (81,)
    assert obs.shape == (81,)
    assert np.isfinite(obs).all()
    assert np.allclose(obs[-2:], [0.0, 1.0])


def test_episode_progress() -> None:
    env = make_env(RewardConfig(700.0, 1000.0), max_steps=3)
    obs, _ = env.reset(seed=201)
    assert np.allclose(obs[-2:], [0.0, 1.0])
    obs, _, terminated, _, _ = env.step(np.zeros(12))
    assert not terminated
    assert np.allclose(obs[-2:], [0.5, 0.5])
    obs, _, terminated, _, _ = env.step(np.zeros(12))
    assert not terminated
    assert np.allclose(obs[-2:], [1.0, 0.0])


def test_training_weights_preserve_full_objective() -> None:
    control = make_env(
        RewardConfig(
            1000.0,
            1000.0,
            subtract_idle_cost=True,
        ),
        max_steps=16,
    )
    variant = make_env(
        RewardConfig(
            700.0,
            2000.0,
            subtract_idle_cost=True,
        ),
        max_steps=16,
    )
    control.reset(seed=201)
    variant.reset(seed=201)
    action = np.zeros(12)
    _, control_reward, _, _, control_info = control.step(action)
    _, variant_reward, _, _, variant_info = variant.step(action)
    assert np.isclose(
        control_info["total_cost"],
        variant_info["total_cost"],
    )
    assert control.backlog_weight == variant.backlog_weight == 1000.0
    assert (
        control.batch_completion_weight
        == variant.batch_completion_weight
        == 1000.0
    )
    assert not np.isclose(control_reward, variant_reward)
    assert np.isclose(
        variant_info["reward_training_cost"],
        variant_info["total_cost"]
        + variant_info["reward_penalty_adjustment"]
        - variant_info["reward_idle_cost"]
        - variant_info["reward_potential_delta"],
    )


def test_potential_telescopes() -> None:
    env = make_env(
        RewardConfig(
            700.0,
            1500.0,
            subtract_idle_cost=True,
            urgency_potential_weight=250.0,
        ),
        max_steps=32,
    )
    env.reset(seed=201)
    initial = env._batch_urgency_potential(0)
    deltas = []
    while True:
        _, _, terminated, _, info = env.step(np.zeros(12))
        deltas.append(info["reward_potential_delta"])
        if terminated:
            break
    assert np.isclose(sum(deltas), -initial, atol=1e-8)


def test_penalty_floor_rejects_unsafe_weights() -> None:
    try:
        make_env(RewardConfig(100.0, 1000.0))
    except ValueError as exc:
        assert "economic floor" in str(exc)
    else:
        raise AssertionError("unsafe service weight was accepted")


def test_action_shape_fails_closed() -> None:
    env = make_env(RewardConfig(700.0, 1000.0), max_steps=2)
    env.reset(seed=201)
    try:
        env.step(np.zeros(8))
    except ValueError as exc:
        assert "action shape" in str(exc)
    else:
        raise AssertionError("joint environment accepted a truncated action")


def main() -> None:
    tests = [
        test_deadline_queries,
        test_observation_compatibility,
        test_episode_progress,
        test_training_weights_preserve_full_objective,
        test_potential_telescopes,
        test_penalty_floor_rejects_unsafe_weights,
        test_action_shape_fails_closed,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print("PPO v3 smoke tests passed.")


if __name__ == "__main__":
    main()
