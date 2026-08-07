"""Adversarial and full-episode checks for the v4 hard safety layer."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from env.data_loader import load_scenario
from env.reward import RewardConfig
from env.safe_multi_dc_env import SafeMultiDCEnv
from env.safety_layer import (
    SafetyConfig,
    SafetyInfeasibleError,
    exact_transport,
    project_capped_simplex,
    project_joint_action,
)
from env.workload_generator import BatchPool
from evaluate import compute_summary, run_episode


EDGES = (1, 3, 6, 12, 24)


def make_pool(entries: list[tuple[float, int]]) -> BatchPool:
    pool = BatchPool()
    for amount, deadline in entries:
        pool.add(amount, deadline)
    return pool


def make_env(
    scenario: str = "env/scenarios/us_model_v2_2025.yaml",
    *,
    flush: bool = False,
    max_steps: int | None = None,
) -> SafeMultiDCEnv:
    sites, power_model, batch = load_scenario(
        Path(scenario),
        batch_enabled=True,
        seed=42,
    )
    return SafeMultiDCEnv(
        sites,
        power_model,
        max_steps=max_steps,
        batch_enabled=True,
        peak_penalty_weight=0.015,
        flexibility_factor=batch["flexibility_factor"],
        deadline_penalty_weight=batch["deadline_penalty_weight"],
        urgency_horizon_steps=batch["urgency_horizon_steps"],
        enforce_batch_completion=True,
        reward_config=RewardConfig(700.0, 1000.0),
        observe_episode_progress=True,
        deadline_bucket_edges=EDGES,
        safety_config=SafetyConfig(
            negative_demand_flush=flush,
        ),
    )


def test_projection_primitives() -> None:
    projected = project_capped_simplex(
        np.array([3.0, 1.0, 0.0]),
        2.0,
        np.array([0.5, 1.0, 2.0]),
    )
    assert np.isclose(projected.sum(), 2.0)
    assert np.all(projected >= 0.0)
    assert np.all(projected <= np.array([0.5, 1.0, 2.0]))
    flow = exact_transport(
        np.array([0.3, 0.7]),
        np.array([0.4, 0.6]),
    )
    assert np.allclose(flow.sum(axis=1), [0.3, 0.7])
    assert np.allclose(flow.sum(axis=0), [0.4, 0.6])


def test_deadline_semantics_and_actionable_state() -> None:
    pool = make_pool([(1.0, 4), (2.0, 5)])
    assert np.isclose(pool.actionable_total_demand(4), 2.0)
    histogram = pool.actionable_deadline_histogram(4, EDGES)
    assert np.isclose(sum(histogram), 2.0)
    assert np.isclose(pool.expire(4), 1.0)
    assert np.isclose(pool.total_demand, 2.0)


def test_wrong_origin_override_and_exact_endpoints() -> None:
    pools = [
        make_pool([(0.5, 1)]),
        make_pool([(0.5, 10)]),
    ]
    result = project_joint_action(
        np.array([0.0, 0.0, -3.0, 3.0, 0.0, 0.0]),
        pools,
        current_step=0,
        max_steps=20,
        total_service=0.0,
        current_batch_arrival=0.0,
        effective_capacity=np.array([0.25, 0.25]),
        net_demand=np.ones(2),
        price=np.zeros(2),
        config=SafetyConfig(
            service_envelope_total=0.0,
            batch_arrival_envelope_total=0.0,
            future_fleet_capacity_total=0.5,
        ),
    )
    assert np.allclose(result.origin_batch, [0.5, 0.0])
    assert result.drain_rates[0] == 1.0

    full = project_joint_action(
        np.array([0.0, -3.0, 0.0]),
        [make_pool([(1.0, 10)])],
        current_step=0,
        max_steps=1,
        total_service=0.0,
        current_batch_arrival=0.0,
        effective_capacity=np.array([1.0]),
        net_demand=np.array([1.0]),
        price=np.array([0.0]),
        config=SafetyConfig(
            service_envelope_total=0.0,
            batch_arrival_envelope_total=0.0,
            future_fleet_capacity_total=1.0,
        ),
    )
    assert full.drain_rates[0] == 1.0

    empty = project_joint_action(
        np.array([0.0, 3.0, 0.0]),
        [make_pool([])],
        current_step=0,
        max_steps=20,
        total_service=0.0,
        current_batch_arrival=0.0,
        effective_capacity=np.array([1.0]),
        net_demand=np.array([1.0]),
        price=np.array([0.0]),
        config=SafetyConfig(
            service_envelope_total=0.0,
            batch_arrival_envelope_total=0.0,
            future_fleet_capacity_total=1.0,
        ),
    )
    assert empty.drain_rates[0] == 0.0


def test_fail_closed_certificates() -> None:
    try:
        project_joint_action(
            np.zeros(3),
            [make_pool([])],
            current_step=0,
            max_steps=2,
            total_service=1.1,
            current_batch_arrival=0.0,
            effective_capacity=np.array([1.0]),
            net_demand=np.array([0.0]),
            price=np.array([0.0]),
            config=SafetyConfig(
                service_envelope_total=1.1,
                batch_arrival_envelope_total=0.0,
                future_fleet_capacity_total=1.1,
            ),
        )
    except SafetyInfeasibleError as exc:
        assert exc.certificate["reason"] == "service_capacity_deficit"
    else:
        raise AssertionError("service deficit did not fail closed")

    try:
        project_joint_action(
            np.zeros(3),
            [make_pool([(1.0, 0)])],
            current_step=0,
            max_steps=2,
            total_service=0.0,
            current_batch_arrival=0.0,
            effective_capacity=np.array([1.0]),
            net_demand=np.array([0.0]),
            price=np.array([0.0]),
            config=SafetyConfig(
                service_envelope_total=0.0,
                batch_arrival_envelope_total=0.0,
                future_fleet_capacity_total=1.0,
            ),
        )
    except SafetyInfeasibleError as exc:
        assert exc.certificate["reason"] == "pre_action_deadline_miss"
    else:
        raise AssertionError("already-lost work did not fail closed")


def test_preexisting_backlog_fails_closed() -> None:
    env = make_env(max_steps=2)
    env.reset(seed=301)
    env.sites[0].backlog = 0.1
    try:
        env.step(np.zeros(12))
    except SafetyInfeasibleError as exc:
        assert (
            exc.certificate["reason"]
            == "preexisting_local_service_backlog"
        )
    else:
        raise AssertionError("pre-existing local backlog was hidden")


def test_no_future_trace_leakage() -> None:
    common = dict(
        raw_action=np.zeros(3),
        pools=[make_pool([(0.4, 3)])],
        current_step=0,
        max_steps=10,
        total_service=0.2,
        current_batch_arrival=0.0,
        effective_capacity=np.array([1.0]),
        net_demand=np.array([-0.1]),
        price=np.array([0.01]),
        config=SafetyConfig(
            service_envelope_total=0.5,
            batch_arrival_envelope_total=0.0,
            future_fleet_capacity_total=1.0,
        ),
    )
    first = project_joint_action(**common)
    # No future trace is an input. Reconstructing an identical current state
    # must therefore give an identical projection.
    common["pools"] = [make_pool([(0.4, 3)])]
    second = project_joint_action(**common)
    assert np.array_equal(first.service, second.service)
    assert np.array_equal(first.origin_batch, second.origin_batch)


def test_negative_flush_is_optional() -> None:
    base = dict(
        raw_action=np.array([0.0, -3.0, 0.0]),
        pools=[make_pool([(1.0, 10)])],
        current_step=0,
        max_steps=20,
        total_service=0.0,
        current_batch_arrival=0.0,
        effective_capacity=np.array([1.0]),
        net_demand=np.array([-0.2]),
        price=np.array([-0.01]),
    )
    off = project_joint_action(
        **base,
        config=SafetyConfig(
            service_envelope_total=0.0,
            batch_arrival_envelope_total=0.0,
            future_fleet_capacity_total=1.0,
            negative_demand_flush=False,
        ),
    )
    base["pools"] = [make_pool([(1.0, 10)])]
    on = project_joint_action(
        **base,
        config=SafetyConfig(
            service_envelope_total=0.0,
            batch_arrival_envelope_total=0.0,
            future_fleet_capacity_total=1.0,
            negative_demand_flush=True,
        ),
    )
    assert off.origin_batch[0] < 0.1
    assert on.origin_batch[0] == 1.0


def test_random_projection_invariants() -> None:
    rng = np.random.default_rng(4)
    config = SafetyConfig(
        service_envelope_total=2.25,
        batch_arrival_envelope_total=1.0,
        future_fleet_capacity_total=4.0,
    )
    returned = 0
    for _ in range(250):
        pools = []
        for _origin in range(4):
            entries = [
                (
                    float(rng.uniform(0.0, 0.12)),
                    int(rng.integers(1, 28)),
                )
                for _ in range(int(rng.integers(0, 5)))
            ]
            pools.append(make_pool(entries))
        service = float(rng.uniform(0.0, 2.25))
        try:
            result = project_joint_action(
                rng.uniform(-3.0, 3.0, size=12),
                pools,
                current_step=0,
                max_steps=30,
                total_service=service,
                current_batch_arrival=0.0,
                effective_capacity=np.ones(4),
                net_demand=rng.uniform(-1.0, 1.0, size=4),
                price=rng.uniform(-0.02, 0.08, size=4),
                config=config,
            )
        except SafetyInfeasibleError:
            continue
        returned += 1
        assert np.all(result.service >= -1e-10)
        assert np.all(result.service <= 1.0 + 1e-10)
        assert np.isclose(result.service.sum(), service)
        assert np.all(
            result.destination_batch
            <= result.residual_capacity + 1e-10
        )
        assert np.all(
            result.origin_batch
            <= np.array([p.total_demand for p in pools]) + 1e-10
        )
        assert np.allclose(
            result.transport.sum(axis=1),
            result.origin_batch,
        )
        assert np.allclose(
            result.transport.sum(axis=0),
            result.destination_batch,
        )
        assert result.minimum_deadline_slack >= -1e-8
    assert returned >= 200


def test_optional_grid_and_ramp_caps() -> None:
    env = make_env(max_steps=2)
    grid_caps = []
    ramp_caps = []
    for site in env.sites:
        power_model = site.power_model or env.power_model
        grid_caps.append(
            (
                power_model.idle_power
                + 0.8 * power_model.slope
            )
            * site.rated_power_mw
        )
        ramp_caps.append(
            0.8 * power_model.slope * site.rated_power_mw
        )
    env.safety_config = SafetyConfig(
        service_envelope_total=2.25,
        batch_arrival_envelope_total=0.9,
        future_fleet_capacity_total=3.2,
        max_grid_mw=tuple(grid_caps),
        max_upward_ramp_mw=tuple(ramp_caps),
    )
    env.safety_config.validate(env.n_dc)
    env.reset(seed=301)
    _, _, _, _, info = env.step(np.zeros(12))
    for index, dc in enumerate(info["per_dc"]):
        assert dc["grid_mw"] <= grid_caps[index] + 1e-8

    env = make_env(max_steps=2)
    idle_caps = tuple(
        (site.power_model or env.power_model).idle_power
        * site.rated_power_mw
        - 0.1
        for site in env.sites
    )
    try:
        env.safety_config = SafetyConfig(
            future_fleet_capacity_total=0.0,
            service_envelope_total=0.0,
            batch_arrival_envelope_total=0.0,
            max_grid_mw=idle_caps,
        )
        env._static_future_capacity_bounds()
    except ValueError as exc:
        assert "below unavoidable idle draw" in str(exc)
    else:
        raise AssertionError("grid cap below idle draw was accepted")


def test_full_episode_hold_policy() -> None:
    for scenario in (
        "env/scenarios/us_model_v2_2025.yaml",
        "env/scenarios/global_model_v2_2025.yaml",
    ):
        env = make_env(scenario)

        def hold(obs, current_env):
            return np.full(
                current_env.action_space.shape,
                -3.0,
                dtype=np.float32,
            )

        _, history = run_episode(env, hold, is_sb3=False)
        summary = compute_summary(history, batch_enabled=True)
        assert summary["terminal_backlog"] == 0.0
        assert summary["terminal_batch_pool"] < 1e-8
        assert summary["total_batch_expired"] == 0.0
        assert summary["batch_completion_fraction"] >= 1.0 - 1e-10
        assert max(
            row["safety_transport_conservation_error"]
            for row in history
        ) < 1e-8


def main() -> None:
    tests = [
        test_projection_primitives,
        test_deadline_semantics_and_actionable_state,
        test_wrong_origin_override_and_exact_endpoints,
        test_fail_closed_certificates,
        test_preexisting_backlog_fails_closed,
        test_no_future_trace_leakage,
        test_negative_flush_is_optional,
        test_random_projection_invariants,
        test_optional_grid_and_ramp_caps,
        test_full_episode_hold_policy,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print("PPO v4 safety tests passed.")


if __name__ == "__main__":
    main()
