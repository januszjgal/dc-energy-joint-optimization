"""Smoke coverage for the residual-safe off-policy v5 environment."""

from __future__ import annotations

import sys
import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import stable_baselines3 as sb3
from stable_baselines3.common.monitor import Monitor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from env.data_loader import load_scenario
from env.residual_safe_offpolicy_env import ResidualSafeOffPolicyEnv
from env.reward import RewardConfig
from env.safety_layer import SafetyConfig
from scripts.run_offpolicy_campaign_v5 import (
    EXPECTED_SB3_VERSION,
    PROTOCOL_ID,
    PROTOCOL_SHA256,
    TD3BehaviorCloning,
    actor_target_sync_summary,
    behavior_clone_actor,
    campaign_job,
    evaluate_model,
    make_model,
    module_sha256,
    prefill_replay_buffer,
    resolve_stage_config,
    teacher_action,
)
from scripts.build_offpolicy_evidence_v5 import (
    classify_workspace_state,
    post_rl_seed_provenance_pass,
    saved_model_hashes,
    source_commit_contract_pass,
    validate_campaign_roles,
)


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


def test_teacher_action_stays_native_safe() -> None:
    env = make_env()
    env.reset(seed=11)
    action = env.marginal_cost_teacher_action()
    _, _, _, _, info = env.step(action)
    assert info["native_decoder_used"] is True
    assert info["safety_intervened"] is False
    assert info["total_batch_expired"] == 0.0
    env.close()


def test_exact_teacher_action_stays_native_safe() -> None:
    env = make_env()
    env.reset(seed=13)
    action = teacher_action(env, "exact_native")
    assert env.action_space.contains(action)
    _, _, _, _, info = env.step(action)
    assert info["native_decoder_used"] is True
    assert info["safety_intervened"] is False
    assert info["total_batch_expired"] == 0.0
    assert np.allclose(np.asarray(info["executed_action"], dtype=np.float32), action)
    env.close()


def test_prefill_normalizes_td3_actions() -> None:
    env = Monitor(make_env())
    obs, _ = env.reset(seed=17)
    action = env.unwrapped.status_quo_action()
    next_obs, reward, terminated, truncated, info = env.step(action)
    model = make_model(
        "td3_bc",
        env,
        17,
        {
            "learning_rate": 1e-5,
            "buffer_size": 32,
            "learning_starts": 0,
            "batch_size": 8,
            "tau": 0.01,
            "train_freq": 1,
            "gradient_steps": 1,
            "action_noise_sigma": 0.01,
            "net_arch": (32, 32),
            "bc_alpha": 1.0,
            "td3bc_lambda_alpha": 2.5,
        },
    )
    dataset = {
        "observations": np.asarray([obs], dtype=np.float32),
        "next_observations": np.asarray([next_obs], dtype=np.float32),
        "actions": np.asarray([action], dtype=np.float32),
        "rewards": np.asarray([reward], dtype=np.float32),
        "dones": np.asarray([terminated or truncated], dtype=np.float32),
        "infos": [info],
    }
    prefill_replay_buffer(model, dataset, env)
    stored = np.asarray(model.replay_buffer.actions[0, 0], dtype=np.float32)
    assert np.max(np.abs(stored)) <= 1.0 + 1e-6
    env.close()


def test_teacher_step_store_pairing_uses_executed_action() -> None:
    env = make_env()
    obs, _ = env.reset(seed=19)
    for _ in range(3):
        action = teacher_action(env, "exact_native")
        assert env.action_space.contains(action)
        next_obs, reward, terminated, truncated, info = env.step(action)
        executed = np.asarray(info["executed_action"], dtype=np.float32)
        assert np.allclose(executed, action, atol=1e-8)
        obs = next_obs
        if terminated or truncated:
            break
    assert np.isfinite(obs).all()
    assert np.isfinite(reward)
    env.close()


def test_campaign_cli_uses_td3_behavior_cloning() -> None:
    env = Monitor(make_env())
    job = campaign_job("td3bc_bconly_frozen_v3", region="us", seed=23)
    config = resolve_stage_config(job)
    model = make_model(job.algorithm, env, job.seed, config)
    assert isinstance(model, TD3BehaviorCloning)
    env.close()


def test_teacher_guard_blocks_inference_teacher_calls() -> None:
    env = make_env()
    env.set_teacher_policy_calls_allowed(False)
    try:
        env.marginal_cost_teacher_action()
    except RuntimeError as exc:
        assert "disabled" in str(exc)
    else:
        raise AssertionError("teacher action should be blocked during inference")
    env.close()

    eval_env = Monitor(make_env())
    job = campaign_job("td3bc_bconly_frozen_v3", region="us", seed=29)
    config = resolve_stage_config(job)
    model = make_model(job.algorithm, eval_env, job.seed, config)
    baseline = {
        "total_cost": 1.0,
    }
    summary = evaluate_model(
        model,
        "us",
        float(config["reward_scale_by_region"]["us"]),
        baseline=baseline,
        scenario_kind="development",
        forbid_teacher_policy_calls=True,
    )
    assert summary["teacher_call_guard_enabled"] is True
    eval_env.close()


def test_decoder_and_emergency_telemetry_are_separate() -> None:
    env = make_env()
    env.reset(seed=31)
    action = teacher_action(env, "exact_native")
    _, _, _, _, info = env.step(action)
    assert "safety_decoder_adjustment_l2" in info
    assert "safety_emergency_adjustment_l2" in info
    assert "safety_decoder_adjusted" in info
    assert "safety_emergency_intervened" in info
    assert info["safety_emergency_intervened"] is False
    summary = evaluate_model(
        make_model(
            "td3_bc",
            Monitor(make_env()),
            31,
            {
                **resolve_stage_config(
                campaign_job("td3bc_bconly_frozen_v3", region="us", seed=31)
                ),
            },
        ),
        "us",
        float(
            resolve_stage_config(
                campaign_job("td3bc_bconly_frozen_v3", region="us", seed=31)
            )["reward_scale_by_region"]["us"]
        ),
        baseline={"total_cost": 1.0},
        scenario_kind="development",
        forbid_teacher_policy_calls=True,
    )
    safety = summary["safety"]
    assert "decoder_adjustment_rate" in safety
    assert "emergency_intervention_rate" in safety
    env.close()


def test_behavior_clone_synchronizes_actor_target() -> None:
    env = Monitor(make_env())
    obs, _ = env.reset(seed=37)
    action = env.unwrapped.status_quo_action()
    next_obs, reward, terminated, truncated, info = env.step(action)
    job = campaign_job("td3bc_bconly_frozen_v3", region="us", seed=37)
    config = {
        **resolve_stage_config(job),
        "buffer_size": 32,
        "batch_size": 8,
        "net_arch": (32, 32),
    }
    model = make_model(job.algorithm, env, job.seed, config)
    dataset = {
        "observations": np.repeat(np.asarray([obs], dtype=np.float32), 8, axis=0),
        "next_observations": np.repeat(
            np.asarray([next_obs], dtype=np.float32),
            8,
            axis=0,
        ),
        "actions": np.repeat(np.asarray([action], dtype=np.float32), 8, axis=0),
        "rewards": np.repeat(np.asarray([reward], dtype=np.float32), 8, axis=0),
        "dones": np.repeat(
            np.asarray([terminated or truncated], dtype=np.float32),
            8,
            axis=0,
        ),
        "infos": [info] * 8,
    }
    summary = behavior_clone_actor(
        model,
        env,
        dataset,
        algorithm=job.algorithm,
        steps=2,
        batch_size=8,
        seed=37,
    )
    assert summary["distance_l2_before"] > 0.0
    assert summary["distance_l2_after"] <= 1e-12
    assert summary["synchronized"] is True
    assert summary["actor_hash"] == summary["actor_target_hash"]
    assert actor_target_sync_summary(model, synchronize=False)["synchronized"] is True
    env.close()


def test_origin_decoder_adjustment_is_measured() -> None:
    env = make_env()
    env.reset(seed=41)
    context = env.residual_decoder_context()
    action = env.status_quo_action()
    action[env.n_dc] = env.decoder_logit_bound
    origin_start = env.n_dc + 1
    origin_logits = np.full(env.n_dc, -env.decoder_logit_bound, dtype=np.float32)
    origin_logits[int(np.argmin(context.pool_totals))] = env.decoder_logit_bound
    action[origin_start : origin_start + env.n_dc] = origin_logits
    projection = env._decode_residual_action(
        action,
        context,
        current_step=env.step_index,
    )
    origin_delta = float(
        np.linalg.norm(
            projection.desired_origin_batch - projection.origin_batch,
        )
    )
    assert origin_delta > 1e-6
    assert projection.decoder_adjustment_l2 >= origin_delta - 1e-10
    assert projection.decoder_adjusted is True
    env.close()


def test_evidence_rejects_wrong_roles_dirty_source_and_tampered_hashes() -> None:
    validate_campaign_roles(
        "td3bc_bconly_frozen_v3",
        "td3bc_postrl_frozen_v3",
    )
    try:
        validate_campaign_roles(
            "td3bc_bconly_frozen_v3",
            "td3bc_bconly_frozen_v3",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("evidence accepted the same BC and post-RL campaign")

    dirty = classify_workspace_state(
        [" M scripts/run_offpolicy_campaign_v5.py"],
        unstaged_tracked_clean=False,
        staged_tracked_clean=True,
    )
    assert dirty["tracked_source_clean"] is False
    nongenerated = classify_workspace_state(
        ["?? unexpected.txt"],
        unstaged_tracked_clean=True,
        staged_tracked_clean=True,
    )
    assert nongenerated["untracked_generated_only"] is False
    generated = classify_workspace_state(
        ["?? models/offpolicy_v5_continuous/example/model.zip"],
        unstaged_tracked_clean=True,
        staged_tracked_clean=True,
    )
    assert generated["tracked_source_clean"] is True
    assert generated["untracked_generated_only"] is True

    valid = {
        "campaign_role": "post_rl",
        "training_mode": "td3_bc_postrl",
        "n_updates": 1,
        "actor_hash_before": "actor-before",
        "actor_hash_after": "actor-after",
        "critic_hash_before": "critic-before",
        "critic_hash_after": "critic-after",
        "saved_actor_hash": "actor-after",
        "saved_critic_hash": "critic-after",
        "source_bc_model_sha256": "bc-model-sha",
        "verified_source_bc_model_sha256": "bc-model-sha",
        "source_bc_actor_hash": "actor-before",
        "source_bc_critic_hash": "critic-before",
        "weight_update_evidence": {
            "actor_hash_changed": True,
            "critic_hash_changed": True,
            "actor_parameter_delta_l2": 1.0,
            "critic_parameter_delta_l2": 1.0,
        },
        "warm_start_actor_target_sync": {
            "synchronized": True,
            "distance_l2_after": 0.0,
        },
        "teacher_present_during_rl": False,
        "teacher_present_at_inference": False,
        "teacher_action_audit": {"stored_action_matches_executed": True},
        "stable_baselines3_version": EXPECTED_SB3_VERSION,
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": PROTOCOL_SHA256,
    }
    assert post_rl_seed_provenance_pass(valid) is True
    tampered = deepcopy(valid)
    tampered["actor_hash_after"] = "arbitrary-unequal-tampered-hash"
    assert post_rl_seed_provenance_pass(tampered) is False
    replaced_bc = deepcopy(valid)
    replaced_bc["verified_source_bc_model_sha256"] = "replacement-sha"
    assert post_rl_seed_provenance_pass(replaced_bc) is False
    bc_only = deepcopy(valid)
    bc_only["campaign_role"] = "bc_only"
    bc_only["training_mode"] = "teacher_bc_only"
    bc_only["n_updates"] = 0
    assert post_rl_seed_provenance_pass(bc_only) is False
    assert source_commit_contract_pass(
        {"source-head"},
        current_head="source-head",
        committed_protocol_sha256=PROTOCOL_SHA256,
    )
    assert not source_commit_contract_pass(
        {"41aff32-old-source"},
        current_head="source-head",
        committed_protocol_sha256=PROTOCOL_SHA256,
    )
    assert not source_commit_contract_pass(
        {"source-head"},
        current_head="source-head",
        committed_protocol_sha256="tampered-protocol",
    )


def test_saved_model_hashes_are_recomputed() -> None:
    env = Monitor(make_env())
    job = campaign_job("td3bc_bconly_frozen_v3", region="us", seed=43)
    config = {
        **resolve_stage_config(job),
        "buffer_size": 32,
        "batch_size": 8,
        "net_arch": (32, 32),
    }
    model = make_model(job.algorithm, env, job.seed, config)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model"
        model.save(path)
        hashes = saved_model_hashes(path.with_suffix(".zip"))
    assert hashes["actor"] == module_sha256(model.actor)
    assert hashes["actor_target"] == module_sha256(model.actor_target)
    assert hashes["critic"] == module_sha256(model.critic)
    env.close()


def test_sb3_runtime_is_pinned() -> None:
    assert sb3.__version__ == EXPECTED_SB3_VERSION


def main() -> None:
    test_observation_contains_residual_features()
    test_status_quo_action_stays_native_safe()
    test_random_action_preserves_hard_safety()
    test_teacher_action_stays_native_safe()
    test_exact_teacher_action_stays_native_safe()
    test_prefill_normalizes_td3_actions()
    test_teacher_step_store_pairing_uses_executed_action()
    test_campaign_cli_uses_td3_behavior_cloning()
    test_teacher_guard_blocks_inference_teacher_calls()
    test_decoder_and_emergency_telemetry_are_separate()
    test_behavior_clone_synchronizes_actor_target()
    test_origin_decoder_adjustment_is_measured()
    test_evidence_rejects_wrong_roles_dirty_source_and_tampered_hashes()
    test_saved_model_hashes_are_recomputed()
    test_sb3_runtime_is_pinned()
    print("smoke_test_offpolicy_v5: PASS")


if __name__ == "__main__":
    main()
