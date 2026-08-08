"""Deterministic miniature PPO/SAC, attribution, resumption, and gate tests."""

from __future__ import annotations

import json
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ramp_rl.contract import EnvRequest, RampContractError, RampEnvAdapter  # noqa: E402
from ramp_rl.campaign import (  # noqa: E402
    assert_long_campaign_ready,
    extension_allowed,
    plan_stage,
    promotion_decision,
)
from ramp_rl.evaluation import evaluate_checkpoint  # noqa: E402
from ramp_rl.evidence import (  # noqa: E402
    FrozenLagrangian,
    record_pareto_sensitivities,
    select_on_validation,
    verify_pure_rl_manifest,
)
from ramp_rl.fixture_env import DeterministicRampFixtureEnv, make_fixture_env  # noqa: E402
from ramp_rl.runner import run_training  # noqa: E402
from ramp_rl.schema import load_protocol  # noqa: E402


def test_contract_and_evaluation_guard() -> None:
    train_request = EnvRequest(split="train", seed=1, training=True)
    env = RampEnvAdapter(make_fixture_env(train_request), train_request)
    observation, info = env.reset(seed=1)
    first_window = info["episode_context"]["window_id"]
    _, next_info = env.reset()
    assert next_info["episode_context"]["window_id"] != first_window
    assert observation.shape == env.observation_space.shape
    assert info["episode_context"]["future_realized_features_exposed"] is False
    try:
        env.evaluation_action("status_quo")
    except RampContractError:
        pass
    else:
        raise AssertionError("training adapter exposed an evaluation controller")
    env.close()

    request = EnvRequest(split="train", seed=2, training=True)
    raw = DeterministicRampFixtureEnv(request)
    original_contract = raw.ramp_rl_contract()
    raw.ramp_rl_contract = lambda: {**original_contract, "raw_redundant_projected_logits": True}  # type: ignore[method-assign]
    try:
        RampEnvAdapter(raw, request)
    except RampContractError:
        pass
    else:
        raise AssertionError("adapter accepted raw projected logits")


def test_lagrangian_and_selection_seal_test() -> None:
    dual = FrozenLagrangian(multiplier=0.0, learning_rate=0.1, maximum=10.0)
    assert dual.update(103.0, 100.0, 2.0, split="validation") > 0.0
    try:
        dual.update(103.0, 100.0, 2.0, split="test")
    except ValueError:
        pass
    else:
        raise AssertionError("sealed test updated the Lagrangian")
    candidates = [
        {"name": "a", "safety_pass": True, "energy_budget_pass": True, "mean_incremental_ramp_impact": -0.1},
        {"name": "b", "safety_pass": True, "energy_budget_pass": True, "mean_incremental_ramp_impact": -0.2},
    ]
    assert select_on_validation(candidates)["name"] == "b"
    pareto = record_pareto_sensitivities(
        [
            {
                "name": "b",
                "safety_pass": True,
                "energy_cost": 102.0,
                "status_quo_energy_cost": 100.0,
                "mean_incremental_ramp_impact": -0.2,
            }
        ],
        split="test",
    )
    assert [row["epsilon_pct"] for row in pareto] == [0.0, 2.0, 5.0]
    assert all(row["selected_on_test"] is False for row in pareto)
    try:
        select_on_validation(candidates, split="test")
    except ValueError:
        pass
    else:
        raise AssertionError("selection used sealed test")


def test_staged_campaign_protocol() -> None:
    protocol = load_protocol()
    screen = plan_stage(protocol, "screen")
    confirmation = plan_stage(protocol, "confirmation", algorithms=["sac"])
    assert len(screen) == 6
    assert all(job.timesteps == 100_000 for job in screen)
    assert len(confirmation) == 5
    rows = [
        {
            "split": "validation",
            "algorithm": algorithm,
            "seed": seed,
            "safety_pass": True,
            "energy_budget_pass": True,
            "mean_incremental_ramp_impact": -0.1,
        }
        for algorithm in ("ppo", "sac")
        for seed in (2601, 2602, 2603)
    ]
    assert promotion_decision(rows, expected_seed_count=3)["promoted_algorithms"]
    curve = [
        {"split": "validation", "mean_incremental_ramp_impact": -0.10},
        {"split": "validation", "mean_incremental_ramp_impact": -0.102},
    ]
    assert extension_allowed(protocol, curve)["allowed"] is True
    try:
        assert_long_campaign_ready({})
    except RuntimeError:
        pass
    else:
        raise AssertionError("long campaign was not blocked before integration")


def test_miniature_training_resumption_and_determinism() -> dict[str, object]:
    protocol = load_protocol()
    assert protocol["algorithms"]["ppo"]["gamma"] == 1.0
    assert protocol["algorithms"]["sac"]["n_steps"] == 36
    evidence: dict[str, object] = {
        "schema_version": "ramp-pure-rl-fixture-evidence-v1",
        "protocol_id": protocol["protocol"]["id"],
        "protocol_sha256": protocol["_sha256"],
        "final_long_campaign_launched": False,
        "jobs": [],
    }
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        ppo = run_training(
            factory=make_fixture_env,
            protocol=protocol,
            algorithm="ppo",
            seed=2601,
            target_timesteps=64,
            output_dir=root / "ppo",
            n_envs=2,
            resume=False,
            fixture_profile=True,
        )
        sac_first = run_training(
            factory=make_fixture_env,
            protocol=protocol,
            algorithm="sac",
            seed=2602,
            target_timesteps=96,
            output_dir=root / "sac",
            n_envs=2,
            resume=False,
            fixture_profile=True,
        )
        assert sac_first["interaction_count"] == 96
        assert sac_first["interaction_count"] % sac_first["checkpoint_boundary_quantum"] == 0
        assert sac_first["multiobjective"]["lagrangian_updates"]
        assert sac_first["training_data_provenance"]["episodes"]
        sac_manifest_path = root / "sac" / "training_manifest.json"
        original_manifest_text = sac_manifest_path.read_text(encoding="utf-8")
        incompatible = json.loads(original_manifest_text)
        incompatible["job_identity"]["seed"] = -1
        sac_manifest_path.write_text(
            json.dumps(incompatible, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        try:
            run_training(
                factory=make_fixture_env,
                protocol=protocol,
                algorithm="sac",
                seed=2602,
                target_timesteps=192,
                output_dir=root / "sac",
                n_envs=2,
                resume=True,
                fixture_profile=True,
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("resume accepted an incompatible job identity")
        sac_manifest_path.write_text(original_manifest_text, encoding="utf-8")
        sac_resumed = run_training(
            factory=make_fixture_env,
            protocol=protocol,
            algorithm="sac",
            seed=2602,
            target_timesteps=192,
            output_dir=root / "sac",
            n_envs=2,
            resume=True,
            fixture_profile=True,
        )
        assert sac_resumed["resume"]["resumed"] is True
        assert sac_resumed["resume"]["prior_final_policy_sha256"] == sac_resumed["resume"]["loaded_policy_sha256"]
        assert sac_resumed["interaction_count"] > sac_first["interaction_count"]
        assert not verify_pure_rl_manifest(ppo)
        assert not verify_pure_rl_manifest(sac_resumed)
        tampered = deepcopy(sac_resumed)
        tampered["pure_rl_assertions"]["teacher"] = True
        assert verify_pure_rl_manifest(tampered)
        eval_one = evaluate_checkpoint(
            factory=make_fixture_env,
            algorithm="ppo",
            checkpoint_dir=root / "ppo",
            split="validation",
            seeds=[77],
            windows=["m-07-sealed-0000"],
        )
        eval_two = evaluate_checkpoint(
            factory=make_fixture_env,
            algorithm="ppo",
            checkpoint_dir=root / "ppo",
            split="validation",
            seeds=[77],
            windows=["m-07-sealed-0000"],
        )
        assert json.dumps(eval_one, sort_keys=True) == json.dumps(eval_two, sort_keys=True)
        assert eval_one["success_gate"]["split"] == "validation"
        assert eval_one["service_unserved"] == 0.0
        assert eval_one["batch_unfinished"] == 0.0
        assert eval_one["batch_expired"] == 0.0
        assert eval_one["terminal_work"] == 0.0
        assert eval_one["certificate_violations"] == 0
        assert eval_one["emergency_feasibility_rate"] == 0.0
        assert eval_one["future_leakage_detected"] is False
        assert len(eval_one["policy_episodes"][0]["ramp_h1"]) == 48 + 36
        assert all(
            episode["split"] == "train"
            for episode in sac_resumed["training_data_provenance"]["episodes"]
        )
        for record in (ppo, sac_first, sac_resumed):
            evidence["jobs"].append(
                {
                    key: record[key]
                    for key in (
                        "algorithm",
                        "seed",
                        "semantic_feasible_action",
                        "raw_redundant_projected_logits",
                        "interaction_count",
                        "update_count",
                        "initial_policy_sha256",
                        "initial_critic_sha256",
                        "final_policy_sha256",
                        "final_critic_sha256",
                        "replay_provenance",
                        "pure_rl_assertions",
                        "normalization",
                        "data_split",
                        "training_data_provenance",
                        "forecast_identity",
                        "source_bundle_sha256",
                        "job_identity",
                        "protocol_sha256",
                        "credit_assignment",
                        "multiobjective",
                        "pure_rl_verification",
                    )
                }
            )
        evidence["deterministic_evaluation_sha256"] = __import__("hashlib").sha256(
            json.dumps(eval_one, sort_keys=True).encode("utf-8")
        ).hexdigest()
        evidence["resumption_proven"] = True
    return evidence


def main() -> None:
    test_contract_and_evaluation_guard()
    test_lagrangian_and_selection_seal_test()
    test_staged_campaign_protocol()
    evidence = test_miniature_training_resumption_and_determinism()
    output = ROOT / "output" / "ramp_rl_v6" / "fixture_smoke_evidence.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("smoke_test_ramp_rl_v6: PASS")


if __name__ == "__main__":
    main()
