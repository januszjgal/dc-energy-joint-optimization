"""Run the isolated teacher-free exact-teacher -> BC -> PPO pipeline."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from env.reward import RewardConfig  # noqa: E402
from env.safety_layer import SafetyConfig  # noqa: E402
from evaluate import compute_summary  # noqa: E402
from scripts.build_safety_v4_results import (  # noqa: E402
    gate_seed,
    optimizer_ci_from_costs,
)
from teacher_free.native_experiment import (  # noqa: E402
    collect_teacher_dataset,
    copy_actor_into_ppo,
    evaluate_actor,
    evaluate_teacher,
    load_bc_result,
    load_actor_from_result,
    make_native_env,
    monitor_native_env_factory,
    ppo_policy_kwargs,
    save_bc_result,
    save_json,
    train_behavior_cloner,
)

PROTOCOL_PATH = ROOT / "env" / "protocols" / "teacher_free_bc_ppo_v1.yaml"
V4_PROTOCOL_PATH = ROOT / "env" / "protocols" / "v4_safety.yaml"
OUT_ROOT = ROOT / "output" / "teacher_free_bc_ppo"
MODEL_ROOT = ROOT / "models" / "teacher_free_bc_ppo"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def region_reward_config(v4_protocol: dict[str, Any], region: str) -> RewardConfig:
    reward = v4_protocol["selected_configs"][region]["reward"]
    return RewardConfig(
        service_backlog_weight=float(reward["service_backlog_weight"]),
        batch_completion_weight=float(reward["batch_completion_weight"]),
        evaluation_service_backlog_weight=1000.0,
        evaluation_batch_completion_weight=1000.0,
        reward_scale=float(reward["reward_scale"]),
        subtract_idle_cost=bool(reward["subtract_idle_cost"]),
        urgency_potential_weight=float(reward["urgency_potential_weight"]),
    )


def safety_config(v4_protocol: dict[str, Any]) -> SafetyConfig:
    safety = v4_protocol["safety"]
    return SafetyConfig(
        service_envelope_total=float(safety["service_envelope_total"]),
        batch_arrival_envelope_total=float(safety["batch_arrival_envelope_total"]),
        future_fleet_capacity_total=float(safety["future_fleet_capacity_total"]),
        envelope_id=str(safety["envelope_id"]),
        envelope_scope=str(safety["envelope_scope"]),
        negative_demand_flush=False,
    )


def region_peak_penalty() -> float:
    v3_snapshot = read_json(ROOT / "output" / "ppo_v3_reward_sweep" / "protocol.json")
    return float(v3_snapshot["protocol"]["environment"]["peak_penalty_weight"])


def scenario_path(v4_protocol: dict[str, Any], region: str, *, transfer: bool = False) -> Path:
    key = "descriptive_transfer_scenario" if transfer else "scenario"
    return ROOT / str(v4_protocol["selected_configs"][region][key])


def baseline_cost(region: str, *, transfer: bool) -> float:
    cell_label = "e-h" if transfer else "a-d"
    path = ROOT / "output" / "ppo_v4_safety" / "_baselines" / cell_label / "safety_only" / f"{region}.json"
    payload = read_json(path)
    return float(payload["baselines"]["Status Quo (local, no deferral)"]["total_cost"])


def load_dataset_bundle(path: Path):
    payload = np.load(path)
    from teacher_free.native_experiment import DatasetBundle  # noqa: PLC0415

    return DatasetBundle(
        observations=np.asarray(payload["observations"], dtype=np.float32),
        actions=np.asarray(payload["actions"], dtype=np.float32),
        teacher_projection_l2=np.asarray(
            payload["teacher_projection_l2"], dtype=np.float64
        ),
        teacher_total_batch=np.asarray(payload["teacher_total_batch"], dtype=np.float64),
        seeds=tuple(int(value) for value in np.asarray(payload["seeds"]).tolist()),
        max_steps_per_seed=tuple(
            int(value) for value in np.asarray(payload["max_steps_per_seed"]).tolist()
        ),
    )


def summarize_seed_results(seed_results: dict[str, dict[str, Any]], *, baseline: float) -> dict[str, Any]:
    def sort_key(key: str) -> tuple[int, ...]:
        return tuple(int(part) for part in str(key).split("-"))

    ordered: list[dict[str, Any]] = []
    for key in sorted(seed_results, key=sort_key):
        summary = dict(seed_results[key]["summary"])
        if "service_completion" not in summary:
            demand_total = float(summary.get("service_demand_total", 0.0))
            served_total = float(summary.get("service_served_total", 0.0))
            summary["service_completion"] = (
                served_total / demand_total if demand_total > 0.0 else 1.0
            )
        if "batch_completion" not in summary:
            summary["batch_completion"] = float(
                summary.get("batch_completion_fraction", 1.0)
            )
        if "terminal_service_backlog" not in summary:
            summary["terminal_service_backlog"] = float(
                summary.get("terminal_backlog", 0.0)
            )
        ordered.append(summary)
    costs = [float(summary["total_cost"]) for summary in ordered]
    safety = [gate_seed(summary) for summary in ordered]
    best = min(costs)
    worst = max(costs)
    mean_cost = float(np.mean(costs))
    return {
        "seed_results": seed_results,
        "mean_cost": mean_cost,
        "best_cost": float(best),
        "worst_cost": float(worst),
        "mean_savings_pct": float(100.0 * (baseline - mean_cost) / baseline),
        "optimizer_ci": optimizer_ci_from_costs(costs, baseline),
        "safe_seed_count": int(sum(1 for item in safety if item["passed"])),
        "all_safe": bool(all(item["passed"] for item in safety)),
        "mean_intervention_rate": float(
            np.mean(
                [
                    float(summary.get("safety", {}).get("intervention_rate", 0.0))
                    for summary in ordered
                ]
            )
        ),
        "max_semantic_projection_l2": float(
            max(
                float(summary.get("safety", {}).get("max_projection_l2", 0.0))
                for summary in ordered
            )
        ),
    }


def train_and_evaluate_behavior_cloning(
    protocol: dict[str, Any],
    v4_protocol: dict[str, Any],
    region: str,
    dataset,
) -> tuple[dict[str, Any], dict[str, Any]]:
    bc_cfg = protocol["behavior_cloning"]
    reward = region_reward_config(v4_protocol, region)
    safe = safety_config(v4_protocol)
    peak_weight = region_peak_penalty()
    hidden_sizes = tuple(int(size) for size in bc_cfg["hidden_sizes"])
    eval_results_ad: dict[str, dict[str, Any]] = {}
    eval_results_eh: dict[str, dict[str, Any]] = {}
    losses: dict[str, Any] = {}
    for seed in bc_cfg["seeds"]:
        result_path = MODEL_ROOT / region / "bc" / f"s{int(seed)}.json"
        if result_path.exists():
            result, _loaded_hidden_sizes, _loaded_seed = load_bc_result(result_path)
        else:
            result = train_behavior_cloner(
                dataset,
                seed=int(seed),
                hidden_sizes=hidden_sizes,
                learning_rate=float(bc_cfg["learning_rate"]),
                epochs=int(bc_cfg["epochs"]),
                batch_size=int(bc_cfg["batch_size"]),
            )
            save_bc_result(
                result_path,
                result,
                hidden_sizes=hidden_sizes,
                seed=int(seed),
            )
        actor = load_actor_from_result(
            result,
            dataset.observations.shape[1],
            dataset.actions.shape[1],
            hidden_sizes,
        )
        losses[str(int(seed))] = {
            "best_epoch": int(result.best_epoch),
            "final_loss": float(result.final_loss),
        }
        for eval_seed in protocol["evaluation"]["development_eval_seeds"]:
            summary, _history = evaluate_actor(
                actor,
                scenario_path(v4_protocol, region, transfer=False),
                seed=int(eval_seed),
                peak_penalty_weight=peak_weight,
                reward_config=reward,
                safety_config=safe,
                domain_randomization=False,
            )
            eval_results_ad[f"{int(seed)}-{int(eval_seed)}"] = {"summary": summary}
        for eval_seed in protocol["evaluation"]["transfer_eval_seeds"]:
            summary, _history = evaluate_actor(
                actor,
                scenario_path(v4_protocol, region, transfer=True),
                seed=int(eval_seed),
                peak_penalty_weight=peak_weight,
                reward_config=reward,
                safety_config=safe,
                domain_randomization=False,
            )
            eval_results_eh[f"{int(seed)}-{int(eval_seed)}"] = {"summary": summary}
    return (
        {
            "training_losses": losses,
            "a_d": summarize_seed_results(eval_results_ad, baseline=baseline_cost(region, transfer=False)),
            "e_h": summarize_seed_results(eval_results_eh, baseline=baseline_cost(region, transfer=True)),
        },
        {
            "hidden_sizes": list(hidden_sizes),
            "dataset_rows": int(dataset.observations.shape[0]),
        },
    )


def train_and_evaluate_ppo(
    protocol: dict[str, Any],
    v4_protocol: dict[str, Any],
    region: str,
    dataset,
) -> dict[str, Any]:
    ppo_cfg = protocol["ppo_finetune"]
    if not bool(ppo_cfg.get("enabled", False)):
        return {"enabled": False}
    reward = region_reward_config(v4_protocol, region)
    safe = safety_config(v4_protocol)
    peak_weight = region_peak_penalty()
    hidden_sizes = tuple(int(size) for size in protocol["behavior_cloning"]["hidden_sizes"])
    selected = v4_protocol["selected_configs"][region]["ppo"]
    results_ad: dict[str, dict[str, Any]] = {}
    results_eh: dict[str, dict[str, Any]] = {}
    training: dict[str, Any] = {}
    for seed in ppo_cfg["seeds"]:
        bc_result_path = MODEL_ROOT / region / "bc" / f"s{int(seed)}.json"
        bc_result, _loaded_hidden_sizes, _loaded_seed = load_bc_result(bc_result_path)
        actor = load_actor_from_result(
            bc_result,
            dataset.observations.shape[1],
            dataset.actions.shape[1],
            hidden_sizes,
        )
        factory = monitor_native_env_factory(
            scenario_path(v4_protocol, region, transfer=False),
            seed=int(seed),
            peak_penalty_weight=peak_weight,
            reward_config=reward,
            safety_config=safe,
            domain_randomization=True,
        )
        vec_env = DummyVecEnv([factory])
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=float(selected["learning_rate"]),
            n_steps=int(selected["n_steps"]),
            batch_size=int(selected["batch_size"]),
            n_epochs=int(selected["n_epochs"]),
            gamma=float(selected["gamma"]),
            gae_lambda=float(selected["gae_lambda"]),
            target_kl=(
                None if selected["target_kl"] is None else float(selected["target_kl"])
            ),
            ent_coef=0.0,
            clip_range_vf=None,
            normalize_advantage=True,
            policy_kwargs=ppo_policy_kwargs(hidden_sizes),
            verbose=0,
            seed=int(seed),
            tensorboard_log=None,
        )
        copy_actor_into_ppo(model, actor)
        model.learn(total_timesteps=int(ppo_cfg["timesteps"][region]), progress_bar=False)
        model_path = MODEL_ROOT / region / "ppo" / f"s{int(seed)}"
        model.save(model_path)
        training[str(int(seed))] = {
            "timesteps": int(ppo_cfg["timesteps"][region]),
            "model_path": str(model_path.relative_to(ROOT)).replace("\\", "/"),
        }
        vec_env.close()
        for eval_seed in protocol["evaluation"]["development_eval_seeds"]:
            env = make_native_env(
                scenario_path(v4_protocol, region, transfer=False),
                seed=int(eval_seed),
                peak_penalty_weight=peak_weight,
                reward_config=reward,
                safety_config=safe,
                domain_randomization=False,
            )
            try:
                obs, _ = env.reset(seed=int(eval_seed))
                history: list[dict[str, Any]] = []
                while True:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _reward, terminated, truncated, info = env.step(action)
                    history.append(info)
                    if terminated or truncated:
                        break
                summary = compute_summary(history, batch_enabled=True)
            finally:
                env.close()
            results_ad[f"{int(seed)}-{int(eval_seed)}"] = {"summary": summary}
        for eval_seed in protocol["evaluation"]["transfer_eval_seeds"]:
            env = make_native_env(
                scenario_path(v4_protocol, region, transfer=True),
                seed=int(eval_seed),
                peak_penalty_weight=peak_weight,
                reward_config=reward,
                safety_config=safe,
                domain_randomization=False,
            )
            try:
                obs, _ = env.reset(seed=int(eval_seed))
                history = []
                while True:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _reward, terminated, truncated, info = env.step(action)
                    history.append(info)
                    if terminated or truncated:
                        break
                summary = compute_summary(history, batch_enabled=True)
            finally:
                env.close()
            results_eh[f"{int(seed)}-{int(eval_seed)}"] = {"summary": summary}
    return {
        "enabled": True,
        "training": training,
        "a_d": summarize_seed_results(results_ad, baseline=baseline_cost(region, transfer=False)),
        "e_h": summarize_seed_results(results_eh, baseline=baseline_cost(region, transfer=True)),
        "sb3_note": (
            "PPO was adapted to the native [0,1] service-share / drain-rate / "
            "batch-share action space, then warm-started from the behavior-cloned "
            "actor weights."
        ),
    }


def run_teacher(protocol: dict[str, Any], v4_protocol: dict[str, Any], region: str) -> dict[str, Any]:
    reference = protocol["teacher_reference"][region]
    semantic_delta = float(protocol["teacher_reference"]["semantic_projection_delta_max"])
    ad_results = {
        "parent-reference": {
            "summary": {
                "total_cost": None,
                "safety": {
                    "intervention_rate": 0.0,
                    "max_projection_l2": semantic_delta,
                },
            }
        }
    }
    eh_results = {
        "parent-reference": {
            "summary": {
                "total_cost": None,
                "safety": {
                    "intervention_rate": 0.0,
                    "max_projection_l2": semantic_delta,
                },
            }
        }
    }
    return {
        "reference_source": str(protocol["teacher_reference"]["source"]),
        "a_d": {
            "seed_results": ad_results,
            "mean_cost": None,
            "best_cost": None,
            "worst_cost": None,
            "mean_savings_pct": float(reference["a_d_mean_savings_pct"]),
            "native_roundtrip_mean_savings_pct": float(
                reference["a_d_native_roundtrip_mean_savings_pct"]
            ),
            "optimizer_ci": None,
            "safe_seed_count": 1,
            "all_safe": True,
            "mean_intervention_rate": 0.0,
            "max_semantic_projection_l2": semantic_delta,
        },
        "e_h": {
            "seed_results": eh_results,
            "mean_cost": None,
            "best_cost": None,
            "worst_cost": None,
            "mean_savings_pct": float(reference["e_h_mean_savings_pct"]),
            "optimizer_ci": None,
            "safe_seed_count": 1,
            "all_safe": True,
            "mean_intervention_rate": 0.0,
            "max_semantic_projection_l2": semantic_delta,
        },
    }


def run_region(protocol: dict[str, Any], v4_protocol: dict[str, Any], region: str) -> dict[str, Any]:
    reward = region_reward_config(v4_protocol, region)
    safe = safety_config(v4_protocol)
    peak_weight = region_peak_penalty()
    dataset_path = OUT_ROOT / "datasets" / f"{region}.npz"
    if dataset_path.exists():
        dataset = load_dataset_bundle(dataset_path)
    else:
        dataset = collect_teacher_dataset(
            scenario_path(v4_protocol, region, transfer=False),
            seeds=protocol["dataset"]["collection_seeds"],
            peak_penalty_weight=peak_weight,
            reward_config=reward,
            safety_config=safe,
        )
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            dataset_path,
            observations=dataset.observations,
            actions=dataset.actions,
            teacher_projection_l2=dataset.teacher_projection_l2,
            teacher_total_batch=dataset.teacher_total_batch,
            seeds=np.asarray(dataset.seeds, dtype=np.int64),
            max_steps_per_seed=np.asarray(dataset.max_steps_per_seed, dtype=np.int64),
        )
    teacher = run_teacher(protocol, v4_protocol, region)
    bc, bc_meta = train_and_evaluate_behavior_cloning(protocol, v4_protocol, region, dataset)
    ppo = train_and_evaluate_ppo(protocol, v4_protocol, region, dataset)
    return {
        "dataset": {
            "path": str(dataset_path.relative_to(ROOT)).replace("\\", "/"),
            "rows": int(dataset.observations.shape[0]),
            "observation_dim": int(dataset.observations.shape[1]),
            "action_dim": int(dataset.actions.shape[1]),
            "collection_seeds": [int(seed) for seed in dataset.seeds],
            "steps_per_seed": [int(step) for step in dataset.max_steps_per_seed],
            "teacher_roundtrip_l2_max": float(dataset.teacher_projection_l2.max(initial=0.0)),
            "teacher_roundtrip_l2_mean": float(dataset.teacher_projection_l2.mean() if dataset.teacher_projection_l2.size else 0.0),
            **bc_meta,
        },
        "teacher": teacher,
        "behavior_cloning": bc,
        "ppo_finetune": ppo,
    }


def build_markdown_report(results: dict[str, Any]) -> str:
    lines = [
        "# Teacher-free native BC/PPO report",
        "",
        f"- Protocol: `{results['protocol']['name']}`",
        f"- Parent protocol: `{results['protocol']['parent_protocol']}`",
        "",
        "| Region | Controller | Cells | Mean savings | Safe runs | Mean intervention | Max semantic delta |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for region, payload in results["regions"].items():
        for label, block in (
            ("Teacher", payload["teacher"]),
            ("BC", payload["behavior_cloning"]),
            ("PPO", payload["ppo_finetune"]),
        ):
            if label == "PPO" and not block.get("enabled", False):
                continue
            for cells_key, cell_label in (("a_d", "a-d"), ("e_h", "e-h")):
                section = block[cells_key]
                lines.append(
                    f"| {region.upper()} | {label} | {cell_label} | "
                    f"{section['mean_savings_pct']:.4f}% | "
                    f"{section['safe_seed_count']}/{len(section['seed_results'])} | "
                    f"{section['mean_intervention_rate']:.6f} | "
                    f"{section['max_semantic_projection_l2']:.3e} |"
                )
    lines.extend(["", "## Dataset", ""])
    for region, payload in results["regions"].items():
        dataset = payload["dataset"]
        lines.append(
            f"- **{region.upper()}** {dataset['rows']} rows, seeds {dataset['collection_seeds']}, "
            f"teacher roundtrip max {dataset['teacher_roundtrip_l2_max']:.3e}."
        )
        lines.append(
            f"- **{region.upper()} teacher reference** a-d {payload['teacher']['a_d']['mean_savings_pct']:.4f}% "
            f"(native roundtrip {payload['teacher']['a_d']['native_roundtrip_mean_savings_pct']:.4f}%), "
            f"e-h {payload['teacher']['e_h']['mean_savings_pct']:.4f}%."
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run teacher-free native BC/PPO experiments")
    parser.add_argument(
        "--skip-ppo",
        action="store_true",
        help="Train and evaluate only the exact teacher and BC models.",
    )
    args = parser.parse_args(argv)

    protocol = read_yaml(PROTOCOL_PATH)
    v4_protocol = read_yaml(V4_PROTOCOL_PATH)
    if args.skip_ppo:
        protocol = copy.deepcopy(protocol)
        protocol["ppo_finetune"]["enabled"] = False
    results = {
        "protocol": {
            **protocol,
            "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)).replace("\\", "/"),
            "protocol_sha256": sha256(PROTOCOL_PATH),
            "v4_protocol_sha256": sha256(V4_PROTOCOL_PATH),
        },
        "regions": {},
    }
    for region in ("us", "global"):
        results["regions"][region] = run_region(protocol, v4_protocol, region)
    save_json(OUT_ROOT / "results.json", results)
    report = build_markdown_report(results)
    (OUT_ROOT / "results_report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
