"""US-only iterative DAgger follow-up for the teacher-free experiment."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from env.reward import RewardConfig  # noqa: E402
from env.safety_layer import SafetyConfig, project_capped_simplex  # noqa: E402
from evaluate import compute_summary  # noqa: E402
from teacher_free.native_experiment import (  # noqa: E402
    decode_native_action,
    exact_teacher_action,
    load_actor_from_result,
    load_bc_result,
    make_native_env,
    predict_actor_action,
)

PROTOCOL_PATH = ROOT / "env" / "protocols" / "teacher_free_us_v2.yaml"
V4_PROTOCOL_PATH = ROOT / "env" / "protocols" / "v4_safety.yaml"
OUT_ROOT = ROOT / "output" / "teacher_free_us_v2_dagger"
MODEL_ROOT = ROOT / "models" / "teacher_free_us_v2_dagger"
FAILED_RESULT_PATH = ROOT / "output" / "teacher_free_bc_ppo" / "failed_resume_us_result.json"


@dataclass(frozen=True)
class FactoredDataset:
    observations: np.ndarray
    service_targets: np.ndarray
    origin_targets: np.ndarray
    batch_total_targets: np.ndarray
    destination_targets: np.ndarray
    service_totals: np.ndarray
    max_batch_totals: np.ndarray
    service_amount_targets: np.ndarray
    origin_amount_targets: np.ndarray
    destination_amount_targets: np.ndarray
    sample_weights: np.ndarray
    intervention_flags: np.ndarray
    capacity_binding_flags: np.ndarray
    ranking_error_flags: np.ndarray
    source: str
    rollout_seed: int


class FactoredActor(torch.nn.Module):
    def __init__(self, obs_dim: int, hidden_sizes: Sequence[int], n_dc: int) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = []
        in_dim = int(obs_dim)
        for hidden in [int(size) for size in hidden_sizes]:
            linear = torch.nn.Linear(in_dim, hidden)
            torch.nn.init.xavier_uniform_(linear.weight)
            torch.nn.init.zeros_(linear.bias)
            layers.extend([linear, torch.nn.Tanh()])
            in_dim = hidden
        self.body = torch.nn.Sequential(*layers)
        self.service_head = torch.nn.Linear(in_dim, n_dc)
        self.origin_head = torch.nn.Linear(in_dim, n_dc)
        self.destination_head = torch.nn.Linear(in_dim, n_dc)
        self.total_head = torch.nn.Linear(in_dim, 1)
        for head in (
            self.service_head,
            self.origin_head,
            self.destination_head,
            self.total_head,
        ):
            torch.nn.init.xavier_uniform_(head.weight)
            torch.nn.init.zeros_(head.bias)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.body(obs)
        return (
            torch.softmax(self.service_head(features), dim=-1),
            torch.softmax(self.origin_head(features), dim=-1),
            torch.softmax(self.destination_head(features), dim=-1),
            torch.sigmoid(self.total_head(features)).squeeze(-1),
        )


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


def region_reward_config(v4_protocol: dict[str, Any]) -> RewardConfig:
    reward = v4_protocol["selected_configs"]["us"]["reward"]
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


def peak_penalty_weight() -> float:
    return float(read_json(ROOT / "output" / "ppo_v3_reward_sweep" / "protocol.json")["protocol"]["environment"]["peak_penalty_weight"])


def scenario_path(v4_protocol: dict[str, Any], *, transfer: bool = False) -> Path:
    key = "descriptive_transfer_scenario" if transfer else "scenario"
    return ROOT / str(v4_protocol["selected_configs"]["us"][key])


def build_post_arrival_state(env) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, float]:
    t = int(env.step_index)
    copied_pools = []
    for index, site in enumerate(env.sites):
        pool = type(site.batch_pool)()
        pool.entries.extend(site.batch_pool.entries)
        arrival = float(site.get_batch_demand(t))
        if arrival > 0.0:
            pool.add(arrival, t + env._deadline_offsets[index])
        expired = float(pool.expire(t))
        if expired > env.safety_config.tolerance:
            raise RuntimeError("unsalvageable carryover reached v2 data collection")
        copied_pools.append(pool)
    pool_totals = np.asarray([pool.total_demand for pool in copied_pools], dtype=np.float64)
    service_total = float(sum(site.get_service_demand(t) for site in env.sites))
    effective_capacity = env._effective_capacities()
    service_vector = np.asarray([site.get_service_demand(t) for site in env.sites], dtype=np.float64)
    max_batch_total = max(
        0.0,
        min(float(pool_totals.sum()), float(effective_capacity.sum()) - service_total),
    )
    return pool_totals, service_total, effective_capacity, service_vector, max_batch_total


def factored_targets_from_teacher(env, teacher_action: np.ndarray, total_batch_target: float) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    n_dc = env.n_dc
    pool_totals, service_total, effective_capacity, _service_vector, max_batch_total = build_post_arrival_state(env)
    service_frac = np.asarray(teacher_action[:n_dc], dtype=np.float64)
    drain_rates = np.asarray(teacher_action[n_dc : 2 * n_dc], dtype=np.float64)
    batch_frac = np.asarray(teacher_action[2 * n_dc :], dtype=np.float64)
    teacher_service = project_capped_simplex(
        service_frac * service_total,
        service_total,
        effective_capacity,
    )
    teacher_total_batch = min(float(total_batch_target), float(max_batch_total))
    origin_batch = drain_rates * pool_totals
    origin_total = float(origin_batch.sum())
    if origin_total > 1e-12:
        origin_simplex = origin_batch / origin_total
    else:
        origin_simplex = np.full(n_dc, 1.0 / n_dc, dtype=np.float64)
    batch_total_scalar = (
        teacher_total_batch / max_batch_total if max_batch_total > 1e-12 else 0.0
    )
    return (
        np.divide(
            teacher_service,
            service_total,
            out=np.zeros_like(teacher_service),
            where=service_total > 1e-12,
        ),
        origin_simplex,
        float(np.clip(batch_total_scalar, 0.0, 1.0)),
        batch_frac,
    )


def pairwise_ranking_error(reference: np.ndarray, candidate: np.ndarray, *, tolerance: float = 1e-6) -> float:
    disagreements = 0.0
    comparisons = 0.0
    for i in range(len(reference)):
        for j in range(i + 1, len(reference)):
            ref_delta = float(reference[i] - reference[j])
            if abs(ref_delta) <= tolerance:
                continue
            cand_delta = float(candidate[i] - candidate[j])
            comparisons += 1.0
            if ref_delta * cand_delta < -tolerance:
                disagreements += 1.0
    if comparisons <= 0.0:
        return 0.0
    return disagreements / comparisons


def factored_action_components(
    model: FactoredActor,
    env,
    obs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with torch.no_grad():
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        service_simplex, origin_simplex, destination_simplex, batch_total = model(obs_tensor)
    service_simplex_np = service_simplex.squeeze(0).cpu().numpy().astype(np.float64)
    origin_simplex_np = origin_simplex.squeeze(0).cpu().numpy().astype(np.float64)
    destination_simplex_np = destination_simplex.squeeze(0).cpu().numpy().astype(np.float64)
    batch_total_np = float(batch_total.squeeze(0).cpu().item())
    pool_totals, service_total, effective_capacity, _service_vector, max_batch_total = build_post_arrival_state(env.unwrapped)
    desired_service = project_capped_simplex(
        service_simplex_np * service_total,
        service_total,
        effective_capacity,
    )
    total_batch = float(np.clip(batch_total_np, 0.0, 1.0) * max_batch_total)
    if total_batch > 1e-12:
        desired_origin = project_capped_simplex(
            origin_simplex_np * total_batch,
            total_batch,
            pool_totals,
        )
        residual_capacity = effective_capacity - desired_service
        desired_destination = project_capped_simplex(
            destination_simplex_np * total_batch,
            total_batch,
            residual_capacity,
        )
    else:
        desired_origin = np.zeros_like(pool_totals)
        desired_destination = np.zeros_like(pool_totals)
    service_fraction = np.divide(
        desired_service,
        service_total,
        out=np.zeros_like(desired_service),
        where=service_total > 1e-12,
    )
    drain_rates = np.divide(
        desired_origin,
        pool_totals,
        out=np.zeros_like(desired_origin),
        where=pool_totals > 1e-12,
    )
    batch_fraction = np.divide(
        desired_destination,
        total_batch,
        out=np.zeros_like(desired_destination),
        where=total_batch > 1e-12,
    )
    semantic_action = np.concatenate([service_fraction, drain_rates, batch_fraction]).astype(np.float32)
    return semantic_action, desired_service, desired_origin, desired_destination


def policy_action_components(
    controller_actor,
    env,
    obs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(controller_actor, FactoredActor):
        action, desired_service, _desired_origin, desired_destination = factored_action_components(controller_actor, env, obs)
    else:
        action = predict_actor_action(controller_actor, obs)
        pool_totals, service_total, _effective_capacity, _service_vector, _max_batch_total = build_post_arrival_state(env.unwrapped)
        desired_service, _desired_origin, desired_destination = decode_native_action(action, service_total, pool_totals)
    total_load = desired_service + desired_destination
    return action, total_load


def collect_factored_rollout(
    protocol: dict[str, Any],
    v4_protocol: dict[str, Any],
    *,
    source: str,
    rollout_seed: int,
    controller_actor=None,
    domain_randomization: bool,
) -> FactoredDataset:
    reward = region_reward_config(v4_protocol)
    safe = safety_config(v4_protocol)
    env = make_native_env(
        scenario_path(v4_protocol, transfer=False),
        seed=int(rollout_seed),
        peak_penalty_weight=peak_penalty_weight(),
        reward_config=reward,
        safety_config=safe,
        domain_randomization=domain_randomization,
    )
    obs_rows: list[np.ndarray] = []
    service_rows: list[np.ndarray] = []
    origin_rows: list[np.ndarray] = []
    total_rows: list[float] = []
    dest_rows: list[np.ndarray] = []
    service_total_rows: list[float] = []
    max_batch_total_rows: list[float] = []
    service_amount_rows: list[np.ndarray] = []
    origin_amount_rows: list[np.ndarray] = []
    dest_amount_rows: list[np.ndarray] = []
    weight_rows: list[float] = []
    intervention_rows: list[float] = []
    capacity_rows: list[float] = []
    ranking_rows: list[float] = []
    obs, _ = env.reset(seed=int(rollout_seed))
    while True:
        teacher_action, diagnostics = exact_teacher_action(env.unwrapped)
        service_tgt, origin_tgt, total_tgt, dest_tgt = factored_targets_from_teacher(
            env.unwrapped,
            teacher_action,
            diagnostics.total_batch_target,
        )
        pool_totals, _service_total, effective_capacity, _service_vector, max_batch_total = build_post_arrival_state(
            env.unwrapped
        )
        service_total = float(_service_total)
        capacity_binding = float(
            max_batch_total < float(pool_totals.sum()) - env.unwrapped.safety_config.tolerance
        )
        edge_state = float(
            np.any((service_tgt < 0.02) | (service_tgt > 0.98))
            or np.any((origin_tgt < 0.02) | (origin_tgt > 0.98))
            or total_tgt < 0.02
            or total_tgt > 0.98
            or np.any((dest_tgt < 0.02) | (dest_tgt > 0.98))
        )
        teacher_total_batch = float(total_tgt * max_batch_total)
        teacher_service_amount = service_tgt.astype(np.float64) * service_total
        teacher_origin_amount = origin_tgt.astype(np.float64) * teacher_total_batch
        teacher_dest_amount = dest_tgt.astype(np.float64) * teacher_total_batch
        teacher_total_load = teacher_service_amount + teacher_dest_amount
        _ = effective_capacity
        ranking_error = 0.0
        obs_rows.append(np.asarray(obs, dtype=np.float32))
        service_rows.append(service_tgt.astype(np.float32))
        origin_rows.append(origin_tgt.astype(np.float32))
        total_rows.append(float(total_tgt))
        dest_rows.append(dest_tgt.astype(np.float32))
        service_total_rows.append(service_total)
        max_batch_total_rows.append(float(max_batch_total))
        service_amount_rows.append(teacher_service_amount.astype(np.float32))
        origin_amount_rows.append(teacher_origin_amount.astype(np.float32))
        dest_amount_rows.append(teacher_dest_amount.astype(np.float32))
        if controller_actor is None:
            action = teacher_action
        else:
            action, actor_total_load = policy_action_components(controller_actor, env, obs)
            ranking_error = pairwise_ranking_error(teacher_total_load, actor_total_load)
        obs, _reward, terminated, truncated, info = env.step(action)
        intervened = float(info.get("safety_intervened", False))
        weight_rows.append(
            float(
                1.0
                + 2.0 * capacity_binding
                + 3.0 * edge_state
                + 4.0 * ranking_error
                + 8.0 * intervened
            )
        )
        intervention_rows.append(intervened)
        capacity_rows.append(capacity_binding)
        ranking_rows.append(float(ranking_error > 0.0))
        if terminated or truncated:
            break
    env.close()
    return FactoredDataset(
        observations=np.asarray(obs_rows, dtype=np.float32),
        service_targets=np.asarray(service_rows, dtype=np.float32),
        origin_targets=np.asarray(origin_rows, dtype=np.float32),
        batch_total_targets=np.asarray(total_rows, dtype=np.float32),
        destination_targets=np.asarray(dest_rows, dtype=np.float32),
        service_totals=np.asarray(service_total_rows, dtype=np.float32),
        max_batch_totals=np.asarray(max_batch_total_rows, dtype=np.float32),
        service_amount_targets=np.asarray(service_amount_rows, dtype=np.float32),
        origin_amount_targets=np.asarray(origin_amount_rows, dtype=np.float32),
        destination_amount_targets=np.asarray(dest_amount_rows, dtype=np.float32),
        sample_weights=np.asarray(weight_rows, dtype=np.float32),
        intervention_flags=np.asarray(intervention_rows, dtype=np.float32),
        capacity_binding_flags=np.asarray(capacity_rows, dtype=np.float32),
        ranking_error_flags=np.asarray(ranking_rows, dtype=np.float32),
        source=source,
        rollout_seed=int(rollout_seed),
    )


def concat_datasets(datasets: Sequence[FactoredDataset]) -> FactoredDataset:
    return FactoredDataset(
        observations=np.concatenate([dataset.observations for dataset in datasets], axis=0),
        service_targets=np.concatenate([dataset.service_targets for dataset in datasets], axis=0),
        origin_targets=np.concatenate([dataset.origin_targets for dataset in datasets], axis=0),
        batch_total_targets=np.concatenate([dataset.batch_total_targets for dataset in datasets], axis=0),
        destination_targets=np.concatenate([dataset.destination_targets for dataset in datasets], axis=0),
        service_totals=np.concatenate([dataset.service_totals for dataset in datasets], axis=0),
        max_batch_totals=np.concatenate([dataset.max_batch_totals for dataset in datasets], axis=0),
        service_amount_targets=np.concatenate([dataset.service_amount_targets for dataset in datasets], axis=0),
        origin_amount_targets=np.concatenate([dataset.origin_amount_targets for dataset in datasets], axis=0),
        destination_amount_targets=np.concatenate([dataset.destination_amount_targets for dataset in datasets], axis=0),
        sample_weights=np.concatenate([dataset.sample_weights for dataset in datasets], axis=0),
        intervention_flags=np.concatenate([dataset.intervention_flags for dataset in datasets], axis=0),
        capacity_binding_flags=np.concatenate([dataset.capacity_binding_flags for dataset in datasets], axis=0),
        ranking_error_flags=np.concatenate([dataset.ranking_error_flags for dataset in datasets], axis=0),
        source="+".join(dataset.source for dataset in datasets),
        rollout_seed=-1,
    )


def train_factored_actor(dataset: FactoredDataset, seed: int, hidden_sizes: Sequence[int], learning_rate: float, epochs: int, batch_size: int) -> tuple[FactoredActor, list[float]]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    model = FactoredActor(dataset.observations.shape[1], hidden_sizes, dataset.service_targets.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate), weight_decay=1e-6)
    rng = np.random.default_rng(int(seed))
    obs = torch.from_numpy(dataset.observations)
    svc = torch.from_numpy(dataset.service_targets)
    org = torch.from_numpy(dataset.origin_targets)
    tot = torch.from_numpy(dataset.batch_total_targets)
    dst = torch.from_numpy(dataset.destination_targets)
    svc_total = torch.from_numpy(dataset.service_totals)
    max_batch_total = torch.from_numpy(dataset.max_batch_totals)
    svc_amount = torch.from_numpy(dataset.service_amount_targets)
    org_amount = torch.from_numpy(dataset.origin_amount_targets)
    dst_amount = torch.from_numpy(dataset.destination_amount_targets)
    sample_weights = np.asarray(dataset.sample_weights, dtype=np.float64)
    sample_prob = sample_weights / float(sample_weights.sum())
    losses: list[float] = []
    best_loss = math.inf
    best_state = None
    for _epoch in range(int(epochs)):
        permutation = torch.from_numpy(
            rng.choice(
                len(obs),
                size=len(obs),
                replace=True,
                p=sample_prob,
            )
        )
        batch_losses = []
        for start in range(0, len(obs), int(batch_size)):
            idx = permutation[start : start + int(batch_size)]
            optimizer.zero_grad(set_to_none=True)
            pred_svc, pred_org, pred_dst, pred_tot = model(obs[idx])
            batch_amount = pred_tot * max_batch_total[idx]
            pred_svc_amount = pred_svc * svc_total[idx].unsqueeze(-1)
            pred_org_amount = pred_org * batch_amount.unsqueeze(-1)
            pred_dst_amount = pred_dst * batch_amount.unsqueeze(-1)
            service_loss = torch.nn.functional.mse_loss(pred_svc, svc[idx])
            origin_loss = torch.nn.functional.mse_loss(pred_org, org[idx])
            destination_loss = torch.nn.functional.mse_loss(pred_dst, dst[idx])
            total_loss = torch.nn.functional.binary_cross_entropy(
                pred_tot.clamp(1e-6, 1.0 - 1e-6),
                tot[idx],
            )
            service_scale = svc_total[idx].unsqueeze(-1).clamp_min(1.0)
            batch_scale = max_batch_total[idx].unsqueeze(-1).clamp_min(1.0)
            service_amount_loss = torch.nn.functional.mse_loss(
                pred_svc_amount / service_scale,
                svc_amount[idx] / service_scale,
            )
            origin_amount_loss = torch.nn.functional.mse_loss(
                pred_org_amount / batch_scale,
                org_amount[idx] / batch_scale,
            )
            destination_amount_loss = torch.nn.functional.mse_loss(
                pred_dst_amount / batch_scale,
                dst_amount[idx] / batch_scale,
            )
            loss = (
                service_loss
                + origin_loss
                + destination_loss
                + total_loss
                + service_amount_loss
                + origin_amount_loss
                + destination_amount_loss
            ) / 7.0
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu().item()))
        epoch_loss = float(np.mean(batch_losses))
        losses.append(epoch_loss)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("factored actor did not train")
    model.load_state_dict(best_state)
    model.eval()
    return model, losses


def decode_factored_action(model: FactoredActor, env, obs: np.ndarray) -> np.ndarray:
    action, _desired_service, _desired_origin, _desired_destination = factored_action_components(model, env, obs)
    return action


def evaluate_us_actor(model: FactoredActor, v4_protocol: dict[str, Any], seed: int, *, transfer: bool = False) -> dict[str, Any]:
    reward = region_reward_config(v4_protocol)
    safe = safety_config(v4_protocol)
    env = make_native_env(
        scenario_path(v4_protocol, transfer=transfer),
        seed=int(seed),
        peak_penalty_weight=peak_penalty_weight(),
        reward_config=reward,
        safety_config=safe,
        domain_randomization=False,
    )
    obs, _ = env.reset(seed=int(seed))
    history = []
    while True:
        action = decode_factored_action(model, env, obs)
        obs, _reward, terminated, truncated, info = env.step(action)
        history.append(info)
        if terminated or truncated:
            break
    env.close()
    return compute_summary(history, batch_enabled=True)


def baseline_cost(transfer: bool) -> float:
    cell_label = "e-h" if transfer else "a-d"
    path = ROOT / "output" / "ppo_v4_safety" / "_baselines" / cell_label / "safety_only" / "us.json"
    payload = read_json(path)
    return float(payload["baselines"]["Status Quo (local, no deferral)"]["total_cost"])


def save_dataset(path: Path, dataset: FactoredDataset) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        observations=dataset.observations,
        service_targets=dataset.service_targets,
        origin_targets=dataset.origin_targets,
        batch_total_targets=dataset.batch_total_targets,
        destination_targets=dataset.destination_targets,
        service_totals=dataset.service_totals,
        max_batch_totals=dataset.max_batch_totals,
        service_amount_targets=dataset.service_amount_targets,
        origin_amount_targets=dataset.origin_amount_targets,
        destination_amount_targets=dataset.destination_amount_targets,
        sample_weights=dataset.sample_weights,
        intervention_flags=dataset.intervention_flags,
        capacity_binding_flags=dataset.capacity_binding_flags,
        ranking_error_flags=dataset.ranking_error_flags,
    )


def train_iteration(
    dataset: FactoredDataset,
    protocol: dict[str, Any],
    v4_protocol: dict[str, Any],
    iteration_index: int,
) -> tuple[dict[str, Any], FactoredActor]:
    eval_rows: dict[str, Any] = {}
    best_cost = math.inf
    best_model = None
    best_seed = None
    for seed in protocol["behavior_cloning"]["seeds"]:
        model, losses = train_factored_actor(
            dataset,
            seed=int(seed),
            hidden_sizes=protocol["behavior_cloning"]["hidden_sizes"],
            learning_rate=float(protocol["behavior_cloning"]["learning_rate"]),
            epochs=int(protocol["behavior_cloning"]["epochs"]),
            batch_size=int(protocol["behavior_cloning"]["batch_size"]),
        )
        model_path = MODEL_ROOT / "us" / f"iter_{iteration_index:02d}" / f"s{int(seed)}.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "losses": losses}, model_path)
        summary = evaluate_us_actor(model, v4_protocol, int(protocol["evaluation"]["a_d_eval_seed"]), transfer=False)
        summary["training_loss_best"] = float(min(losses))
        summary["training_loss_final"] = float(losses[-1])
        eval_rows[str(int(seed))] = summary
        if float(summary["total_cost"]) < best_cost:
            best_cost = float(summary["total_cost"])
            best_seed = int(seed)
            best_model = copy.deepcopy(model)
    if best_model is None or best_seed is None:
        raise RuntimeError("iteration did not produce a best factored actor")
    baseline = baseline_cost(False)
    costs = np.asarray([float(summary["total_cost"]) for summary in eval_rows.values()], dtype=np.float64)
    interventions = np.asarray(
        [float(summary["safety"]["intervention_rate"]) for summary in eval_rows.values()],
        dtype=np.float64,
    )
    max_proj = max(float(summary["safety"]["max_projection_l2"]) for summary in eval_rows.values())
    safe_count = sum(
        (
            float(summary.get("total_batch_expired", 0.0)) == 0.0
            and float(summary.get("terminal_batch_pool", 0.0)) <= 1e-8
            and float(summary["terminal_backlog"]) <= 1e-8
            and float(summary["batch_completion_fraction"]) >= 1.0 - 1e-9
            and float(summary["service_served_total"]) / float(summary["service_demand_total"]) >= 1.0 - 1e-9
        )
        for summary in eval_rows.values()
    )
    iteration_summary = {
        "iteration": int(iteration_index),
        "dataset_rows": int(dataset.observations.shape[0]),
        "dataset_sources": dataset.source.split("+"),
        "mean_sample_weight": float(np.mean(dataset.sample_weights)),
        "intervention_state_rate": float(np.mean(dataset.intervention_flags)),
        "capacity_binding_rate": float(np.mean(dataset.capacity_binding_flags)),
        "ranking_error_rate": float(np.mean(dataset.ranking_error_flags)),
        "seed_results": eval_rows,
        "best_seed": int(best_seed),
        "best_total_cost": float(best_cost),
        "mean_savings_pct": float(100.0 * (baseline - float(costs.mean())) / baseline),
        "safe_runs": f"{safe_count}/{len(eval_rows)}",
        "mean_intervention_rate": float(interventions.mean()),
        "max_projection_l2": float(max_proj),
    }
    return iteration_summary, best_model


def persist_failed_v1(protocol: dict[str, Any]) -> None:
    payload = {
        "protocol": protocol["failed_v1_reference"],
        "decision": "stop-global-same-model-loss",
    }
    FAILED_RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FAILED_RESULT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_report(results: dict[str, Any]) -> str:
    iterations = results["iterations"]
    final_iteration = iterations[-1]
    lines = [
        "# Teacher-free US iterative DAgger",
        "",
        f"- Protocol: `{results['protocol']['name']}`",
        f"- Cached failed v1 result persisted at `{FAILED_RESULT_PATH.relative_to(ROOT).as_posix()}`",
        f"- DAgger iterations executed: **{results['protocol']['dagger']['iterations']}**",
        "",
        "## Final US a-d gate",
        "",
        f"- Mean savings: **{final_iteration['mean_savings_pct']:.4f}%**",
        f"- Safe runs: **{final_iteration['safe_runs']}**",
        f"- Mean intervention: **{final_iteration['mean_intervention_rate']:.6f}**",
        f"- Max semantic delta: **{final_iteration['max_projection_l2']:.3e}**",
        "",
        "| Iteration | Rows | Mean savings | Mean intervention | Intervention-state rate | Ranking-error rate | Best seed |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in iterations:
        lines.append(
            f"| {row['iteration']} | {row['dataset_rows']} | {row['mean_savings_pct']:.4f}% | "
            f"{row['mean_intervention_rate']:.6f} | {row['intervention_state_rate']:.6f} | "
            f"{row['ranking_error_rate']:.6f} | {row['best_seed']} |"
        )
    lines.extend(
        [
            "",
            f"- PPO attempted: **{results['ppo_attempted']}**",
            f"- Global preserved without retraining: **{results['global_preserved']}**",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run US-only v2 teacher-free BC")
    args = parser.parse_args(argv)
    del args
    protocol = read_yaml(PROTOCOL_PATH)
    v4_protocol = read_yaml(V4_PROTOCOL_PATH)
    persist_failed_v1(protocol)

    checkpoint_path = ROOT / str(protocol["dataset"]["dagger_policy_checkpoint"])
    bc_result, hidden_sizes, _seed = load_bc_result(checkpoint_path)
    failed_actor = load_actor_from_result(
        bc_result,
        81,
        12,
        hidden_sizes,
    )

    teacher_dataset = collect_factored_rollout(
        protocol,
        v4_protocol,
        source="teacher-dr",
        rollout_seed=int(protocol["dataset"]["teacher_rollout_seed"]),
        controller_actor=None,
        domain_randomization=bool(protocol["dataset"]["teacher_domain_randomization"]),
    )
    dagger_dataset = collect_factored_rollout(
        protocol,
        v4_protocol,
        source="dagger-failed-bc",
        rollout_seed=int(protocol["dataset"]["dagger_rollout_seed"]),
        controller_actor=failed_actor,
        domain_randomization=False,
    )
    datasets: list[FactoredDataset] = [teacher_dataset, dagger_dataset]
    iteration_rows: list[dict[str, Any]] = []
    current_policy = None
    dagger_iterations = int(protocol["dagger"]["iterations"])
    for iteration_index in range(dagger_iterations + 1):
        if iteration_index > 0:
            rollout_dataset = collect_factored_rollout(
                protocol,
                v4_protocol,
                source=f"dagger-iter-{iteration_index:02d}",
                rollout_seed=int(protocol["dagger"]["rollout_seed"]),
                controller_actor=current_policy,
                domain_randomization=bool(protocol["dagger"]["policy_domain_randomization"]),
            )
            datasets.append(rollout_dataset)
        dataset = concat_datasets(datasets)
        dataset_path = OUT_ROOT / "datasets" / f"us_factored_iter_{iteration_index:02d}.npz"
        save_dataset(dataset_path, dataset)
        iteration_summary, current_policy = train_iteration(
            dataset,
            protocol,
            v4_protocol,
            iteration_index,
        )
        iteration_summary["dataset_path"] = str(dataset_path.relative_to(ROOT)).replace("\\", "/")
        iteration_rows.append(iteration_summary)
    final_iteration = iteration_rows[-1]
    ppo_attempted = False
    results = {
        "protocol": {
            **protocol,
            "protocol_sha256": sha256(PROTOCOL_PATH),
            "v4_protocol_sha256": sha256(V4_PROTOCOL_PATH),
        },
        "iterations": iteration_rows,
        "final_gate": {
            "mean_savings_pct": float(final_iteration["mean_savings_pct"]),
            "safe_runs": str(final_iteration["safe_runs"]),
            "mean_intervention_rate": float(final_iteration["mean_intervention_rate"]),
            "max_projection_l2": float(final_iteration["max_projection_l2"]),
        },
        "ppo_attempted": ppo_attempted,
        "global_preserved": True,
    }
    save_json = OUT_ROOT / "results.json"
    save_json.parent.mkdir(parents=True, exist_ok=True)
    save_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    report = build_report(results)
    (OUT_ROOT / "results_report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
