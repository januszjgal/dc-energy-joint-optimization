"""Teacher-free native-action helpers built on top of the frozen v4 safety stack."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from env.reward import RewardConfig
from env.safe_multi_dc_env import SafeMultiDCEnv
from env.safety_layer import (
    SafetyConfig,
    SafetyInfeasibleError,
    SafetyProjectionResult,
    _mandatory_edf_by_origin,
    _minimum_deadline_slack,
    exact_transport,
    project_box_sum_range,
    project_capped_simplex,
)
from env.workload_generator import BatchEntry, BatchPool
from evaluate import compute_summary
from train_v4 import DEFAULT_DEADLINE_BUCKET_EDGES, make_safe_env

DT_HOURS = 5.0 / 60.0
SEMANTIC_TOLERANCE = 1e-8
PPO_NATIVE_ACTION_BOUND = 1.0


@dataclass(frozen=True)
class ExactTeacherDiagnostics:
    total_batch_target: float
    total_load_target: float
    marginal_cost: float
    mandatory_total: float
    maximum_total: float
    native_roundtrip_l2: float


@dataclass(frozen=True)
class DatasetBundle:
    observations: np.ndarray
    actions: np.ndarray
    teacher_projection_l2: np.ndarray
    teacher_total_batch: np.ndarray
    seeds: tuple[int, ...]
    max_steps_per_seed: tuple[int, ...]


@dataclass(frozen=True)
class BCTrainResult:
    model_state: dict[str, torch.Tensor]
    obs_mean: np.ndarray
    obs_std: np.ndarray
    losses: list[float]
    best_epoch: int
    final_loss: float


def clone_batch_pool(pool: BatchPool) -> BatchPool:
    cloned = BatchPool()
    cloned.entries.extend(pool.entries)
    return cloned


def clone_pools(pools: Sequence[BatchPool]) -> list[BatchPool]:
    return [clone_batch_pool(pool) for pool in pools]


def normalize_simplex(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 0.0, None)
    total = float(clipped.sum())
    if total <= 0.0:
        if clipped.size == 0:
            return clipped
        return np.full(clipped.shape, 1.0 / clipped.size, dtype=np.float64)
    return clipped / total


def drain_exact_edf(
    pools: Sequence[BatchPool],
    amount: float,
) -> tuple[np.ndarray, list[tuple[int, int, float]]]:
    remaining = [
        [int(entry.deadline_step), origin, float(entry.cpu_demand)]
        for origin, pool in enumerate(pools)
        for entry in pool.entries
        if float(entry.cpu_demand) > 0.0
    ]
    remaining.sort(key=lambda item: (item[0], item[1]))
    drained = np.zeros(len(pools), dtype=np.float64)
    left = float(amount)
    for entry in remaining:
        if left <= 1e-12:
            break
        take = min(float(entry[2]), left)
        if take > 0.0:
            drained[int(entry[1])] += take
            entry[2] -= take
            left -= take
    if left > 1e-8:
        raise RuntimeError("exact EDF drain exceeded available pool demand")
    return drained, [
        (int(deadline), int(origin), float(cpu))
        for deadline, origin, cpu in remaining
        if cpu > 1e-12
    ]


def urgency_segments(
    remaining_entries: Sequence[tuple[int, int, float]],
    current_step: int,
    urgency_weight: float,
) -> list[tuple[float, float]]:
    segments: list[tuple[float, float]] = []
    if urgency_weight <= 0.0:
        return segments
    for deadline, _origin, amount in remaining_entries:
        remaining = max(int(deadline) - int(current_step), 1)
        benefit = float(urgency_weight) / float(remaining)
        segments.append((benefit, float(amount)))
    segments.sort(key=lambda item: (-item[0], item[1]))
    return segments


def total_alloc_for_mu(
    mu: float,
    linear: np.ndarray,
    quad: np.ndarray,
    capacities: np.ndarray,
) -> float:
    total = 0.0
    for a_i, b_i, cap_i in zip(linear, quad, capacities):
        if b_i <= 0.0:
            if mu > a_i + 1e-12:
                total += float(cap_i)
            continue
        total += float(np.clip((mu - a_i) / (2.0 * b_i), 0.0, cap_i))
    return total


def economic_dispatch(
    linear: np.ndarray,
    quad: np.ndarray,
    capacities: np.ndarray,
    total_load: float,
) -> tuple[np.ndarray, float, float]:
    linear = np.asarray(linear, dtype=np.float64)
    quad = np.asarray(quad, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    total_load = float(total_load)
    capacity_total = float(capacities.sum())
    if total_load < -1e-12 or total_load > capacity_total + 1e-12:
        raise ValueError(
            f"requested load {total_load} outside feasible range [0, {capacity_total}]"
        )
    if total_load <= 1e-12:
        return (
            np.zeros_like(capacities),
            0.0,
            float(np.min(linear)) if linear.size else 0.0,
        )
    if capacity_total - total_load <= 1e-12:
        alloc = capacities.copy()
        cost = float(np.dot(linear, alloc) + np.dot(quad, alloc * alloc))
        mu = float(np.max(linear + 2.0 * quad * alloc))
        return alloc, cost, mu
    if np.all(quad <= 1e-12):
        alloc = np.zeros_like(capacities)
        remaining = total_load
        for index in np.argsort(linear):
            take = min(float(capacities[index]), remaining)
            alloc[index] = take
            remaining -= take
            if remaining <= 1e-12:
                break
        if remaining > 1e-8:
            raise RuntimeError("linear dispatch failed to place the requested load")
        cost = float(np.dot(linear, alloc))
        active = alloc > 1e-12
        mu = float(np.max(linear[active])) if np.any(active) else float(np.min(linear))
        return alloc, cost, mu

    low = float(np.min(linear) - 1.0)
    high = float(np.max(linear + 2.0 * quad * capacities) + 1.0)
    for _ in range(120):
        midpoint = 0.5 * (low + high)
        if total_alloc_for_mu(midpoint, linear, quad, capacities) >= total_load:
            high = midpoint
        else:
            low = midpoint
    mu = high
    alloc = np.zeros_like(capacities)
    residual = total_load
    for index, (a_i, b_i, cap_i) in enumerate(zip(linear, quad, capacities)):
        if b_i <= 0.0:
            if mu > a_i + 1e-10:
                alloc[index] = float(cap_i)
                residual -= alloc[index]
            continue
        value = float(np.clip((mu - a_i) / (2.0 * b_i), 0.0, cap_i))
        alloc[index] = value
        residual -= value
    if residual > 1e-8:
        marginals = linear + 2.0 * quad * alloc
        room = capacities - alloc
        candidates = np.argsort(marginals)
        for index in candidates:
            if room[index] <= 1e-12:
                continue
            take = min(float(room[index]), residual)
            alloc[index] += take
            residual -= take
            if residual <= 1e-8:
                break
    elif residual < -1e-8:
        marginals = linear + 2.0 * quad * alloc
        candidates = np.argsort(-marginals)
        for index in candidates:
            if alloc[index] <= 1e-12:
                continue
            take = min(float(alloc[index]), -residual)
            alloc[index] -= take
            residual += take
            if residual >= -1e-8:
                break
    if abs(residual) > 1e-6:
        raise RuntimeError(f"dispatch residual too large: {residual}")
    cost = float(np.dot(linear, alloc) + np.dot(quad, alloc * alloc))
    return alloc, cost, float(mu)


def economic_coefficients(env: SafeMultiDCEnv, step_index: int) -> tuple[np.ndarray, np.ndarray]:
    linear: list[float] = []
    quad: list[float] = []
    for site in env.sites:
        power_model = site.power_model or env.power_model
        rated_power = site.rated_power_mw
        a_i = float(site.get_price(step_index)) * 1000.0 * DT_HOURS * power_model.slope * rated_power
        d_positive = max(float(site.get_net_demand(step_index)), 0.0)
        b_i = (
            env.peak_penalty_weight
            * d_positive
            * (power_model.slope * rated_power) ** 2
        )
        a_i += (
            2.0
            * env.peak_penalty_weight
            * d_positive
            * power_model.idle_power
            * rated_power
            * power_model.slope
            * rated_power
        )
        linear.append(a_i)
        quad.append(b_i)
    return np.asarray(linear, dtype=np.float64), np.asarray(quad, dtype=np.float64)


def choose_total_batch_target(
    env: SafeMultiDCEnv,
    pools: Sequence[BatchPool],
    effective_capacity: np.ndarray,
    total_service: float,
    mandatory_total: float,
    current_step: int,
) -> tuple[float, float]:
    urgency_weight = (
        float(env.reward_config.urgency_potential_weight)
        if env.reward_config is not None
        else 0.0
    )
    maximum_total = min(
        float(sum(pool.total_demand for pool in pools)),
        max(float(effective_capacity.sum()) - float(total_service), 0.0),
    )
    if mandatory_total > maximum_total + 1e-8:
        raise SafetyInfeasibleError(
            {
                "reason": "teacher_mandatory_batch_capacity_deficit",
                "step": current_step,
                "mandatory_batch": mandatory_total,
                "available_batch_capacity": maximum_total,
            }
        )
    if maximum_total - mandatory_total <= 1e-12:
        _, _, mu = economic_dispatch(
            *economic_coefficients(env, current_step),
            effective_capacity,
            total_service + mandatory_total,
        )
        return float(mandatory_total), float(mu)

    mandatory_drain, remaining_entries = drain_exact_edf(pools, mandatory_total)
    del mandatory_drain
    segments = urgency_segments(remaining_entries, current_step, urgency_weight)
    linear, quad = economic_coefficients(env, current_step)
    batch_weight = float(env.reward_batch_completion_weight)
    current = float(mandatory_total)
    segment_index = 0

    while current < maximum_total - 1e-12:
        benefit = segments[segment_index][0] if segment_index < len(segments) else 0.0
        segment_room = (
            segments[segment_index][1] if segment_index < len(segments) else maximum_total - current
        )
        upper = min(maximum_total, current + segment_room)
        target_mu = batch_weight + benefit
        low_load = total_service + current
        high_load = total_service + upper
        _, _, mu_low = economic_dispatch(linear, quad, effective_capacity, low_load)
        if mu_low >= target_mu - 1e-10:
            return current, mu_low
        _, _, mu_high = economic_dispatch(linear, quad, effective_capacity, high_load)
        if mu_high <= target_mu + 1e-10:
            current = upper
            if segment_index < len(segments):
                segment_index += 1
            continue
        left = current
        right = upper
        for _ in range(80):
            midpoint = 0.5 * (left + right)
            _, _, mu_mid = economic_dispatch(
                linear,
                quad,
                effective_capacity,
                total_service + midpoint,
            )
            if mu_mid >= target_mu:
                right = midpoint
            else:
                left = midpoint
        _, _, mu_star = economic_dispatch(
            linear,
            quad,
            effective_capacity,
            total_service + right,
        )
        return float(right), float(mu_star)
    _, _, mu = economic_dispatch(
        linear,
        quad,
        effective_capacity,
        total_service + maximum_total,
    )
    return float(maximum_total), float(mu)


def project_native_action(
    desired_service: np.ndarray,
    desired_origin: np.ndarray,
    desired_destination: np.ndarray,
    pools: Sequence[BatchPool],
    *,
    current_step: int,
    max_steps: int,
    total_service: float,
    effective_capacity: np.ndarray,
    net_demand: np.ndarray,
    price: np.ndarray,
    config: SafetyConfig,
) -> SafetyProjectionResult:
    n_dc = len(pools)
    config.validate(n_dc)
    desired_service = np.asarray(desired_service, dtype=np.float64)
    desired_origin = np.asarray(desired_origin, dtype=np.float64)
    desired_destination = np.asarray(desired_destination, dtype=np.float64)
    capacity = np.asarray(effective_capacity, dtype=np.float64)
    tolerance = config.tolerance
    if desired_service.shape != (n_dc,):
        raise ValueError(f"expected desired_service shape {(n_dc,)}, got {desired_service.shape}")
    if desired_origin.shape != (n_dc,):
        raise ValueError(f"expected desired_origin shape {(n_dc,)}, got {desired_origin.shape}")
    if desired_destination.shape != (n_dc,):
        raise ValueError(
            f"expected desired_destination shape {(n_dc,)}, got {desired_destination.shape}"
        )
    if np.any(desired_service < -tolerance):
        raise ValueError("desired_service must be non-negative")
    if np.any(desired_origin < -tolerance):
        raise ValueError("desired_origin must be non-negative")
    if np.any(desired_destination < -tolerance):
        raise ValueError("desired_destination must be non-negative")
    fleet_capacity = float(capacity.sum())
    if total_service > config.service_envelope_total + tolerance:
        raise SafetyInfeasibleError(
            {
                "reason": "service_envelope_exceeded",
                "step": current_step,
                "observed_service": float(total_service),
                "service_envelope": config.service_envelope_total,
            }
        )
    if total_service > fleet_capacity + tolerance:
        raise SafetyInfeasibleError(
            {
                "reason": "service_capacity_deficit",
                "step": current_step,
                "required_service": float(total_service),
                "available_capacity": fleet_capacity,
                "deficit": float(total_service - fleet_capacity),
            }
        )
    service = project_capped_simplex(
        desired_service,
        total_service,
        capacity,
        tolerance=tolerance,
    )
    residual = capacity - service
    pool_totals = np.asarray([pool.total_demand for pool in pools], dtype=np.float64)
    guaranteed_capacity = max(0.0, config.guaranteed_carried_batch_capacity)
    mandatory, mandatory_total, binding = _mandatory_edf_by_origin(
        pools,
        desired_origin,
        current_step,
        max_steps,
        guaranteed_capacity,
        tolerance=tolerance,
    )
    batch_capacity = float(residual.sum())
    if mandatory_total > batch_capacity + tolerance:
        raise SafetyInfeasibleError(
            {
                "reason": "mandatory_batch_capacity_deficit",
                "step": current_step,
                "mandatory_batch": mandatory_total,
                "available_batch_capacity": batch_capacity,
                "deficit": mandatory_total - batch_capacity,
                "binding_deadline_step": binding,
            }
        )
    flush_mask = np.asarray(net_demand, dtype=np.float64) < 0.0
    if config.negative_demand_flush_price_ceiling is not None:
        flush_mask &= np.asarray(price, dtype=np.float64) <= config.negative_demand_flush_price_ceiling
    flush_active = bool(config.negative_demand_flush and np.any(flush_mask))
    minimum_total = mandatory_total
    desired_total = float(np.clip(desired_origin, 0.0, pool_totals).sum())
    if flush_active:
        minimum_total = max(
            minimum_total,
            min(float(pool_totals.sum()), float(residual[flush_mask].sum())),
        )
    maximum_total = min(float(pool_totals.sum()), batch_capacity)
    origin_batch = project_box_sum_range(
        desired_origin,
        mandatory,
        pool_totals,
        minimum_total,
        maximum_total,
        tolerance=tolerance,
    )
    total_batch = float(origin_batch.sum())
    if flush_active and total_batch > tolerance:
        negative_target = min(total_batch, float(residual[flush_mask].sum()))
        negative_upper = residual * flush_mask.astype(np.float64)
        negative_desired = desired_destination * flush_mask.astype(np.float64)
        negative_placement = project_capped_simplex(
            negative_desired,
            negative_target,
            negative_upper,
            tolerance=tolerance,
        )
        remaining_target = total_batch - negative_target
        destination_batch = negative_placement
        if remaining_target > tolerance:
            remaining_upper = residual - negative_placement
            destination_batch = destination_batch + project_capped_simplex(
                desired_destination - negative_placement,
                remaining_target,
                remaining_upper,
                tolerance=tolerance,
            )
    else:
        destination_batch = project_capped_simplex(
            desired_destination,
            total_batch,
            residual,
            tolerance=tolerance,
        )
    transport = exact_transport(origin_batch, destination_batch, tolerance=tolerance)
    service_fractions = np.divide(
        service,
        total_service,
        out=np.zeros_like(service),
        where=total_service > tolerance,
    )
    drain_rates = np.divide(
        origin_batch,
        pool_totals,
        out=np.zeros_like(origin_batch),
        where=pool_totals > tolerance,
    )
    batch_fractions = np.divide(
        destination_batch,
        total_batch,
        out=np.zeros_like(destination_batch),
        where=total_batch > tolerance,
    )
    projection_l2 = float(
        math.sqrt(
            float(np.square(service - desired_service).sum())
            + float(np.square(origin_batch - desired_origin).sum())
            + float(np.square(destination_batch - desired_destination).sum())
        )
    )
    minimum_slack = _minimum_deadline_slack(
        pools,
        origin_batch,
        current_step,
        max_steps,
        guaranteed_capacity,
        tolerance=tolerance,
    )
    if minimum_slack < -1e-7:
        raise RuntimeError("native projection violated cumulative deadline feasibility")
    return SafetyProjectionResult(
        service=service,
        origin_batch=origin_batch,
        destination_batch=destination_batch,
        transport=transport,
        service_fractions=service_fractions,
        drain_rates=drain_rates,
        batch_fractions=batch_fractions,
        mandatory_by_origin=mandatory,
        mandatory_total=mandatory_total,
        desired_service=desired_service,
        desired_origin_batch=desired_origin,
        desired_destination_batch=desired_destination,
        effective_capacity=capacity,
        residual_capacity=residual,
        projection_l2=projection_l2,
        intervened=projection_l2 > SEMANTIC_TOLERANCE,
        binding_deadline_step=binding,
        minimum_deadline_slack=minimum_slack,
        service_capacity_slack=fleet_capacity - total_service,
        batch_capacity_slack=batch_capacity - total_batch,
        negative_flush_active=flush_active,
        exact_zero_drain_count=int(np.count_nonzero(drain_rates == 0.0)),
        exact_full_drain_count=int(np.count_nonzero(drain_rates == 1.0)),
    )


def decode_native_action(
    action: np.ndarray,
    total_service: float,
    pool_totals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    action = np.asarray(action, dtype=np.float64)
    n_dc = int(pool_totals.shape[0])
    if action.shape != (3 * n_dc,):
        raise ValueError(f"expected semantic action shape {(3 * n_dc,)}, got {action.shape}")
    service_fraction = normalize_simplex(action[:n_dc])
    drain_rates = np.clip(action[n_dc : 2 * n_dc], 0.0, 1.0)
    desired_service = service_fraction * float(total_service)
    desired_origin = drain_rates * np.asarray(pool_totals, dtype=np.float64)
    total_desired_origin = float(desired_origin.sum())
    batch_fraction = normalize_simplex(action[2 * n_dc :])
    desired_destination = batch_fraction * total_desired_origin
    return desired_service, desired_origin, desired_destination


def exact_teacher_action(env: SafeMultiDCEnv) -> tuple[np.ndarray, ExactTeacherDiagnostics]:
    step_index = int(env.step_index)
    copied_pools = clone_pools([site.batch_pool for site in env.sites])
    batch_arrivals = np.zeros(env.n_dc, dtype=np.float64)
    for index, site in enumerate(env.sites):
        arrival = float(site.get_batch_demand(step_index))
        batch_arrivals[index] = arrival
        if arrival > 0.0:
            copied_pools[index].add(arrival, step_index + env._deadline_offsets[index])
    for pool in copied_pools:
        expired = float(pool.expire(step_index))
        if expired > env.safety_config.tolerance:
            raise SafetyInfeasibleError(
                {
                    "reason": "teacher_pre_action_deadline_miss",
                    "step": step_index,
                    "expired_total": expired,
                }
            )
    total_arrival = float(batch_arrivals.sum())
    if total_arrival > env.safety_config.batch_arrival_envelope_total + env.safety_config.tolerance:
        raise SafetyInfeasibleError(
            {
                "reason": "batch_arrival_envelope_exceeded",
                "step": step_index,
                "observed_batch_arrival": total_arrival,
                "batch_arrival_envelope": env.safety_config.batch_arrival_envelope_total,
            }
        )
    service = np.asarray(
        [site.get_service_demand(step_index) for site in env.sites],
        dtype=np.float64,
    )
    total_service = float(service.sum())
    effective_capacity = env._effective_capacities()
    mandatory, mandatory_total, _binding = _mandatory_edf_by_origin(
        copied_pools,
        np.zeros(env.n_dc, dtype=np.float64),
        step_index,
        env.max_steps,
        env.safety_config.guaranteed_carried_batch_capacity,
        tolerance=env.safety_config.tolerance,
    )
    del mandatory
    total_batch_target, marginal_cost = choose_total_batch_target(
        env,
        copied_pools,
        effective_capacity,
        total_service,
        mandatory_total,
        step_index,
    )
    drain_by_origin, _remaining = drain_exact_edf(copied_pools, total_batch_target)
    linear, quad = economic_coefficients(env, step_index)
    total_load_target = total_service + total_batch_target
    total_load_alloc, _dispatch_cost, _ = economic_dispatch(
        linear,
        quad,
        effective_capacity,
        total_load_target,
    )
    if total_load_target > 1e-12:
        load_share = total_load_alloc / total_load_target
    else:
        load_share = np.full(env.n_dc, 1.0 / env.n_dc, dtype=np.float64)
    service_fractions = load_share.copy()
    pool_totals = np.asarray([pool.total_demand for pool in copied_pools], dtype=np.float64)
    drain_rates = np.divide(
        drain_by_origin,
        pool_totals,
        out=np.zeros_like(drain_by_origin),
        where=pool_totals > 1e-12,
    )
    batch_fractions = load_share.copy()
    action = np.concatenate(
        [
            np.clip(service_fractions, 0.0, 1.0),
            np.clip(drain_rates, 0.0, 1.0),
            np.clip(batch_fractions, 0.0, 1.0),
        ]
    ).astype(np.float32)
    desired_service, desired_origin, desired_destination = decode_native_action(
        action,
        total_service,
        pool_totals,
    )
    projection = project_native_action(
        desired_service,
        desired_origin,
        desired_destination,
        copied_pools,
        current_step=step_index,
        max_steps=env.max_steps,
        total_service=total_service,
        effective_capacity=effective_capacity,
        net_demand=np.asarray(
            [site.get_net_demand(step_index) for site in env.sites],
            dtype=np.float64,
        ),
        price=np.asarray(
            [site.get_price(step_index) for site in env.sites],
            dtype=np.float64,
        ),
        config=env.safety_config,
    )
    diagnostics = ExactTeacherDiagnostics(
        total_batch_target=float(total_batch_target),
        total_load_target=float(total_load_target),
        marginal_cost=float(marginal_cost),
        mandatory_total=float(mandatory_total),
        maximum_total=float(
            min(
                float(sum(pool.total_demand for pool in copied_pools)),
                max(float(effective_capacity.sum()) - float(total_service), 0.0),
            )
        ),
        native_roundtrip_l2=float(projection.projection_l2),
    )
    return action, diagnostics


def run_native_step(
    env: SafeMultiDCEnv,
    action: np.ndarray,
) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
    action_array = np.asarray(action, dtype=np.float64)
    if action_array.shape != (3 * env.n_dc,):
        raise ValueError(f"expected semantic action shape {(3 * env.n_dc,)}, got {action_array.shape}")
    if not np.isfinite(action_array).all():
        raise ValueError("native action must contain only finite values")
    t = int(env.step_index)
    tolerance = env.safety_config.tolerance
    preexisting_backlog = np.asarray([site.backlog for site in env.sites], dtype=np.float64)
    if np.any(preexisting_backlog > tolerance):
        raise SafetyInfeasibleError(
            {
                "reason": "preexisting_local_service_backlog",
                "step": t,
                "per_site_backlog": preexisting_backlog.tolist(),
            }
        )
    potential_before = env._batch_urgency_potential(t)
    batch_arrivals = np.zeros(env.n_dc, dtype=np.float64)
    for index, site in enumerate(env.sites):
        arrival = float(site.get_batch_demand(t))
        batch_arrivals[index] = arrival
        if arrival > 0.0:
            site.batch_pool.add(arrival, t + env._deadline_offsets[index])
        if env.burst_aware:
            site.record_arrival(t)
    expired_per_dc = np.asarray([site.batch_pool.expire(t) for site in env.sites], dtype=np.float64)
    if float(expired_per_dc.sum()) > tolerance:
        raise SafetyInfeasibleError(
            {
                "reason": "pre_action_deadline_miss",
                "step": t,
                "expired_per_site": expired_per_dc.tolist(),
                "expired_total": float(expired_per_dc.sum()),
            }
        )
    current_service = np.asarray(
        [site.get_service_demand(t) for site in env.sites],
        dtype=np.float64,
    )
    total_service = float(current_service.sum())
    effective_capacity = env._effective_capacities()
    pool_totals = np.asarray([site.batch_pool.total_demand for site in env.sites], dtype=np.float64)
    desired_service, desired_origin, desired_destination = decode_native_action(
        action_array,
        total_service,
        pool_totals,
    )
    projection = project_native_action(
        desired_service,
        desired_origin,
        desired_destination,
        [site.batch_pool for site in env.sites],
        current_step=t,
        max_steps=env.max_steps,
        total_service=total_service,
        effective_capacity=effective_capacity,
        net_demand=np.asarray(
            [site.get_net_demand(t) for site in env.sites],
            dtype=np.float64,
        ),
        price=np.asarray(
            [site.get_price(t) for site in env.sites],
            dtype=np.float64,
        ),
        config=env.safety_config,
    )
    total_cost = 0.0
    total_energy = 0.0
    total_peak = 0.0
    total_demand_charge = 0.0
    total_grid_mw = 0.0
    info_per_dc: list[dict[str, Any]] = []
    for index, site in enumerate(env.sites):
        power_model = site.power_model or env.power_model
        previous_grid_mw = power_model.compute(site.current_load) * site.rated_power_mw
        service_served = float(projection.service[index])
        batch_served = float(projection.destination_batch[index])
        served = service_served + batch_served
        if served > effective_capacity[index] + tolerance:
            raise RuntimeError("projected site capacity was exceeded")
        (
            dc_cost,
            energy,
            peak,
            grid_mw,
            net_demand,
            backlog_cost,
            capacity_cost,
            demand_charge,
        ) = env._compute_dc_cost(site, served, 0.0, t, index)
        if (
            env.safety_config.max_grid_mw is not None
            and grid_mw > env.safety_config.max_grid_mw[index] + tolerance
        ):
            raise SafetyInfeasibleError(
                {
                    "reason": "post_projection_grid_cap_violation",
                    "step": t,
                    "site": site.name,
                    "grid_mw": float(grid_mw),
                    "max_grid_mw": env.safety_config.max_grid_mw[index],
                }
            )
        if (
            env.safety_config.max_upward_ramp_mw is not None
            and grid_mw - previous_grid_mw > env.safety_config.max_upward_ramp_mw[index] + tolerance
        ):
            raise SafetyInfeasibleError(
                {
                    "reason": "post_projection_ramp_cap_violation",
                    "step": t,
                    "site": site.name,
                    "upward_ramp_mw": float(grid_mw - previous_grid_mw),
                    "max_upward_ramp_mw": env.safety_config.max_upward_ramp_mw[index],
                }
            )
        site.backlog = 0.0
        site.current_load = served
        if env.memory_enabled:
            site.current_memory_load = served * site.memory_cpu_ratio
        total_cost += dc_cost
        total_energy += energy
        total_peak += peak
        total_demand_charge += demand_charge
        total_grid_mw += grid_mw
        info_per_dc.append(
            {
                "name": site.name,
                "service_assigned": service_served,
                "service_served": service_served,
                "batch_arrival": float(batch_arrivals[index]),
                "batch_assigned": batch_served,
                "batch_served": batch_served,
                "served": float(served),
                "backlog": 0.0,
                "batch_expired": 0.0,
                "drain_rate": float(projection.drain_rates[index]),
                "grid_mw": float(grid_mw),
                "net_demand": float(net_demand),
                "net_demand_mw": site.get_net_demand_mw(t),
                "energy_cost": float(energy),
                "peak_penalty": float(peak),
                "demand_charge": float(demand_charge),
                "billed_peak_mw": float(env._billed_peak_mw[index]),
                "backlog_cost": float(backlog_cost),
                "capacity_cost": float(capacity_cost),
                "deadline_cost": 0.0,
                "mandatory_batch_origin": float(projection.mandatory_by_origin[index]),
                "batch_transport_from_origin": projection.transport[index].tolist(),
            }
        )
    total_batch_served = float(projection.origin_batch.sum())
    for index, site in enumerate(env.sites):
        drained = site.batch_pool.drain_amount(float(projection.origin_batch[index]))
        if not math.isclose(drained, float(projection.origin_batch[index]), rel_tol=0.0, abs_tol=1e-8):
            raise RuntimeError("exact origin drain lost work")
        info_per_dc[index]["batch_drained"] = float(drained)
        info_per_dc[index]["batch_pool_size"] = float(site.batch_pool.total_demand)
    if not math.isclose(
        float(projection.destination_batch.sum()),
        total_batch_served,
        rel_tol=0.0,
        abs_tol=1e-8,
    ):
        raise RuntimeError("origin drain and destination execution differ")
    batch_balance = float(batch_arrivals.sum()) - total_batch_served
    total_batch_accounting_cost = 0.0
    if env.batch_completion_shaping_enabled:
        total_batch_accounting_cost = env.deadline_penalty_weight * batch_balance
        total_cost += total_batch_accounting_cost
    billing_info = env._billing_info(t)
    env.step_index += 1
    terminated = env.step_index >= env.max_steps
    truncated = False
    potential_after = 0.0 if terminated else env._batch_urgency_potential(env.step_index)
    penalty_adjustment = (env.reward_batch_completion_weight - env.batch_completion_weight) * batch_balance
    reward, reward_training_cost, reward_idle_cost, reward_potential_delta = env._training_reward(
        total_cost,
        t,
        potential_before,
        potential_after,
        penalty_adjustment,
    )
    if not terminated:
        env._start_billing_period(env.step_index)
    info = {
        "total_demand": total_service,
        "total_batch_expired": 0.0,
        "total_batch_pool": float(sum(site.batch_pool.total_demand for site in env.sites)),
        "total_batch_accounting_cost": float(total_batch_accounting_cost),
        "fractions": projection.service_fractions.tolist(),
        "drain_rates": projection.drain_rates.tolist(),
        "batch_fractions": projection.batch_fractions.tolist(),
        "batch_transport": projection.transport.tolist(),
        "total_cost": float(total_cost),
        "total_energy_cost": float(total_energy),
        "total_peak_penalty": float(total_peak),
        "total_demand_charge": float(total_demand_charge),
        "total_grid_mw": float(total_grid_mw),
        "reward_training_cost": float(reward_training_cost),
        "reward_idle_cost": float(reward_idle_cost),
        "reward_potential_delta": float(reward_potential_delta),
        "reward_penalty_adjustment": float(penalty_adjustment),
        "per_dc": info_per_dc,
        "safety_enabled": True,
        "safety_intervened": bool(projection.intervened),
        "safety_projection_l2": float(projection.projection_l2),
        "safety_mandatory_batch": float(projection.mandatory_total),
        "safety_mandatory_by_origin": projection.mandatory_by_origin.tolist(),
        "safety_binding_deadline_step": projection.binding_deadline_step,
        "safety_binding_deadline_steps_remaining": (
            projection.binding_deadline_step - t if projection.binding_deadline_step is not None else None
        ),
        "safety_minimum_deadline_slack": float(projection.minimum_deadline_slack),
        "safety_service_capacity_slack": float(projection.service_capacity_slack),
        "safety_batch_capacity_slack": float(projection.batch_capacity_slack),
        "safety_negative_flush_active": bool(projection.negative_flush_active),
        "safety_exact_zero_drain_count": int(projection.exact_zero_drain_count),
        "safety_exact_full_drain_count": int(projection.exact_full_drain_count),
        "safety_transport_conservation_error": float(
            max(
                np.max(np.abs(projection.transport.sum(axis=1) - projection.origin_batch), initial=0.0),
                np.max(np.abs(projection.transport.sum(axis=0) - projection.destination_batch), initial=0.0),
            )
        ),
        "safety_service_envelope_total": float(env.safety_config.service_envelope_total),
        "safety_batch_arrival_envelope_total": float(env.safety_config.batch_arrival_envelope_total),
        "safety_guaranteed_carried_batch_capacity": float(env.safety_config.guaranteed_carried_batch_capacity),
        "safety_envelope_id": env.safety_config.envelope_id,
        "safety_envelope_scope": env.safety_config.envelope_scope,
        "native_semantic_desired_service": desired_service.tolist(),
        "native_semantic_desired_origin_batch": desired_origin.tolist(),
        "native_semantic_desired_destination_batch": desired_destination.tolist(),
        **billing_info,
    }
    observation = env._get_obs() if not terminated else np.zeros(env.observation_space.shape, dtype=np.float32)
    return observation, float(reward), terminated, truncated, info


class NativeActionEnv(gym.Env):
    """Gym wrapper that exposes the v4 safe batch controller in native action units."""

    metadata = {"render_modes": []}

    def __init__(self, base_env: SafeMultiDCEnv) -> None:
        super().__init__()
        self.base_env = base_env
        self.observation_space = base_env.observation_space
        self.action_space = spaces.Box(
            low=0.0,
            high=PPO_NATIVE_ACTION_BOUND,
            shape=(3 * base_env.n_dc,),
            dtype=np.float32,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_env, name)

    @property
    def unwrapped(self) -> SafeMultiDCEnv:  # type: ignore[override]
        return self.base_env

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        return self.base_env.reset(seed=seed, options=options)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        return run_native_step(self.base_env, action)

    def close(self) -> None:
        self.base_env.close()


def make_native_env(
    scenario_path: Path,
    *,
    seed: int,
    peak_penalty_weight: float,
    reward_config: RewardConfig,
    safety_config: SafetyConfig,
    domain_randomization: bool,
    deadline_bucket_edges: tuple[int, ...] = DEFAULT_DEADLINE_BUCKET_EDGES,
) -> NativeActionEnv:
    base_env = make_safe_env(
        scenario_path,
        seed=seed,
        peak_penalty_weight=peak_penalty_weight,
        reward_config=reward_config,
        safety_config=safety_config,
        domain_randomization=domain_randomization,
        deadline_bucket_edges=deadline_bucket_edges,
    )
    return NativeActionEnv(base_env)


def rollout_controller(
    env: gym.Env,
    controller: Callable[[gym.Env, np.ndarray], np.ndarray],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    obs, _ = env.reset()
    history: list[dict[str, Any]] = []
    while True:
        action = controller(env, obs)
        obs, _reward, terminated, truncated, info = env.step(action)
        history.append(info)
        if terminated or truncated:
            break
    summary = compute_summary(history, batch_enabled=True)
    return summary, history


def collect_teacher_dataset(
    scenario_path: Path,
    *,
    seeds: Sequence[int],
    peak_penalty_weight: float,
    reward_config: RewardConfig,
    safety_config: SafetyConfig,
    max_steps: int | None = None,
) -> DatasetBundle:
    obs_rows: list[np.ndarray] = []
    act_rows: list[np.ndarray] = []
    projection_rows: list[float] = []
    batch_rows: list[float] = []
    steps_per_seed: list[int] = []
    for seed in seeds:
        env = make_native_env(
            scenario_path,
            seed=int(seed),
            peak_penalty_weight=peak_penalty_weight,
            reward_config=reward_config,
            safety_config=safety_config,
            domain_randomization=True,
        )
        obs, _ = env.reset(seed=int(seed))
        step_count = 0
        while True:
            teacher_action, diagnostics = exact_teacher_action(env.unwrapped)
            obs_rows.append(np.asarray(obs, dtype=np.float32))
            act_rows.append(np.asarray(teacher_action, dtype=np.float32))
            projection_rows.append(float(diagnostics.native_roundtrip_l2))
            batch_rows.append(float(diagnostics.total_batch_target))
            obs, _reward, terminated, truncated, _info = env.step(teacher_action)
            step_count += 1
            if terminated or truncated or (max_steps is not None and step_count >= max_steps):
                break
        steps_per_seed.append(step_count)
        env.close()
    return DatasetBundle(
        observations=np.asarray(obs_rows, dtype=np.float32),
        actions=np.asarray(act_rows, dtype=np.float32),
        teacher_projection_l2=np.asarray(projection_rows, dtype=np.float64),
        teacher_total_batch=np.asarray(batch_rows, dtype=np.float64),
        seeds=tuple(int(seed) for seed in seeds),
        max_steps_per_seed=tuple(int(step_count) for step_count in steps_per_seed),
    )


class ActorMLP(torch.nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: Sequence[int]) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = []
        in_dim = int(obs_dim)
        hidden_list = [int(size) for size in hidden_sizes]
        self.hidden_sizes = tuple(hidden_list)
        self.input_dim = in_dim
        self.output_dim = int(action_dim)
        modules: list[torch.nn.Linear] = []
        for hidden in hidden_list:
            linear = torch.nn.Linear(in_dim, hidden)
            torch.nn.init.xavier_uniform_(linear.weight)
            torch.nn.init.zeros_(linear.bias)
            layers.extend([linear, torch.nn.Tanh()])
            modules.append(linear)
            in_dim = hidden
        final = torch.nn.Linear(in_dim, self.output_dim)
        torch.nn.init.xavier_uniform_(final.weight)
        torch.nn.init.zeros_(final.bias)
        layers.append(final)
        modules.append(final)
        self.layers = torch.nn.Sequential(*layers)
        self.linear_layers = modules

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.layers(obs)


def semantic_action_from_raw_tensor(raw: torch.Tensor) -> torch.Tensor:
    n_dc = raw.shape[-1] // 3
    service = torch.softmax(raw[..., :n_dc], dim=-1)
    drain = torch.sigmoid(raw[..., n_dc : 2 * n_dc])
    batch = torch.softmax(raw[..., 2 * n_dc :], dim=-1)
    return torch.cat([service, drain, batch], dim=-1)


def semantic_action_from_raw_numpy(raw: np.ndarray) -> np.ndarray:
    tensor = torch.as_tensor(raw, dtype=torch.float32)
    semantic = semantic_action_from_raw_tensor(tensor).detach().cpu().numpy()
    return semantic.astype(np.float32)


def fold_observation_normalization(
    model_state: dict[str, torch.Tensor],
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
) -> dict[str, torch.Tensor]:
    adjusted = {key: value.detach().clone() for key, value in model_state.items()}
    first_weight = adjusted["layers.0.weight"]
    first_bias = adjusted["layers.0.bias"]
    mean_tensor = torch.as_tensor(obs_mean, dtype=first_weight.dtype)
    std_tensor = torch.as_tensor(obs_std, dtype=first_weight.dtype)
    adjusted["layers.0.weight"] = first_weight / std_tensor.unsqueeze(0)
    adjusted["layers.0.bias"] = first_bias - torch.mv(first_weight, mean_tensor / std_tensor)
    return adjusted


def train_behavior_cloner(
    dataset: DatasetBundle,
    *,
    seed: int,
    hidden_sizes: Sequence[int],
    learning_rate: float,
    epochs: int,
    batch_size: int,
    weight_decay: float = 1e-6,
) -> BCTrainResult:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    obs = dataset.observations.astype(np.float32)
    actions = dataset.actions.astype(np.float32)
    obs_mean = obs.mean(axis=0, dtype=np.float64)
    obs_std = obs.std(axis=0, dtype=np.float64)
    obs_std = np.where(obs_std < 1e-6, 1.0, obs_std)
    normalized_obs = ((obs - obs_mean) / obs_std).astype(np.float32)
    obs_tensor = torch.from_numpy(normalized_obs)
    action_tensor = torch.from_numpy(actions)
    model = ActorMLP(obs.shape[1], actions.shape[1], hidden_sizes)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    loss_fn = torch.nn.MSELoss()
    rng = np.random.default_rng(int(seed))
    best_loss = math.inf
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    losses: list[float] = []
    for epoch in range(int(epochs)):
        permutation = torch.from_numpy(rng.permutation(len(obs_tensor)))
        epoch_losses: list[float] = []
        for start in range(0, len(obs_tensor), int(batch_size)):
            batch_indices = permutation[start : start + int(batch_size)]
            batch_obs = obs_tensor[batch_indices]
            batch_actions = action_tensor[batch_indices]
            optimizer.zero_grad(set_to_none=True)
            prediction = semantic_action_from_raw_tensor(model(batch_obs))
            loss = loss_fn(prediction, batch_actions)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu().item()))
        epoch_loss = float(np.mean(epoch_losses))
        losses.append(epoch_loss)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("behavior cloning training did not produce a model state")
    exported_state = fold_observation_normalization(best_state, obs_mean, obs_std)
    return BCTrainResult(
        model_state=exported_state,
        obs_mean=obs_mean,
        obs_std=obs_std,
        losses=losses,
        best_epoch=best_epoch,
        final_loss=best_loss,
    )


def load_actor_from_result(result: BCTrainResult, obs_dim: int, action_dim: int, hidden_sizes: Sequence[int]) -> ActorMLP:
    actor = ActorMLP(obs_dim, action_dim, hidden_sizes)
    actor.load_state_dict(result.model_state)
    actor.eval()
    return actor


def predict_actor_action(actor: ActorMLP, obs: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        raw = actor(obs_tensor).squeeze(0)
        action = semantic_action_from_raw_tensor(raw).cpu().numpy()
    return action.astype(np.float32)


def copy_actor_into_ppo(model: PPO, actor: ActorMLP) -> None:
    policy_net = list(model.policy.mlp_extractor.policy_net)
    actor_linears = [module for module in actor.layers if isinstance(module, torch.nn.Linear)]
    target_linears = [module for module in policy_net if isinstance(module, torch.nn.Linear)]
    if len(actor_linears) != len(target_linears) + 1:
        raise ValueError("unexpected actor/ppo architecture mismatch")
    for source, target in zip(actor_linears[:-1], target_linears):
        if source.weight.shape != target.weight.shape:
            raise ValueError("hidden layer shape mismatch during PPO warm start")
        target.weight.data.copy_(source.weight.data)
        target.bias.data.copy_(source.bias.data)
    final_source = actor_linears[-1]
    if final_source.weight.shape != model.policy.action_net.weight.shape:
        raise ValueError("action head shape mismatch during PPO warm start")
    model.policy.action_net.weight.data.copy_(final_source.weight.data)
    model.policy.action_net.bias.data.copy_(final_source.bias.data)
    model.policy.log_std.data.fill_(-2.0)


def save_bc_result(
    path: Path,
    result: BCTrainResult,
    *,
    hidden_sizes: Sequence[int],
    seed: int,
) -> None:
    payload = {
        "seed": int(seed),
        "hidden_sizes": [int(size) for size in hidden_sizes],
        "best_epoch": int(result.best_epoch),
        "final_loss": float(result.final_loss),
        "losses": [float(value) for value in result.losses],
        "obs_mean": result.obs_mean.tolist(),
        "obs_std": result.obs_std.tolist(),
        "state_dict": {
            key: value.detach().cpu().numpy().tolist()
            for key, value in result.model_state.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_bc_result(path: Path) -> tuple[BCTrainResult, tuple[int, ...], int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    state = {
        key: torch.as_tensor(value, dtype=torch.float32)
        for key, value in payload["state_dict"].items()
    }
    result = BCTrainResult(
        model_state=state,
        obs_mean=np.asarray(payload["obs_mean"], dtype=np.float64),
        obs_std=np.asarray(payload["obs_std"], dtype=np.float64),
        losses=[float(value) for value in payload["losses"]],
        best_epoch=int(payload["best_epoch"]),
        final_loss=float(payload["final_loss"]),
    )
    return result, tuple(int(size) for size in payload["hidden_sizes"]), int(payload["seed"])


def evaluate_actor(
    actor: ActorMLP,
    scenario_path: Path,
    *,
    seed: int,
    peak_penalty_weight: float,
    reward_config: RewardConfig,
    safety_config: SafetyConfig,
    domain_randomization: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env = make_native_env(
        scenario_path,
        seed=int(seed),
        peak_penalty_weight=peak_penalty_weight,
        reward_config=reward_config,
        safety_config=safety_config,
        domain_randomization=bool(domain_randomization),
    )
    try:
        return rollout_controller(env, lambda _env, obs: predict_actor_action(actor, obs))
    finally:
        env.close()


def evaluate_teacher(
    scenario_path: Path,
    *,
    seed: int,
    peak_penalty_weight: float,
    reward_config: RewardConfig,
    safety_config: SafetyConfig,
    domain_randomization: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env = make_native_env(
        scenario_path,
        seed=int(seed),
        peak_penalty_weight=peak_penalty_weight,
        reward_config=reward_config,
        safety_config=safety_config,
        domain_randomization=bool(domain_randomization),
    )
    teacher_roundtrip: list[float] = []

    def controller(native_env: gym.Env, _obs: np.ndarray) -> np.ndarray:
        action, diagnostics = exact_teacher_action(native_env.unwrapped)
        teacher_roundtrip.append(float(diagnostics.native_roundtrip_l2))
        return action

    try:
        summary, history = rollout_controller(env, controller)
    finally:
        env.close()
    summary["teacher_native_roundtrip_l2_max"] = float(max(teacher_roundtrip, default=0.0))
    summary["teacher_native_roundtrip_l2_mean"] = float(np.mean(teacher_roundtrip) if teacher_roundtrip else 0.0)
    return summary, history


def monitor_native_env_factory(
    scenario_path: Path,
    *,
    seed: int,
    peak_penalty_weight: float,
    reward_config: RewardConfig,
    safety_config: SafetyConfig,
    domain_randomization: bool,
) -> Callable[[], Monitor]:
    def factory() -> Monitor:
        return Monitor(
            make_native_env(
                scenario_path,
                seed=seed,
                peak_penalty_weight=peak_penalty_weight,
                reward_config=reward_config,
                safety_config=safety_config,
                domain_randomization=domain_randomization,
            )
        )

    return factory


def ppo_policy_kwargs(hidden_sizes: Sequence[int]) -> dict[str, Any]:
    return {
        "net_arch": list(int(size) for size in hidden_sizes),
    }


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
