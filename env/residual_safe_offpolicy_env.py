"""Residual-safe continuous-control environment for off-policy v5 experiments."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
from gymnasium import spaces

from env.multi_dc_env import INTERVAL_HOURS
from env.safe_multi_dc_env import SafeMultiDCEnv
from env.safety_layer import (
    SafetyConfig,
    SafetyInfeasibleError,
    SafetyProjectionResult,
    _mandatory_edf_by_origin,
    _minimum_deadline_slack,
    exact_transport,
    project_capped_simplex,
    softmax,
)
from env.workload_generator import BatchEntry, BatchPool

RESIDUAL_ACTION_LOGIT_BOUND = 6.0
_EPS = 1e-9


@dataclass(frozen=True)
class ResidualDecoderContext:
    """Causal per-step safety state exposed to the residual controller."""

    total_service: float
    batch_arrivals: np.ndarray
    effective_capacity: np.ndarray
    pools_after_arrival: tuple[BatchPool, ...]
    pool_totals: np.ndarray
    guaranteed_future_capacity: float
    fleet_service_slack: float
    conservative_local_residual: np.ndarray
    mandatory_hint_by_origin: np.ndarray
    mandatory_hint_total: float
    binding_deadline_step: int | None
    minimum_deadline_slack: float
    marginal_energy_cost: np.ndarray
    marginal_peak_cost: np.ndarray
    marginal_demand_charge_cost: np.ndarray


def _clone_pool(pool: BatchPool) -> BatchPool:
    return BatchPool(
        deque(
            BatchEntry(float(entry.cpu_demand), int(entry.deadline_step))
            for entry in pool.entries
        )
    )


def _shares_to_logits(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    total = float(values.sum())
    if total <= _EPS:
        return np.zeros(values.shape, dtype=np.float64)
    shares = np.clip(values / total, _EPS, None)
    logits = np.log(shares)
    return logits - float(np.mean(logits))


def _greedy_linear_allocation(
    total: float,
    upper: np.ndarray,
    costs: np.ndarray,
) -> np.ndarray:
    total = float(total)
    upper = np.asarray(upper, dtype=np.float64)
    costs = np.asarray(costs, dtype=np.float64)
    allocation = np.zeros_like(upper)
    if total <= _EPS:
        return allocation
    remaining = min(total, float(upper.sum()))
    order = np.argsort(costs, kind="stable")
    for index in order:
        capacity = float(max(upper[index], 0.0))
        if capacity <= _EPS:
            continue
        take = min(remaining, capacity)
        allocation[index] = take
        remaining -= take
        if remaining <= _EPS:
            break
    return allocation


def _drain_exact_edf(
    pools: tuple[BatchPool, ...],
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
        if left <= _EPS:
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


def _rates_to_logits(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    clipped = np.clip(values, _EPS, 1.0 - _EPS)
    return np.log(clipped / (1.0 - clipped))


def _urgency_segments(
    remaining_entries: list[tuple[int, int, float]],
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


def _total_alloc_for_mu(
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


def _economic_dispatch(
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
        mu = (
            float(np.max(linear[active]))
            if np.any(active)
            else float(np.min(linear))
        )
        return alloc, cost, mu

    low = float(np.min(linear) - 1.0)
    high = float(np.max(linear + 2.0 * quad * capacities) + 1.0)
    for _ in range(120):
        midpoint = 0.5 * (low + high)
        if _total_alloc_for_mu(midpoint, linear, quad, capacities) >= total_load:
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


class ResidualSafeOffPolicyEnv(SafeMultiDCEnv):
    """SafeMultiDCEnv with a native residual action decoder for SAC/TD3."""

    def __init__(
        self,
        *args: Any,
        decoder_logit_bound: float = RESIDUAL_ACTION_LOGIT_BOUND,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.decoder_logit_bound = float(decoder_logit_bound)
        base_obs_dim = int(self.observation_space.shape[0])
        self._per_site_extra_obs_dim = 7
        self._global_extra_obs_dim = 4
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                base_obs_dim
                + self.n_dc * self._per_site_extra_obs_dim
                + self._global_extra_obs_dim,
            ),
            dtype=np.float32,
        )
        # Action layout:
        #   service logits (N)
        #   optional batch utilization scalar (1)
        #   optional batch origin logits (N)
        #   batch destination logits (N)
        self.action_space = spaces.Box(
            low=-self.decoder_logit_bound,
            high=self.decoder_logit_bound,
            shape=(3 * self.n_dc + 1,),
            dtype=np.float32,
        )

    def _post_arrival_pools(
        self,
        t: int,
        batch_arrivals: np.ndarray,
    ) -> tuple[BatchPool, ...]:
        pools: list[BatchPool] = []
        for index, site in enumerate(self.sites):
            pool = _clone_pool(site.batch_pool)
            arrival = float(batch_arrivals[index])
            if arrival > 0.0:
                pool.add(arrival, t + self._deadline_offsets[index])
            pools.append(pool)
        return tuple(pools)

    def _marginal_cost_features(
        self,
        effective_capacity: np.ndarray,
        current_service: np.ndarray,
        t: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        energy = np.zeros(self.n_dc, dtype=np.float64)
        peak = np.zeros(self.n_dc, dtype=np.float64)
        demand = np.zeros(self.n_dc, dtype=np.float64)
        for index, site in enumerate(self.sites):
            power_model = site.power_model or self.power_model
            marginal_mw = power_model.slope * site.rated_power_mw
            energy[index] = (
                site.get_price(t) * marginal_mw * 1000.0 * INTERVAL_HOURS
            )
            current_grid_mw = (
                power_model.compute(min(current_service[index], effective_capacity[index]))
                * site.rated_power_mw
            )
            peak[index] = (
                2.0
                * self.peak_penalty_weight
                * max(site.get_net_demand(t), 0.0)
                * current_grid_mw
                * marginal_mw
            )
            if self.demand_charge_enabled:
                demand[index] = (
                    self.demand_charge_rate * marginal_mw * 1000.0
                    if current_grid_mw + marginal_mw > self._billed_peak_mw[index]
                    else 0.0
                )
        return energy, peak, demand

    def residual_decoder_context(
        self,
        t: int | None = None,
    ) -> ResidualDecoderContext:
        current_step = self.step_index if t is None else int(t)
        current_service = np.array(
            [site.get_service_demand(current_step) for site in self.sites],
            dtype=np.float64,
        )
        batch_arrivals = np.array(
            [site.get_batch_demand(current_step) for site in self.sites],
            dtype=np.float64,
        )
        effective_capacity = self._effective_capacities()
        pools_after_arrival = self._post_arrival_pools(
            current_step,
            batch_arrivals,
        )
        pool_totals = np.array(
            [pool.total_demand for pool in pools_after_arrival],
            dtype=np.float64,
        )
        guaranteed_capacity = max(
            0.0, self.safety_config.guaranteed_carried_batch_capacity
        )
        mandatory_hint, mandatory_total, binding = _mandatory_edf_by_origin(
            pools_after_arrival,
            pool_totals,
            current_step,
            self.max_steps,
            guaranteed_capacity,
            tolerance=self.safety_config.tolerance,
        )
        minimum_slack = _minimum_deadline_slack(
            pools_after_arrival,
            mandatory_hint,
            current_step,
            self.max_steps,
            guaranteed_capacity,
            tolerance=self.safety_config.tolerance,
        )
        conservative_local_residual = np.maximum(
            effective_capacity - current_service,
            0.0,
        )
        marginal_energy, marginal_peak, marginal_demand = (
            self._marginal_cost_features(
                effective_capacity,
                current_service,
                current_step,
            )
        )
        return ResidualDecoderContext(
            total_service=float(current_service.sum()),
            batch_arrivals=batch_arrivals,
            effective_capacity=effective_capacity,
            pools_after_arrival=pools_after_arrival,
            pool_totals=pool_totals,
            guaranteed_future_capacity=guaranteed_capacity,
            fleet_service_slack=float(
                max(float(effective_capacity.sum()) - float(current_service.sum()), 0.0)
            ),
            conservative_local_residual=conservative_local_residual,
            mandatory_hint_by_origin=mandatory_hint,
            mandatory_hint_total=float(mandatory_total),
            binding_deadline_step=binding,
            minimum_deadline_slack=float(minimum_slack),
            marginal_energy_cost=marginal_energy,
            marginal_peak_cost=marginal_peak,
            marginal_demand_charge_cost=marginal_demand,
        )

    def _economic_coefficients(
        self,
        current_step: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        linear = np.zeros(self.n_dc, dtype=np.float64)
        quad = np.zeros(self.n_dc, dtype=np.float64)
        for index, site in enumerate(self.sites):
            power_model = site.power_model or self.power_model
            rated_power = site.rated_power_mw
            linear[index] = (
                float(site.get_price(current_step))
                * 1000.0
                * INTERVAL_HOURS
                * power_model.slope
                * rated_power
            )
            positive_demand = max(float(site.get_net_demand(current_step)), 0.0)
            quad[index] = (
                self.peak_penalty_weight
                * positive_demand
                * (power_model.slope * rated_power) ** 2
            )
            linear[index] += (
                2.0
                * self.peak_penalty_weight
                * positive_demand
                * power_model.idle_power
                * rated_power
                * power_model.slope
                * rated_power
            )
        return linear, quad

    def _optional_total_from_logit(
        self,
        logit: float,
        upper: float,
    ) -> float:
        if upper <= _EPS:
            return 0.0
        if logit >= self.decoder_logit_bound - 1e-9:
            return float(upper)
        if logit <= -self.decoder_logit_bound + 1e-9:
            return 0.0
        return float(upper / (1.0 + math.exp(-logit)))

    def _legacy_fallback_action(self, action: np.ndarray) -> np.ndarray:
        n_dc = self.n_dc
        service_logits = np.asarray(action[:n_dc], dtype=np.float64)
        optional_total = float(action[n_dc])
        origin_logits = np.asarray(action[n_dc + 1 : 2 * n_dc + 1], dtype=np.float64)
        destination_logits = np.asarray(action[2 * n_dc + 1 :], dtype=np.float64)
        drain_logits = origin_logits + optional_total
        return np.concatenate(
            [service_logits, drain_logits, destination_logits],
            dtype=np.float64,
        )

    def _optional_total_logit(
        self,
        total: float,
        upper: float,
    ) -> float:
        if upper <= _EPS or total <= _EPS:
            return -self.decoder_logit_bound
        if upper - total <= _EPS:
            return self.decoder_logit_bound
        fraction = float(np.clip(total / upper, _EPS, 1.0 - _EPS))
        logit = math.log(fraction / (1.0 - fraction))
        return float(
            np.clip(
                logit,
                -self.decoder_logit_bound,
                self.decoder_logit_bound,
            )
        )

    def marginal_cost_teacher_action(self) -> np.ndarray:
        """Return a causal native-decoder action from current marginal costs."""
        t = self.step_index
        context = self.residual_decoder_context(t)
        site_cost = (
            context.marginal_energy_cost
            + context.marginal_peak_cost
            + context.marginal_demand_charge_cost
        )
        service = _greedy_linear_allocation(
            context.total_service,
            context.effective_capacity,
            site_cost,
        )
        residual_capacity = np.maximum(
            context.effective_capacity - service,
            0.0,
        )
        mandatory = np.asarray(
            context.mandatory_hint_by_origin,
            dtype=np.float64,
        )
        optional_upper = np.maximum(context.pool_totals - mandatory, 0.0)
        optional_pool_total = float(optional_upper.sum())
        optional_capacity_total = float(residual_capacity.sum())
        optional_max_total = min(optional_pool_total, optional_capacity_total)
        net_demand = np.asarray(
            [site.get_net_demand(t) for site in self.sites],
            dtype=np.float64,
        )
        if optional_max_total <= _EPS:
            optional_total = 0.0
            target_mask = residual_capacity > _EPS
        else:
            spread = float(site_cost.max() - site_cost.min())
            cheap_threshold = float(site_cost.min() + 0.25 * spread)
            cheap_mask = site_cost <= cheap_threshold + _EPS
            negative_mask = net_demand < 0.0
            target_mask = negative_mask | cheap_mask
            target_capacity = float(
                residual_capacity[target_mask].sum()
            )
            optional_total = min(optional_max_total, target_capacity)
            if (
                optional_total <= _EPS
                and math.isfinite(context.minimum_deadline_slack)
            ):
                urgency_fraction = float(
                    np.clip(
                        (6.0 - context.minimum_deadline_slack) / 6.0,
                        0.0,
                        1.0,
                    )
                )
                optional_total = min(
                    optional_max_total,
                    optional_capacity_total * urgency_fraction,
                )
                if optional_total > _EPS:
                    target_mask = residual_capacity > _EPS
            if not np.any(target_mask):
                target_mask = residual_capacity > _EPS

        optional_origin = np.zeros(self.n_dc, dtype=np.float64)
        if optional_total > _EPS and optional_pool_total > _EPS:
            desired_origin = np.divide(
                optional_upper,
                optional_pool_total,
                out=np.zeros_like(optional_upper),
                where=optional_pool_total > _EPS,
            ) * optional_total
            optional_origin = project_capped_simplex(
                desired_origin,
                optional_total,
                optional_upper,
                tolerance=self.safety_config.tolerance,
            )
        origin_batch = mandatory + optional_origin

        candidate_capacity = np.where(
            target_mask,
            residual_capacity,
            0.0,
        )
        if float(candidate_capacity.sum()) < optional_total - _EPS:
            candidate_capacity = residual_capacity
        destination_batch = _greedy_linear_allocation(
            optional_total,
            candidate_capacity,
            site_cost,
        )

        service_logits = _shares_to_logits(
            service if float(service.sum()) > _EPS else context.effective_capacity
        )
        optional_total_logit = self._optional_total_logit(
            optional_total,
            optional_max_total,
        )
        origin_logits = _shares_to_logits(
            optional_origin
            if float(optional_origin.sum()) > _EPS
            else np.maximum(optional_upper, _EPS)
        )
        destination_logits = _shares_to_logits(
            destination_batch
            if float(destination_batch.sum()) > _EPS
            else np.maximum(candidate_capacity, _EPS)
        )
        return np.concatenate(
            [
                service_logits,
                np.array([optional_total_logit], dtype=np.float64),
                origin_logits,
                destination_logits,
            ]
        ).astype(np.float32)

    def exact_native_teacher_action(self) -> np.ndarray:
        """Translate the exact native teacher into the residual decoder action space."""
        current_step = self.step_index
        context = self.residual_decoder_context(current_step)
        maximum_total = min(
            float(context.pool_totals.sum()),
            max(
                float(context.effective_capacity.sum()) - float(context.total_service),
                0.0,
            ),
        )
        mandatory_total = float(context.mandatory_hint_total)
        if mandatory_total > maximum_total + 1e-8:
            raise SafetyInfeasibleError(
                {
                    "reason": "teacher_mandatory_batch_capacity_deficit",
                    "step": current_step,
                    "mandatory_batch": mandatory_total,
                    "available_batch_capacity": maximum_total,
                }
            )
        urgency_weight = (
            float(self.reward_config.urgency_potential_weight)
            if self.reward_config is not None
            else 0.0
        )
        linear, quad = self._economic_coefficients(current_step)
        if maximum_total - mandatory_total <= _EPS:
            total_batch_target = mandatory_total
        else:
            _mandatory_drain, remaining_entries = _drain_exact_edf(
                context.pools_after_arrival,
                mandatory_total,
            )
            segments = _urgency_segments(
                remaining_entries,
                current_step,
                urgency_weight,
            )
            batch_weight = float(self.reward_batch_completion_weight)
            current = mandatory_total
            segment_index = 0
            total_batch_target = maximum_total
            while current < maximum_total - 1e-12:
                benefit = (
                    segments[segment_index][0]
                    if segment_index < len(segments)
                    else 0.0
                )
                segment_room = (
                    segments[segment_index][1]
                    if segment_index < len(segments)
                    else maximum_total - current
                )
                upper = min(maximum_total, current + segment_room)
                target_mu = batch_weight + benefit
                low_load = context.total_service + current
                high_load = context.total_service + upper
                _low_alloc, _low_cost, mu_low = _economic_dispatch(
                    linear,
                    quad,
                    context.effective_capacity,
                    low_load,
                )
                if mu_low >= target_mu - 1e-10:
                    total_batch_target = current
                    break
                _high_alloc, _high_cost, mu_high = _economic_dispatch(
                    linear,
                    quad,
                    context.effective_capacity,
                    high_load,
                )
                if mu_high <= target_mu + 1e-10:
                    current = upper
                    if segment_index < len(segments):
                        segment_index += 1
                    total_batch_target = current
                    continue
                left = current
                right = upper
                for _ in range(80):
                    midpoint = 0.5 * (left + right)
                    _mid_alloc, _mid_cost, mu_mid = _economic_dispatch(
                        linear,
                        quad,
                        context.effective_capacity,
                        context.total_service + midpoint,
                    )
                    if mu_mid >= target_mu:
                        right = midpoint
                    else:
                        left = midpoint
                total_batch_target = right
                break

        origin_batch, _remaining = _drain_exact_edf(
            context.pools_after_arrival,
            total_batch_target,
        )
        total_load_target = context.total_service + float(total_batch_target)
        if total_load_target <= _EPS:
            load_share = np.full(self.n_dc, 1.0 / self.n_dc, dtype=np.float64)
        else:
            total_load_alloc, _dispatch_cost, _mu = _economic_dispatch(
                linear,
                quad,
                context.effective_capacity,
                total_load_target,
            )
            load_share = total_load_alloc / total_load_target

        service = load_share * context.total_service
        residual_capacity = np.maximum(context.effective_capacity - service, 0.0)
        destination_batch = load_share * float(origin_batch.sum())
        optional_origin = np.maximum(
            origin_batch - np.asarray(context.mandatory_hint_by_origin, dtype=np.float64),
            0.0,
        )
        optional_total = float(optional_origin.sum())
        optional_upper_total = max(
            min(
                float(context.pool_totals.sum()),
                float(residual_capacity.sum()),
            )
            - mandatory_total,
            0.0,
        )
        optional_upper = np.maximum(
            np.asarray(context.pool_totals, dtype=np.float64)
            - np.asarray(context.mandatory_hint_by_origin, dtype=np.float64),
            0.0,
        )
        service_logits = _shares_to_logits(
            service if float(service.sum()) > _EPS else context.effective_capacity
        )
        optional_total_logit = self._optional_total_logit(
            optional_total,
            optional_upper_total,
        )
        origin_logits = _shares_to_logits(
            optional_origin
            if float(optional_origin.sum()) > _EPS
            else np.maximum(optional_upper, _EPS)
        )
        destination_logits = _shares_to_logits(
            destination_batch
            if float(destination_batch.sum()) > _EPS
            else np.maximum(residual_capacity, _EPS)
        )
        return np.concatenate(
            [
                service_logits,
                np.array([optional_total_logit], dtype=np.float64),
                origin_logits,
                destination_logits,
            ]
        ).astype(np.float32)

    def _decode_residual_action(
        self,
        action: np.ndarray,
        context: ResidualDecoderContext,
        *,
        current_step: int,
    ) -> SafetyProjectionResult:
        n_dc = self.n_dc
        service_logits = np.asarray(action[:n_dc], dtype=np.float64)
        optional_total_logit = float(action[n_dc])
        origin_logits = np.asarray(
            action[n_dc + 1 : 2 * n_dc + 1],
            dtype=np.float64,
        )
        destination_logits = np.asarray(
            action[2 * n_dc + 1 :],
            dtype=np.float64,
        )
        if destination_logits.shape != (n_dc,):
            raise ValueError(
                f"expected {n_dc} destination logits, got {destination_logits.shape}"
            )

        tolerance = self.safety_config.tolerance
        service_pref = softmax(service_logits)
        desired_service = service_pref * context.total_service
        service = project_capped_simplex(
            desired_service,
            context.total_service,
            context.effective_capacity,
            tolerance=tolerance,
        )
        residual_capacity = context.effective_capacity - service

        mandatory, mandatory_total, binding = _mandatory_edf_by_origin(
            context.pools_after_arrival,
            np.maximum(context.pool_totals, _EPS),
            current_step,
            self.max_steps,
            context.guaranteed_future_capacity,
            tolerance=tolerance,
        )
        batch_capacity = float(residual_capacity.sum())
        if mandatory_total > batch_capacity + tolerance:
            raise SafetyInfeasibleError(
                {
                    "reason": "mandatory_batch_capacity_deficit",
                    "step": current_step,
                    "mandatory_batch": float(mandatory_total),
                    "available_batch_capacity": float(batch_capacity),
                    "deficit": float(mandatory_total - batch_capacity),
                    "binding_deadline_step": binding,
                }
            )

        flush_mask = np.asarray(
            [site.get_net_demand(current_step) for site in self.sites],
            dtype=np.float64,
        ) < 0.0
        flush_active = bool(
            self.safety_config.negative_demand_flush and np.any(flush_mask)
        )
        maximum_total = min(float(context.pool_totals.sum()), batch_capacity)
        minimum_total = mandatory_total
        if flush_active:
            minimum_total = max(
                minimum_total,
                min(
                    float(context.pool_totals.sum()),
                    float(residual_capacity[flush_mask].sum()),
                ),
            )
        optional_upper = np.maximum(context.pool_totals - mandatory, 0.0)
        optional_total = self._optional_total_from_logit(
            optional_total_logit,
            max(maximum_total - mandatory_total, 0.0),
        )
        optional_origin = np.zeros(n_dc, dtype=np.float64)
        if optional_total > tolerance:
            desired_optional = softmax(origin_logits) * optional_total
            optional_origin = project_capped_simplex(
                desired_optional,
                optional_total,
                optional_upper,
                tolerance=tolerance,
            )
        origin_batch = mandatory + optional_origin
        total_batch = float(origin_batch.sum())

        destination_batch = np.zeros(n_dc, dtype=np.float64)
        desired_destination = np.zeros(n_dc, dtype=np.float64)
        if total_batch > tolerance:
            desired_destination = softmax(destination_logits) * total_batch
            if flush_active:
                negative_target = min(
                    total_batch,
                    float(residual_capacity[flush_mask].sum()),
                )
                negative_upper = residual_capacity * flush_mask.astype(np.float64)
                negative_placement = project_capped_simplex(
                    desired_destination * flush_mask,
                    negative_target,
                    negative_upper,
                    tolerance=tolerance,
                )
                destination_batch = negative_placement
                remaining_target = total_batch - negative_target
                if remaining_target > tolerance:
                    destination_batch = (
                        destination_batch
                        + project_capped_simplex(
                            desired_destination - negative_placement,
                            remaining_target,
                            residual_capacity - negative_placement,
                            tolerance=tolerance,
                        )
                    )
            else:
                destination_batch = project_capped_simplex(
                    desired_destination,
                    total_batch,
                    residual_capacity,
                    tolerance=tolerance,
                )

        transport = exact_transport(
            origin_batch,
            destination_batch,
            tolerance=tolerance,
        )
        service_fractions = np.divide(
            service,
            context.total_service,
            out=np.zeros_like(service),
            where=context.total_service > tolerance,
        )
        drain_rates = np.divide(
            origin_batch,
            context.pool_totals,
            out=np.zeros_like(origin_batch),
            where=context.pool_totals > tolerance,
        )
        batch_fractions = np.divide(
            destination_batch,
            total_batch,
            out=np.zeros_like(destination_batch),
            where=total_batch > tolerance,
        )
        minimum_slack = _minimum_deadline_slack(
            context.pools_after_arrival,
            origin_batch,
            current_step,
            self.max_steps,
            context.guaranteed_future_capacity,
            tolerance=tolerance,
        )
        if minimum_slack < -1e-7:
            raise RuntimeError(
                "residual decoder violated cumulative deadline feasibility"
            )
        return SafetyProjectionResult(
            service=service,
            origin_batch=origin_batch,
            destination_batch=destination_batch,
            transport=transport,
            service_fractions=service_fractions,
            drain_rates=drain_rates,
            batch_fractions=batch_fractions,
            mandatory_by_origin=mandatory,
            mandatory_total=float(mandatory_total),
            desired_service=desired_service,
            desired_origin_batch=mandatory + optional_origin,
            desired_destination_batch=desired_destination,
            effective_capacity=context.effective_capacity,
            residual_capacity=residual_capacity,
            projection_l2=0.0,
            intervened=False,
            binding_deadline_step=binding,
            minimum_deadline_slack=float(minimum_slack),
            service_capacity_slack=float(
                context.effective_capacity.sum() - context.total_service
            ),
            batch_capacity_slack=float(batch_capacity - total_batch),
            negative_flush_active=flush_active,
            exact_zero_drain_count=int(np.count_nonzero(drain_rates == 0.0)),
            exact_full_drain_count=int(np.count_nonzero(drain_rates == 1.0)),
        )

    def _get_obs(self) -> np.ndarray:
        base = super()._get_obs()
        if self.step_index >= self.max_steps:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        context = self.residual_decoder_context(self.step_index)
        extra: list[float] = []
        marginal_total = (
            context.marginal_energy_cost
            + context.marginal_peak_cost
            + context.marginal_demand_charge_cost
        )
        for index in range(self.n_dc):
            extra.extend(
                [
                    float(context.effective_capacity[index]),
                    float(context.conservative_local_residual[index]),
                    float(context.mandatory_hint_by_origin[index]),
                    float(context.marginal_energy_cost[index]),
                    float(context.marginal_peak_cost[index]),
                    float(context.marginal_demand_charge_cost[index]),
                    float(marginal_total[index]),
                ]
            )
        extra.extend(
            [
                float(context.fleet_service_slack),
                float(context.mandatory_hint_total),
                float(
                    max(
                        min(
                            float(context.pool_totals.sum()),
                            float(context.fleet_service_slack),
                        )
                        - float(context.mandatory_hint_total),
                        0.0,
                    )
                ),
                (
                    float(context.minimum_deadline_slack)
                    if math.isfinite(context.minimum_deadline_slack)
                    else float(self.max_steps)
                ),
            ]
        )
        observation = np.concatenate(
            [base.astype(np.float32), np.asarray(extra, dtype=np.float32)]
        )
        return np.nan_to_num(
            observation,
            nan=0.0,
            posinf=float(self.max_steps),
            neginf=-float(self.max_steps),
        ).astype(np.float32)

    def _step_batch(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        t = self.step_index
        tolerance = self.safety_config.tolerance
        preexisting_backlog = np.array(
            [site.backlog for site in self.sites],
            dtype=np.float64,
        )
        if np.any(preexisting_backlog > tolerance):
            raise SafetyInfeasibleError(
                {
                    "reason": "preexisting_local_service_backlog",
                    "step": t,
                    "per_site_backlog": preexisting_backlog.tolist(),
                    "detail": (
                        "The hard zero-backlog guarantee applies only from "
                        "a clean safe state; local backlog is not pooled."
                    ),
                }
            )
        potential_before = self._batch_urgency_potential(t)
        batch_arrivals = np.array(
            [site.get_batch_demand(t) for site in self.sites],
            dtype=np.float64,
        )
        for index, site in enumerate(self.sites):
            if self.burst_aware:
                site.record_arrival(t)
        expired_per_dc = np.array(
            [site.batch_pool.expire(t) for site in self.sites],
            dtype=np.float64,
        )
        if float(expired_per_dc.sum()) > tolerance:
            raise SafetyInfeasibleError(
                {
                    "reason": "pre_action_deadline_miss",
                    "step": t,
                    "expired_per_site": expired_per_dc.tolist(),
                    "expired_total": float(expired_per_dc.sum()),
                }
            )

        context = self.residual_decoder_context(t)
        emergency_reason: str | None = None
        try:
            projection = self._decode_residual_action(
                np.asarray(action, dtype=np.float64),
                context,
                current_step=t,
            )
        except SafetyInfeasibleError:
            raise
        except (RuntimeError, ValueError) as exc:
            emergency_reason = str(exc)
            fallback_action = self._legacy_fallback_action(
                np.asarray(action, dtype=np.float64)
            )
            projection = self._decode_fallback_with_v4_shield(
                fallback_action,
                context,
                current_step=t,
            )

        total_cost = 0.0
        total_energy = 0.0
        total_peak = 0.0
        total_demand_charge = 0.0
        total_grid_mw = 0.0
        info_per_dc: list[dict[str, Any]] = []
        for index, site in enumerate(self.sites):
            power_model = site.power_model or self.power_model
            previous_grid_mw = (
                power_model.compute(site.current_load) * site.rated_power_mw
            )
            service_served = float(projection.service[index])
            batch_served = float(projection.destination_batch[index])
            served = service_served + batch_served
            if served > context.effective_capacity[index] + tolerance:
                raise RuntimeError("residual decoder exceeded site capacity")
            (
                dc_cost,
                energy,
                peak,
                grid_mw,
                net_demand,
                backlog_cost,
                capacity_cost,
                demand_charge,
            ) = self._compute_dc_cost(
                site,
                served,
                0.0,
                t,
                index,
            )
            if (
                self.safety_config.max_grid_mw is not None
                and grid_mw > self.safety_config.max_grid_mw[index] + tolerance
            ):
                raise SafetyInfeasibleError(
                    {
                        "reason": "post_projection_grid_cap_violation",
                        "step": t,
                        "site": site.name,
                        "grid_mw": float(grid_mw),
                        "max_grid_mw": self.safety_config.max_grid_mw[index],
                    }
                )
            if (
                self.safety_config.max_upward_ramp_mw is not None
                and grid_mw - previous_grid_mw
                > self.safety_config.max_upward_ramp_mw[index] + tolerance
            ):
                raise SafetyInfeasibleError(
                    {
                        "reason": "post_projection_ramp_cap_violation",
                        "step": t,
                        "site": site.name,
                        "upward_ramp_mw": float(grid_mw - previous_grid_mw),
                        "max_upward_ramp_mw": (
                            self.safety_config.max_upward_ramp_mw[index]
                        ),
                    }
                )
            site.backlog = 0.0
            site.current_load = served
            if self.memory_enabled:
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
                    "billed_peak_mw": float(self._billed_peak_mw[index]),
                    "backlog_cost": float(backlog_cost),
                    "capacity_cost": float(capacity_cost),
                    "deadline_cost": 0.0,
                    "mandatory_batch_origin": float(
                        projection.mandatory_by_origin[index]
                    ),
                    "batch_transport_from_origin": (
                        projection.transport[index].tolist()
                    ),
                }
            )

        total_batch_served = float(projection.origin_batch.sum())
        for index, site in enumerate(self.sites):
            if batch_arrivals[index] > 0.0:
                site.batch_pool.add(
                    float(batch_arrivals[index]),
                    t + self._deadline_offsets[index],
                )
            drained = site.batch_pool.drain_amount(
                float(projection.origin_batch[index])
            )
            if not math.isclose(
                drained,
                float(projection.origin_batch[index]),
                rel_tol=0.0,
                abs_tol=1e-8,
            ):
                raise RuntimeError("residual origin drain lost work")
            info_per_dc[index]["batch_drained"] = float(drained)
            info_per_dc[index]["batch_pool_size"] = float(
                site.batch_pool.total_demand
            )

        if not math.isclose(
            float(projection.destination_batch.sum()),
            total_batch_served,
            rel_tol=0.0,
            abs_tol=1e-8,
        ):
            raise RuntimeError("origin drain and destination execution differ")

        batch_balance = float(batch_arrivals.sum()) - total_batch_served
        total_batch_accounting_cost = 0.0
        if self.batch_completion_shaping_enabled:
            total_batch_accounting_cost = (
                self.deadline_penalty_weight * batch_balance
            )
            total_cost += total_batch_accounting_cost

        billing_info = self._billing_info(t)
        self.step_index += 1
        terminated = self.step_index >= self.max_steps
        truncated = False
        potential_after = (
            0.0
            if terminated
            else self._batch_urgency_potential(self.step_index)
        )
        penalty_adjustment = (
            self.reward_batch_completion_weight
            - self.batch_completion_weight
        ) * batch_balance
        (
            reward,
            reward_training_cost,
            reward_idle_cost,
            reward_potential_delta,
        ) = self._training_reward(
            total_cost,
            t,
            potential_before,
            potential_after,
            penalty_adjustment,
        )
        if not terminated:
            self._start_billing_period(self.step_index)

        emergency_intervened = emergency_reason is not None
        safety_info = {
            "safety_enabled": True,
            "safety_intervened": emergency_intervened,
            "safety_projection_l2": float(projection.projection_l2),
            "safety_mandatory_batch": float(projection.mandatory_total),
            "safety_mandatory_by_origin": (
                projection.mandatory_by_origin.tolist()
            ),
            "safety_binding_deadline_step": projection.binding_deadline_step,
            "safety_binding_deadline_steps_remaining": (
                projection.binding_deadline_step - t
                if projection.binding_deadline_step is not None
                else None
            ),
            "safety_minimum_deadline_slack": (
                projection.minimum_deadline_slack
            ),
            "safety_service_capacity_slack": (
                projection.service_capacity_slack
            ),
            "safety_batch_capacity_slack": projection.batch_capacity_slack,
            "safety_negative_flush_active": projection.negative_flush_active,
            "safety_exact_zero_drain_count": (
                projection.exact_zero_drain_count
            ),
            "safety_exact_full_drain_count": (
                projection.exact_full_drain_count
            ),
            "safety_transport_conservation_error": float(
                max(
                    np.max(
                        np.abs(
                            projection.transport.sum(axis=1)
                            - projection.origin_batch
                        ),
                        initial=0.0,
                    ),
                    np.max(
                        np.abs(
                            projection.transport.sum(axis=0)
                            - projection.destination_batch
                        ),
                        initial=0.0,
                    ),
                )
            ),
            "safety_service_envelope_total": (
                self.safety_config.service_envelope_total
            ),
            "safety_batch_arrival_envelope_total": (
                self.safety_config.batch_arrival_envelope_total
            ),
            "safety_guaranteed_carried_batch_capacity": (
                self.safety_config.guaranteed_carried_batch_capacity
            ),
            "safety_envelope_id": self.safety_config.envelope_id,
            "safety_envelope_scope": self.safety_config.envelope_scope,
            "safety_emergency_reason": emergency_reason,
            "native_decoder_used": not emergency_intervened,
        }
        info = {
            "total_demand": float(context.total_service),
            "total_batch_expired": 0.0,
            "total_batch_pool": float(
                sum(site.batch_pool.total_demand for site in self.sites)
            ),
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
            "reward_training_cost": reward_training_cost,
            "reward_idle_cost": reward_idle_cost,
            "reward_potential_delta": reward_potential_delta,
            "reward_penalty_adjustment": float(penalty_adjustment),
            "per_dc": info_per_dc,
            **safety_info,
            **billing_info,
        }
        observation = (
            self._get_obs()
            if not terminated
            else np.zeros(
                self.observation_space.shape,
                dtype=np.float32,
            )
        )
        return observation, float(reward), terminated, truncated, info

    def _decode_fallback_with_v4_shield(
        self,
        fallback_action: np.ndarray,
        context: ResidualDecoderContext,
        *,
        current_step: int,
    ) -> SafetyProjectionResult:
        from env.safety_layer import project_joint_action

        return project_joint_action(
            fallback_action,
            context.pools_after_arrival,
            current_step=current_step,
            max_steps=self.max_steps,
            total_service=context.total_service,
            current_batch_arrival=float(context.batch_arrivals.sum()),
            effective_capacity=context.effective_capacity,
            net_demand=np.array(
                [site.get_net_demand(current_step) for site in self.sites],
                dtype=np.float64,
            ),
            price=np.array(
                [site.get_price(current_step) for site in self.sites],
                dtype=np.float64,
            ),
            config=self.safety_config,
        )

    def status_quo_action(self) -> np.ndarray:
        """Return a local/immediate reference action in residual coordinates."""
        t = self.step_index
        service = np.array(
            [site.get_service_demand(t) + site.backlog for site in self.sites],
            dtype=np.float64,
        )
        pending_batch = np.array(
            [
                site.batch_pool.total_demand + site.get_batch_demand(t)
                for site in self.sites
            ],
            dtype=np.float64,
        )
        service_logits = _shares_to_logits(service)
        origin_logits = _shares_to_logits(pending_batch)
        destination_logits = origin_logits.copy()
        return np.concatenate(
            [
                service_logits,
                np.array([20.0], dtype=np.float64),
                origin_logits,
                destination_logits,
            ]
        ).astype(np.float32)
