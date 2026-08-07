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


def _rates_to_logits(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    clipped = np.clip(values, _EPS, 1.0 - _EPS)
    return np.log(clipped / (1.0 - clipped))


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
