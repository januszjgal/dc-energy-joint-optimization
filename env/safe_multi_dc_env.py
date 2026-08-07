"""Hard-feasible joint environment used by the post-v3 safety protocol."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from env.multi_dc_env import MultiDCEnv
from env.safety_layer import (
    SafetyConfig,
    SafetyInfeasibleError,
    project_joint_action,
)


class SafeMultiDCEnv(MultiDCEnv):
    """MultiDCEnv with a causal one-step feasibility projection."""

    def __init__(
        self,
        *args: Any,
        safety_config: SafetyConfig,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not self.batch_enabled or not self.batch_spatial_routing:
            raise ValueError(
                "SafeMultiDCEnv requires joint batch spatial routing"
            )
        self.safety_config = safety_config
        self.safety_config.validate(self.n_dc)
        static_capacity = self._static_future_capacity_bounds()
        static_fleet_capacity = float(static_capacity.sum())
        if (
            static_fleet_capacity + self.safety_config.tolerance
            < self.safety_config.future_fleet_capacity_total
        ):
            raise ValueError(
                "future_fleet_capacity_total exceeds the provable fleet "
                "capacity lower bound under memory/grid/ramp constraints: "
                f"{self.safety_config.future_fleet_capacity_total} > "
                f"{static_fleet_capacity}"
            )
        # Preserve the frozen v3 observation shape while removing carryover
        # that expires before the current action can serve it.
        self.actionable_deadline_state = True

    def _static_future_capacity_bounds(self) -> np.ndarray:
        """Return per-site utilization guaranteed under optional hard caps."""
        bounds = np.zeros(self.n_dc, dtype=np.float64)
        tolerance = self.safety_config.tolerance
        for index, site in enumerate(self.sites):
            cap = float(site.capacity)
            if self.memory_enabled:
                if site.memory_cpu_ratio <= 0.0:
                    raise ValueError("memory_cpu_ratio must be positive")
                cap = min(
                    cap,
                    float(
                        site.memory_capacity / site.memory_cpu_ratio
                    ),
                )
            power_model = site.power_model or self.power_model
            idle_grid = power_model.idle_power * site.rated_power_mw
            if self.safety_config.max_grid_mw is not None:
                grid_cap = self.safety_config.max_grid_mw[index]
                if grid_cap < idle_grid - tolerance:
                    raise ValueError(
                        f"max_grid_mw[{index}]={grid_cap} is below "
                        f"unavoidable idle draw {idle_grid}"
                    )
                if power_model.slope > 0.0:
                    cap = min(
                        cap,
                        (
                            grid_cap / site.rated_power_mw
                            - power_model.idle_power
                        )
                        / power_model.slope,
                    )
            if (
                self.safety_config.max_upward_ramp_mw is not None
                and power_model.slope > 0.0
            ):
                # From any non-negative current load, the next step can always
                # increase by at least this utilization amount. It is therefore
                # a conservative lower bound for every future step.
                cap = min(
                    cap,
                    self.safety_config.max_upward_ramp_mw[index]
                    / (power_model.slope * site.rated_power_mw),
                )
            bounds[index] = max(0.0, cap)
        return bounds

    def _effective_capacities(self) -> np.ndarray:
        capacities = np.zeros(self.n_dc, dtype=np.float64)
        for index, site in enumerate(self.sites):
            cap = float(site.capacity)
            if self.memory_enabled:
                if site.memory_cpu_ratio <= 0.0:
                    raise SafetyInfeasibleError(
                        {
                            "reason": "invalid_memory_cpu_ratio",
                            "step": self.step_index,
                            "site": site.name,
                        }
                    )
                cap = min(
                    cap,
                    float(
                        site.memory_capacity / site.memory_cpu_ratio
                    ),
                )
            power_model = (
                site.power_model
                if site.power_model is not None
                else self.power_model
            )
            if self.safety_config.max_grid_mw is not None:
                grid_cap = self.safety_config.max_grid_mw[index]
                if power_model.slope <= 0.0:
                    cap = 0.0 if (
                        power_model.idle_power * site.rated_power_mw
                        > grid_cap
                    ) else cap
                else:
                    cap = min(
                        cap,
                        (
                            grid_cap / site.rated_power_mw
                            - power_model.idle_power
                        )
                        / power_model.slope,
                    )
            if self.safety_config.max_upward_ramp_mw is not None:
                previous_grid = (
                    power_model.compute(site.current_load)
                    * site.rated_power_mw
                )
                grid_cap = (
                    previous_grid
                    + self.safety_config.max_upward_ramp_mw[index]
                )
                if power_model.slope <= 0.0:
                    cap = 0.0 if (
                        power_model.idle_power * site.rated_power_mw
                        > grid_cap
                    ) else cap
                else:
                    cap = min(
                        cap,
                        (
                            grid_cap / site.rated_power_mw
                            - power_model.idle_power
                        )
                        / power_model.slope,
                    )
            capacities[index] = max(0.0, cap)
        return capacities

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

        batch_arrivals = np.zeros(self.n_dc, dtype=np.float64)
        for index, site in enumerate(self.sites):
            arrival = site.get_batch_demand(t)
            batch_arrivals[index] = arrival
            if arrival > 0.0:
                site.batch_pool.add(
                    arrival,
                    t + self._deadline_offsets[index],
                )
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

        current_service = np.array(
            [site.get_service_demand(t) for site in self.sites],
            dtype=np.float64,
        )
        total_service = float(current_service.sum())
        effective_capacity = self._effective_capacities()
        projection = project_joint_action(
            action,
            [site.batch_pool for site in self.sites],
            current_step=t,
            max_steps=self.max_steps,
            total_service=total_service,
            current_batch_arrival=float(batch_arrivals.sum()),
            effective_capacity=effective_capacity,
            net_demand=np.array(
                [site.get_net_demand(t) for site in self.sites],
                dtype=np.float64,
            ),
            price=np.array(
                [site.get_price(t) for site in self.sites],
                dtype=np.float64,
            ),
            config=self.safety_config,
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
                power_model.compute(site.current_load)
                * site.rated_power_mw
            )
            service_served = float(projection.service[index])
            batch_served = float(
                projection.destination_batch[index]
            )
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
            ) = self._compute_dc_cost(
                site,
                served,
                0.0,
                t,
                index,
            )
            if (
                self.safety_config.max_grid_mw is not None
                and grid_mw
                > self.safety_config.max_grid_mw[index] + tolerance
            ):
                raise SafetyInfeasibleError(
                    {
                        "reason": "post_projection_grid_cap_violation",
                        "step": t,
                        "site": site.name,
                        "grid_mw": float(grid_mw),
                        "max_grid_mw": (
                            self.safety_config.max_grid_mw[index]
                        ),
                    }
                )
            if (
                self.safety_config.max_upward_ramp_mw is not None
                and grid_mw - previous_grid_mw
                > self.safety_config.max_upward_ramp_mw[index]
                + tolerance
            ):
                raise SafetyInfeasibleError(
                    {
                        "reason": "post_projection_ramp_cap_violation",
                        "step": t,
                        "site": site.name,
                        "upward_ramp_mw": float(
                            grid_mw - previous_grid_mw
                        ),
                        "max_upward_ramp_mw": (
                            self.safety_config.max_upward_ramp_mw[
                                index
                            ]
                        ),
                    }
                )
            site.backlog = 0.0
            site.current_load = served
            if self.memory_enabled:
                site.current_memory_load = (
                    served * site.memory_cpu_ratio
                )
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
                    "drain_rate": float(
                        projection.drain_rates[index]
                    ),
                    "grid_mw": float(grid_mw),
                    "net_demand": float(net_demand),
                    "net_demand_mw": site.get_net_demand_mw(t),
                    "energy_cost": float(energy),
                    "peak_penalty": float(peak),
                    "demand_charge": float(demand_charge),
                    "billed_peak_mw": float(
                        self._billed_peak_mw[index]
                    ),
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
            drained = site.batch_pool.drain_amount(
                float(projection.origin_batch[index])
            )
            if not math.isclose(
                drained,
                float(projection.origin_batch[index]),
                rel_tol=0.0,
                abs_tol=1e-8,
            ):
                raise RuntimeError("exact origin drain lost work")
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

        safety_info = {
            "safety_enabled": True,
            "safety_intervened": projection.intervened,
            "safety_projection_l2": projection.projection_l2,
            "safety_mandatory_batch": projection.mandatory_total,
            "safety_mandatory_by_origin": (
                projection.mandatory_by_origin.tolist()
            ),
            "safety_binding_deadline_step": (
                projection.binding_deadline_step
            ),
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
            "safety_batch_capacity_slack": (
                projection.batch_capacity_slack
            ),
            "safety_negative_flush_active": (
                projection.negative_flush_active
            ),
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
        }
        info = {
            "total_demand": total_service,
            "total_batch_expired": 0.0,
            "total_batch_pool": float(
                sum(site.batch_pool.total_demand for site in self.sites)
            ),
            "total_batch_accounting_cost": float(
                total_batch_accounting_cost
            ),
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
