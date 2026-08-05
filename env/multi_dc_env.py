"""Multi-DC Gymnasium environment for grid-aware workload routing.

Each DC is a pure grid-connected load (no on-site solar self-consumption).
The agent decides how to distribute incoming demand spatially across DCs
and, in batch mode, when to drain deferrable batch pools.

The reward combines three objectives:

  1. **Energy cost**:    price[t] × grid_mw × Δt    (operator cost)
  2. **Peak penalty**:   α × grid_mw² × net_demand_normalized[t]
                                              (grid demand smoothing)
  3. **Demand charge**:  c × max_t(grid_mw) per billing period, optional
                                              (operator tariff)

The peak penalty is quadratic in load (so concentrating draw is penalized
more than spreading it out) and scaled by current grid net demand (so the
penalty only bites near the duck-curve neck — late-night consumption is
essentially free). It is a *grid-stress shadow price*, not a tariff: α is
calibrated to a target share of total cost, not derived from a rate schedule.

The demand charge (3) is the actual commercial tariff term — commercial and
industrial customers are billed on the single highest demand interval of the
billing period, typically 30-50% of the total bill. It is off by default
(`demand_charge_rate=0.0`) so results predating it are unaffected; see
`_demand_charge` for the telescoping formulation used to keep a period-max
cost Markov within a per-step reward.
"""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.dc_site import DataCenterSite
from env.power_model import PowerModel


INTERVAL_HOURS = 5.0 / 60.0  # 5-minute intervals

# Bound on the continuous action logits (see action_space comment in __init__).
ACTION_LOGIT_BOUND = 3.0

# Steps in a nominal 30-day billing month (30 × 288). Used to convert a
# $/kW-month tariff into the rate for whatever billing window is in force, so
# the total charge over a month of flat load is the same regardless of window.
STEPS_PER_MONTH = 30 * 288

# Reference industrial demand-charge rate ($/kW-month) used for the
# evaluation-only reporting metric. Mid-range of published US industrial
# tariffs; the reported charge scales linearly, so it can be rescaled post hoc.
REFERENCE_DEMAND_CHARGE_RATE = 15.0


class MultiDCEnv(gym.Env):
    """Gymnasium environment for multi-DC workload routing.

    Observation dimensions (with memory_enabled adding 1 dim/DC legacy or 2 dims/DC batch):
        Legacy: 6*N + 2
        Batch:  8*N + 3
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        sites: list[DataCenterSite],
        power_model: PowerModel,
        max_steps: int | None = None,
        # Service-backlog penalty per unit-step. Calibrated like the deadline
        # penalty (§7.8 logic): delaying a unit of service one step must never be
        # cheaper than the maximum price-spread arbitrage it could capture.
        # Serving one unit-step ≈ slope·R·Δh·1000 ≈ 3,700 kWh; a peak-to-trough
        # price spread of ~$0.05/kWh makes ~$180/unit-step the largest plausible
        # saving from delay, so at 25 a three-hour hold costs $900 ≫ $180 —
        # parking SLO service traffic in the backlog is never profitable.
        # (The old 1.5 made a 3 h hold cost $54: exploitable. Peer-review M4.)
        backlog_weight: float = 25.0,
        capacity_penalty_weight: float = 5.0,
        peak_penalty_weight: float = 0.0,
        # Demand charge ($/kW-month). The real commercial tariff term: billed on
        # the single highest demand interval of each billing period, and in
        # practice 30-50% of a large customer's bill. Default 0.0 leaves it OUT
        # of the reward, so every result produced before this term existed
        # remains reproducible; evaluation reports the charge either way.
        demand_charge_rate: float = 0.0,
        # Billing window in steps. Default 288 = daily. A monthly window (8640)
        # is the realistic tariff but is 89x the γ=0.99 effective horizon
        # (~100 steps), so the once-a-month record-setting reward is invisible
        # to the agent. Daily billing keeps the same total magnitude for flat
        # load while firing often enough to carry gradient — and daily-demand
        # tariffs are themselves real.
        demand_charge_period_steps: int = 288,
        steps_per_day: int = 288,
        # Batch scheduling
        batch_enabled: bool = False,
        flexibility_factor: float = 1.0,
        deadline_penalty_weight: float = 2.0,
        urgency_horizon_steps: int = 12,
        interval_seconds: int = 300,
        # Batch spatial routing: give the agent a second routing head that
        # distributes *drained* batch work across DCs. Deferrable batch has no
        # latency SLO, so (unlike service) it can be executed at any DC. When
        # False, drained batch is executed at its home DC (legacy behavior).
        batch_spatial_routing: bool = True,
        # Memory constraint
        memory_enabled: bool = False,
        # Burst-aware augmentation: add per-DC burst_severity feature to
        # observation (= current batch arrival / rolling 24h mean arrival).
        # Motivated by the burst-window analysis showing the optimization
        # signal concentrates in high-arrival periods (§7-burst).
        burst_aware: bool = False,
        # Static site context in the observation: per-DC (idle_power, slope,
        # capacity, batch_fraction). Without these, a policy can only exploit
        # per-site power/capacity heterogeneity by MEMORIZING which site slot
        # has which hidden constants — which is exactly what the held-out
        # cells e-h evaluation exposed (frozen a-d policies inverted their
        # slope-arbitrage routing on unseen cells, -15% to -33% vs status
        # quo). With the context observed, the same strategy is learnable as
        # a transferable *function* of site parameters. (Peer-review M2.)
        site_context: bool = True,
        # Domain randomization (TRAIN-time only; leave False for evaluation).
        # Observability alone proved insufficient for transfer: the context is
        # constant within every episode trained on fixed cells a-d, so the
        # network receives it as a bias term with no gradient signal to learn
        # dependence on it (ctx-only policies still failed on e-h). At each
        # reset this (i) permutes the compute bundles (workload/tier curves/
        # capacity/power/deadlines) across the market slots, destroying slot
        # identity so the policy must read the context features, and (ii)
        # resamples each site's power parameters uniformly within the range
        # spanned by all eight PowerData2019-fitted cells (idle 0.35-0.60,
        # slope 0.30-0.65), so held-out cells lie inside the training support.
        domain_randomization: bool = False,
    ):
        super().__init__()

        self.sites = sites
        self.power_model = power_model
        self.n_dc = len(sites)

        min_timesteps = min(s.num_timesteps for s in sites)
        self.max_steps = max_steps if max_steps else min_timesteps

        self.backlog_weight = backlog_weight
        self.capacity_penalty_weight = capacity_penalty_weight
        self.peak_penalty_weight = peak_penalty_weight
        self.demand_charge_rate = demand_charge_rate
        self.demand_charge_period_steps = max(1, int(demand_charge_period_steps))
        # $/kW charged per billing window, so that a month of flat load costs
        # exactly `demand_charge_rate` $/kW regardless of window length.
        self.demand_charge_per_period = demand_charge_rate * (
            self.demand_charge_period_steps / STEPS_PER_MONTH
        )
        self.demand_charge_enabled = demand_charge_rate > 0.0
        # Running per-DC max grid draw within the current billing window (MW).
        self._billed_peak_mw = np.zeros(self.n_dc, dtype=np.float64)
        # Rate in force for the window currently open. Set per window rather
        # than once, so a partial trailing window (episode length is rarely an
        # exact multiple of the billing period) is prorated by its true length
        # instead of being charged a full period — which would otherwise nearly
        # double the bill for a 31-day episode on a 30-day window.
        self._period_rate = self.demand_charge_per_period
        self.steps_per_day = steps_per_day

        self.batch_enabled = batch_enabled
        self.flexibility_factor = flexibility_factor
        self.deadline_penalty_weight = deadline_penalty_weight
        self.urgency_horizon_steps = urgency_horizon_steps
        self.interval_seconds = interval_seconds
        self.batch_spatial_routing = batch_spatial_routing and batch_enabled

        self.memory_enabled = memory_enabled
        self.burst_aware = burst_aware and batch_enabled  # only meaningful in batch mode
        self.site_context = site_context
        self.domain_randomization = domain_randomization
        # Original compute bundles for permutation (captured once at init)
        self._bundle_attrs = [
            "workload", "service_curve", "batch_curve", "capacity",
            "memory_capacity", "memory_cpu_ratio", "batch_fraction",
            "batch_mean_duration_sec", "power_model",
            "fleet_cpu_total", "fleet_memory_total", "fleet_machine_count",
        ]
        self._bundles = [
            {a: getattr(s, a) for a in self._bundle_attrs} for s in sites
        ]

        # Per-site deadline offsets (timesteps)
        self._deadline_offsets = self._compute_deadline_offsets()

        # Observation & action spaces
        ctx_dims = 4 if self.site_context else 0  # idle, slope, capacity, batch_fraction
        # Running billed peak. A period-max cost is only Markov if the agent can
        # see the max so far — without it the same (load, price, net demand)
        # state has two different marginal costs depending on unobserved history.
        dc_dims = 1 if self.demand_charge_enabled else 0
        if self.batch_enabled:
            mem_dims = 2 if memory_enabled else 0
            burst_dims = 1 if self.burst_aware else 0
            obs_dim = (8 + mem_dims + burst_dims + ctx_dims + dc_dims) * self.n_dc + 3
            # service routing (N) + drain (N) [+ batch routing (N) if enabled]
            action_dim = (3 if self.batch_spatial_routing else 2) * self.n_dc
        else:
            mem_dims = 1 if memory_enabled else 0
            obs_dim = (6 + mem_dims + ctx_dims + dc_dims) * self.n_dc + 2
            action_dim = self.n_dc

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        # Action bound ±3 (was ±1): SB3 clips continuous actions to this Box, so
        # the bound sets the agent's reachable range after softmax/sigmoid decode.
        # At ±1 PPO was structurally capped to routing shares in [4.3%, 71%] and
        # drain rates in [27%, 73%] — multi-hour holding and full concentration
        # were impossible by construction, while the discrete wrappers (logits
        # ±2/±3) and heuristics (±3/±5) were not so limited. ±3 gives the
        # continuous agent the same reach (shares to ~98.5%, drain 4.7–95.3%).
        # Heuristic baselines and DQN wrappers call step() directly and are
        # unaffected. (Peer-review M5.)
        self.action_space = spaces.Box(
            low=-ACTION_LOGIT_BOUND, high=ACTION_LOGIT_BOUND,
            shape=(action_dim,), dtype=np.float32,
        )

        self.step_index = 0

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _compute_deadline_offsets(self) -> list[int]:
        if not self.batch_enabled:
            return []
        return [
            max(1, math.ceil(site.batch_mean_duration_sec
                             * (1.0 + self.flexibility_factor)
                             / self.interval_seconds))
            for site in self.sites
        ]

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.step_index = 0
        self._billed_peak_mw[:] = 0.0
        if self.domain_randomization:
            # (i) permute compute bundles across market slots — slot identity
            # carries no information, forcing the policy onto the context
            # features; (ii) resample power params within the measured range
            # of all eight PowerData2019-fitted cells.
            perm = self.np_random.permutation(self.n_dc)
            for i, site in enumerate(self.sites):
                for a in self._bundle_attrs:
                    setattr(site, a, self._bundles[int(perm[i])][a])
                idle = float(self.np_random.uniform(0.35, 0.60))
                slope = float(self.np_random.uniform(0.30, 0.65))
                site.power_model = PowerModel(
                    idle_power=idle, slope=slope, peak_power=idle + slope
                )
            self._deadline_offsets = self._compute_deadline_offsets()
        for i, site in enumerate(self.sites):
            site_seed = seed + i if seed is not None else None
            site.reset(seed=site_seed)
        return self._get_obs(), {}

    # ------------------------------------------------------------------
    # Step (dispatcher)
    # ------------------------------------------------------------------

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self.batch_enabled:
            return self._step_batch(action)
        return self._step_legacy(action)

    # ------------------------------------------------------------------
    # Per-DC cost (shared)
    # ------------------------------------------------------------------

    def _demand_charge(self, i: int, grid_mw: float) -> float:
        """Incremental demand charge for site `i` drawing `grid_mw` this step.

        A demand charge bills `c × max_t(grid_mw)` over the billing period — a
        max, not a sum, so it is not directly expressible as a per-step cost.
        The telescoping form used here charges only the amount by which this
        step *raises* the running maximum:

            Σ_t c · max(0, g_t − D_{t−1})  =  c · max_t g_t,   D_0 = 0

        which is exact (each step pays the increment it is responsible for),
        and Markov as long as D_{t−1} is observable — which is why the running
        peak is added to the observation when this term is enabled.

        Cost is sparse by construction: after the first days of a window most
        steps set no record and pay nothing. That is faithful to the tariff but
        hard on credit assignment, hence the daily default window.
        """
        if not self.demand_charge_enabled:
            return 0.0
        excess_mw = max(0.0, grid_mw - self._billed_peak_mw[i])
        if excess_mw > 0.0:
            self._billed_peak_mw[i] = grid_mw
        # ×1000: MW → kW, since the rate is quoted per kW.
        return self._period_rate * excess_mw * 1000.0

    def _roll_billing_period(self, t: int) -> None:
        """Open a new billing window: clear the running peak and set its rate.

        The window's rate is prorated by its actual length, so the total charge
        over an episode is `demand_charge_rate × (max_steps / STEPS_PER_MONTH)`
        $/kW of billed peak regardless of how the episode divides into windows.
        """
        if not self.demand_charge_enabled:
            return
        if t % self.demand_charge_period_steps == 0:
            self._billed_peak_mw[:] = 0.0
            window_len = min(self.demand_charge_period_steps, self.max_steps - t)
            self._period_rate = self.demand_charge_rate * (
                max(window_len, 0) / STEPS_PER_MONTH
            )

    def _compute_dc_cost(
        self,
        site: DataCenterSite,
        served: float,
        new_backlog: float,
        t: int,
        i: int = 0,
    ) -> tuple[float, float, float, float, float, float, float, float]:
        """Compute energy + peak + demand + backlog + capacity costs for one DC.

        Returns (dc_cost, energy_cost, peak_penalty, grid_mw, net_demand,
        backlog_cost, capacity_cost, demand_charge).
        """
        # Per-cell calibrated model when available (R² 0.75-0.80 vs pooled 0.43;
        # §3.2), else the pooled fleet model.
        pm = site.power_model if site.power_model is not None else self.power_model
        power_util = pm.compute(served)
        power_mw = power_util * site.rated_power_mw
        grid_mw = power_mw  # no on-site solar; full draw from grid

        price = site.get_price(t)
        nd = site.get_net_demand(t)

        energy_cost = price * grid_mw * 1000.0 * INTERVAL_HOURS
        peak_penalty = self.peak_penalty_weight * (grid_mw * grid_mw) * nd
        demand_charge = self._demand_charge(i, grid_mw)

        backlog_cost = self.backlog_weight * new_backlog
        cap_penalty = self.capacity_penalty_weight * max(0.0, served - site.capacity)

        dc_cost = energy_cost + peak_penalty + demand_charge + backlog_cost + cap_penalty
        return (dc_cost, energy_cost, peak_penalty, grid_mw, nd, backlog_cost,
                cap_penalty, demand_charge)

    # ------------------------------------------------------------------
    # Legacy step (spatial routing only)
    # ------------------------------------------------------------------

    def _step_legacy(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        t = self.step_index

        # Softmax routing
        action = np.asarray(action, dtype=np.float64)
        exp_a = np.exp(action - action.max())
        fractions = exp_a / exp_a.sum()

        total_demand = sum(s.get_local_demand(t) for s in self.sites)

        self._roll_billing_period(t)

        total_cost = 0.0
        total_energy = 0.0
        total_peak = 0.0
        total_demand_charge = 0.0
        total_grid_mw = 0.0
        info_per_dc = []

        for i, site in enumerate(self.sites):
            assigned = fractions[i] * total_demand
            total_to_serve = assigned + site.backlog

            served = min(total_to_serve, site.capacity)
            if self.memory_enabled:
                mem_required = served * site.memory_cpu_ratio
                if mem_required > site.memory_capacity:
                    served = site.memory_capacity / site.memory_cpu_ratio

            new_backlog = total_to_serve - served

            (dc_cost, energy, peak, grid_mw, nd, backlog_cost, cap_cost,
             dem_charge) = self._compute_dc_cost(site, served, new_backlog, t, i)

            site.backlog = new_backlog
            site.current_load = served
            if self.memory_enabled:
                site.current_memory_load = served * site.memory_cpu_ratio

            total_cost += dc_cost
            total_energy += energy
            total_peak += peak
            total_demand_charge += dem_charge
            total_grid_mw += grid_mw

            info_per_dc.append(
                {
                    "name": site.name,
                    "assigned": float(assigned),
                    "served": float(served),
                    "backlog": float(new_backlog),
                    "grid_mw": float(grid_mw),
                    "net_demand": float(nd),
                    "energy_cost": float(energy),
                    "peak_penalty": float(peak),
                    "demand_charge": float(dem_charge),
                    "billed_peak_mw": float(self._billed_peak_mw[i]),
                    "backlog_cost": float(backlog_cost),
                    "capacity_cost": float(cap_cost),
                }
            )

        reward = -total_cost

        self.step_index += 1
        terminated = self.step_index >= self.max_steps
        truncated = False

        info = {
            "total_demand": float(total_demand),
            "fractions": fractions.tolist(),
            "total_cost": float(total_cost),
            "total_energy_cost": float(total_energy),
            "total_peak_penalty": float(total_peak),
            "total_demand_charge": float(total_demand_charge),
            "total_grid_mw": float(total_grid_mw),
            "per_dc": info_per_dc,
        }

        obs = (
            self._get_obs()
            if not terminated
            else np.zeros(self.observation_space.shape, dtype=np.float32)
        )

        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Batch step (spatial + temporal)
    # ------------------------------------------------------------------

    def _step_batch(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        t = self.step_index
        N = self.n_dc
        action = np.asarray(action, dtype=np.float64)

        routing_logits = action[:N]
        drain_logits = action[N:2 * N]

        exp_a = np.exp(routing_logits - routing_logits.max())
        fractions = exp_a / exp_a.sum()

        drain_rates = 1.0 / (1.0 + np.exp(-drain_logits))

        # Optional independent spatial routing for deferrable batch work.
        # Batch has no latency SLO, so drained batch may execute at any DC.
        if self.batch_spatial_routing:
            batch_logits = action[2 * N:3 * N]
            exp_b = np.exp(batch_logits - batch_logits.max())
            batch_fractions = exp_b / exp_b.sum()
        else:
            batch_fractions = None

        # Phase 1: inject new batch demand into each DC's local pool
        for i, site in enumerate(self.sites):
            new_batch = site.get_batch_demand(t)
            if new_batch > 0:
                site.batch_pool.add(new_batch, t + self._deadline_offsets[i])
            # Track arrival in rolling buffer for burst-severity observation
            if self.burst_aware:
                site.record_arrival(t)

        # Phase 2: expire overdue entries
        expired_per_dc = [site.batch_pool.expire(t) for site in self.sites]

        # Phase 3: each DC's *intended* batch release (drain request). Work is NOT
        # removed from the pool yet — only what actually gets served leaves it (Phase 7).
        # This way capacity-blocked batch keeps its ORIGINAL deadline and waits for a
        # later trough, instead of being re-queued with a 1-step fuse and expiring next
        # step. (Borg's beb tier is queued/best-effort; it does not discard work that
        # merely could not run this instant — see thesis_overview §7.9.)
        drain_request = [
            drain_rates[i] * site.batch_pool.total_demand
            for i, site in enumerate(self.sites)
        ]

        # Phase 4: spatially route the requested batch.
        #   routing on : pool all requested batch and re-split by batch_fractions
        #   routing off: each DC executes its own requested batch (legacy behavior)
        if self.batch_spatial_routing:
            total_request = float(sum(drain_request))
            batch_assigned = [batch_fractions[j] * total_request for j in range(N)]
        else:
            batch_assigned = list(drain_request)

        # Phase 5: route service demand
        total_service = sum(s.get_service_demand(t) for s in self.sites)

        # Phase 6: per-DC serve (service first, then assigned batch) + cost
        self._roll_billing_period(t)

        total_cost = 0.0
        total_energy = 0.0
        total_peak = 0.0
        total_demand_charge = 0.0
        total_grid_mw = 0.0
        info_per_dc = []
        batch_served_list = []

        for i, site in enumerate(self.sites):
            service_assigned = fractions[i] * total_service
            service_to_serve = service_assigned + site.backlog

            batch_work = batch_assigned[i]

            max_serve = site.capacity
            if self.memory_enabled:
                mem_limit = site.memory_capacity / site.memory_cpu_ratio
                max_serve = min(max_serve, mem_limit)

            served = min(service_to_serve + batch_work, max_serve)
            service_served = min(service_to_serve, served)
            batch_served = served - service_served
            new_backlog = service_to_serve - service_served
            batch_served_list.append(batch_served)

            (dc_cost, energy, peak, grid_mw, nd, backlog_cost, cap_cost,
             dem_charge) = self._compute_dc_cost(site, served, new_backlog, t, i)
            deadline_cost = self.deadline_penalty_weight * expired_per_dc[i]
            dc_cost += deadline_cost

            site.backlog = new_backlog
            site.current_load = served
            if self.memory_enabled:
                site.current_memory_load = served * site.memory_cpu_ratio

            total_cost += dc_cost
            total_energy += energy
            total_peak += peak
            total_demand_charge += dem_charge
            total_grid_mw += grid_mw

            info_per_dc.append(
                {
                    "name": site.name,
                    "service_assigned": float(service_assigned),
                    "service_served": float(service_served),
                    "batch_assigned": float(batch_work),     # routed to this DC to execute
                    "batch_served": float(batch_served),
                    "served": float(served),
                    "backlog": float(new_backlog),
                    "batch_expired": float(expired_per_dc[i]),
                    "drain_rate": float(drain_rates[i]),
                    "grid_mw": float(grid_mw),
                    "net_demand": float(nd),
                    "energy_cost": float(energy),
                    "peak_penalty": float(peak),
                    "demand_charge": float(dem_charge),
                    "billed_peak_mw": float(self._billed_peak_mw[i]),
                    "backlog_cost": float(backlog_cost),
                    "capacity_cost": float(cap_cost),
                    "deadline_cost": float(deadline_cost),
                }
            )

        # Phase 7: remove ONLY the served batch from the pools (urgency-first). Each DC
        # gives up the share of its offered work that was actually served; everything
        # unserved stays in its pool with its original deadline (no 1-step requeue), so
        # it can be retried at a later trough and expires only when genuinely overdue.
        total_request_sum = float(sum(drain_request))
        total_served_sum = float(sum(batch_served_list))
        serve_ratio = (
            total_served_sum / total_request_sum if total_request_sum > 1e-12 else 0.0
        )
        for i, site in enumerate(self.sites):
            served_from_i = (
                drain_request[i] * serve_ratio
                if self.batch_spatial_routing
                else batch_served_list[i]
            )
            pool_total = site.batch_pool.total_demand
            if served_from_i > 1e-12 and pool_total > 1e-12:
                site.batch_pool.drain(min(served_from_i / pool_total, 1.0))
            info_per_dc[i]["batch_drained"] = float(served_from_i)  # served & removed from pool
            info_per_dc[i]["batch_pool_size"] = float(site.batch_pool.total_demand)

        reward = -total_cost

        self.step_index += 1
        terminated = self.step_index >= self.max_steps
        truncated = False

        info = {
            "total_demand": float(total_service),
            "total_batch_expired": float(sum(expired_per_dc)),
            "total_batch_pool": float(
                sum(s.batch_pool.total_demand for s in self.sites)
            ),
            "fractions": fractions.tolist(),
            "drain_rates": drain_rates.tolist(),
            "batch_fractions": batch_fractions.tolist() if batch_fractions is not None else None,
            "total_cost": float(total_cost),
            "total_energy_cost": float(total_energy),
            "total_peak_penalty": float(total_peak),
            "total_demand_charge": float(total_demand_charge),
            "total_grid_mw": float(total_grid_mw),
            "per_dc": info_per_dc,
        }

        obs = (
            self._get_obs()
            if not terminated
            else np.zeros(self.observation_space.shape, dtype=np.float32)
        )

        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _site_ctx(self, site: DataCenterSite) -> list[float]:
        """Static per-site context (constant within an episode, varies across
        scenarios/cells): power idle & slope, capacity, deferrable fraction."""
        pm = site.power_model if site.power_model is not None else self.power_model
        return [
            float(pm.idle_power),
            float(pm.slope),
            float(site.capacity),
            float(site.batch_fraction),
        ]

    def _get_obs(self) -> np.ndarray:
        t = self.step_index
        obs_parts: list[float] = []

        if self.batch_enabled:
            total_service = 0.0
            total_batch_pool = 0.0
            for i, site in enumerate(self.sites):
                svc = site.get_service_demand(t)
                total_service += svc
                pool_size = site.batch_pool.total_demand
                total_batch_pool += pool_size
                urgency = site.batch_pool.urgency(t, self.urgency_horizon_steps)
                per_dc = [
                    svc,
                    pool_size,
                    urgency,
                    site.backlog,
                    site.get_price(t),
                    site.get_net_demand(t),
                    site.get_solar_fraction(t),
                    site.current_load,
                ]
                if self.memory_enabled:
                    per_dc.append(site.current_memory_load)
                    per_dc.append(site.memory_backlog)
                if self.burst_aware:
                    per_dc.append(site.get_burst_severity(t))
                if self.site_context:
                    per_dc.extend(self._site_ctx(site))
                if self.demand_charge_enabled:
                    per_dc.append(self._billed_peak_mw[i] / site.rated_power_mw)
                obs_parts.extend(per_dc)
            hour_of_day = (t % self.steps_per_day) / self.steps_per_day
            obs_parts.extend([total_service, total_batch_pool, hour_of_day])
        else:
            total_demand = 0.0
            for i, site in enumerate(self.sites):
                demand = site.get_local_demand(t)
                total_demand += demand
                per_dc = [
                    demand,
                    site.backlog,
                    site.get_price(t),
                    site.get_net_demand(t),
                    site.get_solar_fraction(t),
                    site.current_load,
                ]
                if self.memory_enabled:
                    per_dc.append(site.current_memory_load)
                if self.site_context:
                    per_dc.extend(self._site_ctx(site))
                if self.demand_charge_enabled:
                    per_dc.append(self._billed_peak_mw[i] / site.rated_power_mw)
                obs_parts.extend(per_dc)
            hour_of_day = (t % self.steps_per_day) / self.steps_per_day
            obs_parts.extend([total_demand, hour_of_day])

        return np.array(obs_parts, dtype=np.float32)
