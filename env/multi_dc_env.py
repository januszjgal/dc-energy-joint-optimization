"""Multi-DC Gymnasium environment for grid-aware workload routing.

Each DC is a pure grid-connected load (no on-site solar self-consumption).
The agent decides how to distribute incoming demand spatially across DCs
and, in batch mode, when to drain deferrable batch pools.

The reward combines three objectives:

  1. **Energy cost**:    price[t] × grid_mw × Δt    (operator cost)
  2. **Grid-stress penalty**:
     α × grid_mw² × max(net_demand_signed[t], 0)
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
from bisect import bisect_left
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.dc_site import DataCenterSite
from env.power_model import PowerModel
from env.reward import RewardConfig


INTERVAL_HOURS = 5.0 / 60.0  # 5-minute intervals

# Bound on the continuous action logits (see action_space comment in __init__).
ACTION_LOGIT_BOUND = 3.0

# Reference US commercial/industrial demand-charge rate ($/kW per full-episode
# billing cycle) used for reporting and sensitivity analysis. The study's
# episode is one 31-day billing cycle. It is not a Dutch/Singapore tariff
# reconstruction; the reported charge scales linearly for site-specific rates.
REFERENCE_DEMAND_CHARGE_RATE = 15.0
DEMAND_CHARGE_REWARD_SCALE = 1e-4
PENALTY_SAFETY_MARGIN = 1.05


def resolve_training_gamma(
    requested_gamma: float | None,
    demand_charge_rate: float,
    enforce_batch_completion: bool = False,
) -> float:
    """Resolve an RL discount factor without distorting demand charges.

    Incremental running-maximum charges telescope only in an undiscounted
    return. Demand-charge training therefore requires gamma=1.0. Runs with the
    term disabled retain Stable-Baselines3's historical default of 0.99.
    """
    undiscounted_required = (
        demand_charge_rate > 0.0 or enforce_batch_completion
    )
    gamma = (
        1.0 if undiscounted_required else 0.99
    ) if requested_gamma is None else float(requested_gamma)
    if not math.isfinite(gamma) or not 0.0 < gamma <= 1.0:
        raise ValueError(f"gamma must be in (0, 1], got {gamma}")
    if undiscounted_required and not math.isclose(
        gamma, 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            "demand-charge or completion-guard training requires gamma=1.0 "
            "so dense shaping remains equivalent to the finite-horizon cost"
        )
    return gamma


class MultiDCEnv(gym.Env):
    """Gymnasium environment for multi-DC workload routing.

    Base observation dimensions:
        Spatial-only: 6*N + 2
        Batch:  9*N + 3

    Site context, memory, burst, and demand-charge state add optional
    dimensions. Demand charging adds one running peak per site plus two global
    billing-period features.
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
        # Demand charge ($/kW per configured billing period). The real
        # commercial tariff term is billed on the single highest demand
        # interval of that period. Default 0.0 leaves it OUT of the reward, so
        # every result produced before this term existed remains reproducible;
        # evaluation reports a full-episode reference charge either way.
        demand_charge_rate: float = 0.0,
        # Billing period in steps. None means the complete episode is one
        # billing cycle, matching the paper's post-hoc monthly metric. Shorter
        # periods represent a genuinely different tariff and therefore require
        # a rate quoted for that shorter period; they are not a monthly proxy.
        demand_charge_period_steps: int | None = None,
        # A shorter period charges the full configured rate once per period.
        # Require an explicit acknowledgement so stale legacy values cannot
        # silently multiply a nominal monthly rate.
        allow_multiple_demand_charge_periods: bool = False,
        # Demand charges make the value of dropping or delaying one normalized
        # CPU unit hundreds of thousands of dollars. When enabled, raise the
        # backlog/expiry weights above a conservative marginal-cost bound so
        # the tariff cannot make unserved work economically optimal.
        demand_charge_penalty_guard: bool = True,
        # Enforce economically meaningful completion even without a demand
        # tariff. This enables gamma=1 dense arrival-minus-completion shaping,
        # a terminal-pool value, and a conservative penalty floor.
        enforce_batch_completion: bool = False,
        completion_penalty_weight: float | None = None,
        # Scale only the reward returned to RL; info retains raw dollar costs.
        # None selects 1e-4 with demand charges and 1.0 otherwise.
        reward_scale: float | None = None,
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
        # False, drained batch is executed at its home DC (home-site behavior).
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
        # capacity/power/deadlines) across market slots, destroying slot
        # identity, and (ii) resamples power parameters only within the range
        # spanned by the CURRENT training scenario. Test-fold cells never set
        # training support bounds.
        domain_randomization: bool = False,
        # V3 recovery controls are opt-in so frozen v2 models retain their
        # original observation and reward semantics.
        reward_config: RewardConfig | None = None,
        observe_episode_progress: bool = False,
        deadline_bucket_edges: tuple[int, ...] | None = None,
    ):
        super().__init__()

        self.sites = sites
        self.power_model = power_model
        self.n_dc = len(sites)
        self.memory_enabled = memory_enabled
        self.domain_randomization = domain_randomization

        min_timesteps = min(s.num_timesteps for s in sites)
        self.max_steps = max_steps if max_steps else min_timesteps
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")

        if not math.isfinite(backlog_weight) or backlog_weight < 0.0:
            raise ValueError(
                f"backlog_weight must be finite and non-negative, got {backlog_weight}"
            )
        if (
            not math.isfinite(deadline_penalty_weight)
            or deadline_penalty_weight < 0.0
        ):
            raise ValueError(
                "deadline_penalty_weight must be finite and non-negative, "
                f"got {deadline_penalty_weight}"
            )
        self.configured_backlog_weight = float(backlog_weight)
        self.configured_deadline_penalty_weight = float(
            deadline_penalty_weight
        )
        self.backlog_weight = self.configured_backlog_weight
        self.capacity_penalty_weight = capacity_penalty_weight
        self.peak_penalty_weight = peak_penalty_weight
        if not math.isfinite(demand_charge_rate) or demand_charge_rate < 0.0:
            raise ValueError(
                "demand_charge_rate must be a finite non-negative value, "
                f"got {demand_charge_rate}"
            )
        self.demand_charge_rate = float(demand_charge_rate)
        if demand_charge_period_steps is None:
            self.demand_charge_period_steps = self.max_steps
        else:
            self.demand_charge_period_steps = int(demand_charge_period_steps)
            if self.demand_charge_period_steps <= 0:
                raise ValueError(
                    "demand_charge_period_steps must be positive or None, "
                    f"got {demand_charge_period_steps}"
                )
        self.demand_charge_enabled = self.demand_charge_rate > 0.0
        self.allow_multiple_demand_charge_periods = bool(
            allow_multiple_demand_charge_periods
        )
        self.enforce_batch_completion = bool(enforce_batch_completion)
        if self.enforce_batch_completion and not batch_enabled:
            raise ValueError(
                "enforce_batch_completion requires batch_enabled=True"
            )
        self.reward_config = reward_config
        if self.reward_config is not None:
            if not self.enforce_batch_completion:
                raise ValueError(
                    "RewardConfig requires enforce_batch_completion=True"
                )
            if completion_penalty_weight is not None:
                raise ValueError(
                    "RewardConfig cannot be combined with the legacy "
                    "completion_penalty_weight"
                )
            if reward_scale is not None:
                raise ValueError(
                    "RewardConfig owns reward_scale; do not also pass "
                    "reward_scale"
                )
            if (
                self.reward_config.subtract_idle_cost
                and self.demand_charge_enabled
            ):
                raise ValueError(
                    "idle-cost subtraction is not defined for a "
                    "history-dependent demand charge"
                )
        self.demand_charge_penalty_guard = bool(
            demand_charge_penalty_guard and self.demand_charge_enabled
        )
        self.economic_penalty_guard_enabled = bool(
            self.demand_charge_penalty_guard
            or self.enforce_batch_completion
        )
        self.economic_penalty_floor = 0.0
        self.computed_economic_penalty_floor = 0.0
        if self.economic_penalty_guard_enabled:
            computed_floor = self._economic_penalty_floor()
            self.computed_economic_penalty_floor = computed_floor
            if self.reward_config is not None:
                self.reward_config.validate(computed_floor)
                self.economic_penalty_floor = computed_floor
                self.backlog_weight = (
                    self.reward_config.evaluation_service_backlog_weight
                )
                deadline_penalty_weight = (
                    self.reward_config.evaluation_batch_completion_weight
                )
            elif completion_penalty_weight is not None:
                explicit_floor = float(completion_penalty_weight)
                if (
                    not math.isfinite(explicit_floor)
                    or explicit_floor < computed_floor
                ):
                    raise ValueError(
                        "completion_penalty_weight must be finite and at least "
                        f"the modeled economic floor ${computed_floor:,.2f}; "
                        f"got {completion_penalty_weight}"
                    )
                self.economic_penalty_floor = explicit_floor
            elif self.reward_config is None:
                self.economic_penalty_floor = computed_floor
            if self.reward_config is None:
                self.backlog_weight = max(
                    self.configured_backlog_weight,
                    self.economic_penalty_floor,
                )
                deadline_penalty_weight = max(
                    self.configured_deadline_penalty_weight,
                    self.economic_penalty_floor,
                )
        if self.reward_config is not None:
            self.reward_scale = self.reward_config.reward_scale
        elif reward_scale is None:
            self.reward_scale = (
                DEMAND_CHARGE_REWARD_SCALE
                if (
                    self.demand_charge_enabled
                    or self.enforce_batch_completion
                )
                else 1.0
            )
        else:
            self.reward_scale = float(reward_scale)
        if not math.isfinite(self.reward_scale) or self.reward_scale <= 0.0:
            raise ValueError(
                f"reward_scale must be finite and positive, got {reward_scale}"
            )
        # Running per-DC max grid draw within the current billing window (MW).
        self._billed_peak_mw = np.zeros(self.n_dc, dtype=np.float64)
        self._billing_period_start = 0
        self._billing_period_length = min(
            self.demand_charge_period_steps, self.max_steps
        )
        self._billing_period_index = 0
        if (
            self.demand_charge_enabled
            and self.max_steps % self.demand_charge_period_steps != 0
        ):
            raise ValueError(
                "demand-charge episodes must contain complete billing periods: "
                f"max_steps={self.max_steps} is not divisible by "
                f"demand_charge_period_steps={self.demand_charge_period_steps}"
            )
        if (
            self.demand_charge_enabled
            and self.demand_charge_period_steps != self.max_steps
            and not self.allow_multiple_demand_charge_periods
        ):
            raise ValueError(
                "demand_charge_period_steps shorter than the episode charges "
                "the full demand_charge_rate once per period; pass "
                "allow_multiple_demand_charge_periods=True only when that "
                "per-period tariff is intentional"
            )
        if (
            batch_enabled
            and self.demand_charge_enabled
            and self.demand_charge_period_steps != self.max_steps
        ):
            raise ValueError(
                "batch demand-charge training requires one full-episode "
                "billing cycle so the observed billing progress is also "
                "episode progress"
            )
        self.batch_completion_shaping_enabled = bool(
            batch_enabled
            and (
                self.demand_charge_enabled
                or self.enforce_batch_completion
            )
        )
        self._period_rate = self.demand_charge_rate
        self.steps_per_day = steps_per_day

        self.batch_enabled = batch_enabled
        self.flexibility_factor = flexibility_factor
        self.deadline_penalty_weight = float(deadline_penalty_weight)
        self.batch_completion_weight = float(deadline_penalty_weight)
        self.reward_service_backlog_weight = float(
            self.reward_config.service_backlog_weight
            if self.reward_config is not None
            else self.backlog_weight
        )
        self.reward_batch_completion_weight = float(
            self.reward_config.batch_completion_weight
            if self.reward_config is not None
            else self.batch_completion_weight
        )
        self.urgency_horizon_steps = urgency_horizon_steps
        self.interval_seconds = interval_seconds
        self.batch_spatial_routing = batch_spatial_routing and batch_enabled
        self.observe_episode_progress = bool(observe_episode_progress)
        raw_bucket_edges = (
            tuple(deadline_bucket_edges)
            if deadline_bucket_edges is not None
            else ()
        )
        if any(
            isinstance(edge, bool) or int(edge) != edge or int(edge) <= 0
            for edge in raw_bucket_edges
        ):
            raise ValueError(
                "deadline_bucket_edges must contain positive integers"
            )
        self.deadline_bucket_edges = tuple(int(edge) for edge in raw_bucket_edges)
        if any(
            left >= right
            for left, right in zip(
                self.deadline_bucket_edges,
                self.deadline_bucket_edges[1:],
            )
        ):
            raise ValueError(
                "deadline_bucket_edges must be strictly increasing"
            )
        if self.deadline_bucket_edges and not self.batch_enabled:
            raise ValueError(
                "deadline_bucket_edges require batch_enabled=True"
            )

        self.burst_aware = burst_aware and batch_enabled  # only meaningful in batch mode
        self.site_context = site_context
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
        training_power_models = [
            b["power_model"] if b["power_model"] is not None else self.power_model
            for b in self._bundles
        ]
        self._domain_power_pairs = np.array(
            [
                [pm.idle_power, pm.slope]
                for pm in training_power_models
            ],
            dtype=np.float64,
        )
        self._domain_idle_bounds = (
            min(pm.idle_power for pm in training_power_models),
            max(pm.idle_power for pm in training_power_models),
        )
        self._domain_slope_bounds = (
            min(pm.slope for pm in training_power_models),
            max(pm.slope for pm in training_power_models),
        )

        # Per-site deadline offsets (timesteps)
        self._deadline_offsets = self._compute_deadline_offsets()

        # Observation & action spaces
        ctx_dims = 4 if self.site_context else 0  # idle, slope, capacity, batch_fraction
        # Running billed peak. A period-max cost is only Markov if the agent can
        # see the max so far — without it the same (load, price, net demand)
        # state has two different marginal costs depending on unobserved history.
        dc_dims = 1 if self.demand_charge_enabled else 0
        # Billing-cycle progress and active-rate fraction expose reset timing.
        billing_dims = 2 if self.demand_charge_enabled else 0
        if self.batch_enabled:
            mem_dims = 2 if memory_enabled else 0
            burst_dims = 1 if self.burst_aware else 0
            deadline_dims = (
                len(self.deadline_bucket_edges) + 1
                if self.deadline_bucket_edges
                else 0
            )
            episode_dims = 2 if self.observe_episode_progress else 0
            obs_dim = (
                (
                    9
                    + mem_dims
                    + burst_dims
                    + ctx_dims
                    + dc_dims
                    + deadline_dims
                )
                * self.n_dc
                + 3
                + billing_dims
                + episode_dims
            )
            # service routing (N) + drain (N) [+ batch routing (N) if enabled]
            action_dim = (3 if self.batch_spatial_routing else 2) * self.n_dc
        else:
            mem_dims = 1 if memory_enabled else 0
            episode_dims = 2 if self.observe_episode_progress else 0
            obs_dim = (
                (6 + mem_dims + ctx_dims + dc_dims) * self.n_dc
                + 2
                + billing_dims
                + episode_dims
            )
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

    def _economic_penalty_floor(self) -> float:
        """Bound the one-step value of delaying or dropping one CPU unit.

        The bound includes the largest marginal demand-charge reduction plus
        the largest same-step energy and quadratic grid-penalty reduction. A
        safety margin makes one unit of backlog/expiry strictly more expensive
        than avoiding all three costs.
        """
        max_saving = 0.0
        scenario_models = [
            (
                site.power_model
                if site.power_model is not None
                else self.power_model
            )
            for site in self.sites
        ]
        randomized_idle_bound = max(
            pm.idle_power for pm in scenario_models
        )
        randomized_slope_bound = max(pm.slope for pm in scenario_models)
        randomized_util_bound = max(
            min(
                site.capacity,
                (
                    site.memory_capacity / site.memory_cpu_ratio
                    if self.memory_enabled and site.memory_cpu_ratio > 0.0
                    else site.capacity
                ),
            )
            for site in self.sites
        )
        for site in self.sites:
            pm = (
                site.power_model
                if site.power_model is not None
                else self.power_model
            )
            slope = (
                randomized_slope_bound
                if self.domain_randomization
                else pm.slope
            )
            idle = (
                randomized_idle_bound
                if self.domain_randomization
                else pm.idle_power
            )
            marginal_mw = slope * site.rated_power_mw
            max_util = site.capacity
            if self.memory_enabled and site.memory_cpu_ratio > 0.0:
                max_util = min(
                    max_util,
                    site.memory_capacity / site.memory_cpu_ratio,
                )
            if self.domain_randomization:
                max_util = randomized_util_bound
            max_grid_mw = (
                idle + slope * max_util
            ) * site.rated_power_mw
            max_price = max(
                site.get_price(t) for t in range(self.max_steps)
            )
            max_net_demand = max(
                site.get_net_demand(t) for t in range(self.max_steps)
            )
            energy = max_price * marginal_mw * 1000.0 * INTERVAL_HOURS
            peak = (
                2.0
                * self.peak_penalty_weight
                * max_net_demand
                * max_grid_mw
                * marginal_mw
            )
            demand = self.demand_charge_rate * marginal_mw * 1000.0
            max_saving = max(max_saving, energy + peak + demand)
        return PENALTY_SAFETY_MARGIN * max_saving

    def _episode_context(self, t: int) -> tuple[float, float]:
        """Return pre-action month progress and remaining-horizon fractions."""
        denominator = max(self.max_steps - 1, 1)
        progress = min(max(t / denominator, 0.0), 1.0)
        remaining = min(max((self.max_steps - 1 - t) / denominator, 0.0), 1.0)
        return float(progress), float(remaining)

    def _action_independent_idle_cost(self, t: int) -> float:
        """Return the energy/stress cost paid even at zero utilization."""
        if (
            self.reward_config is None
            or not self.reward_config.subtract_idle_cost
        ):
            return 0.0
        total = 0.0
        for site in self.sites:
            pm = (
                site.power_model
                if site.power_model is not None
                else self.power_model
            )
            idle_mw = pm.compute(0.0) * site.rated_power_mw
            total += (
                site.get_price(t)
                * idle_mw
                * 1000.0
                * INTERVAL_HOURS
            )
            total += (
                self.peak_penalty_weight
                * idle_mw
                * idle_mw
                * max(site.get_net_demand(t), 0.0)
            )
        return float(total)

    def _batch_urgency_potential(self, t: int) -> float:
        """Potential over observable pending/current batch work.

        The negative sign means reducing deadline-weighted pending demand raises
        the potential and therefore provides a positive shaping reward.
        """
        if (
            self.reward_config is None
            or self.reward_config.urgency_potential_weight <= 0.0
            or t >= self.max_steps
        ):
            return 0.0
        score = 0.0
        for i, site in enumerate(self.sites):
            for entry in site.batch_pool.entries:
                remaining = max(entry.deadline_step - t, 1)
                score += entry.cpu_demand / remaining
            arrival = site.get_batch_demand(t)
            if arrival > 0.0:
                score += arrival / max(self._deadline_offsets[i], 1)
        return float(-self.reward_config.urgency_potential_weight * score)

    def _training_reward(
        self,
        total_cost: float,
        t: int,
        potential_before: float = 0.0,
        potential_after: float = 0.0,
        penalty_adjustment: float = 0.0,
    ) -> tuple[float, float, float, float]:
        """Return scaled reward plus its training-only decomposition."""
        idle_cost = self._action_independent_idle_cost(t)
        potential_delta = potential_after - potential_before
        training_cost = (
            total_cost
            + penalty_adjustment
            - idle_cost
            - potential_delta
        )
        reward = -training_cost * self.reward_scale
        return (
            float(reward),
            float(training_cost),
            float(idle_cost),
            float(potential_delta),
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.step_index = 0
        self._start_billing_period(0, force=True)
        if self.domain_randomization:
            # (i) permute compute bundles across market slots — slot identity
            # carries no information, forcing the policy onto the context
            # features; (ii) resample power params within the measured range
            # of all eight PowerData2019-fitted cells.
            perm = self.np_random.permutation(self.n_dc)
            for i, site in enumerate(self.sites):
                for a in self._bundle_attrs:
                    setattr(site, a, self._bundles[int(perm[i])][a])
                left = int(
                    self.np_random.integers(len(self._domain_power_pairs))
                )
                right = int(
                    self.np_random.integers(len(self._domain_power_pairs))
                )
                mix = float(self.np_random.random())
                idle, slope = (
                    (1.0 - mix) * self._domain_power_pairs[left]
                    + mix * self._domain_power_pairs[right]
                )
                site.power_model = PowerModel(
                    idle_power=float(idle),
                    slope=float(slope),
                    peak_power=float(idle + slope),
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
        action_array = np.asarray(action)
        if action_array.shape != self.action_space.shape:
            raise ValueError(
                "action shape must exactly match the configured controller: "
                f"expected {self.action_space.shape}, got {action_array.shape}"
            )
        if not np.isfinite(action_array).all():
            raise ValueError("action must contain only finite values")
        if self.batch_enabled:
            return self._step_batch(action_array)
        return self._step_spatial(action_array)

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

        Cost is sparse by construction: after the early records in a window,
        most steps pay nothing. Training uses gamma=1.0 when this term is
        enabled so record timing cannot change the value of an identical bill.
        """
        if not self.demand_charge_enabled:
            return 0.0
        excess_mw = max(0.0, grid_mw - self._billed_peak_mw[i])
        if excess_mw > 0.0:
            self._billed_peak_mw[i] = grid_mw
        # ×1000: MW → kW, since the rate is quoted per kW.
        return self._period_rate * excess_mw * 1000.0

    def _start_billing_period(self, t: int, *, force: bool = False) -> None:
        """Prepare billing state before the policy observes step ``t``.

        Every configured period is complete; partial trailing periods are
        rejected at construction because their future rate/reset state would
        otherwise require additional episode-position state. Resetting here
        before observation prevents the first action of a period from seeing
        the previous period's peak.
        """
        if t >= self.max_steps:
            return
        if not force and t % self.demand_charge_period_steps != 0:
            return
        self._billed_peak_mw[:] = 0.0
        self._billing_period_start = t
        self._billing_period_length = self.demand_charge_period_steps
        self._billing_period_index = t // self.demand_charge_period_steps
        self._period_rate = self.demand_charge_rate

    def _billing_context(self, t: int) -> tuple[float, float]:
        """Return observable period progress and active-rate fraction."""
        elapsed = max(0, t - self._billing_period_start)
        progress = elapsed / max(self._billing_period_length, 1)
        rate_fraction = (
            self._period_rate / self.demand_charge_rate
            if self.demand_charge_rate > 0.0
            else 0.0
        )
        return float(progress), float(rate_fraction)

    def _billing_info(self, t: int) -> dict[str, Any]:
        progress, rate_fraction = self._billing_context(t)
        return {
            "demand_charge_enabled": self.demand_charge_enabled,
            "demand_charge_rate": float(self.demand_charge_rate),
            "demand_charge_rate_unit": "USD_per_kW_per_billing_period",
            "demand_charge_period_steps": int(self.demand_charge_period_steps),
            "demand_charge_period_index": int(self._billing_period_index),
            "demand_charge_period_length": int(self._billing_period_length),
            "demand_charge_period_rate": float(self._period_rate),
            "demand_charge_period_progress": progress,
            "demand_charge_period_rate_fraction": rate_fraction,
            "allow_multiple_demand_charge_periods": (
                self.allow_multiple_demand_charge_periods
            ),
            "demand_charge_penalty_guard": self.demand_charge_penalty_guard,
            "economic_penalty_guard_enabled": (
                self.economic_penalty_guard_enabled
            ),
            "enforce_batch_completion": self.enforce_batch_completion,
            "completion_penalty_weight": (
                float(self.batch_completion_weight)
                if self.enforce_batch_completion
                else None
            ),
            "economic_penalty_floor": float(self.economic_penalty_floor),
            "computed_economic_penalty_floor": float(
                self.computed_economic_penalty_floor
            ),
            "configured_backlog_weight": float(
                self.configured_backlog_weight
            ),
            "effective_backlog_weight": float(self.backlog_weight),
            "configured_deadline_penalty_weight": float(
                self.configured_deadline_penalty_weight
            ),
            "effective_deadline_penalty_weight": float(
                self.deadline_penalty_weight
            ),
            "batch_completion_weight": float(
                self.batch_completion_weight
            ),
            "reward_service_backlog_weight": float(
                self.reward_service_backlog_weight
            ),
            "reward_batch_completion_weight": float(
                self.reward_batch_completion_weight
            ),
            "reward_scale": float(self.reward_scale),
            "reward_idle_cost_subtraction": bool(
                self.reward_config is not None
                and self.reward_config.subtract_idle_cost
            ),
            "reward_urgency_potential_weight": float(
                self.reward_config.urgency_potential_weight
                if self.reward_config is not None
                else 0.0
            ),
            "observe_episode_progress": self.observe_episode_progress,
            "deadline_bucket_edges": list(self.deadline_bucket_edges),
            "batch_completion_shaping_enabled": (
                self.batch_completion_shaping_enabled
            ),
        }

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
        stress = max(nd, 0.0)

        energy_cost = price * grid_mw * 1000.0 * INTERVAL_HOURS
        peak_penalty = (
            self.peak_penalty_weight
            * (grid_mw * grid_mw)
            * stress
        )
        demand_charge = self._demand_charge(i, grid_mw)

        backlog_cost = self.backlog_weight * new_backlog
        cap_penalty = self.capacity_penalty_weight * max(0.0, served - site.capacity)

        dc_cost = energy_cost + peak_penalty + demand_charge + backlog_cost + cap_penalty
        return (dc_cost, energy_cost, peak_penalty, grid_mw, nd, backlog_cost,
                cap_penalty, demand_charge)

    # ------------------------------------------------------------------
    # Spatial-only step
    # ------------------------------------------------------------------

    def _step_spatial(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        t = self.step_index

        # Softmax routing
        action = np.asarray(action, dtype=np.float64)
        exp_a = np.exp(action - action.max())
        fractions = exp_a / exp_a.sum()

        total_demand = sum(s.get_local_demand(t) for s in self.sites)

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
                    "net_demand_mw": site.get_net_demand_mw(t),
                    "energy_cost": float(energy),
                    "peak_penalty": float(peak),
                    "demand_charge": float(dem_charge),
                    "billed_peak_mw": float(self._billed_peak_mw[i]),
                    "backlog_cost": float(backlog_cost),
                    "capacity_cost": float(cap_cost),
                }
            )

        billing_info = self._billing_info(t)
        (
            reward,
            reward_training_cost,
            reward_idle_cost,
            reward_potential_delta,
        ) = self._training_reward(total_cost, t)

        self.step_index += 1
        terminated = self.step_index >= self.max_steps
        truncated = False
        if not terminated:
            self._start_billing_period(self.step_index)

        info = {
            "total_demand": float(total_demand),
            "fractions": fractions.tolist(),
            "total_cost": float(total_cost),
            "total_energy_cost": float(total_energy),
            "total_peak_penalty": float(total_peak),
            "total_demand_charge": float(total_demand_charge),
            "total_grid_mw": float(total_grid_mw),
            "reward_training_cost": reward_training_cost,
            "reward_idle_cost": reward_idle_cost,
            "reward_potential_delta": reward_potential_delta,
            "per_dc": info_per_dc,
            **billing_info,
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
        potential_before = self._batch_urgency_potential(t)
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
        batch_arrivals = []
        for i, site in enumerate(self.sites):
            new_batch = site.get_batch_demand(t)
            batch_arrivals.append(new_batch)
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
        #   routing off: each DC executes its own requested batch (home-site behavior)
        if self.batch_spatial_routing:
            total_request = float(sum(drain_request))
            batch_assigned = [batch_fractions[j] * total_request for j in range(N)]
        else:
            batch_assigned = list(drain_request)

        # Phase 5: route service demand
        total_service = sum(s.get_service_demand(t) for s in self.sites)

        # Phase 6: per-DC serve (service first, then assigned batch) + cost
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
            deadline_cost = (
                0.0
                if self.batch_completion_shaping_enabled
                else self.deadline_penalty_weight * expired_per_dc[i]
            )
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
                    "batch_arrival": float(batch_arrivals[i]),
                    "batch_assigned": float(batch_work),     # routed to this DC to execute
                    "batch_served": float(batch_served),
                    "served": float(served),
                    "backlog": float(new_backlog),
                    "batch_expired": float(expired_per_dc[i]),
                    "drain_rate": float(drain_rates[i]),
                    "grid_mw": float(grid_mw),
                    "net_demand": float(nd),
                    "net_demand_mw": site.get_net_demand_mw(t),
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

        total_batch_accounting_cost = 0.0
        batch_balance = float(sum(batch_arrivals)) - total_served_sum
        if self.batch_completion_shaping_enabled:
            # With gamma=1 this telescopes exactly:
            # Σ λ(arrivals_t - completed_t) = λ(expired + terminal_pool).
            # It preserves the finite-horizon objective while crediting
            # completion immediately instead of only at the terminal step.
            total_batch_accounting_cost = self.deadline_penalty_weight * (
                batch_balance
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
        total_backlog_units = float(
            sum(dc["backlog"] for dc in info_per_dc)
        )
        penalty_adjustment = (
            (
                self.reward_service_backlog_weight
                - self.backlog_weight
            )
            * total_backlog_units
            + (
                self.reward_batch_completion_weight
                - self.batch_completion_weight
            )
            * batch_balance
        )
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

        info = {
            "total_demand": float(total_service),
            "total_batch_expired": float(sum(expired_per_dc)),
            "total_batch_pool": float(
                sum(s.batch_pool.total_demand for s in self.sites)
            ),
            "total_batch_accounting_cost": float(
                total_batch_accounting_cost
            ),
            "fractions": fractions.tolist(),
            "drain_rates": drain_rates.tolist(),
            "batch_fractions": batch_fractions.tolist() if batch_fractions is not None else None,
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
            **billing_info,
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
                batch_arrival = site.get_batch_demand(t)
                total_service += svc
                pool_size = site.batch_pool.total_demand
                total_batch_pool += pool_size
                urgency = site.batch_pool.urgency(t, self.urgency_horizon_steps)
                per_dc = [
                    svc,
                    # This arrival is injected later in the same step and can
                    # be released by the current action, so it must be observed.
                    batch_arrival,
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
                if self.deadline_bucket_edges:
                    deadline_buckets = list(
                        site.batch_pool.deadline_histogram(
                            t,
                            self.deadline_bucket_edges,
                        )
                    )
                    arrival_bucket = bisect_left(
                        self.deadline_bucket_edges,
                        self._deadline_offsets[i],
                    )
                    deadline_buckets[arrival_bucket] += batch_arrival
                    per_dc.extend(deadline_buckets)
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

        if self.demand_charge_enabled:
            obs_parts.extend(self._billing_context(t))
        if self.observe_episode_progress:
            obs_parts.extend(self._episode_context(t))

        return np.array(obs_parts, dtype=np.float32)
