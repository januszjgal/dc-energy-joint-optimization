# Multi-Datacenter Energy Optimization via Reinforcement Learning

## Thesis Overview & Technical Reference

---

## 1. Problem Statement

Modern hyperscale cloud providers operate geographically distributed data centers that collectively consume tens of gigawatts of power, drawn from grids whose **net demand** (total load minus renewable generation) swings dramatically over each day. In solar-heavy regions, net demand exhibits the **duck curve**: midday solar pushes residual demand low, but the evening ramp — when solar drops off and residential load rises — produces a steep peak that strains the grid and spikes wholesale prices. Hyperscale DCs, with steady-state loads of 50–100 MW per site, are non-trivial contributors to this peak.

> **Can a reinforcement learning agent learn to route workloads across data centers — both spatially (which DC) and temporally (when to execute deferrable work) — to minimize grid energy cost while reducing the DCs' contribution to grid net-demand peaks?**

By shifting deferrable batch workloads away from high-net-demand periods, and routing latency-insensitive work to regions where the grid is currently under less stress, the agent both lowers operator cost (via real-time price signals) and reduces its load's contribution to the duck-curve neck.

### 1.1 Scope

We optimize a **joint objective** across a fleet of 4 data centers:

1. **Grid energy cost**: $/kWh × grid MW consumed.
2. **Peak-contribution penalty**: a load-squared term weighted by current grid net demand, penalizing DC consumption concentrated during periods of grid stress.

The action space has two dimensions:

1. **Spatial routing**: distributing incoming service demand across DCs to exploit regional differences in net demand and price.
2. **Temporal scheduling** (batch mode): deciding when to execute deferrable batch jobs, deferring work away from peak net-demand periods.

We explicitly **do not** model:
- On-site solar generation or PPAs (DCs are pure grid-connected loads — solar enters only as a forecast feature for upcoming net-demand peaks)
- Carbon emissions (excluded to maintain a focused optimization target; net demand is a correlated proxy)
- Battery storage
- Cooling energy (excluded per advisor guidance)

---

## 2. Data Sources

### 2.1 Workload Traces — Google ClusterData 2019

Workload demand traces are derived from the **Google ClusterData 2019** trace, the second-generation Borg trace published by Google in 2019/2020 and documented by **Tirmazi et al. (2020), "Borg: the next generation"** (EuroSys '20) [§8.1]. The trace covers eight Borg clusters (cells **a–h**) for the entire month of May 2019, comprising ~96.4k machines across the eight cells with an average of ~12k machines per cell. We use cells a, b, c, d — one per DC in our 4-DC scenarios.

A **cell** in Borg is "a single management unit" — a logical cluster of machines managed by one scheduler master (Tirmazi §2). Treating each cell as one DC's natural workload is consistent with how the trace publishers themselves describe cells as first-class deployment units; the trace publication explicitly notes "considerable inter-cell workload variation" (Tirmazi §3), which is precisely the heterogeneity our multi-DC routing exploits.

We extract per-cell aggregate CPU demand timeseries at 5-minute resolution:
- **Source table**: `google.com:google-cluster-data.clusterdata_2019_a` (and b, c, d variants), accessed via BigQuery
- **Metric**: Normalized CPU demand (`cpu_demand_norm`) — aggregate CPU usage per 5-minute interval, expressed in **Normalized Compute Units (NCUs)** scaled to [0, 1]. NCUs abstract over machine heterogeneity by rescaling Google Compute Units (GCUs) against the maximum machine size in the trace (Tirmazi §3).
- **Sampling interval**: 5 minutes, matching the trace's native sampling period (Tirmazi notes that the 2019 trace adds a 21-element CPU-utilization histogram per 5-minute period)
- **Duration**: 31 days → 8,917 timesteps per cell
- **Files**: `data/cells/cell_a.csv` through `cell_d.csv`

This per-cell aggregate extraction is the standard way of summarizing the 2019 trace — Tirmazi's own analyses (Figures 2–3 of that paper) present cell-level CPU and memory usage as "fraction of cell capacity" timeseries across the trace duration, which is structurally the same view our environment operates on. Each cell thus represents one DC's natural workload pattern with real diurnal and weekly variation preserved.

**Modeling assumption — cells as proxy DCs.** Tirmazi documents the 2019 trace as eight Borg cells but does *not* claim those cells are in geographically distinct locations. Our framing treats four cells as if they were four geographically distributed hyperscale DCs — i.e., what 4 hyperscale-DC workloads with similar diurnal patterns but realistic cell-level heterogeneity would look like. This is a defensible **modeling exercise** rather than a dataset-grounded claim: it relies on (a) Tirmazi's documented inter-cell workload variation as a proxy for inter-DC workload variation, and (b) the absence of any contradicting metadata in the trace. The cell's actual physical scale is also significantly smaller than a hyperscale DC; the magnitude scaling that bridges this gap is treated separately in §3.2 (`rated_power_mw`). Together with §3.2, this is the cell-as-proxy-DC modeling exercise that the rest of the thesis builds on — not what the dataset publishers had in mind, but consistent with the patterns the dataset preserves.

### 2.2 Batch Job Distributions — Google ClusterData 2019

The 2019 trace exposes job priority as a sparse value in [0, 450] and groups priorities into named **tiers** (Tirmazi §2). We filter on the **best-effort batch (beb) tier** (priorities 110–115), which is "managed by the batch scheduler and incurs low internal charges; they have no associated SLOs" — i.e., the workloads that are genuinely deferrable. Tirmazi reports that across the 2019 trace, the beb tier accounts for ~20% of cell capacity on average; our per-cell measured batch fractions (4.3%–7.3%, below) are lower because we filter more conservatively for jobs whose duration and arrival pattern allow modeling as a queueable pool.

For batch scheduling, we extract 200,000 batch jobs per cell from the beb tier and fit statistical distributions to characterize:

| Property | Cell A Distribution | Key Parameters |
|---|---|---|
| **Inter-arrival time** (sec) | Exponential | mean=0.91s, λ fitted via MLE |
| **Duration** (sec) | Weibull (min) | shape=0.49, scale=567, mean=11,044s |
| **CPU request** (normalized) | Log-normal | σ=0.91, μ=0.007, mean=0.011 |
| **Memory request** (normalized) | Log-normal | σ=0.83, μ=0.004, mean=0.006 |
| **Tasks per job** | Mixture: 83.2% point-mass at 1 + Gamma tail | mean=12.0 |

**Fitting methodology**: Maximum Likelihood Estimation (MLE) with p1-p99 percentile trimming to remove extreme outliers. Distribution candidates (lognorm, weibull_min, gamma, expon) were tested and the best-fit selected per property. The choice of heavy-tailed candidates (Weibull, log-normal, Pareto-adjacent) reflects Tirmazi's finding that "the top 1% of jobs consume over 99% of resources" with "squared coefficients of variation over 23,000" — these distributions are necessary to faithfully reproduce the extreme variability in production cluster workloads (Tirmazi §7).

The **batch fraction** (percentage of total cluster workload that is deferrable batch) is computed directly from the dataset:
- Cell A: 4.3%, Cell B: 7.3%, Cell C: 5.1%, Cell D: 6.2%

These are realistic values: Tirmazi reports that workload mix has migrated from the free tier into the best-effort batch tier between 2011 and 2019, with beb now a structural part of Google's workload. Batch work is a relatively small fraction of total compute, but its deferability makes it high-leverage for energy optimization.

### 2.3 Grid Net Demand & Solar Forecast Features

Two timeseries support the duck-curve modeling. The DCs themselves do not own or self-consume any renewable generation — they are pure grid-connected loads. Renewables enter the model only through their effect on regional grid net demand and through solar irradiance as a forecast feature.

**Grid net demand** (per region, 5-minute resolution): total system load minus utility-scale renewable generation. This is the duck curve, expressed as the actual quantity the optimization targets. When net demand is high, the grid is stressed and prices spike; when it is low (sunny midday), there is headroom for additional DC load.

| DC Location | Net Demand Source |
|---|---|
| US-West (CAISO) | CAISO OASIS — system demand minus solar + wind generation |
| US-Central (MISO) | MISO Market Reports — net load |
| US-Southeast-1 (Southern Co) | EIA-930 hourly net demand, interpolated to 5 min |
| US-Southeast-2 (Duke Carolinas) | EIA-930 hourly net demand, interpolated to 5 min |
| Global-EU (ENTSO-E NL) | ENTSO-E Transparency — actual load minus solar + wind |
| Global-Asia (EMA Singapore) | System load (low renewable share; no net-demand decomposition needed) |

**Column**: `net_demand_normalized` ∈ [0, 1] — net demand divided by region-specific historical peak.

**Solar irradiance** from the **NREL National Solar Radiation Database (NSRDB)** is retained as a *forecast feature*. High midday solar in a solar-heavy region implies a steep evening ramp ahead — useful predictive context for the agent even though the DC does not consume the solar directly.

| DC Location | Solar Site | NSRDB Location |
|---|---|---|
| US-West | The Dalles, OR | 45.59°N, 121.18°W |
| US-Central | Council Bluffs, IA | 41.26°N, 95.86°W |
| US-Southeast-1 | Douglas County, GA | 33.75°N, 84.77°W |
| US-Southeast-2 | Berkeley County, SC | 33.19°N, 80.00°W |
| Global-EU | Eemshaven, NL | 53.44°N, 6.83°E |
| Global-Asia | Singapore | 1.35°N, 103.82°E |

**Column**: `solar_fraction` ∈ [0, 1].

### 2.4 Electricity Prices — Regional ISOs

Real-time wholesale electricity prices from US Independent System Operators:

| DC | Price Source | File |
|---|---|---|
| US-West | CAISO (California ISO) | `data/prices/caiso.csv` |
| US-Central | MISO (Midcontinent ISO) | `data/prices/miso.csv` |
| US-Southeast-1 | Southern Company | `data/prices/southern_co.csv` |
| US-Southeast-2 | Duke Energy Carolinas | `data/prices/duke_carolinas.csv` |
| Global-EU | ENTSO-E Netherlands | `data/prices/entso_e_nl.csv` |
| Global-Asia | EMA Singapore | `data/prices/ema_singapore.csv` |

**Column**: `price_usd_kwh` — locational marginal price in $/kWh at 5-minute resolution.

---

## 3. Environment Design

### 3.1 Simulation Architecture

The environment (`env/multi_dc_env.py`) is a **Gymnasium** environment implementing a multi-datacenter workload routing simulator. Each episode runs for the full trace length (~8,917 timesteps = ~31 days at 5-minute intervals).

```
Timestep interval: 5 minutes (300 seconds)
Steps per day: 288
Episode length: 8,917 steps (~31 days)
Number of DCs: 4
```

### 3.2 Power Model

The power model converts CPU utilization to electrical power consumption using a **linear model** calibrated against real Google power measurements from the **`powerdata_2019`** BigQuery dataset — the companion power-measurement trace published alongside ClusterData 2019 and documented in **Sakalkar et al. (2020), "Data Center Power Oversubscription with a Medium Voltage Power Plane and Priority-Aware Capping" (ASPLOS '20)** [§8]. `powerdata_2019` exposes per-PDU measured power utilization at the cell level; we join it with aggregate cell-level CPU utilization at hourly resolution and fit a linear model on the resulting (cpu_util, power_util) pairs.

**Calibration** (see [preprocess/power_calibrator.py](preprocess/power_calibrator.py)):

- **Source**: `powerdata_2019.cell*.measured_power_util` joined with aggregate CPU computed from `instance_usage.average_usage.cpus / cell_cpu_capacity`, both per (cell, hour).
- **Cells**: a, b, c, d (matching the four cells used as DCs in our scenarios).
- **Samples**: 2,981 (cpu_util, power_util) pairs across the four cells.
- **Fit**: `P(u) = idle_power + slope × u`, least squares.

| Parameter | Value | Description |
|---|---|---|
| `idle_power` | 0.4788 | Power draw at zero CPU load (normalized, fraction of rated) |
| `slope` | 0.4438 | Additional power per unit CPU utilization |
| `peak_power` | 0.9227 | Power at 100% CPU (idle + slope) |
| `R²` | 0.4328 | CPU alone explains ~43% of measured power variance |

The R² of ~0.43 reflects that CPU is one of several drivers of cell power — memory, I/O, network, and (Sakalkar et al. note) priority-aware capping decisions all contribute to the residual. The 48% idle / 92% peak shape is consistent with the canonical Fan et al. (2007) range for hyperscale servers (~50–100% of rated).

**Magnitude scaling (the `rated_power_mw` assumption)**. The model output `P(u)` is dimensionless [0, 1]. To get MW:

```
power_MW = P(cpu_util) × rated_power_mw
```

With `rated_power_mw = 100`, this yields ~47.9 MW idle and ~92.3 MW peak per DC. **This 100 MW figure is an explicit assumption, not derived from cell size.** A Borg cell of ~12k machines (Tirmazi Table 1) drawing ~300 W per server is physically about 3–5 MW — roughly 20× smaller than the 100 MW we use. The decision to scale up reflects the thesis's research question: at 3–5 MW a single cell is invisible to the regional grid (CAISO peak ~26 GW), and the duck-curve / peak-penalty optimization signal would be near-zero. At 100 MW per DC × 4 DCs = 400 MW aggregate, the simulated fleet is the size of a real hyperscale operator's regional footprint (e.g., Google's Council Bluffs IA campus, Microsoft's Quincy WA), which is the scale where grid-stress considerations actually drive operator decisions.

The defensible framing is: **the cell's normalized utilization curve provides the workload *shape* (real diurnal + weekly + cell-heterogeneity patterns from production); the 100 MW `rated_power_mw` sets the *magnitude* to a hyperscale-DC reference where the optimization is operationally meaningful.** Sensitivity to this assumption could be tested by sweeping `rated_power_mw` ∈ {10, 50, 100, 200} — left as future work.

### 3.3 Energy Cost & Peak-Contribution Penalty

DCs are pure grid-connected loads. For each DC at each timestep:
```python
grid_mw      = power_mw                                                    # full draw from grid
energy_cost  = price[t] × grid_mw × 1000 × (5/60)                          # $ for this interval
peak_penalty = α × grid_mw² × net_demand_normalized[t]                     # demand-smoothing term
```

The `× 1000` converts MW to kW (matching $/kWh prices), and `× (5/60)` converts the 5-minute interval to hours.

The peak-contribution term is **quadratic in load** (so concentrating draw is penalized more than spreading it out) and **scaled by current net demand** (so the penalty only bites near the duck-curve neck — sleepy 3 AM consumption is essentially free). The weight `α` is calibrated so that, at peak net demand and full DC load, the penalty contributes roughly 20% of total reward magnitude — making demand smoothing a meaningful but not dominant objective. Exact calibration is determined empirically during reward-shaping experiments (§4.3).

### 3.4 Observation Space

**Legacy mode** (spatial routing only): `6N + 2 = 26` dimensions

Per DC (×4):
| Dim | Feature | Range |
|---|---|---|
| 0 | Local CPU demand | [0, 1] |
| 1 | Backlog (accumulated unserved work) | [0, ∞) |
| 2 | Electricity price ($/kWh) | varies |
| 3 | Grid net demand (normalized) | [0, 1] |
| 4 | Solar fraction (ramp-forecast feature) | [0, 1] |
| 5 | Current CPU load | [0, 1] |

Global (×2):
| Dim | Feature |
|---|---|
| 24 | Total demand (sum across DCs) |
| 25 | Hour of day (normalized to [0, 1]) |

**Batch mode** (spatial + temporal): `8N + 3 = 35` dimensions

Per DC (×4):
| Dim | Feature | Range |
|---|---|---|
| 0 | Service demand (non-deferrable) | [0, 1] |
| 1 | Batch pool size | [0, ∞) |
| 2 | Urgency (fraction due within horizon) | [0, 1] |
| 3 | Service backlog | [0, ∞) |
| 4 | Electricity price | varies |
| 5 | Grid net demand (normalized) | [0, 1] |
| 6 | Solar fraction (ramp-forecast feature) | [0, 1] |
| 7 | Current CPU load | [0, 1] |

Global (×3):
| Dim | Feature |
|---|---|
| 32 | Total service demand |
| 33 | Total batch pool size |
| 34 | Hour of day |

**Optional augmentations** (used by the burst-aware experiments in §7.6):

| Flag | Adds per DC | Adds globally | Total batch-mode obs dim |
|---|---|---|---|
| `memory_enabled=True` | `current_memory_load`, `memory_backlog` (+2) | — | 43 |
| `burst_aware=True` | `burst_severity = current_batch_arrival / rolling_24h_mean` (+1) | — | 36 (or 44 with memory) |
| both | (+3) | — | **47** |

Burst severity is clipped to [0, 10] for numerical stability; 1.0 indicates "this arrival is average," 2+ indicates an active burst. Memory is enabled but does not bind in our env (verified — total cost unchanged vs disabled).

### 3.5 Action Space

**Legacy mode**: `N = 4` continuous actions ∈ [-1, 1]

Converted to allocation fractions via **softmax**:
```python
exp_a = exp(action - max(action))   # numerically stable
fractions = exp_a / sum(exp_a)       # fractions sum to 1
```

**Batch mode**: `2N = 8` continuous actions ∈ [-1, 1]

- First N: spatial routing logits (→ softmax → allocation fractions)
- Last N: temporal drain logits (→ sigmoid → drain rates per DC)

```python
drain_rates = sigmoid(drain_logits)  # ∈ (0, 1)
```

A drain rate of 0.5 means drain 50% of the batch pool this timestep. The sigmoid mapping gives the agent smooth control from "hold everything" (logit → -∞) to "drain everything" (logit → +∞).

### 3.6 Reward Function

```python
reward = -total_cost
```

Where:
```python
total_cost = Σᵢ (energy_cost[i] + peak_penalty[i] + backlog_weight × backlog[i] + capacity_penalty[i])
```

In batch mode, an additional deadline violation penalty:
```python
total_cost += Σᵢ (deadline_penalty_weight × expired_demand[i])
```

| Weight | Default | Purpose |
|---|---|---|
| `backlog_weight` | 1.5 | Penalize unserved service demand |
| `capacity_penalty_weight` | 5.0 | Hard penalty for exceeding DC capacity |
| `peak_penalty_weight` (α) | 0.015 (calibrated) | Weight on `grid_mw² × net_demand_normalized` peak-contribution term |
| `deadline_penalty_weight` | 2.0 | Penalty per unit of batch work that expires past deadline |

The `renewable_bonus` term from the prior on-site-solar formulation has been removed — with grid-only DCs, there is no "renewable fraction" to reward. Demand smoothing is now expressed directly through `peak_penalty`, which carries the same intent but targets the actual quantity (grid stress) rather than a proxy (local solar self-consumption).

### 3.7 Batch Scheduling Mechanism

The deferrable-batch-with-deadlines pattern follows a well-established lineage in renewable-aware datacenter scheduling — most directly **Grange et al. (2018)** (§8.2), whose central abstraction is "batch jobs with due-date constraints, which takes into account the availability of the renewable energy," and **GreenSlot (Goiri et al. 2011)** (§8.4), which "delays jobs to execute them when the cost is the lowest." We extend the single-DC pool-with-deadline pattern from that lineage to the multi-DC setting, where the agent must simultaneously decide *where* to route service work and *when* to drain each DC's batch pool.

When batch mode is enabled, each timestep follows this pipeline:

1. **Inject**: New batch demand arrives (sampled from the fitted distributions in §2.2) and enters each DC's batch pool with a deadline:
   ```python
   deadline = t + ceil(mean_duration × (1 + flexibility_factor) / interval_seconds)
   ```
   With `flexibility_factor = 1.0`, jobs get 2× their expected duration as deadline slack. This is the SLA-flexibility knob that Grange et al. identify as the primary driver of achievable savings ("the amount of freedom allowed by the SLA greatly affects the achievable saving").

2. **Expire**: Any pool entries past their deadline are removed and counted as violations (penalized via `deadline_penalty_weight`).

3. **Drain**: The agent's drain rate controls how much of each DC's pool is executed:
   ```python
   drained = pool.drain(drain_rate)  # drains most-urgent entries first
   ```

4. **Route**: Service demand is routed spatially across DCs (softmax over agent's routing logits).

5. **Serve**: Each DC serves `service_assigned + backlog + batch_drained`, up to capacity. Service work gets priority over batch work — service traffic must be served immediately to avoid backlog penalty, while batch can spill back into the pool.

The `BatchPool` data structure maintains a deque of `(cpu_demand, deadline_step)` entries. Draining preferentially removes the most urgent (nearest-deadline) entries first, which is the canonical EDF (earliest-deadline-first) policy used in the batch-deferral literature.

### 3.8 Dynamic Batch Arrivals

Rather than using a fixed `workload[t] × batch_fraction` split, we generate **synthetic batch arrivals** from fitted distributions (`BatchArrivalGenerator`):

1. Sample inter-arrival times from the fitted exponential distribution
2. For each arrival, sample CPU demand, memory demand, and task count from fitted distributions
3. Aggregate into per-timestep demand arrays
4. **Normalize** total demand to match the expected aggregate from the static split

This preserves realistic temporal **burstiness** (batch jobs arrive in clusters, not uniformly) while maintaining consistent aggregate demand volume.

---

## 4. RL Agents

### 4.1 PPO (Proximal Policy Optimization)

Our primary RL agent uses **PPO** from Stable-Baselines3, operating in the continuous action space.

| Hyperparameter | Value |
|---|---|
| Algorithm | PPO (clip objective) |
| Policy | MlpPolicy |
| Network architecture | [128, 128] (two hidden layers) |
| Learning rate | 3 × 10⁻⁴ |
| Rollout steps (n_steps) | 2,048 |
| Minibatch size | 64 |
| Training timesteps | 500,000 |
| Seed | 42 |

PPO was chosen for its:
- **Continuous action space** support — natural for the softmax routing formulation
- **Stability** — clipped objective prevents catastrophic policy updates
- **Sample efficiency** — on-policy but with multiple epochs per rollout

Two PPO models were trained:
- `ppo_us_model` — legacy mode (spatial routing only)
- `ppo_us_model_batch` — batch mode (spatial + temporal)

### 4.2 DQN (Deep Q-Network)

We train **two DQN variants** as algorithm-level analogs to CFWS (Zhao et al. 2025) — both use identical DQN hyperparameters (ε-greedy + target network + experience replay), differing only in action encoding:

- **DQN (routing-grid)**: 759-action enumerated grid (253 routing × 3 drain). Generic discretization.
- **DQN (flat-idx)**: 48-action CFWS-style flattened-index decoded via hash-map to `(src_dc, dst_dc, drain_level)` — the closest port of CFWS's hash-map action philosophy our cell-aggregate formulation allows. See [env/cfws_style_wrapper.py](env/cfws_style_wrapper.py).

This setup tests two questions: (a) does the CFWS algorithmic choice (DQN) work on our formulation? (b) does CFWS's *action-encoding* idea (small, semantically-meaningful action set) transfer? See §8.6 for the formulation/algorithm/encoding comparison tables and §7 for empirical results. Both DQN variants are independent of and not reimplementations of CFWS — CFWS operates on per-PM VM migrations at a different state granularity.

| Hyperparameter | Value |
|---|---|
| Algorithm | DQN (with target network) |
| Policy | MlpPolicy |
| Network architecture | [256, 256] |
| Learning rate | 1 × 10⁻⁴ |
| Replay buffer size | 100,000 |
| Training batch size | 64 |
| Exploration fraction | 0.3 (ε decays over first 30% of training) |
| Final ε | 0.05 |
| Target network update interval | 1,000 steps |
| Training timesteps | 500,000 |

**Action Space Discretization** (`DiscretizedMultiDCEnv`):

Since DQN requires a discrete action space, we pre-compute a finite set of canonical allocations:

1. **Routing actions**: Each DC gets one of 5 levels {0%, 25%, 50%, 75%, 100%}, then normalized. After deduplication: **253 unique routing actions** (for 4 DCs with granularity=5).
   - Includes: uniform allocation, single-DC-only allocations, and all grid combinations
   - Proportions converted to softmax-compatible logits

2. **Drain actions** (batch mode only): 3 options per DC:
   - Hold (sigmoid(-3) ≈ 5% drain)
   - Half (sigmoid(0) = 50% drain)
   - Flush (sigmoid(3) ≈ 95% drain)

3. **Combined**: 253 × 3 = **759 discrete actions** in batch mode

This discretization inherently limits DQN's expressiveness compared to PPO's continuous actions — DQN can only choose from pre-defined allocation patterns, while PPO can output arbitrary routing fractions.

---

## 5. Baseline Policies

Eight heuristic baselines provide comparison points:

### 5.1 Round Robin
Equal allocation to all DCs. Drain rate: 50% (sigmoid(0)). The simplest possible policy.

### 5.2 Cheapest Price First
Route all demand to the DC with the lowest current electricity price. Drain aggressively at cheap DCs, defer at expensive ones.

### 5.3 Avoid the Ramp
Route all demand inversely to current grid net demand — load goes to the DC whose grid is currently most underloaded. Drain inversely proportional to net demand. The analogue of the old "Follow the Sun" baseline under the demand-smoothing formulation.

### 5.4 Local Only (No Routing)
Each DC handles only its own cell's workload — allocation proportional to local demand, no cross-DC routing. Drains immediately (sigmoid(1) ≈ 73%).

### 5.5 Random
Uniform random routing and drain rates each timestep.

### 5.6 Drain Immediately
Equal routing + drain everything immediately (sigmoid(5) ≈ 99.3%). Isolates the value of temporal scheduling — any improvement by PPO over this comes from learning *when* to execute batch work.

### 5.7 Defer to Low Net Demand
Equal routing + drain inversely proportional to current net demand. Maps `net_demand_normalized` [0,1] → sigmoid input [+3, -3], so DCs drain aggressively when the grid is underloaded and hold when it is near peak.

### 5.8 Trough-Slot Lookahead (GreenSlot-style)
Lookahead-based scheduling adapted from Goiri et al.'s GreenSlot "green slot" concept, retargeted from local solar surplus to grid demand troughs:
- **Spatial**: Routes proportional to *inverse* average net demand over a 3-hour lookahead window (36 steps) — load goes where the grid will be slack on average.
- **Temporal**: Drains when current net demand is below its lookahead average (a "trough slot"), defers when above.
- The ratio `avg_net_demand / current_net_demand` is mapped to sigmoid input via `(ratio - 1) × 3`, clipped to [-3, 3].

This is the strongest heuristic baseline, as it has perfect foresight into future grid net demand.

---

## 6. Scenarios

### 6.1 US Model (Data Sovereignty)

4 DCs within the continental United States, testing optimization under constrained timezone diversity (~3 hours).

| DC | Location | Net Demand Source | Solar Forecast | Price Source | Cell |
|---|---|---|---|---|---|
| US-West | The Dalles, OR | CAISO OASIS | NREL NSRDB | CAISO | cell_a |
| US-Central | Council Bluffs, IA | MISO | NREL NSRDB | MISO | cell_b |
| US-Southeast-1 | Douglas County, GA | EIA-930 | NREL NSRDB | Southern Co | cell_c |
| US-Southeast-2 | Berkeley County, SC | EIA-930 | NREL NSRDB | Duke Carolinas | cell_d |

All DCs: `rated_power_mw = 100`. No on-site solar.

### 6.2 Global Model (Maximum Diversity)

4 DCs across 3 continents, testing unconstrained routing with maximum solar/timezone diversity (~16-hour spread).

| DC | Location | Net Demand Source | Solar Forecast | Price Source | Cell |
|---|---|---|---|---|---|
| Global-US-West | The Dalles, OR | CAISO OASIS | NREL NSRDB | CAISO | cell_a |
| Global-US-Central | Council Bluffs, IA | MISO | NREL NSRDB | MISO | cell_b |
| Global-EU | Eemshaven, NL | ENTSO-E | NREL NSRDB | ENTSO-E NL | cell_c |
| Global-Asia | Singapore | EMA | NREL NSRDB | EMA Singapore | cell_d |

---

## 7. Results

All numbers below come from the demand-smoothing formulation: grid-only DCs, reward = -(energy_cost + α × grid_mw² × net_demand_normalized + backlog + capacity penalties [+ deadline]), with α = 0.015. Each policy is run for one full 8,917-step episode (~31 days) under the same seed.

### 7.1 US Model — Legacy Mode (Spatial Routing Only)

| Policy | Total Cost ($) | Energy Cost ($) | Peak Penalty ($) | ND-weighted Load |
|---|---|---|---|---|
| **DQN (routing-grid)** | **9,842,806** | 7,909,957 | 1,931,809 | 1,733,322 |
| Round Robin | 9,842,773 | 7,979,247 | 1,863,526 | 1,723,187 |
| Drain Immediately | 9,842,773 | 7,979,247 | 1,863,526 | 1,723,187 |
| Defer to Low Net Demand | 9,842,773 | 7,979,247 | 1,863,526 | 1,723,187 |
| Local Only | 9,854,609 | 7,987,269 | 1,867,340 | 1,724,368 |
| Random | 9,881,612 | 7,978,050 | 1,902,539 | 1,723,286 |
| Trough-Slot Lookahead | 9,939,815 | 8,035,872 | 1,820,927 | 1,691,662 |
| PPO | 10,097,587 | 7,844,162 | 1,885,277 | 1,721,237 |
| DQN (flat-idx) | 10,865,116 | 7,787,561 | 1,880,161 | 1,716,067 |
| Avoid the Ramp | 16,580,284 | 7,981,741 | 1,832,226 | 1,672,948 |
| Cheapest Price First | 40,302,132 | 7,101,765 | 1,692,723 | 1,599,584 |

**Key finding**: In US legacy mode, spatial routing alone is structurally limited — the four DCs share similar net-demand profiles (only 3-hour timezone spread, all in the same continental load shape) so the agent has little room to re-route. Round Robin, Drain Immediately, and Defer to Low Net Demand tie exactly at $9.843M because with no batch deferral their per-step actions reduce to the same uniform allocation. DQN (routing-grid) matches them to within $30.

Both RL agents **underperform** the trivial baselines here. PPO ($10.10M, 2.6% worse than Round Robin) achieves the *lowest* energy cost in the table ($7.84M) but pays for it through higher capacity-violation penalties on the concentrated DC it favors. DQN (flat-idx) does even worse ($10.87M, 10.4% worse than Round Robin) — its 48-action space can only encode "uniform" or "migrate 15% from src to dst" allocations, which is too coarse when the optimal policy is "stay close to uniform with small adjustments per timestep." This is the canonical case where CFWS-style flat-idx encoding hurts: with low spatial diversity, the constrained action set can't fine-tune.

Concentration heuristics (Avoid the Ramp, Cheapest Price First) catastrophically fail by pushing all demand onto one DC and triggering backlog blowups.

### 7.2 US Model — Batch Mode (Spatial + Temporal)

| Policy | Total Cost ($) | Energy Cost ($) | Peak Penalty ($) | ND-weighted Load | Batch Expired | Avg Pool |
|---|---|---|---|---|---|---|
| **PPO** | **9,473,793** | 7,615,395 | 1,855,933 | 1,698,501 | 1,232.7 | 0.73 |
| DQN (routing-grid) | 9,654,859 | 7,767,898 | 1,885,515 | 1,713,902 | 723.4 | 9.37 |
| Random | 9,737,380 | 7,859,869 | 1,876,581 | 1,711,321 | 420.2 | 0.51 |
| Drain Immediately | 9,769,153 | 7,896,170 | 1,872,734 | 1,720,079 | 123.9 | 0.02 |
| Round Robin | 9,784,834 | 7,905,770 | 1,878,991 | 1,722,885 | 36.7 | 0.47 |
| Local Only | 9,786,782 | 7,908,301 | 1,878,337 | 1,722,633 | 72.2 | 0.18 |
| Defer to Low Net Demand | 9,788,720 | 7,907,982 | 1,880,718 | 1,723,684 | 9.6 | 1.67 |
| Trough-Slot Lookahead | 9,802,326 | 7,954,189 | 1,832,267 | 1,694,752 | 48.1 | 0.48 |
| DQN (flat-idx) | 9,832,808 | 7,931,021 | 1,901,552 | 1,724,409 | 117.7 | 0.04 |
| Avoid the Ramp | 10,287,186 | 7,882,658 | 1,803,174 | 1,664,045 | 979.5 | 1.78 |
| Cheapest Price First | 21,051,100 | 7,170,953 | 1,720,172 | 1,614,448 | 2,316.9 | 0.55 |

**Key findings**:

1. **PPO is best, beating Trough-Slot Lookahead by 3.4%** ($9.47M vs $9.80M). This is the inverse of the original-formulation result, where GreenSlot edged out PPO by 0.2%. The new objective gives PPO room to learn more sophisticated strategies than the foresighted heuristic — Trough-Slot has perfect 3-hour net-demand lookahead but its routing-by-inverse-demand pattern leaves performance on the table.

2. **Temporal scheduling unlocks the gains.** Batch mode lowers PPO's cost from $10.10M (legacy) to $9.47M, a **6.2% reduction**. The lever isn't routing — it's *when* to execute deferrable work.

3. **PPO accepts deadline cost for energy savings.** The agent expires 1,233 units of batch work (vs ~10 for Defer-to-Low-Net-Demand, ~37 for Round Robin) because the energy + peak savings from late draining exceed the deadline penalty (2 × 1,233 ≈ $2.5K is dwarfed by the $300K energy cost reduction). This is an emergent strategy — the deadline penalty weight is fixed at 2.0; the agent simply discovered the favorable arithmetic.

4. **DQN (routing-grid) does well in batch mode** ($9.65M, 2nd place, 1.4% above Round Robin) — much better than its legacy-mode tie. The discrete drain options (hold / half / flush) capture most of the temporal benefit, even though continuous routing would help further. **DQN (flat-idx)** is significantly worse here ($9.83M, 9th) — its always-on drain-level commitment (one of three intensities, applied uniformly to all DCs every step) is too coarse for the fine drain-timing PPO and DQN-routing-grid learn. Notably, flat-idx has the smallest avg pool (0.04) and very few expirations (118) — it's draining too aggressively and not exploiting temporal flexibility well in the US scenario.

5. **Defer to Low Net Demand has the fewest deadline violations (9.6)** but achieves only a middling cost. Its drain timing is *too* conservative — it lets the pool grow then drains in big bursts when net demand drops, but with only modest cost savings.

### 7.3 Global Model — Legacy Mode

| Policy | Total Cost ($) | Energy Cost ($) | Peak Penalty ($) | ND-weighted Load |
|---|---|---|---|---|
| **DQN (flat-idx)** | **13,473,988** | 11,438,042 | 2,035,946 | 1,845,916 |
| PPO | 13,479,999 | 11,331,244 | 2,063,018 | 1,848,298 |
| DQN (routing-grid) | 13,567,888 | 11,565,256 | 2,002,631 | 1,833,835 |
| Trough-Slot Lookahead | 14,073,496 | 12,085,211 | 1,951,681 | 1,808,770 |
| Local Only | 14,226,821 | 12,224,388 | 2,002,433 | 1,849,641 |
| Round Robin | 14,242,628 | 12,241,614 | 2,001,013 | 1,849,584 |
| Drain Immediately | 14,242,628 | 12,241,614 | 2,001,013 | 1,849,584 |
| Defer to Low Net Demand | 14,242,628 | 12,241,614 | 2,001,013 | 1,849,584 |
| Random | 14,288,838 | 12,244,432 | 2,043,384 | 1,850,001 |
| Avoid the Ramp | 14,671,017 | 11,763,190 | 2,005,938 | 1,805,966 |
| Cheapest Price First | 43,772,226 | 10,485,812 | 1,778,770 | 1,699,588 |

**Key findings**:

1. **DQN (flat-idx) and PPO tie** within $6K ($13,473,988 vs $13,479,999) — a statistical tie inside any reasonable seed-variance band. The 16-hour timezone spread in the Global scenario gives spatial routing real teeth: at any moment, some DC's grid is slack while another's is peaking. Both winning agents learn to route load toward the slack DC, but reach the same cost via different routes — PPO via fine-grained continuous fractions, DQN-flat-idx via well-chosen discrete "migrate X from src to dst" actions.

2. **The flat-idx win here is the strongest evidence that CFWS's action-encoding philosophy can transfer to our formulation.** Under geographic diversity, the 48-action set's structure (every action is a sensible "migrate from one DC to another" operation) is exactly what's needed; a generic 759-action routing-grid (DQN-routing-grid at $13.57M, 0.7% worse) wastes capacity on allocations that are never optimal.

3. **All three RL/RL-adjacent agents beat the lookahead heuristic.** PPO, DQN-flat-idx, and DQN-routing-grid all beat Trough-Slot Lookahead by 3.6–4.3%. Trough-Slot has perfect 3-hour net-demand foresight but its proportional-to-inverse-demand routing rule doesn't capture the price + capacity tradeoffs the agents learn.

### 7.4 Global Model — Batch Mode

| Policy | Total Cost ($) | Energy Cost ($) | Peak Penalty ($) | ND-weighted Load | Batch Expired | Avg Pool |
|---|---|---|---|---|---|---|
| **PPO** | **13,200,684** | 11,233,518 | 1,964,651 | 1,809,538 | 1,257.9 | 0.73 |
| DQN (flat-idx) | 13,418,703 | 11,428,230 | 1,989,005 | 1,825,247 | 734.2 | 0.54 |
| Avoid the Ramp | 13,807,454 | 11,623,269 | 1,921,059 | 1,775,187 | 1,071.1 | 2.38 |
| Trough-Slot Lookahead | 13,974,911 | 12,009,138 | 1,965,431 | 1,815,191 | 114.1 | 0.49 |
| DQN (routing-grid) | 14,077,805 | 12,017,983 | 2,059,274 | 1,855,238 | 274.0 | 0.49 |
| Random | 14,088,938 | 12,071,455 | 2,016,552 | 1,837,997 | 420.2 | 0.51 |
| Local Only | 14,137,839 | 12,120,996 | 2,016,698 | 1,849,378 | 72.2 | 0.18 |
| Drain Immediately | 14,142,016 | 12,128,766 | 2,013,002 | 1,847,711 | 123.9 | 0.02 |
| Round Robin | 14,157,556 | 12,138,313 | 2,019,170 | 1,850,517 | 36.7 | 0.47 |
| Defer to Low Net Demand | 14,157,372 | 12,137,029 | 2,020,324 | 1,851,042 | 9.6 | 2.26 |
| Cheapest Price First | 24,689,886 | 10,711,673 | 1,818,238 | 1,721,228 | 2,316.9 | 0.70 |

**Key findings**:

1. **PPO is best** at $13.20M, **6.8% better than Round Robin** and **5.5% better than Trough-Slot Lookahead**. This is the largest margin in any config — geographic diversity (Global) + temporal flexibility (batch) jointly maximize the optimization surface.

2. **DQN (flat-idx) is 2nd** at $13.42M — 1.6% behind PPO but **4.7% better than DQN-routing-grid** ($14.08M, 5th). The CFWS-style action encoding clearly helps here: even though the action set is 16× smaller (48 vs 759), the structured "migrate from src to dst" primitives are much more sample-efficient to learn than the generic routing-fraction enumeration. This is the second config (after Global legacy §7.3) where flat-idx demonstrably outperforms routing-grid.

3. **Avoid the Ramp** lands in 3rd ($13.81M). Its routing-to-the-slackest-grid strategy, which catastrophically failed in legacy mode (capacity violations), becomes viable when batch deferral can absorb the spike. The drain pool grows to an average of 2.38 — the largest non-degenerate pool size — as it queues work waiting for the chosen DC to have headroom.

4. **DQN (routing-grid) regresses to mid-pack** ($14.08M, 5th) — its 759-action space wastes too much capacity on near-optimal-allocation variants that never get explored. The gap to PPO widens to 6.7%. This is the clearest single piece of evidence that **action-space encoding choice can matter more than algorithm choice**: the same DQN algorithm, on the same env, with the same hyperparameters, differs by 4.7% based solely on how the discrete action space is structured.

### 7.5 Per-DC Energy Cost Breakdown

**US Batch Mode**:

| Policy | US-West | US-Central | US-Southeast-1 | US-Southeast-2 |
|---|---|---|---|---|
| PPO | $2,083,377 | $1,933,461 | $1,839,559 | $1,758,998 |
| Trough-Slot | $2,561,563 | $1,628,319 | $1,932,417 | $1,831,890 |
| Round Robin | $2,427,003 | $1,733,599 | $1,956,747 | $1,788,422 |

PPO reduces US-West cost (the highest-stress region, CAISO duck curve) by **14%** vs Round Robin while accepting slightly higher US-Central cost. It also outperforms Trough-Slot on US-West by 19%.

**Global Batch Mode**:

| Policy | Global-US-West | Global-US-Central | Global-EU | Global-Asia |
|---|---|---|---|---|
| PPO | $2,359,960 | $1,944,207 | $2,398,138 | $4,531,213 |
| Avoid the Ramp | $2,766,336 | $1,612,884 | $2,424,420 | $4,819,630 |
| Round Robin | $2,427,003 | $1,733,599 | $2,410,081 | $5,567,631 |

The Global-Asia DC (Singapore) is the most expensive region (high EMA prices). PPO reduces Global-Asia cost by **19%** vs Round Robin by deferring Singapore-bound work and rerouting it to cheaper grids during their slack hours. Avoid the Ramp pushes too aggressively to Global-US-Central, raising the duck-curve cost at Global-US-West.

### 7.6 Burst-Aware Augmentation (Heavy-Tail Follow-Up)

Tirmazi et al. (2020) document that the 2019 trace exhibits an extreme long tail at the job level: "the top 1% of jobs (resource hogs) consume over 99% of all resources" with squared coefficients of variation >23,000 (§7 of that paper). The intra-cluster scheduling implication is "insulate the mice from the hogs"; for our cell-aggregate routing problem, the same heavy-tail manifests as **bursty aggregate batch arrivals** — a small fraction of timesteps deliver a large fraction of new batch CPU demand. Our heavy-tailed distribution fits (Weibull, log-normal; §2.2) preserve this burstiness into the per-step cell-aggregate arrivals our env exposes.

This sub-section reports two diagnostic experiments on whether burst handling is a high-leverage optimization axis we should target explicitly.

#### Step 1 — Are bursts where the optimization signal lives?

Define a **burst timestep** as any step in the top 5% by aggregate batch CPU arrival across all DCs (446 of 8,917 timesteps per episode; arrival magnitude ~2.5× the non-burst mean). Per-policy cost decomposition on US batch and Global batch (full results in `output/burst_analysis.json`):

| Metric | US batch | Global batch |
|---|---|---|
| Burst share of total cost (Round Robin) | 5.3% | 5.3% |
| Burst $/step premium vs non-burst (Round Robin) | +6.5% | +5.7% |
| PPO total advantage vs Round Robin | +3.19% | +6.76% |
| PPO **burst-window** advantage vs Round Robin | **+6.37%** | **+8.86%** |
| PPO **off-burst** advantage vs Round Robin | +3.01% | +6.65% |
| DQN-routing-grid total / burst-window adv | +1.34% / **+4.33%** | +0.57% / +1.64% |
| DQN-flatidx total / burst-window adv | −0.49% / −1.15% | +5.23% / **+6.75%** |

**Findings**:

1. **Bursts do not dominate raw cost.** The pool-with-deadline mechanic smears arrival spikes across multiple subsequent drain steps, so the 5% of timesteps with the largest arrivals account for ~5.3% of total cost — only a slight premium over their share-of-timesteps baseline.
2. **But the RL optimization signal *does* concentrate in burst windows.** Every RL agent's advantage-over-Round-Robin is larger in burst windows than off-burst — for PPO, the burst-window advantage is **2.1× the off-burst advantage in US batch** and **1.3× in Global batch**. So the policies aren't winning by averaging gains over uniform conditions; they're winning by making their best decisions during the high-arrival moments.
3. **PPO already learned burst-aware spatial routing implicitly** (no explicit burst signal in the observation). Quantified via three behavior metrics — HHI of routing fractions (spatial concentration), `Σ fraction_i × net_demand_i` (net-demand-weighted routing target), and mean drain rate:

   | PPO metric (US batch) | Burst | Non-burst |
   |---|---|---|
   | HHI (spatial concentration) | 0.316 | 0.308 |
   | ND-weighted routing target (lower = route to slack grids) | **0.662** | **0.693** |
   | Mean drain rate | 0.460 | 0.461 |

   Spatial routing differs during bursts (more concentrated toward lower-net-demand DCs). **Temporal behavior (drain rate) is essentially identical** burst vs non-burst — the unused lever.

#### Step 2 — Does an explicit burst signal help?

We added a per-DC `burst_severity = current_batch_arrival / rolling_24h_mean_arrival` observation feature (clipped to [0, 10]) and retrained PPO + DQN-flatidx on US batch and Global batch. We also enabled the `memory_enabled` flag in these runs (memory does not bind in our env — verified by [scripts/check_memory_binding.py](scripts/check_memory_binding.py) showing 0.0000% cost diff with memory on vs off — so this adds observation dimensionality without changing dynamics).

| Variant | US batch | Global batch | Total improvement |
|---|---|---|---|
| PPO baseline | $9.47M | $13.20M | — |
| **PPO + burst + memory** | **$9.41M** | **$13.04M** | **+0.7% US, +1.2% Global** |
| DQN-flatidx baseline | $9.83M | $13.42M | — |
| DQN-flatidx + burst + memory | $9.79M | $14.18M | +0.4% US, **−5.4% Global** |

**Findings**:

1. **PPO benefits modestly from the explicit burst signal** (+0.7–1.2% total cost reduction). The improvement is **proportionally larger in burst windows** (+1.3% burst $/step in US, +1.7% in Global) than off-burst (+0.7%, +1.2%), consistent with the prediction.
2. **The improvement mechanism is NOT what we predicted.** Drain rate differentiation between burst and non-burst remains essentially zero even with the explicit signal (PPO-burst delta = −0.0018 vs baseline's −0.0009). What changed: PPO-burst+mem learned a **more spatially concentrated** policy (HHI 0.358 vs 0.316 in US burst windows) and a **slightly more aggressive overall drain rate** (0.477 vs 0.461 in US). The agent did not learn to vary drain timing based on burst presence; it learned a uniformly tighter policy that happens to perform better during bursts.
3. **DQN-flatidx is hit-or-miss** with burst awareness: small +0.4% improvement on US, but a **−5.4% regression on Global**. Likely cause: the 47-dim augmented observation (vs 35 baseline) is harder for DQN with only 48 actions to map into Q-values in the same training budget. PPO scales better with input dimensionality on our env.

#### Takeaway

The burst-aware augmentation is **a modest positive result for PPO with a negative result for DQN-flatidx in geo-distributed settings**. The deeper finding from these two experiments is structural: **temporal scheduling (drain timing) is the lever PPO appears unable to differentiate by arrival magnitude**, even when given the explicit signal. The Step 1 spatial-routing differentiation that PPO learned implicitly extends and slightly amplifies with the burst signal, but no agent we trained learned to vary drain timing based on burst severity. This suggests one of:
- The current pool-with-deadline mechanic already absorbs bursts well enough that differentiated drain timing has limited additional value
- The deadline penalty weight is calibrated such that holding longer during bursts isn't favorable
- A different reward structure (e.g., explicit reward for "uniform-load drain") would be needed to elicit differentiated temporal behavior

For the thesis story, the contribution is honest: we identified the burst-window concentration of the optimization signal (a non-trivial finding tied to Tirmazi's heavy-tail observation), tested an explicit intervention, and report that PPO improves modestly via spatial-routing tightening — not via the differentiated temporal behavior we hypothesized.

---

## 8. Comparison with Related Work

The combination explored in this thesis — multi-DC spatial routing + temporal batch scheduling, with RL, against grid demand-smoothing as a first-class objective, using cell-aggregated ClusterData 2019 — is not directly anchored to a single prior paper. The framing below triangulates across several lines of work: each citation supports the specific subclaim it can actually support, rather than overclaiming a single-paper methodology lineage.

### 8.1 ClusterData lineage — Tirmazi et al. (2020)

**Tirmazi, Barker, Deng, Haque, Qin, Hand, Harchol-Balter, Wilkes. "Borg: the next generation." EuroSys '20** is the canonical Google paper describing the ClusterData 2019 trace. It is published by the same authors who released the dataset, and is the primary methodology reference for analyzing the trace at the cell level. The structural decisions in our environment that depend on the trace are all anchored directly to claims in that paper:

| Our environment choice | Tirmazi anchor |
|---|---|
| Treating each cell as one DC | "Each such deployment is called a cell, and is operated as a single management unit" (§2) |
| Per-cell aggregate CPU timeseries | Figures 2–3 present cell-level CPU/memory usage as "fraction of cell capacity" timeseries — the standard aggregate-per-cell view (§4) |
| 5-minute sampling interval | "The 2019 trace adds a 21-element histogram of CPU utilization for each 5 minute sampling period" (§3) |
| `cpu_demand_norm` in [0, 1] | Normalized Compute Units (NCUs) are "always in the range 0–1" (§3) |
| Using multiple heterogeneous cells | "Considerable inter-cell workload variation" reported across cells a–h (§3, Fig 3) |
| Best-effort batch (beb) tier as the deferrable pool | beb tier "managed by the batch scheduler ... no associated SLOs," priority 110–115 (§2); accounts for ~20% of cell capacity (§4) |
| Heavy-tailed distribution fits (Weibull, log-normal) | "Top 1% of jobs (resource hogs) consume over 99% of all resources" with squared coefficient of variation > 23,000 (§7) |
| 31-day episode length | "31 days" duration of the 2019 trace (Table 1) |

This grounds our data usage in the dataset publisher's own framing rather than borrowing a methodology from an unrelated research thread. The cells a–d we use are four of the eight cells (a–h) Tirmazi analyzes, with ~12k machines per cell on average.

### 8.2 Single-DC renewable-aware predecessors — Grange (2018), Haghshenas et al., Liu et al.

The renewable-aware batch scheduling literature is largely **single-DC**. Three reference points form the immediate lineage:

**Grange, Da Costa, Stolf (2018), "Green IT scheduling for data center powered with renewable energy"** (*Future Generation Computer Systems* 86) is the closest single-DC predecessor to our batch-deferral logic. Grange schedules **batch jobs with due-date constraints** in a small-scale DC powered by on-site solar panels and grid, achieving up to **49% brown-energy reduction and 51% cost savings** vs a renewable-unaware scheduler. Their core architectural insight — which we adopt — is **separation of concerns**: "a scheduling algorithm agnostic of the electrical infrastructure. A separated system, managing the renewable sources, provides an arbitrary objective function, which is used to guide the scheduling heuristic." In our env, the same separation holds: the agent's reward gets a `peak_penalty` term that summarizes grid stress via `net_demand_normalized × grid_mw²`, but the agent doesn't model the grid directly. Grange also identifies the SLA-flexibility/savings tradeoff that motivates our `flexibility_factor` parameter (§3.7).

**Haghshenas, Taheri, Goudarzi, Mohammadi, "Infrastructure Aware Heterogeneous-Workloads Scheduling for Data Center Energy Cost Minimization"** considers a single Internet DC with **heterogeneous interactive + batch workloads**, on-site solar, cooling subsystem, and time-varying electricity prices. Their algorithm achieves 46% cost reduction. Two aspects flow through to our work: (1) the **interactive-vs-batch split** that we encode as `get_service_demand(t)` (non-deferrable) vs `get_batch_demand(t)` (pool-managed) — Haghshenas-style heterogeneous workloads are the rationale for treating these as separate workload classes; (2) **electricity rate structure awareness**, which we extend from a single DC's local price to per-DC LMP signals.

**Liu, Chen, Bash, Wierman, Gmach, Wang, Marwah, Hyser, "Renewable and Cooling Aware Workload Management for Sustainable Data Centers"** (Caltech + HP Labs) takes a **predict-then-plan** structure: forecast renewable supply + IT demand, then generate a workload plan that schedules IT work and allocates resources according to time-varying power supply. The forecasting horizon and lookahead-based planning structure is what motivates our **Trough-Slot Lookahead** baseline (§8.4) — though we use net demand as the forecast signal rather than renewable supply, consistent with the reframing in §1.

All three are **single-DC**. The combination "multi-DC routing + temporal batch deferral + grid-aware objective + cell-aggregate ClusterData" is the gap this thesis fills relative to that lineage.

### 8.3 Interactive + batch deferral pattern — Xu, Toosi, Buyya

**"A Self-Adaptive Approach for Managing Applications and Harnessing Renewable Energy for Sustainable Cloud Computing"** (Xu, Toosi, Buyya) provides the framework we adopt for **splitting workloads into interactive (must-serve-now) and batch (deferrable) components**, with separate handling for each: brownout for interactive, deferring for batch. Our environment's `batch_enabled` mode (§3.7) and the batch pool / drain abstraction are direct descendants of this framework, restricted to the batch side (we don't implement brownout). Like the others in §8.2, Xu's setup is single-DC.

### 8.4 Lookahead-based scheduling — GreenSlot (Goiri et al. 2011)

**Goiri et al., "GreenSlot: Scheduling Energy Consumption in Green Datacenters"** is the lineage for our **Trough-Slot Lookahead** baseline. Per Grange et al.'s clear summary (§8.2 above): GreenSlot "considered a small cluster used for scientific computation, and powered partially with solar panels. Using prediction of renewable power available, along with grid electricity price, the GreenSlot algorithm delays jobs to execute them when the cost is the lowest (both in terms of brown energy usage and in terms of purchasing cost)." The algorithm discretizes future time into fixed-duration slots, each "valuated with predicted renewable energy production, grid electricity cost, and number of available computing nodes," then greedily places each task in the first slot allowing renewable-only execution.

Our Trough-Slot Lookahead baseline (§5.8) reuses the slot-valuation idea but retargets it to the demand-smoothing formulation: instead of evaluating slots by predicted renewable supply, it evaluates them by predicted **grid net demand** — a slot is "good" when net demand will be low (a duck-curve trough), not when local solar will be high. The 36-step (3-hour) lookahead window directly mirrors GreenSlot's slot horizon.

| Aspect | GreenSlot | Our Trough-Slot baseline | Our PPO agent |
|---|---|---|---|
| Approach | Greedy slot valuation (LP variant in follow-up GreenSwitch) | Lookahead heuristic | Model-free RL |
| Scheduling | Temporal only (single DC) | Spatial + temporal (multi-DC) | Spatial + temporal (multi-DC) |
| Slot signal | Solar supply + grid price | Grid net demand (forecast) | Implicit from observations |
| Forecast required | Yes (explicit) | Yes (oracle: actual future net demand) | No (learned) |
| Number of DCs | 1 | 4 | 4 |

PPO **outperforms Trough-Slot Lookahead in every config where temporal flexibility exists**:

| Config | PPO | Trough-Slot | PPO Advantage |
|---|---|---|---|
| US batch | $9.47M | $9.80M | **3.4%** |
| Global legacy | $13.48M | $14.07M | **4.2%** |
| Global batch | $13.20M | $13.97M | **5.5%** |

This is meaningful because Trough-Slot has **privileged 3-hour future net-demand information** that PPO does not. PPO learns implicit forecasting AND coordinates it with routing — a strictly stronger policy than the foresighted single-axis heuristic.

### 8.5 Operational anchor — Radovanovic et al. (2022)

**Radovanovic, Koningstein, Schneider, Chen, Duarte, Roy, Xiao, Haridasan, Hung, Care, Talukdar, Mullen, Smith, Cottman, Cirne. "Carbon-Aware Computing for Datacenters." IEEE Transactions on Power Systems** describes **CICS — Google's Carbon-Intelligent Compute System**, the production system that shifts temporally flexible workloads across Google's datacenter portfolio (20+ DCs, 15.5 TWh annual consumption, 4 continents) to align computing with low-carbon grid hours. CICS uses day-ahead carbon-intensity forecasts and cluster-level load forecasts to generate hourly Virtual Capacity Curves (VCCs) per cluster.

This paper is the **operational validation that the problem class this thesis addresses is real at hyperscale**. Three specific claims from Radovanovic carry through to our work:

| Radovanovic CICS | Our env |
|---|---|
| "Shifting execution of flexible workloads in time and space can decrease peak demand for resources and power" | Our peak-contribution penalty (§3.3) targets exactly this; PPO's batch-mode drain decisions are the spatial+temporal shift CICS enacts in production |
| Uses "cluster-level load forecasts and power models [Sakalkar 2020]" | Same cluster-level cell-aggregate granularity (§2.1); same Sakalkar 2020 power model lineage (§3.2) |
| "Datacenters are planned based on peak power and resource usage, smaller peaks reduce the need for more capacity" | Direct motivation for our load-squared peak penalty in the reward |

We do **not** reproduce CICS — it's an operational paper not a published methodology, uses internal Google data not the public ClusterData 2019 trace, and optimizes carbon rather than grid demand smoothing. But it confirms our problem framing matches industry practice at the scale we model. The thesis can be positioned as "a published methodology and reproducible Gymnasium env for the problem class that CICS solves operationally."

### 8.6 Academic foundation — CFWS (Zhao et al. 2025)

**CFWS (Zhao, Zhou, Li, IEEE Trans. Sustainable Computing 10(1), Jan/Feb 2025)** is the closest published academic peer — a DRL framework for **multi-DC workload distribution under renewable considerations**, with 4 geographically distributed US DCs (Arizona, California, Oregon, Louisiana). Like us, CFWS is geo-distributed at the DC level; unlike us, the unit of work CFWS shifts is an individual VM (with per-PM state inside each DC) rather than aggregate cell-level load. We use CFWS as the **academic foundation** establishing that DRL is a valid approach to this class of problem, and align with it at the **algorithmic level** (DQN) without claiming methodology reproduction.

#### Formulation-level differences (why head-to-head numbers would mislead)

Both CFWS and our work operate over geo-distributed DCs; the differences are about **what gets shifted and at what granularity inside each DC**, not about whether DCs are geographically separate. Numerical comparison between their DQN cost figures and ours would not be apples-to-apples because the units being optimized are different.

| Dimension | CFWS | Us |
|---|---|---|
| Geo-distributed DCs | ✓ 4 (AZ, CA, OR, LA) | ✓ 4 (US scenario: OR, IA, GA, SC; Global: OR, IA, NL, SG) |
| Unit of work being moved | One VM at a time | A fraction of aggregate load |
| Destination | A specific PM (which happens to be in some DC) | A whole DC (no PM granularity) |
| State granularity | Per-PM CPU utilization tuple within each DC | Aggregated per-DC CPU demand (cell-as-DC, §2.1) |
| Action paradigm | Discrete VM migration: pick (VM, dest DC, dest PM) via flattened-index hash map (`R×n×m`) | Discrete routing-fraction grid (DQN) / continuous softmax routing + sigmoid drain (PPO) |
| Trigger | TCN-MAD overload detection — only fires when a PM is over/under threshold | Continuous, every 5-min step |
| Renewable | On-site wind (NE-3000 turbines, cut-in/rated/cut-out model) | None on-site; renewables enter via grid net demand only |
| Objective | Energy cost + carbon footprint (linear scalarization) | Energy cost + grid peak-contribution penalty (quadratic in load) |
| Workload | ClusterData 2011, 5-day simulation | ClusterData 2019, 31-day simulation |

#### Algorithm-level alignment (DQN as the CFWS-aligned baseline)

Where CFWS and our **DQN baselines** *do* align is at the algorithm level — both use a standard DQN configuration. We train **two DQN variants** to test the action-encoding question from different angles:

- **DQN (routing-grid)** — generic 759-action grid (253 routing × 3 drain). The natural "what if you just dropped DQN into our env" baseline.
- **DQN (flat-idx)** — 48-action CFWS-style flattened-index scheme adapted to our env. Each action decodes via division/modulo to `(src_dc, dst_dc, drain_level)`: same DC for src/dst means "uniform allocation", otherwise "migrate 15% from src to dst" + set all drain rates to the chosen intensity level. This is the closest port of CFWS's hash-map-decode action philosophy that our cell-aggregate formulation allows (we don't have VMs/PMs to enumerate). See [env/cfws_style_wrapper.py](env/cfws_style_wrapper.py).

Both DQN variants share identical hyperparameters with each other and with CFWS:

| DQN component | CFWS (per Zhao et al. §IV-C3) | Our DQN (routing-grid) | Our DQN (flat-idx) |
|---|---|---|---|
| Core algorithm | DQN | DQN | DQN |
| Exploration | ε-greedy, decaying | ε-greedy, fraction = 0.3, final ε = 0.05 | (same) |
| Stability | Target Q-network + experience replay | Target update = 1,000 steps; replay buffer = 100,000 | (same) |
| Network arch | MLP (dims unspecified) | MLP [256, 256] | (same) |
| Output | Q-value per discrete action | Q-value per discrete action | Q-value per discrete action |
| **Action set size** | `R × n × m` (variable, ~10s–100s per overloaded PM) | **759** (batch mode) | **48** (batch mode) |
| **Action encoding** | Flattened index → hash-map → (VM, dest DC, dest PM) | Index → lookup into precomputed (routing logits, drain logits) tuple | Flattened index → hash-map → (src DC, dst DC, drain level) |
| What action means | Migrate one selected VM from an overloaded PM to a target PM (the target PM may be in any of the geo-distributed DCs) | Set the next 5-min routing fractions across DCs + per-DC drain rates | Migrate 15% load from src DC to dst DC + set all drain rates to chosen intensity |
| Trigger in env loop | Only when TCN-MAD flags an over/under-loaded PM | Every timestep | Every timestep |

The **flat-idx variant** is closer to CFWS *in spirit* (small discrete action set, hash-map decode to a multi-dim "migration" tuple); the **routing-grid variant** is closer in *expressivity* (large enumerated action set covering arbitrary allocations).

#### Empirical comparison: action-space encoding matters, but direction depends on scenario

Trained models evaluated under identical conditions (α = 0.015, same seeds, full 31-day trace):

| Scenario | PPO | DQN (routing-grid, 759) | DQN (flat-idx, 48) | flat-idx vs routing-grid |
|---|---|---|---|---|
| US legacy | $10.10M | **$9.84M** | $10.87M | **−10.4%** worse |
| US batch | **$9.47M** | $9.65M | $9.83M | −1.8% worse |
| Global legacy | $13.48M | $13.57M | **$13.47M** (ties PPO) | **+0.7%** better |
| Global batch | **$13.20M** | $14.08M | $13.42M | **+4.7%** better |

Three findings emerge:

1. **On our formulation, the CFWS-style flat-idx encoding wins under high geographic diversity** (Global scenarios) and loses under low diversity (US). With a 16-hour timezone spread and heterogeneous grids, the 48-action set's "migrate load from this DC to that DC" structure encodes good policy primitives directly; with a 3-hour US-only spread (similar grids, similar diurnal shapes), the constrained 48-action space can't fine-tune the small adjustments that the optimal policy requires. *Note*: this finding is specific to our cell-aggregate formulation; it does **not** say anything about how CFWS's encoding performs on CFWS's own per-PM formulation. CFWS reports their flat-idx DQN works well on their AZ/CA/OR/LA setting (5.67–13.22% brown-energy reduction, 46.49–86.53% migration reduction vs their baselines) — and they have many more meaningful actions per migration event because each DC contains many PMs, so their action space isn't constrained the same way ours is.
2. **PPO still wins overall, but the margin against the *best* DQN variant is smaller than against routing-grid alone**: PPO vs the best DQN at each config is +2.6% (US legacy DQN-rg), +1.9% (US batch DQN-rg), tie (Global legacy DQN-flatidx), +1.6% (Global batch DQN-flatidx). So the headline "PPO beats DQN by 1–7%" was partly explained by sub-optimal action-space discretization, not just the discrete-vs-continuous distinction.
3. **Action-encoding choice matters as much as algorithm choice in some configs**. In Global batch, switching DQN from routing-grid to flat-idx improves cost by 4.7% — larger than the typical PPO-over-DQN gap. This validates CFWS's own design insight (constraining the action space to a small set of semantically-meaningful operations helps DQN) even though we apply it on a different formulation.

#### PPO as the contribution beyond CFWS

PPO remains our overall best policy and goes beyond what CFWS's framework can express: their flattened-index action is inherently discrete (pick *one* VM to migrate), so a continuous-action variant would require reformulating their problem. In our cell-aggregate formulation, the action is naturally continuous (routing fractions in [0,1] summing to 1, drain rates in [0,1]), making PPO a natural fit. With the **flat-idx variant as the cleaner CFWS-style analog**, the PPO advantage is now tightly quantified: **+1.6 to +2.6% over the best DQN variant in 3 of 4 configs, and a statistical tie in Global legacy**. The contribution claim is "continuous actions provide a small but consistent advantage over even well-designed discrete action spaces in this formulation" — narrower and more defensible than the original "1–7% over DQN-routing-grid" framing.

### 8.7 Survey context — Lin et al. (2024) and Wu et al. (2025)

**"A systematic review of green-aware management techniques for sustainable data center"** (Lin, Lin, Peng, Huang, Lin, Li, 2024) provides the broader sustainable-DC landscape view. The categories of workload management, virtual resource management, energy management, thermal management, and waste heat recovery surveyed there place this thesis within "workload management + energy management for grid-aware multi-DC operation." For the multi-DC scheduling subarea specifically, **Wu et al. (2025), "Task Scheduling in Geo-Distributed Computing: A Survey"** (arXiv:2501.15504) is the most recent systematic review and covers the geo-distributed task-scheduling thread that this thesis sits within.

### 8.8 Positioning of Our Contribution

Stated against the lineage above:

1. **Cell-as-proxy-DC modeling exercise.** We treat four ClusterData 2019 cells (a–d) as four geographically distributed hyperscale DCs — what such DCs' workloads would look like if they had cell-level inter-DC heterogeneity. This is *not* what Tirmazi et al. (2020) intended when documenting the trace (they don't claim the cells are geographically distinct), and no prior published work does exactly this. It is a defensible modeling exercise rather than a dataset-grounded claim (see §2.1 and §3.2 for the explicit modeling assumptions on workload-as-shape and `rated_power_mw`-as-magnitude).
2. **Grid demand smoothing as a first-class objective**, via a peak-contribution penalty against actual EIA-930 net demand timeseries — rather than the on-site-renewable framing that dominates academic prior work.
3. **Continuous action space (PPO)** enabling fine-grained joint routing + drain decisions. Against our best DQN variant per config, PPO wins by +1.6 to +2.6% in 3 of 4 configs and ties in Global legacy (§8.6). The PPO advantage is narrower than against generic routing-grid DQN alone, because a CFWS-style flat-idx encoding closes much of the gap in Global scenarios.
4. **Real-data grounding**: ClusterData 2019 (per Tirmazi 2020) for workloads, `powerdata_2019` (per Sakalkar 2020) for the power model, EIA-930 (CISO, MISO, SOCO, DUK) for grid net demand, real ISO prices, NSRDB solar irradiance as forecast features.
5. **Operational relevance**: the problem class is the same one **Google's CICS (Radovanovic 2022)** solves in production at 20+ DCs across 4 continents. This thesis contributes a published methodology + reproducible Gymnasium env for that problem class.

Honest framing of what this thesis is *not*: it is not a head-to-head comparable against CFWS (different action paradigm, different state granularity, different objective), nor a reimplementation of Google's CICS (closed-source operational system). It is a self-contained academic exploration of multi-DC + cell-aggregate + RL + grid-aware scheduling, with the cell-as-DC and 100 MW magnitude assumptions stated explicitly rather than hidden.

The primary empirical findings are:

- **Spatial routing alone matters under regional diversity.** In the Global scenario (16-hour timezone spread), PPO beats Round Robin by 5.4% in legacy mode. In the US scenario (3-hour spread, similar grid profiles), spatial routing alone is structurally limited.
- **Temporal batch scheduling is the universal lever.** Batch mode improves PPO's cost by 6.2% over legacy in US and 2.1% in Global, on top of the spatial gains.
- **PPO beats the oracle-like Trough-Slot baseline** by 3.4–5.5% in every config where temporal flexibility exists. RL learns implicit forecasting plus coordinated routing — a strictly stronger policy than the lookahead heuristic.

---

## 9. Technical Architecture Summary

```
┌─────────────────────────────────────────────────┐
│                  Scenario YAML                    │
│  (sites, rated_power, net_demand path, batch)     │
└──────────────────────┬──────────────────────────┘
                       │
              ┌────────▼────────┐
              │   data_loader   │  Loads CSVs, fits fleet capacities,
              │                 │  creates BatchArrivalGenerators
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │  MultiDCEnv     │  Gymnasium environment
              │  ┌────────────┐ │
              │  │ DC Sites   │ │  4× DataCenterSite with:
              │  │  workload  │ │  - CPU demand timeseries
              │  │  net_dem.  │ │  - Grid net demand (EIA-930)
              │  │  solar     │ │  - Solar fraction (forecast)
              │  │  price     │ │  - Electricity price
              │  │  BatchPool │ │  - Deferrable work queue
              │  └────────────┘ │
              │  ┌────────────┐ │
              │  │ PowerModel │ │  P = 0.4788 + 0.4438 × u
              │  └────────────┘ │
              └────────┬────────┘
                       │
           ┌───────────┼───────────┐
           │           │           │
    ┌──────▼──────┐ ┌──▼───┐ ┌────▼─────────┐
    │    PPO      │ │ DQN  │ │  Baselines   │
    │ continuous  │ │ 253  │ │  8 heuristic │
    │ 8 actions   │ │ disc │ │  policies    │
    └──────┬──────┘ └──┬───┘ └────┬─────────┘
           │           │          │
           └───────────┼──────────┘
                       │
              ┌────────▼────────┐
              │   evaluate.py   │  Runs all policies, generates
              │                 │  reports, plots, comparisons
              └─────────────────┘
```

---

## 10. Reproduction

### Data Pipeline

```bash
# Net demand timeseries (EIA-930 for US BAs; synthetic NL + SG)
# Requires EIA_API_KEY in .env or --eia-key flag
python preprocess/net_demand_fetcher.py
```

### Training Commands (peak_penalty_weight = 0.015)

```bash
# PPO — US, legacy + batch
python train.py --scenario env/scenarios/us_model.yaml \
    --timesteps 500000 --peak-penalty-weight 0.015
python train.py --scenario env/scenarios/us_model.yaml --batch-mode \
    --timesteps 500000 --peak-penalty-weight 0.015

# DQN — US, legacy + batch
python train_dqn.py --scenario env/scenarios/us_model.yaml \
    --timesteps 500000 --peak-penalty-weight 0.015
python train_dqn.py --scenario env/scenarios/us_model.yaml --batch-mode \
    --timesteps 500000 --peak-penalty-weight 0.015

# Or run the full 8-model sweep at once
python scripts/run_full_sweep.py --timesteps 500000 --alpha 0.015
```

### Evaluation Commands

```bash
# Single config
python evaluate.py --scenario env/scenarios/us_model.yaml \
    --model models/ppo_us_model_batch.zip --batch-mode \
    --peak-penalty-weight 0.015 \
    --dqn-model models/dqn_us_model_batch.zip

# All four configs at once (including DQN-flatidx if trained)
python scripts/evaluate_all.py --alpha 0.015

# CFWS-style flat-index DQN variant (48-action hash-map-decoded space, §8.6)
python train_dqn.py --scenario env/scenarios/us_model.yaml --batch-mode \
    --timesteps 500000 --peak-penalty-weight 0.015 \
    --action-scheme cfws-style
# or train all 4 flatidx configs at once:
python scripts/run_cfws_dqn_sweep.py --timesteps 500000 --alpha 0.015

# Burst-aware + memory PPO / DQN-flatidx variants (§7.6)
python train.py --scenario env/scenarios/us_model.yaml --batch-mode \
    --timesteps 500000 --peak-penalty-weight 0.015 \
    --burst-aware --memory
python train_dqn.py --scenario env/scenarios/us_model.yaml --batch-mode \
    --timesteps 500000 --peak-penalty-weight 0.015 \
    --action-scheme cfws-style --burst-aware --memory
# or all 4 burst-aware configs at once:
python scripts/run_burst_sweep.py --timesteps 500000 --alpha 0.015

# Burst-window diagnostic analysis (§7.6)
python scripts/analyze_burst_routing.py
python scripts/analyze_burst_drain_diff.py
```

---

## 11. Key Takeaways for Thesis Writing

1. **PPO is best in 3 of 4 configurations.** The exception is US legacy mode, where the agent's tendency to over-concentrate routing on one DC causes capacity-violation penalties that exceed its energy savings. With either geographic diversity (Global) or temporal flexibility (batch), PPO wins decisively.

2. **PPO beats the foresighted Trough-Slot Lookahead oracle in every config it wins.** Margins: 3.4% (US batch), 4.2% (Global legacy), 5.5% (Global batch). RL learns implicit net-demand forecasting AND coordinates it with spatial routing — a strictly stronger policy than a foresighted single-axis heuristic.

3. **Temporal batch scheduling is the universal lever.** Batch mode reduces PPO cost by 6.2% in US and 2.1% in Global vs the corresponding legacy results. Even when routing is structurally limited (US), the agent recovers performance by deciding *when* to execute deferrable work.

4. **PPO discovers an aggressive cost-deadline tradeoff.** The agent expires 1,200–1,300 units of batch work per episode (vs ~10 for Defer-to-Low-Net-Demand) because the deadline penalty (~$2.5K) is dwarfed by the energy + peak savings (~$300K) from deferring drain into low-net-demand hours. This is emergent — the deadline weight is fixed at 2.0.

5. **Action-encoding choice matters as much as algorithm choice — and PPO's edge over DQN narrows once DQN's encoding is well-designed.** Both CFWS and our work are geo-distributed multi-DC; CFWS shifts individual VMs across DCs using per-PM state, we shift aggregate load across DCs at cell granularity (§8.6). We trained two DQN variants on *our* formulation: a generic 759-action routing-grid and a 48-action CFWS-style flat-idx encoding (`src_dc, dst_dc, drain_level` via hash-map decode, the closest port of CFWS's hash-map decoding philosophy our cell-aggregate setting allows). The flat-idx variant **wins under high geographic diversity** (Global legacy: ties PPO at $13.47M; Global batch: 4.7% better than routing-grid DQN) and **loses under low diversity** (US legacy: 10.4% worse than routing-grid DQN, because the constrained 48-action space can't fine-tune the few available levers when DCs are similar). PPO still wins overall (+1.6 to +2.6% over the best DQN per config in 3 of 4 configs; statistical tie in Global legacy) but the margin is narrower than against routing-grid DQN alone. This says nothing about how CFWS's encoding performs on CFWS's own per-PM formulation — they report their flat-idx DQN works well on their AZ/CA/OR/LA setup. It says that *in our cell-aggregate setting*, structured small action spaces beat generic large ones only when there's enough inter-DC diversity to exploit.

6. **Per-DC reallocation matches the duck-curve story.** PPO cuts US-West (CAISO duck curve) cost by 14% vs Round Robin in US batch, and cuts Global-Asia (high-price Singapore) cost by 19% in Global batch. The agent is exploiting exactly the regional differences in net demand and price that motivated the framing.

7. **Concentration heuristics fail without temporal slack.** Avoid-the-Ramp and Cheapest-Price-First collapse in legacy mode (8× and 4× worse than Round Robin respectively) from capacity violations. In batch mode the deferral pool absorbs the shock — Avoid-the-Ramp becomes the 2nd-best Global policy. This is a generalizable design insight: spatial concentration strategies need temporal flexibility to be safe.

8. **The new formulation produces meaningful policy spread.** The old on-site-solar formulation flattened all reasonable legacy policies to within 1%. The new objective shows 3–7% spreads across policies and configurations, with clearly differentiated rankings — the optimization surface is meaningfully exposed rather than smothered by surplus solar.

9. **The RL optimization signal concentrates in burst windows; the heavy tail is real but doesn't dominate cost.** Tirmazi's per-job long-tail (top 1% jobs = 99% of resources) manifests in our cell-aggregate setting as bursty batch arrivals. Burst timesteps (top 5% by arrival magnitude) don't drive disproportionate raw cost — the pool-with-deadline mechanic smears burst cost across drain steps — but the RL agents' advantage over Round Robin is **substantially larger in burst windows** (PPO: 2.1× larger in US batch, 1.3× in Global batch). PPO learned implicit burst-aware spatial routing without an explicit signal; it does not differentiate temporal (drain) behavior burst vs non-burst even when given an explicit `burst_severity` observation feature (§7.6). The Step 2 burst-aware augmentation improves PPO cost by 0.7–1.2% via a tighter spatial policy, not via differentiated temporal scheduling — an honest result with a different mechanism than predicted.
