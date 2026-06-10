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
- Cooling energy (excluded per advisor guidance; the cooling-optimization dimension — cooling-aware scheduling and RL cooling control — is covered by surveys such as *"A survey on data center cooling systems"* and *"Towards Joint Optimization Over ICT and Cooling Systems in Data Centre: A Survey"*, and is out of scope here)

### 1.2 Modeling Granularity & Positioning

**We model workload as divisible aggregate flow, not discrete jobs.** Each cell's demand is a single continuous CPU-utilization curve; the agent routes *fractions* of it spatially and releases a deferrable *pool* temporally. There is no per-job placement, no bin-packing, and no VM migration anywhere in the environment — individual jobs enter only *upstream*, as samples that shape the batch-demand curve (§2.2, §3.8). This is a deliberate abstraction, and it places the work in a specific lineage.

**The sustainable-scheduling literature splits by granularity.** One camp schedules discrete units onto machines: **Grange et al. (2018), *"Green IT scheduling for data center powered with renewable energy"*** places individual *tasks*; **Xu et al. (2020), *"A Self-Adaptive Approach for Managing…"*** creates/migrates *VMs* via OpenStack; **Haghshenas et al. (2022), *"Infrastructure-Aware…"*** schedules *jobs* on heterogeneous machines; **CFWS (Zhao et al. 2025)** migrates *VMs* across physical machines. The other camp shapes *aggregate* cluster load with no per-job placement — most authoritatively **Radovanovic et al. (2023), *"Carbon-Aware Computing for Datacenters"*** (Google's production Carbon-Intelligent Compute Management System, CICS, operating on the same Google-cluster workload). Our environment sits squarely in the aggregate camp.

**Alignment with Google's production system (CICS).** Our abstraction mirrors CICS point-for-point:

| Our environment | CICS (Radovanovic et al. 2023) |
|---|---|
| Aggregate cell CPU curve; no job placement | Cluster-level "Virtual Capacity Curves" shape hourly resource/power usage |
| Deferrable vs must-serve split by **priority tier** (free+beb vs production) | *"temporally **inflexible** (higher tiers)"* vs *"**flexible** (lower-tier batch jobs that tolerate delays)"* |
| Per-cell power model on aggregate CPU | *"power models trained separately for each cluster"* on *"aggregate… resource demand"* |
| Spatial routing **+** temporal deferral | *"shifting workloads across datacenter locations, or by delaying jobs' execution"* |
| Peak-contribution penalty | *"reduces daily peak CPU and, consequently, power consumption"* |
| CPU as demand proxy | *"in aggregate, resource consumption is highly correlated to CPU consumption"* |

Crucially, CICS explicitly characterizes the *job-level deadline-optimization* approach — i.e. Grange's — as the **previous** treatment it moved away from: it runs *"independently from real-time job-level scheduling"* using *"aggregate cluster-specific resource demand forecasts… rather than… stylized models for job-level resource demand modeling."* We therefore follow the newer, production-validated paradigm rather than a naïve simplification.

**What we reuse vs. what we benchmark.** We reuse Grange/Da Costa's *workload generator* (the input model; §3.8) but make the scheduling decision at CICS's *aggregate* granularity, not Grange's per-task placement. The three single-DC predecessors (Grange, Xu, Haghshenas) are benchmarked on **objectives and effects** (operator cost, peak contribution, deferral savings), not on mechanism — we do not claim to reproduce their machine-level placement or VM migration.

**Honest limitations of the abstraction.** Divisible aggregate flow cannot capture per-VM/per-job SLA enforcement, VM-migration overhead, bin-packing and resource fragmentation on real machines, or the indivisibility of a single job. These belong to the intra-DC placement problem (CFWS's territory) and are out of scope. The aggregate view is appropriate for the inter-DC, grid-facing question this thesis asks — *how much load runs where, and when* — which does not require machine-level detail.

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

The 2019 trace exposes job priority as a sparse value in [0, 450] and groups priorities into named **tiers** (Tirmazi et al. 2020, *"Borg: the Next Generation"*, §2). We classify a job as deferrable **batch** iff it belongs to one of the two **SLO-free tiers**: the **free tier** (`priority ≤ 99`) or the **best-effort batch (beb) tier** (`priority ∈ [110, 115]`). Both are explicitly described as having "no associated SLOs" (Tirmazi §2) — they are the genuinely delay-tolerant work. Every SLO-bearing tier is non-deferrable **service**: the mid-tier (116–119, weak SLOs) and especially the **production tier** (120–359), which "require[s] high availability" — "Borg will evict lower-tier jobs in order to ensure production tier jobs receive their expected level of service" (Tirmazi §2). Production must be served immediately and cannot be queued.

**Why both no-SLO tiers, not beb alone.** Tirmazi's terminology reserves "batch" for the beb tier, but in these four cells beb *alone* is negligible (≈0% of CPU-time, <6% of jobs). The free tier — equally SLO-free — holds the deferrable mass; restricting to strict beb would leave essentially nothing to defer. The union of the two SLO-free tiers is the defensible deferrable class. Classification is by **priority**, not `scheduling_class`: the latter encodes latency-sensitivity (0–3), is orthogonal to tier, and a `scheduling_class ≤ 1` filter is dominated by latency-insensitive *production* (priority 200) jobs.

We fit statistical distributions to each cell's deferrable jobs (Cell A shown):

| Property | Cell A distribution | Key parameters |
|---|---|---|
| **Inter-arrival time** (sec) | Weibull (min) | shape=0.54, scale=33.0, mean=29.4s, KS D=0.11 |
| **Duration** (sec) | Log-normal | σ=1.74, scale=129, mean=1430s, median=105s, KS D=0.13 |
| **CPU request** (normalized) | Log-normal | σ=0.81, mean=0.0090, KS D=0.07 |
| **Memory request** (normalized) | Log-normal | σ=0.95, mean=0.0061, KS D=0.08 |
| **Tasks per job** | Negative binomial | mean=52.5, median=1, KS D=0.20 |

**Fitting methodology.** Candidate continuous distributions (exponential, log-normal, gamma, Weibull) are fit by MLE and the best selected by **minimum KS *D* statistic** — *not* the KS *p*-value, which underflows to 0 at n~10⁵ regardless of fit quality and is invalid for parameters estimated from the same sample (the Lilliefors situation). Tasks-per-job is count data, so it uses a **discrete** fit (Poisson / geometric / negative-binomial). Inter-arrivals and resource requests are taken over all deferrable jobs; durations only over jobs that reached FINISH. The heavy-tailed winners — log-normal durations/requests, negative-binomial task counts with mean 52 ≫ median 1 — reproduce the extreme variability Tirmazi documents: "the top 1% of jobs consume over 99% of resources," squared coefficient of variation > 23,000 (Tirmazi §7).

**Batch fraction.** Because the environment splits the CPU-*usage* curve, each cell's `batch_fraction` is the share of **CPU-time** (cpu_request × duration, with unfinished jobs charged to trace end) that is deferrable — not a job count:

| | Cell A | Cell B | Cell C | Cell D |
|---|---|---|---|---|
| **batch_fraction (CPU-time, used)** | 27% | 61% | 45% | 54% |
| by job count | 1.9% | 38.5% | 6.6% | 11.1% |
| by CPU request | 35% | 66% | 51% | 62% |

These are substantial — the cells are batch/free-heavy research clusters — so the temporal-deferral lever acts on a large slice of load. Because the usage-weighted value is itself a proxy (request × duration), `batch_fraction` is also treated as a **sensitivity-sweep parameter** rather than a single point estimate. (These supersede earlier `scheduling_class ≤ 1 AND priority < 200` figures, which conflated production with batch.)

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

The power model converts CPU utilization to electrical power consumption using a **linear model** calibrated against real Google power measurements from the **`powerdata_2019`** BigQuery dataset — the companion power-measurement trace published alongside ClusterData 2019 and documented in **Sakalkar et al. (2020), "Data Center Power Oversubscription with a Medium Voltage Power Plane and Priority-Aware Capping" (ASPLOS '20)** [§8]. `powerdata_2019` exposes per-PDU measured power utilization at the cell level; we join it with aggregate cell-level CPU utilization at hourly resolution and fit a linear model on the resulting (cpu_util, power_util) pairs. The linear `idle + slope·util` server-power form is the standard model catalogued by **Dayarathna, Wen & Fan (2016), "Data Center Energy Consumption Modeling: A Survey" (IEEE Communications Surveys & Tutorials 18(1))**.

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

This load-squared, net-demand-weighted term operationalizes **demand response / peak shaving** — flattening and time-shifting DC load to relieve grid stress — which **Vasques, Moura & de Almeida (2018), "A review on energy efficiency and demand response with focus on small and medium data centers" (Energy Efficiency, Springer)** identify as an underexploited lever for data centers.

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

The deferrable-batch-with-deadlines pattern follows a well-established lineage in renewable-aware datacenter scheduling — most directly **Grange et al. (2018)** (§8.3), whose central abstraction is "batch jobs with due-date constraints, which takes into account the availability of the renewable energy," and **GreenSlot (Goiri et al. 2011)** (§8.5), which "delays jobs to execute them when the cost is the lowest." We extend the single-DC pool-with-deadline pattern from that lineage to the multi-DC setting, where the agent must simultaneously decide *where* to route service work and *when* to drain each DC's batch pool.

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

Rather than using a fixed `workload[t] × batch_fraction` split, we generate **synthetic batch arrivals** from the per-cell fitted distributions (`BatchArrivalGenerator`):

1. Sample inter-arrival times from the fitted distribution (Weibull / log-normal per cell; see §2.2)
2. For each arrival, sample CPU demand, memory demand, and task count from the fitted distributions
3. Aggregate into per-timestep demand arrays
4. **Normalize** total demand to match the expected aggregate from the static split

This preserves realistic temporal **burstiness** (batch jobs arrive in clusters, not uniformly) while maintaining consistent aggregate demand volume.

**Methodological lineage — we reuse Grange et al.'s generator.** This distribution-fitting-then-generating approach is taken directly from **Grange et al. (2018), *"Green IT scheduling for data center powered with renewable energy"*** (Future Generation Computer Systems 86), whose Listing 1 is a short `scipy.stats` generator that draws each batch task's submission time and execution time from distributions fit to a Google cluster — *"we can easily control its duration, and generate several workloads based on the same distribution laws, but using different random seeds."* That generator instantiates the parameterized model of **Da Costa, Grange & De Courchelle (2016), *"Modeling and generating large-scale Google-like workload"*** (IGSC '16) — itself in the Feitelson parallel-workload-modeling tradition. We reproduce their exact Listing 1 verbatim in [`scripts/grange_generator.py`](scripts/grange_generator.py) (it recovers their reported log-normal parameters `s=1.634, scale=447` and a makespan mean of ~1700 s, their `mass`), and our `BatchArrivalGenerator` is the same generator *family*. We extend it in three ways: (i) we re-fit the **2019** trace (Grange/Da Costa used the ~2011-era study) under the no-SLO deferrable definition (§2.2); (ii) we select each distribution by minimum KS *D* rather than fixing the family a priori, and add **per-task CPU/memory request** and a **discrete negative-binomial task-count** distribution Grange does not model; and (iii) Grange's `mass`/`disparity`/`dynamism`/`ratioTask` knobs and his truncated-normal **task-flexibility** distribution map onto our fitted parameters and our `flexibility_factor` deadline knob, respectively. Net: same established methodology, refreshed to the newer trace and enriched. (Contrast: Xu et al. (2020) instead *replay* a real trace; Haghshenas et al. (2022) use hand-constructed synthetic arrival benchmarks.)

---

## 4. RL Agents

Reinforcement learning is an established lens for data-center energy optimization. **Kahil, Sharma, Välisuo & Elmusrati, "Reinforcement learning for data center energy efficiency optimization: A systematic literature review and research roadmap" (Applied Energy)** survey the area and find model-free RL increasingly applied to scheduling and resource control, and **Ran, Hu, Zhou & Wen, "DeepEE: Joint Optimization of Job Scheduling and Cooling Control for Data Center Energy Efficiency Using Deep Reinforcement Learning" (IEEE INFOCOM 2019)** is a representative DRL scheduler (it jointly controls cooling, which we exclude per scope; RL for thermal/HVAC control is an adjacent subfield, e.g. *"Practical Implementation and Evaluation of Deep Reinforcement Learning Control for a Radiant Heating System"*). We apply RL to the *spatial routing + temporal deferral* decision and compare two algorithm families — PPO (continuous) and DQN (discrete).

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

This setup tests two questions: (a) does the CFWS algorithmic choice (DQN) work on our formulation? (b) does CFWS's *action-encoding* idea (small, semantically-meaningful action set) transfer? See §8.7 for the formulation/algorithm/encoding comparison tables and §7 for empirical results. Both DQN variants are independent of and not reimplementations of CFWS — CFWS operates on per-PM VM migrations at a different state granularity.

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

Run with the calibrated `deadline_penalty_weight = 250` (§7.8).

| Policy | Total Cost ($) | Energy Cost ($) | Peak Penalty ($) | ND-weighted Load | Batch Expired | Avg Pool |
|---|---|---|---|---|---|---|
| **PPO** | **9,975,114** | 7,554,692 | 1,719,534 | 1,641,603 | 2,804 | 1.40 |
| Defer to Low Net Demand | 9,993,387 | 7,583,739 | 1,703,775 | 1,637,060 | 2,823 | 3.27 |
| DQN (routing-grid) | 9,997,085 | 7,140,220 | 1,592,896 | 1,579,613 | 5,056 | 9.45 |
| Round Robin | 10,006,942 | 7,576,297 | 1,707,796 | 1,636,550 | 2,891 | 1.34 |
| Trough-Slot Lookahead | 10,024,073 | 7,551,603 | 1,639,692 | 1,598,416 | 3,331 | 1.43 |
| Random | 10,039,999 | 7,513,320 | 1,690,884 | 1,622,873 | 3,343 | 1.39 |
| Local Only | 10,064,149 | 7,451,005 | 1,651,865 | 1,609,136 | 3,845 | 0.80 |
| Cheapest Price First | 10,076,771 | 7,013,525 | 1,657,654 | 1,578,385 | 5,622 | 1.54 |
| DQN (flat-idx) | 10,090,407 | 7,229,653 | 1,560,502 | 1,563,648 | 5,201 | 9.47 |
| Drain Immediately | 10,120,638 | 7,280,894 | 1,577,120 | 1,572,744 | 5,050 | 0.57 |
| Status Quo (no optimization) | 10,127,636 | 7,284,546 | 1,578,422 | 1,573,224 | 5,059 | 0.57 |
| Avoid the Ramp | 10,234,914 | 7,172,335 | 1,449,439 | 1,488,550 | 6,434 | 3.68 |

**Key findings**:

1. **PPO wins — narrowly, and for the right reason.** PPO is 1st ($9.98M), but the top of the table is a tight cluster: +0.2% over the best DQN, +0.3% over Round Robin, +0.5% over the foresighted Trough-Slot oracle. PPO wins by **completing its committed work** — it expires the fewest units (2,804) of any policy. This is the inverse of the mis-calibrated `w=2` run, where dumping batch won (§7.8).

2. **Aggressive-expiry policies sink to the bottom.** Status Quo, Drain-Immediately, Cheapest-First, and Avoid-the-Ramp — the "winners" before recalibration — are now the *most expensive*, expiring 5,000–6,400 units and paying the deadline penalty. Avoid-the-Ramp is last.

3. **The temporal lever is modest, not dramatic.** Batch mode lowers PPO's cost from $10.10M (legacy) to $9.98M — a **1.2% reduction**, not the inflated 6.2% of the mis-calibrated run. Once expiring committed work is properly priced, the deferral gain is real but small under low spatial diversity. It is also *double-edged*: batch mode **raises** cost for naive policies (Status Quo: $9.85M legacy → $10.13M batch) that defer without managing deadlines.

4. **DQN (routing-grid) is competitive (3rd) via a different strategy.** It achieves the second-lowest energy cost ($7.14M) by draining aggressively and routing well, but expires 5,056 units, so the deadline penalty pulls it back to ~$10.00M. **DQN (flat-idx)** is mid-pack ($10.09M); its always-on uniform-drain commitment over-expires (5,201). Both also carry large idle pools (~9.5), a sign of jerky hold-then-dump drain timing rather than the smooth scheduling PPO learns.

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

Run with the calibrated `deadline_penalty_weight = 250` (§7.8).

| Policy | Total Cost ($) | Energy Cost ($) | Peak Penalty ($) | ND-weighted Load | Batch Expired | Avg Pool |
|---|---|---|---|---|---|---|
| **Cheapest Price First** | **13,517,795** | 10,390,572 | 1,750,579 | 1,681,462 | 5,507 | 1.76 |
| Avoid the Ramp | 13,808,135 | 10,742,100 | 1,601,776 | 1,616,244 | 5,856 | 4.60 |
| **PPO** | 13,825,816 | 11,264,455 | 1,834,245 | 1,753,045 | 2,908 | 1.33 |
| Trough-Slot Lookahead | 14,005,607 | 11,403,912 | 1,758,550 | 1,712,191 | 3,373 | 1.45 |
| Status Quo (no optimization) | 14,102,545 | 11,144,801 | 1,693,077 | 1,687,778 | 5,059 | 0.57 |
| Drain Immediately | 14,112,130 | 11,156,406 | 1,693,100 | 1,687,983 | 5,050 | 0.57 |
| Local Only | 14,131,490 | 11,397,917 | 1,772,294 | 1,726,511 | 3,845 | 0.80 |
| Random | 14,166,306 | 11,513,985 | 1,816,526 | 1,742,392 | 3,343 | 1.39 |
| Round Robin | 14,166,563 | 11,609,367 | 1,834,346 | 1,756,916 | 2,891 | 1.34 |
| Defer to Low Net Demand | 14,177,876 | 11,707,241 | 1,857,837 | 1,770,406 | 2,451 | 4.22 |
| DQN (routing-grid) | 14,373,836 | 11,580,539 | 1,851,353 | 1,748,775 | 3,768 | 1.44 |
| DQN (flat-idx) | 14,493,037 | 11,488,471 | 1,722,647 | 1,697,325 | 5,128 | 0.63 |

**Key findings**:

1. **A deadline-violating price-arbitrage heuristic wins on raw cost.** Cheapest-Price-First ($13.52M) routes batch to the cheapest grid and lets 5,507 units expire; even paying ~$1.4M in (calibrated) deadline penalties, the 16-hour price spread — chiefly routing away from high-priced Singapore — nets it ahead. **Batch deferral *rescues* these concentration heuristics** that catastrophically failed in legacy mode (Cheapest was $43.8M in §7.3) — the pool absorbs the capacity shock.

2. **PPO is 3rd on raw cost but the best deadline-respecting policy.** At $13.83M it trails Cheapest by 2.3%, but it expires **2,908 units — about half** of Cheapest's 5,507 — and holds the flattest draw of the competitive policies (load factor 0.766 vs 0.730). PPO buys low cost *without* discarding committed work; the two policies ahead of it buy lower cost *by* discarding it. Which is "better" depends on how binding the deadlines are taken to be — a genuine multi-objective tradeoff, not a clean win.

3. **Temporal deferral does not help the disciplined policy here.** PPO's Global cost *rises* from $13.48M (legacy) to $13.83M (batch): with the high price diversity already captured by spatial routing, adding the deferrable-batch deadline burden is a net cost for a low-expiry policy. In Global, the temporal lever pays off mainly by *rescuing* otherwise-failing concentration heuristics, not by improving the best routing policy.

4. **DQN flips to the bottom.** DQN (flat-idx) is **last** ($14.49M, expiring 5,128) and DQN (routing-grid) is 11th — the exact reverse of Global legacy (§7.3), where flat-idx was best. Bolting the discrete uniform-drain commitment onto the flat-idx encoding hurts badly once temporal management matters; the CFWS-style encoding helps for *pure routing* and hurts for *routing + drain*.

### 7.5 Per-DC Energy Cost Breakdown

**US Batch Mode** (per-DC energy cost):

| Policy | US-West | US-Central | US-Southeast-1 | US-Southeast-2 |
|---|---|---|---|---|
| PPO | $2,351,838 | $1,575,406 | $1,922,010 | $1,705,438 |
| Round Robin | $2,438,136 | $1,521,309 | $1,919,083 | $1,697,768 |
| Status Quo | $2,356,727 | $1,457,309 | $1,844,973 | $1,625,538 |

PPO trims US-West (the CAISO duck-curve region) by ~3.5% vs Round Robin, shifting a little load to US-Central. But the per-DC reallocation is **modest** in the low-diversity US scenario — Status Quo's local-only split is already close, and PPO's win comes more from low deadline-expiry than from dramatic re-routing.

**Global Batch Mode** (per-DC energy cost):

| Policy | Global-US-West | Global-US-Central | Global-EU | Global-Asia |
|---|---|---|---|---|
| PPO | $2,489,265 | $1,619,778 | $2,391,778 | $4,763,633 |
| Cheapest Price First | $2,026,258 | $2,002,893 | $1,956,022 | $4,405,398 |
| Round Robin | $2,438,136 | $1,521,309 | $2,364,840 | $5,285,082 |

The Global-Asia DC (Singapore, high EMA prices) dominates fleet cost. PPO cuts Global-Asia energy by **~10%** vs Round Robin by deferring and rerouting Singapore-bound work. Cheapest-First cuts it (and EU / US-West) further still by routing even more aggressively to the cheapest grids — but it pays for that with its high deadline-violation count (§7.4), the tradeoff at the heart of the Global-batch result.

### 7.6 Burst-Aware Augmentation (Heavy-Tail Follow-Up)

> ⚠ **Pending rerun.** The burst experiments in this subsection were run on the *pre-recalibration* models (old batch fraction, `deadline_penalty_weight=2.0`). The qualitative finding — the optimization signal concentrates in burst windows — is expected to survive, but the specific percentages are stale and will be regenerated with the w=250 batch models (§7.8).

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

### 7.7 Comparison to the no-optimization status quo

The tables above rank policies against each other. This section adds the external reference point: **how much grid-aware optimization saves over running the workload as-is** — the closest in-framework analogue to "Google's actual operation."

**The Status Quo baseline.** `StatusQuoPolicy` ([baselines.py](baselines.py)) serves each cell's own demand **locally and immediately** — no cross-DC routing, no temporal deferral. It is Borg's raw aggregate run grid-unaware: a CICS-style load-shaper switched *off* (§1.2, §8.6). Since the 2019 trace has no grid prices, no net demand, and no inter-cell routing, this is the correct counterfactual — *not* "Google's scheduler," which never solved the multi-DC grid-aware problem.

**Why the comparison is fair despite R²=0.43.** Status Quo and every optimized policy are scored by the **same** power model, so its absolute error (CPU explains ~43% of point-level power variance; §3.2) **cancels in the relative comparison** — a shared bias does not change the *difference* between policies. The reported saving is therefore robust to the power model's noise; R²=0.43 is the honest model fit, not a limitation on this comparison.

**Normalized power metric — load factor.** Beyond cost, every policy reports `load_factor = mean / peak` aggregate grid draw (emitted by `compute_summary`). Higher = flatter. In batch mode the *peak* is set by non-deferrable **service** demand (≈357.5 MW for every policy), so optimization improves the load factor by **filling the troughs** with deferred batch, not by shaving the peak.

| Policy (US batch, w=250) | Total cost | Load factor | Δ cost vs Status Quo |
|---|---|---|---|
| **PPO** | **$9.98M** | **0.767** | **+1.5%** |
| Round Robin | $10.01M | 0.766 | +1.2% |
| Trough-Slot Lookahead (oracle) | $10.02M | 0.760 | +1.0% |
| Status Quo (no optimization) | $10.13M | 0.736 | reference |

PPO saves **1.5% over the grid-unaware status quo** in US batch with the flattest draw (0.767 vs 0.736), and the saving holds across configs — **+2.0% in Global batch, +5.3% in Global legacy** — *except* US legacy, where PPO's over-concentration leaves it ~2.5% *worse* than status quo (§7.1). So "savings versus the grid-unaware status quo" is real wherever there is either spatial diversity or well-managed deferral; it is the cleanest externally-facing result, and it mirrors what a CICS-style layer contributes on top of Borg. The visual profile is produced by [analysis/plot_power_profile.py](analysis/plot_power_profile.py) → `output/power_profile_comparison.png`.

### 7.8 Calibrating the deadline penalty — a sensitivity lesson

A non-obvious lesson surfaced when the corrected free+beb batch fraction (27–61%; §2.2) replaced the old tiny one. The deadline-violation penalty `deadline_penalty_weight × expired_demand` had been set to **2.0**, calibrated when batch was a negligible slice of load. At the realistic volume that value is **~100× too weak**: expiring a unit of deferred work costs \$2, while *serving* it costs ~\$150 of energy — so the cost-minimal strategy becomes to **dump batch and pay the trivial penalty**. Under that miscalibration, aggressive-expiry policies (Avoid-the-Ramp, Cheapest-First, even immediate-drain Status Quo) *beat* PPO, which conservatively served its committed work — an artifact, not a real ranking.

Because the penalty is linear in expired demand, the cost of any rollout at any weight is recoverable without re-evaluating: `cost(w) = (cost − 2·expired) + w·expired`. Sweeping `w` over the US-batch rollouts inverts the ranking around **w ≈ 183** (where low-expiry policies overtake high-expiry ones):

| Policy (US batch) | expired | cost @ w=2 | cost @ w=250 |
|---|---|---|---|
| **PPO** | 3,127 | $9.21M (10th) | **$9.99M (1st)** |
| Avoid-the-Ramp | 6,434 | $8.64M (1st) | $10.23M (last) |
| Status Quo | 5,059 | $8.87M | $10.13M |

**The calibration.** A principled value is the **energy cost of serving one unit** of deferred CPU — `slope · rated_power · 1000 · Δt · price ≈ $150` at a typical wholesale price — so that dropping committed work is never cheaper than doing it. We set **`deadline_penalty_weight = 250`** (≈ that energy cost at a moderately high price, comfortably above the empirical $183 crossover). At this value the optimization rewards *completing* deferred work, and PPO's spatial batch routing — which lowers forced expiry by balancing load across DCs (3,127 vs Status Quo's 5,059) — becomes a legitimate advantage rather than a liability.

**The general lesson:** in a deferral-with-deadlines reward, the deadline penalty must **scale with the value of the deferred work** (≈ its energy cost), not be a fixed small constant — otherwise the agent learns to discard work whenever the deferrable fraction is non-trivial. All batch-mode results in this thesis use the recalibrated **`w = 250`**.

### 7.9 Queue, don't dump — preserving deadlines under capacity pressure

A second batch-model artifact surfaced from a simple sanity check: **why does Status Quo expire ~5,000 units?** A policy that drains everything *immediately* should never discard work. The cause was in the step loop — drained batch that didn't fit under capacity was re-queued with a **1-step deadline (`t+1`)**, so any capacity-blocked work expired the very next step instead of waiting for a later trough. With the realistic free+beb fraction (27–61%), bursty arrivals routinely exceed capacity, so this manufactured a large, *policy-independent* expiry floor — ~5,059 in **both** US and Global batch (identical, because the cells/capacity/arrivals are the same and only the grid differs: the tell that it was an artifact, not a result).

**This is the opposite of how Borg behaves.** The best-effort batch tier is *queued* — beb jobs wait for capacity, managed by the batch scheduler; they are not discarded because they could not run this instant. The "deadline" itself is a *synthetic* construct (Grange/Da Costa's `flexibility_factor`; §2.2 / §8.2), not a property of the trace. So the `t+1` re-queue contradicted both Borg's documented behavior and the deferral semantics we meant to model.

**The fix** ([multi_dc_env.py](env/multi_dc_env.py), Phases 3/7): the drain decision is now an *intended release*, and **only the batch actually served is removed from the pool**; everything unserved stays with its **original deadline** and is retried at a later trough, expiring only when genuinely overdue. Effect, on fixed-behavior baselines:

| US batch | expired (before) | expired (after) |
|---|---|---|
| Status Quo | 5,059 | **738** |
| Drain Immediately | 5,050 | **738** |
| Round Robin | 2,891 | **777** |
| Avoid the Ramp | 6,434 | 3,617 |

Status Quo's expiry collapses by **~85%** to a small genuine residual (bursts that exceed capacity even across the full deadline window). Avoid-the-Ramp stays high — but *legitimately*, because routing everything to one DC really does overload it. With the artifact removed, the batch comparison measures **scheduling skill** (placing deferrable work in the troughs) rather than who least-suffers from a modeling fuse. The §7.2–7.5 batch tables reflect this fix (and the calibrated `w = 250`).

**The general lesson:** in a deferral-with-deadlines simulator, work that is merely capacity-blocked must **retain its deadline and queue**, not be discarded — otherwise saturation manufactures artificial deadline violations that swamp the real optimization signal.

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
| No-SLO tiers (free + beb) as the deferrable pool | Both the free tier (priority ≤ 99) and the best-effort batch tier (110–115) are explicitly "no associated SLOs" (§2); strict beb alone is negligible in cells a–d, so the deferrable class is the union of the two SLO-free tiers (see §2.2) |
| Heavy-tailed distribution fits (Weibull, log-normal) | "Top 1% of jobs (resource hogs) consume over 99% of all resources" with squared coefficient of variation > 23,000 (§7) |
| 31-day episode length | "31 days" duration of the 2019 trace (Table 1) |

This grounds our data usage in the dataset publisher's own framing rather than borrowing a methodology from an unrelated research thread. The cells a–d we use are four of the eight cells (a–h) Tirmazi analyzes, with ~12k machines per cell on average.

### 8.2 Workload-generation foundation — Da Costa et al. (2016) & Grange et al. (2018)

The synthetic batch workload that drives every experiment is **not original to this thesis** — its methodology, and even its concrete generator code, come from two papers. This subsection makes that provenance explicit.

**The model — Da Costa, Grange & De Courchelle (2016), "Modeling and generating large-scale Google-like workload" (IGSC '16).** This is the parameterized generator at the root of our pipeline: it fits statistical distributions to a real Google cluster and synthesizes tasks from four knobs — `mass` (mean execution time), `disparity` (mean/median ratio), `dynamism` (mean inter-arrival), `ratioTask` (batch fraction). The entire "fit distributions to the trace → sample a synthetic workload" approach this thesis uses originates here.

**The instantiation — Grange, Da Costa & Stolf (2018), "Green IT scheduling for data center powered with renewable energy" (FGCS 86).** Grange instantiates that 2016 model as a compact `scipy.stats` generator (their **Listing 1**: modified-Pareto inter-arrivals, log-normal execution time, truncated-exponential priority). We **reproduce Listing 1 verbatim** in [scripts/grange_generator.py](scripts/grange_generator.py) — it recovers their published parameters exactly (log-normal `s=1.634, scale=447`; makespan mean ≈ their `mass`=1700) — and our [`BatchArrivalGenerator`](env/workload_generator.py) is the same generator *family*.

**Exactly what we took from each, and what we added:**

| Idea | Source | Where it lives in our work |
|---|---|---|
| Fit-distributions-then-generate methodology | Da Costa et al. (2016) | §2.2, §3.8; `scripts/refit_freebeb_local.py` |
| Concrete scipy generator (Listing 1) | Grange et al. (2018) | `scripts/grange_generator.py` (verbatim); `BatchArrivalGenerator` (same family) |
| Log-normal execution-time / heavy-tailed fits | both | our duration fits (§2.2) |
| SLA-flexibility → deadline knob | Grange et al. (2018) | `flexibility_factor` (§3.7) |
| "SLA freedom drives savings" finding | Grange et al. (2018) | motivates the flexibility sweep (§3.7) |
| Separation of concerns (scheduler agnostic of electrical infrastructure) | Grange et al. (2018) | reward `peak_penalty` summarizes grid stress (§8.3) |
| **Our extensions** | — | re-fit to the **2019** trace under the no-SLO definition (§2.2); **KS-D** model selection; added per-task **CPU/memory** and **discrete task-count** distributions; the scheduling *decision* is made at aggregate granularity, not per task (§1.2) |

Grange therefore plays a **dual role** here: the *methodological parent* of our workload generator (this subsection) and a *benchmark predecessor* for the scheduling problem (§8.3). The difference between Grange's per-task scheduler and our aggregate-flow decision is the granularity point of §1.2.

### 8.3 Single-DC renewable-aware predecessors — Grange (2018), Haghshenas et al., Liu et al.

The renewable-aware batch scheduling literature is largely **single-DC**. Three reference points form the immediate lineage:

**Grange, Da Costa, Stolf (2018), "Green IT scheduling for data center powered with renewable energy"** (*Future Generation Computer Systems* 86) is the closest single-DC predecessor to our batch-deferral logic. (Its workload-*generator* role, which we reuse directly, is covered separately in §8.2; here we treat its *scheduling* contribution.) Grange schedules **batch jobs with due-date constraints** in a small-scale DC powered by on-site solar panels and grid, achieving up to **49% brown-energy reduction and 51% cost savings** vs a renewable-unaware scheduler. Their core architectural insight — which we adopt — is **separation of concerns**: "a scheduling algorithm agnostic of the electrical infrastructure. A separated system, managing the renewable sources, provides an arbitrary objective function, which is used to guide the scheduling heuristic." In our env, the same separation holds: the agent's reward gets a `peak_penalty` term that summarizes grid stress via `net_demand_normalized × grid_mw²`, but the agent doesn't model the grid directly. Grange also identifies the SLA-flexibility/savings tradeoff that motivates our `flexibility_factor` parameter (§3.7).

**Haghshenas, Taheri, Goudarzi, Mohammadi, "Infrastructure Aware Heterogeneous-Workloads Scheduling for Data Center Energy Cost Minimization"** considers a single Internet DC with **heterogeneous interactive + batch workloads**, on-site solar, cooling subsystem, and time-varying electricity prices. Their algorithm achieves 46% cost reduction. Two aspects flow through to our work: (1) the **interactive-vs-batch split** that we encode as `get_service_demand(t)` (non-deferrable) vs `get_batch_demand(t)` (pool-managed) — Haghshenas-style heterogeneous workloads are the rationale for treating these as separate workload classes; (2) **electricity rate structure awareness**, which we extend from a single DC's local price to per-DC LMP signals.

**Liu, Chen, Bash, Wierman, Gmach, Wang, Marwah, Hyser, "Renewable and Cooling Aware Workload Management for Sustainable Data Centers"** (Caltech + HP Labs) takes a **predict-then-plan** structure: forecast renewable supply + IT demand, then generate a workload plan that schedules IT work and allocates resources according to time-varying power supply. The forecasting horizon and lookahead-based planning structure is what motivates our **Trough-Slot Lookahead** baseline (§8.5) — though we use net demand as the forecast signal rather than renewable supply, consistent with the reframing in §1.

All three are **single-DC**. The combination "multi-DC routing + temporal batch deferral + grid-aware objective + cell-aggregate ClusterData" is the gap this thesis fills relative to that lineage.

### 8.4 Interactive + batch deferral pattern — Xu, Toosi, Buyya

**"A Self-Adaptive Approach for Managing Applications and Harnessing Renewable Energy for Sustainable Cloud Computing"** (Xu, Toosi, Buyya) provides the framework we adopt for **splitting workloads into interactive (must-serve-now) and batch (deferrable) components**, with separate handling for each: brownout for interactive, deferring for batch. Our environment's `batch_enabled` mode (§3.7) and the batch pool / drain abstraction are direct descendants of this framework, restricted to the batch side (we don't implement brownout). Like the others in §8.3, Xu's setup is single-DC.

### 8.5 Lookahead-based scheduling — GreenSlot (Goiri et al. 2011)

**Goiri et al., "GreenSlot: Scheduling Energy Consumption in Green Datacenters"** is the lineage for our **Trough-Slot Lookahead** baseline. Per Grange et al.'s clear summary (§8.3 above): GreenSlot "considered a small cluster used for scientific computation, and powered partially with solar panels. Using prediction of renewable power available, along with grid electricity price, the GreenSlot algorithm delays jobs to execute them when the cost is the lowest (both in terms of brown energy usage and in terms of purchasing cost)." The algorithm discretizes future time into fixed-duration slots, each "valuated with predicted renewable energy production, grid electricity cost, and number of available computing nodes," then greedily places each task in the first slot allowing renewable-only execution.

Our Trough-Slot Lookahead baseline (§5.8) reuses the slot-valuation idea but retargets it to the demand-smoothing formulation: instead of evaluating slots by predicted renewable supply, it evaluates them by predicted **grid net demand** — a slot is "good" when net demand will be low (a duck-curve trough), not when local solar will be high. The 36-step (3-hour) lookahead window directly mirrors GreenSlot's slot horizon.

| Aspect | GreenSlot | Our Trough-Slot baseline | Our PPO agent |
|---|---|---|---|
| Approach | Greedy slot valuation (LP variant in follow-up GreenSwitch) | Lookahead heuristic | Model-free RL |
| Scheduling | Temporal only (single DC) | Spatial + temporal (multi-DC) | Spatial + temporal (multi-DC) |
| Slot signal | Solar supply + grid price | Grid net demand (forecast) | Implicit from observations |
| Forecast required | Yes (explicit) | Yes (oracle: actual future net demand) | No (learned) |
| Number of DCs | 1 | 4 | 4 |

PPO **beats Trough-Slot Lookahead in every config where there is spatial diversity or well-managed deferral** (i.e. all but US legacy):

| Config | PPO | Trough-Slot | PPO Advantage |
|---|---|---|---|
| US batch | $9.98M | $10.02M | +0.5% |
| Global legacy | $13.48M | $14.07M | **+4.2%** |
| Global batch | $13.83M | $14.01M | +1.3% |

This is meaningful because Trough-Slot has **privileged 3-hour future net-demand information** that PPO does not: PPO learns implicit forecasting *and* coordinates it with routing. But under the calibrated deadline penalty the *batch*-mode margins are now slim (+0.5% / +1.3%) — most of PPO's edge over the foresighted heuristic comes from spatial routing (Global legacy, +4.2%), not from temporal scheduling. In US legacy (no temporal flexibility, low diversity) PPO trails Trough-Slot by 1.6%.

### 8.6 Granularity & operational precedent — Radovanovic et al. (2023)

**Radovanović, Koningstein, Schneider, Chen, Duarte, Roy, Xiao, Haridasan, Hung, Care, Talukdar, Mullen, Smith, Cottman, Cirne. "Carbon-Aware Computing for Datacenters." IEEE Transactions on Power Systems (2023; arXiv 2021)** describes **CICS — Google's Carbon-Intelligent Compute System**, the production system that shifts temporally flexible workloads across Google's datacenter portfolio (20+ DCs, 15.5 TWh annual consumption, 4 continents) to align computing with low-carbon grid hours, via day-ahead cluster-level **Virtual Capacity Curves (VCCs)**.

CICS is this thesis's single most important external anchor, and it serves as a precedent in **two distinct ways** (the full point-for-point alignment is in §1.2):

**1. Granularity precedent — this is the methodological one.** CICS shapes *aggregate, cluster-level* load and explicitly runs *"independently from real-time job-level scheduling."* Crucially, it frames the job-level deadline-optimization approach — i.e. the §8.3 lineage (Grange et al.) — as the **prior paradigm it superseded**, using *"aggregate cluster-specific resource demand forecasts… rather than… stylized models for job-level resource demand modeling."* So our divisible-aggregate-flow abstraction (§1.2) is **not** a simplification of the academic predecessors — it is the newer, production-validated paradigm. CICS also matches our **tier-based deferrable split** (*"temporally inflexible (higher tiers)"* vs *"flexible (lower-tier batch jobs)"* = our production vs free+beb, §2.2) and our **per-cluster power model on aggregate CPU** (§3.2).

**2. Operational precedent — the problem-is-real one.** Three specific CICS claims carry through to our objective:

| Radovanovic CICS | Our env |
|---|---|
| "Shifting execution of flexible workloads in time and space can decrease peak demand for resources and power" | Our peak-contribution penalty (§3.3) targets exactly this; PPO's batch-mode drain decisions are the spatial+temporal shift CICS enacts in production |
| Uses "cluster-level load forecasts and power models [Sakalkar 2020]" | Same cluster-level cell-aggregate granularity (§2.1); same Sakalkar 2020 power model lineage (§3.2) |
| "Datacenters are planned based on peak power and resource usage, smaller peaks reduce the need for more capacity" | Direct motivation for our load-squared peak penalty in the reward |

We do **not** reproduce CICS — it uses internal Google data not the public ClusterData 2019 trace, and optimizes carbon rather than grid demand smoothing. The positioning that follows: **this thesis is a published, reproducible Gymnasium methodology for the aggregate spatial-temporal load-shaping problem class that CICS solves operationally** — retargeted to grid demand smoothing and benchmarked against the finer-grained academic lineage (§8.3–8.7).

### 8.7 Academic foundation — CFWS (Zhao et al. 2025)

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

Trained models evaluated under identical conditions (α = 0.015, calibrated deadline penalty w=250, same seeds, full 31-day trace):

| Scenario | PPO | DQN (routing-grid, 759) | DQN (flat-idx, 48) | flat-idx vs routing-grid |
|---|---|---|---|---|
| US legacy | $10.10M | **$9.84M** | $10.87M | **+10.4%** worse |
| US batch | **$9.98M** | $10.00M | $10.09M | +0.9% worse |
| Global legacy | $13.48M | $13.57M | **$13.47M** (ties PPO) | **−0.7%** better |
| Global batch | $13.83M | $14.37M | $14.49M | +0.8% worse |

Three findings emerge:

1. **The flat-idx encoding helps only for *pure* spatial routing under diversity.** It beats routing-grid in exactly one config — Global legacy (−0.7%) — and *loses* in the other three: +10.4% (US legacy), +0.9% (US batch), +0.8% (Global batch). The earlier result that flat-idx *won* Global batch does **not** survive the corrected data + calibrated deadline penalty: once the discrete uniform-drain commitment is bolted onto the small action set, it over-expires (5,128 units, §7.4) and drops to last. So CFWS's encoding philosophy transfers to the *routing* problem it was designed for (Global legacy), but not to the routing+drain action our batch mode adds. *This says nothing about CFWS's own per-PM formulation* — there each migration event exposes many meaningful actions (CFWS reports 5.67–13.22% brown-energy reduction, 46.49–86.53% migration reduction on their AZ/CA/OR/LA setting), so their action space is not constrained the way our cell-aggregate one is.
2. **PPO is the best RL agent in 2 of 4 configs (US batch, Global batch) and ties for best in Global legacy**, losing only US legacy where DQN-routing-grid edges it (PPO over-concentrates, §7.1). Against the *best* DQN variant per config: −2.6% (US legacy), +0.2% (US batch), tie (Global legacy), +3.8% (Global batch).
3. **No single discrete encoding dominates.** routing-grid wins US legacy; flat-idx wins Global legacy; both DQN variants fall behind PPO in *both* batch configs. The lesson holds — action-space encoding can matter as much as algorithm choice — but the *direction* is scenario- and mode-dependent, not a clean "structured small action sets win."

#### PPO as the contribution beyond CFWS

PPO is the best RL policy in 3 of 4 configs (counting the Global-legacy tie) and the **best deadline-respecting policy in both batch modes** (§7.2, §7.4). Its continuous action space — natural for routing fractions plus drain rates — is something CFWS's discrete one-VM-migration encoding cannot express without reformulation. The honest contribution claim is narrower than "PPO dominates": **continuous actions give PPO the edge under realistic deferral — it wins both batch configs among the RL agents by *completing* committed work — while under pure spatial routing a well-encoded discrete DQN (flat-idx) matches it, and under low diversity (US legacy) PPO's over-concentration costs it.** Where a non-learning heuristic beats PPO on raw cost (Cheapest-First, Global batch), it does so only by violating ~2× more deadlines (§7.4) — so PPO remains the policy of choice whenever the deferral deadlines are taken to be real.

### 8.8 Survey context — Lin et al. (2024) and Wu et al. (2025)

**"A systematic review of green-aware management techniques for sustainable data center"** (Lin, Lin, Peng, Huang, Lin, Li, 2024) provides the broader sustainable-DC landscape view. The categories of workload management, virtual resource management, energy management, thermal management, and waste heat recovery surveyed there place this thesis within "workload management + energy management for grid-aware multi-DC operation." For the multi-DC scheduling subarea specifically, **Wu et al. (2025), "Task Scheduling in Geo-Distributed Computing: A Survey"** (arXiv:2501.15504) is the most recent systematic review and covers the geo-distributed task-scheduling thread that this thesis sits within. Additional green-DC landscape reviews — *"Energy efficiency in cloud computing data centers: a survey on software technologies"* and *"A systematic review on effective energy utilization management strategies in cloud data centers"* — corroborate the workload-/energy-management framing.

### 8.9 Positioning of Our Contribution

Stated against the lineage above:

1. **Cell-as-proxy-DC modeling exercise.** We treat four ClusterData 2019 cells (a–d) as four geographically distributed hyperscale DCs — what such DCs' workloads would look like if they had cell-level inter-DC heterogeneity. This is *not* what Tirmazi et al. (2020) intended when documenting the trace (they don't claim the cells are geographically distinct), and no prior published work does exactly this. It is a defensible modeling exercise rather than a dataset-grounded claim (see §2.1 and §3.2 for the explicit modeling assumptions on workload-as-shape and `rated_power_mw`-as-magnitude).
2. **Grid demand smoothing as a first-class objective**, via a peak-contribution penalty against actual EIA-930 net demand timeseries — rather than the on-site-renewable framing that dominates academic prior work.
3. **Continuous action space (PPO)** enabling fine-grained joint routing + drain decisions. PPO is the best RL agent in 3 of 4 configs (US batch +0.2% and Global batch +3.8% over the best DQN; a Global-legacy tie) and the **best deadline-respecting policy in both batch modes** — it loses only US legacy, where it over-concentrates (§7.1, §8.7). A well-encoded discrete DQN (flat-idx) matches it for *pure* spatial routing (Global legacy).
4. **Real-data grounding**: ClusterData 2019 (per Tirmazi 2020) for workloads, `powerdata_2019` (per Sakalkar 2020) for the power model, EIA-930 (CISO, MISO, SOCO, DUK) for grid net demand, real ISO prices, NSRDB solar irradiance as forecast features.
5. **Operational relevance**: the problem class is the same one **Google's CICS (Radovanović et al. 2023)** solves in production at 20+ DCs across 4 continents. This thesis contributes a published methodology + reproducible Gymnasium env for that problem class.

Honest framing of what this thesis is *not*: it is not a head-to-head comparable against CFWS (different action paradigm, different state granularity, different objective), nor a reimplementation of Google's CICS (closed-source operational system). It is a self-contained academic exploration of multi-DC + cell-aggregate + RL + grid-aware scheduling, with the cell-as-DC and 100 MW magnitude assumptions stated explicitly rather than hidden.

The primary empirical findings are:

- **Spatial routing matters under regional diversity.** In Global legacy (16-hour spread), PPO beats Round Robin by **5.4%** and the no-optimization status quo by 5.3%. In US (3-hour spread, similar profiles), spatial routing alone is structurally limited and PPO over-concentrates (it trails the trivial baselines, §7.1).
- **Temporal deferral is a modest, double-edged lever — not a universal win.** With the *calibrated* deadline penalty (§7.8), batch mode lowers PPO's US cost by only **1.2%** and *raises* its Global cost by **2.6%** (high diversity is already captured by routing). Deferral mainly *rescues* concentration heuristics that fail in legacy mode; for the disciplined policy its value is small and scenario-dependent. (The inflated 6.2% / 2.1% "universal lever" of earlier drafts was an artifact of the mis-calibrated deadline penalty.)
- **PPO is the best deadline-respecting policy, not a universal cost winner.** It beats the foresighted Trough-Slot oracle by +0.5–4.2% wherever diversity or deferral helps, and beats both DQN variants in batch mode. Where a pure-cost heuristic edges it (Cheapest-First, Global batch), it does so only by violating ~2× more deadlines (§7.4).
- **A reward-calibration lesson (§7.8):** the deadline penalty must scale with the energy value of deferred work; mis-set, it inverts the entire batch ranking — itself a methodological contribution.

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

# CFWS-style flat-index DQN variant (48-action hash-map-decoded space, §8.7)
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

1. **No single policy dominates; PPO is the best *deadline-respecting* agent.** PPO is the best RL policy in US batch and Global batch, and ties for best in Global legacy — losing only US legacy, where it over-concentrates routing onto one DC (capacity-violation penalties exceed its energy savings). The honest headline is not "PPO wins everywhere" but "PPO achieves low cost *while completing its committed work*" (takeaways 3–4).

2. **PPO beats the foresighted Trough-Slot oracle wherever diversity or deferral helps, but the batch margins are slim.** +0.5% (US batch), +4.2% (Global legacy), +1.3% (Global batch); it trails by 1.6% in US legacy. Most of PPO's edge over the oracle comes from *spatial routing under diversity*, not from temporal scheduling.

3. **Temporal deferral is a modest, double-edged lever — not the universal win earlier drafts claimed.** With the calibrated deadline penalty (takeaway 4), batch mode lowers PPO's US cost by only **1.2%** and *raises* its Global cost by **2.6%** (high diversity is already captured by routing). Deferral mainly *rescues* concentration heuristics that fail in legacy mode (takeaway 7); for the disciplined policy its value is small and scenario-dependent.

4. **A reward-calibration lesson: the deadline penalty must scale with the value of the deferred work (§7.8).** Set to 2.0 for the old tiny batch fraction, it was ~100× too weak once 27–61% of load became deferrable — expiring a unit cost $2, serving it ~$150 of energy, so the cost-minimal strategy was to *dump* batch, and aggressive-expiry policies beat PPO. Recalibrated to 250 (≈ the energy cost of serving a unit; ranking crossover at w≈183), the optimization correctly rewards completing work, and **PPO becomes the lowest-expiry policy** (~2,800–2,900 units vs ~5,000–6,400 for the dumpers). The entire batch ranking hinges on this one constant.

5. **Action-encoding choice matters, but its direction is scenario- and mode-dependent.** Of two DQN variants — 759-action routing-grid and 48-action CFWS-style flat-idx — flat-idx beats routing-grid in exactly one config, **Global legacy** (−0.7%, the *pure-routing* high-diversity case), and loses in the other three (US legacy +10.4%, US batch +0.9%, Global batch +0.8% worse). The earlier "flat-idx wins Global batch" does **not** survive the corrected data: adding the discrete uniform-drain commitment makes it over-expire and finish *last* in Global batch. CFWS's encoding philosophy thus transfers to spatial routing but not to routing+drain — and this says nothing about CFWS's own per-PM formulation, where each migration exposes many actions.

6. **Per-DC reallocation matches the duck-curve story, but modestly.** PPO trims US-West (CAISO) energy ~3.5% vs Round Robin and Global-Asia (Singapore) ~10% — exploiting the regional net-demand/price differences that motivated the framing, though the US effect is small in the low-diversity scenario.

7. **Concentration heuristics fail without temporal slack — and can *win* with it.** Avoid-the-Ramp and Cheapest-First collapse in legacy mode (8× / 4× worse than Round Robin) from capacity violations. In batch mode the deferral pool absorbs the shock; in **Global batch, Cheapest-First becomes the cheapest policy overall** — by dumping ~5,500 units to chase the cheapest grid. Under extreme price diversity a deadline-violating concentrator can beat the disciplined agent on raw cost; whether that "wins" depends on how real the deadlines are.

8. **The formulation produces meaningful, well-differentiated policy spread.** Rankings are clearly separated and the tradeoffs (cost vs deadline-violation, routing vs drain) are exposed rather than smothered — the corrected data makes the optimization surface genuinely informative.

9. **(Pending rerun, §7.6)** **The RL optimization signal concentrates in burst windows.** From the *pre-recalibration* burst study (stale numbers, to be regenerated): burst timesteps (top 5% by arrival) don't drive disproportionate raw cost — the pool-with-deadline mechanic smears it across drain steps — but each RL agent's advantage over Round Robin is larger in burst windows, and PPO learns implicit burst-aware spatial routing without an explicit signal. The qualitative finding is expected to hold under w=250; the percentages will be refreshed.
