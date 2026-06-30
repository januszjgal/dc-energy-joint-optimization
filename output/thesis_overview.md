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

The 2019 trace exposes job priority as a sparse value in [0, 450] and groups priorities into named **tiers**. We classify a job as deferrable **batch** iff it belongs to one of the two **SLO-free tiers**: the **free tier** (`priority ≤ 99`) or the **best-effort batch (beb) tier** (`priority ∈ [100, 115]`) — i.e., simply **`priority ≤ 115`**. Tier bounds follow the authoritative trace documentation (**Wilkes, *"Google cluster-usage traces v3"*, 2020-08 revision**), which explicitly notes that the beb range was *"mistakenly reported as 110–115"* in **Tirmazi et al. (2020), *"Borg: the Next Generation"*, §2** — an erratum our pipeline initially inherited by citing the paper faithfully (see §12, row 11). Both no-SLO tiers are the genuinely delay-tolerant work. Every SLO-bearing tier is non-deferrable **service**: the mid-tier (116–119, weak SLOs) and especially the **production tier** (120–359), which "require[s] high availability" — "Borg will evict lower-tier jobs in order to ensure production tier jobs receive their expected level of service" (Tirmazi §2). Monitoring (≥ 360) is likewise service. Production must be served immediately and cannot be queued. Classification is by **priority**, not `scheduling_class`: the latter encodes latency-sensitivity (0–3), is orthogonal to tier ("scheduling class is *not* a priority," trace docs v3), and a `scheduling_class ≤ 1` filter is dominated by latency-insensitive *production* (priority 200) jobs.

All numbers below are the **corrected `≤ 115` (free + beb) extraction** — the re-extraction the earlier erratum banner anticipated is done, and the deferrable shares duly rose (the 100–109 band the inherited 110–115 filter had dropped was the bulk of best-effort batch). Exact per-cell fit parameters live in `data/jobs/batch_distributions_{a..d}.json`.

We fit statistical distributions to each cell's deferrable jobs (Cell A shown; illustrative — full params in the JSON):

| Property | Cell A distribution | Key parameters |
|---|---|---|
| **Inter-arrival time** (sec) | Weibull (min) | shape=0.54, scale=33.0, mean=29.4s, KS D=0.11 |
| **Duration** (sec) | Log-normal | σ=1.74, scale=129, mean=1430s, median=105s, KS D=0.13 |
| **CPU request** (normalized) | Log-normal | σ=0.81, mean=0.0090, KS D=0.07 |
| **Memory request** (normalized) | Log-normal | σ=0.95, mean=0.0061, KS D=0.08 |
| **Tasks per job** | Negative binomial | mean=52.5, median=1, KS D=0.20 |

**Fitting methodology.** Candidate continuous distributions (exponential, log-normal, gamma, Weibull) are fit by MLE and the best selected by **minimum KS *D* statistic** — *not* the KS *p*-value, which underflows to 0 at n~10⁵ regardless of fit quality and is invalid for parameters estimated from the same sample (the Lilliefors situation). Tasks-per-job is count data, so it uses a **discrete** fit (Poisson / geometric / negative-binomial). Inter-arrivals and resource requests are taken over all deferrable jobs; durations only over jobs that reached FINISH. The heavy-tailed winners — log-normal durations/requests, negative-binomial task counts with mean 52 ≫ median 1 — reproduce the extreme variability Tirmazi documents: "the top 1% of jobs consume over 99% of resources," squared coefficient of variation > 23,000 (Tirmazi §7).

**Batch fraction.** The environment splits the CPU-*usage* curve, so the operative `batch_fraction` is the no-SLO tiers' share of **measured usage** — extracted ground-truth from `instance_usage` split by priority tier ([extract_tier_curves.ipynb](extract_tier_curves.ipynb)):

| | Cell A | Cell B | Cell C | Cell D |
|---|---|---|---|---|
| **measured usage share (used by the env)** | **16.6%** | **15.7%** | **22.3%** | **26.4%** |
| by CPU request | 63% | 85% | 81% | 79% |
| by CPU-time proxy (request × duration) | 60% | 82% | 77% | 75% |
| by job count | 2.4% | 49.8% | 9.1% | 14.3% |

**The request-based proxies overestimate the deferrable *usage* share by ~3–5×.** This is exactly the over-allocation Tirmazi documents: the best-effort tiers request far more than Borg actually runs them at — *"cell c has allocated ~140% of the cell's memory capacity just to the best-effort batch tier"* (§5 of that paper) while tier *usage* is a smaller fraction of that. Requests measure intent; usage measures the schedulable reality. The corrected measured deferrable share — **16–26% of load** — sits right at the **~20%-of-capacity best-effort average** Tirmazi reports for the trace, an independent consistency check. `batch_fraction` remains a natural **sensitivity-sweep parameter** (the synthetic generator, §3.8, can instantiate counterfactually batch-heavier mixes). (All figures supersede earlier `scheduling_class ≤ 1 AND priority < 200` numbers, which conflated production with batch, and the inherited 110–115 band, which dropped most of beb.)

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

### 2.4 Electricity Prices — documented synthetic (⚠ provenance, to resolve)

> **⚠ Honesty note — the price series are SYNTHETIC, not real LMP.** This is the thesis's single most important data caveat, surfaced during the peer-review response. The committed `data/prices/*.csv` are **byte-identical** to the output of the calibrated diurnal generator in [preprocess/price_fetcher.py](preprocess/price_fetcher.py) (`generate_synthetic_prices`) — confirmed by regenerating and diffing (max |Δ| = 9.7×10⁻¹⁷). The fetcher's "real EIA" path was never functional: the EIA API v2 exposes **no hourly wholesale-price route** (the route it queried 404s for every year), and two US-scenario territories (**Southern Co./GA, Duke/SC are vertically integrated utilities with no public wholesale market**) have no LMP that could be fetched at all. So a fully-real price series is *not currently obtainable* for this footprint.

What the prices **are**: a calibrated diurnal model (regional May-2019 average × diurnal shape + noise) — structurally realistic *dynamics* (diurnal spread, cross-region differences) anchored to plausible levels.

| DC | Price model (calibrated to) | File |
|---|---|---|
| US-West | "CAISO-exposed" Western DC, ~$48/MWh | `data/prices/caiso.csv` |
| US-Central | MISO-like, ~$30/MWh | `data/prices/miso.csv` |
| US-Southeast-1 | Southern Co. territory, ~$38/MWh | `data/prices/southern_co.csv` |
| US-Southeast-2 | Duke Carolinas territory, ~$34/MWh | `data/prices/duke_carolinas.csv` |
| Global-EU | ENTSO-E NL-like, ~$46/MWh | `data/prices/entso_e_nl.csv` |
| Global-Asia | EMA Singapore (USEP)-like, ~$105/MWh | `data/prices/ema_singapore.csv` |

**Net demand, by contrast, IS real** (EIA-930; §2.3). **Why this is defensible for now:** every policy is scored against the *same* price series, so any price bias **cancels in the relative comparison** — and the comparisons (PPO vs baselines) are all we report. **What resolves it (future work, top priority):** real CAISO/MISO LMP (CAISO OASIS works but is rate-limited and historical access is finicky; `gridstatus` would help but its wheels would not build in this env) plus documented proxies for the non-ISO territories. The fetchers already accept `--year` and read the key from `.env`; the generalization study (§7.12) shifts the *real* net-demand signal to 2024 as a partial market-shift test in the meantime.

**Column**: `price_usd_kwh` — modeled $/kWh at 5-minute resolution.

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

**Why this is credible without leaning on R² (advisor's framing).** The justification for the power model is *not* its goodness-of-fit but its **agreement with the established literature model and its published constants**. The form `idle + slope·u` is the standard server model (Dayarathna survey Eq. 22; DeepEE Eq. 2; Fan et al. 2007), and `powerdata_2019` even defines "power utilization" exactly as our normalization (actual power / theoretical peak; Sakalkar et al.). Our fitted **idle fractions sit squarely inside the published range**: per-cell idle is **0.40–0.61 of peak draw** (table below), bracketed by the measured-server values in the literature — ≈0.38 (Haghshenas, idle 103 W / peak 270 W), ≈0.60 (Qureshi, "idle power around 60% of peak"), up to ~0.70 (older systems). So the constants are defensible on their own terms. **R² is a diagnostic, not the justification**: because *every* policy is scored by the *same* power model, any residual error cancels in the relative comparisons we report — the model's job is fair comparison, not point prediction. (The pooled R² ≈ 0.43 is low only because it blends four distinct per-cell lines; see below.)

**Where the missing variance actually lives — per-cell heterogeneity.** Extended diagnostics (per-cell fits, CPU+memory regression, binned means; all stored in `power_model_params.json`) decompose the low pooled R²:

| Model | R² | Note |
|---|---|---|
| Pooled CPU-only (above) | 0.433 | fleet-wide reference / fallback |
| Pooled CPU + memory | 0.460 | memory adds little (coef 0.08) |
| Binned means (20 CPU bins) | 0.950 | the mean CPU→power relationship is cleanly linear |
| **Per-cell CPU-only (used by the env)** | **0.75–0.80** | a: idle 0.53/slope 0.34 · b: 0.55/0.38 · c: 0.44/0.53 · d: 0.38/0.57 |

Most of the pooled residual is **between-cell heterogeneity**, not noise: each cell's machine mix has its own idle/slope (cell d is markedly more energy-proportional than cell a), and pooling four different lines into one scatters the fit. Memory as a second regressor is *not* the fix (+0.03). **The environment therefore uses the per-cell models** (`per_cell_power: true` in the scenario YAMLs; each site matched to its cell's calibration) — which both lifts the fit to R² 0.75–0.80 and makes the §1.2 CICS alignment literal: *"power models trained separately for each cluster"* (Radovanović et al. 2023). Per-DC power heterogeneity is also a real routing signal: the agent can prefer the more energy-proportional fleet (cell d) at high load.

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
| `backlog_weight` (λ_b) | 25 (calibrated; §7.14) | Penalize unserved service demand |
| `capacity_penalty_weight` (λ_κ) | 5.0 | Hard penalty for exceeding DC capacity |
| `peak_penalty_weight` (α) | 0.015 (calibrated) | Weight on `grid_mw² × net_demand_normalized` peak-contribution term |
| `deadline_penalty_weight` (λ_x) | 250 (calibrated; §7.8) | Penalty per unit of batch work that expires past deadline |

The `renewable_bonus` term from the prior on-site-solar formulation has been removed — with grid-only DCs, there is no "renewable fraction" to reward. Demand smoothing is now expressed directly through `peak_penalty`, which carries the same intent but targets the actual quantity (grid stress) rather than a proxy (local solar self-consumption).

### 3.6.1 Formal Problem Statement — what we optimize

The reward above is the per-step signal; stated as an optimization, the agent solves a **constrained cost-minimization over the full episode**. Indices: **i, j ∈ {1,…,N}** (data centers; `i` = site under consideration, `j` = running index for fleet-wide sums), **t, τ ∈ {1,…,T}** (time steps; `t` = current, `τ` = running index over earlier steps). Per-step controls: **f**ₜ (service-routing fractions), **δ**ₜ (drain rates), **h**ₜ (batch placement). Served CPU splits into service-served **σᵢ,ₜ** (service has first claim on capacity) and batch-served **Sᵢ,ₜ** (runs in the remainder):

```
minimize   J = Σₜ Σᵢ [ Eᵢ,ₜ + Φᵢ,ₜ + λ_b·Bᵢ,ₜ + λ_x·Xᵢ,ₜ ]         (P)
 f, δ, h

subject to, for all sites i and steps t:
  gᵢ,ₜ = (Pᵢ_idle + sᵢ·uᵢ,ₜ)·R                       power model
  σᵢ,ₜ = min(fᵢ,ₜ·Dₜ + Bᵢ,ₜ₋₁, κᵢ)                    service served (priority)
  Sᵢ,ₜ = min(hᵢ,ₜ·Σⱼ δⱼ,ₜ·Qⱼ,ₜ, κᵢ − σᵢ,ₜ)            batch served (remainder)
  uᵢ,ₜ = σᵢ,ₜ + Sᵢ,ₜ  ≤ κᵢ                            total served / capacity
  Σᵢ fᵢ,ₜ = 1,  fᵢ,ₜ ≥ 0,  δ,h ∈ [0,1]               valid allocations
  Bᵢ,ₜ = Bᵢ,ₜ₋₁ + fᵢ,ₜ·Dₜ − σᵢ,ₜ                      service conservation
  Qᵢ,ₜ₊₁ = Qᵢ,ₜ + aᵢ,ₜ − Sᵢ,ₜ − Xᵢ,ₜ                 batch queue
  Σ_{τ≤t} Sᵢ,τ ≥ (arrivals at i due by t)            deadline feasibility
```

where `Eᵢ,ₜ = πᵢ,ₜ·gᵢ,ₜ·1000·Δh` is energy cost (§3.3), `Φᵢ,ₜ = α·gᵢ,ₜ²·dᵢ,ₜ` the grid peak-contribution penalty weighted by **grid net demand** `dᵢ,ₜ` (§3.3), `B`/`X` are service backlog and expired batch, and **λ_x = 250** is the deadline (expiry) weight — renamed from a bare `w` so it does not clash with the demand symbol `wᵢ,ₜ` (= `vᵢ,ₜ + aᵢ,ₜ`). An arrival entering at step `τ` is due by `τ + Hᵢ`, where `Hᵢ = ⌈μᵢ(1+φ)/Δ⌉` is cell `i`'s deadline horizon (§3.7). **In words:** route the inflexible service load and defer the flexible batch so as to **minimize electricity cost plus grid-peak contribution, while serving all demand within capacity and completing batch by its deadline.**

**How we solve it.** Problem (P) is an *offline, full-information* statement; the deployed controller is **causal** — it cannot see future prices or demand. So we solve it with **model-free RL**: PPO maximizes the expected discounted return `E[Σₜ γᵗ rₜ]` with per-step reward `rₜ = −(`the bracketed cost`)` — i.e. the reward function (§3.6) is the negative per-step integrand of J. A **clairvoyant convex relaxation** of the same J (fluid allocation, known future) is a quadratic program whose optimum lower-bounds any policy — this is the optimality bound of §7.11.

**Lineage of the formulation (what to cite).** Problem (P) follows the **aggregate flexible/inflexible load-shaping problem of Google's CICS** (Radovanović et al. 2023) — retargeted from carbon to grid net-demand smoothing + electricity cost — with the **cost-minimization-over-distributed-DCs** structure of geographic load balancing (Qureshi 2009; Rao 2010; Liu-Wierman 2011), a **linear idle+slope power model** (Fan 2007; Dayarathna 2016), a **demand-charge-style peak term** (Liu et al. 2012; Vasques 2019), and **aggregate batch-with-deadline dynamics** (Grange 2018; Liu et al. 2012). The contribution is unifying these under a single learned policy with a grid-net-demand objective, rather than CICS's day-ahead carbon caps or the GLB papers' static convex programs (§8.6).

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

### 3.8 Batch Demand Input — Real Per-Tier Curves (primary) + Synthetic Generator (sensitivity)

**Primary input: real per-tier demand curves.** Each site's batch demand comes from `data/cells/cell_{x}_tiers.csv` — the measured cell aggregate split into non-deferrable `service_demand_norm` (SLO tiers) and deferrable `batch_demand_norm` (no-SLO tiers), with `service + batch = measured aggregate` at every timestep. This follows the data organization of Google's CICS (real aggregate flexible vs inflexible demand per cluster; §1.2, §8.6) and was adopted after the synthetic generator was found to produce unrealistically bursty arrivals (peak/mean ≈ 195 vs ≈ 1.24 in the trace — the full diagnosis and lesson are in §7.10). Curves are derived locally from the job extracts ([scripts/derive_tier_curves.py](scripts/derive_tier_curves.py)); a ground-truth BigQuery version is [extract_tier_curves.ipynb](extract_tier_curves.ipynb).

**Sensitivity tool: the synthetic generator.** The distribution-fitted `BatchArrivalGenerator` (sample inter-arrivals → per-job CPU/memory/task-count → aggregate → normalize volume) is retained for *controlled* experiments — varying load intensity, job mix, and arrival patterns independent of the trace — which is its original purpose in the Grange/Da Costa methodology below. It is no longer the primary demand path.

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

All numbers below come from the demand-smoothing formulation: grid-only DCs, reward = -(energy_cost + α × grid_mw² × net_demand_normalized + backlog + capacity penalties [+ deadline]), with α = 0.015. Each policy is run for one full 8,917-step episode (~31 days) under the same seed. **These are the final, artifact-free results**: ground-truth per-tier demand curves (§7.10), per-cell calibrated power models (§3.2), deadline-preserving queueing (§7.9), and the calibrated deadline penalty λ_x=250 (§7.8).

### 7.1 Multi-Seed Campaign — Canonical Results

The headline numbers come from the **multi-seed review campaign** ([scripts/run_review_campaign.py](scripts/run_review_campaign.py)): 5 training seeds × 3 algorithms × 4 configurations = 60 models, each evaluated deterministically on the full 31-day episode, with seed-level statistics (mean ± std, paired PPO-vs-best-DQN sign/Wilcoxon tests, bootstrap CIs). This is the **canonical context-aware (`ctx`) model set** — PPO's observation includes the static per-DC site context (idle, slope, capacity, batch_fraction; §3.4) that makes per-site heterogeneity learnable rather than memorized (§7.12). Costs in **millions of dollars**; "spatial-only" = the former "legacy" mode (routing only); "batch" = spatial + temporal.

| Config | Status Quo | Trough-Slot | Best DQN | **PPO (mean ± sd)** | **Δ vs SQ** | QP optimum | PPO gap | PPO vs DQN | PPO wins |
|---|---|---|---|---|---|---|---|---|---|
| US spatial-only | 9.848 | 11.235 | 9.809 | **9.570 ± 0.034** | **+2.82%** | 9.462 | 1.14% | +2.50% | 5/5 |
| US batch | 9.848 | 10.090 | 9.893 | **9.586 ± 0.080** | **+2.66%** | 9.443 | 1.51% | +3.20% | 5/5 |
| Global spatial-only | 14.062 | 14.324 | 13.348 | **12.621 ± 0.249** | **+10.25%** | 11.947 | 5.64% | +5.77% | 4/5 |
| Global batch | 14.062 | 13.757 | 13.457 | **12.246 ± 0.139** | **+12.91%** | 11.921 | 2.73% | +9.88% | 5/5 |

**Headline.** PPO is the best policy — learned or heuristic — in **all four configurations**: **+2.7–12.9%** vs the grid-unaware status quo, **+5.0–14.8%** vs the foresighted Trough-Slot heuristic, **+2.5–9.9%** vs the best DQN variant. It beats the best DQN in **19 of 20 seed-paired comparisons** (sign/Wilcoxon p = 0.031 in three configs; Global spatial-only is the exception at 4/5, p = 0.19). It sits **1.1–5.6% from the clairvoyant QP lower bound** (§7.11), serves **100% of service demand with zero deadline violations** (§7.14), and holds the flattest, lowest fleet draw. Savings scale with exploitable structure: modest under US-only diversity, large under global price/timezone spread.

**Where the savings come from (slope arbitrage).** Per-cell power calibration (§3.2) turned the low-diversity US scenario from "nothing to learn" into a real optimization: idle power is sunk, so each marginal unit of CPU is cheapest where the *slope* is lowest. PPO inverts the load distribution relative to the do-nothing policies — toward low-slope cells a/b (slope 0.34/0.38), away from high-slope c/d (0.53/0.57) — worth +2.8% with no price diversity at all (§7.5). In the Global scenario this compounds with price arbitrage (route away from high-priced Singapore toward cheap, low-slope US capacity).

**The foresighted heuristic loses everywhere.** Trough-Slot Lookahead has privileged 3-hour net-demand foresight yet loses every config by 5.0–14.8%, and in the US is the single most expensive policy (worse than doing nothing): its slack-grid routing *concentrates* load, raising the fleet peak the quadratic penalty punishes (load factor ≈ 0.86 vs PPO's ≈ 0.96), and it is blind to per-cell power. Foresight does not compensate for optimizing the wrong surface.

**The temporal lever, honestly sized.** At the measured 16–26% deferrable fractions (§2.2), batch deferral adds **+2.7 points** over spatial-only in the Global scenario (+10.3% → +12.9%) but is essentially neutral in the US (+2.8% → +2.7%), which has little price/timezone diversity to shift into. Spatial routing is the primary lever; deferral is a real but secondary, scenario-dependent one. (Earlier drafts' inflated +6% estimates were artifacts L1–L3, §7.8–§7.10.)

**DQN stability is the discrete-encoding story.** With 5 seeds, DQN's weakness is variance, not a single bad run: DQN std reaches ±$0.6M and flat-idx ±$0.9M, versus PPO's ≤ ±$0.25M. Neither discrete encoding dominates the other across configs, and the one PPO near-miss (Global spatial-only, 4/5) is where a flat-idx seed got close. PPO trained reliably in all four with one hyperparameter set. (This concerns our cell-aggregate formulation, not CFWS's per-VM setting where the encoding is reported effective — §8.7.)

> **Note on the single-seed tables.** Earlier drafts of §7.1–§7.4 carried full 11-policy baseline tables from a single pre-context model per config. Those are superseded by the campaign above; the full per-policy field (every heuristic) is still produced per run in `output/review_campaign_ctx/{config}.json` if the complete ranking is needed.

### 7.5 Per-DC Energy Cost Breakdown

**US Batch Mode** (per-DC energy cost):

| Policy | US-West (a) | US-Central (b) | US-Southeast-1 (c) | US-Southeast-2 (d) |
|---|---|---|---|---|
| PPO | $2,799,894 | $2,008,816 | $1,580,562 | $1,279,795 |
| Status Quo | $2,563,208 | $1,661,717 | $2,031,473 | $1,714,272 |
| Round Robin | $2,545,111 | $1,667,545 | $2,028,545 | $1,728,637 |

PPO **inverts** the load distribution relative to the do-nothing policies: it shifts work *toward* cells a/b and *away from* c/d. This is the **slope-arbitrage** behavior the per-cell power models enable: cells a/b have high idle but *low marginal* power (slope 0.34/0.38), c/d the reverse (0.53/0.57). Idle power is sunk regardless of routing, so each marginal unit of CPU is cheapest at the low-slope DCs — an emergent strategy that no baseline encodes, and the main source of PPO's US-legacy and US-batch wins.

**Global Batch Mode** (per-DC energy cost):

| Policy | Global-US-West | Global-US-Central | Global-EU | Global-Asia |
|---|---|---|---|---|
| PPO | $2,948,880 | $1,964,278 | $1,963,324 | $3,627,328 |
| Trough-Slot | $2,714,831 | $1,620,596 | $2,486,795 | $4,981,876 |
| Status Quo | $2,563,208 | $1,661,717 | $2,500,212 | $5,331,309 |

Global-Asia (Singapore, high EMA prices) dominates fleet cost, and PPO attacks exactly that: **−32% Singapore energy** and **−21% EU** vs the Status Quo, paid for with more (cheap, low-slope) US-West/Central consumption. Price arbitrage and slope arbitrage compound — this is where the 11.7% total saving over the status quo (§7.7) comes from.

### 7.6 Burst-Awareness Revisited (superseded experiment)

> This subsection documents an experiment whose **premise was dissolved by the environment corrections** of §7.8–§7.10. It is retained because the dissolution is itself a finding, and because the original experiment is part of the artifact narrative (§12).

**The original premise and experiments.** Tirmazi et al. (2020) document an extreme per-job heavy tail ("the top 1% of jobs consume over 99% of all resources," squared coefficient of variation > 23,000). The hypothesis: this tail manifests as *bursty aggregate batch arrivals*, making burst handling a high-leverage optimization axis. Two experiments tested it on the pre-correction environment: (1) a decomposition showing the RL agents' advantage over Round Robin was up to **2.1× larger in burst windows** (top-5% arrival timesteps); (2) an explicit `burst_severity` observation feature, which improved PPO by +0.7–1.2% (via tighter spatial routing, not differentiated drain timing) and *hurt* DQN-flatidx by 5.4%.

**Why the corrected environment dissolved the premise.** Each correction removed a leg:

1. **The burstiness was manufactured.** The synthetic generator injected each job's full demand at its arrival timestep, producing batch curves with peak/mean ≈ 195 vs ≈ 1.24 in the measured trace (§7.10). The "burst windows" the agents exploited were two orders of magnitude more extreme than reality — the 2.1× concentration finding was a property of the bug, not the workload.
2. **The measured deferrable slice is bounded.** At the ground-truth 16–26% batch fractions (§2.2), and with the aggregate batch curve being smooth (peak/mean ≈ 1.24), bursty *arrivals* barely perturb total fleet demand.
3. **The damage channel is closed.** With deadline-preserving queueing (§7.9), arrival spikes no longer cause expiry — violations are ≈ 0 for every reasonable policy — so there is nothing for burst-awareness to protect against.
4. **The real curves are deterministic.** The generator re-sampled arrivals per seed, making bursts stochastic events worth detecting; the measured tier curves are a fixed 31-day series whose "bursts" sit at known calendar positions an agent can learn implicitly. An explicit burst feature is a derived column of fixed data.

**What replaced it — where does the advantage actually live in time?** A no-retrain diagnostic ([scripts/analyze_peak_windows.py](scripts/analyze_peak_windows.py)) decomposes PPO's per-timestep advantage over the Status Quo by fleet-mean net-demand quartile:

| Net-demand quartile | US batch: share of savings | Global batch: share of savings |
|---|---|---|
| Q1 (slack grid) | 23.3% | 21.1% |
| Q2 | 23.5% | 26.7% |
| Q3 | 24.1% | 26.4% |
| Q4 (duck-curve neck) | **29.1%** | 25.8% |

PPO's savings are **nearly uniform in time** — a mild tilt toward high-net-demand windows in the US (Q4 earns 1.25× the per-step savings of Q1, driven by the peak penalty), and effectively flat in Global. This is the honest final picture: the slope- and price-arbitrage that drive the savings (§7.5) operate *continuously*, not in rare crisis windows. The old "advantage concentrates in bursts" result inverted into "the advantage is steady" once the bursts stopped being synthetic.

**What survives of the heavy-tail story.** The tail is real at the job level, but **it does not survive aggregation**: thousands of concurrent jobs smooth the cell-level curve to peak/mean ≈ 1.24 (§7.10). For cell-aggregate scheduling, Tirmazi's heavy tail matters through the *distribution fits* used in sensitivity experiments (§3.8), not as an operational burst phenomenon. The burst-aware models and sweep scripts are retained in the repository as the historical record of the superseded experiment.

### 7.7 Comparison to the no-optimization status quo

The tables above rank policies against each other. This section adds the external reference point: **how much grid-aware optimization saves over running the workload as-is** — the closest in-framework analogue to "Google's actual operation."

**The Status Quo baseline.** `StatusQuoPolicy` ([baselines.py](baselines.py)) serves each cell's own demand **locally and immediately** — no cross-DC routing, no temporal deferral. It is Borg's raw aggregate run grid-unaware: a CICS-style load-shaper switched *off* (§1.2, §8.6). Since the 2019 trace has no grid prices, no net demand, and no inter-cell routing, this is the correct counterfactual — *not* "Google's scheduler," which never solved the multi-DC grid-aware problem.

**Why the comparison is fair despite imperfect power fits.** Status Quo and every optimized policy are scored by the **same** per-cell power models (R² 0.75–0.80; §3.2), so any remaining model error **cancels in the relative comparison** — a shared bias does not change the *difference* between policies. The reported savings are therefore robust to residual power-model noise.

**Normalized power metric — load factor.** Beyond cost, every policy reports `load_factor = mean / peak` aggregate grid draw (emitted by `compute_summary`). Higher = flatter.

Multi-seed PPO means vs the status quo:

| Config | PPO mean cost | Status Quo cost | **Δ vs Status Quo** |
|---|---|---|---|
| US spatial-only | $9.570M | $9.848M | **+2.82%** |
| US batch | $9.586M | $9.848M | **+2.66%** |
| Global spatial-only | $12.621M | $14.062M | **+10.25%** |
| Global batch | $12.246M | $14.062M | **+12.91%** |

**PPO saves 2.7–12.9% over the grid-unaware status quo in every configuration**, with zero deadline violations, while *also* shaving the fleet peak ~10–14 MW (≈303 → ≈289). The saving scales with exploitable structure: modest under US-only diversity (where it comes from per-cell slope arbitrage, §7.5), large under global price/timezone diversity. This is the cleanest externally-facing result of the thesis, and it mirrors exactly what a CICS-style grid-aware layer contributes on top of Borg's grid-unaware operation. The visual profile is produced by [analysis/plot_power_profile.py](analysis/plot_power_profile.py) → `output/power_profile_comparison.png`.

### 7.8 Calibrating the deadline penalty — a sensitivity lesson

A non-obvious lesson surfaced when the corrected free+beb batch fraction (27–61%; §2.2) replaced the old tiny one. The deadline-violation penalty `deadline_penalty_weight × expired_demand` had been set to **2.0**, calibrated when batch was a negligible slice of load. At the realistic volume that value is **~100× too weak**: expiring a unit of deferred work costs \$2, while *serving* it costs ~\$150 of energy — so the cost-minimal strategy becomes to **dump batch and pay the trivial penalty**. Under that miscalibration, aggressive-expiry policies (Avoid-the-Ramp, Cheapest-First, even immediate-drain Status Quo) *beat* PPO, which conservatively served its committed work — an artifact, not a real ranking.

Because the penalty is linear in expired demand, the cost of any rollout at any weight is recoverable without re-evaluating: `cost(λ_x) = (cost − 2·expired) + λ_x·expired`. Sweeping `λ_x` over the US-batch rollouts inverts the ranking around **λ_x ≈ 183** (where low-expiry policies overtake high-expiry ones):

| Policy (US batch) | expired | cost @ w=2 | cost @ w=250 |
|---|---|---|---|
| **PPO** | 3,127 | $9.21M (10th) | **$9.99M (1st)** |
| Avoid-the-Ramp | 6,434 | $8.64M (1st) | $10.23M (last) |
| Status Quo | 5,059 | $8.87M | $10.13M |

**The calibration.** A principled value is the **energy cost of serving one unit** of deferred CPU — `slope · rated_power · 1000 · Δt · price ≈ $150` at a typical wholesale price — so that dropping committed work is never cheaper than doing it. We set **`deadline_penalty_weight = 250`** (≈ that energy cost at a moderately high price, comfortably above the empirical $183 crossover). At this value the optimization rewards *completing* deferred work, and PPO's spatial batch routing — which lowers forced expiry by balancing load across DCs (3,127 vs Status Quo's 5,059) — becomes a legitimate advantage rather than a liability.

**The general lesson:** in a deferral-with-deadlines reward, the deadline penalty must **scale with the value of the deferred work** (≈ its energy cost), not be a fixed small constant — otherwise the agent learns to discard work whenever the deferrable fraction is non-trivial. All batch-mode results in this thesis use the recalibrated **`λ_x = 250`** (the `deadline_penalty_weight` config key).

*Postscript: the expiry counts in this subsection are from the interim (request-based) batch fractions under which the lesson was learned. In the final environment — measured 16–26% fractions (§2.2) plus deadline-preserving queueing (§7.9) — deadline violations essentially vanish for all reasonable policies (§7.1, §7.14), and the calibrated penalty matters mainly for counterfactual batch-heavy sensitivity sweeps.*

### 7.9 Queue, don't dump — preserving deadlines under capacity pressure

A second batch-model artifact surfaced from a simple sanity check: **why does Status Quo expire ~5,000 units?** A policy that drains everything *immediately* should never discard work. The cause was in the step loop — drained batch that didn't fit under capacity was re-queued with a **1-step deadline (`t+1`)**, so any capacity-blocked work expired the very next step instead of waiting for a later trough. With the interim request-based free+beb fraction (27–61%; later measured at 16–26% of usage, §2.2), bursty arrivals routinely exceeded capacity, so this manufactured a large, *policy-independent* expiry floor — ~5,059 in **both** US and Global batch (identical, because the cells/capacity/arrivals are the same and only the grid differs: the tell that it was an artifact, not a result).

**This is the opposite of how Borg behaves.** The best-effort batch tier is *queued* — beb jobs wait for capacity, managed by the batch scheduler; they are not discarded because they could not run this instant. The "deadline" itself is a *synthetic* construct (Grange/Da Costa's `flexibility_factor`; §2.2 / §8.2), not a property of the trace. So the `t+1` re-queue contradicted both Borg's documented behavior and the deferral semantics we meant to model.

**The fix** ([multi_dc_env.py](env/multi_dc_env.py), Phases 3/7): the drain decision is now an *intended release*, and **only the batch actually served is removed from the pool**; everything unserved stays with its **original deadline** and is retried at a later trough, expiring only when genuinely overdue. Effect, on fixed-behavior baselines:

| US batch | expired (before) | expired (after) |
|---|---|---|
| Status Quo | 5,059 | **738** |
| Drain Immediately | 5,050 | **738** |
| Round Robin | 2,891 | **777** |
| Avoid the Ramp | 6,434 | 3,617 |

Status Quo's expiry collapses by **~85%** to a small genuine residual (bursts that exceed capacity even across the full deadline window). Avoid-the-Ramp stays high — but *legitimately*, because routing everything to one DC really does overload it. With the artifact removed, the batch comparison measures **scheduling skill** (placing deferrable work in the troughs) rather than who least-suffers from a modeling fuse. The §7.2–7.5 batch tables reflect this fix (and the calibrated `λ_x = 250`).

**The general lesson:** in a deferral-with-deadlines simulator, work that is merely capacity-blocked must **retain its deadline and queue**, not be discarded — otherwise saturation manufactures artificial deadline violations that swamp the real optimization signal.

### 7.10 Real per-tier curves replace synthetic arrival pulses — the burstiness lesson

The third (and final) batch-model artifact was found by comparing the synthetic batch curve against the trace it was supposed to mimic:

| Curve | peak / mean | max (fraction of capacity) |
|---|---|---|
| **Real cell-a aggregate** (includes all batch, post-Borg) | **1.24** | 0.69 |
| Synthetic batch curve (`BatchArrivalGenerator`) | **195** | **29.7** (≈30× one DC's capacity in a single 5-min bucket) |

The generator sampled each job's `cpu × tasks` and injected **all of it into the single arrival timestep** — the sampled *duration* was used only for deadlines, never for the curve shape. In reality (and in the Grange/Da Costa model it descends from), a job occupies capacity *at its rate over its runtime*; thousands of concurrent long jobs overlap, and the aggregate is smooth — which is precisely why the measured cell curves have peak/mean ≈ 1.2 despite Tirmazi's extreme per-job heavy tail. The single-step pulses explain the earlier anomaly that batch mode was *peakier* than legacy (load factor 0.796 vs 0.954) and every policy saturated at the same 357.5 MW: a modeling artifact, not a property of the workload.

**The fix — adopt CICS's data organization.** Rather than repairing the generator, we switched the batch-demand input to **real per-tier demand curves**, exactly the organization of Radovanović et al. (2023), *"Carbon-Aware Computing for Datacenters"* (real aggregate flexible vs inflexible demand per cluster, no synthetic generation in the loop):

- `data/cells/cell_{x}_tiers.csv`: per-timestep `service_demand_norm` (SLO tiers) and `batch_demand_norm` (no-SLO tiers), with **`service + batch = measured aggregate`** at every timestep (verified to machine precision, and the aggregate matches `cell_X.csv` exactly).
- **Ground truth** ([extract_tier_curves.ipynb](extract_tier_curves.ipynb)): `instance_usage` split by priority tier in BigQuery — measured usage, no reconstruction. Batch usage shares: **a=2.1%, b=6.1%, c=8.1%, d=9.7%** (§2.2). The batch curves are legitimately burstier than the total (peak/mean 3.5–21 vs 1.24) because the deferrable slice is small; that burstiness is now *real*, not manufactured.
- An interim local approximation ([scripts/derive_tier_curves.py](scripts/derive_tier_curves.py)) reconstructed the split from job *request* windows and **overestimated the deferrable share ~3–5×** (request-weighted ≈ 60–85% vs measured usage 16–26%) — a second instance of the requests-vs-usage gap (§2.2). It remains useful only when BigQuery is unavailable.

**Validation:** with the real curves (and the §7.9 queueing fix), a serve-everything-now policy in batch mode reproduces the legacy demand exactly — Status Quo: load factor 0.952, peak 302.8 MW, **zero** expiry, total cost equal to its legacy run **to the dollar** ($9,847,536 both modes; the invariant also holds under the per-cell power models and in the final §7.2 results). The batch machinery is now demand-neutral by construction; any cost difference between policies is *scheduling*, not artifacts.

**Two general lessons:** (i) a synthetic workload generator must conserve not just total volume but the **demand-presentation process** — jobs present their rate *over their duration*; validating the generated curve's shape statistics against the source trace (peak/mean here) is a one-line check that would have caught this immediately. (ii) **Resource *requests* are not resource *usage*** — any quantity weighted by requests (batch fractions, tier shares) can be off by an order of magnitude in an over-allocated system; only measured usage settles it. The generator is retained for controlled load-intensity sensitivity experiments (its original purpose in Grange et al.), no longer as the primary input.

---

### 7.11 Optimality gap — a clairvoyant QP lower bound

Reviewers rightly note that "beats the best heuristic we wrote" is weaker than "near-optimal." Our environment is **convex in the serving decisions** (linear power Eq. 1, linear energy cost, convex-quadratic peak penalty, linear queue/backlog dynamics), so a clairvoyant planner with full-episode foresight and fluid (unconstrained) allocation solves a **quadratic program whose optimum lower-bounds any policy** — causal or not. [scripts/compute_qp_optimum.py](scripts/compute_qp_optimum.py) assembles it directly for the Clarabel solver (per-origin-stream cumulative deadline constraints are exact because each stream has one deadline offset; the manual sparse assembly was **validated against a cvxpy reference at 10⁻⁹** before the full solve, since cvxpy's conic reduction overflows 32-bit indices at this scale on Windows). All four configs solve in 1–6 s.

| Config | QP optimum | PPO mean | **PPO gap** | Share of clairvoyant savings captured |
|---|---|---|---|---|
| US spatial-only | $9.462M | $9.570M | **1.14%** | 72% |
| US batch | $9.443M | $9.586M | **1.51%** | 65% |
| Global spatial-only | $11.947M | $12.621M | **5.64%** | 68% |
| Global batch | $11.921M | $12.246M | **2.73%** | 85% |

PPO captures **65–85% of the total clairvoyant savings over the status quo** and is **1.1–5.6% from a bound no causal policy can reach** (the bound has perfect foresight and no action-parameterization limits, so the true achievable gap is smaller). A useful by-product: the QP's spatial-only and batch optima differ by only **~0.2%**, independently confirming that the *intrinsic* value of temporal flexibility at these deferrable fractions is small (§7.1, temporal-lever paragraph) — PPO's larger empirical batch gains come mostly from the batch formulation training better routing, not from deferral per se.

### 7.12 Generalization — transfer tracks observability

A model evaluated on the same 31-day episode it trained on could be memorizing the calendar. We test the **frozen** policies (no retraining) on two distribution shifts via [scripts/eval_generalization.py](scripts/eval_generalization.py); the result is a clean, mechanistic story.

**Stage 1 — unobserved shift (held-out cells e–h).** Swap cells a–d for the four cells the agent never trained on (new workloads, tier mixes, per-cell power), via `extract_cells_eh.ipynb` → `*_eh.yaml`. The context-aware policies **fail where their edge is purely spatial** (US spatial-only −19.0%, Global spatial-only −27.5% vs the held-out status quo): the policy had learned per-site routing keyed to static parameters that — although now *in* the observation (§3.4) — never *varied* across the fixed-a–d training episodes, so the network received them as a constant bias with no gradient signal to depend on them. Adding the observation feature is **necessary but not sufficient**.

**Stage 2 — the fix (domain randomization).** Train with the env's `domain_randomization` flag ([env/multi_dc_env.py](env/multi_dc_env.py) `reset`): each episode permutes the compute bundles (workload/tiers/capacity/power/deadlines) across the market slots — destroying slot identity so the policy *must* read the context — and resamples power parameters within the measured 8-cell range (idle 0.35–0.60, slope 0.30–0.65) so held-out cells lie inside the training support. This **restores positive transfer in 3 of 4 configs** (US spatial +4.7%, US batch +4.0%, Global batch +5.2%). Global spatial-only stays brittle (−19.8%): its aggressive price-concentration strategy has no deferral pool to absorb mistakes on unseen capacity, and the (correctly expensive) backlog penalty amplifies the overload — while its **batch** counterpart, which *does* have the pool, transfers at +5.2% with the tightest variance of all. **Deferral capacity buys distribution-shift robustness** — a finding that did not exist before this experiment.

**Stage 3 — observed shift (real 2019→2024 net demand, M2b).** Keep cells a–d but swap net demand to **real EIA-930 May-2024** (a genuine duck-curve shift, corr ≈ 0.74; prices held fixed — see provenance caveat §2.4), via the `--year` fetchers → `*_yshift.yaml`. The **same frozen context-aware policies transfer near-perfectly** (US +3.4%/+3.1%, Global +10.7%/+13.2% — within ~0.5 point of their on-training result, two configs *better*).

**The unified statement:** generalization **succeeds along the axes the policy observes** (net demand, prices) and **fails along the one it cannot** (static site parameters) — until domain randomization makes the static axis learnable. This answers the memorization concern for the demand signal and prescribes the fix.

### 7.13 Robustness to inter-site movement cost

The spatial lever assumes free, instant fungibility. [scripts/movement_cost_sensitivity.py](scripts/movement_cost_sensitivity.py) charges a per-unit cost `w` on service routed away from its home cell (the SustainCluster transmission-cost idea, §8.8) and sweeps `w`, scoring the existing policies post-hoc — a **conservative lower bound**, since a movement-aware policy would route less and recover more. The Status Quo serves locally (zero movement), so it is the invariant reference.

| Config | Break-even `w*` ($/unit) | As multiple of unit-energy cost | Savings at $50/unit |
|---|---|---|---|
| Global spatial-only | $219 | 0.35× | +7.9% |
| Global batch | $331 | 0.53× | +11.0% |
| US spatial-only | $56 | 0.14× | +0.3% |

The 10–13% Global advantage **survives until movement costs reach 35–53% of the energy cost of serving a unit** — far above realistic data-egress prices — so it is *not* an artifact of free fungibility. The US margin is thinner (0.14×), honestly reflecting its smaller spatial diversity.

### 7.14 Deadline & backlog audit; environment calibration

**Backlog audit (are the savings real scheduling, not penalty-dodging?).** Because every policy is scored on the shaped objective (§3.6), `compute_summary` ([evaluate.py](evaluate.py)) now emits a full cost-component breakdown plus terminal state. Across all **ten PPO batch-mode seeds**: service served / demand = **1.0000**, terminal backlog = **0**, terminal batch pool ≤ 0.95 units, **zero** expired batch, peak transient backlog ≤ 3.9 units, backlog penalty ≤ 1% of total cost. PPO serves everything, on time — the savings are scheduling, not leakage.

**Two reward-calibration changes underpin that** (both following the §7.8 "scale the penalty to the value of the work" logic):
- **Service-backlog weight λ_b: 1.5 → 25.** At 1.5, parking a unit of SLO service for 3 hours cost ~$54 against ~$180 of price arbitrage — an exploitable loophole (delay interactive work when it's expensive). At 25 a 3-hour hold costs ~$900 ≫ $180, closing it; the audit above confirms it is closed.
- **Continuous action bound: ±1 → ±3** ([env/multi_dc_env.py](env/multi_dc_env.py) `ACTION_LOGIT_BOUND`). SB3 clips actions to the action box *before* the softmax/sigmoid decode, so the default ±1 structurally capped routing shares to [4.3%, 71%] and drain rates to [27%, 73%] — full concentration and multi-hour holding were impossible by construction, while the discrete wrappers (logits ±2/±3) were not so limited. ±3 restores expressiveness parity (shares to ~98.5%, drain 4.7–95.3%). Baselines and DQN bypass the box and are unaffected.

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
| No-SLO tiers (priority ≤ 115 = free + beb) as the deferrable pool | Both tiers are explicitly "no associated SLOs" (§2); beb is 100–115 per the trace documentation v3, which corrects the 110–115 range "mistakenly reported" in this paper (see §2.2, §12 row 11) |
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

PPO **beats Trough-Slot Lookahead in all four configurations**:

| Config | PPO (mean) | Trough-Slot | PPO Advantage |
|---|---|---|---|
| US spatial-only | $9.570M | $11.235M | **+14.8%** |
| US batch | $9.586M | $10.090M | **+5.0%** |
| Global spatial-only | $12.621M | $14.324M | **+11.9%** |
| Global batch | $12.246M | $13.757M | **+11.0%** |

This is meaningful because Trough-Slot has **privileged 3-hour future net-demand information** that PPO does not: PPO learns implicit forecasting *and* coordinates it with routing. Two structural weaknesses cost the foresighted heuristic: (i) its route-to-the-slack-grid rule **concentrates** load, raising the fleet peak (load factor ~0.86, peak ~333 MW vs PPO's ~0.96/289), which the quadratic peak penalty punishes — in the US this makes it the single most expensive policy, worse than doing nothing; (ii) it has no notion of the **per-cell power heterogeneity** (§3.2) — it optimizes against net demand only, while PPO additionally arbitrages each cell's marginal power slope (§7.5). Foresight does not compensate for optimizing the wrong surface.

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

Multi-seed campaign (5 seeds; mean ± std), ground-truth tier curves, per-cell power, w=250, λ_b=25, action bound ±3, full 31-day trace (§7.1):

| Scenario | PPO (mean ± sd) | best DQN (mean) | DQN std | PPO vs best DQN | PPO wins |
|---|---|---|---|---|---|
| US spatial-only | **$9.570M ± 0.034** | $9.809M | ±0.08–0.31M | **+2.50%** | 5/5 |
| US batch | **$9.586M ± 0.080** | $9.893M | ±0.08–0.16M | **+3.20%** | 5/5 |
| Global spatial-only | **$12.621M ± 0.249** | $13.348M | ±0.62–0.63M | **+5.77%** | 4/5 |
| Global batch | **$12.246M ± 0.139** | $13.457M | ±0.30–0.55M | **+9.88%** | 5/5 |

Three findings emerge:

1. **PPO is the best RL agent in all four configurations**, winning **19 of 20 seed-paired comparisons** (sign/Wilcoxon p = 0.031 in three configs; Global spatial-only 4/5, p = 0.19). In the multi-seed view the DQN story is **variance, not a single catastrophic run**: DQN std reaches ±$0.6M and flat-idx ±$0.9M versus PPO's ≤ ±$0.25M. (The earlier single-seed campaign *did* show a $77.8M routing-grid divergence in Global spatial-only and a flat-idx no-op in Global batch; with 5 seeds these read as the tails of a high-variance optimizer, which is the more honest framing.) PPO trained to a strong policy in all four configs with one hyperparameter set.
2. **Neither discrete encoding dominates the other**, and with per-cell power heterogeneity the 48-action migrate-fraction primitives are too coarse to capture the slope-arbitrage routing PPO learns (§7.5). *This says nothing about CFWS's own per-PM formulation* — there each migration event exposes many meaningful actions (CFWS reports 5.67–13.22% brown-energy reduction, 46.49–86.53% migration reduction on their AZ/CA/OR/LA setting); on their formulation the encoding is reportedly effective.
3. **Continuous actions matter most where the optimization surface is richest.** PPO's margin over the best DQN grows with available structure: +2.5% (US spatial-only, slope arbitrage only) → +9.9% (Global batch, slope + price + timezone + deferral). Fine-grained fractional routing is what converts surface richness into savings.

#### PPO as the contribution beyond CFWS

PPO is the best policy — RL or heuristic — in **all four configurations**, with **zero deadline violations** in both batch modes (§7.1, §7.14). Its continuous action space (routing fractions + drain rates + batch routing) is something CFWS's discrete one-VM-migration encoding cannot express without reformulation, and the results tie its advantage to exactly that expressiveness: the slope- and price-arbitrage routing it learns (§7.5) requires fine-grained, per-DC fractional control. The contribution claim: **a continuous-action PPO formulation of aggregate multi-DC load-shaping beats foresighted heuristics (+5.0–14.8%), discrete DQN variants (+2.5–9.9%, 19/20 seeds), and the grid-unaware status quo (+2.7–12.9%) across every tested configuration — landing 1.1–5.6% from a clairvoyant QP bound (§7.11), completing all committed work** — with the caveat that DQN's instability is a training-stability property of our formulation, not evidence against CFWS's design on theirs.

### 8.8 Survey context — Lin et al. (2024) and Wu et al. (2025)

**"A systematic review of green-aware management techniques for sustainable data center"** (Lin, Lin, Peng, Huang, Lin, Li, 2024) provides the broader sustainable-DC landscape view. The categories of workload management, virtual resource management, energy management, thermal management, and waste heat recovery surveyed there place this thesis within "workload management + energy management for grid-aware multi-DC operation." For the multi-DC scheduling subarea specifically, **Wu et al. (2025), "Task Scheduling in Geo-Distributed Computing: A Survey"** (arXiv:2501.15504) is the most recent systematic review and covers the geo-distributed task-scheduling thread that this thesis sits within. Additional green-DC landscape reviews — *"Energy efficiency in cloud computing data centers: a survey on software technologies"* and *"A systematic review on effective energy utilization management strategies in cloud data centers"* — corroborate the workload-/energy-management framing.

**Open benchmark environments — and why we do not run inside them.** **SustainDC (Naug et al., NeurIPS 2024, "SustainDC: Benchmarking for Sustainable Data Center Control")** provides multi-agent Gymnasium environments for *within*-DC control (workload shifting, cooling, battery); **SustainCluster** (Hewlett Packard Enterprise, github.com/HewlettPackard/sustain-cluster) is its geo-distributed successor — a centralized scheduler dispatching or deferring *individual tasks* of the Alibaba 2020 GPU trace across 20+ locations, optimizing energy cost, carbon, SLA, and per-GB transmission overheads. These environments mean our novelty claim must be (and is) narrower than "no open formulation exists": our contribution is the **measured per-tier demand decomposition of ClusterData 2019, PowerData2019-calibrated per-cell power, and the grid net-demand (duck-curve) objective**, at CICS-style aggregate granularity. We deliberately do **not** run our experiments inside SustainCluster: answering our question there would require replacing both its objective (carbon → grid net demand) and its action model (per-task dispatch → aggregate tier control — the granularity §1.2 argues for), while abandoning the Google-trace data contributions; at that point the environment would no longer be SustainCluster. The two are complementary benchmarks: SustainCluster's transmission-cost model is the template for our movement-cost sensitivity analysis, and cross-validating our policies inside it is listed as future work.

### 8.9 Positioning of Our Contribution

Stated against the lineage above:

1. **Cell-as-proxy-DC modeling exercise.** We treat four ClusterData 2019 cells (a–d) as four geographically distributed hyperscale DCs — what such DCs' workloads would look like if they had cell-level inter-DC heterogeneity. This is *not* what Tirmazi et al. (2020) intended when documenting the trace (they don't claim the cells are geographically distinct), and no prior published work does exactly this. It is a defensible modeling exercise rather than a dataset-grounded claim (see §2.1 and §3.2 for the explicit modeling assumptions on workload-as-shape and `rated_power_mw`-as-magnitude).
2. **Grid demand smoothing as a first-class objective**, via a peak-contribution penalty against actual EIA-930 net demand timeseries — rather than the on-site-renewable framing that dominates academic prior work.
3. **Continuous action space (PPO)** enabling fine-grained joint routing + drain decisions. PPO is the best policy — RL or heuristic — in **all four configurations** (+2.5–9.9% over the best DQN variant across 19/20 seeds, +5.0–14.8% over the foresighted Trough-Slot heuristic, within 1.1–5.6% of the QP optimum), with **zero deadline violations** in batch mode. Its advantage is tied to expressiveness: the slope- and price-arbitrage routing it learns (§7.5) requires per-DC fractional control that the discrete encodings cannot represent (§8.7).
4. **Real-data grounding** *(with one documented exception)*: ClusterData 2019 (per Tirmazi 2020) for workloads, `powerdata_2019` (per Sakalkar 2020) for per-cell power, EIA-930 (CISO, MISO, SOCO, DUK) for grid net demand, NSRDB solar irradiance as forecast features — **but prices are documented synthetic, not real LMP (§2.4)**, the thesis's primary data limitation and top future-work item.
5. **Operational relevance**: the problem class is the same one **Google's CICS (Radovanović et al. 2023)** solves in production at 20+ DCs across 4 continents. This thesis contributes a published methodology + reproducible Gymnasium env for that problem class.

Honest framing of what this thesis is *not*: it is not a head-to-head comparable against CFWS (different action paradigm, different state granularity, different objective), nor a reimplementation of Google's CICS (closed-source operational system). It is a self-contained academic exploration of multi-DC + cell-aggregate + RL + grid-aware scheduling, with the cell-as-DC and 100 MW magnitude assumptions stated explicitly rather than hidden.

The primary empirical findings (5-seed campaign) are:

- **PPO wins every configuration** — +2.7% (US batch) to +12.9% (Global batch) over the grid-unaware status quo, +5.0–14.8% over the foresighted Trough-Slot heuristic, +2.5–9.9% over the best DQN (19/20 seeds), within **1.1–5.6% of a clairvoyant QP bound** (§7.11), with zero deadline violations and the lowest fleet peaks (~289 vs ~303 MW).
- **Spatial routing is the primary lever; its strength scales with exploitable structure.** Under global diversity PPO saves 10.3% (spatial-only) over the status quo from price + timezone + power-heterogeneity arbitrage. Even in the low-diversity US scenario, **per-cell power calibration (§3.2) creates a real signal** — marginal-cost (slope) arbitrage (§7.5) — worth +2.8%.
- **Temporal deferral is a secondary, scenario-dependent lever: +2.7 points in Global, ~neutral in US**, at the *measured* 16–26% deferrable fractions; the QP confirms the intrinsic value of deferral here is small (§7.11).
- **Generalization tracks observability (§7.12):** frozen policies transfer along observed axes (real 2019→2024 net-demand shift, +3–13%) but not the unobserved static-site axis (held-out cells) until domain randomization restores it (+4–5%).
- **Three modeling lessons as methodological contributions (§7.8–§7.10):** the deadline penalty must scale with the energy value of deferred work; capacity-blocked work must queue, not be discarded; and synthetic workload generation must conserve the demand-presentation process (requests ≠ usage). Each artifact, while present, *inverted* the experimental ranking.

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

### 9.1 File-by-file reference (what every file does)

The authoritative map of the repository — grouped by role. Paths are clickable.

**Environment (the simulator the agent acts against)**
- [env/multi_dc_env.py](env/multi_dc_env.py) — the Gymnasium `MultiDCEnv`. Observation/action spaces, the `_step_legacy` (spatial-only) and `_step_batch` (spatial + temporal) loops, reward (§3.6), `_compute_dc_cost` (per-cell power → energy + peak + backlog + capacity components), the deadline/queue dynamics (§7.9), the **site-context observation** and **domain-randomization** in `reset` (§7.12), and the `ACTION_LOGIT_BOUND = 3` action box (§7.14). The heart of the project.
- [env/dc_site.py](env/dc_site.py) — `DataCenterSite`: per-DC demand/price/net-demand accessors, the real per-tier `service_curve`/`batch_curve`, the per-cell `power_model`, the `BatchPool` deferrable queue, and `batch_fraction` (Tirmazi-cited).
- [env/power_model.py](env/power_model.py) — linear idle+slope `PowerModel`; `per_cell_from_json` builds the four per-cell models (§3.2).
- [env/data_loader.py](env/data_loader.py) — `load_scenario`: reads a scenario YAML, loads all CSVs, fits fleet capacities, wires real tier curves + per-cell power onto each site.
- [env/discrete_wrapper.py](env/discrete_wrapper.py) — 759-action routing-grid wrapper for DQN.
- [env/cfws_style_wrapper.py](env/cfws_style_wrapper.py) — 48-action CFWS-style flattened-index wrapper for DQN (§8.7).
- [env/scenarios/](env/scenarios/) — `us_model.yaml`, `global_model.yaml` (primary); `*_eh.yaml` (held-out cells e–h, §7.12); `*_yshift.yaml` (2024 net-demand shift, §7.12).

**Agents, baselines, training, evaluation**
- [train.py](train.py) — PPO trainer (`make_env` + SB3 PPO); flags include `--batch-mode`, `--peak-penalty-weight`, `--seed`, `--domain-rand`.
- [train_dqn.py](train_dqn.py) — DQN trainer for both discrete wrappers (`--action-scheme cfws-style` selects flat-idx).
- [baselines.py](baselines.py) — the 9 heuristic policies incl. `StatusQuoPolicy` (local, no deferral — the headline reference) and `TroughSlotLookaheadPolicy` (foresighted heuristic).
- [evaluate.py](evaluate.py) — `run_episode` + `compute_summary` (cost, load factor, peak, **and the §7.14 component breakdown + backlog audit metrics**).

**Review-response experiment scripts (one per major result)**
- [scripts/run_review_campaign.py](scripts/run_review_campaign.py) — the **multi-seed campaign** (M1): trains 5 seeds × 3 algos × 4 configs (resumable via `--tag`, `--algos`, `--domain-rand`) and emits seed-level stats → `output/review_campaign[_ctx|_dr]/`.
- [scripts/compute_qp_optimum.py](scripts/compute_qp_optimum.py) — the **clairvoyant QP lower bound** (M3, §7.11); Clarabel sparse assembly, cvxpy-validated, `--validate` for the small-T cross-check.
- [scripts/eval_generalization.py](scripts/eval_generalization.py) — **frozen-policy held-out evaluation** (M2/M2b, §7.12); `--axis cells|market`, `--model-dir`.
- [scripts/movement_cost_sensitivity.py](scripts/movement_cost_sensitivity.py) — **movement-cost sweep** (M6, §7.13).
- [scripts/analyze_peak_windows.py](scripts/analyze_peak_windows.py) — the net-demand-quartile advantage diagnostic that replaced the burst study (§7.6).
- [scripts/validate_caiso_prices.py](scripts/validate_caiso_prices.py) — attempted real-CAISO price provenance check (blocked by tooling; see §2.4) — retained as the documented attempt.

**Data pipeline (extraction & fitting)**
- [extract_clusterdata2019_full.ipynb](extract_clusterdata2019_full.ipynb) — the original Colab notebook: per-cell aggregate demand, machine fleets, **per-cell power calibration** (Dataset 2), and job metadata with the ≤115 tier classification (Dataset 4).
- [extract_tier_curves.ipynb](extract_tier_curves.ipynb) — standalone Colab: ground-truth per-tier (service/batch) usage curves, `priority ≤ 115` (§7.10).
- [extract_cells_eh.ipynb](extract_cells_eh.ipynb) — standalone Colab: held-out cells e–h (curves, machines, power, durations) for the generalization test (§7.12).
- [scripts/refit_freebeb_local.py](scripts/refit_freebeb_local.py) — refits batch distributions from `jobs_*.csv` under free+beb (§2.2).
- [scripts/derive_tier_curves.py](scripts/derive_tier_curves.py) — local (request-window) tier-curve approximation; superseded by the notebook but kept.
- [scripts/grange_generator.py](scripts/grange_generator.py) — verbatim Grange Listing 1 reproduction (§8.2).
- [preprocess/price_fetcher.py](preprocess/price_fetcher.py) — price series generator (**documented synthetic**, §2.4); `--year`, reads `.env`.
- [preprocess/net_demand_fetcher.py](preprocess/net_demand_fetcher.py) — **real EIA-930** net demand; `--year`, reads `.env`.

**Figures & paper**
- [scripts/build_paper_figures.py](scripts/build_paper_figures.py) — all 7 paper figures (results+QP, tiers, power, profile, peak-window, generalization, movement cost).
- [scripts/build_thesis_paper.py](scripts/build_thesis_paper.py) — assembles `thesis_paper.docx` (two-column, Google-Docs-importable).
- [analysis/plot_power_profile.py](analysis/plot_power_profile.py) — the Status-Quo-vs-PPO fleet power-profile plot.

**Key data & outputs**
- `data/cells/cell_{a..h}.csv` + `cell_*_tiers.csv` — aggregate + per-tier demand. `data/machines/`, `data/power_model_params.json` (per-cell fits, all 8 cells), `data/net_demand/` (real 2019), `data/net_demand_2024/` (real 2024), `data/prices/` (synthetic), `data/jobs/batch_distributions_*.json`.
- `output/review_campaign[_ctx|_dr]/` — per-config eval JSONs, `stats.json`, `qp_optimum.json`, `generalization_*.json`, `movement_cost_sensitivity.json`. `models/review[_ctx|_dr]/s{seed}/` — the trained models.
- `.env` — `EIA_API_KEY` (gitignored; read by both fetchers).

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

*(All numbers are the 5-seed multi-seed campaign, §7.1.)*

1. **PPO wins all four configurations — against every heuristic, both DQN variants, and the status quo — with zero deadline violations, across five seeds.** Margins: **+2.7–12.9%** over the grid-unaware Status Quo, **+5.0–14.8%** over the foresighted Trough-Slot heuristic, **+2.5–9.9%** over the best DQN (winning **19 of 20** seed-paired comparisons), while shaving the fleet peak ~303 → ~289 MW. This headline only became trustworthy after the modeling artifacts were fixed (takeaway 5), the per-cell power calibration exposed the routing surface (takeaway 3), and the multi-seed campaign confirmed the margins are not seed noise.

2. **PPO is near-optimal, not just "best heuristic we wrote."** It lands **1.1–5.6% from a clairvoyant convex-QP lower bound** (§7.11) that no causal policy can reach — capturing 65–85% of the total achievable savings over the status quo. The QP also shows the intrinsic value of temporal deferral here is small (spatial-only and batch optima differ ~0.2%).

3. **Per-cell power calibration turned the US scenario from "nothing to learn" into a real optimization.** Under pooled power all DCs were energetically identical; with each cell's measured idle/slope (R² 0.75–0.80, §3.2), **marginal-cost (slope) arbitrage** emerges — idle power is sunk, so load is cheapest where the slope is lowest — worth **+2.8%** in the US with no price diversity at all. An emergent strategy no baseline encodes, and a direct payoff of calibrating power per cluster as CICS does.

4. **Spatial routing is the primary lever; temporal deferral is secondary and scenario-dependent.** Global spatial-only already saves +10.3% over the status quo (price + timezone + slope arbitrage). At the measured 16–26% deferrable fractions, batch deferral adds **+2.7 points in Global** but is **~neutral in the US** — and the QP confirms the small intrinsic temporal value.

5. **Generalization tracks observability (§7.12) — a three-stage result.** Frozen policies transfer near-perfectly across a real 2019→2024 *net-demand* shift (observed axis, +3–13%) but **fail** on held-out *cells* (unobserved static-site axis, −19 to −27%) — until **domain randomization** makes the static context learnable and restores transfer (+4–5% in 3/4 configs). Corollary: **deferral capacity buys distribution-shift robustness** (the one brittle config, Global spatial-only, has no pool; its batch counterpart transfers fine).

6. **The spatial advantage is robust to realistic movement costs (§7.13).** The 10–13% Global savings survive a per-unit inter-site movement charge up to **0.35–0.53× the energy cost of serving a unit** — far above realistic egress prices — so it is not an artifact of free fungibility.

7. **The foresighted heuristic loses to learning on every config (+5.0–14.8%).** Trough-Slot has perfect 3-hour foresight but optimizes the wrong surface: its slack-grid concentration raises the peak (load factor 0.86 vs PPO's 0.96) and it is blind to per-cell power. In the US it is the single most expensive policy. Foresight does not compensate for a mis-specified objective.

8. **DQN's story is training variance, not encoding.** With 5 seeds, DQN std reaches ±$0.6M and flat-idx ±$0.9M vs PPO's ≤±$0.25M; neither discrete encoding dominates. PPO trained reliably everywhere with one hyperparameter set. The continuous/discrete choice matters for both expressiveness (slope arbitrage needs fractional control) and stability.

9. **Four reward/observation calibrations, each tied to the value of the work (§7.8, §7.14):** deadline (expiry) penalty λ_x = 250 (scale to deferred-work energy value, or the agent discards it); service-backlog λ_b = 25 (or it parks SLO traffic to dodge price spikes); queue-don't-dump (Borg queues, it doesn't expire blocked work); and the ±3 action bound (or SB3 clipping structurally caps concentration/holding). Each, mis-set, inverted or distorted the ranking. The backlog audit confirms PPO serves 100% of demand with zero leakage.

10. **Three modeling lessons as transferable methodological contributions (§7.8–§7.10):** the deadline penalty must scale with deferred-work energy value; capacity-blocked work must queue, not dump; and synthetic generation must conserve the demand-presentation process — with **requests ≠ usage** (request-weighted tier shares overestimated the deferrable fraction 3–5×). Each artifact, while present, inverted the ranking; the corrected results exist because each was found and fixed.

11. **Honest provenance (§2.4):** demand, per-cell power, and net demand are real measured data; **prices are documented synthetic** (no real hourly LMP is obtainable for the non-ISO US territories, and the EIA API has no hourly price route). Relative comparisons are unaffected (shared price series), and resolving this with real LMP is the top future-work item.

## 12. Change Log — How the Final Results Were Reached

The results in §7 were not produced by a single clean run; they are the product of an iterative validation process in which **five substantive modeling corrections** were found, fixed, retrained, and documented. This log records the journey honestly — both because several corrections inverted the experimental rankings (making the lineage essential context for anyone comparing against earlier drafts), and because the corrections themselves are among the thesis's contributions.

| # | Change | Trigger | Effect on results |
|---|---|---|---|
| 1 | **Distribution-fitting fixes**: select by KS *D* statistic (the p-value underflows to 0 at n≈10⁵ and silently defaulted every fit to the first candidate); discrete (negative-binomial) fit for task counts; FINISH-only durations; last-terminal (not first-eviction) end times | "Isn't it suspicious we're getting KS=0 for every result?" | Fits became meaningful: log-normal durations, Weibull/log-normal inter-arrivals, nbinom task counts — instead of "exponential everywhere" |
| 2 | **Batch definition corrected twice**: `scheduling_class ≤ 1 AND priority < 200` → strict beb tier (110–115) → **no-SLO tiers (free ≤ 99 ∪ beb 110–115)** per Tirmazi §2 | Validation showed the old extraction was 94% *production* (priority-200) jobs; strict beb alone was ~0% of CPU | The deferrable class became defensible and citable; all batch data regenerated |
| 3 | **Batch spatial routing added** (action 2N → 3N): drained batch is pooled and routed by a second softmax head | Deferrable work has no latency SLO — pinning it to its home DC while routing latency-sensitive service inverted physical reality | Deferred work can shift in space *and* time (CICS's two levers); all batch models retrained |
| 4 | **Status Quo baseline + load-factor metric**: serve-locally-immediately reference; `load_factor = mean/peak` | "Can I tell how much my run is better than Google's actual data?" | The externally-facing result (§7.7) — savings vs the grid-unaware status quo — became measurable |
| 5 | **Deadline penalty calibrated 2.0 → 250** (§7.8) | At realistic batch volumes, expiring a unit cost \$2 vs ~\$150 to serve it — dumping won; ranking inverted at w≈183 | Aggressive-expiry "winners" sank; completing work became rational |
| 6 | **Queue-don't-dump** (§7.9): capacity-blocked batch keeps its original deadline instead of a 1-step fuse | "Does the status quo dump any? In theory it shouldn't, right?" — it expired 5,059 units, identically in both scenarios (the tell) | The policy-independent expiry floor vanished (5,059 → ~0); the "concentrators win by dumping" anomaly disappeared |
| 7 | **Real per-tier demand curves replace the synthetic generator** (§7.10): first a local request-window approximation, then ground truth from `instance_usage` split by tier | Generated batch curves had peak/mean ≈ 195 vs ≈ 1.24 in the trace — the generator injected whole jobs as single-step pulses | Batch demand became measured reality; the demand-neutrality invariant (Status Quo batch ≡ legacy to the dollar) holds by construction |
| 8 | **Requests ≠ usage**: ground-truth tier split showed the deferrable share is **2–10%** of usage, not the 27–61% suggested by request-based proxies (Borg over-allocates best-effort tiers ~5–13×) | Comparing the local approximation against the BigQuery ground truth | Honest sizing of the temporal lever (+1.5–1.6%); batch-heavy mixes moved to sensitivity territory |
| 9 | **Per-cell power calibration** (§3.2): R² 0.43 (pooled) → 0.75–0.80 (per cell); env consumes per-cell models | Extended power diagnostics showed the pooled residual was *between-cell heterogeneity*, not noise | **US flipped from "PPO loses" to "PPO wins"** — per-DC slope heterogeneity is a real routing signal (slope arbitrage, §7.5); full 12-model retrain |
| 10 | **Burst study superseded** (§7.6): premise dissolved by #7; replaced with a net-demand-window diagnostic | The burstiness the burst study analyzed was the #7 artifact | "Advantage concentrates in bursts (2.1×)" inverted to "advantage is nearly uniform in time (~1.25×)" |
| 11 | **beb tier band corrected to 100–115** (`deferrable = priority ≤ 115`): the trace documentation v3 explicitly corrects the 110–115 range "mistakenly reported" in Tirmazi et al. (2020) §2, which our pipeline had inherited by citing the paper | Peer review (M9) flagged the paper-vs-trace-docs discrepancy; our own earlier analysis had flagged the unassigned 100–109 band ("?") holding 16–30% of requested CPU | Re-extracted: deferrable shares rose to **16–26%** (matching Tirmazi's ~20% best-effort average); requests-overestimate now 3–5× |

**Peer-review response (M1–M9) — second campaign.** A reviewer pass prompted a settled-environment rebuild and a multi-seed campaign; these moved the headline from single-seed to statistically-backed.

| # | Change | Trigger | Effect on results |
|---|---|---|---|
| 12 | **Action bound ±1 → ±3** (M5) + **service-backlog λ_b 1.5 → 25** (M4) | SB3 clips actions before decode, capping PPO's reachable concentration/drain; λ_b=1.5 made parking SLO traffic profitable | Expressiveness parity restored; backlog audit confirms 100% served, zero leakage (§7.14) |
| 13 | **Multi-seed campaign** (M1): 5 seeds × 3 algos × 4 configs = 60 models + seed-level stats | Single seed can't support "wins every config" | **PPO wins 19/20 seed comparisons** (p=0.031); DQN reframed as high-variance, not single-divergence (§7.1) |
| 14 | **Clairvoyant QP optimum** (M3): convex-QP lower bound, Clarabel, cvxpy-validated 1e-9 | "Beats best heuristic" is weaker than "near-optimal" | PPO **1.1–5.6% from optimum**; temporal value shown intrinsically small (§7.11) |
| 15 | **Site-context observation + domain randomization** (M2): static per-DC params observed, then randomized in training | Held-out cells e–h exposed slot-memorization (frozen policies −19 to −27%) | DR restores transfer (+4–5%, 3/4); **generalization tracks observability** (§7.12) |
| 16 | **Real 2019→2024 net-demand shift** (M2b) + **movement-cost sweep** (M6) | Test transfer on a real market signal; test free-fungibility assumption | Observed-axis transfer +3–13%; Global savings survive to 0.35–0.53× unit energy (§7.13) |
| 17 | **Prices found synthetic, not real LMP** (M7): committed CSVs byte-identical to the generator; EIA has no hourly price route; GA/SC have no market | Tried to fetch real prices for M2b; discovered the fetch path never worked | Provenance corrected (§2.4): net demand real, **prices documented synthetic**; real LMP is top future work |
| 18 | **Related work + novelty softened** (M8): Qureshi/Rao/Liu-Wierman geographic-load-balancing lineage; SustainDC/SustainCluster differentiated | Reviewer noted missing lineages and too-strong novelty claim | Novelty narrowed to measured per-tier demand + per-cell power + grid-net-demand objective (§8.7–§8.9) |

**The arc in one sentence:** every correction moved the environment *toward the measured trace* — and each step toward reality first *shrank* an inflated finding (the temporal lever, the burst concentration) and then *revealed* a genuine one (slope arbitrage, the steady-state advantage), ending with PPO winning all four configurations on an environment whose batch machinery is provably demand-neutral.

**Reproducibility of the lineage:** each correction is an individual commit on `master` with the diagnosis in its commit message; the corrected data artifacts are regenerable via `extract_tier_curves.ipynb` (ground truth), `scripts/refit_freebeb_local.py` + `scripts/derive_tier_curves.py` (local approximations), and the enhanced Dataset 2 cell of `extract_clusterdata2019_full.ipynb` (per-cell power).
