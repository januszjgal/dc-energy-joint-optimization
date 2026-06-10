# Slide Section 1 — Google ClusterData 2019

> One section of the slide deck. Each `### Slide X` block below maps to a single slide. Tables/lists can stay as bullet-style; long prose should be broken into talking points or moved to speaker notes.

---

### Slide 1 — What is Google ClusterData 2019?

**The dataset**

- Published by Google as part of the `google-cluster-data` public BigQuery project
- Companion paper: **Tirmazi et al. (2020), "Borg: the next generation."** EuroSys '20
- 2nd-generation trace (the 2011 version is the highly-cited predecessor)
- Covers **8 Borg cells (a–h)** for the entire month of **May 2019** — ~31 days
- **~96,400 machines** across the 8 cells (~12,000 machines per cell)
- Published as queryable BigQuery tables (~350 GiB compressed per cell)

**A "cell" in Borg terminology** = a single management unit — one cluster scheduler master + the machines it runs on. Cells are the natural unit of cluster-level analysis in this dataset.

**Why this trace** (for our thesis)
- The standard benchmark for cloud-scheduling research (hundreds of citations)
- More modern than the 2011 trace (different schema, richer job-priority structure, batch-tier metadata, alloc sets)
- Includes a **companion PowerData2019** dataset with measured per-PDU power — lets us calibrate a real power model rather than guess

> **Speaker note**: Tirmazi documents these 8 cells but does not claim they are geographically distributed. Treating cells as proxy hyperscale DCs is our modeling choice (covered later); the trace itself is unit-agnostic about geography.

---

### Slide 2 — What's actually in the trace

Per cell, BigQuery exposes these tables (the ones we touch):

| Table | What it contains | Granularity |
|---|---|---|
| `clusterdata_2019_{cell}.instance_usage` | Per-VM resource usage samples (CPU cores, memory) | 5-minute samples per running instance |
| `clusterdata_2019_{cell}.machine_events` | Machine join/leave/update events + capacity | Per-machine, event-driven |
| `clusterdata_2019_{cell}.collection_events` | Job submit/schedule/finish events + priority + scheduling_class | Per-job, event-driven |
| `clusterdata_2019_{cell}.instance_events` | Per-VM events + resource_request | Per-instance, event-driven |
| `powerdata_2019.cell{a..h}` | **Measured power per PDU** + `bad_measurement_data` flag | Per-PDU, sub-minute |

**Two key numbers normalize everything**
- **NCU** (Normalized Compute Units): CPU rescaled to [0, 1] by max machine size → CPU values are always dimensionless
- **NMU** (Normalized Memory Units): same for memory
- **`measured_power_util`** in PowerData2019 is also normalized to [0, 1] per PDU

This means all numbers we work with are dimensionless fractions of capacity — clean for modeling, but means "actual CPU cores" or "actual watts" require multiplying back by physical capacity (which isn't public).

---

### Slide 3 — What we extract, and why

We pull **6 derived datasets** from the trace via [`extract_clusterdata2019_full.ipynb`](../extract_clusterdata2019_full.ipynb) (run in Colab against BigQuery). Each dataset feeds a specific part of the env:

| # | Dataset | What it is | Feeds |
|---|---|---|---|
| 1 | **Per-cell aggregate CPU** | Sum of `average_usage.cpus` over all instances per 5-min bucket, divided by cell capacity → a single [0,1] timeseries per cell, ~8,917 timesteps (31 days × 288 steps/day) | Workload signal for each DC in the env |
| 2 | **Power model** | Hourly (CPU util, measured power util) joined across cells a–d → linear regression → `P(u) = 0.479 + 0.444·u`, R² = 0.43 | Converts CPU utilization to MW in the env |
| 3 | **Machine fleet** | Per-machine (CPU capacity, memory capacity, cell) — ~10K machines per cell, 5–6 unique machine shapes | Fleet-capacity calibration (which DC has more compute) |
| 4 | **Per-job metadata** | One row per job: submit/end time, duration, scheduling_class, priority, num_tasks, resource requests | Used to classify batch vs service, fit batch distributions |
| 5 | **Batch job distribution fits** | Per-cell MLE fits (Weibull / log-normal / gamma) for inter-arrival, duration, CPU/memory request, tasks/job | Drives the synthetic batch arrival generator |
| 6 | **Workload generator params** | Aggregated cross-cell summary of #5 + deadline-flexibility methodology from Grange | One-file handoff to the env's `BatchArrivalGenerator` |

**Classification rule for batch vs service** (Borg priority tiers; Tirmazi et al. 2020, "Borg: the Next Generation", §2):
- `priority ≤ 99` (free) **OR** `priority ∈ [110, 115]` (beb) → **batch** — the two **SLO-free** tiers, genuinely delay-tolerant. (Strict beb alone is ~0% of these cells' CPU, so the deferrable class is the union of both no-SLO tiers; the **free tier holds the mass**.)
- otherwise → **service** (must-serve-now). In particular the **production tier** (priority 120–359) "require[s] high availability" and Borg evicts lower-tier jobs to protect it — so it is *not* deferrable.

Classification is by **priority**, not `scheduling_class`: the latter is latency-sensitivity (0–3), orthogonal to tier — `scheduling_class ≤ 1` is dominated by latency-insensitive *production* (priority 200) jobs.

The batch class is the **deferrable** subset — what the agent's drain decisions act on in batch mode. Per-cell deferrable share, by CPU-time: **A 27%, B 61%, C 45%, D 54%**.

---

### Slide 4 — Where the data lands in the repo

After running the notebook + helper scripts, the local `data/` tree looks like:

```
data/
├── cells/
│   ├── cell_a.csv             ─┐  per-cell aggregate CPU timeseries
│   ├── cell_b.csv              │  (timestep, cpu_demand_norm)
│   ├── cell_c.csv              │  ~8,917 rows each, ~30 KB
│   └── cell_d.csv             ─┘
│
├── machines/
│   ├── machines_a.csv         ─┐  per-machine fleet
│   ├── machines_b.csv          │  (machine_id, cpu_capacity, memory_capacity, cell)
│   ├── machines_c.csv          │  ~10K rows per cell
│   ├── machines_d.csv          │
│   └── machines_all.csv       ─┘  union across cells (~40K rows)
│
├── jobs/
│   ├── jobs_{a..d}.csv               (gitignored — ~1M rows each, too big)
│   ├── jobs_{a..d}_truncated.csv     (1,000-row preview, committed for reference)
│   ├── batch_raw_{a..d}.csv          200,000 batch jobs per cell, raw
│   └── batch_distributions_{a..d}.json   Refined distribution fits (input to env)
│
├── power_model_params.json          {idle_power: 0.479, slope: 0.444, peak_power: 0.923, R²: 0.43}
├── power_model_scatter.csv          2,981 (cpu_util, power_util) pairs used to fit it
├── workload_generator_params.json   Cross-cell aggregated generator config
├── extraction_summary.png           6-panel validation plot (committed)
└── metadata.json                    {total_timesteps: 8917, cells: [a,b,c,d], ...}
```

**Two pieces of the directory are NOT from the cluster data** and are introduced later in the deck:
- `data/solar/` — NREL NSRDB irradiance per DC location
- `data/prices/` — real ISO electricity prices per DC's grid region
- `data/net_demand/` — EIA-930 net demand per US balancing authority

Those slot in alongside the ClusterData-derived files but come from different sources (will cover in their own sections).

---

### Slide 5 — Extraction pipeline (visualized)

```
                  google-cluster-data (BigQuery project, public)
                                  │
        ┌─────────────────────────┼────────────────────────────┐
        │                         │                            │
clusterdata_2019_{a..d}     powerdata_2019.cell*        (instance_events,
   instance_usage              measured_power_util       collection_events,
   machine_events             bad_measurement_data       instance_events)
   collection_events
   instance_events
        │                         │                            │
        └─────────────────────────┼────────────────────────────┘
                                  │
                                  ▼
            ┌────────────────────────────────────────────┐
            │  extract_clusterdata2019_full.ipynb        │   ← run in Colab
            │  (20 cells, 6 derived datasets)            │
            └────────────────────────────────────────────┘
                                  │
                                  │  zip → download → unzip into data/
                                  ▼
        ┌──────────────┬──────────────┬──────────────┬──────────────────┐
        │              │              │              │                  │
   cells/cell_*.csv  machines/*  jobs/jobs_*.csv  power_model_*  workload_generator_params.json
        │                              │                                │
        │                              │                                │
        │              scripts/colab_extract_batch_jobs.py  (separate Colab cell)
        │                              │
        │                              ▼
        │                    data/jobs/batch_raw_*.csv  (200K batch jobs/cell, raw)
        │                              │
        │                              ▼
        │                    scripts/refit_distributions.py  (local, MLE refit)
        │                              │
        │                              ▼
        │                    data/jobs/batch_distributions_*.json  (used by env)
        │
        ▼
   env/data_loader.py  (loads cells/* + jobs/batch_distributions_* + others into the env)
```

The notebook is "run once per major dataset change" — Colab/BigQuery work; output is downloaded as a zip and committed (excluding the huge full `jobs_*.csv` files).

---

### Slide 6 — What we DON'T use (and why)

Even though the trace is rich, we deliberately leave a lot on the table:

| Trace feature | Why we skip it |
|---|---|
| Per-PM (per-machine) job placement decisions | We work at cell-aggregate granularity — see "cell-as-DC modeling exercise" section. Per-PM analysis is what CFWS does at intra-DC; we do inter-DC. |
| Job parent-child dependencies (new in 2019 trace) | At aggregate level, individual job dependencies are abstracted away. We're routing aggregate CPU demand, not individual jobs. |
| Alloc sets (resource reservations) | Same reason — relevant for VM-placement decisions, not aggregate routing. |
| Vertical autoscaling (Autopilot) info | Out of scope; we don't model job resource adjustment. |
| Per-PDU power measurements (we use cell-aggregate only) | The cell-level aggregation is sufficient for fitting CPU→power; per-PDU adds resolution we don't act on. |
| Service jobs' deadlines | Service is treated as must-serve-now (no deferral). The deferral lever is batch only. |
| 4 of the 8 cells (e, f, g, h) | We use cells a–d to model 4 DCs. Cells e–h are held out and would be an obvious "does this generalize?" follow-up experiment. |

> **Speaker note**: The unused cells e–h are sitting there ready if a reviewer asks "does the result generalize beyond your chosen cells?" That's a one-day re-eval, not new training.

---

### Slide 7 — Two assumptions we make about the data (state these clearly)

These are real modeling choices that go beyond what Tirmazi documents:

**Assumption 1: Each cell represents a geographically distributed hyperscale DC.**
- Tirmazi describes 8 cells but does not claim they're in different geographic locations
- We treat 4 cells as 4 geo-distributed DCs (US scenario: OR/IA/GA/SC; Global: OR/IA/NL/SG)
- Justified by Tirmazi's documented "considerable inter-cell workload variation" (§3) — the cells *behave* like independent workloads even if their physical locations aren't disclosed

**Assumption 2: We scale cell aggregate up to hyperscale-DC magnitude.**
- A real Borg cell of ~12K machines × ~300W/server ≈ **3–5 MW** of actual power
- We use `rated_power_mw = 100` per DC — about 20–30× the cell's physical scale
- Why: at 5 MW, a single cell is invisible to the regional grid (CAISO peaks at ~26 GW). At 100 MW × 4 DCs = 400 MW aggregate, the simulated fleet matches a real hyperscale operator's regional footprint (Google Council Bluffs, Microsoft Quincy)
- The cell's normalized utilization curve gives us the **shape**; `rated_power_mw` sets the **magnitude** to where grid-stress matters
- Easy to sweep: `rated_power_mw ∈ {25, 50, 100, 200, 500}` would show how PPO advantage scales with DC size

Both are stated explicitly in §2.1 and §3.2 of the thesis overview document, not buried.

---

## Notes for the deck

- **Visuals to grab**: `data/extraction_summary.png` is a ready-to-use 6-panel validation figure (cell utilization, power-model scatter + fit, machine heterogeneity, job-duration histogram, workload mix bar chart, inter-arrival distribution). It's already in the repo; can drop straight into Slide 3 or Slide 5.
- **Numbers to verify before showing**: anything attributed to Tirmazi (cell counts, machine counts, table 1 stats) — the PDF is at `references/borg the next generation.pdf`.
- **Things to defer to later sections**: the cell-as-DC and 100 MW assumptions (Slide 7) are *introduced* here but get their full justification in the "Env design" section of the deck.
