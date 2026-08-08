# Safe Joint Energy Optimization for Geo-Distributed Data Centers

The preregistered, additive pure-PPO/SAC harness for the forthcoming ramp-aware
v6 environment is documented in [`docs/ramp_rl_v6.md`](docs/ramp_rl_v6.md).
It remains fixture-only until the v6 ramp environment and energy-model v3 are
integrated; frozen v2-v5 behavior and artifacts are not imported or modified.

This repository is the executable companion to a thesis on joint spatial and
temporal workload scheduling across four proxy data centers.

The final controller is:

```text
trained TD3+BC network chooses routing/timing preferences
  -> constraint-only decoder enforces feasibility
  -> emergency fallback handles unexpected decoder failures
```

The analytic teacher is used only for offline demonstrations. It is disabled
during reward-driven TD3 updates and at inference.

## Final verified result

The corrected frozen-v3 study uses five training seeds per region. Cells a-d
are the development/frozen-confirmation scope; cells e-h are descriptive
transfer only.

| Region | BC-only a-d mean | Post-RL a-d mean | Post-RL minimum | e-h descriptive mean | Emergency fallback |
|---|---:|---:|---:|---:|---:|
| US | 6.289% | **6.290%** | 6.112% | 5.685% | 0.00% |
| Global | 14.017% | **14.000%** | 13.822% | 12.890% | 0.00% |

Every post-RL seed has:

- exact service and batch completion;
- zero expiry, terminal pool, and terminal backlog;
- zero infeasibility certificates;
- zero emergency fallback;
- changed actor and critic hashes; and
- 2,048 genuine TD3 reward updates.

The result clears the requested >5% US and >10% Global thresholds on every
a-d seed. The e-h descriptive means also exceed those thresholds, but e-h is
not fresh confirmatory data.

### Attribution

The result is primarily **teacher imitation preserved by reward training**:

- paired post-RL minus BC: **+0.0013 percentage points US**;
- paired post-RL minus BC: **-0.0172 percentage points Global**.

The short TD3+BC phase is genuine RL, but it does not explain most of the
absolute savings. The frozen greedy demonstration teacher scores 6.089% US and
13.886% Global on a-d. A separate exact-native analytic benchmark scores
7.216% and 15.715%; the post-RL networks capture 87.17% and 89.09% of that
stronger benchmark.

Normal decoder adjustment is frequent (98.22% US, 95.12% Global on a-d).
That is expected constraint enforcement, not emergency intervention, but it
means the executed solution is properly attributed to the learned preference
network **plus** the deterministic feasibility decoder.

## Demand-charge finding

Demand charge is excluded from the primary training objective and reported as
a secondary sensitivity at an illustrative $15/kW-cycle.

| Scope | Demand-charge change | Primary + demand-charge sensitivity |
|---|---:|---:|
| US a-d | +18.536% (+$870,104) | **-4.058%** |
| Global a-d | +18.536% (+$870,104) | **+0.475%** |
| US e-h descriptive | +16.152% (+$747,240) | **-3.630%** |
| Global e-h descriptive | +16.152% (+$747,240) | **+0.566%** |

The energy/grid-stress controller concentrates load spatially and increases
the sum of site peaks. Optimizing primary energy cost does **not** guarantee
demand-charge savings. This is not a production tariff forecast: the model
uses five-minute peaks, while real tariffs commonly use 15-minute windows,
ratchets, and utility-specific rules.

## Data contract

### Workload

- Google ClusterData 2019 cells a-h.
- 96,580 source machines (metadata only).
- Measured five-minute aggregate CPU usage.
- Measured aggregate usage and classified no-SLO batch usage for every cell:
  - batch/no-SLO: matched priority <=115;
  - service/residual: aggregate minus classified batch, including unmatched
    priority metadata.
- Exact `service + batch = aggregate` conservation.
- Experimental deadline horizons, not Borg SLOs.

See [`data/README.md`](data/README.md).

### Energy

- CAISO Today's Outlook native five-minute net demand and solar.
- CAISO OASIS NP15 hourly day-ahead LMP expanded stepwise.
- May 1-June 1, 2025: 8,928 five-minute intervals.
- Signed negative net demand and negative prices preserved.
- One CAISO tuple shifted together by IANA local wall time across US and
  Global slots.

The slots are a controlled archetype, not eight independent electricity
markets. See
[`data/energy_model_v2/README.md`](data/energy_model_v2/README.md).

### Proxy facilities

Each site is an equal 100 MW proxy with normalized compute capacity 1.0.
Per-cell affine PowerData2019 fits map CPU utilization to power:

```text
power_utilization = idle + slope * cpu_utilization
grid_power_mw = 100 MW * power_utilization
```

Routing is unrestricted. Latency, residency, transfer bandwidth, transfer
energy, and movement cost are out of scope.

## Primary objective

For five-minute interval duration `delta_h = 1/12`:

```text
energy_cost_i
  = price_i_usd_per_kwh * grid_power_i_mw * 1000 * delta_h

grid_stress_i
  = 0.015 * grid_power_i_mw^2 * max(signed_net_demand_i, 0)
```

The primary objective sums energy cost and grid stress. Carbon intensity is
not an input and carbon is not optimized.

## Final v5 policy

The network action has 13 dimensions:

1. four service-routing logits;
2. one optional batch-total scalar;
3. four batch-origin logits; and
4. four batch-destination logits.

The decoder applies:

- capped-simplex service allocation;
- cumulative EDF deadline lower bounds;
- bounded batch-origin drain;
- residual-capacity destination allocation; and
- exact origin-to-destination transport.

The v4 shield remains only as a separately measured emergency fallback.

## Authoritative evidence

| Artifact | Purpose |
|---|---|
| [`env/protocols/v5_offpolicy_td3bc.yaml`](env/protocols/v5_offpolicy_td3bc.yaml) | Frozen BC and post-RL protocol |
| [`output/offpolicy_v5_continuous/canonical_v5_v3/evidence_manifest.json`](output/offpolicy_v5_continuous/canonical_v5_v3/evidence_manifest.json) | Cryptographic source/model/record provenance |
| [`output/offpolicy_v5_continuous/canonical_v5_v3/canonical_results.json`](output/offpolicy_v5_continuous/canonical_v5_v3/canonical_results.json) | Canonical metrics, attribution, demonstration teacher, exact-native benchmark, and demand-charge sensitivity |
| [`output/offpolicy_v5_continuous/canonical_v5_v3/results_report.md`](output/offpolicy_v5_continuous/canonical_v5_v3/results_report.md) | Human-readable final result |
| [`output/offpolicy_v5_continuous/final_td3bc_manifest_us_v3.json`](output/offpolicy_v5_continuous/final_td3bc_manifest_us_v3.json) | Per-seed US hashes and metrics |
| [`output/offpolicy_v5_continuous/final_td3bc_manifest_global_v3.json`](output/offpolicy_v5_continuous/final_td3bc_manifest_global_v3.json) | Per-seed Global hashes and metrics |
| [`models/offpolicy_v5_continuous/frozen_confirmation/`](models/offpolicy_v5_continuous/frozen_confirmation/) | Twenty persisted BC/post-RL model and record pairs |

Published model-training source commit:
`81d50713b85e5f96809b37c16855289d13b1ad4d`

Protocol SHA-256:
`af656597d178f64ccf1f37ce635e3d1c9dc9167f13efe104deee86b44c4e1349`

## Setup

```powershell
python -m pip install -r requirements.txt
npm install
```

Tested final RL stack:

- Stable-Baselines3 2.9.0
- PyTorch 2.11 CPU
- Gymnasium-compatible environment

## Validation

```powershell
python scripts\preflight_energy_model_v2.py
python scripts\smoke_test_demand_charge.py
python scripts\smoke_test_ppo_v3.py
python scripts\smoke_test_safety_v4.py
python scripts\smoke_test_offpolicy_v5.py
```

## Reproduce the final campaign

The published models record training-source commit
`81d50713b85e5f96809b37c16855289d13b1ad4d`. Use a separate worktree to
regenerate that exact source identity without overwriting the final branch's
committed artifacts:

```powershell
git worktree add ..\dc-energy-v5-training 81d50713b85e5f96809b37c16855289d13b1ad4d
Push-Location ..\dc-energy-v5-training

python scripts\run_offpolicy_campaign_v5.py --campaign td3bc_bconly_frozen_v3 --regions us global --seeds 301 302 303 304 305 --workers 4

python scripts\run_offpolicy_campaign_v5.py --campaign td3bc_postrl_frozen_v3 --regions us global --seeds 301 302 303 304 305 --workers 4

python scripts\build_offpolicy_evidence_v5.py --bc-campaign td3bc_bconly_frozen_v3 --postrl-campaign td3bc_postrl_frozen_v3 --seeds 301 302 303 304 305 --workers 4 --suffix v3

Pop-Location
```

On the checked-out final package commit, rebuild the canonical report from the committed
verified package:

```powershell
python scripts\build_v5_results.py
```

The final results builder and completed paper/artifact package were added after
the frozen training commit; the thesis builder infrastructure itself predates
that commit. This multi-revision provenance is intentional and documented in
the final paper. Running training from a later commit creates a new
`source_commit` and a new evidence package.

## Build the thesis

The final paper is authored at the repository root and regenerated from
scratch:

```powershell
python scripts\build_final_thesis.py
```

Outputs:

- [`thesis_paper.md`](thesis_paper.md)
- [`thesis_paper.docx`](thesis_paper.docx)

The DOCX builder uses the pinned local `docx` package from `package.json`.

## Repository map

| Path | Role |
|---|---|
| `data/` | Active Borg, PowerData2019, and CAISO inputs |
| `env/` | Simulator, accounting, safety layer, and v5 decoder |
| `env/scenarios/` | US/Global a-d and e-h scenario wiring |
| `env/protocols/` | Frozen v2-v5 protocol files |
| `scripts/run_offpolicy_campaign_v5.py` | Teacher collection, BC, TD3+BC, evaluation |
| `scripts/build_offpolicy_evidence_v5.py` | Cryptographic evidence validation |
| `scripts/build_v5_results.py` | Final metrics, report, and figures |
| `scripts/build_final_thesis.py` | Final DOCX entry point |
| `models/offpolicy_v5_continuous/` | Final network artifacts |
| `output/offpolicy_v5_continuous/` | Final evidence and results |
| `archive/` | Historical/invalid/superseded material only |

Nothing under `archive/` is required to run or explain the final solution.

## Version lineage

- **v1:** mixed/synthetic market and accounting issues; invalidated.
- **v2:** frozen 80-model PPO study; only Global spatial robust; joint unsafe.
- **v3:** observability/reward recovery; economics improved but only 1/10 safe.
- **v4:** exact hard safety; Global +3.262% but 18.786% projector intervention.
- **v5:** structured network preferences, constraint decoder, offline causal
  teacher, BC, and verified post-BC TD3+BC.

The detailed history and limitations are in `thesis_paper.md`.
