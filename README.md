# Anticipatory Grid-Ramp Smoothing for Geo-Distributed Data Centers

The final thesis is [`thesis_paper.md`](thesis_paper.md) with the reproducible
Word deliverable [`thesis_paper.docx`](thesis_paper.docx).

## Evidence authority

The authoritative publication wrapper is
[`output/ramp_rl_v6/recovered_v4r_posthoc_metrics_v1/canonical_posthoc_metrics.json`](output/ramp_rl_v6/recovered_v4r_posthoc_metrics_v1/canonical_posthoc_metrics.json).
Its canonical-JSON SHA-256 is
`42bbcf4c4cd2cca58cfce0e07319d5280145e0fac948a7fe66d2a6582e379d4b`.

That wrapper **references rather than replaces** the immutable single-open
sealed evidence at
[`output/ramp_rl_v6/recovered_v4r_resealed_v2/canonical_evidence.json`](output/ramp_rl_v6/recovered_v4r_resealed_v2/canonical_evidence.json),
whose canonical-JSON SHA-256 remains
`f642bd5868abdd9f7cda2a6fffb228250f3570fd0c6d440085da68c976892d9b`.
The additive wrapper corrects only:

- absolute per-market/per-timestep physical 1 h and 3 h ramp summaries;
- explicit native-grid versus operational status-quo labels and deltas; and
- real semantic decoder-adjustment telemetry.

The replay occurred after unblinding and is not a second sealed generalization
test. The primary squared incremental objective, policy, members, equal weights,
normalizers, data, forecasts, constraints, and original chronology are
unchanged.

Checkpoint reconstruction is independently bound by
[`output/ramp_rl_v6/recovered_v4r_posthoc_metrics_v1/metric_replay_recovery_manifest.json`](output/ramp_rl_v6/recovered_v4r_posthoc_metrics_v1/metric_replay_recovery_manifest.json).
Its canonical-JSON SHA-256 is
`26e9c77a1d932ff7edb66fe8b7d55b6f769a2e2029a49c95411b62465b14f064`.
It references the full raw verifier output, stable recovery commit `d1d4828`,
all initial/final tensor and training-provenance checks, the new ZIP hashes,
and byte-identical normalizer hashes. Recovered binaries remain under the
ignored `models/ramp_rl_v6/recovery_v3_metric_replay/` root.

## Final V4R result

V4R combines five reward-only V3 PPO members with a fixed equal environment-
action mean. Every decision invokes all five deterministic members at weights
of 0.2. There is no member selection, learned weighting, trainable combiner,
teacher, behavior cloning, MPC, analytic policy, or optimizer action.

| Split | Episodes | Policy native-relative ramp impact | DA cost ratio | Historical gates |
|---|---:|---:|---:|---|
| February validation | 28 | -1.4086907198e-05 | 0.9858280673 | pass |
| March-April sealed test | 60 | -1.4258710514e-05 | 0.9775584402 | pass |

All six policy impacts are negative relative to native grid ramps. Direct
comparison with operational status quo is favorable in **five of six** markets;
MISO is the exception. Overall macro impact strongly favors policy. Secondary
modeled day-ahead cost is $21,098,314 versus $21,582,662 for status quo, a
$484,349 (2.244%) difference. Cost is not the primary objective, and demand
charges and retail bills are out of scope.

## Rebuild and verify

Restricted raw market files and recovered model containers may remain local.
Committed manifests, hashes, sealed evidence, and corrected metrics are the
publication boundary.

```powershell
python scripts\bind_v4r_metric_replay_recovery.py verify
python scripts\recompute_v4r_posthoc_metrics.py verify
python scripts\build_ramp_rl_thesis_results.py --corrected
python scripts\build_ramp_thesis_figures.py
python scripts\materialize_ramp_thesis.py --output thesis_paper.md
python scripts\validate_ramp_thesis.py --source thesis_paper.md
python scripts\build_final_thesis.py --source thesis_paper.md --output thesis_paper.docx
```

To reproduce the post-hoc metrics rather than only verify the committed
package, first stage the exact hash-bound frozen model and normalizer binaries,
then run `scripts\recompute_v4r_posthoc_metrics.py recompute` with an explicit
UTC timestamp. The command performs no tuning, selection, or policy update.

In the committed correction run, ignored V4R containers were unavailable, so
the frozen V3 recipe was deterministically re-executed solely to reconstruct
loadable checkpoints. The wrapper records this training computation, the new
outer container hashes, exact policy/critic/normalizer equivalence, unchanged
action chains, and the fact that the replay is not fresh test evidence.

Frozen V2-V5 behavior, models, evidence, and reproduction commands below remain
historical and unchanged.

## Legacy V5 study

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

## Frozen V5 verified result

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
python scripts\build_energy_model_v3.py preflight
python scripts\smoke_test_demand_charge.py
python scripts\smoke_test_ppo_v3.py
python scripts\smoke_test_safety_v4.py
python scripts\smoke_test_offpolicy_v5.py
python -m unittest tests.test_ramp_thesis -v
python -m unittest discover -s tests\ramp_v6 -p "test_*.py" -v
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

## Build the legacy V5 thesis source directly

The DOCX builder remains compatible with the preserved V5-era Markdown and
builder tests. The final ramp-aware publication should use the hash-gated
five-command pipeline at the top of this README.

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
| `env/protocols/` | Frozen v2-v6 environment, panel, and campaign contracts |
| `env/ramp_v6/` | Hourly six-market ramp environment and trainer factories |
| `ramp_rl/` | Pure PPO/SAC campaign, resume, evaluation, and evidence harness |
| `ramp_rl/v4r_thesis.py` | V4R hash/recovery/claim-ledger publication gate |
| `scripts/run_offpolicy_campaign_v5.py` | Teacher collection, BC, TD3+BC, evaluation |
| `scripts/build_offpolicy_evidence_v5.py` | Cryptographic evidence validation |
| `scripts/build_v5_results.py` | Final metrics, report, and figures |
| `scripts/build_final_thesis.py` | Final DOCX entry point |
| `models/offpolicy_v5_continuous/` | Final network artifacts |
| `output/offpolicy_v5_continuous/` | Final evidence and results |
| `output/ramp_rl_v6/recovered_v4r/` | Canonical V4R evidence and thesis tables |
| `archive/` | Historical/invalid/superseded material only |

Nothing under `archive/` is required to run or explain the final solution.

## Version lineage

- **v1:** mixed/synthetic market and accounting issues; invalidated.
- **v2:** frozen 80-model PPO study; only Global spatial robust; joint unsafe.
- **v3:** observability/reward recovery; economics improved but only 1/10 safe.
- **v4:** exact hard safety; Global +3.262% but 18.786% projector intervention.
- **v5:** structured network preferences, constraint decoder, offline causal
  teacher, BC, and verified post-BC TD3+BC.
- **v6:** six-market ramp objective; V1-V3 strict failures, blocked original
  V4, and successful recovered-policy equal-action ensemble V4R.

The detailed history and limitations are in `thesis_paper.md`.
