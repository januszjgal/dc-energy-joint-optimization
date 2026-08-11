# Anticipatory Grid-Ramp Smoothing for Geo-Distributed Data Centers

This repository contains the final six-market V6/V4R thesis implementation,
evidence package, and LaTeX publication source. Historical V1-V5 studies,
superseded artifacts, and the retired Markdown/DOCX pipeline are retained under
[`archive/`](archive/).

The active thesis source is [`latex/thesis.tex`](latex/thesis.tex).

## Evidence authority

The authoritative publication wrapper is
[`output/ramp_rl_v6/recovered_v4r_posthoc_metrics_v1/canonical_posthoc_metrics.json`](output/ramp_rl_v6/recovered_v4r_posthoc_metrics_v1/canonical_posthoc_metrics.json).
Its canonical-JSON SHA-256 is
`42bbcf4c4cd2cca58cfce0e07319d5280145e0fac948a7fe66d2a6582e379d4b`.

The wrapper references rather than replaces the immutable single-open sealed
evidence at
[`output/ramp_rl_v6/recovered_v4r_resealed_v2/canonical_evidence.json`](output/ramp_rl_v6/recovered_v4r_resealed_v2/canonical_evidence.json),
whose canonical-JSON SHA-256 is
`f642bd5868abdd9f7cda2a6fffb228250f3570fd0c6d440085da68c976892d9b`.

The additive wrapper corrects only:

- absolute per-market/per-timestep physical 1 h and 3 h ramp summaries;
- explicit native-grid versus operational status-quo labels and deltas; and
- measured semantic decoder-adjustment telemetry.

The equivalence-bound replay occurred after unblinding and is not a second
sealed generalization test. The primary objective, policy, members, equal
weights, normalizers, data, forecasts, constraints, and original chronology
remain unchanged.

Checkpoint reconstruction is independently bound by
[`output/ramp_rl_v6/recovered_v4r_posthoc_metrics_v1/metric_replay_recovery_manifest.json`](output/ramp_rl_v6/recovered_v4r_posthoc_metrics_v1/metric_replay_recovery_manifest.json).
Its canonical-JSON SHA-256 is
`26e9c77a1d932ff7edb66fe8b7d55b6f769a2e2029a49c95411b62465b14f064`.

## Final V4R result

V4R combines five independently trained reward-only PPO members using a fixed
equal mean of their deterministic environment-space actions. Every member is
invoked at every decision with weight 0.2. There is no teacher, behavior
cloning, learned combiner, member selection, MPC, or optimizer action.

| Split | Episodes | Policy native-relative ramp impact | DA cost ratio | Historical gates |
|---|---:|---:|---:|---|
| February 2026 validation | 28 | -1.4086907198e-05 | 0.9858280673 | pass |
| March-April 2026 sealed test | 60 | -1.4258710514e-05 | 0.9775584402 | pass |

All six policy impacts are negative relative to native grid ramps. Direct
comparison with operational status quo is favorable in five of six markets;
MISO is the exception. Secondary modeled day-ahead cost is $21,098,314 versus
$21,582,662 for status quo, a $484,349 (2.244%) difference. Cost is not the
primary objective, and retail tariffs and demand charges are out of scope.

## Final study inputs

### Workload

- Google ClusterData2019 cells a-f for the primary six-site map.
- Cells c-h for an explicitly overlapping, non-out-of-fold robustness map.
- Measured five-minute CPU usage split into immediate service/residual and
  no-SLO batch work using matched collection priority <=115.
- A 744-hour measured profile repeated across the eight-month market calendar.
- Experimental hourly deadlines derived from completed no-SLO durations.

See [`data/README.md`](data/README.md) and
[`latex/thesis.tex`](latex/thesis.tex).

### Power

- Per-cell affine CPU-to-power fits derived from Google PowerData2019.
- Six equal 100 MW proxy facilities in the primary 600 MW study.
- Workload, compute capacity, rated power, and warm power scaled together in
  capacity sensitivities.

### Electricity

The common hourly UTC panel covers September 2025 through April 2026:

| Site | Price location | Physical grid geography | Borg cell |
|---|---|---|---|
| Northern California | CAISO NP15 | CAISO balancing authority | a |
| North Texas | ERCOT LZ_NORTH | ERCOT system | b |
| New York City | NYISO Zone J | Zone J load with NYCA renewable context | c |
| Minnesota | MISO MINN.HUB | MISO balancing authority | d |
| SPP North | SPPNORTH_HUB | SPP balancing authority | e |
| Boston/NEMA | ISO-NE location 4008 | ISO New England balancing authority | f |

Operator day-ahead prices remain separate from physical demand, wind, solar,
and net load. Explicit same-balancing-authority EIA bulk fallbacks are used
where a complete operator physical history is unavailable. No primary market
series is interpolated or silently substituted. PJM DOM/Northern Virginia is
credential-blocked and excluded.

See [`data/energy_model_v3/README.md`](data/energy_model_v3/README.md).

## Calendar and leakage boundary

- September 2025 through January 2026: training.
- February 2026: validation.
- March-April 2026: sealed test.

Market scales, ramp thresholds, normalization, forecast fitting, and reward
calibration use training data only. Forecasts are causal reconstructed 1 h,
2 h, and 3 h trajectories; realized future values are forbidden as features.

## Rebuild and verify

Restricted raw market files and recovered model containers may remain local.
Committed manifests, hashes, sealed evidence, and corrected metrics define the
publication boundary.

```powershell
python scripts\bind_v4r_metric_replay_recovery.py verify
python scripts\recompute_v4r_posthoc_metrics.py verify
python scripts\build_ramp_rl_thesis_results.py --corrected
python scripts\build_ramp_thesis_figures.py
python scripts\build_energy_thesis_figures.py

Push-Location latex
xelatex -interaction=nonstopmode -halt-on-error thesis.tex
xelatex -interaction=nonstopmode -halt-on-error thesis.tex
Pop-Location
```

To reproduce post-hoc metrics rather than verify the committed package, stage
the exact hash-bound model and normalizer binaries and run
`scripts\recompute_v4r_posthoc_metrics.py recompute` with an explicit UTC
timestamp. The command performs no tuning, selection, or policy update.

To validate the locally staged market panel:

```powershell
python scripts\build_energy_model_v3.py preflight
python scripts\build_energy_v3_ramp_factory.py
```

## Setup

```powershell
python -m pip install -r requirements.txt
```

The final RL stack uses Stable-Baselines3 2.9.0, PyTorch, and a
Gymnasium-compatible environment. XeLaTeX builds the active thesis.

## Repository map

| Path | Role |
|---|---|
| `data/cells/` | Active ClusterData2019 aggregate and tier curves |
| `data/jobs/batch_distributions_*.json` | Active per-cell deadline statistics |
| `data/energy_model_v3/` | Six-market source contracts, manifests, forecasts, and panel configuration |
| `energy_model_v3/` | Acquisition, canonicalization, forecasts, derivations, and factory |
| `env/ramp_v6/` | Hourly six-market ramp environment and constraint decoder |
| `env/protocols/v6_*` | Frozen V6 environment and campaign contracts |
| `ramp_rl/` | PPO campaign, recovery, evaluation, and evidence harness |
| `models/ramp_rl_v6/` | Final and recovery model containers; large binaries may remain local |
| `output/ramp_rl_v6/` | Canonical V4R evidence and thesis result packages |
| `scripts/` | Active V6/V4R build, verification, recovery, and publication commands |
| `latex/thesis.tex` | Active thesis source |
| `archive/` | Historical, invalidated, and superseded material |

Nothing under `archive/` is required to run or verify the final V6/V4R
solution. See [`archive/README.md`](archive/README.md) for provenance and
original-path mappings.
