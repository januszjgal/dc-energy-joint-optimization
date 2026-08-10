# Ramp v6 pure-RL integration checkpoint

The pure PPO/SAC harness is integrated with the actual hourly ramp-core
environment and its deterministic six-market, 1 GW fixture. The long campaign
was not launched.

## Frozen contract decisions

- The environment contract uses semantic action ID
  `ramp-v6-constraint-decoded-preferences-2n-plus-1-v1`.
- Six sites expose the real `2N+1 = 13` action with exact `[-6, 6]` bounds:
  service-allocation preferences, optional total batch execution, and batch
  destination preferences.
- The frozen constraint-only capped-simplex, EDF origin drain, residual-capacity
  allocation, and exact transport implementation remains the only decoder.
- The fixture episode has 3 real warm-history hours, 6 arrival decisions, and 3
  no-arrival tail decisions. All 9 steps are scored. Terminal evidence repeats
  the 3 tail hours in ramp, cost, and safety arrays without evaluator double
  counting.
- The trainer protocol is hourly and binds the existing ramp environment
  protocol and its single panel schema. SAC retains the stronger frozen
  `n_steps=36`, covering the real 3-step delayed 3h ramp credit.
- The environment accepts projected-dual multiplier updates and adds the signed
  cost-budget Lagrangian term to the ramp reward. Validation/test optimizer and
  multiplier guards remain sealed.
- Training/replay attribution is random initialization plus SAC safe-random
  warmup only. Teacher, BC, demonstrations, expert replay, analytic economic
  actions, MPC, optimizer actions, oracle shaping, and curriculum trajectories
  remain explicitly false.

Protocol bundle SHA-256:
`03271cd1a3c8cf66c5c8be1e324bf1e07e096ffb650fccb1ac53f87d3672a8fd`.

## Miniature checkpoint evidence

| Metric | PPO | SAC |
|---|---:|---:|
| Interactions | 288 | 198 |
| Updates | 18 | 95 |
| Resume | no | yes, from 108 interactions |
| Mean incremental ramp impact | 0.0000660092268874879 | 0.0000752449492121297 |
| DA energy cost ratio | 1.0009368383919233 | 1.00095399483131 |
| Adjusted 1h ramp p95 / max | 0.0347992766785827 / 0.0379923108733994 | 0.0348556263612196 / 0.0380078949914589 |
| Adjusted 3h ramp p95 / max | 0.0294699583324415 / 0.0320373755263946 | 0.0294420803991097 / 0.0320104483506029 |
| Service unserved | 0 | 0 |
| Batch unfinished / expired | 0 / 0 | 0 / 0 |
| Terminal work / certificates | 0 / 0 | 0 / 0 |
| Emergency feasibility rate | 0 | 0 |
| Semantic adjustment L2 | 0 | 0 |

Both miniature random policies satisfy exact workload safety and the 2% cost
budget. Both fail the preregistered mean and every-market ramp-improvement
gates, which is an allowed publishable smoke result rather than a promotion.

`integration_evidence_manifest.json` re-hashes the saved PPO/SAC models,
normalization state, SAC replay, training manifests, factory source, integrated
source bundle, and protocol bundle. It passes with no errors. No expert or
analytic action appears in training or replay. Resume identity includes the
integrated source-bundle hash, so pre-change interactions cannot be mixed into
a post-change checkpoint.

## Validation

- Ramp core plus integration: 29 tests passed, including independent
  status-quo shadow accounting and bundled-tail evaluation/training regressions.
- Integrated miniature PPO/SAC smoke, deterministic evaluation, attribution,
  incompatible-resume rejection, valid SAC replay resume, and evidence checks:
  passed.
- Frozen v5 off-policy smoke: passed.
- PPO v3 compatibility smoke: passed.
- v4 hard-safety smoke: passed.
- Demand-charge regression: all checks passed.

## Remaining energy-model v3 handoff

The sole long-campaign blocker is
`output/energy_model_v3/ramp_v6/factory_manifest.json`. It must use schema
`energy-model-v3-ramp-v6-factory-v1`, identify the forecast model, and map
train/validation/test window IDs to artifact roots, months, days, forecast
vintages, file SHA-256 hashes, and one manifest-level frozen-statistics
SHA-256 shared by every window.

Each artifact root must contain:

1. `canonical_panel.csv`: six complete hourly markets satisfying
   `env/protocols/v6_ramp_panel.schema.json`, including causal h1/h2/h3
   gross/net forecasts, issue time, vintage, quality, physical tuple, market
   scale, and day-ahead LMP.
2. `fixture.json`: six aligned site definitions, hourly service/batch workload,
   deadlines, three warm-power rows, and train-only frozen ramp statistics.

`env.ramp_v6.factory:make_energy_model_v3_env` selects only the requested split
and verifies both file hashes. Train environments resample chronological
windows from a seeded split-local permutation on reset; validation/test remain
fixed. Declared months, panel timestamps, and frozen-stat fit months are checked
against the consolidated split before the same real environment used by the
fixture integration is loaded. Until the manifest exists, the factory fails
with `MissingEnergyModelV3PanelError`; there is no remaining
trainer/environment interface blocker.
