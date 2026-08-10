# Ramp-aware pure-RL protocol v6

Version 6 is additive. It does not import, rewrite, or reinterpret frozen v2-v5
controllers, protocols, models, or evidence. Its runtime is isolated under
`env/ramp_v6`, its contract under `env/protocols/v6_*`, its deterministic input
under `tests/fixtures/ramp_v6`, and its sanity evidence under
`output/ramp_v6_fixture`.

## Integration boundary

The energy-model v3 handoff must produce the long-form columns in
`env/protocols/v6_ramp_panel.schema.json`, one row per market and contiguous
UTC hour. Each controller row includes gross demand, net load, wind, solar,
market scale, day-ahead LMP, quality, and h1/h2/h3 forecast endpoints with an
issue timestamp and vintage ID. `forecast_issue_time_utc` must not exceed the
controller timestamp. Realized future demand and real-time price are not
policy inputs.

The primary study uses six separately sourced market/BA series at 100 MW/site.
They are fixed evaluated cases, not statistically independent samples. The core
supports arbitrary N and multiple sites attached to one market; site power is
summed exactly once before market ramp scoring. Physical sensitivities scale
rated MW, compute capacity, arrivals, and warm power together:

- `1.0`: 100 MW/site fixture and primary per-site convention;
- `1000 / (6 * 100)`: 1 GW total across six equal primary sites;
- `10.0`: non-primary 1 GW/site stress.

Frozen gross-demand Q95 scales, level normalizers, native-ramp Q90 thresholds,
and forecast fits come only from complete chronological training months.
Validation and test months are later complete calendar months.

## State, action, and feasibility

The state exposes current plus three trailing normalized gross/net levels,
closed 1h/3h native ramps, causal h1/h2/h3 forecast endpoints and maximum
upward forecast ramps, previous modeled power, service/batch arrivals, EDF
queue/deadline state, site power/capacity parameters, DA price, vintage age,
quality, and UTC time features.

For N sites the action is `2N+1`: N service preferences, one optional total
batch-execution preference, and N batch-destination preferences. The decoder
contains no economic objective. It applies capped-simplex service conservation,
deterministic EDF origin drainage, capped-simplex residual destination
capacity, and exact transport. There is no teacher, behavior cloning,
demonstration set, economic base policy, MPC, or policy/training optimizer.

Batch work is conserved exactly. A three-hour no-arrival tail is part of every
episode, all tail power and ramp windows are scored, and the queue must be empty
at termination. Frozen protocol design envelopes for service and batch arrivals
reserve causal carried-work capacity and make work mandatory before it can be
dumped into the final step. Integration preflight must validate these envelopes;
runtime violations fail closed rather than consulting future workload traces.

## Ramp and cost accounting

For market net load `N`, aggregate data-center power `P`, training gross-load
Q95 `S`, and `h` in `{1, 3}`:

```text
b_h = (N_t - N_t-h) / (S * h)
a_h = ((N_t + P_t) - (N_t-h + P_t-h)) / (S * h)
I_h = a_h^2 - b_h^2
```

Weights are 0.40/0.60. A smaller term measures the incremental squared
absolute-ramp residual above the frozen training Q90. The scalar RL interface
is the negative weighted dimensionless stress score. It is not labeled or
interpreted as dollars.

Actual hourly DA energy cost is separately
`LMP_USD_per_MWh * P_MW * 1h`. The primary incremental objective remains the
per-market 0.40/0.60 weighted squared impact above. Physical p95 and maximum
telemetry instead pools `abs(a_h)` over every market and scored timestep with
equal market-timestep weight. Signed cross-market means are compatibility
telemetry only and are never inputs to physical p95 or maximum metrics.

The preregistered per-market gate asks whether each policy impact is negative
relative to the native grid ramp. That gate is not a claim that policy beats
the operational status quo in each market. Evaluation therefore reports a
separate policy-versus-status-quo block with overall and per-market deltas,
better-market count/share, and the diagnostic
`every_market_outperforms_status_quo`.

The normal decoder adjustment is the Euclidean distance between requested and
executed vectors in decoded `2N+1` work-allocation coordinates: N service
amounts, one total batch amount, and N batch-destination amounts. Its units are
compute-work units per hourly decision. Raw preference logits are never
subtracted from allocations. Normal projection telemetry is reported
separately from the emergency fallback path.

## Deterministic verification

```powershell
python -m unittest discover -s tests\ramp_v6 -t . -v
python scripts\run_ramp_v6_fixture.py
python scripts\smoke_test_ramp_rl_v6.py
python scripts\run_ramp_v6_fixture.py --scale-multiplier 10 --output-root output\ramp_v6_fixture_1gw_per_site
```

The committed fixture evidence is an implementation check only. Miniature
random-initialized PPO/SAC metrics are smoke evidence, not a trained-policy
claim. The later V1-V4R campaign lineage, immutable single-open result, and
additive post-hoc metric correction are described in `docs/ramp_rl_v6.md` and
`docs/ramp_v6_result_handoff.md`; fixture outputs must not be substituted for
that evidence.
