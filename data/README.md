# Active Four-Market Data Contract

The active experiment uses one raw-workload protocol and one generated factory.
Cells a-d are active; cells e-h remain tier-curve workload holdouts.

## Workload and power

`cells/cell_a_tiers.csv` through `cells/cell_d_tiers.csv` provide the raw measured
five-minute normalized CPU, service, and batch curves. The factory averages twelve
samples per hour and uses the service and batch columns unchanged. It uses fixed
provisional batch deadlines `[2, 1, 2, 3]`; they are protocol parameters, not
observed customer deadlines.

`power_model_params.json` contains the committed affine CPU-to-power coefficients.
Every proxy site has normalized compute capacity 1 and a 500 MW rating:

```text
power_mw = 500 * (idle_power_fraction + dynamic_power_fraction * executed_work)
```

## Electricity calendar

`four_market_v2/calendar.json` is the source contract. It lists 114
October-January training dates, the exact 28 February validation dates, each
referenced path under `four_market_v2/windows/`, and
`four_market_v2/frozen_stats.json`. Daily panels contain four market cases with
gross demand, wind, solar, derived net load, day-ahead LMP, and causal 1-3 hour
gross/net forecasts. The contract has no test split.

Forecasts remain per-date causal forecasts: a fit using later training targets
would leak them into earlier training observations. Forecast age, vintage, and
quality are not policy observations.

## Factory and campaign

Run:

```powershell
python scripts\build_four_market_v2_factory.py
```

This produces one fixture set and
`output/four_market_v2/factory/factory.json`. The factory reads `calendar.json`
and central frozen statistics, and references data panels rather than copying them.
It validates the 114/28 split, panel coverage, hard capacity, and raw workload
contract.

The locked ten-seed campaign is complete. Reproduce it with five workers, then
aggregate after all ten summaries exist:

```powershell
python scripts\run_four_market_v2_campaign.py run --workers 5
python scripts\aggregate_four_market_v2_campaign.py
```

Seeds 4101-4110 request 2,000,000 interactions and complete uninterrupted
(`resumed_from_interactions=0`) at the safe boundary of 2,045,952 per seed.
Learning curves use raw rollout metrics; final paired analysis uses the optimizer
seed (`n=10`), exact two-sided Wilcoxon, matched-pairs
rank-biserial correlation, Hodges-Lehmann shift, 10,000-draw seed bootstrap, and
post-hoc cost/safety reporting.

All ten optimizer seeds beat status quo on the macro ramp metric
(`p=0.001953125`, exact two-sided Wilcoxon; rank-biserial `1.0`). CAISO, SPP,
and ISO-NE improved in every seed, while MISO worsened in every seed. Mean
day-ahead cost was 1.001233 times status quo and all modeled safety/work
violations were zero.
