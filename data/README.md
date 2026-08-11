# Active Four-Market Data Contract

The experiment combines Google ClusterData2019 workload, Google PowerData2019
power fits, and four hourly electricity-market panels. Cells a-d are active.
Cells e-h remain committed but untouched as a future workload holdout.

## Workload

`cells/cell_a_tiers.csv` through `cells/cell_d_tiers.csv` contain measured
five-minute aggregate, service/residual, and no-SLO batch CPU curves. Twelve
complete samples are averaged per hour. The service and batch columns reproduce
the aggregate curve before any experimental admission rule.

`jobs/batch_distributions_a.json` through
`jobs/batch_distributions_d.json` contain completed no-SLO duration summaries.
They produce synthetic 2, 1, 2, and 3 hour maximum deferral windows. These are
not observed customer deadlines.

## Power

`power_model_params.json` contains one affine CPU-to-power fit per cell. The
active factory assigns normalized compute capacity 1 and 500 MW rated power to
each site:

```text
power_mw = 500 * (idle_power_fraction + slope * executed_work)
```

Executed work is bounded to `[0, 1]`. The fitted full-utilization values remain
below 500 MW because the measured affine coefficients do not sum to one.

## Electricity

`four_market_v2/source_manifest.json` binds 142 daily canonical panels:

| Market case | Physical series | Day-ahead price |
|---|---|---|
| CAISO NP15 | EIA CISO demand/wind/solar | CAISO OASIS NP15 |
| MISO Minnesota | EIA MISO demand/wind/solar | MISO MINN.HUB |
| SPP North | EIA SWPP demand/wind/solar | SPPNORTH_HUB |
| ISO-NE NEMA | EIA ISNE demand/wind/solar | ISO-NE location 4008 |

Each canonical row contains demand, wind, solar, derived net load, training-only
market scale, day-ahead LMP, causal 1-3 hour forecasts, forecast identity, and
quality status. The source has 114 October-January learning windows and 28
February development-validation windows. It exposes no test split.

## Generated factories

Run:

```powershell
python scripts\build_four_market_v2_factory.py
```

This creates `output/four_market_v2/factory/envelope_on/` and
`output/four_market_v2/factory/envelope_off/`. Both use the same four market
panels, direct 500 MW sites, and deadline windows.

- `envelope_on` caps new batch at 10% of fleet capacity and reclassifies excess
  batch as immediate service.
- `envelope_off` preserves the measured service/batch decomposition exactly.

Validate the data and runtime contract with:

```powershell
python scripts\run_four_market_v2_campaign.py preflight
python -m unittest tests.energy_model_v3.test_four_market_v2 -v
```
