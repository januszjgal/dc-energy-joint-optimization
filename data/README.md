# Active Four-Market Data Contract

The joint ramp-and-monthly-net-load-peak experiment uses one raw-workload
protocol, `env/protocols/four_market_v2.yaml`, and one generated factory per
split specification (four cross-validation folds and the final factory).
Cells a-d form the training fleet; cells e-h form the test fleet, which serves
only the May 2026 test month.

## Workload and power

[`extract_cells.ipynb`](../extract_cells.ipynb) is the single extraction notebook.
Run it top to bottom in Colab or from the repository root with local Google
credentials. It defaults to project `aeee-thesis` and all eight cells; set
`INCLUDE_TEST_CELLS = False` to extract only a-d. It writes tier and aggregate
curves, power coefficients, cell capacities, and the a-d fitting scatter to
`build/clusterdata2019/data/`, then packages only that run's outputs in a zip.
Review those files before copying them into `data/` and rebuilding the factories.
The committed e-h files were extracted on 2026-06-12 with the same queries.

`cells/cell_a_tiers.csv` through `cells/cell_h_tiers.csv` provide the raw measured
five-minute normalized CPU, service, and batch curves from ClusterData2019.
The factory averages twelve samples per hour and uses the service and batch
columns without reshaping them. The 744-hour workload restarts each month;
shorter months use only the hours they need. Measured execution is treated as
synthetic arrival demand, not as observed job submissions or deadlines.

Service must run immediately. Every site uses a 24-slot batch completion window,
including the arrival slot: batch may run immediately or in the following
23 hours. Windows are clipped at month-end, when queues must be empty. Batch
releases follow global earliest-deadline-first order, with ties broken by origin
index and then arrival order. The decoder enforces service, capacity, and
deadlines as hard constraints, retaining the conservative deadline safety guard.

`power_model_params.json` contains the committed affine CPU-to-power coefficients
for all eight cells. CPU utilization is the sole predictor of measured
power-domain utilization. The power measurements include data-center-floor
cooling, as documented in
[Google Data Center Power Trace, p. 2](https://raw.githubusercontent.com/google/cluster-data/master/power_trace_documentation.pdf#page=2).
`cell_capacity.json` holds each cell's machine count and reference CPU capacity
(the sum of each machine's largest `capacity.cpus`, over machines that report
both a CPU and a memory capacity). Each synthetic site keeps its cell's capacity
relative to cell c, the largest training cell: compute capacity
`K = cpu_capacity / cpu_capacity_c` (0.897, 0.995, 1.000, 0.846 for a-d; 0.954,
1.443, 1.045, 1.354 for e-h) and rated power `500 * K` MW. Arrivals are the
cell's utilization times `K`, so one work unit is one hour of cell c's site at
full utilization, the same computation everywhere:

```text
power_mw = 500 * K * (idle_power_fraction + dynamic_power_fraction * executed_work / K)
```

Cells a-d map to the CAISO, MISO, SPP, and ISO-NE markets (IDs `CAISO`, `MISO`,
`SPP`, `ISONE`), respectively, and cells e-h to the same markets in the same order. Google gives
each cell's time zone in "Google cluster-usage traces v3" (p. 16): a, c, d
America/New_York; b, f America/Chicago; e Europe/Helsinki; g Asia/Singapore;
h Europe/Brussels. Those zones cannot give each fleet one cell per U.S. market,
so the study uses its own mapping; it is an assumption, not a claim about where
the cells are.

## Electricity calendar and splits

`four_market_2025/calendar.json` lists the twelve 2025 development months, the
four cross-validation folds (k1 Jan/May/Sep, k2 Feb/Jun/Oct, k3 Mar/Jul/Nov,
k4 Apr/Aug/Dec, each holding out three months and training on nine), and the
May 2026 test month, with each panel path and `four_market_2025/frozen_stats.json`.
The same split is defined in `energy_model_v3/splits.py` and the protocol; a test
keeps the three in step. The final policies train on all twelve months, and the
test month is scored once, with the final policies only.

Each panel contains one continuous month and one preceding warm hour, with
four market rows per hour. Queues and site-power history carry across midnight
and reset only at the month boundary. Every decision hour is scored. The warm
hour supplies the first observation and ramp starting point but is excluded
from monthly peak maxima.

Panels use demand and wind/solar generation from the CISO, MISO, SWPP, and ISNE
balancing authorities. EIA's hour-ending labels are shifted back one hour to
hour-beginning UTC. Monthly panels follow the Google trace's clock, a fixed
UTC-7: each month runs from 07:00 UTC on its first day to 07:00 UTC on the
first day of the next. Net load is demand minus wind and solar. Day-ahead prices
are not fetched, and cost is not part of the study. `four_market_2026_test/`
holds the raw test pull (23 April to 18:00 UTC on 1 June 2026, so the forecasts
have their week of history and their twelve-hour targets) and the May 2026 test panel.
One implausible value is kept as published: CAISO demand at 2025-07-31 20:00 UTC
is 11,819 MW against about 29,880 MW on either side.

Each row carries gross-demand and net-load forecasts issued at that row's hour,
with leads 1, 3, 6, and 12. The committed 2025 panels and the test panel hold
the final forecasts, fitted on all of 2025; each fold's panels, generated under
`output/four_market_joint_v3/cv-kX/data/`, hold forecasts fitted on the fold's
nine training months. A decision at hour t reads the realized grid row and
forecasts from t-1, targeting t, t+2, t+5, and t+11, together with current
arrivals. The 82-feature joint observation also includes running adjusted and
original normalized peaks over completed decision hours and the remaining-month
fraction. Forecast vintage and quality fields are provenance, not policy
observations.

## Fixed objective calibration

Regional scales $S_m$ are the gross-demand 95th percentiles over a split's
training hours. Each factory computes $C_R$ and $C_Q$ as arithmetic means of its
training months' no-flexibility reference scores with the training fleet: the
total sum of squared adjusted normalized ramps and the sum of separate regional
normalized peaks. Both are whole-grid totals with the fleet, not signed impacts
or hourly averages, so dividing the objective's impact terms by them expresses
each change as a fraction of a grid total. Both positive scales stay frozen for
the split's training and evaluation. For the final policies,
$C_R\approx6.10907155$ and $C_Q\approx3.65083499$; they also score the test month.

The default objective is
$J_e=0.5\sum_{t,m}I_{m,t}/C_R+0.5\sum_m\Phi_{m,e}/C_Q$, where
$I_{m,t}$ subtracts the original grid's squared ramp from the adjusted
squared ramp and $\Phi_{m,e}$ subtracts the original normalized monthly peak
from the adjusted one. Both subtractions are constant across schedules. There is no division by month length. Equal weights value
equal proportional changes relative to those fixed training references, not
equal achieved savings or learned influence. Report $J_{\mathrm{NF}}-J_{\mathrm{policy}}$
(positive is favorable) alongside signed monthly ramp totals and regional
peak scores. Use differences rather than percentages for the signed ramp
component. Regional peak reductions in MW are measured against the same
no-flexibility fleet, not the native no-fleet peak.
The [thesis](../latex/thesis_draft.tex) defines the objective.

## Factories, cross-validation, and campaign

From the repository root, run:

```powershell
python scripts\build_data_splits.py
python scripts\build_four_market_v2_factory.py
python scripts\run_four_market_v2_campaign.py preflight
```

The factory builder writes one factory per split under
`output/four_market_joint_v3/<spec>/factory/`, with monthly fixtures and a
`factory.json` that references the calendar, the spec's frozen statistics, and
its panels rather than copying them. Each window records the fleet that serves
it. The fixed objective calibration records the objective version and input
identity. Script and module names retain `four_market_v2` and `ramp_v6`.

Cross-validation outputs use `output/four_market_joint_v3/cv/<tag>/`, final
campaign outputs use `output/four_market_joint_v3/campaign/`, and checkpoints use
`models/four_market_joint_v3/`. The pilots under `output/four_market_joint_v3/pilot/`
used the old 11-month/May split and are superseded. Old `four_market_v2`,
`four_market_joint_v1`, and `four_market_joint_v2` (absolute-peak objective)
artifacts are historical and must not be resumed as monthly-impact runs. The
objective and normalization versions are checked alongside data, observation,
and training identities.

**Full joint ten-seed results are pending.** The planned final seeds are
4101-4110, trained on all twelve 2025 months and scored once on May 2026 against
the same no-flexibility fleet. Ramp-only, peak-only, and equal-weight joint runs
are objective variants, not additional scheduling baselines. See the
[repository README](../README.md) for the cross-validation and campaign commands.
