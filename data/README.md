# Active Four-Market Data Contract

The joint ramp-and-monthly-net-load-peak experiment uses one raw-workload
protocol, `env/protocols/four_market_v2.yaml`, and one generated factory.
Cells a-d are active; cells e-h remain unused tier-curve workload holdouts,
not a separate test split.

## Workload and power

[`extract_cells.ipynb`](../extract_cells.ipynb) is the single extraction notebook.
Run it top to bottom in Colab or from the repository root with local Google
credentials. It defaults to project `aeee-thesis` and cells a-d; set
`INCLUDE_UNUSED_CELLS = True` to also extract e-h. It writes tier and aggregate
curves, power coefficients, and the a-d fitting scatter to
`build/clusterdata2019/data/`, then packages only that run's outputs in a zip.
Review those files before copying them into `data/` and rebuilding the factory.
The committed data and historical runs are unchanged by the consolidation.

`cells/cell_a_tiers.csv` through `cells/cell_d_tiers.csv` provide the raw measured
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

`power_model_params.json` contains the committed affine CPU-to-power coefficients.
CPU utilization is the sole predictor of measured power-domain utilization.
The power measurements include data-center-floor cooling, as documented in
[Google Data Center Power Trace, p. 2](https://raw.githubusercontent.com/google/cluster-data/master/power_trace_documentation.pdf#page=2).
Every synthetic site has normalized compute capacity 1 and a 500 MW rating:

```text
power_mw = 500 * (idle_power_fraction + dynamic_power_fraction * executed_work)
```

Cells a-d map to CAISO NP15, MISO Minnesota, SPP North, and ISO-NE NEMA,
respectively. This mapping is assumed, not a claim about Google's cell locations.

## Electricity calendar

`four_market_2025/calendar.json` lists the eleven training months of 2025
(all except May), the May validation month, each panel under
`four_market_2025/months/`, and `four_market_2025/frozen_stats.json`.
There are 334 training days and 31 validation days, with no test split.
May has already informed forecast-lead selection and is not an untouched test set.

Each panel contains one continuous month and one preceding warm hour, with
four market rows per hour. Queues and site-power history carry across midnight
and reset only at the month boundary. Every decision hour is scored. The warm
hour supplies the first observation and ramp starting point but is excluded
from monthly peak maxima.

Panels use demand and wind/solar generation from the CISO, MISO, SWPP, and ISNE
balancing authorities. EIA's hour-ending labels are shifted back one hour to
hour-beginning UTC. Net load is demand minus wind and solar. Day-ahead prices
are not fetched, and cost is not part of the study.

Forecast models are fitted on the eleven training months. Each row carries
gross-demand and net-load forecasts issued at that row's hour, with leads
1, 3, 6, and 12. A decision at hour t reads the realized grid row and forecasts
from t-1, targeting t, t+2, t+5, and t+11, together with current arrivals.
The 82-feature joint observation also includes running adjusted and original
normalized peaks over completed decision hours and the remaining-month
fraction. Forecast vintage and quality fields are provenance, not policy
observations.

## Fixed objective calibration

Regional scales $S_m$ are the gross-demand 95th percentiles over training
decision hours. The factory computes $C_R$ and $C_Q$ as arithmetic means of
the eleven monthly no-flexibility reference scores: the total sum of squared
adjusted normalized ramps and the sum of separate regional normalized peaks.
Both are whole-grid totals with the fleet, not signed impacts or hourly
averages, so dividing the objective's impact terms by them expresses each
change as a fraction of a grid total. Both positive scales stay frozen for
training and May validation. For the committed data, $C_R\approx6.03405775$ and
$C_Q\approx3.66459854$.

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
The [thesis](../latex/thesis_problemstatement.tex) defines the objective.

## Factory and campaign

From the repository root, run:

```powershell
python scripts\build_four_market_v2_factory.py
python scripts\run_four_market_v2_campaign.py preflight
```

The factory writes monthly fixtures and
`output/four_market_joint_v3/factory/factory.json`, referencing the calendar,
central frozen statistics, and panels rather than copying panels. Its fixed
objective calibration records the objective version and input identity.
Script and module names retain `four_market_v2` and `ramp_v6`.

Campaign outputs use `output/four_market_joint_v3/campaign/`, tagged pilots
use `output/four_market_joint_v3/pilot/<tag>/`, and checkpoints use
`models/four_market_joint_v3/`. Old `four_market_v2`,
`four_market_joint_v1`, and `four_market_joint_v2` (absolute-peak objective)
artifacts are historical and must not be resumed as
monthly-impact runs. The objective and normalization versions are checked
alongside data, observation, and training identities.

**Full joint ten-seed results are pending.** The planned seeds are 4101-4110,
trained on eleven months and paired with the same no-flexibility fleet on May.
Ramp-only, peak-only, and equal-weight joint runs are objective variants, not
additional scheduling baselines. See the [repository README](../README.md)
for the short pilot and full campaign commands.
