# Joint Grid-Ramp Smoothing and Monthly Net-Load Peak Reduction

This is a master's thesis repository for UNC Charlotte written by Janusz 'Jay' Gal. The 
experiment schedules raw measured service and batch CPU curves from
ClusterData2019 cells a-d (the training fleet) and e-h (the test fleet) against
four hourly market panels.
The working thesis document is
[`latex/thesis_draft.tex`](latex/thesis_draft.tex).

## Active design

- One protocol: [`env/protocols/four_market_v2.yaml`](env/protocols/four_market_v2.yaml),
  including objective weights and fixed-reference normalization.
- Four sites map cells a-d to the CAISO, MISO, SPP, and ISO-NE balancing
  authorities (market IDs `CAISO`, `MISO`, `SPP`, `ISONE`). Each keeps its cell's measured CPU capacity relative to cell c, the largest
  training cell (`data/cell_capacity.json`): compute capacity 0.897 / 0.995 / 1.000
  / 0.846 and reference power 500 MW times that, 449 / 498 / 500 / 423 MW. Arrivals
  are the cell's utilization times the site's capacity, in work units that add
  across sites.
- The test fleet maps cells e-h to the same four markets in the same order and is
  sized by the same rule: 0.954 / 1.443 / 1.045 / 1.354, or 477 / 721 / 523 / 677
  MW (2,398 MW against the training fleet's 1,869 MW). It serves only the test month.
- **Continuous months.** Each month of 2025 is one episode. Months follow the
  Google trace's clock, a fixed UTC-7 (`env/ramp_v6/clock.py`), so each starts at
  07:00 UTC on its first day and the trace's first hour lands on the month's
  first hour. Batch queues
  and site-power history carry across every midnight inside the month and reset
  only at the month boundary. One warm hour before the month gives the first
  decision its hour t-1 observation and the first ramp its starting point; it
  never enters a monthly peak maximum. Every hour of the month is scored.
  Work arriving in the last 23 hours has its execution window cut at the month's
  final hour, so the month closes with empty queues and nothing escapes the
  objective.
- **Causal timeline inside hour t.** The hour t-1 grid row, and the
  forecasts issued at t-1 with leads 1, 3, 6, and 12 (targeting t, t+2, t+5,
  and t+11: the current decision hour and 2, 5, and 11 hours later), are observed; the hour-t
  arrivals are revealed; the policy chooses service destinations, the batch
  volume to run now, and batch destinations; the realized hour-t grid values are
  revealed; site power, adjusted net load, and the reward are computed.
  [`tests/ramp_v6/test_continuous_month.py`](tests/ramp_v6/test_continuous_month.py)
  perturbs every row at or after hour t and asserts the hour-t observation is
  unchanged.
- **Batch completion window** `H = 24` slots for every site, including the
  arrival slot. Batch can execute immediately or in any of the next 23 hourly
  slots and must finish within 24 hours; it is never required to wait.
  Global earliest-deadline-first drainage breaks ties by lowest origin index, then
  arrival order. Month-end clipping can give different arrival hours the same
  deadline. The decoder enforces service, capacity, and deadlines as hard
  constraints, not reward penalties. Its conservative deadline safety guard
  releases batch earlier when needed to preserve room to finish the backlog.
- The action shape remains 9. The joint observation shape is 82 (73 + 8 + 1):
  it adds each region's running adjusted and original normalized net-load peaks
  over completed decision hours, plus the remaining-month fraction. These are
  causal features; peaks start at zero before the first decision.
- **Splits** ([`energy_model_v3/splits.py`](energy_model_v3/splits.py), mirrored in
  [`data/four_market_2025/calendar.json`](data/four_market_2025/calendar.json) and the
  protocol). The twelve months of 2025 are the development data. Four-fold grouped
  cross-validation holds out three interleaved months per fold (k1 Jan/May/Sep,
  k2 Feb/Jun/Oct, k3 Mar/Jul/Nov, k4 Apr/Aug/Dec) and trains on the other nine, so
  every 2025 month is validated once. The final policies train on all twelve
  months. May 2026, served by the e-h fleet, is the test month; it is scored once,
  with the final policies, after every design choice is fixed.
- Everything fitted to data is fitted on a split's training months only: forecast
  weights, the market scales $S_m$ and observation standardization, and the
  objective references $C_R$ and $C_Q$. The committed panels and
  `frozen_stats.json` hold the final (all-2025) values; each fold's own copies are
  generated under `output/four_market_joint_v3/cv-kX/data/`.
- There is one factory per split specification (`cv-k1` .. `cv-k4`, `final`) under
  `output/four_market_joint_v3/<spec>/factory/`. Each records which fleet serves
  each window, freezes the spec's training no-flexibility calibration, and records
  its objective version and input identity.

Day-ahead prices are not fetched and cost is not part of the study.

## Joint monthly objective

Adjusted regional net load is $A_{m,t}=N_{m,t}+P_{m,t}$. Each fixed regional
scale $S_m$ is the training gross-demand 95th percentile. Every interval is one
hour, so $\Delta^{\mathrm{adj}}_{m,t}=(A_{m,t}-A_{m,t-1})/S_m$ is a fraction of the
market scale per hour.
The incremental impact subtracts the original grid's squared ramp. Each region
also has its own monthly peak:

$$
\begin{aligned}
I_{m,t}(\mu)&=\left(\Delta^{\mathrm{adj}}_{m,t}(\mu)\right)^2
  -\left(\Delta^{\mathrm{original}}_{m,t}\right)^2,\\
Q^{\mathrm{adj}}_{m,e}(\mu)&=\max_{t\in\mathcal T_e}\frac{A_{m,t}(\mu)}{S_m},
\qquad
Q^{\mathrm{original}}_{m,e}=\max_{t\in\mathcal T_e}\frac{N_{m,t}}{S_m},\\
\Phi_{m,e}(\mu)&=Q^{\mathrm{adj}}_{m,e}(\mu)-Q^{\mathrm{original}}_{m,e}.
\end{aligned}
$$

The objective sums $I_{m,t}$ over the month; negative impacts are valid.
Like $I_{m,t}$, the peak impact $\Phi_{m,e}$ subtracts the original grid. That
shifts every schedule's $J_e$ by the same constant, leaving the best schedule and
every improvement unchanged, and keeps the grid's own peak records out of the
reward. It sums the separate regional peak impacts, not the maximum of the combined regions,
data-center power peaks, or hourly squared loads. The warm hour participates
in the first ramp only. The fixed positive references are arithmetic means
over a split's training months (nine per fold, twelve for the final policies):

$$
C_R=\frac{1}{|\mathcal E_{\mathrm{train}}|}\sum_{e\in\mathcal E_{\mathrm{train}}}
  \sum_{t\in\mathcal T_e}\sum_m(\Delta^{\mathrm{adj}}_{m,t}(\mathrm{NF}))^2,
\qquad
C_Q=\frac{1}{|\mathcal E_{\mathrm{train}}|}\sum_{e\in\mathcal E_{\mathrm{train}}}\sum_m Q^{\mathrm{adj}}_{m,e}(\mathrm{NF}).
$$

Both references are whole-grid totals with the no-flexibility fleet: total
squared adjusted ramp and summed regional adjusted peaks. Dividing the impact
terms by them expresses each change as a fraction of a grid total, so equal
weights value equal fractional reductions. The fleet can move total ramps
proportionally more than total peaks, so ramp gains usually dominate. For the
final policies, $C_R\approx6.10907155$ and $C_Q\approx3.65083499$; the folds give
$C_R$ from 5.86 to 6.62 and $C_Q$ from 3.57 to 3.74. Each stays frozen for its
split's training and evaluation, and the final values also score the test month.
The objective is

$$
J_e(\mu)=\frac{\lambda_r}{C_R}\sum_{t\in\mathcal T_e}\sum_m I_{m,t}(\mu)
       +\frac{\lambda_p}{C_Q}\sum_m \Phi_{m,e}(\mu),
\qquad \lambda_r+\lambda_p=1.
$$

Weights are nonnegative, with $\lambda_r=\lambda_p=0.5$ by default. Equal weights
give the same value to equal proportional changes relative to the fixed training
references, not equal achieved savings or learned influence. A better joint
score need not improve both components.

The raw step reward is

$$
r_t=-\frac{\lambda_r}{C_R}\sum_m I_{m,t}
    -\frac{\lambda_p}{C_Q}\sum_m\left[(M_{m,t}-M_{m,t-1})-(O_{m,t}-O_{m,t-1})\right],
$$

where $M_{m,t}$ and $O_{m,t}$ are the running normalized adjusted and original
peaks over decision hours. Set $M_{m,-1}=O_{m,-1}=0$; the first peak increment is
therefore $P_{m,0}/S_m$, and later updates take running maxima. Original-grid
records cancel, so they never enter the reward. No warm-hour peak is included.
There is no episode-length divisor. Negative ramp impacts earn positive ramp
rewards. The raw, undiscounted monthly sum is exactly $-J_e$. PPO uses gamma 0.99 and
reward normalization, so it is an approximate solution method, not an exact
optimizer of the undiscounted objective or a guarantee of global optimality.

## The one baseline

Exactly one comparison policy is implemented, called `status_quo` in the code
and the no-flexibility baseline in the thesis: every service and batch arrival
executes at its origin site in its arrival hour, with no deferral and no
routing, on the same arrivals, power models, grid trajectory, forecasts, and
evaluation window as PPO. The reported improvement is

    improvement = J(status quo) - J(PPO)

so a positive value means PPO improved the joint score. Report the monthly
sum of ramp impacts, summed regional peak impacts, and regional adjusted peaks
in MW separately. Hourly mean impacts and absolute squared ramps remain
diagnostics, not inputs to $J_e$. Ramp percentages are not reported because
this component is signed; use raw or training-reference-scaled differences.

The native grid without the fleet supplies the reference in $I_{m,t}$, not a
second scheduling baseline. With nonnegative
added site power, peak reduction means relative to the same fleet without
flexibility, never a peak below the native grid without the fleet.

## Cross-validation

[`scripts/run_four_market_cv.py`](scripts/run_four_market_cv.py) trains one fresh
policy per (fold, seed) on the fold's nine training months and evaluates every
saved milestone on each of its three held-out months, one continuous episode per
month, against the status quo on the same hours. Results go to
`output/four_market_joint_v3/cv/<tag>/<fold>/seed-<seed>/`, and `cv.json` collects
the final improvement of every (month, seed) and its averages. The test month is
never touched here.

## Ten-seed final campaign and test

The final campaign uses seeds 4101-4110, each requesting 1,000,000
interactions, the length the cross-validation tested, and stopping at the first
checkpoint boundary at or beyond it (1,024,000 interactions at 4 environments,
512 steps per rollout, and a checkpoint every 25 rollouts).
Each seed trains on all twelve months of 2025 and is then evaluated once on the
continuous May 2026 test episode (744 hourly decisions on the trace clock), served by
the e-h fleet and paired with the status quo on exactly the same hours. Nothing
about the design may change after this evaluation. The optimizer seed is the
statistical unit (`n=10`). Aggregation uses exact two-sided Wilcoxon signed-rank testing,
matched-pairs rank-biserial correlation, Hodges-Lehmann shift, 10,000-draw
seed-bootstrap intervals, per-UTC-day tables, safety totals, and learning-curve
slopes.

**Test-month result (May 2026, ten seeds):** mean improvement 0.0147 (0.0125-0.0171)
over no flexibility, every seed positive; per-seed summaries and
`aggregation.json` are under `output/four_market_joint_v3/campaign/`, and
`scripts/build_test_month_results.py` writes the thesis numbers to
`docs/figures/ppo_test/test_month.json`. Earlier ramp-only artifacts under
`four_market_v2` and mean-squared joint artifacts under `four_market_joint_v1`
are historical. The current objective and calibration have new version
identifiers; incompatible checkpoints cannot be resumed or silently reused.
A short pilot does not establish a ten-seed result.

## Reproduce

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

To re-extract the Google workload and power inputs, run
[`extract_cells.ipynb`](extract_cells.ipynb) from top to bottom in Colab or
locally from the repository root. It defaults to billing project `aeee-thesis`
and all eight cells (a-d training, e-h test); set `INCLUDE_TEST_CELLS = False` for
a-d only. It stages a zip under `build/clusterdata2019/` for review before
replacing committed data. The existing inputs can be used without rerunning it.

Rebuild the EIA panels, the test pull, and every split's data (optional; the
committed panels already carry the final forecasts, and the factory builder
regenerates the fold data it needs):

```powershell
python scripts\fetch_eia_panels.py --write
python scripts\fetch_eia_panels.py --test-raw
python scripts\build_causal_forecasts.py --select
python scripts\build_data_splits.py
```

Build the five factories and preflight the final one, then run the tests:

```powershell
python scripts\build_four_market_v2_factory.py
python scripts\run_four_market_v2_campaign.py preflight
python -m pytest -q tests
```

Run the four-fold cross-validation (three seeds, six workers):

```powershell
python scripts\run_four_market_cv.py --seeds 4101 4102 4103 --timesteps 1000000 --tag cv-raw-obs-1m --workers 6
```

The default ramp weight is 0.5 from the protocol. Use `--ramp-weight 0` for
peak-only, `0.5` for joint, or `1` for ramp-only comparisons, with distinct tags.
These are three objective variants, not three additional scheduling baselines.

The thesis's July routing example is a regression test against the final
factory: baseline $J=0.0051421164$, routed $J=0.0044403453$, improvement
$0.0007017711$. This includes every July hour and the following hour's rebound
ramp.

**Earlier pilots:** `output/four_market_joint_v3/pilot/` holds two three-seed
pilots under the old 11-month/May split: `impact-50-50-1m` (equal-size fleet,
ridge forecasts) and `ols-forecast-50-50-1m` (sized fleet, five-input
least-squares forecasts). Both are superseded by the cross-validation and are
not comparable with the current factories.

Run and aggregate the locked campaign (only aggregate after all ten summaries
exist):

```powershell
python scripts\run_four_market_v2_campaign.py run --workers 5
python scripts\aggregate_four_market_v2_campaign.py
```

Rebuild the two electricity figures:

```powershell
python scripts\build_energy_thesis_figures.py
```

## Repository map

Script and module names retain `four_market_v2` and `ramp_v6`; new artifacts
use `four_market_joint_v3` to keep them separate from historical runs.

| Path | Role |
|---|---|
| `extract_cells.ipynb` | Single Google workload and power extraction notebook |
| `data/cells/cell_X_tiers.csv` | Retained measured CPU/service/batch tier curves |
| `data/four_market_2025/calendar.json` | Development months, folds, test month, panel paths, and frozen-statistics path |
| `data/four_market_2025/months/` | One continuous hourly EIA panel per 2025 month (one warm hour plus the month, hour-beginning UTC), with final scales and forecasts |
| `data/four_market_2026_test/` | Raw May 2026 test pull and the test panel |
| `energy_model_v3/splits.py` | Development months, cross-validation folds, test month, and both fleets |
| `scripts/fetch_eia_panels.py` | Rebuilds the monthly panels and fetches the test window from the EIA API |
| `scripts/build_causal_forecasts.py` | Selects the forecast inputs by cross-validation and fits the forecasts |
| `scripts/build_data_splits.py` | Writes final statistics, calendar, forecasts, test panel, and each fold's data |
| `scripts/run_four_market_cv.py` | Four-fold cross-validation of PPO |
| `data/power_model_params.json` | Committed CPU-to-power coefficients |
| `data/cell_capacity.json` | Per-cell CPU capacity that sets each site's relative size |
| `env/protocols/four_market_v2.yaml` | Sole active protocol |
| `energy_model_v3/four_market_v2.py` | Factory runtime (one continuous episode per month, one factory per split) |
| `env/ramp_v6/` | Scheduler, queue, observation, and reward implementation |
| `ramp_rl/` | Trainer boundary, PPO runner, evaluation, and campaign statistics |
| `output/four_market_joint_v3/<spec>/` | Generated fold data and per-split factories with fixed objective calibration (not tracked) |
| `output/four_market_joint_v3/cv/<tag>/` | Cross-validation summaries and learning curves |
| `output/four_market_joint_v3/campaign/` | Final ten-seed test summaries, statistics, and learning curves (pending) |
| `output/four_market_joint_v3/pilot/<tag>/` | Superseded pilots under the old split |
| `models/four_market_joint_v3/` | Cross-validation, campaign, and pilot checkpoints |
| `output/four_market_v2/`, `models/four_market_v2/` | Historical ramp-only artifacts |
| `output/four_market_joint_v1/`, `models/four_market_joint_v1/` | Historical mean-squared joint artifacts |
| `output/four_market_joint_v2/`, `models/four_market_joint_v2/` | Historical joint runs under the absolute-peak objective |
| `scripts/` | Data, factory, campaign, aggregation, and figure commands |
| `tests/` | Unit and integration tests, including causality, continuity, and deadline checks |
