# Joint Grid-Ramp Smoothing and Monthly Net-Load Peak Reduction

This is Janusz Gal's applied master's thesis at UNC Charlotte. The active
experiment schedules raw measured service and batch CPU curves from
ClusterData2019 cells a-d against four hourly market panels. It is a single,
fixed-workload proxy experiment, not a claim about identified Google facilities.
The working thesis document is
[`latex/thesis_problemstatement.tex`](latex/thesis_problemstatement.tex).
It currently contains the problem statement and will be expanded into the thesis.
Build it with the Tectonic recipe in VS Code; the PDF and SyncTeX output go in
`latex/build/`, which is ignored by Git. Superseded drafts remain in Git history.

## Active design

- One protocol: [`env/protocols/four_market_v2.yaml`](env/protocols/four_market_v2.yaml),
  including objective weights and fixed-reference normalization.
- Four normalized-capacity sites, 500 MW each, map cells a-d to CAISO NP15, MISO
  Minnesota, SPP North, and ISO-NE NEMA.
- **Continuous months.** Each calendar month of 2025 is one episode. Batch queues
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
- [`data/four_market_2025/calendar.json`](data/four_market_2025/calendar.json)
  lists the eleven training months (2025 outside May), the May validation month,
  the monthly panel paths, and the frozen-statistics path. May has already been
  consulted to select forecast lead times, so it is validation rather than an
  untouched test set. There is no separate test split.
- The factory creates one fixture per month and
  `output/four_market_joint_v3/factory/factory.json`; it references the panels in
  `data/four_market_2025/months/` rather than copying them. It freezes the training
  no-flexibility calibration and records its objective version and input identity.

Day-ahead prices are not fetched and cost is not part of the study.

## Joint monthly objective

Adjusted regional net load is $A_{m,t}=N_{m,t}+P_{m,t}$. Each fixed regional
scale $S_m$ is the training gross-demand 95th percentile. With $\Delta t=1$ hour,
$\Delta^{\mathrm{adj}}_{m,t}=(A_{m,t}-A_{m,t-1})/(S_m\Delta t)$.
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
over the eleven training months:

$$
C_R=\frac{1}{11}\sum_{e\in\mathcal E_{\mathrm{train}}}
  \sum_{t\in\mathcal T_e}\sum_m(\Delta^{\mathrm{adj}}_{m,t}(\mathrm{NF}))^2,
\qquad
C_Q=\frac{1}{11}\sum_{e\in\mathcal E_{\mathrm{train}}}\sum_m Q^{\mathrm{adj}}_{m,e}(\mathrm{NF}).
$$

Both references are whole-grid totals with the no-flexibility fleet: total
squared adjusted ramp and summed regional adjusted peaks. Dividing the impact
terms by them expresses each change as a fraction of a grid total, so equal
weights value equal fractional reductions. The fleet can move total ramps
proportionally more than total peaks, so ramp gains usually dominate. For the
committed data, $C_R\approx6.03405775$ and $C_Q\approx3.66459854$. Both remain frozen for training and May validation.
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

## Ten-seed validation campaign

The planned joint campaign uses seeds 4101-4110, each requesting 2,000,000
interactions and stopping at the first checkpoint boundary at or beyond it
(2,048,000 interactions at 4 environments,
512 steps per rollout, and a checkpoint every 25 rollouts).
Each seed trains on the same eleven months and is evaluated on the single
continuous May episode (744 hourly decisions, 31 UTC days), paired with the
status quo on exactly the same hours. The optimizer seed is the statistical
unit (`n=10`). Aggregation uses exact two-sided Wilcoxon signed-rank testing,
matched-pairs rank-biserial correlation, Hodges-Lehmann shift, 10,000-draw
seed-bootstrap intervals, per-UTC-day tables, safety totals, and learning-curve
slopes.

**Full monthly-impact results are pending.** Earlier ramp-only artifacts under
`four_market_v2` and mean-squared joint artifacts under `four_market_joint_v1`
are historical. The current objective and calibration have new version
identifiers; incompatible checkpoints cannot be resumed or silently reused.
A short pilot does not establish a ten-seed result.

## Reproduce

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Rebuild the data (optional; the committed monthly panels already carry the
causal forecast columns):

```powershell
python scripts\fetch_eia_panels.py --write
python scripts\build_causal_forecasts.py --write
```

Build and preflight the one factory, then run the tests:

```powershell
python scripts\build_four_market_v2_factory.py
python scripts\run_four_market_v2_campaign.py preflight
python -m pytest -q tests
```

Run a short joint pilot:

```powershell
python scripts\run_four_market_v2_pilot.py --ramp-weight 0.5 --seeds 4101 --timesteps 51200 --tag impact-50-50-smoke --workers 1
```

The default ramp weight is 0.5 from the protocol. Use `--ramp-weight 0` for
peak-only, `0.5` for joint, or `1` for ramp-only comparisons, with distinct tags.
These are three objective variants, not three additional scheduling baselines.

The unchanged thesis's July routing example is a regression test: baseline
$J=0.0056208656$, routed $J=0.0048005770$, improvement $0.0008202886$.
This includes every July hour and the following hour's rebound ramp.

**Pilot results:** the three-seed pilot in
`output/four_market_joint_v3/pilot/impact-50-50-1m/` uses the current
calibration. On May its final policies improve $J$ by 0.0153 to 0.0170 over the
no-flexibility baseline; `latex/ppo_solution.tex` reports the details.

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
| `data/cells/cell_X_tiers.csv` | Retained measured CPU/service/batch tier curves |
| `data/four_market_2025/calendar.json` | Month split, monthly panel paths, and frozen-statistics path |
| `data/four_market_2025/months/` | One continuous hourly EIA panel per month (one warm hour plus the month, hour-beginning UTC) |
| `scripts/fetch_eia_panels.py` | Rebuilds the monthly panels from the EIA API |
| `scripts/build_causal_forecasts.py` | Fits and writes the causal forecast columns |
| `data/power_model_params.json` | Committed CPU-to-power coefficients |
| `env/protocols/four_market_v2.yaml` | Sole active protocol |
| `energy_model_v3/four_market_v2.py` | One-factory runtime (one continuous episode per month) |
| `env/ramp_v6/` | Scheduler, queue, observation, and reward implementation |
| `ramp_rl/` | Trainer boundary, PPO runner, evaluation, and campaign statistics |
| `output/four_market_joint_v3/factory/` | Monthly fixtures, `factory.json`, and fixed objective calibration |
| `output/four_market_joint_v3/campaign/` | Monthly-impact ten-seed summaries, statistics, and learning curves (full results pending) |
| `output/four_market_joint_v3/pilot/<tag>/` | Tagged short pilot outputs |
| `models/four_market_joint_v3/` | Monthly-impact campaign and pilot checkpoints |
| `output/four_market_v2/`, `models/four_market_v2/` | Historical ramp-only artifacts |
| `output/four_market_joint_v1/`, `models/four_market_joint_v1/` | Historical mean-squared joint artifacts |
| `output/four_market_joint_v2/`, `models/four_market_joint_v2/` | Historical joint runs under the absolute-peak objective |
| `scripts/` | Data, factory, campaign, aggregation, and figure commands |
| `tests/` | Unit and integration tests, including causality, continuity, and deadline checks |
