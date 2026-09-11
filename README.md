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
  `output/four_market_joint_v1/factory/factory.json`; it references the panels in
  `data/four_market_2025/months/` rather than copying them. It freezes the training
  no-flexibility calibration and records its objective version and input identity.

Day-ahead prices are not fetched and cost is not part of the study.

## Joint monthly objective

Adjusted regional net load is $A_{m,t}=N_{m,t}+P_{m,t}$. Each fixed regional
scale $S_m$ is the training gross-demand 95th percentile. With $\Delta t=1$ hour,
$\Delta^{\mathrm{adj}}_{m,t}=(A_{m,t}-A_{m,t-1})/(S_m\Delta t)$.
For a month $e$ with $T_e$ decision hours:

$$
\begin{aligned}
R_e(\mu)&=\frac{1}{T_e}\sum_{t\in\mathcal T_e}\sum_m
  \left(\Delta^{\mathrm{adj}}_{m,t}(\mu)\right)^2,\\
Q_e(\mu)&=\sum_m\max_{t\in\mathcal T_e}\frac{A_{m,t}(\mu)}{S_m}.
\end{aligned}
$$

$R_e$ is the mean hourly fleet ramp score. $Q_e$ sums separate regional monthly
net-load peaks, not the maximum of the combined regions, data-center power peaks,
or hourly squared loads. The warm hour participates in the first ramp only.
The fixed positive references are arithmetic means over the eleven training months:

$$
C_R=\frac{1}{11}\sum_{e\in\mathcal E_{\mathrm{train}}}R_e(\mathrm{NF}),
\qquad
C_Q=\frac{1}{11}\sum_{e\in\mathcal E_{\mathrm{train}}}Q_e(\mathrm{NF}).
$$

They use absolute adjusted-grid scores, never signed incremental ramp impacts,
and remain frozen for all training and May validation. The objective is

$$
J_e(\mu)=\lambda_r\frac{R_e(\mu)}{C_R}+\lambda_p\frac{Q_e(\mu)}{C_Q},
\qquad \lambda_r+\lambda_p=1.
$$

Weights are nonnegative, with $\lambda_r=\lambda_p=0.5$ by default. Equal weights
give the same value to equal proportional changes relative to the fixed training
references, not equal achieved savings or learned influence. A better joint
score need not improve both components.

The raw step reward is

$$
r_t=-\frac{\lambda_r}{T_eC_R}\sum_m(\Delta^{\mathrm{adj}}_{m,t})^2
    -\frac{\lambda_p}{C_Q}\sum_m(M_{m,t}-M_{m,t-1}),
$$

where $M_{m,t}$ is the running normalized adjusted peak over decision hours.
Set $M_{m,-1}=0$; the first increment initializes $M_{m,0}=A_{m,0}/S_m$,
and later updates take the running maximum. No warm-hour peak is included.
The raw, undiscounted monthly sum is exactly $-J_e$. PPO uses gamma 0.99 and
reward normalization, so it is an approximate solution method, not an exact
optimizer of the undiscounted objective or a guarantee of global optimality.

## The one baseline

Exactly one comparison policy is implemented, called `status_quo` in the code
and the no-flexibility baseline in the thesis: every service and batch arrival
executes at its origin site in its arrival hour, with no deferral and no
routing, on the same arrivals, power models, grid trajectory, forecasts, and
evaluation window as PPO. The reported improvement is

    improvement = J(status quo) - J(PPO)

so a positive value means PPO improved the joint score. Report raw ramp score
$R_e$, incremental ramp diagnostics
$I_{m,t}=(\Delta^{\mathrm{adj}}_{m,t})^2-(\Delta^{\mathrm{original}}_{m,t})^2$,
peak score $Q_e$, and regional adjusted peaks in MW and their reductions from
the same no-flexibility fleet separately. The incremental-ramp comparison is
not the sign-reversed joint improvement.

The native grid without the fleet is a diagnostic reference, not a second
scheduling baseline; $J_e$ is not a signed comparison with it. With nonnegative
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

**Full joint results are pending.** Earlier ramp-only campaign results and
checkpoints under `output/four_market_v2/` and `models/four_market_v2/` are
historical, not joint results or inputs to reuse. A short pilot does not
establish a ten-seed result.

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
python scripts\run_four_market_v2_pilot.py --ramp-weight 0.5 --seeds 4101 --timesteps 51200 --tag joint-50-50-smoke --workers 1
```

The default ramp weight is 0.5 from the protocol. Use `--ramp-weight 0` for
peak-only, `0.5` for joint, or `1` for ramp-only comparisons, with distinct tags.
These are three objective variants, not three additional scheduling baselines.

**Smoke result:** Seed 4101 completed 51,200 interactions and the full 744-hour
May evaluation with no unserved service, unfinished/expired batch, terminal work,
or certificate violations. Joint improvement was -0.00032585 (policy
$J=0.92269678$, no-flexibility $J=0.92237093$): feasible execution, not evidence
of performance improvement. Results are in
`output/four_market_joint_v1/pilot/joint-50-50-smoke/pilot.json`.

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
use `four_market_joint_v1` to keep them separate from historical runs.

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
| `output/four_market_joint_v1/factory/` | Monthly fixtures, `factory.json`, and fixed objective calibration |
| `output/four_market_joint_v1/campaign/` | Joint ten-seed summaries, statistics, and learning curves (full results pending) |
| `output/four_market_joint_v1/pilot/<tag>/` | Tagged short pilot outputs |
| `models/four_market_joint_v1/` | Joint campaign and pilot checkpoints |
| `output/four_market_v2/`, `models/four_market_v2/` | Historical artifacts only; not reused by joint runs |
| `scripts/` | Data, factory, campaign, aggregation, and figure commands |
| `tests/` | Unit and integration tests, including causality, continuity, and deadline checks |
