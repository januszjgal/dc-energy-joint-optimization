# Four-Market Grid-Ramp Smoothing for Geo-Distributed Data Centers

The active experiment schedules raw measured service and batch CPU curves from
ClusterData2019 cells a-d against four hourly market panels. It is a single,
fixed-workload proxy experiment, not a claim about identified Google facilities.
The current problem-statement chapter is
[`latex/problemstatement_ieee_revised.tex`](latex/problemstatement_ieee_revised.tex);
[`latex/thesis.tex`](latex/thesis.tex) is an older draft.

## Active design

- One protocol: [`env/protocols/four_market_v2.yaml`](env/protocols/four_market_v2.yaml).
- Four normalized-capacity sites, 500 MW each, map cells a-d to CAISO NP15, MISO
  Minnesota, SPP North, and ISO-NE NEMA.
- **Continuous months.** Each calendar month of 2025 is one episode. Batch queues
  and site-power history carry across every midnight inside the month and reset
  only at the month boundary. One warm hour before the month gives the first
  decision its hour t-1 observation and the first ramp its starting point. Every hour
  of the month is scored. Work arriving in the last 23 hours has its execution
  window cut at the month's final hour, so the month closes with empty queues and
  nothing escapes the objective.
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
  Earliest-deadline-first drainage breaks ties by lowest origin index, then
  arrival order. Month-end clipping can give different arrival hours the same
  deadline. The projection layer enforces capacity and deadlines, releasing
  batch earlier when necessary to leave enough capacity to finish the backlog.
- The action shape is 9 and the observation shape is 73. The reward is the
  negative one-hour incremental squared ramp impact summed over the four
  markets, so a month's undiscounted return is exactly -J from the paper.
- [`data/four_market_2025/calendar.json`](data/four_market_2025/calendar.json)
  lists the eleven training months (2025 outside May), the May validation month,
  the monthly panel paths, and the frozen-statistics path. May has already been
  consulted to select forecast lead times, so it is validation rather than an
  untouched test set. There is no separate test split.
- The factory creates one fixture per month and
  [`factory.json`](output/four_market_v2/factory/factory.json); it references
  the panels in `data/four_market_2025/months/` rather than copying them.

Day-ahead prices are not fetched and cost is not part of the study.

## The one baseline

Exactly one comparison policy is implemented, called `status_quo` in the code
and the no-flexibility baseline in the paper: every service and batch arrival
executes at its origin site in its arrival hour, with no deferral and no
routing, on the same arrivals, power models, grid trajectory, forecasts, and
evaluation window as PPO. The reported improvement is

    improvement = J(status quo) - J(PPO)

so a positive value means PPO left the four regions with gentler ramps. The
evaluation JSON also keeps `policy_minus_status_quo_mean_incremental_ramp_impact`,
which is the same quantity with the opposite sign. The native grid without the
fleet is the reference built into the impact term itself, not a second
scheduling baseline.

## Ten-seed validation campaign

Seeds 4101-4110 each request 2,000,000 interactions and stop at the first
checkpoint boundary at or beyond it (2,048,000 interactions at 4 environments,
512 steps per rollout, and a checkpoint every 25 rollouts). PPO uses gamma 0.99.
Each seed trains on the same eleven months and is evaluated on the single
continuous May episode (744 hourly decisions, 31 UTC days), paired with the
status quo on exactly the same hours. The optimizer seed is the statistical
unit (`n=10`). Aggregation uses exact two-sided Wilcoxon signed-rank testing,
matched-pairs rank-biserial correlation, Hodges-Lehmann shift, 10,000-draw
seed-bootstrap intervals, per-UTC-day tables, safety totals, and learning-curve
slopes.

**Results: pending a rerun.** The previous ten-seed result (August 2026) was
trained on October 2025 to January 2026 and validated on February 2026 using
30-hour daily panels. It is superseded and not comparable: under that design
the three post-midnight hours, in which arrivals were zeroed and every site
collapsed to idle, carried 59% of the scored impact for the status quo alone
and flipped the sign of the daily objective. Continuous months remove that
artifact. The old `output/four_market_v2/campaign/` files are kept only as a
record of the earlier design.

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
| `output/four_market_v2/factory/` | Generated monthly fixtures and `factory.json` |
| `output/four_market_v2/campaign/` | Ten-seed summaries, statistics, and learning curves (pending rerun) |
| `scripts/` | Data, factory, campaign, aggregation, and figure commands |
| `tests/` | Unit and integration tests, including causality, continuity, and deadline checks |
