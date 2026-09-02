# Four-Market Grid-Ramp Smoothing for Geo-Distributed Data Centers

The active experiment schedules raw measured service and batch CPU curves from
ClusterData2019 cells a-d against four hourly market panels. It is a single,
fixed-workload proxy experiment, not a claim about identified Google facilities.
The thesis source is [`latex/thesis.tex`](latex/thesis.tex).

## Active design

- One protocol: [`env/protocols/four_market_v2.yaml`](env/protocols/four_market_v2.yaml).
- Four normalized-capacity sites, 500 MW each, map cells a-d to CAISO NP15, MISO
  Minnesota, SPP North, and ISO-NE NEMA.
- The raw service/batch tier curves are used unchanged. Hard capacity, EDF
  deadlines `[2, 1, 2, 3]`, and conservative data-derived future batch capacity
  enforce feasibility.
- The action shape is 9 and the observation shape is 100. The reward is negative
  equal-market, 1 h/3 h weighted incremental squared ramp impact.
- [`data/four_market_2025/calendar.json`](data/four_market_2025/calendar.json) lists
  334 training dates spanning 2025 outside May, 31 May validation dates, daily panel
  paths, and the central frozen-statistics path. There is no test split.
- The factory creates one fixture set and
  [`factory.json`](output/four_market_v2/factory/factory.json); it references
  panels in `data/four_market_2025/windows/` rather than copying them.

Day-ahead cost is reported post-hoc against status quo. It is not part of the
reward or a pass/fail rule.

## Locked ten-seed validation campaign

Seeds 4101-4110 each request 2,000,000 interactions and complete uninterrupted
(`resumed_from_interactions=0`) at the safe boundary of 2,045,952 interactions.
Each trains on the same 334 days and is paired with
status quo on the exact same 31 May validation days. The optimizer seed is
the statistical unit (`n=10`). Aggregation uses exact two-sided Wilcoxon
signed-rank testing, matched-pairs rank-biserial correlation (positive favors the
policy), Hodges-Lehmann shift, 10,000-draw seed-bootstrap intervals, post-hoc
cost/safety summaries, raw rollout learning curves, and descriptive slopes around
110,592 interactions and in the final 20%.

All ten policies beat status quo on the equal-market macro metric. The mean
policy-minus-status-quo impact was `-6.6161e-05` with a 95% seed-bootstrap
interval of `[-6.6486e-05, -6.5861e-05]`. The exact two-sided Wilcoxon result
was `p=0.001953125`; rank-biserial correlation was `1.0`.

The result is heterogeneous: CAISO, SPP, and ISO-NE improved in 10/10 seeds,
while MISO worsened in 10/10. Mean day-ahead cost was 0.123% above status quo
and all modeled work/safety violations were zero. Learning was still improving
at 110,592 interactions and was near a plateau by 2,045,952.

See [`aggregation.json`](output/four_market_v2/campaign/aggregation.json) and
[`learning_curve_aggregate.png`](output/four_market_v2/campaign/learning_curve_aggregate.png).

## Reproduce

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Build and preflight the one factory:

```powershell
python scripts\build_four_market_v2_factory.py
python scripts\run_four_market_v2_campaign.py preflight
python -m unittest tests.energy_model_v3.test_four_market_v2 tests.ramp_v6.test_ramp_math -v
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
| `data/four_market_2025/calendar.json` | Date split, daily panel paths, and frozen-statistics path |
| `data/four_market_2025/windows/` | Referenced canonical daily electricity panels (EIA grid, hour-beginning UTC) |
| `scripts/fetch_eia_panels.py` | Rebuilds the panels from the EIA API |
| `scripts/build_causal_forecasts.py` | Fits and writes the causal forecast columns |
| `data/power_model_params.json` | Committed CPU-to-power coefficients |
| `env/protocols/four_market_v2.yaml` | Sole active protocol |
| `energy_model_v3/four_market_v2.py` | One-factory runtime |
| `env/ramp_v6/` | Scheduler, queue, observation, and reward implementation |
| `output/four_market_v2/factory/` | Generated one-factory fixtures and `factory.json` |
| `output/four_market_v2/campaign/` | Ten-seed validation summaries, statistics, and learning curves |
| `scripts/` | Factory, campaign, aggregation, and figure commands |
| `latex/thesis.tex` | Publication source |
