# Four-Market Grid-Ramp Smoothing for Geo-Distributed Data Centers

This repository models four 500 MW data-center sites attached to CAISO, MISO,
SPP, and ISO-NE. PPO schedules immediate and deferrable compute so changes in
data-center demand counter changes in grid net load. The objective compares the
combined trajectory \(N+P\) with the native grid trajectory \(N\); it does not
try to smooth data-center power \(P\) by itself.

The active thesis source is [`latex/thesis.tex`](latex/thesis.tex).

## Active design

- Google ClusterData2019 cells a-d supply measured workload.
- Cells e-h remain untouched as a future workload-generalization test.
- Each site has normalized compute capacity 1 and 500 MW rated power.
- Cell-specific PowerData2019 fits convert utilization to power as
  `500 * (idle_fraction + dynamic_fraction * utilization)`.
- Hourly decisions retain synthetic 2, 1, 2, and 3 hour batch windows for
  cells a-d.
- The action space has 9 preferences; the observation has 128 causal features.
- The factory contains 114 October-January learning days and 28 February
  development-validation days.
- No test split is exposed.

Two protocols differ only in workload admission:

- [`four_market_v2_envelope_on.yaml`](env/protocols/four_market_v2_envelope_on.yaml)
  caps newly admitted batch at 10% of fleet capacity and reclassifies excess
  measured batch as immediate service.
- [`four_market_v2_envelope_off.yaml`](env/protocols/four_market_v2_envelope_off.yaml)
  preserves the measured service/batch split without reclassification.

Both variants retain hard capacity, conservation, earliest-deadline-first
completion, causal forecasts, and the same ramp reward.

## Paired PPO validation

Each variant was trained from random initialization with paired seeds 4101,
4102, and 4103. Each run requested 100,000 interactions and ended on the first
complete rollout/episode boundary at 110,592 interactions. Every policy was
evaluated on all 28 February days.

| Mean February metric | Envelope on | Envelope off |
|---|---:|---:|
| Native-relative incremental ramp impact | -1.4058e-05 | -1.3983e-05 |
| Policy minus status-quo impact | -7.6006e-06 | -7.5253e-06 |
| Day-ahead cost ratio | 0.97481 | 0.97465 |
| Unserved, expired, or terminal work | 0 | 0 |
| Emergency fallback rate | 0 | 0 |

Negative impact means the data-center schedule reduced the grid's squared
net-load ramp. Removing the envelope won two of three paired seeds but did not
improve results consistently; its mean native-relative impact was
`7.53e-08` less favorable. The ablation therefore establishes that the raw
workload is feasible without reclassification, not that envelope removal
improves PPO performance. These results are development evidence, not a final
sealed experiment.

The compact comparison is
[`paired_aggregation.json`](output/four_market_v2/campaign/paired_aggregation.json).

## Data and calendar

The active market source is
[`data/four_market_v2/source_manifest.json`](data/four_market_v2/source_manifest.json).
Its four physical tuples use EIA balancing-authority demand, wind, and solar;
operator day-ahead LMP supplies separate economic context. Net load is demand
minus wind and solar.

The single measured May 2019 workload month is repeated across energy dates.
Grid conditions, prices, and causal forecasts vary across the 114 learning
days. February provides development-validation evidence on unseen energy dates,
while cells e-h remain unseen workload.

## Reproduce

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Build and validate both factories:

```powershell
python scripts\build_four_market_v2_factory.py
python scripts\run_four_market_v2_campaign.py preflight
python -m unittest tests.energy_model_v3.test_four_market_v2 tests.ramp_v6.test_ramp_math -v
```

Run the complete paired campaign and aggregate it:

```powershell
python scripts\run_four_market_v2_campaign.py run
python scripts\aggregate_four_market_v2_campaign.py
```

The six training runs took about 2 hours 39 minutes on the development machine.
PPO model containers remain local; tracked manifests and validation evidence
record the run.

Rebuild figures and the thesis:

```powershell
python scripts\build_energy_thesis_figures.py
Push-Location latex
xelatex -interaction=nonstopmode -halt-on-error thesis.tex
xelatex -interaction=nonstopmode -halt-on-error thesis.tex
Pop-Location
```

## Repository map

| Path | Role |
|---|---|
| `data/cells/` | ClusterData2019 aggregate and service/batch curves |
| `data/jobs/` | Per-cell completed no-SLO duration summaries |
| `data/four_market_v2/` | Hash-bound physical, price, and forecast source panels |
| `data/power_model_params.json` | Cell-specific CPU-to-power fits |
| `energy_model_v3/four_market_v2.py` | Active two-variant runtime factory |
| `env/ramp_v6/` | Reusable hourly scheduler, queue, reward, and constraint decoder |
| `ramp_rl/` | PPO training, evaluation, and evidence harness |
| `output/four_market_v2/campaign/` | Paired campaign evidence and aggregate findings |
| `scripts/` | Factory, campaign, aggregation, and figure commands |
| `latex/thesis.tex` | Sole publication source |
