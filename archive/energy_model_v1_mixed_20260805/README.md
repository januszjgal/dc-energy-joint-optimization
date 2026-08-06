# Energy Model v1 — Mixed/Synthetic Archive

Archived on 2026-08-05 after the symmetric cross-cell OOF campaign completed.

## Why it was retired

The v1 experiment mixed:

- EIA-930-derived US net-demand arrays with discarded timestamps;
- synthetic Netherlands/Singapore net demand;
- six independently parameterized synthetic price curves;
- common seeded price noise across markets;
- fixed timezone offsets that did not account for May daylight saving;
- positional array joins to a rebased workload clock whose UTC origin was not
  retained; and
- 8,917-step episodes caused by interpolation/truncation rather than a complete
  8,928-slot May calendar.

The resulting experiment is reproducible as a synthetic objective but cannot
support claims about actual synchronized May-2019 market prices or local
duck-curve economics.

## Contents

- `data/` — exact v1 price, net-demand, year-shift, and solar inputs;
- `env/scenarios/` — exact mixed/synthetic and year-shift scenario YAMLs;
- `preprocess/`, `analysis/`, and `scripts/` — v1-only fetchers, figures,
  diagnostics, baseline/generalization analyses, and campaign orchestration;
- `models/oof/`, `logs/oof/`, `output/oof/` — completed 80-model symmetric
  a–d/e–h OOF campaign;
- `models/demand_charge/`, `logs/demand_charge/`,
  `output/demand_charge_models/` — demand-aware single-seed PPO campaign;
- `models/review*` and `output/review_campaign*` — earlier multi-seed PPO
  campaigns and analyses;
- `output/paper_figs/` and root-level evaluation artifacts — v1 manuscript and
  policy visualizations;
- `thesis_paper_v1.docx` — manuscript snapshot before the energy-model rewrite.

These artifacts are historical and must not be used by active training scripts.

## Successor

Energy model v2 is built and at its pretraining review gate as:

> A controlled geo-distributed workload-shaping experiment driven by one real
> CAISO duck-curve/price archetype, time-shifted across US and global market
> slots.

The US slots use Pacific, Mountain, Central, and Eastern time. The Global slots
use Pacific, Central, Amsterdam, and Singapore time. Net demand and price are
shifted together from one co-timestamped source, with no manual re-averaging in
the primary experiment.
