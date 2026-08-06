# Grid-Aware Spatio-Temporal Load Shaping

This repository is transitioning from the retired mixed/synthetic energy model
to a single-source CAISO energy archetype.

## Current status: v2 objective frozen; training protocol not launched

**Do not launch PPO retraining yet.**

The completed v1 campaigns, exact energy inputs, models, logs, outputs, and
manuscript snapshot are archived under:

```text
archive/energy_model_v1_mixed_20260805/
```

Historical DQN/CFWS work is separately archived under:

```text
archive/dqn_cfws_20260805/
```

## Energy model v2

The primary experiment is:

> A controlled geo-distributed workload-shaping experiment driven by one real
> May-2025 CAISO duck-curve/price archetype, time-shifted across US and global
> market slots.

Market slots:

- **US:** Pacific, Mountain, Central, Eastern.
- **Global:** Pacific, Central, Amsterdam, Singapore.

Net demand and price:

- come from one co-timestamped CAISO calendar;
- retain UTC and local timestamps;
- shift together using IANA time zones and daylight-saving rules;
- use the same price level in the primary experiment;
- cover a validated complete May calendar with adjacent-day buffers; and
- are graphed and reviewed before any new training begins.

The source build and market diagnostics are complete:

- 8,928 five-minute intervals (744 hours);
- real CAISO Today's Outlook net demand and solar;
- real CAISO OASIS NP15 hourly DAM LMP expanded stepwise;
- negative prices preserved;
- average local trough near 12:00 PDT and peak near 20:00 PDT;
- price/net-demand correlation about 0.895.

The corrected gate finds 8.0–17.4% joint headroom and 0.6–2.3% incremental
temporal headroom. Spatial routing remains the dominant lever.

Time-zone shifts are continuous, not circular. Each slot has equal duration,
but adjacent real boundary hours can produce small monthly-mean differences.
Singapore's 0.59% higher mean is such a boundary effect, not a regional price
premium or extra simulated time.

Regional price-level and real multi-market data are future sensitivity work,
not assumptions in the primary model. Historical 2019/2024 duck-curve
comparison and multi-year extrapolation are future work; they are not additional
training scenarios.

## Frozen historical conclusion

Energy-model v1 is reproducible as an 8,917-step synthetic objective (v1 only;
v2 has 8,928 steps) but is not valid evidence of synchronized real market
economics. Its OOF campaign failed the frozen joint-shaping headline criterion
and is retained only for historical analysis.

## Training-ready checkpoint

The protocol freezes **501,760 steps = 245 PPO rollouts**, 10 seeds per fold,
`gamma=1`, domain randomization, and every objective/assumption above.

1. Commit this checkpoint.
2. Run `python scripts/run_oof_campaign_v2.py --phase preflight`; it refuses
   dirty tracked source and writes the immutable hash/provenance record.
3. Launch with `python scripts/run_oof_campaign_v2.py --phase train --workers N`.
4. Evaluate only after all 80 completion records exist.

## Frozen simplifying assumptions

- Every modeled site is an equal 100 MW proxy with normalized capacity 1.0.
- Current measured batch arrival is present in the pre-action observation.
- Net demand retains a signed `[-1,1]` scale; real low/negative LMP rewards
  execution during renewable oversupply.
- Routing is unrestricted across all four slots; latency, residency, and
  movement costs are future work. Spatial headroom is therefore an optimistic
  upper bound, not a deployment-feasibility claim.
- MPC is future work; QP is a clairvoyant diagnostic only.
- Ramp rate is not directly penalized. Evaluation reports per-region 1h/3h
  maximum and p95 upward ramps; a ramp-aware reward is added only if v2
  policies worsen them.
- The primary deadline uses `flexibility_factor=1`; factors 0 and 2 are
  robustness cases.
- OOF holds out workload cells, not the single May-2025 energy calendar.

No-training ablations use a–d only. They show shifted market phase is the
dominant Global lever, while per-cell power heterogeneity materially increases
US headroom. Lower-slope routing is treated as a direct model consequence, not
a novel RL discovery. Fixed-α headroom is also reported for 50/100/200 MW.

`output/thesis_overview.md` is the living technical reference.
`thesis_paper.docx` is regenerated from `scripts/build_thesis_paper.py`; v1
learned-policy sections remain explicitly historical until v2 results replace
them.
