# Grid-Aware Spatio-Temporal Load Shaping

This repository implements and evaluates a controlled single-source CAISO
energy archetype for geo-distributed workload shaping.

## Current status: frozen v2 campaign complete

**80/80 PPO models trained; all eight held-out evaluations completed.**

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

## Frozen held-out result

The broad joint-shaping headline criterion **failed**.

- **Global spatial PPO is the only robust learned success:** +0.90% and +1.19%
  mean savings on the two held-out workload folds, with positive optimizer CIs
  and complete service.
- **US spatial savings are not established:** −0.29% and +0.07%; both CIs cross
  zero, and the negative fold also loses to Round Robin.
- **Joint batch control is unstable:** only 9/40 batch seeds meet the frozen
  99.99% completion floor; it does not reliably improve over spatial-only PPO.
- Spatial PPO lowers the secondary demand-charge reference; joint PPO raises it
  and worsens rare maximum three-hour ramps.

Canonical results: `output/oof_v2_2025/results_report.md`.

## Post-v2 exploratory PPO v3 recovery study (non-headline)

**The frozen v2 held-out result above remains the primary evidence and is
unchanged.** After the frozen campaign, a post-hoc, exploratory-only PPO
recovery study (`ppo-reward-sweep-v3`) trained and reward-tuned joint
controllers on development cells a–d, to test whether the v2 joint-negative
result was an algorithmic/observability artifact rather than a ceiling on PPO
itself. Full raw artifacts: `output/ppo_v3_reward_sweep/`; full narrative:
`output/thesis_overview.md` §7A.

- **A genuine v2 state defect, reframed rather than retracted.** Frozen v2
  charged joint policies for terminal unfinished batch work while omitting
  episode/month position from the observation whenever the primary
  demand-charge rate was zero (the default). The v2 joint-negative result is
  therefore conditional on that partially observable state and the frozen
  budget — it is not clean proof that PPO cannot learn joint spatio-temporal
  control.
- **v3 partially repairs the state** (episode progress +
  `[1,3,6,12,24]`-step deadline buckets), keeps a fixed full-dollar evaluation
  objective regardless of
  training reward weights, and runs a successive-halving reward/training
  sweep (6 candidates → 4 reward-weight variants → full 10-seed development)
  at exact rollout-aligned budgets with matched seeds and a safety-first
  selection rule (`all_seeds_safe → safe_seed_count → worst-seed cost →
  mean cost`). **v3 trains and evaluates joint control only** — no
  spatial-only, temporal-only, or MPC controller is trained or activated in
  this protocol; **Status Quo remains the primary comparator**, not
  spatial-only PPO.
- **Budget scaling is a standalone compute-limitation diagnostic, not a
  retroactive rescue of v2.** More training budget monotonically improves
  Global's mean savings (0.89% → 1.23% → 1.76% at 5 seeds) but actively hurts
  US at 1M steps (0.21% → 0.35% → **−0.24%**, with batch completion falling to
  99.54% and 16.7 units expiring).
- **Selected final a–d results (10-seed replication):** US (151,552 steps)
  is 1/10 safe with **+0.051%** mean savings, optimizer CI crossing zero;
  Global (1,003,520 steps) is 1/10 safe with **+2.032%** mean savings and a
  fully positive optimizer CI. **Both fail the frozen 99.99% completion
  reliability requirement and the final all-seeds-safe gate**
  (`output/ppo_v3_reward_sweep/budget_selected_gate.json`: `"passed": false`
  for both regions).
- **Exact reading:** more compute improves Global's cost but not its safety;
  reward tuning and better observability reduce some catastrophic behavior
  (e.g., service backlog no longer spikes to millions of dollars) but do not
  yield a trustworthy unconstrained joint PPO controller.
- **e–h transfer is descriptive only.** A one-time post-selection check that
  re-evaluates the *same* region-selected budget models used for the a–d
  gate above — US R3_P1 at 151,552 steps, Global R0_P3 at 1,003,520 steps —
  on cells already exposed by frozen v2 (not fresh confirmatory data): US
  −0.155% (7/10 safe), Global +2.826% (5/10 safe). Not headline-eligible.
- **Limitations carried over:** v2's spatial-vs-joint reward-scale confound
  is not resolved because v3 never trains or compares a spatial-only
  controller; the equal 100 MW/unit-capacity proxy still removes real
  fleet-size heterogeneity and may help explain the weaker US opportunity
  seen in both v2 and v3.
- **A deadline-boundary audit found one remaining v3 observation defect.**
  At step `t`, the observation can include carried pool entries with
  `deadline_step <= t`, although the transition expires those entries before
  service. Selected US training had no expiry, but **9/10 selected Global
  trainings accumulated 127.597 expired units across their randomized
  episodes**. All selected a–d and descriptive e–h evaluations had zero expiry,
  so their reported failures are terminal-pool failures rather than this
  boundary artifact; nevertheless, v3 is not a fully repaired Markov/state-
  faithful formulation. A future protocol must compute actionable pool,
  urgency, and deadline buckets only from entries with `deadline_step > t`
  (optionally exposing unavoidable due-now mass separately), then retrain.
- **Safety remains soft, not hard.** Feasibility is still enforced only
  through reward penalties; the drain action is sigmoid-bounded by the
  inherited ±3 action clip, so a single step can reach at most
  `sigmoid(3) ≈ 95.257%` drain — but this cap is not the sole failure cause,
  since some seeds carry terminal batch pools far larger than one step's
  residual would explain. Widening the action bound alone would still never
  reach exact 100%, since a sigmoid link only approaches 1.0 asymptotically;
  the priority fix is instead a non-MPC safety layer with an **exact
  0.0–100% drain-reachability decoder / hard feasibility override** (not
  merely a wider sigmoid), which is future work, not part of v3.
- **Negative net-demand behavior — now with a deterministic conditioned
  probe against a uniform-routing benchmark, not just whole-episode
  averages.** The objective structurally rewards concentrating batch
  execution during negative-net-demand/low-price windows, and selected
  policies do show much lower whole-episode average drain rates than Status
  Quo (~0.43–0.63 vs. ~1.0). A deterministic probe
  (`negative_net_demand_probe` in `output/ppo_v3_reward_sweep/canonical_results.json`;
  `output/ppo_v3_reward_sweep/results_report.md`;
  `output/ppo_v3_reward_sweep/ppo_v3_negative_net_demand_probe.png`) — all 10
  selected a–d models/region re-evaluated with domain randomization
  disabled — splits mean drain, actual pool clearance, and batch routing
  share by negative- vs. non-negative-demand step, and compares routing
  share against the **conditioned uniform benchmark** (mean fraction of
  destinations negative when available), not raw negative-step incidence:
  - **US:** negative-step share 0.238911; mean drain 0.509108 (any
    destination negative) vs. 0.513233 (otherwise); clearance 0.508187 vs.
    0.507867; actual route share to negative destinations **0.611085** vs.
    uniform benchmark **0.611814** (**−0.07 pp**); 0 negative steps with mean
    drain > 90%.
  - **Global:** negative-step share 0.482527; mean drain 0.520032 vs.
    0.521231; clearance 0.501521 vs. 0.502063; actual route share
    **0.332631** vs. uniform benchmark **0.300487** (**+3.21 pp**); 0
    negative steps with mean drain > 90%.
  - **Exact reading:** neither region increases drain or clearance during
    negative-demand windows (both are slightly lower). Once compared against
    the uniform benchmark rather than zero, **the US shows no measurable
    negative-site routing preference** (essentially uniform, −0.07 pp), while
    **Global shows only a modest +3.21 pp positive spatial tilt** — the
    opposite of, and a correction to, an earlier reading that had the US
    showing "much stronger" redirection than Global. Neither region ever
    "blasts through" the batch pool on a negative-demand step. Descriptive
    only (domain randomization disabled, already-selected models, no
    retraining) — **no causal or confirmatory claim**.
- The demand-charge opt-in guard (`rate=0.0` by default, γ=1 required when
  enabled, period must evenly divide `max_steps`) is unchanged from v2 and is
  covered by two already-published exact identities in
  `scripts/smoke_test_demand_charge.py` — incremental demand charge sums to
  `rate × period max`, and dense arrival-minus-completion shaping sums to
  `λ_x × (expired + terminal pool)` — plus v3's new potential-shaping
  telescoping identity (`scripts/smoke_test_ppo_v3.py::test_potential_telescopes`).
  All three are exact-accounting regression tests, not statistical checks.

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

## Frozen campaign protocol

The protocol freezes **501,760 steps = 245 PPO rollouts**, 10 seeds per fold,
`gamma=1`, domain randomization, and every objective/assumption above.

- `output/oof_v2_2025/protocol.json` — frozen source/data/package provenance.
- `models/oof_v2_2025/manifest.json` — 80 model hashes and completion records.
- `output/oof_v2_2025/summary.json` — raw canonical held-out evaluation.
- `output/oof_v2_2025/canonical_results.json` — compact publication source.

The separate, non-frozen exploratory v3 recovery protocol and raw sweep/gate
results are under `output/ppo_v3_reward_sweep/` (`protocol.json`,
`round1_results.json`, `round2_results.json`, `full_results.json`/`full_gate.json`,
`budget_curve_results.json`, `budget_selection.json`,
`budget_selected_results.json`/`budget_selected_gate.json`,
`final_eh_transfer.json`, `budget_selected_eh_results.json`,
`canonical_results.json`, `results_report.md`,
`ppo_v3_negative_net_demand_probe.png`); see the "Post-v2 exploratory PPO v3
recovery study" section above.

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

`output/thesis_overview.md` is the living end-to-end technical reference.
`thesis_paper.docx` is regenerated from `scripts/build_thesis_paper.py`; v1
results remain explicitly historical.
