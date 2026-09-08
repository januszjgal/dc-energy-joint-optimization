Review of `latex/problemstatement_ieee_revised.tex` — 7 September 2026

The core problem is valid as a synthetic, hourly, divisible-work scheduling experiment. It is a reasonable master's thesis scope: construct a reproducible environment from public traces, train a causal scheduler, and measure whether flexible execution reduces a clearly defined regional net-load ramp score. It does not require job-level simulation, a power-flow model, a new RL algorithm, or a detailed network model. The main needs are tighter claims, a few data-pipeline corrections, and simpler notation and controls.

This review covers all 27 numbered equations, the unnumbered arithmetic and impact expansion, the README, extraction notebooks, forecasting and factory scripts, selected implementation tests, and the stored May fixture. Equation numbers below follow their order in the current source; line references refer to this revision. The thesis source and implementation were not edited, and training was not run. Direct reads of the raw data, environment, and preprocessing directories were unavailable, so raw-data completeness, interval alignment, and full runtime behavior are not certified here. Old PPO results are explicitly superseded in the README and are not evidence for this design.

**1. Clarify what the experiment represents.**

The controls optimize where and when synthetic compute work executes. EIA demand and generation are an exogenous historical background. The two datasets are joined to construct an experiment; neither original dataset is itself optimized.

At source lines 41, 47, 51, and 74–77, replace references to running every original job, real arrival demand, and delivering the same end usage with precise language about retained aggregate CPU work. The model preserves the volume of its synthetic work parcels. It does not reconstruct original submission times, task execution order, application outputs, or actual service guarantees. The measured workload has already passed through Google's scheduler before being reused as new arrivals.

Suggested central question:

> Using historical EIA grid data and aggregate Google workload traces, how much can an hourly scheduler reduce regional net-load ramps by shifting modeled compute work across sites and within assumed completion windows, compared with executing all work at its original site and hour?

Keep the fluid approximation. State near the first workload equation that each hour's aggregate work is assumed known at the beginning of that simulated hour. This is an input convention, not evidence that real service demand is known in advance.

Priority is a reasonable way to define a modeled flexible fraction, but “no SLO” does not establish that arbitrary one- or two-hour delays preserve application behavior. Replace the “safe to delay” assertion at lines 167–172 with an explicit assumption. Google's schema distinguishes scheduling priority from latency sensitivity. Its usage records contain average resource rates over measurement intervals, including shorter intervals when instances start or finish. [Google trace schema](https://github.com/google/cluster-data/blob/master/clusterdata_trace_format_v3.proto)

The selected priority bands agree with Google's example analysis. That example also uses a fixed capacity reference, so keeping this approach is defensible; describe the resulting fraction as utilization relative to the fixed reference, rather than the exact fraction of currently active machines that was busy. [Google analysis notebook](https://github.com/google/cluster-data/blob/master/clusterdata_analysis_colab.ipynb)

**2. Correct the forecast holdout mask before any new results.**

`scripts/build_causal_forecasts.py:207–213` shifts the target forward, then applies `is_training` only to the issue timestamp. Consequently, eligible April 30 issue rows can contain May 1 targets. With complete boundary data, the one-, two-, and three-hour fits admit one, two, and three May target observations respectively. The model-versus-persistence comparison at lines 266–275 uses the same issue-only rule. This contradicts the paper's statement at line 420 that forecast fitting never uses May.

Require both issue and target timestamps to belong to the intended training set, and apply that rule consistently in model selection. Fit feature standardization within each rolling cross-validation training fold, too; currently it precedes the folds. These are small pipeline fixes, not a reason to redesign the thesis.

Separately, `scripts/fetch_eia_panels.py:153–156` interpolates in both directions. If missing historical observations enter policy features after this operation, their values can depend on observations unavailable at the simulated decision time. Inspect the actual missingness report before asserting this happened. Use causal filling for observations, or explicitly distinguish retrospective reconstruction from online information. `quality_ok=True` at line 225 currently does not distinguish repaired observations.

A model fitted offline on eleven months can still define a causal policy within a held-out simulated month. Do not confuse that retrospective experiment with prospective forecasting through 2025. May also holds out grid conditions only: its workload sequence is reused during training in other calendar months.

**3. Keep the grid model, but narrow its interpretation.**

The equation `N = D - wind - solar` is a valid residual-load definition. It does not identify actual fossil generation, reserve procurement, or the entire amount of dispatchable generation needed locally. Other generation, storage, and interchange also participate in balancing. EIA explicitly includes both net generation and electricity interchange in supply. [EIA data explanation](https://www.eia.gov/tools/faqs/faq.php?id=100)

Use “proxy for regional ramping burden” in place of the stronger operational claims at lines 53 and 343. Report reductions in the modeled score; do not translate them into reserve, cost, emissions, or reliability savings without another model.

Use CAISO/CISO, MISO, SPP/SWPP, and ISO-NE/ISNE as the region names. The data used are balancing-authority totals; NP15, Minnesota, North, and NEMA imply a geographic resolution the experiment does not have. Renaming is simpler than adding local grid data.

Adding synthetic site power to historical demand is consistent with the stated additional-fleet experiment (lines 1011–1016). There is no need to subtract an estimated existing Google load. Keep the explanation that the experiment adds the same hypothetical fleet under each policy.

**4. Treat power as a calibrated proxy, not a measured idle-to-peak curve.**

The affine model is adequate for this scope. The observed CPU ranges in the paper are roughly 0.35–0.68, so evaluating it at zero or one is extrapolation. The intercept is the model's predicted idle power, not a direct idle measurement. State that at lines 272–274, rather than only in the limitations. The 500 MW parameter is a chosen reference scale, not observed site size or exactly the modeled full-utilization draw.

The extraction notebook averages `measured_power_util` over cell/hour without explicit power-domain capacity weighting (`extract_clusterdata2019_full.ipynb:185–196`). Google's public dataset describes measurements for power domains. Verify what the selected table represents and how its normalized observations should be aggregated before calling this an aggregate cell-power measurement. If capacity weights are unavailable, describe the fitted signal as a normalized power proxy. The present review did not establish whether a weighting correction is required. [PowerData2019 description](https://github.com/google/cluster-data/blob/master/PowerData2019.md)

Add PowerData2019 to the bibliography. A nonlinear power model or uncertainty model is not necessary merely to make this thesis valid. A small power-scale sensitivity is a more useful bounded addition.

Constant idle power cancels from ramp differences. It matters for reported MW, but it does not create scheduling flexibility. Different destination slopes mean conserved work need not conserve fleet energy; the worked transfer's unequal MW changes are therefore consistent.

**5. All numbered equations.**

| Equation | Source line | Assessment and appropriate action |
|---|---:|---|
| 1: fixed reference capacity | 121 | Correct definition of the chosen fixed denominator. It is not instantaneous active capacity. Keep; move extraction detail to methods if shortening. |
| 2: five-minute utilization | 142 | Dimensionally consistent. Its interpretation as bucket-average execution requires duration/alignment checks: accepting records of at least 300 seconds does not by itself prove they cover exactly the assigned bucket. Longer or misaligned rows need overlap weighting. Removing short rows means this is retained execution, not all execution. |
| 3: batch utilization | 201 | Correct partition of retained usage under the priority assumption. It does not establish real deadline flexibility. |
| 4: service residual | 207 | Correct and preserves the retained total, including unmatched priority as service. |
| 5: normalized power fit | 266 | Correct affine form. Qualify extrapolated idle/full values and verify the aggregation used as its target. |
| 6: grid net load | 349 | Correct residual-load proxy. Do not equate it with measured dispatchable generation. |
| 7: market scale | 388 | Valid positive, fixed normalization. Q95 demand is an experimental scaling choice, not demonstrated reserve capability. |
| 8: hourly service work | 561 | Correct averaging and units: fraction times work/hour times hour gives work. The start-of-hour revelation is a modeling assumption. |
| 9: hourly batch work | 567 | Correct on the same assumptions. Since `h=t`, a separate source-hour symbol is unnecessary in the main formulation. |
| 10: grid timestamp mapping | 612 | Correct; can be stated in prose. |
| 11: service conservation | 787 | Correct for divisible service with unrestricted destination eligibility. It imposes same-hour execution, not real latency guarantees. |
| 12: total batch execution | 804 | Correct connection between origin drainage and destination execution. The total can be derived from other decisions. |
| 13: queue balance | 824 | Correct conservation. Explicitly state the initial condition `B_i,0=0`. |
| 14: queue availability | 829 | Correct no-execution-before-arrival condition. Together with initial balance, it preserves nonnegative queues. |
| 15: inclusive deadline | 841 | Correct: H=3 permits arrival hour plus two later hours. End-of-month truncation is a reasonable finite-episode convention. |
| 16: cumulative deadline feasibility | 851 | Correct with the stated EDF rule, which serves earlier deadlines first within each origin. Keep this mathematical guarantee. |
| 17: terminal queue | 863 | Correct but redundant: truncated deadlines, availability, and conservation already imply zero terminal backlog. It can be prose. |
| 18: total site work | 874 | Correct; supports eliminating separate destination vectors from the control. |
| 19: capacity bound | 880 | Correct work-unit bound. Baseline feasibility requires this at every individual site, not only fleet-wide. |
| 20: runtime power | 886 | Correct dimensional conversion. The displayed coefficients give sensible positive modeled values over [0,1], subject to extrapolation limits. |
| 21: adjusted net load | 910 | Correct for adding a hypothetical fleet to the historical background. |
| 22: native ramp | 925 | Correct normalized average rate, units h^-1. A three-hour endpoint change is not a measure of all variation inside those three hours. |
| 23: adjusted ramp | 930 | Correct on the same basis. Required history must be initialized consistently. |
| 24: incremental squared impact | 946 | Correct sign and algebra; units h^-2. Subtracting the native term does not change which schedule minimizes the score. |
| 25: objective | 966 | Correct weighted monthly score. Positive weights yield a sensible convex quadratic objective over physical allocations for a known trajectory. Normalize by month length for cross-month comparisons. Offline optimization of a known trajectory and learning a causal policy are different solution settings. |
| 26: no-flexibility baseline | 1023 | Correct and feasible for the stored May workload, as checked below. |
| 27: improvement | 1030 | Correct sign: positive is better than no flexibility. Do not calculate percentage improvement by dividing by a potentially negative or near-zero incremental baseline score. |

The expansion at lines 994–1004 is correct: `(2 deltaN deltaP + deltaP^2)/(S*l*dt)^2`. The implied ISO-NE/MISO coefficient ratio is approximately 31.6. This is an explicit weighting preference, not an algebra error.

The worked power calculation gives 362.21 MW using the rounded work value and listed coefficients. Moving 0.3982 work units removes about 74.81 MW in MISO and adds 67.64 MW in CAISO. The summed idle-to-full range is about 905.80 MW. The displayed one-hour impact means are also consistent: 0.9575 and -6.145 in the table's units.

There is a small numerical wording error at lines 1154–1156: 4.03 is about 16.8 times 0.24, the largest other displayed absolute contribution, not four times. The table's mean is the unweighted one-hour component, not the full 0.40/0.60 reward. Explain that this example holds preceding power fixed and does not establish the total multi-hour benefit of the move.

**6. Two simplifications preserve the current physical scheduling problem.**

First, optimize total work at each destination instead of separate service and batch destination vectors. Let `L_t = sum_m s_m,t`. Choose batch release `z_t` and total site work `w_m,t`, satisfying

\[
\sum_m w_{m,t}=L_t+z_t,
\qquad 0\leq w_{m,t}\leq K_m\Delta t.
\]

EDF determines which batch parcels are drained. Any such allocation can be decomposed into service and batch without changing site power. For example, when total work is positive, allocate the fractions `L_t/(L_t+z_t)` and `z_t/(L_t+z_t)` of each site's work to the respective classes. Zero total work simply gives zero execution everywhere.

Thus one batch-release action and four destination logits can replace the existing nine raw controls. This equivalence depends on the current unrestricted routing assumptions. It would not automatically hold after imposing service-specific location restrictions.

Second, use the mean adjusted squared-ramp score directly:

\[
C(\mu)=\frac{1}{T|\mathcal M|}
\sum_{t,m}\sum_{\ell\in\{1,3\}}\omega_\ell
\left(\frac{N_{m,t}+P_{m,t}-N_{m,t-\ell}-P_{m,t-\ell}}
{S_m\ell\Delta t}\right)^2.
\]

For a fixed evaluation month, this has exactly the same minimizing schedules as the current J. Report `C(NF)-C(PPO)` and, when the baseline is positive,

\[
100\,\frac{C(\mathrm{NF})-C(\mathrm{PPO})}{C(\mathrm{NF})}.
\]

This is a reduction in the specified squared-ramp score, not automatically the same percentage reduction in ramp magnitude. Report physical per-region MW/h metrics alongside it. The native-subtracted reward can remain an implementation choice if helpful for training.

**7. Optional scope reductions, with their tradeoffs.**

- One common batch window, such as three slots, removes arbitrary differences between origins and permits a shared queue with a few age buckets. This changes the scenario, so it is optional. Keep 2/1/2/3 only if heterogeneous flexibility is part of the question.
- A one-hour primary objective with three-hour ramps as a secondary metric removes the arbitrary 0.40/0.60 tradeoff. This also changes the objective; keeping both horizons is valid if the weights are plainly described as choices.
- Retain unrestricted service routing only as an explicit optimistic assumption. With median service about 1.72 work units versus 0.44 batch units, geography can dominate the measured benefit. A simpler, more conservative alternative is local service with geographically and temporally flexible batch. That still studies both forms of flexibility, but narrows the control authority.
- Move CSV column inventories, extraction minutiae, repeated clock/index explanations, and the equation inventory to methods or an appendix. Keep one end-to-end worked example. The central problem statement needs the research question, assumptions, controls, conservation/capacity/deadline rules, score, and evaluation scope.

An unrestricted model's optimal improvement may bound improvement in a suitably nested constrained model. A PPO result in the unrestricted model is not itself a certified upper bound. Replace “upper bound” at lines 903 and 1205 with “optimistic scenario” unless an actual bound is established.

**8. The available workload supports simple feasibility safeguards.**

The stored May fixture (`output/four_market_v2/factory/windows/2025-05/fixture.json`) has 744 hours. Its maximum no-flexibility utilization per site is approximately 0.68198, 0.65410, 0.66715, and 0.62678, all below one. Its maximum new fleet work is 2.42440260, and maximum hourly batch arrivals are 0.77863597.

With at most two hours of deferral, a deadline-respecting backlog before current arrivals can contain only batch from the previous two hours. A conservative bound for clearing that entire backlog together with current arrivals is

\[
2.42440260+2(0.77863597)=3.98167454<4.
\]

Because routing is unrestricted, total work below four can be distributed within the four unit capacities. For this fixed workload, the bound is a simple sufficient check that deadline enforcement need not require sophisticated forecasts of future workload. It assumes previous deadlines have been respected; arbitrary invalid queues are not covered. Recheck it when changing arrival intensity, capacities, or windows. A hard allocation layer still must enforce due release, availability, and individual destination capacity.

**9. Clarify information and episode boundaries.**

Line 420 incorrectly says the available forecasts reach as far as the scheduler ever needs. Choosing `P_t` affects the ramp scored at `t+3`, while forecasts issued at `t-1` cover only `t` through `t+2`. The short forecast horizon is a reasonable design choice; it is not a sufficiency result based on the batch deadline.

Define the observation's queue urgency as work remaining by deadline or age, and specify the prior site powers needed by the score. Include time remaining in the episode if terminal deadline truncation is to be anticipated. There is no need to write a full MDP chapter here.

Keep continuous months and empty terminal queues. Specify exactly which four workload hours generate warm power. The terminal convention is acceptable, but only the last two arrival hours having truncated windows does not imply its influence on earlier decisions is confined to those hours. Three-hour ramp consequences after the end of the month are outside the finite scoring horizon; describe the horizon honestly rather than claiming no boundary effect exists.

**10. A modest, credible PPO evaluation.**

The immediate local baseline is necessary. On its own, beating it shows the benefit of a learned flexible schedule relative to no flexibility; it does not show PPO outperforms simple scheduling. I recommend one additional causal greedy scheduler using the same forecasts, capacity rules, and deadlines. No large algorithm benchmark is needed.

Under the current affine power model and linear physical constraints, an offline version can be written as a convex quadratic program. PPO is therefore a chosen approach for learning a causal policy, not a mathematical necessity. A full perfect-foresight benchmark is optional; a small forecast-based greedy allocation is enough to make the comparison more informative.

Reserve an untouched evaluation month if another month is used for tuning. If May is evaluated only after the protocol is fixed, one held-out grid-month case study is a defensible limited result. Do not simultaneously tune repeatedly on May and describe it as untouched testing. Multiple PPO seeds measure training variability on that grid trajectory; they do not create independent grid-month samples.

Report the primary score and percentage reduction, per-region ramp magnitudes, and zero late/dropped work or capacity violations. A few seeds, learning curves, and a small set of sensitivities are enough to begin. Avoid building a large significance-testing campaign before the environment and reward are stable. The planned PPO methods should also distinguish discounted training return from the undiscounted monthly evaluation score.

The immediate revision priorities are: tighten the synthetic-data and grid claims; correct the forecast split and verify missing-data handling; simplify controls and objective notation; clarify power extrapolation; and set a small, honest evaluation protocol. These improve validity without expanding the thesis into a much larger project.
