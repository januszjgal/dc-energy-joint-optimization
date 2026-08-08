# Corrected v5 TD3+BC Results

## Verdict

The corrected, cryptographically verified post-RL TD3+BC network clears the predeclared a-d savings and safety gates in both regions. The teacher is absent during reward updates and inference. Every post-RL seed has nonzero actor and critic changes and 2,048 TD3 updates.

| Region | BC-only mean | Post-RL mean | Post-RL minimum | RL - BC mean | e-h descriptive mean | Normal decoder adjustment | Emergency fallback |
|---|---:|---:|---:|---:|---:|---:|---:|
| US | 6.289% | 6.290% | 6.112% | +0.0013 pp | 5.685% | 98.22% | 0.00% |
| GLOBAL | 14.017% | 14.000% | 13.822% | -0.0172 pp | 12.890% | 95.12% | 0.00% |

The result is primarily an imitation result preserved by genuine reward training: post-RL minus BC is nearly zero in US and slightly negative in Global. It is therefore inaccurate to attribute the full absolute savings to the short TD3 phase.

Routine constraint-decoder adjustment is frequent and must not be confused with emergency shielding. The decoder maps network preferences to feasible service, origin, and destination amounts on most steps; the separate emergency fallback remains unused.

## Development and transfer statistics

### US

- a-d post-RL mean: **6.290226%**; minimum: **6.112360%**; optimization-seed interval: [6.106133%, 6.474319%].
- e-h descriptive post-RL mean: **5.684974%**; minimum: **5.319743%**.
- Frozen greedy demonstration teacher: 6.088525% a-d and 6.343857% e-h. This is the offline label policy, not an RL result.
- Separate exact-native analytic benchmark: 7.215991% a-d and 7.227750% e-h. It is not the demonstration teacher.
- Post-RL network captures 87.17% of the exact-native a-d benchmark.

### GLOBAL

- a-d post-RL mean: **14.000051%**; minimum: **13.822014%**; optimization-seed interval: [13.792241%, 14.207862%].
- e-h descriptive post-RL mean: **12.889597%**; minimum: **12.037511%**.
- Frozen greedy demonstration teacher: 13.886388% a-d and 13.914521% e-h. This is the offline label policy, not an RL result.
- Separate exact-native analytic benchmark: 15.715358% a-d and 16.361134% e-h. It is not the demonstration teacher.
- Post-RL network captures 89.09% of the exact-native a-d benchmark.

## Secondary demand-charge finding

Demand charge was excluded from the primary training objective and is reported at the illustrative $15/kW-cycle reference. The final policy concentrates load spatially, increasing the summed site peaks and the secondary demand-charge estimate.

| Region/scope | Primary savings | Demand-charge change | Combined primary + demand-charge sensitivity |
|---|---:|---:|---:|
| US a-d | 6.290% | +18.536% ($870,104) | -4.058% |
| US e-h descriptive | 5.685% | +16.152% ($747,240) | -3.630% |
| GLOBAL a-d | 14.000% | +18.536% ($870,104) | +0.475% |
| GLOBAL e-h descriptive | 12.890% | +16.152% ($747,240) | +0.566% |

This sensitivity is not a tariff forecast: the simulator uses five-minute peaks and one illustrative rate, while real demand tariffs commonly use 15-minute windows and utility-specific ratchets. It nevertheless shows that optimizing energy and grid stress alone does not guarantee peak-demand savings.

## Scope

- a-d is development/frozen confirmation, not an untouched holdout.
- e-h is descriptive transfer only.
- All seeds share one deterministic May 2025 CAISO archetype.
- The optimization-seed intervals describe training variability, not independent months or markets.

## Reproduction

The published model records were generated from source commit `81d50713b85e5f96809b37c16855289d13b1ad4d`. Recreate that source identity in a separate worktree; final reporting and thesis builders live on the later final branch tip.

```powershell
git worktree add ..\dc-energy-v5-training 81d50713b85e5f96809b37c16855289d13b1ad4d
Push-Location ..\dc-energy-v5-training
python scripts\run_offpolicy_campaign_v5.py --campaign td3bc_bconly_frozen_v3 --regions us global --seeds 301 302 303 304 305 --workers 4
python scripts\run_offpolicy_campaign_v5.py --campaign td3bc_postrl_frozen_v3 --regions us global --seeds 301 302 303 304 305 --workers 4
python scripts\build_offpolicy_evidence_v5.py --bc-campaign td3bc_bconly_frozen_v3 --postrl-campaign td3bc_postrl_frozen_v3 --seeds 301 302 303 304 305 --workers 4 --suffix v3
Pop-Location
# Back on the final branch tip:
python scripts\build_v5_results.py
```
