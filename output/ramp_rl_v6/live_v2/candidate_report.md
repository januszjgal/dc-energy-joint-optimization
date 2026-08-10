# Ramp-RL v2-A validation report

Protocol `v6-ramp-pure-rl-preregistered-v2` (`d7dab14572eb3c2465f7cba3a863d532c4cca97bbc81b8768529b69056acda5d`) was trained from random initialization for a nominal 100k steps (110,592 complete-boundary interactions) on seeds 2701-2703.

| Seed | Ramp impact | Cost ratio | Strict gate | Failed gates |
|---:|---:|---:|---|---|
| 2701 | 5.38879285551e-07 | 1.0371496846 | False | mean_ramp_improves, every_market_ramp_improves, primary_energy_budget, behavior_lower_ramp_power |
| 2702 | -2.39548983666e-07 | 0.9944911462 | False | every_market_ramp_improves, behavior_lower_ramp_power |
| 2703 | -1.79444564729e-06 | 0.9959583182 | False | behavior_lower_ramp_power |

Three-seed mean ramp impact was `-4.98371781802e-07` and mean cost ratio was `1.0091997163`. Ramp-improvement magnitude changed by `-96.2667%` versus the frozen v1 100k PPO screen.

All three seeds failed at least one strict validation gate. Candidate A therefore stops at validation; seeds 2704-2705 were not launched and the sealed March-April test remains unopened.
