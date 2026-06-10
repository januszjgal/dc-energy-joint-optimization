# Evaluation Report: global_model_batch

## Summary

| Policy | Total Cost | Energy Cost | Peak Penalty | ND-weighted Load | Batch Expired | Deadline Cost | Avg Pool Size |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PPO | 13825816.15 | 11264454.77 | 1834245.28 | 1753045 | 2908.4644 | 727116.10 | 1.3266 |
| DQN | 14373835.74 | 11580538.59 | 1851352.84 | 1748775 | 3767.7773 | 941944.31 | 1.4355 |
| DQN-flatidx | 14493037.00 | 11488471.20 | 1722647.09 | 1697325 | 5127.6748 | 1281918.71 | 0.6255 |
| Status Quo (local, no deferral) | 14102545.43 | 11144801.06 | 1693076.88 | 1687778 | 5058.6699 | 1264667.49 | 0.5741 |
| Round Robin | 14166562.77 | 11609366.95 | 1834346.37 | 1756916 | 2891.3978 | 722849.44 | 1.3372 |
| Cheapest Price First | 13517794.82 | 10390571.85 | 1750578.89 | 1681462 | 5506.5763 | 1376644.08 | 1.7649 |
| Avoid the Ramp | 13808134.59 | 10742099.55 | 1601776.47 | 1616244 | 5856.2498 | 1464062.45 | 4.5987 |
| Local Only (No Routing) | 14131489.78 | 11397917.18 | 1772294.09 | 1726511 | 3845.1140 | 961278.51 | 0.8039 |
| Random | 14166306.31 | 11513984.70 | 1816526.35 | 1742392 | 3343.1810 | 835795.24 | 1.3881 |
| Drain Immediately | 14112129.76 | 11156406.26 | 1693100.38 | 1687983 | 5050.4925 | 1262623.13 | 0.5732 |
| Defer to Low Net Demand | 14177876.36 | 11707240.72 | 1857836.92 | 1770406 | 2451.1949 | 612798.72 | 4.2168 |
| Trough-Slot Lookahead | 14005607.12 | 11403911.89 | 1758549.53 | 1712191 | 3372.5828 | 843145.69 | 1.4462 |

## Per-DC Energy Cost Breakdown

| Policy | Global-US-West | Global-US-Central | Global-EU | Global-Asia |
| --- | --- | --- | --- | --- |
| PPO | 2489265.37 | 1619778.34 | 2391777.98 | 4763633.08 |
| DQN | 1729248.70 | 1638447.65 | 2547183.85 | 5665658.39 |
| DQN-flatidx | 2008719.64 | 1468286.25 | 2281642.78 | 5729822.54 |
| Status Quo (local, no deferral) | 2356726.54 | 1457308.58 | 2271805.37 | 5058960.57 |
| Round Robin | 2438136.19 | 1521308.99 | 2364840.16 | 5285081.62 |
| Cheapest Price First | 2026258.01 | 2002893.45 | 1956022.16 | 4405398.23 |
| Avoid the Ramp | 2556344.67 | 1396476.32 | 2283421.20 | 4505857.35 |
| Local Only (No Routing) | 2411379.61 | 1490421.37 | 2324762.00 | 5171354.20 |
| Random | 2414579.21 | 1509569.53 | 2346647.17 | 5243188.78 |
| Drain Immediately | 2343127.78 | 1461708.66 | 2271313.87 | 5080255.94 |
| Defer to Low Net Demand | 2458072.14 | 1531458.56 | 2382460.31 | 5335249.71 |
| Trough-Slot Lookahead | 2581564.47 | 1468767.58 | 2330525.77 | 5023054.08 |

## Plots

![Cumulative Cost](global_model_batch_cumulative_cost.png)

![Peak Contribution](global_model_batch_peak_contribution.png)

![Allocation Heatmap](global_model_batch_allocation_heatmap.png)

![Batch Pool](global_model_batch_batch_pool.png)

![Drain Heatmap](global_model_batch_drain_heatmap.png)
