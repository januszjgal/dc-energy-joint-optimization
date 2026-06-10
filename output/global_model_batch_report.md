# Evaluation Report: global_model_batch

## Summary

| Policy | Total Cost | Energy Cost | Peak Penalty | ND-weighted Load | Batch Expired | Deadline Cost | Avg Pool Size |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PPO | 12941218.43 | 11115627.69 | 1819339.51 | 1743508 | 3125.6149 | 6251.23 | 1.2228 |
| DQN | 12880618.37 | 11221946.97 | 1647636.27 | 1657485 | 5517.5612 | 11035.12 | 9.5026 |
| DQN-flatidx | 12537850.41 | 10913606.14 | 1611764.29 | 1649262 | 6239.9881 | 12479.98 | 2.7064 |
| Status Quo (local, no deferral) | 12847995.28 | 11144801.06 | 1693076.88 | 1687778 | 5058.6699 | 10117.34 | 0.5741 |
| Round Robin | 13449496.12 | 11609366.95 | 1834346.37 | 1756916 | 2891.3978 | 5782.80 | 1.3372 |
| Cheapest Price First | 12152163.90 | 10390571.85 | 1750578.89 | 1681462 | 5506.5763 | 11013.15 | 1.7649 |
| Avoid the Ramp | 12355784.64 | 10742099.55 | 1601776.47 | 1616244 | 5856.2498 | 11712.50 | 4.5987 |
| Local Only (No Routing) | 13177901.51 | 11397917.18 | 1772294.09 | 1726511 | 3845.1140 | 7690.23 | 0.8039 |
| Random | 13337197.43 | 11513984.70 | 1816526.35 | 1742392 | 3343.1810 | 6686.36 | 1.3881 |
| Drain Immediately | 12859607.62 | 11156406.26 | 1693100.38 | 1687983 | 5050.4925 | 10100.99 | 0.5732 |
| Defer to Low Net Demand | 13569980.03 | 11707240.72 | 1857836.92 | 1770406 | 2451.1949 | 4902.39 | 4.2168 |
| Trough-Slot Lookahead | 13169206.59 | 11403911.89 | 1758549.53 | 1712191 | 3372.5828 | 6745.17 | 1.4462 |

## Per-DC Energy Cost Breakdown

| Policy | Global-US-West | Global-US-Central | Global-EU | Global-Asia |
| --- | --- | --- | --- | --- |
| PPO | 2545385.30 | 1623876.42 | 2383429.43 | 4562936.55 |
| DQN | 2891921.42 | 1260777.05 | 1959133.29 | 5110115.22 |
| DQN-flatidx | 2314066.87 | 1410681.34 | 2224722.33 | 4964135.61 |
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
