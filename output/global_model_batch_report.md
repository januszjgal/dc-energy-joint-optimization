# Evaluation Report: global_model_batch

## Summary

| Policy | Total Cost | Energy Cost | Peak Penalty | ND-weighted Load | Batch Expired | Deadline Cost | Avg Pool Size |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PPO | 13953068.84 | 11770363.14 | 1988938.79 | 1820705 | 775.0676 | 193766.91 | 3.9389 |
| DQN | 13769298.73 | 11574756.62 | 2006783.96 | 1820180 | 751.0326 | 187758.15 | 3.6277 |
| DQN-flatidx | 13992674.28 | 11032305.65 | 1700919.39 | 1689169 | 5037.7970 | 1259449.24 | 9.7262 |
| Status Quo (local, no deferral) | 14208298.51 | 12018763.35 | 2005037.51 | 1825353 | 737.9906 | 184497.65 | 3.5225 |
| Round Robin | 14215889.75 | 12031326.45 | 1990337.63 | 1824060 | 776.9027 | 194225.67 | 3.9419 |
| Cheapest Price First | 13728488.72 | 11399640.41 | 2023324.77 | 1817416 | 1222.0942 | 305523.54 | 5.9046 |
| Avoid the Ramp | 13879540.13 | 11207767.06 | 1742077.71 | 1688447 | 3717.9970 | 929499.25 | 8.3994 |
| Local Only (No Routing) | 14206742.52 | 12020894.56 | 1998986.69 | 1825049 | 747.4451 | 186861.27 | 3.6601 |
| Random | 14224729.88 | 12019706.37 | 1998076.92 | 1822417 | 827.7863 | 206946.58 | 4.1171 |
| Drain Immediately | 14216552.28 | 12027411.55 | 2004652.48 | 1825288 | 737.9530 | 184488.24 | 3.5204 |
| Defer to Low Net Demand | 14191139.92 | 11851971.52 | 1913040.27 | 1793744 | 1704.5125 | 426128.13 | 5.7621 |
| Trough-Slot Lookahead | 14076037.62 | 11911217.20 | 1940360.30 | 1793367 | 897.8405 | 224460.12 | 4.2739 |

## Per-DC Energy Cost Breakdown

| Policy | Global-US-West | Global-US-Central | Global-EU | Global-Asia |
| --- | --- | --- | --- | --- |
| PPO | 2589566.53 | 1646967.55 | 2493759.20 | 5040069.87 |
| DQN | 2628726.64 | 1756976.56 | 2401246.55 | 4787806.86 |
| DQN-flatidx | 2352931.70 | 1682929.41 | 1919522.40 | 5076922.15 |
| Status Quo (local, no deferral) | 2533529.24 | 1586976.21 | 2475904.36 | 5422353.54 |
| Round Robin | 2524489.92 | 1585539.45 | 2467444.89 | 5453852.19 |
| Cheapest Price First | 2264723.08 | 2026032.73 | 2199358.74 | 4909525.85 |
| Avoid the Ramp | 2620102.25 | 1458177.68 | 2379553.18 | 4749933.95 |
| Local Only (No Routing) | 2535860.62 | 1584939.36 | 2473033.17 | 5427061.41 |
| Random | 2521399.84 | 1584727.25 | 2464344.42 | 5449234.86 |
| Drain Immediately | 2522557.89 | 1590113.81 | 2475078.71 | 5439661.13 |
| Defer to Low Net Demand | 2487276.17 | 1554121.85 | 2419253.99 | 5391319.51 |
| Trough-Slot Lookahead | 2664090.23 | 1547554.42 | 2450555.15 | 5249017.40 |

## Plots

![Cumulative Cost](global_model_batch_cumulative_cost.png)

![Peak Contribution](global_model_batch_peak_contribution.png)

![Allocation Heatmap](global_model_batch_allocation_heatmap.png)

![Batch Pool](global_model_batch_batch_pool.png)

![Drain Heatmap](global_model_batch_drain_heatmap.png)
