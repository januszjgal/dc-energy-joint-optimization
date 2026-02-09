# Evaluation Report: us_model

## Summary

| Policy | Total Cost | Avg Renewable % | Total Grid (MW-steps) |
| --- | --- | --- | --- |
| PPO | 6426673.12 | 17.8% | 2111895.95 |
| Round Robin | 6477228.53 | 17.8% | 2111895.95 |
| Cheapest Price First | 36816919.82 | 19.5% | 1894169.87 |
| Follow the Sun | 8043751.28 | 17.0% | 2109106.94 |
| Local Only (No Routing) | 6485250.35 | 17.8% | 2111895.95 |
| Random | 6476565.44 | 17.8% | 2111895.95 |

## Per-DC Energy Cost Breakdown

| Policy | US-West | US-Central | US-Southeast-1 | US-Southeast-2 |
| --- | --- | --- | --- | --- |
| PPO | 1945919.77 | 1396511.25 | 1586240.06 | 1498002.03 |
| Round Robin | 2084185.94 | 1341595.14 | 1604941.26 | 1446506.19 |
| Cheapest Price First | 1552912.67 | 1787858.60 | 1188854.22 | 1074986.34 |
| Follow the Sun | 2722019.84 | 1104451.43 | 1391127.12 | 1473828.23 |
| Local Only (No Routing) | 2107823.49 | 1334705.36 | 1607402.17 | 1435319.33 |
| Random | 2079240.02 | 1342477.83 | 1606148.81 | 1448179.01 |

## Plots

![Cumulative Cost](us_model_cumulative_cost.png)

![Renewable Utilization](us_model_renewable.png)

![Allocation Heatmap](us_model_allocation_heatmap.png)
