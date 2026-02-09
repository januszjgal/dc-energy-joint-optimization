# Evaluation Report: global_model

## Summary

| Policy | Total Cost | Avg Renewable % | Total Grid (MW-steps) |
| --- | --- | --- | --- |
| PPO | 9753843.06 | 15.9% | 2160993.93 |
| Round Robin | 10140503.69 | 15.9% | 2160993.93 |
| Cheapest Price First | 39601875.12 | 17.4% | 1943267.85 |
| Follow the Sun | 11031203.29 | 16.0% | 2159559.91 |
| Local Only (No Routing) | 10123277.12 | 15.9% | 2160993.93 |
| Random | 10143932.66 | 15.9% | 2160993.93 |

## Per-DC Energy Cost Breakdown

| Policy | Global-US-West | Global-US-Central | Global-EU | Global-Asia |
| --- | --- | --- | --- | --- |
| PPO | 2125653.41 | 1466361.56 | 2095212.62 | 4066615.47 |
| Round Robin | 2084185.94 | 1341595.14 | 2072836.50 | 4641886.11 |
| Cheapest Price First | 1552912.67 | 1787858.60 | 1560348.50 | 3488447.36 |
| Follow the Sun | 2111416.26 | 1231164.39 | 2138433.83 | 5015442.59 |
| Local Only (No Routing) | 2107823.49 | 1334705.36 | 2074430.29 | 4606317.98 |
| Random | 2079240.02 | 1342477.83 | 2074447.07 | 4647247.96 |

## Plots

![Cumulative Cost](global_model_cumulative_cost.png)

![Renewable Utilization](global_model_renewable.png)

![Allocation Heatmap](global_model_allocation_heatmap.png)
