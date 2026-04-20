# Evaluation Report: global_model

## Summary

| Policy | Total Cost | Avg Renewable % | Total Grid (MW-steps) |
| --- | --- | --- | --- |
| PPO | 4809312.63 | 46.9% | 1364260.33 |
| Round Robin | 5118260.19 | 47.2% | 1356780.28 |
| Cheapest Price First | 35815997.47 | 47.0% | 1244591.13 |
| Follow the Sun | 5434143.32 | 51.9% | 1234726.50 |
| Local Only (No Routing) | 5117101.43 | 47.1% | 1360057.61 |
| Random | 5147302.80 | 47.0% | 1361159.66 |
| Drain Immediately | 5118260.19 | 47.2% | 1356780.28 |
| Defer to Sun | 5118260.19 | 47.2% | 1356780.28 |
| GreenSlot | 4836801.73 | 53.1% | 1207041.42 |

## Per-DC Energy Cost Breakdown

| Policy | Global-US-West | Global-US-Central | Global-EU | Global-Asia |
| --- | --- | --- | --- | --- |
| PPO | 1005503.71 | 1017980.05 | 915691.99 | 1766163.92 |
| Round Robin | 1049723.90 | 743407.44 | 962837.04 | 2362291.81 |
| Cheapest Price First | 788660.58 | 1018235.79 | 710560.54 | 1790897.01 |
| Follow the Sun | 1006591.84 | 598831.11 | 879489.17 | 2262799.68 |
| Local Only (No Routing) | 1068691.74 | 740214.43 | 967009.02 | 2341186.24 |
| Random | 1052222.96 | 747324.81 | 972662.63 | 2374070.17 |
| Drain Immediately | 1049723.90 | 743407.44 | 962837.04 | 2362291.81 |
| Defer to Sun | 1049723.90 | 743407.44 | 962837.04 | 2362291.81 |
| GreenSlot | 951931.87 | 685435.23 | 992933.67 | 2179299.56 |

## Plots

![Cumulative Cost](global_model_cumulative_cost.png)

![Renewable Utilization](global_model_renewable.png)

![Allocation Heatmap](global_model_allocation_heatmap.png)
