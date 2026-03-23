# Evaluation Report: us_model

## Summary

| Policy | Total Cost | Avg Renewable % | Total Grid (MW-steps) |
| --- | --- | --- | --- |
| PPO | 3931338.68 | 47.5% | 1346687.64 |
| Round Robin | 3930168.95 | 48.1% | 1331164.58 |
| Cheapest Price First | 35011042.72 | 47.7% | 1227692.90 |
| Follow the Sun | 9544312.60 | 48.8% | 1241468.11 |
| Local Only (No Routing) | 3940875.91 | 48.1% | 1332548.08 |
| Random | 3943000.40 | 48.0% | 1335133.43 |
| Drain Immediately | 3930168.95 | 48.1% | 1331164.58 |
| Defer to Sun | 3930168.95 | 48.1% | 1331164.58 |
| GreenSlot | 3909396.16 | 50.0% | 1286338.00 |

## Per-DC Energy Cost Breakdown

| Policy | US-West | US-Central | US-Southeast-1 | US-Southeast-2 |
| --- | --- | --- | --- | --- |
| PPO | 1249627.93 | 1100914.93 | 753844.76 | 825146.01 |
| Round Robin | 1216497.28 | 850457.36 | 985140.11 | 878074.20 |
| Cheapest Price First | 926294.79 | 1146558.24 | 754784.74 | 675761.38 |
| Follow the Sun | 1536059.38 | 665319.99 | 771782.45 | 817134.95 |
| Local Only (No Routing) | 1238523.26 | 846728.85 | 984062.34 | 871561.47 |
| Random | 1217687.46 | 853788.38 | 989021.81 | 881480.53 |
| Drain Immediately | 1216497.28 | 850457.36 | 985140.11 | 878074.20 |
| Defer to Sun | 1216497.28 | 850457.36 | 985140.11 | 878074.20 |
| GreenSlot | 1216499.83 | 800127.03 | 937768.97 | 858094.43 |

## Plots

![Cumulative Cost](us_model_cumulative_cost.png)

![Renewable Utilization](us_model_renewable.png)

![Allocation Heatmap](us_model_allocation_heatmap.png)
