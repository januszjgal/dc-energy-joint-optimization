# Evaluation Report: global_model_batch

## Summary

| Policy | Total Cost | Avg Renewable % | Batch Expired | Deadline Cost | Avg Pool Size |
| --- | --- | --- | --- | --- | --- |
| PPO | 8960172.34 | 16.5% | 2148.2304 | 4296.46 | 0.7129 |
| Round Robin | 10037275.78 | 15.9% | 36.7074 | 73.41 | 0.4666 |
| Cheapest Price First | 20770537.34 | 17.2% | 2316.8850 | 4633.77 | 0.7046 |
| Follow the Sun | 10307105.56 | 16.4% | 1177.5289 | 2355.06 | 0.9383 |
| Local Only (No Routing) | 10020029.89 | 15.9% | 72.2401 | 144.48 | 0.1782 |
| Random | 9971274.44 | 16.0% | 420.1795 | 840.36 | 0.5088 |
| Drain Immediately | 10027903.27 | 16.0% | 123.9446 | 247.89 | 0.0170 |
| Defer to Sun | 10035539.39 | 16.0% | 101.4973 | 202.99 | 5.0459 |

## Per-DC Energy Cost Breakdown

| Policy | Global-US-West | Global-US-Central | Global-EU | Global-Asia |
| --- | --- | --- | --- | --- |
| PPO | 1710896.06 | 1776666.09 | 1851489.97 | 3593774.59 |
| Round Robin | 1938985.50 | 1478969.24 | 2003577.88 | 4615669.75 |
| Cheapest Price First | 1520519.47 | 1782993.15 | 1599827.62 | 3707222.25 |
| Follow the Sun | 1949524.91 | 1347497.64 | 2024278.86 | 4740059.30 |
| Local Only (No Routing) | 1957953.95 | 1469301.60 | 2004957.38 | 4587672.49 |
| Random | 1933729.84 | 1449880.48 | 2002665.08 | 4584069.10 |
| Drain Immediately | 1938988.64 | 1469337.65 | 2003582.42 | 4615746.67 |
| Defer to Sun | 1939175.22 | 1472193.26 | 2004370.79 | 4619597.14 |

## Plots

![Cumulative Cost](global_model_batch_cumulative_cost.png)

![Renewable Utilization](global_model_batch_renewable.png)

![Allocation Heatmap](global_model_batch_allocation_heatmap.png)

![Batch Pool](global_model_batch_batch_pool.png)

![Drain Heatmap](global_model_batch_drain_heatmap.png)
