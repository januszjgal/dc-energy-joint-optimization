# Evaluation Report: us_model_batch

## Summary

| Policy | Total Cost | Avg Renewable % | Batch Expired | Deadline Cost | Avg Pool Size |
| --- | --- | --- | --- | --- | --- |
| PPO | 3853319.79 | 47.8% | 631.9145 | 1263.83 | 0.5693 |
| Round Robin | 3904639.42 | 48.0% | 36.7074 | 73.41 | 0.4666 |
| Cheapest Price First | 15705659.31 | 47.7% | 2316.8850 | 4633.77 | 0.5452 |
| Follow the Sun | 4221514.05 | 48.9% | 821.9289 | 1643.86 | 0.8637 |
| Local Only (No Routing) | 3910488.42 | 48.0% | 72.2401 | 144.48 | 0.1782 |
| Random | 3885289.97 | 48.0% | 420.1795 | 840.36 | 0.5088 |
| Drain Immediately | 3898683.88 | 48.0% | 123.9446 | 247.89 | 0.0170 |
| Defer to Sun | 3894280.81 | 48.1% | 101.4938 | 202.99 | 4.9956 |
| GreenSlot | 3778034.37 | 49.5% | 323.4058 | 646.81 | 1.5606 |

## Per-DC Energy Cost Breakdown

| Policy | US-West | US-Central | US-Southeast-1 | US-Southeast-2 |
| --- | --- | --- | --- | --- |
| PPO | 1072267.31 | 1033431.40 | 911773.97 | 834583.29 |
| Round Robin | 1135910.72 | 937671.39 | 955475.07 | 875508.83 |
| Cheapest Price First | 909270.52 | 1146558.24 | 774336.65 | 715519.06 |
| Follow the Sun | 1506013.62 | 778532.28 | 776121.44 | 790213.97 |
| Local Only (No Routing) | 1153289.20 | 932043.36 | 954594.69 | 870416.69 |
| Random | 1136326.58 | 919157.64 | 956642.20 | 872233.62 |
| Drain Immediately | 1135915.18 | 931372.77 | 955539.12 | 875608.92 |
| Defer to Sun | 1135310.50 | 930839.64 | 954158.69 | 873769.00 |
| GreenSlot | 1100977.26 | 882510.25 | 916608.44 | 867674.66 |

## Plots

![Cumulative Cost](us_model_batch_cumulative_cost.png)

![Renewable Utilization](us_model_batch_renewable.png)

![Allocation Heatmap](us_model_batch_allocation_heatmap.png)

![Batch Pool](us_model_batch_batch_pool.png)

![Drain Heatmap](us_model_batch_drain_heatmap.png)
