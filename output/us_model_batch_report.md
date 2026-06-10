# Evaluation Report: us_model_batch

## Summary

| Policy | Total Cost | Energy Cost | Peak Penalty | ND-weighted Load | Batch Expired | Deadline Cost | Avg Pool Size |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PPO | 9211062.38 | 7508991.68 | 1695816.25 | 1630290 | 3127.2280 | 6254.46 | 1.2397 |
| DQN | 8826085.37 | 7220302.62 | 1595598.77 | 1574442 | 5091.9898 | 10183.98 | 0.6215 |
| DQN-flatidx | 9057869.96 | 7402892.34 | 1647049.24 | 1601981 | 3964.1887 | 7928.38 | 1.2571 |
| Status Quo (local, no deferral) | 8873085.95 | 7284546.30 | 1578422.31 | 1573224 | 5058.6699 | 10117.34 | 0.5741 |
| Round Robin | 9289875.55 | 7576296.81 | 1707795.95 | 1636550 | 2891.3978 | 5782.80 | 1.3372 |
| Cheapest Price First | 8682423.70 | 7013524.82 | 1657654.14 | 1578385 | 5622.3676 | 11244.74 | 1.5362 |
| Avoid the Ramp | 8639330.44 | 7172335.08 | 1449439.39 | 1488550 | 6433.8056 | 12867.61 | 3.6756 |
| Local Only (No Routing) | 9110560.30 | 7451004.88 | 1651865.19 | 1609136 | 3845.1140 | 7690.23 | 0.8039 |
| Random | 9210890.15 | 7513319.64 | 1690884.13 | 1622873 | 3343.1810 | 6686.36 | 1.3881 |
| Drain Immediately | 8868115.48 | 7280894.14 | 1577120.35 | 1572744 | 5050.4925 | 10100.99 | 0.5732 |
| Defer to Low Net Demand | 9293161.02 | 7583738.87 | 1703775.17 | 1637060 | 2823.4918 | 5646.98 | 3.2707 |
| Trough-Slot Lookahead | 9197956.75 | 7551602.69 | 1639691.83 | 1598416 | 3331.1148 | 6662.23 | 1.4298 |

## Per-DC Energy Cost Breakdown

| Policy | US-West | US-Central | US-Southeast-1 | US-Southeast-2 |
| --- | --- | --- | --- | --- |
| PPO | 2321897.80 | 1530565.05 | 1900188.02 | 1756340.81 |
| DQN | 2037115.06 | 1430762.33 | 2153395.45 | 1599029.78 |
| DQN-flatidx | 2413044.39 | 1504544.26 | 1588179.88 | 1897123.81 |
| Status Quo (local, no deferral) | 2356726.54 | 1457308.58 | 1844972.73 | 1625538.45 |
| Round Robin | 2438136.19 | 1521308.99 | 1919083.40 | 1697768.23 |
| Cheapest Price First | 2018638.68 | 2003929.80 | 1581629.89 | 1409326.46 |
| Avoid the Ramp | 2531441.88 | 1250925.30 | 1605898.14 | 1784069.76 |
| Local Only (No Routing) | 2411379.61 | 1490421.37 | 1887651.84 | 1661552.06 |
| Random | 2414579.21 | 1509569.53 | 1904312.50 | 1684858.40 |
| Drain Immediately | 2343127.78 | 1461708.66 | 1843852.13 | 1632205.57 |
| Defer to Low Net Demand | 2441870.97 | 1521150.23 | 1918841.11 | 1701876.56 |
| Trough-Slot Lookahead | 2513692.85 | 1416580.16 | 1892156.19 | 1729173.49 |

## Plots

![Cumulative Cost](us_model_batch_cumulative_cost.png)

![Peak Contribution](us_model_batch_peak_contribution.png)

![Allocation Heatmap](us_model_batch_allocation_heatmap.png)

![Batch Pool](us_model_batch_batch_pool.png)

![Drain Heatmap](us_model_batch_drain_heatmap.png)
