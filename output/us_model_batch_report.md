# Evaluation Report: us_model_batch

## Summary

| Policy | Total Cost | Energy Cost | Peak Penalty | ND-weighted Load | Batch Expired | Deadline Cost | Avg Pool Size |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PPO | 9975114.24 | 7554692.21 | 1719534.17 | 1641603 | 2803.5515 | 700887.87 | 1.4032 |
| DQN | 9997085.36 | 7140219.70 | 1592896.04 | 1579613 | 5055.8785 | 1263969.62 | 9.4509 |
| DQN-flatidx | 10090407.10 | 7229653.17 | 1560502.48 | 1563648 | 5201.0058 | 1300251.46 | 9.4671 |
| Status Quo (local, no deferral) | 10127636.09 | 7284546.30 | 1578422.31 | 1573224 | 5058.6699 | 1264667.49 | 0.5741 |
| Round Robin | 10006942.20 | 7576296.81 | 1707795.95 | 1636550 | 2891.3978 | 722849.44 | 1.3372 |
| Cheapest Price First | 10076770.87 | 7013524.82 | 1657654.14 | 1578385 | 5622.3676 | 1405591.90 | 1.5362 |
| Avoid the Ramp | 10234914.22 | 7172335.08 | 1449439.39 | 1488550 | 6433.8056 | 1608451.39 | 3.6756 |
| Local Only (No Routing) | 10064148.58 | 7451004.88 | 1651865.19 | 1609136 | 3845.1140 | 961278.51 | 0.8039 |
| Random | 10039999.03 | 7513319.64 | 1690884.13 | 1622873 | 3343.1810 | 835795.24 | 1.3881 |
| Drain Immediately | 10120637.62 | 7280894.14 | 1577120.35 | 1572744 | 5050.4925 | 1262623.13 | 0.5732 |
| Defer to Low Net Demand | 9993386.98 | 7583738.87 | 1703775.17 | 1637060 | 2823.4918 | 705872.94 | 3.2707 |
| Trough-Slot Lookahead | 10024073.22 | 7551602.69 | 1639691.83 | 1598416 | 3331.1148 | 832778.70 | 1.4298 |

## Per-DC Energy Cost Breakdown

| Policy | US-West | US-Central | US-Southeast-1 | US-Southeast-2 |
| --- | --- | --- | --- | --- |
| PPO | 2351838.36 | 1575406.37 | 1922009.73 | 1705437.75 |
| DQN | 1943829.29 | 1631025.28 | 1885345.11 | 1680020.02 |
| DQN-flatidx | 2350757.74 | 1459795.01 | 1539249.55 | 1879850.86 |
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
