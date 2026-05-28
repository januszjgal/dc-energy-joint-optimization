# Evaluation Report: us_model_batch

## Summary

| Policy | Total Cost | Energy Cost | Peak Penalty | ND-weighted Load | Batch Expired | Deadline Cost | Avg Pool Size |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PPO | 9473793.37 | 7615395.01 | 1855932.98 | 1698501 | 1232.6914 | 2465.38 | 0.7265 |
| DQN | 9654859.21 | 7767897.50 | 1885514.91 | 1713902 | 723.3997 | 1446.80 | 9.3684 |
| DQN-flatidx | 9832807.97 | 7931021.05 | 1901551.52 | 1724409 | 117.7051 | 235.41 | 0.0362 |
| Round Robin | 9784834.25 | 7905770.27 | 1878990.57 | 1722885 | 36.7074 | 73.41 | 0.4666 |
| Cheapest Price First | 21051100.07 | 7170953.35 | 1720171.88 | 1614448 | 2316.8850 | 4633.77 | 0.5452 |
| Avoid the Ramp | 10287186.25 | 7882657.55 | 1803173.80 | 1664045 | 979.5021 | 1959.00 | 1.7769 |
| Local Only (No Routing) | 9786782.26 | 7908300.92 | 1878336.86 | 1722633 | 72.2401 | 144.48 | 0.1782 |
| Random | 9737380.29 | 7859869.13 | 1876581.22 | 1711321 | 420.1795 | 840.36 | 0.5088 |
| Drain Immediately | 9769152.64 | 7896170.27 | 1872734.48 | 1720079 | 123.9446 | 247.89 | 0.0170 |
| Defer to Low Net Demand | 9788719.74 | 7907982.37 | 1880718.24 | 1723684 | 9.5693 | 19.14 | 1.6681 |
| Trough-Slot Lookahead | 9802326.09 | 7954188.84 | 1832266.69 | 1694752 | 48.1381 | 96.28 | 0.4797 |

## Per-DC Energy Cost Breakdown

| Policy | US-West | US-Central | US-Southeast-1 | US-Southeast-2 |
| --- | --- | --- | --- | --- |
| PPO | 2083377.30 | 1933460.70 | 1839558.55 | 1758998.47 |
| DQN | 2290820.46 | 1910801.45 | 2063374.70 | 1502900.89 |
| DQN-flatidx | 2427006.25 | 1724961.11 | 2276126.99 | 1502926.69 |
| Round Robin | 2427003.20 | 1733598.54 | 1956746.79 | 1788421.73 |
| Cheapest Price First | 2008527.15 | 2037622.46 | 1629016.62 | 1495787.12 |
| Avoid the Ramp | 2723276.69 | 1478212.58 | 1665594.19 | 2015574.08 |
| Local Only (No Routing) | 2445971.66 | 1723930.90 | 1958783.13 | 1779615.24 |
| Random | 2421747.55 | 1704509.78 | 1956028.76 | 1777583.05 |
| Drain Immediately | 2427006.34 | 1723966.95 | 1956746.90 | 1788450.08 |
| Defer to Low Net Demand | 2426931.58 | 1736154.62 | 1956602.89 | 1788293.28 |
| Trough-Slot Lookahead | 2561563.03 | 1628319.44 | 1932416.80 | 1831889.57 |

## Plots

![Cumulative Cost](us_model_batch_cumulative_cost.png)

![Peak Contribution](us_model_batch_peak_contribution.png)

![Allocation Heatmap](us_model_batch_allocation_heatmap.png)

![Batch Pool](us_model_batch_batch_pool.png)

![Drain Heatmap](us_model_batch_drain_heatmap.png)
