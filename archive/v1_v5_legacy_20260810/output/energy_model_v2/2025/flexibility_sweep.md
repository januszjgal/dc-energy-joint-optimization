# Energy Model v2 Deadline-Flexibility Sensitivity

> No PPO training is involved. These are clairvoyant QP diagnostics.

ClusterData 2019 does not publish workload deadlines. The model uses fitted mean job duration and an experimental flexibility factor:

`H = ceil(mean_duration * (1 + flexibility_factor) / 300s)`

The primary experiment freezes `flexibility_factor = 1` (`H = 2x mean duration`). Factors 0 and 2 provide tight (`H = 1x`) and loose (`H = 3x`) robustness cases.

| Scenario | Factor | Horizon | Joint QP | Incremental temporal |
|---|---:|---:|---:|---:|
| US a-d | 0 | 1x mean duration | $6.069M | 0.38% |
| US a-d | 1 | 2x mean duration | $6.045M | 0.77% |
| US a-d | 2 | 3x mean duration | $6.024M | 1.12% |
| US e-h | 0 | 1x mean duration | $5.697M | 1.24% |
| US e-h | 1 | 2x mean duration | $5.637M | 2.27% |
| US e-h | 2 | 3x mean duration | $5.585M | 3.17% |
| Global a-d | 0 | 1x mean duration | $5.541M | 0.32% |
| Global a-d | 1 | 2x mean duration | $5.525M | 0.60% |
| Global a-d | 2 | 3x mean duration | $5.514M | 0.82% |
| Global e-h | 0 | 1x mean duration | $5.204M | 0.80% |
| Global e-h | 1 | 2x mean duration | $5.185M | 1.16% |
| Global e-h | 2 | 3x mean duration | $5.176M | 1.33% |
