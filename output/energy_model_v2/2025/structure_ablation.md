# Energy Model v2 Structural Attribution

> No PPO training is involved. Ablations use a-d calibration cells only.

The ablations are diagnostic, not additive causal decompositions; interactions remain between price phase, power heterogeneity, and Φ.

| Scenario | Condition | Status Quo | Spatial QP | Headroom |
|---|---|---:|---:|---:|
| US a-d | primary | $6.568M | $6.092M | 7.24% |
| US a-d | energy_only | $5.555M | $5.138M | 7.52% |
| US a-d | pooled_power | $6.563M | $6.291M | 4.14% |
| US a-d | synchronous_market | $6.564M | $6.192M | 5.67% |
| US a-d | shifted_market_only | $5.553M | $5.311M | 4.36% |
| US a-d | power_heterogeneity_only | $5.553M | $5.202M | 6.33% |
| US a-d | degenerate_control | $5.551M | $5.549M | 0.05% |
| Global a-d | primary | $6.598M | $5.559M | 15.75% |
| Global a-d | energy_only | $5.577M | $4.809M | 13.77% |
| Global a-d | pooled_power | $6.590M | $5.625M | 14.64% |
| Global a-d | synchronous_market | $6.564M | $6.192M | 5.67% |
| Global a-d | shifted_market_only | $5.572M | $4.871M | 12.58% |
| Global a-d | power_heterogeneity_only | $5.553M | $5.202M | 6.33% |
| Global a-d | degenerate_control | $5.551M | $5.549M | 0.05% |

## Rated-power sensitivity (primary objective, fixed alpha)

| Scenario | R | Status Quo | Spatial QP | Headroom |
|---|---:|---:|---:|---:|
| US a-d | 50 MW | $3.031M | $2.808M | 7.34% |
| US a-d | 100 MW | $6.568M | $6.092M | 7.24% |
| US a-d | 200 MW | $15.161M | $14.071M | 7.18% |
| Global a-d | 50 MW | $3.044M | $2.593M | 14.81% |
| Global a-d | 100 MW | $6.598M | $5.559M | 15.75% |
| Global a-d | 200 MW | $15.241M | $12.597M | 17.35% |

![Structural ablation](structure_ablation.png)
