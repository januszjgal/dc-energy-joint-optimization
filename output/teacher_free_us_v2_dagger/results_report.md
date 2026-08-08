# Teacher-free US iterative DAgger

- Protocol: `teacher-free-us-v2-dagger`
- Cached failed v1 result persisted at `output/teacher_free_bc_ppo/failed_resume_us_result.json`
- DAgger iterations executed: **4**

## Final US a-d gate

- Mean savings: **6.9784%**
- Safe runs: **3/3**
- Mean intervention: **0.020871**
- Max semantic delta: **3.041e-07**

| Iteration | Rows | Mean savings | Mean intervention | Intervention-state rate | Ranking-error rate | Best seed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 17856 | 1.2974% | 0.100060 | 0.521001 | 0.420755 | 3102 |
| 1 | 26784 | 6.8913% | 0.063321 | 0.377651 | 0.569930 | 3103 |
| 2 | 35712 | 6.9181% | 0.026247 | 0.296343 | 0.487987 | 3103 |
| 3 | 44640 | 6.9617% | 0.022513 | 0.242092 | 0.431564 | 3101 |
| 4 | 53568 | 6.9784% | 0.020871 | 0.205738 | 0.390326 | 3103 |

- PPO attempted: **False**
- Global preserved without retraining: **True**
