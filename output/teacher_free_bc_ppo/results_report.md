# Teacher-free native BC/PPO report

- Protocol: `teacher-free-native-bc-ppo-v1`
- Parent protocol: `ppo-hard-safety-v4`

| Region | Controller | Cells | Mean savings | Safe runs | Mean intervention | Max semantic delta |
|---|---|---|---:|---:|---:|---:|
| US | Teacher | a-d | 7.1388% | 1/1 | 0.000000 | 1.000e-06 |
| US | Teacher | e-h | 7.1155% | 1/1 | 0.000000 | 1.000e-06 |
| US | BC | a-d | 1.1540% | 3/3 | 0.906362 | 7.787e-01 |
| US | BC | e-h | -0.1987% | 3/3 | 0.962515 | 9.970e-01 |
| GLOBAL | Teacher | a-d | 15.2973% | 1/1 | 0.000000 | 1.000e-06 |
| GLOBAL | Teacher | e-h | 16.0473% | 1/1 | 0.000000 | 1.000e-06 |
| GLOBAL | BC | a-d | 13.0716% | 3/3 | 0.948551 | 6.870e-01 |
| GLOBAL | BC | e-h | 10.3653% | 3/3 | 0.969347 | 9.068e-01 |

## Dataset

- **US** 8928 rows, seeds [1101], teacher roundtrip max 4.434e-08.
- **US teacher reference** a-d 7.1388% (native roundtrip 7.2177%), e-h 7.1155%.
- **GLOBAL** 8928 rows, seeds [1101], teacher roundtrip max 4.607e-08.
- **GLOBAL teacher reference** a-d 15.2973% (native roundtrip 15.7393%), e-h 16.0473%.
