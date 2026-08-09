# Six-market pure-RL campaign closeout

**Result:** `validation_blocker_no_selected_protocol`. PPO remained safe and under the energy budget at 500k, but three of five seeds failed preregistered validation behavior or per-market ramp gates. No protocol qualified for sealed-test access.

## Validation result

| Stage | Algorithm | Seeds | Mean ramp impact | Mean cost ratio | Exact safety | All gates |
|---|---|---:|---:|---:|---|---|
| screen | PPO | 3 | -1.33492978072e-05 | 0.9801356986 | True | True |
| confirmation | PPO | 5 | -1.09572077294e-05 | 0.9783949019 | True | False |

The 500k validation curve changed by **-17.9192%** against a required **1.0%** improvement, so extension toward 2M was rejected.

## Confirmation seeds

| Seed | Ramp impact | Cost ratio | Safety | Success | Failed gates |
|---:|---:|---:|---|---|---|
| 2601 | -1.47394353695e-05 | 0.9919377629 | True | True | - |
| 2602 | -1.58831491944e-05 | 0.9858038052 | True | False | every_market_ramp_improves |
| 2603 | -1.57353810215e-05 | 0.9811123037 | True | True | - |
| 2604 | -4.00807153722e-06 | 0.9742610172 | True | False | behavior_lower_ramp_power |
| 2605 | -4.42000152464e-06 | 0.9588596206 | True | False | every_market_ramp_improves |

## Per-market confirmation ramp

| Market | Five-seed mean impact | All seeds improve | Failed seeds |
|---|---:|---|---|
| CAISO_NP15 | -1.47322963822e-05 | True | - |
| ERCOT_LZ_NORTH | -2.94548234575e-06 | True | - |
| ISONE_NEMA | -1.62251308951e-05 | True | - |
| MISO_MINN_HUB | -7.82329930407e-07 | False | 2602 |
| NYISO_NYC_J | -2.91028819036e-05 | True | - |
| SPP_NORTH_HUB | -1.95512491959e-06 | False | 2605 |

Cost, exact safety, and behavior are reported at the six-market episode aggregate because the frozen evaluator did not persist per-market decompositions for those fields. No values were inferred or synthesized.

## Sealed test and robustness

- Mar-Apr 2026 sealed test opened: **false**.
- 1 GW scaling robustness run: **false**.
- Overlapping c-h robustness run: **false**.

Both robustness studies are post-selection analyses. Running them without a qualifying base protocol would violate the frozen campaign order.

## Reproduction commands

```powershell
python scripts\build_energy_v3_ramp_factory.py
python scripts\run_ramp_rl_v6.py train --env-factory env.ramp_v6.factory:make_energy_model_v3_env --algorithm <ppo|sac> --seed <2601|2602|2603> --timesteps 100000 --n-envs 4 --output models\ramp_rl_v6\live\screen\<algorithm>\<seed>
python scripts\run_ramp_rl_v6.py train --env-factory env.ramp_v6.factory:make_energy_model_v3_env --algorithm ppo --seed <2601|2602|2603|2604|2605> --timesteps 500000 --n-envs 4 --output models\ramp_rl_v6\live\confirmation\ppo\<seed>
$m = Get-Content output\energy_model_v3\ramp_v6\factory_manifest.json -Raw | ConvertFrom-Json; $a = @('scripts\run_ramp_rl_v6.py', 'evaluate', '--env-factory', 'env.ramp_v6.factory:make_energy_model_v3_env', '--algorithm', 'ppo', '--seed', '<seed>', '--output', 'models\ramp_rl_v6\live\confirmation\ppo\<seed>', '--split', 'validation'); foreach ($w in ($m.windows.validation.PSObject.Properties.Name | Sort-Object)) { $a += '--window'; $a += $w }; & python @a
python scripts\summarize_ramp_rl_stage.py --validation-root output\ramp_rl_v6\live\confirmation --expected-seeds 5 --stage confirmation --previous-summary output\ramp_rl_v6\live\screen_summary.json --output output\ramp_rl_v6\live\confirmation_summary.json
python scripts\build_ramp_rl_evidence_v6.py models\ramp_rl_v6\live\confirmation\ppo\2601 models\ramp_rl_v6\live\confirmation\ppo\2602 models\ramp_rl_v6\live\confirmation\ppo\2603 models\ramp_rl_v6\live\confirmation\ppo\2604 models\ramp_rl_v6\live\confirmation\ppo\2605 --output output\ramp_rl_v6\live\confirmation_evidence_index.json
```

## Integrity

- Live panel manifest SHA-256: `489cb39c61c19952fe90b213c1ea20e4f5eb904563425ec8b751a15ceed446df`
- Raw acquisition manifest SHA-256: `64fd78254dabefa8d525f3e43144b2a50bc12c3050bad21efcc8836e3d11b7e7`
- Factory manifest SHA-256: `b2c88f8831eec0e2d1321e779eb9e823d96cd165fd3fd6d2ca3ba34ed1497e8c`
- Screen summary SHA-256: `8865412a59b267f738437c0a19d2c7bc0c7c5e4dd6b2b8dd7cdfc5cf1e4f76ee`
- Confirmation summary SHA-256: `5644b9af598dcb28b82384f421af07015172109e839938d3bc0832673921af72`
- Confirmation evidence index SHA-256: `7a09f5849f70132f3d92b2865854345b73d44e378c61ded1baabaf8c2c18eb20`

All selection and stopping decisions used September-January training and February validation only. The March-April test split remains sealed.
