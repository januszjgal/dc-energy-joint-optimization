# Ramp-aware v6 pure-RL harness

This additive harness trains only randomly initialized PPO or SAC policies against the integrated v6 ramp environment. It does not import or modify frozen v2-v5 trainers, environments, models, or evidence.

## Integration API

Provide a factory as `module:function`. The callable receives `ramp_rl.contract.EnvRequest` and returns a Gymnasium environment. The environment must expose:

```python
def ramp_rl_contract() -> dict:
    return {
        "version": "ramp-v6-semantic-action-v1",
        "semantic_feasible_action": True,
        "semantic_action_id":
            "ramp-v6-constraint-decoded-preferences-2n-plus-1-v1",
        "raw_redundant_projected_logits": False,
        "history_hours": 3,
        "terminal_tail_hours": 3,
        "terminal_tail_emitted_in_step_metrics": True,
        "interval_minutes": 60,
        "actual_terminal": True,
        "decision_steps": 9,
        "active_arrival_steps": 6,
        "action_shape": [13],
        "action_low": [-6.0] * 13,
        "action_high": [6.0] * 13,
    }
```

`reset(options={"split": ..., "window_id": ...})` returns `episode_context` with the split/window, real three-hour warm history, three-hour no-arrival tail, forecast model/vintage, source hashes, and a future-feature leakage assertion. The hourly fixture has six arrival decisions plus three explicitly scored tail decisions. `step()` returns current per-market ramp/cost/safety arrays and macro fields. The terminal transition repeats the three tail hours in dedicated ramp/cost/safety arrays for evidence sealing and returns `terminated=True`, `truncated=False`, `tail_complete=True`, and `actual_terminal=True`; evaluators do not double count tail values already emitted stepwise.

For `N` sites the frozen action is the actual ramp-core `2N+1` bounded preference vector: `N` service-allocation scores, one optional batch-execution score, and `N` batch-destination scores, each in `[-6,6]`. These named semantic coordinates feed the frozen constraint-only capped-simplex/EDF/transport decoder. They are not replaced by the trainer's one-dimensional fixture action or reinterpreted as an unconstrained primitive dispatch vector.

## Commands

```powershell
python scripts\smoke_test_ramp_rl_v6.py
python scripts\run_ramp_rl_v6.py plan
python scripts\run_ramp_rl_v6.py train --algorithm ppo --fixture-profile --output models\ramp_rl_v6\ppo_ramp_core
python scripts\run_ramp_rl_v6.py evaluate --algorithm ppo --output models\ramp_rl_v6\ppo_ramp_core --window ramp-core-m-07-sealed-0000
```

The default factory is `env.ramp_v6.factory:make_fixture_env`, which creates the actual deterministic six-market, 1 GW ramp-core fixture with a 13-dimensional action. Each algorithm/seed/stage/epsilon uses an isolated output directory. Re-running with a larger target resumes the checkpoint. All writes are rounded to complete vector-episode boundaries; SAC verifies and restores replay. Its frozen 36-step return horizon exceeds the real hourly 3h delayed-ramp horizon. Non-fixture campaign jobs enforce four vector environments.

The preregistered campaign is 3 seeds x 100k for PPO/SAC screening, validation-only promotion, 5 seeds x 500k confirmation, and extension toward 2M only while the frozen validation learning-curve rule remains material. The v6 environment interface is integrated. The final campaign is blocked only until the real energy-model v3 ramp panel handoff below exists.

## Energy-model v3 handoff

`env.ramp_v6.factory:make_energy_model_v3_env` expects
`output/energy_model_v3/ramp_v6/factory_manifest.json` with:

```json
{
  "schema_version": "energy-model-v3-ramp-v6-factory-v1",
  "forecast_model": "stable model identifier",
  "frozen_stats_sha256": "64 hex characters shared by every window",
  "windows": {
    "train": {
      "window-id": {
        "artifact_root": "output/energy_model_v3/ramp_v6/windows/window-id",
        "month": 1,
        "day": "YYYY-MM-DD",
        "forecast_vintage": "stable vintage identifier",
        "source_hashes": {
          "canonical_panel": "64 hex characters",
          "fixture": "64 hex characters"
        }
      }
    },
    "validation": {},
    "test": {}
  }
}
```

Every artifact root contains `canonical_panel.csv`, validated by the single
`env/protocols/v6_ramp_panel.schema.json` contract, and `fixture.json` with six
sites plus aligned hourly workload and train-only frozen statistics. The panel
must include causal h1/h2/h3 gross/net forecast endpoints, issue time, vintage,
quality, physical tuple, market scale, and day-ahead LMP. All three split maps
must contain complete chronological windows. File hashes and the manifest-level frozen-statistics hash are verified before
an environment is created. The current independent energy-model v3 builder
does not yet publish this ramp handoff, so the absence of this manifest is the
sole long-campaign blocker.

The real factory verifies declared months and actual panel timestamps against
the frozen split and verifies that ramp statistics were fit only on training
months. Seeded train environments resample windows on reset; validation and
test remain fixed. Resume identity includes the complete integrated source
bundle, so a checkpoint cannot combine interactions from different environment,
reward, runner, or evidence implementations.

## Attribution and evidence

Every training manifest records protocol/source/artifact hashes, random initialization, initial/final policy and critic hashes, interactions and updates, replay provenance, split and normalization statistics, forecast identity, semantic adjustments, emergency feasibility, and explicit false assertions for all prohibited teacher/expert/optimizer inputs. Validation/test normalization is loaded frozen. Status quo and future analytic/QP oracle bounds are reachable only through the evaluation adapter.

The evidence helpers enforce validation-only selection and Lagrangian updates, epsilon sensitivities of 0/2/5%, day/month bootstrap intervals, optimizer-seed ranges, behavior audits, and named success-gate failures. A failed gate is retained as a valid scientific result and cannot authorize sealed-test tuning.
