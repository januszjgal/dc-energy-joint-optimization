# Ramp-aware v6 pure-RL harness

This additive harness trains only randomly initialized PPO or SAC policies against the v6 environment's **semantic feasible action**. It does not import or modify frozen v2-v5 trainers, environments, models, or evidence.

## Integration API

Provide a factory as `module:function`. The callable receives `ramp_rl.contract.EnvRequest` and returns a Gymnasium environment. The environment must expose:

```python
def ramp_rl_contract() -> dict:
    return {
        "version": "ramp-v6-semantic-action-v1",
        "semantic_feasible_action": True,
        "raw_redundant_projected_logits": False,
        "history_hours": 3,
        "terminal_tail_hours": 3,
        "actual_terminal": True,
        "decision_steps": 576,
    }
```

`reset(options={"split": ..., "window_id": ...})` must return `episode_context` with the split, window identity, three-hour history/tail declarations, forecast model/vintage, source/data hashes, and a future-feature leakage assertion. `step()` must return the evidence fields listed in `ramp_rl.contract.REQUIRED_STEP_INFO`. The final decision must settle the tail, expose all tail intervals for 1h/3h distribution metrics, and return `terminated=True`, `truncated=False`, `tail_complete=True`, and `actual_terminal=True`. Training environments also implement `set_lagrangian_multiplier(value)`; the runner applies the frozen projected-dual-ascent rule after each completed training episode.

The action space must be a finite `gymnasium.spaces.Box` whose coordinates directly represent a feasible scheduling decision. Redundant logits subsequently projected by the environment are rejected.

## Commands

```powershell
python scripts\smoke_test_ramp_rl_v6.py
python scripts\run_ramp_rl_v6.py plan
python scripts\run_ramp_rl_v6.py train --algorithm ppo --fixture-profile --output models\ramp_rl_v6\ppo_fixture
python scripts\run_ramp_rl_v6.py evaluate --algorithm ppo --output models\ramp_rl_v6\ppo_fixture
```

For integration, replace `--env-factory ramp_rl.fixture_env:make_fixture_env` with the v6 factory. Each algorithm/seed/stage/epsilon must use an isolated output directory. Re-running with a larger `--timesteps` target resumes the checkpoint. All writes are rounded to complete vector-episode boundaries; SAC additionally verifies and restores replay, so 36-step returns never cross a restarted environment. Integrated jobs enforce the protocol's four vector environments.

The preregistered campaign is 3 seeds x 100k for PPO/SAC screening, validation-only promotion, 5 seeds x 500k confirmation, and extension toward 2M only while the frozen validation learning-curve rule remains material. The command intentionally does not orchestrate the final campaign while the v6 ramp environment and energy-model v3 are absent.

## Attribution and evidence

Every training manifest records protocol/source/artifact hashes, random initialization, initial/final policy and critic hashes, interactions and updates, replay provenance, split and normalization statistics, forecast identity, semantic adjustments, emergency feasibility, and explicit false assertions for all prohibited teacher/expert/optimizer inputs. Validation/test normalization is loaded frozen. Status quo and future analytic/QP oracle bounds are reachable only through the evaluation adapter.

The evidence helpers enforce validation-only selection and Lagrangian updates, epsilon sensitivities of 0/2/5%, day/month bootstrap intervals, optimizer-seed ranges, behavior audits, and named success-gate failures. A failed gate is retained as a valid scientific result and cannot authorize sealed-test tuning.
