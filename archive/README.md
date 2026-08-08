# Archive

Items here are superseded or no-longer-used. Kept for provenance, not for active use.

| Item | Replaced by | Why |
|---|---|---|
| `colab_extract_data.ipynb` | `../extract_clusterdata2019_full.ipynb` | The full extractor (20 cells, 6 datasets) supersedes the limited one (10 cells, 2 datasets). |
| `colab_full_output.md` | Output of `extract_clusterdata2019_full.ipynb` saved to `../data/` | Paste of a prior run's stdout, not a source artifact. |
| `extracted_data.zip` | Latest `../data/` directory | Zip of an earlier extraction run. The current `../data/` reflects the live state. |
| `models_pre_reframe/` | `../models/` (post-reframe) | Models trained under the on-site-solar formulation, before the env was reframed to grid demand smoothing (2026-05-27). Observation spaces differ — these models can't run in the current env. |
| `logs_legacy_naming/` | New sweeps under `../logs/` | Historical sweep logs that called the valid spatial-only mode “legacy.” Active sweeps now use the canonical US/Global × spatial-only/spatial+temporal terminology. |
| `demand_charge_invalid_penalty_20260805/` | Final guarded/dense-shaped demand-charge models | Invalidated iterations: pre-calibration runs, pre-terminal-accounting batch runs, and sparse-terminal-reward batch runs. Retained only for provenance; none may be used for results. |
| `dqn_cfws_20260805/` | Planned PPO-only v2 OOF campaign | Historical DQN and CFWS-inspired trainers, wrappers, orchestrators, models, and logs. Removed from active workflows to eliminate algorithm-horse-race and accidental rerun paths. |
| `oof_invalid_source_change_20260805/` | Final frozen OOF campaign | Partial first-wave artifacts invalidated when active source changed after protocol freeze. No model was accepted or published by the campaign. |
| `energy_model_v1_mixed_20260805/` | CAISO-archetype energy model v2 | Complete mixed/synthetic energy-model scenarios, preprocessing/analysis code, campaigns, inputs, models, logs, results, and manuscript snapshot. Retired after timestamp, DST, provenance, and price/net-demand phase audit. |
| `energy_model_v2_invalid_preflight_20260805/` | Regenerated v2 QP gate after system fixes | Provisional baseline/QP JSON invalidated when corrected Status Quo locality exposed inconsistent held-out-cell capacity units. Must not be cited. |
| `thesis_pre_v5_20260807/` | `../thesis_paper.md`, `../thesis_paper.docx`, and `../scripts/build_final_thesis.py` | Pre-v5 overview, DOCX, and generator. They describe the frozen v2-v4 state and are retained only for historical comparison. |
| `training_logs_pre_v5_20260807/` | Canonical JSON and reports under `../output/oof_v2_2025/`, `../output/ppo_v3_reward_sweep/`, and `../output/ppo_v4_safety/` | Raw historical console logs are not imported by the runtime and are unnecessary for the final controller. Canonical protocols, results, reports, figures, and models remain active. |

Nothing here is needed to run, evaluate, or explain the final v5 controller.
The active extraction pipeline is:

```
extract_clusterdata2019_full.ipynb  (Colab/BigQuery)  → data/cells, data/machines, data/jobs/jobs_*, data/power_model_*, data/workload_generator_params.json
scripts/colab_extract_batch_jobs.py  (Colab/BigQuery) → data/jobs/batch_raw_*
scripts/refit_distributions.py       (local)          → data/jobs/batch_distributions_*
scripts/build_energy_model_v2.py     (local/CAISO)    → data/energy_model_v2/2025
scripts/review_energy_model_v2.py    (no training)    → output/energy_model_v2/2025
```

The final active policy path is:

```text
data + env/scenarios
  -> env/residual_safe_offpolicy_env.py
  -> scripts/run_offpolicy_campaign_v5.py
  -> models/offpolicy_v5_continuous/frozen_confirmation/
  -> output/offpolicy_v5_continuous/canonical_v5_v3/
```
