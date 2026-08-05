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

Nothing here is needed to reproduce the current results. The active extraction pipeline is:

```
extract_clusterdata2019_full.ipynb  (Colab/BigQuery)  → data/cells, data/machines, data/jobs/jobs_*, data/power_model_*, data/workload_generator_params.json
scripts/colab_extract_batch_jobs.py  (Colab/BigQuery) → data/jobs/batch_raw_*
scripts/refit_distributions.py       (local)          → data/jobs/batch_distributions_*
```
