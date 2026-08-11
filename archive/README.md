# Archive

Items here are historical, invalidated, superseded, or noncanonical. They are
retained for provenance and are not imported by the final V6/V4R runtime.

## Current archive bundles

| Item | Replaced by | Why |
|---|---|---|
| `v1_v5_legacy_20260810/` | `../energy_model_v3/`, `../env/ramp_v6/`, `../ramp_rl/`, `../models/ramp_rl_v6/`, and `../output/ramp_rl_v6/` | Preserves the complete V1-V5 code, protocols, models, CAISO-only data, campaigns, result packages, and supporting slide notes after the repository was narrowed to the accepted six-market V6/V4R solution. Paths inside the bundle mirror their former repository locations. |
| `clusterdata2019_stale_extraction_20260810/` | `../data/cells/`, `../data/jobs/batch_distributions_*.json`, the root extraction notebooks, and `../scripts/refit_freebeb_local.py` | Stale raw samples, truncated schema examples, synthetic-generator parameters, superseded extraction/refitting utilities, redundant combined-machine data, and reproducible calibration diagnostics. None are final runtime inputs. |
| `v6_superseded_artifacts_20260810/` | `../output/ramp_rl_v6/recovered_v4r_resealed_v2/`, `../output/ramp_rl_v6/recovered_v4r_posthoc_metrics_v1/`, and the hash-bound figure set in `../docs/figures/ramp_v6/v4r_figure_manifest.json` | Nonfinal V1-style V6 thesis sample, orphaned draft figures, and a noncanonical example manifest. |
| `markdown_docx_pipeline_20260810/` | `../latex/thesis.tex` | Retired Markdown template/materialization, claim-text validator, DOCX builder, Node dependencies, generated Markdown/DOCX deliverables, result-handoff document, and pipeline-specific tests. Canonical evidence and figure/result generators remain active. |

## Earlier archive bundles

| Item | Replaced by | Why |
|---|---|---|
| `colab_extract_data.ipynb` | `../extract_clusterdata2019_full.ipynb` | The comprehensive extractor supersedes the limited early notebook. |
| `colab_full_output.md` | Committed data artifacts and manifests | Paste of an earlier run's stdout, not a source artifact. |
| `extracted_data.zip` | Current committed data and provenance | Zip of an earlier extraction run. |
| `models_pre_reframe/` | Later grid-demand-smoothing models | Models trained under the earlier on-site-solar formulation; observation spaces differ. |
| `logs_legacy_naming/` | Canonical protocol and result manifests | Historical sweep logs using obsolete terminology. |
| `demand_charge_invalid_penalty_20260805/` | Final guarded/dense-shaped historical demand-charge models | Invalid pre-calibration and pre-terminal-accounting iterations. |
| `dqn_cfws_20260805/` | PPO-based successors | Historical DQN and CFWS-inspired trainers and artifacts. |
| `oof_invalid_source_change_20260805/` | Frozen successor campaigns | Partial artifacts invalidated when source data changed after protocol freeze. |
| `energy_model_v1_mixed_20260805/` | Energy-model V2, now itself archived in `v1_v5_legacy_20260810/` | Mixed/synthetic market scenarios retired after timestamp, DST, provenance, and phase audits. |
| `energy_model_v2_invalid_preflight_20260805/` | Corrected historical V2 gate, now in `v1_v5_legacy_20260810/` | Provisional outputs invalidated by corrected Status Quo locality and capacity units. |
| `thesis_pre_v5_20260807/` | `../latex/thesis.tex` | Pre-V5 manuscript and generator retained only for historical comparison. |
| `training_logs_pre_v5_20260807/` | Canonical historical packages in `v1_v5_legacy_20260810/` | Raw console logs are not required by the final V6/V4R evidence chain. |

## Final active path

The active workload and market pipeline is:

```text
Google ClusterData2019 + PowerData2019
  -> data/cells + data/jobs/batch_distributions_*.json
  -> data/energy_model_v3
  -> energy_model_v3/ramp_factory.py
  -> env/ramp_v6
  -> ramp_rl
  -> output/ramp_rl_v6/recovered_v4r_resealed_v2
  -> output/ramp_rl_v6/recovered_v4r_posthoc_metrics_v1
```

Nothing under `archive/` is required to run, verify, or explain the accepted
V6/V4R solution. Historical files remain byte-preserving moves from their
former paths unless an older archive entry explicitly documents invalidation.
