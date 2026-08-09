# Ramp-v6 selected-result handoff

Run:

```powershell
python scripts\build_ramp_rl_thesis_results.py <manifest.json> --artifact-root <repo-root> --output <output-dir>
```

The manifest schema is `ramp-v6-thesis-evidence-manifest-v1`. Paths must be
relative to `--artifact-root`, and every referenced artifact must carry its
exact SHA-256. The audit rejects path traversal and does not open model policy
archives.

## Required bindings

- Frozen protocol artifact, protocol ID, file hash, and protocol bundle hash.
- Factory manifest plus source-panel, acquisition, frozen-stat/data, forecast,
  and source-bundle identities.
- Exact train, validation, and sealed-test month identities.
- Exact selected algorithm, optimizer seed set, six-market set, validation
  split, promotion state, and safety/cost/ramp gate assertions.
- One pure-RL training manifest and one validation result per selected seed.
  Each validation reference must bind the seed, algorithm, training-manifest
  hash, model archive hash, and final-policy hash. A selected result must also
  embed the same `model_identity` values inside each validation artifact. The
  audit compares hashes but never opens policy archives.
- Explicit sealed-test state. An unopened test requires a reason and null
  opening/result references. An opened test requires a hash-bound opening
  record proving authorization after validation and one test result marked
  `used_for_selection: false`.
- The canonical failed v1 result as `negative_protocol`; it is reported only as
  validation evidence and is never pooled with test evidence. The reference is
  pinned to confirmation commit `3391440d168ade7127899a8a2c2d84e3a47a9ad3`
  and canonical result SHA-256
  `ffe94a7c00d8d721940060d06d4aff0880b63ec70a7028b9464a8b188cf6e656`.

Selected results must provide an `analysis_artifact` containing complete
`market_ramps`, `forecast_error_strata`, `dc_ramp_behavior`, `learning_curves`,
and `sensitivities` arrays. Per-seed safety/decoder telemetry is taken directly
from the hash-bound validation results. `market_ramps` must contain every
market x `{1h, 3h}` x `{native, status_quo, policy}` combination. Sensitivities
must include both `base_100mw` and `post_selection` phases.

The selected protocol is restricted to the repository's frozen v2 protocol,
environment, and panel-schema bundle. Its algorithm, five seeds, and split
months must exactly match that bundle. The analysis artifact must include an
`identity` block binding the canonical-result hash, protocol/source bundle
hashes, algorithm, seed set, and every validation/model/policy hash.

Failed validation packages may omit analysis sections only when every omission
has an explicit reason in `figure_omissions`; generated figures then display
the evidence limitation rather than inventing values.

## Sealed-test opening record

An opening record is a JSON object with:

```json
{
  "opened": true,
  "opened_at_utc": "2026-08-09T20:00:00Z",
  "validation_finalized_at_utc": "2026-08-09T19:00:00Z",
  "authorized_after_validation": true,
  "selected_protocol_id": "v6-ramp-pure-rl-preregistered-v2",
  "validation_result_sha256": "<canonical-result-sha256>",
  "test_result_sha256": "<single-test-result-sha256>"
}
```

The result builder verifies both hashes and refuses any test evidence before a
strict validation promotion. The single test result must repeat the protocol,
source, algorithm, seed, market, and per-model bindings from validation and
must remain marked `used_for_selection: false`.
