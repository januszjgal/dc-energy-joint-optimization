# Ramp-v6 V4R corrected thesis-result handoff

The authoritative thesis build uses the additive post-hoc telemetry wrapper:

```powershell
python scripts\bind_v4r_metric_replay_recovery.py verify
python scripts\recompute_v4r_posthoc_metrics.py verify
python scripts\build_ramp_rl_thesis_results.py --corrected
```

Canonical input:

`output/ramp_rl_v6/recovered_v4r_posthoc_metrics_v1/canonical_posthoc_metrics.json`

Canonical-JSON SHA-256:

`42bbcf4c4cd2cca58cfce0e07319d5280145e0fac948a7fe66d2a6582e379d4b`

The wrapper references the immutable sealed evidence at
`output/ramp_rl_v6/recovered_v4r_resealed_v2/canonical_evidence.json`, whose
canonical-JSON SHA-256 remains
`f642bd5868abdd9f7cda2a6fffb228250f3570fd0c6d440085da68c976892d9b`.
The original sealed package is not superseded as test evidence.

## Correction scope

The post-hoc frozen-policy replay corrects:

- physical p95 and maximum inputs to use absolute adjusted ramp magnitudes for
  every market and timestep, with no signed cross-market cancellation;
- native-grid-relative gate labels and first-class policy-versus-status-quo
  overall/per-market deltas; and
- decoder adjustment as an actual Euclidean norm in decoded work-allocation
  coordinates, with distribution and positive-adjustment rate.

The primary per-market squared incremental objective remains unchanged.
Decision-relevant objective, cost, behavior, safety, and ensemble action-chain
identities must reproduce the immutable result before the wrapper is written.
The replay occurred after unblinding and is not a second sealed generalization
test. No model selection, weighting change, retuning, or policy update is
permitted.

The committed run used deterministic checkpoint reconstruction because the
ignored V4R containers were no longer available in the clean worktree. The
wrapper therefore records that reconstruction training code ran after
unblinding, that the new outer ZIP hashes differ, and that policy, critic,
normalizer, and action-chain identities remain exact. The metric replay itself
performs no training. Neither reconstruction nor replay is fresh test evidence.

The tracked checkpoint identity is
`output/ramp_rl_v6/recovered_v4r_posthoc_metrics_v1/metric_replay_recovery_manifest.json`.
Its canonical-JSON SHA-256 is
`26e9c77a1d932ff7edb66fe8b7d55b6f769a2e2029a49c95411b62465b14f064`.
It references the complete raw `recover_ramp_rl_v3.py` verifier output and
binds stable commit `d1d4828`, every required training-manifest comparison,
loaded tensor hashes, new model-container hashes, and exact normalizer bytes.
The corresponding binaries remain ignored under
`models/ramp_rl_v6/recovery_v3_metric_replay/`.

## Baseline interpretation

The historical gate requires every policy market's incremental impact to be
negative relative to native grid ramps. It does not require policy to beat
operational status quo in every market. The corrected sealed-test diagnostic
reports five of six markets better than status quo, with MISO worse. Historical
gate fields remain readable through an explicit compatibility map.

## Fail-closed bindings

The corrected builder verifies:

- the exact corrected-wrapper canonical hash;
- the referenced immutable sealed canonical hash and one-opening chronology;
- historically recorded text artifacts by exact raw, LF-canonical, or
  CRLF-canonical SHA-256 identity, so checkout line endings cannot change data
  semantics; binary hashes remain exact raw bytes;
- protocol, source, data, forecast, policy, critic, normalizer, and controller
  identities;
- five deterministic members invoked once per decision at weights of 0.2;
- no member selection, learned weighting, trainable combiner, teacher,
  behavior cloning, analytic policy, MPC, or optimizer;
- exact invariant reproduction of primary impact, per-market impact, cost,
  behavior, safety, and ensemble action chain;
- an absolute per-market/per-timestep physical aggregation contract;
- status-quo overall and per-market comparison fields;
- a real decoder-adjustment distribution and separate emergency rate; and
- post-hoc labels that prohibit interpreting the replay as fresh test evidence.

Any mismatch blocks claim tables, figures, Markdown, and DOCX.

## Generated artifacts

```powershell
python scripts\build_ramp_thesis_figures.py
python scripts\materialize_ramp_thesis.py --output thesis_paper.md
python scripts\validate_ramp_thesis.py --source thesis_paper.md
python scripts\build_final_thesis.py `
  --source thesis_paper.md `
  --output thesis_paper.docx
```

The publication package writes a claim ledger and CSV tables for split
summary, market/status-quo comparison, corrected physical ramps, decoder
adjustment, behavior, and cost. Figures are bound to the same corrected
canonical hash. Validation and test remain separate. The 1 GW-total and c-h
views remain post-selection robustness.
