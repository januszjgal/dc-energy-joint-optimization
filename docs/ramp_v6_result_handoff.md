# Ramp-v6 V4R corrected thesis-result handoff

The authoritative thesis build uses the additive post-hoc telemetry wrapper:

```powershell
python scripts\recompute_v4r_posthoc_metrics.py verify
python scripts\build_ramp_rl_thesis_results.py --corrected
```

Canonical input:

`output/ramp_rl_v6/recovered_v4r_posthoc_metrics_v1/canonical_posthoc_metrics.json`

Canonical-JSON SHA-256:

`bd1e8a242ec93e389c9e7b9e13ae7f19b60a9445716206b6527938ffe639f842`

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
