# Ramp-v6 V4R thesis-result handoff

The final result builder has two modes. Its legacy manifest mode remains
available for V1-style evidence packages. The final V4R publication uses:

```powershell
python scripts\build_ramp_rl_thesis_results.py --v4r `
  output\ramp_rl_v6\recovered_v4r\canonical_evidence.json `
  --output output\ramp_rl_v6\recovered_v4r\thesis
```

The canonical input SHA-256 must be
`b1742a2e753d9a899be256667c80679cbfcf2da4cf6056a6b471d42e66ee7b30`.
Its protocol ID is
`v6-ramp-pure-rl-recovered-equal-action-ensemble-v4r`, with protocol SHA-256
`57310edca9e7b1e917be2901010352ad928d124beeaa5a10d21ae5fddd4f78dd`.

## Fail-closed bindings

The V4R builder verifies:

- semantic equality to the canonical evidence committed at `7ebd9b2`;
- exact source freeze, source commit, source bundle, recovery binding, and
  single-opening hashes;
- a new recovered-container identity without reusing the blocked V4 identity
  or original V3 containers;
- seeds 2801-2805 and their exact recovered-model hashes;
- exact policy/critic/normalizer/training provenance in the recovery binding;
- all five deterministic actions invoked once per decision at fixed weights
  0.2, with no selection, learned weighting, trainable combiner, teacher,
  behavior cloning, analytic policy, MPC, or optimizer;
- exactly six evaluated markets, with no PJM/Virginia substitution;
- one 28-episode validation evaluation and one 60-episode sealed-test
  evaluation;
- exactly one sealed-test opening after validation, with test selection and
  tuning prohibited;
- every strict ramp, cost, safety, behavior, and leakage gate;
- post-selection-only status for 1 GW-total and c-h robustness; and
- the c-h variant's overlapping/non-independent identity.

Any mismatch blocks the claim ledger, tables, figures, Markdown, and DOCX.
Model archives themselves may remain local; their committed hashes and recovery
equivalence are the publication boundary.

## Generated artifacts

The builder writes a hash-bound `claim_ledger.json`, CSV tables for split
summary, per-market test results, physical ramp extrema, behavior, and
robustness, plus `package_manifest.json`. Figures are then generated from the
same verified canonical evidence:

```powershell
python scripts\build_ramp_thesis_figures.py
python scripts\materialize_ramp_thesis.py --output thesis_paper.md
python scripts\validate_ramp_thesis.py --source thesis_paper.md `
  --results output\ramp_rl_v6\recovered_v4r\canonical_evidence.json
python scripts\build_final_thesis.py --source thesis_paper.md `
  --output thesis_paper.docx
```

Validation and test are rendered separately and never pooled. The 1 GW-total
and c-h views are post-selection robustness, not additional independent
confirmatory tests.
