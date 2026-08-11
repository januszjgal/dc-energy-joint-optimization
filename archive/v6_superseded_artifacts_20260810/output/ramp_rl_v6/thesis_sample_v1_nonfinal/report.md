# Ramp-v6 thesis result package: v1 failed confirmation (validation only)

**NON-FINAL VALIDATION EVIDENCE.** Result status: `failed_validation`. Selection used validation only; sealed test opened: **false**.

## Audited result

| Seed | Validation ramp impact | DA cost ratio | Safety | Cost gate | Every-market ramp gate |
|---:|---:|---:|---|---|---|
| 2601 | -1.47394353695e-05 | 0.99193776 | True | True | True |
| 2602 | -1.58831491944e-05 | 0.98580381 | True | True | False |
| 2603 | -1.57353810215e-05 | 0.98111230 | True | True | True |
| 2604 | -4.00807153722e-06 | 0.97426102 | True | True | True |
| 2605 | -4.42000152464e-06 | 0.95885962 | True | True | False |

## Evidence boundaries

- Protocol: `v6-ramp-pure-rl-preregistered-v1` at `2a993afd22d81b72371c8b66b2e97f3601a635adb193b9ae94bdfcebf2559846`.
- Seeds: `2601, 2602, 2603, 2604, 2605`.
- Markets: `CAISO_NP15, ERCOT_LZ_NORTH, ISONE_NEMA, MISO_MINN_HUB, NYISO_NYC_J, SPP_NORTH_HUB`.
- Selection split: `validation`; test metrics are never mixed into validation tables.
- Sealed-test opening record: `not opened`.

## Figure availability

- `dc_ramp_behavior`: unavailable - The v1 evaluator persisted only six-market aggregate pre-service and ramp-power behavior, not per-market before/during DC power.
- `forecast_error_strata`: unavailable - The v1 evaluator did not persist forecast-error strata joined to policy outcomes.
- `learning_curves`: unavailable - The v1 closeout persisted only the 100k screen and 500k validation endpoints, not per-seed checkpoint curves.
- `market_ramps`: unavailable - The v1 evaluator did not persist per-market native, status-quo, and policy 1h/3h ramp decompositions.
- `sensitivities`: unavailable - The preregistered 1GW and workload sensitivity runs were correctly blocked after failed validation.
- `safety_decoder`: available from hash-bound per-seed validation evidence.

## Claim audit

| Claim | Status |
|---|---|
| The evaluated policies were trained by pure RL under the frozen protocol. | **confirmed** |
| Policy 1h and 3h ramps are lower than native and status quo in every market. | **pending** |
| Ramp improvement remains negative across reported forecast-error strata. | **pending** |
| The policy pre-services deferrable work and lowers DC power during ramps in every market. | **pending** |
| Each selected seed improves its reported validation ramp metric over training. | **pending** |
| Post-selection scale and workload sensitivities retain safety, cost, and ramp gates. | **pending** |
| The protocol passed all preregistered validation promotion gates. | **failed** |
| All selected-seed validation evaluations passed hard safety gates. | **confirmed** |
| All selected-seed validation evaluations met the DA cost gate. | **confirmed** |
| All selected-seed validation evaluations improved ramp impact in every market. | **failed** |
| The validation-selected protocol was confirmed once on the sealed test. | **failed** |
| The v1 confirmation failed validation and did not open the sealed test. | **confirmed** |

The v1 row is retained as a negative validation protocol comparison. It is not pooled with a test result, and no sealed-test metric is inferred.
