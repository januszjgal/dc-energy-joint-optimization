# Independent US Energy Model v3

Energy model v3 is an additive, standalone six-market evidence pipeline. It
does not import, rewrite, or regenerate frozen v2-v5 data, scenarios, models, or
results. The legacy four-site a-d/e-h study remains unchanged.

## Study design

The primary sites are independent market observations on one exact hourly UTC
index:

| Site | Price location | Physical stress geography | Borg cell |
|---|---|---|---|
| Northern California | CAISO NP15 | CAISO system | a |
| North Texas | ERCOT LZ_NORTH | ERCOT system | b |
| New York City | NYISO Zone J PTID 61761 | Zone J load; NYCA renewable context | c |
| Minnesota | MISO MINN.HUB | MISO system | d |
| SPP North | SPPNORTH_HUB | SPP balancing authority | e |
| Boston/NEMA | ISO-NE location 4008 | ISO New England balancing authority | f |

The primary capacity is 100 MW per site and 600 MW total. The primary
sensitivity is 1 GW total, equally allocated (166.6667 MW per site). The
explicitly non-primary price-taking stress is 1 GW per site and 6 GW total.
Every non-stress scenario is subject to a 5% `P/S` penetration gate. Only the
stress scenario declares `penetration_override: true`.

The robustness mapping is cells c-h. It overlaps the primary mapping and is
therefore labeled **overlapping/non-OOF**. It is not an out-of-fold claim.

Live primary panels use the measured Google ClusterData2019 Borg cell a-f
profiles. The 8,928 complete five-minute intervals are averaged to a 744-hour
profile, converted with the committed per-cell power model, and tiled from the
candidate UTC start to cover the eight-month market calendar. The terminal
boundary sample is not treated as a five-minute interval. Exact source,
power-model, artifact hashes, and repetition counts are recorded in
`provenance/workload-power-manifest.json`.

## Calendar and leakage boundary

The locked candidate is `2025-09-01T00:00:00Z` through
`2026-05-01T00:00:00Z` exclusive (5,808 hourly rows):

- September 2025 through January 2026: training;
- February 2026: validation; and
- March-April 2026: sealed test.

If actual source coverage disproves this candidate, the builder may select only
the latest all-six intersection of at least eight consecutive complete months
ending at least 90 days before retrieval. It uses the first five, next one, and
final two months. It must emit `BLOCKED` rather than shorten the window, drop a
market, interpolate, or silently change products.

For each market, `S_m` is training-only Q95 gross demand. The residual-tail
threshold is training-only Q90 of `net_load_mw / S_m`. Validation and test data
cannot influence scales, thresholds, normalization, forecast fitting, or
reward calibration.

## Canonical tables

Native products are retained separately. Canonical tables use UTC interval
start/end and long-form market rows. Required physical evidence is gross
demand, wind, solar, defensible native or derived net load, market scale, and
quality/source flags. Primary price is day-ahead total LMP; real-time LMP and
available price components stay separate.

Every source descriptor records the authoritative feed/product/location,
authentication, licensing, native cadence, units, interval semantics,
timezone/DST handling, preliminary/final and revision handling, retrieval
query, retrieval time, raw hash, quality flags, and redistribution policy.
See `provenance/source_contract.json`.

Subhourly products remain native secondary evidence. Hourly aggregation
requires every native interval; missing intervals fail closed and are never
forward-filled. Five- and 15-minute ramps are not described as primary hourly
evidence.

Forecasts require target, value, issue time, vintage, horizon, capability, and
quality flags. Issue time cannot follow the target. Realized future RT price or
load is rejected as a forecast. A blocked or reconstructed causal capability
must be explicit. Training-period forecasts use daily expanding-window model
vintages whose training observations strictly precede the vintage; the final
first-five-month model is frozen at the February boundary for validation and
sealed test forecasts.

## Physical and economic derivations

For each market and `h` in one and three hours:

```text
b = (N_t - N_t-h) / (S * h)
a = ((N_t + P_t) - (N_t-h + P_t-h)) / (S * h)
I_h = a^2 - b^2
primary incremental ramp = 0.40 I_1h + 0.60 I_3h
```

A smaller 0.10 training-Q90 residual-tail term is persisted separately.
Day-ahead energy cost stays separate. Output also stores current/trailing
normalized gross/net level, prior modeled DC power, `P/S`, and `P/D`. LMP is
economic context, not physical utilization.

Episodes require three real hours of warm history and three terminal-tail
hours. Circular wrap is forbidden. All markets must have the identical common
index, and DC power is aggregated once per market.

## Fail-closed behavior

The pipeline blocks on missing/duplicate rows, mixed preliminary/final
products, unit ambiguity, geography mismatch, a missing primary physical
tuple, credentials, or an incomplete common calendar. No synthetic
substitution exists. Synthetic fixtures are marked
`SYNTHETIC_FIXTURE_ONLY_NOT_MARKET_EVIDENCE` and are used only to test schemas,
derivations, diagnostics, and plots.

**PJM DOM / Northern Virginia is BLOCKED and excluded from the live roster,
results, and claims until `PJM_API_KEY` is provided. It was not evaluated.**
MISO's public daily price reports require no credential. Where a complete
operator historical physical product is unavailable, the pipeline may use only
the same-balancing-authority UTC series from the no-key EIA `EBA.zip` bulk
archive. Every fallback records its source role and reason; it is never silent.
Negative renewable-generation adjustments in EIA bulk series are explicitly
clipped to zero before net-load derivation, with raw minima and affected-hour
counts retained in the live acquisition manifest. This is a documented
non-negativity normalization, not interpolation.

## Commands

From the repository root:

```powershell
python scripts\build_energy_model_v3.py describe-sources
python scripts\build_energy_model_v3.py probe
python scripts\build_energy_model_v3.py download-samples
python scripts\build_energy_model_v3.py probe-parsers
python scripts\build_energy_model_v3.py probe-ercot-physical
python scripts\build_energy_model_v3.py download-ercot-physical
python scripts\build_energy_model_v3.py preflight
python scripts\build_energy_model_v3.py fixture-diagnostics
python scripts\build_energy_model_v3.py build
python scripts\build_energy_model_v3.py acquire-live
python -m unittest discover -s tests\energy_model_v3 -v
```

`download-samples` hashes one locked probe response from each reachable public
endpoint and deletes the raw bytes. It is not a full-calendar downloader and
does not bypass redistribution limits. Full builds consume explicitly staged
native canonical files under `native/<MARKET>/` and modeled DC power under
`modeled_dc_power/`; they fail unless every required file has the exact common
calendar.

`probe-parsers` downloads representative locked public price products, parses
their production schemas into canonical rows, records raw hashes and canonical
coverage, and discards the raw bytes. PJM remains an explicit credential
block. These are parser/schema probes, not a substitute for the complete
all-six physical and price calendar.

`probe-ercot-physical` performs a no-key live schema probe of the public
`Native_Load_2025.zip` and `Native_Load_2026.zip` archives, discovery-selected
report-13052 SCED disclosure, and report-13424 validation workbook. Raw bytes
are hashed and discarded. The committed probe confirms that the two load
archives form the exact 5,808-hour candidate index.

`download-ercot-physical` is the resumable full-calendar path. It discovers
each report-13052 document by operating date plus 60 days, persists the public
ZIP and its DocID/filename/publication/hash metadata, reads `SCED Time Stamp`,
`Repeated Hour Flag`, resource name/type, and telemetered net output, and
normalizes the source `WIND` class to the requested WGR semantic class while
retaining PVGR. It sums resources at each irregular SCED execution and
duration-weights the resulting step values across UTC hour boundaries. Real
warm history and terminal tail are mandatory; circular padding is forbidden.
Execution gaps over 20 minutes and minimum contributing-resource counts are
preserved as diagnostics.

The downloader joins reconstructed wind/solar with public native load, derives
net load exactly once, and writes `physical_hourly.csv`. September-December
2025 is compared independently with annual report 13424. Bias, MAE, RMSE,
quantiles, correlation, and missing-hour counts are diagnostic only and never
calibrate or replace the SCED reconstruction. Reports 13028, 13483, 14787, and
21809 are forbidden for retrospective use because their retention is only
seven days. EIA bulk ERCO demand/wind/solar remains sensitivity-only because
of the observed 24-hour December 5-6 renewable gap.

The preflight also consumes
`provenance/coverage_evidence.json`. Current probes show candidate-boundary
availability for NYISO and CAISO and a covering retention span for SPP, but
they do not clear the exact full-product calendar gate until every required
interval and raw hash is staged. ERCOT's former 2026 physical-source blocker is
resolved: public native-load archives cover all 5,808 hours and report 13052
retention covers all 243 required operating dates. ERCOT remains fail-closed
only because the full 243-document SCED reconstruction and complete price
archives have not yet been staged and hash-bound. EIA-930 is not needed for
the primary tuple and cannot silently replace it.

`acquire-live` downloads and hashes the actual source files, stages restricted
raw/native/panel evidence only in ignored local paths, creates train-only
scales and causal reconstructed 1h/3h forecasts, and writes permissible
manifests and diagnostics. No fixture or synthetic market data can satisfy live
preflight.

The live preflight also requires product-specific canonical coverage files and
raw hash manifests for every trusted product in
`provenance/coverage_evidence.json`. Acquisition alone cannot bypass this gate.

## Current live capability status

The preserved live acquisition provides the complete six-market candidate
calendar and validated local panel. PJM DOM / Northern Virginia remains
credential-blocked, was not evaluated, and is excluded from every result and
claim.

The final ramp-aware publication consumes the hash-bound derivative at
`output/ramp_rl_v6/recovered_v4r/canonical_evidence.json` (SHA-256
`b1742a2e753d9a899be256667c80679cbfcf2da4cf6056a6b471d42e66ee7b30`).
The V4R thesis build validates the committed factory/source identities and does
not reacquire, interpolate, retrain, or retune data or models.

Raw operator downloads, restricted native products, the full local panel, and
recovered model containers may be absent from a fresh clone because of
licensing or artifact-size constraints. Their permissible manifests, hashes,
coverage evidence, source freeze, recovery binding, and canonical evaluation
results are committed. Run the live preflight only in a worktree where the
restricted local files have been staged at the exact paths recorded by the
manifests:

```powershell
python scripts\build_energy_model_v3.py preflight
```

The eight-month September 2025-April 2026 panel does not support a prior-year
or future-year conclusion. Any extension must acquire complete all-six market
coverage, freeze new causal forecast vintages and train-only scales, define a
new sealed split, and publish a new protocol identity. The present
March-April 2026 sealed test cannot be recycled for model or tolerance
development.
