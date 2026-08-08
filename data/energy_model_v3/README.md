# Independent US Energy Model v3

Energy model v3 is an additive, standalone six-market evidence pipeline. It
does not import, rewrite, or regenerate frozen v2-v5 data, scenarios, models, or
results. The legacy four-site a-d/e-h study remains unchanged.

## Study design

The primary sites are independent market observations on one exact hourly UTC
index:

| Site | Price location | Physical stress geography | Borg cell |
|---|---|---|---|
| Northern Virginia | PJM DOM pnode 34964545 | PJM RTO | a |
| New York City | NYISO Zone J PTID 61761 | Zone J load; NYCA renewable context | b |
| Northern California | CAISO NP15 | CAISO system | c |
| North Texas | ERCOT LZ_NORTH | ERCOT system | d |
| Minnesota | MISO MINN.HUB | MISO system | e |
| SPP North | SPPNORTH_HUB | SPP balancing authority | f |

The primary capacity is 100 MW per site and 600 MW total. The primary
sensitivity is 1 GW total, equally allocated (166.6667 MW per site). The
explicitly non-primary price-taking stress is 1 GW per site and 6 GW total.
Every non-stress scenario is subject to a 5% `P/S` penetration gate. Only the
stress scenario declares `penetration_override: true`.

The robustness mapping is cells c-h. It overlaps the primary mapping and is
therefore labeled **overlapping/non-OOF**. It is not an out-of-fold claim.

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
must be explicit.

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

PJM requires `PJM_API_KEY`; MISO historical physical data requires
`MISO_API_KEY` and a subscription. Optional EIA-930 use requires `EIA_API_KEY`,
must match the same balancing authority, and must be activated explicitly. It
is never a silent substitute.

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

## Current live capability status

The committed report under `output/energy_model_v3/` is the source-of-truth for
the retrieval timestamp. Public NYISO, CAISO, ERCOT, and SPP probe endpoints
were reachable. PJM and MISO are blocked because their required credentials
were absent. Consequently the six-market panel, live scale configuration,
live effective-rank/ramp diagnostics, and market-evidence plots remain
correctly blocked rather than being fabricated.
