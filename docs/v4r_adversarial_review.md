# Independent adversarial review: ramp-aware pure-RL V4R

Review date: 2026-08-09  
Reviewed evidence commit: `7ebd9b25e3c830076be4687d73d0cea14ef1e55d`  
Canonical evidence: `output/ramp_rl_v6/recovered_v4r/canonical_evidence.json`

> **Superseded interpretation notice (2026-08-10).** The quoted headline below
> incorrectly says "relative to status quo in every evaluated market." The
> historical gate was relative to native grid ramps. Direct sealed-test
> comparison shows policy better than operational status quo in five of six
> markets, with MISO worse. This review is retained as historical audit
> chronology; current authority is the additive
> `recovered_v4r_posthoc_metrics_v1` wrapper and rebuilt thesis.

## Publication decision

**The headline is supported. Before final release, make one non-scientific
provenance-portability correction and four claim/tooling corrections.** The
sealed numerical evidence itself recomputes exactly, the five recovered models
load with the declared V3-equivalent policy/critic and normalizer identities,
and the narrow headline is supported:

> On the fixed six-market, price-taking simulator and March-April 2026 sealed
> test, the deterministic ensemble of pure-RL-origin PPO policies reduced mean
> normalized incremental squared-ramp impact relative to status quo in every
> evaluated market while satisfying the declared simulated workload, safety,
> and modeled day-ahead-cost gates.

The evidence does **not** support literal opposite-direction "counter-ramping"
at every event, real-grid reliability relief, retail savings, a new V4R
optimizer-seed interval, iid six-market generalization, or future-year
generalization.

## High-confidence findings

### PORTABILITY/TOOLING: some evidence hashes depend on checkout line endings

The recovery binding is correct and recovery equivalence is sound. However,
downstream V4R sidecars mix hashes of Git LF blob bytes with hashes of Windows
CRLF working-tree bytes, creating a cross-platform publication-audit issue.

`source_freeze.json` records:

```text
recovery_source_report_sha256 =
d1ccf86acf6ed8f9a58c1d07546d02678f77f1aeab8c1181e09e16a3173357e6
```

That is the canonical committed-stream hash of
`recovery_transfer_manifest.json` at recovery commit `d1d4828`, and the hash in
the originating V4R worktree, where the corresponding generated recovery-source
manifest remained LF. A Windows checkout that converts that file to CRLF would
instead hash:

```text
a162c2ed8325a3a9d1b966e8b08d633b54ae905a278b9dedf062cac6768b6b4c
```

The reverse inconsistency exists for other chained JSON. Their recorded hashes
are CRLF working-tree hashes, not committed LF blob hashes:

| Artifact | Recorded/Windows CRLF SHA-256 | Commit LF SHA-256 |
|---|---|---|
| `canonical_evidence.json` | `b1742a2e753d9a899be256667c80679cbfcf2da4cf6056a6b471d42e66ee7b30` | `130004efd05e82fe12b21fc7ac6e605c3010728bae0db71dfda3c90c90508a67` |
| `source_freeze.json` | `21e3b036be9e111a4af26ac60697e44993fbc681cd38d2b9ec2ad7c9d9848c02` | `8862a0dbd416387e19c35b05a6032c2607a6a0ddcb35a5a7fa5694a9562ba507` |
| `recovery_binding.json` | `2c125bce0de306aa6606942dba05d85f975153e33dffc41b5b6d8e4d258e98b4` | `a8aca76875077b4fb2e58fa4cb0411f2822a9652680afb90ae66c651cf64eed2` |
| `sealed_test_opening.json` | `fd596881c6f83dc1bc0d77f238727ceb8d2a12b12c42b96a6d4061397c90f837` | `bc6b3908527052e42faac1d65a4b347068be29d2b4e89ca737eee6df7a0b84b2` |

Consequences:

1. `scripts/run_ramp_rl_v4r.py::verify_source_freeze` does not re-check
   `recovery_source_report_sha256`; therefore the prior inference that this
   verifier necessarily fails on a clean Windows checkout was incorrect.
2. Prospective
   `python scripts/build_ramp_rl_thesis_results.py --v4r --audit-only` passes
   in the originating Windows checkout, but its raw-byte checks of the
   canonical evidence, source freeze, recovery binding, and sealed-test opening
   can fail in an LF checkout.
3. That publication verifier does not verify the nested recovery-source report
   hash, so it should be strengthened even though the frozen binding value is
   correct.
4. This is a portability/tooling limitation in publication verification, not a
   scientific or provenance invalidation and not a reason to rerun or reseal
   the test or create a new protocol/evidence identity.

**Required correction:** make an additive provenance/tooling correction before
final release. The minimal preferred policy is a careful `.gitattributes`
mapping that preserves historically hashed bytes across platforms:
`output/ramp_rl_v6/recovered_v4r/recovery_source_manifest.json eol=lf`, with
the other hashed V4R JSON sidecars set to `eol=crlf` and explicit exceptions
where their historical bytes differ. Then make the publication audit verify
the complete nested chain and test it in both Windows and LF checkouts.
Alternatively, add a supplemental provenance manifest defining canonical JSON
or Git-blob hashes. No test rerun, reseal, protocol change, model change, or new
scientific evidence identity is needed. If supplemental hashes are published,
give that erratum its own provenance-manifest identity while retaining the
original `7ebd9b2` evidence identity.

### MAJOR: "counter-ramp" is stronger than the estimand

The metric is

```text
((adjusted normalized ramp)^2 - (native normalized ramp)^2)
```

A negative value proves a lower squared ramp magnitude. It does not prove that
data-center power moved in the opposite direction from the native ramp. For
example, reducing a positive ramp from `+10` to `+8` is favorable under the
metric without being an opposite-direction counter-ramp.

Prospective thesis text archived at
`archive/markdown_docx_pipeline_20260810/thesis_ramp_v6.md:72`, the conclusion
at line 536, and generated table wording in
`archive/markdown_docx_pipeline_20260810/scripts/materialize_ramp_thesis.py:87`
use "counter-ramp" as the direct
interpretation.

**Required correction:** use "reduced normalized squared-ramp impact/stress"
or "attenuated ramp magnitude." Reserve "counter-ramp" for a separately
reported sign-opposition statistic.

### MAJOR: the source figure can report the wrong physical source

Prospective `scripts/build_ramp_thesis_figures.py:80-89` chooses the last
descriptor in `source_contract.json` as the physical source. That contract
describes intended products, not the final selected acquisition path.

The hash-bound live acquisition manifest records:

- CAISO physical data used same-balancing-authority EIA fallback because the
  historical Today's Outlook CSV lacks an authoritative UTC field and exposes
  DST-day display-row ambiguity.
- SPP physical data used same-balancing-authority EIA fallback because the
  GenMix365 snapshot had 67 incomplete five-minute hours and interpolation was
  prohibited.

The current figure logic instead labels CAISO Today's Outlook and SPP GenMix.

**Required correction:** construct the figure from
`live-acquisition-manifest.json` selected `physical_sources`, including
fallback labels and geography, rather than positional source-contract entries.

### MODERATE: blocked V4 is plotted as a numeric zero

Prospective `scripts/build_ramp_thesis_figures.py:111-136` hard-codes original
V4 impact as `0.0` even though V4 was blocked and unevaluated. The annotation
says "no evaluation," but a zero-height bar is still a numerical performance
encoding.

**Required correction:** render V4 as `N/A`/missing with a distinct blocked
marker, not as zero.

### MODERATE: avoid statistical "independent markets" wording

The six series are separately sourced and no market is a shifted copy of
another. They are not demonstrated statistically independent, and the thesis
correctly states later that they are not iid superpopulation draws.
The archived template at
`archive/markdown_docx_pipeline_20260810/thesis_ramp_v6.md:23-24` and line 180,
plus the prospective figure title, still use "independent."

**Required correction:** say "six separately sourced/evaluated market series."

## Verified provenance and protocol chronology

The evaluated chain is a strict ancestor sequence:

```text
46329fe765f596f84eeac71f061dcbd191a90583  Bind second recovery to V4R
fd152948c37ea00c1e0e18383eadfc7ab515e69d  Freeze V4R before evaluation
35969b71009628cf4ca08b0e6f93b0e72f829013  Seal passing V4R validation
96496749cdcecff2ed326a74dba091f41d3b382d  Open V4R sealed test once
864ea8203f704b6e05f007dd6c8595a27ae5d8e8  Seal passing V4R test
7ebd9b25e3c830076be4687d73d0cea14ef1e55d  Publish canonical V4R evidence
```

Commit timestamps run from `2026-08-09T18:56:07-04:00` to
`2026-08-09T19:03:34-04:00` in that order. The source freeze records no
pre-existing validation/test result. Validation passed before the sole opening
record. The opening binds validation commit `35969b7`, result
`e268bedc314c9102d865f7d5896996e2afa7ce74d7d01e1af50bedf1668ecbca`,
and decision
`4d07c14122eeb40601e7a573fc9b8026ea9c347e7932c2cd0923ded7e38af5a1`.
The test decision then authorizes robustness without retuning.

At recovery commit `d1d4828`, the canonical Git-blob SHA-256 values are:

| Artifact | Canonical committed-stream SHA-256 |
|---|---|
| `recovery_transfer_manifest.json` | `d1ccf86acf6ed8f9a58c1d07546d02678f77f1aeab8c1181e09e16a3173357e6` |
| `scripts/recover_ramp_rl_v3.py` | `18f509b1cb7694d7e854e18846ba25481f0d5c89438f3f55f648ed649f8423a1` |
| `.gitignore` | `65a5fc37fdb74b89f2421c7a4a02994fb8dca2102755179ddca890fa5e174ca9` |

The V4R recovery binding correctly uses the first two canonical committed-stream
hashes. All recovered model, normalizer, and artifact hashes remain unchanged.

Original V4 has no `output/ramp_rl_v6/live_v4` evaluation artifact at
`1ddd4c8` or `7ebd9b2`. Canonical V4R explicitly records
`blocked_original_v4_evaluated=false`. V4R uses new model-container hashes and
does not reuse deleted V3 container or blocked V4 identities.

No V2-V5 evidence or source was changed after the V4R source freeze. The only
non-V4R path changed between original V4 and V4R publication was `.gitignore`;
after `fd15294`, changes through `7ebd9b2` are confined to
`output/ramp_rl_v6/recovered_v4r/`.

## Recovered member audit

The five ignored recovered models are absent from this review worktree but
remain available in the originating local V4R worktree. Their hashes match the
frozen binding:

| Seed | Recovered model SHA-256 | Normalizer SHA-256 |
|---:|---|---|
| 2801 | `47e68ae430be484da860b6e1520f1868e5869ccbd49d86fe48920a253c861883` | `0789b19f8999d1cdc79249daa33db92fe4ef9152f7202274d26a90bcae0549c3` |
| 2802 | `93296e54bb61be72b9c41f58175860734208f1785089a11730236891f9a8700f` | `bbb6ccad327453c757fd636ed64b4c5af6012490dbaecfc004aaab57fc0c5bd3` |
| 2803 | `292a24819bb13dd8372b25c932eb3ac44b06bab891237ab331c7ba224873f507` | `f5362840c1fb8c96983ec2d60505e15619fa0bfb3b35ec69a18808f4c178dd9c` |
| 2804 | `ee5ee15ce884d7958da253ea7fb6f95297d395cb0ca5cc816d83ac72bb345fbb` | `778fb2ad91d1ec1e9a16a76f5b1cef139c2c2fbb6256479868473bbb9155b63e` |
| 2805 | `ffba64a2b4574e1a10edfd96e91bcd08f78d380c756a4d3231ac5ee1365e07f6` | `e9e1d4523ddfd16f17744deaa371627fcae2d0316a971a02f140cc73c2d2e8f7` |

Loading all five recovered PPO models independently reproduced every bound
final policy and critic tensor hash. `VecNormalize` state matched each original
V3 training manifest exactly, including mean, variance, sample count, and
train-only fit. All manifests report 110,592 interactions, 540 updates,
distinct random initialization, only `randomly_initialized_policy` replay
rows, zero external/teacher/optimizer/demonstration rows, train-only provenance,
and no future-realized features.

The controller path:

- invokes all five members once per decision;
- applies each member's own persisted normalizer;
- requests deterministic policy actions;
- averages five environment-space float64 actions equally and casts to
  float32;
- passes the mean unchanged to the frozen constraint decoder;
- rejects out-of-bounds actions; and
- contains no teacher, BC, demonstration, expert replay, analytic base, MPC,
  optimizer, evaluation action, trainable combiner, member selection, or
  validation/test-derived weighting.

Invocation counts recompute as `756` per member for validation and `1,620` per
member for test and each robustness run, exactly matching policy decisions.

## Independent metric recomputation

All aggregates below were recomputed from `policy_episodes` and
`status_quo_episodes` in the four raw result artifacts, not copied from summary
fields. Every recomputed scalar, market mean, quantile, bootstrap interval, and
canonical embedding matched exactly.

| Result | Episodes | Mean incremental impact | Cost ratio | 1 h p95 / max | 3 h p95 / max | Day 95% bootstrap |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 28 | `-1.4086907197938803e-05` | `0.98582806727793115` | `0.047103281858883447` / `0.07358149729708896` | `0.039959652419719785` / `0.060072093274429438` | `[-1.4993411574815426e-05, -1.3151365991801374e-05]` |
| Sealed test | 60 | `-1.4258710514255097e-05` | `0.97755844024893435` | `0.037686970112269486` / `0.067176955753757733` | `0.03420304456244664` / `0.054374106734853191` | `[-1.4922706380880133e-05, -1.3524515742754826e-05]` |
| 1 GW total | 60 | `-1.707526521658319e-05` | `0.96812698489041882` | `0.037638862731382908` / `0.066784856898729761` | `0.034097235281036004` / `0.054220895516897488` | `[-1.8054747947133858e-05, -1.6111243268462583e-05]` |
| c-h overlapping | 60 | `-1.7135578243250341e-05` | `0.97201980892317286` | `0.037648572442787014` / `0.066852627947818502` | `0.034070052895764633` / `0.054351210907143177` | `[-1.8116492684846006e-05, -1.6141647164985525e-05]` |

Test month-block interval:
`[-1.5155776265911473e-05, -1.3299778159036212e-05]`.
The one-month validation month interval is necessarily degenerate at
`-1.4086907197938803e-05`.

All six market means are strictly negative in every run. Sealed-test values:

```text
CAISO_NP15      -2.1065969623751118e-05
ERCOT_LZ_NORTH  -2.3454355897116744e-06
ISONE_NEMA      -2.7491577052292107e-05
MISO_MINN_HUB   -5.6524485010333756e-07
NYISO_NYC_J     -3.2978547293241652e-05
SPP_NORTH_HUB   -1.1054886764306852e-06
```

For all four results: service unserved, unfinished batch, expiry, certificate
violations, terminal work, emergency rate, semantic adjustment, and leakage
are zero/false. Deferrable pre-service is positive. Policy ramp-period power is
below status quo:

```text
validation  132335.43064135703 < 135587.93643082705
test        303077.90355961508 < 313167.22669593635
1 GW        507298.02695015143 < 521945.37782656064
c-h         513373.64970121073 < 528004.06877474196
```

The 1 h/3 h values are normalized adjusted-ramp magnitudes, fractions of each
market's train-only gross-demand Q95 scale per hour. They are not a pooled MW
quantity. The ramp-power totals are sums of hourly MW samples across
markets/episodes; their relative reduction is interpretable, but they are not
instantaneous fleet MW.

## Data, causality, and scope audit

- Calendar: 5,808 exact hourly UTC rows from 2025-09-01 through 2026-05-01;
  September-January train, February validation, March-April test.
- Markets: CAISO NP15, ERCOT North, NYISO Zone J, MISO Minnesota Hub, SPP
  North Hub, and ISO-NE NEMA. PJM is `BLOCKED_EXCLUDED`; Northern Virginia was
  not evaluated.
- Prices are wholesale day-ahead operator LMP/LBMP products, not retail bills.
- Physical and price geographies differ and are disclosed. Same-BA EIA
  fallbacks are used where complete operator physical history was unavailable.
- No interpolation or synthetic market data is declared. Negative EIA
  renewable values are explicitly normalized to zero with diagnostics.
- Market scaling is train-only gross-demand Q95; train-only net-load/ramp
  calibration is frozen at 2026-02-01.
- Forecasts are reconstructed causal expanding-window ridge models using
  observed lags and calendar features. Each forecast records issue/vintage and
  training-target bounds, with no future-realized policy feature.
- Three real warm-history hours and a scored three-hour no-arrival terminal
  tail prevent circular-wrap and end-of-window work deferral gaming.
- Primary scale is six 100 MW proxy sites (600 MW total). The 1 GW result is a
  post-selection scale sensitivity. c-h reuses overlapping workload sources
  and is not a holdout.
- The price-taking assumption excludes endogenous LMP, unit commitment,
  reserves, congestion, and network response.

The committed V3 protocol, V3 source freeze, factory manifest, raw acquisition
manifest, and all c-h workload/deadline inputs available in this worktree match
their declared hashes. The ignored live-panel manifest is not available in
this audit worktree, so a full raw-data rebuild was not attempted. This is
consistent with the documented licensing/local-data boundary but must remain a
reproduction limitation.

## Statistical interpretation

- V4R is one deterministic frozen ensemble evaluated with environment seed
  2800. Its reported `evaluation_seed_interval` has `seed_count=1`; it is not a
  new optimizer-seed uncertainty interval.
- Seeds 2801-2805 establish the pure-RL origins and member diversity, not five
  independent evaluations of the final ensemble.
- Day-block intervals describe within-split temporal variation.
- Validation has one month, so its month bootstrap cannot estimate
  between-month generalization. Test has only two months.
- Markets are fixed cases and must not be treated as iid draws from US grids.
- c-h is overlapping/non-independent post-selection robustness.
- Robustness analyses cannot increase the nominal independence of the one
  sealed test.

## Reproduction commands and checks

Representative exact commands used:

```powershell
git show -s --format='%H|%cI|%s' 46329fe fd15294 35969b7 9649674 864ea82 7ebd9b2
git merge-base --is-ancestor 46329fe fd15294
git merge-base --is-ancestor fd15294 35969b7
git merge-base --is-ancestor 35969b7 9649674
git merge-base --is-ancestor 9649674 864ea82
git merge-base --is-ancestor 864ea82 7ebd9b2

Get-FileHash -Algorithm SHA256 output\ramp_rl_v6\recovered_v4r\canonical_evidence.json
git ls-files --eol -- output/ramp_rl_v6/recovered_v4r/*.json

python -m unittest tests.ramp_v6.test_v4r_recovered_ensemble -v

python scripts/build_ramp_rl_thesis_results.py --v4r --audit-only
```

The V4R tests pass without skips in the originating worktree where the ignored
models remain available. The prospective thesis audit currently reports pass
in the originating Windows checkout. Its raw-byte checks of
`canonical_evidence.json`, `source_freeze.json`, `recovery_binding.json`, and
`sealed_test_opening.json` are not cross-checkout portable, and it does not
verify the nested recovery-source report hash. These are publication-verifier
gaps to correct before final release; they do not change the supported headline
or invalidate the sealed evidence.
