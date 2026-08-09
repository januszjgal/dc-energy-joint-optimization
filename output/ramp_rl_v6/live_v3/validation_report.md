# Immutable pure-RL v3 validation result

Protocol `v6-ramp-pure-rl-v1-replication-earlystop-v3` replicated original v1 PPO at the 100k nominal / 110,592 complete-boundary stop on fresh seeds 2801-2805.

| Seed | Raw ramp impact | Cost ratio | Strict pass | Failed gates |
|---:|---:|---:|---|---|
| 2801 | -1.36279664544e-05 | 0.9919667653 | True | - |
| 2802 | -1.38844644958e-05 | 0.9861848878 | True | - |
| 2803 | -1.35565626081e-05 | 0.9820278853 | True | - |
| 2804 | -1.35847085141e-05 | 0.9803329078 | True | - |
| 2805 | -1.39008972319e-05 | 0.9913122200 | False | every_market_ramp_improves |

All-five selection decision: **False**.
Sealed March-April test opened: **false**.
No follow-on protocol was launched automatically.

The protocol is rejected because the all-five rule is conjunctive and seed
2805 failed `every_market_ramp_improves`. Its MISO Minnesota Hub raw macro
incremental ramp impact was `+1.2940080141785231e-07`; all other markets for
that seed were negative. Seed 2805 passed every safety, completion, cost,
emergency, behavior, and leakage check. No tolerance or aggregate override was
applied.

The five-seed reporting-only mean was `-1.371091986086174e-05` at mean cost
ratio `0.9863649332472247`, with four strict passes. This was 2.7089219141%
more ramp-improvement magnitude than the lucky v1 100k three-seed screen
(`-1.3349297807182322e-05`, cost `0.9801356986033037`), but it does not satisfy
the per-seed selection rule. V2-A averaged `-4.983717818021212e-07` at cost
`1.0091997163325592`; v2-B averaged `-1.412252682718701e-06` at cost
`0.9934819185990135`. Both v2 candidates also failed strict fresh-seed
validation.

Source/protocol commits are `f6cd9c9` and `dede685`. The runtime source bundle
hash is `8c4c4c468251bd1d54c230fc9be7b1882ab9b43eeb15c40ccd645f234bbd9fe0`;
the protocol composite hash is
`f692364cd68a6da4228eec25fe246532051cb332f5c22f139f361349e30662df`.
Repository-blob equivalence and deterministic behavior equivalence to
`b1bb302` are both true. Each seed used 110,592 interactions and 540 PPO
updates from a distinct random initialization.

Training manifests are under
`models/ramp_rl_v6/live_v3/confirmation/ppo/<seed>/training_manifest.json`.
Per-seed February evidence is under
`output/ramp_rl_v6/live_v3/validation/ppo_<seed>_validation.json`; the immutable
decision is `output/ramp_rl_v6/live_v3/validation_decision.json`. Because v3
failed validation, the sealed March-April test, 1 GW-total robustness,
c-h-overlapping robustness, and post-selection canonical evidence were not
run.
