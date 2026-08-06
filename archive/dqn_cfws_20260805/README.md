# DQN / CFWS Historical Archive

This directory contains the discrete-action DQN and CFWS-inspired experiments
removed from the active thesis workflow on 2026-08-05.

The active paper now asks a narrower Master's-level question: whether a frozen
continuous-action PPO policy improves the held-out objective versus a
grid-unaware local/immediate Status Quo under a symmetric a-d/e-h two-fold
cross-cell protocol. Round Robin and Drain Immediately remain sanity/ablation
comparators; the clairvoyant QP remains a lower-bound/headroom diagnostic.

Contents preserve their original repository-relative paths:

- `train_dqn.py`, discrete wrappers, CFWS smoke/sweep code;
- mixed historical sweep/review orchestrators that could automatically launch
  DQN jobs;
- DQN/compact model ZIPs and DQN sweep/training logs.

These artifacts are retained for provenance only. They are not active
baselines, are not run by `scripts/run_oof_campaign.py`, and must not be
presented as a CFWS reproduction or PPO-vs-CFWS comparison.
