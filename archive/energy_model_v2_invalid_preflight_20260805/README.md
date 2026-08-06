# Energy Model v2 — Invalidated Preflight Gate

These JSON artifacts were generated before the Status Quo locality correction.
The former batch baseline used total-workload shares for both service and batch
placement, unintentionally moving held-out load across sites.

After correction, the a–d scenarios reproduce spatial and batch Status Quo cost
to numerical precision, but the e–h scenarios expose inconsistent capacity
units: workload is normalized by each source cell's capacity while destination
capacity is normalized again against the largest fleet. Cells e and g exceed
their modeled local capacities in 697 and 307 intervals.

The archived QP percentages and baseline totals are invalid and must not be
cited. Regenerate them only after `scripts/preflight_energy_model_v2.py` passes.

`pretraining-readiness-audit.html` is the visual snapshot from that blocked
state. The active unit-capacity/signed-demand gate supersedes it.
