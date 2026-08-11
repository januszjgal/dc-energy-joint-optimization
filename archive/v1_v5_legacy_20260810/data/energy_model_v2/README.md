# Energy Model v2 Data Contract

Energy model v2 is a controlled **May 2025 CAISO archetype**. Price and net
demand originate from one timestamped real market calendar and are shifted
together across market slots.

No synthetic fallback is permitted.

## Reference source window

The raw source window must include buffers around May 2025:

```text
2025-04-30 through 2025-06-02
```

Buffers allow IANA timezone conversion without circularly wrapping May 31 into
May 1.

## Required raw inputs

### Net demand

CAISO Today's Outlook historical five-minute files:

```text
https://www.caiso.com/outlook/history/YYYYMMDD/netdemand.csv
https://www.caiso.com/outlook/history/YYYYMMDD/fuelsource.csv
```

`netdemand.csv` supplies native five-minute CAISO net demand. `fuelsource.csv`
supplies native five-minute solar generation for the optional forecast feature.
The files use Pacific civil time and are converted with
`America/Los_Angeles`, preserving daylight-saving rules.

### Price

`2025/raw/caiso_dam_np15.csv`

Required schema:

```text
timestamp_utc,price_usd_mwh
```

The primary file contains real hourly CAISO day-ahead total LMP for NP15,
expanded as a step function to five-minute controller intervals. It documents:

- source URL/query;
- market (`DAM`);
- node or trading hub;
- native interval semantics;
- timestamp semantics (`interval_start`);
- timezone;
- units; and
- retrieval timestamp.

Source metadata belongs in:

```text
2025/raw/caiso_dam_np15.metadata.json
```

The build must fail if OASIS returns incomplete data. It never falls back to a
generated curve. Five-minute RTM price is future robustness work.

## Experiment calendar

Processed files use exactly:

```text
2025-05-01T07:00:00Z inclusive (May 1 00:00 PDT)
2025-06-01T07:00:00Z exclusive
5-minute cadence
8,928 rows
```

Google's May-2019 workload timestep 0 is anchored to May 1 00:00 Pacific on the
May-2025 experiment calendar. The workload extraction discarded the original
absolute timestamp and retained only ordinal timesteps, so this anchor is an
explicit modeling convention and cross-year counterfactual, not a
contemporaneous market replay.

## Market slots

### US

- `America/Los_Angeles`
- `America/Denver`
- `America/Chicago`
- `America/New_York`

### Global

- `America/Los_Angeles`
- `America/Chicago`
- `Europe/Amsterdam`
- `Asia/Singapore`

Each target UTC timestamp is converted to target local wall time. The reference
CAISO price/net-demand pair is then looked up at the identical Los Angeles wall
time. This shifts both signals together and preserves daylight-saving rules.

The primary model does **not** re-average regional price levels. Regional
price-level calibration and real multi-market inputs are future sensitivity
work.

## Processed schema

Each market-slot CSV contains:

```text
timestep
timestamp_utc
timestamp_local
reference_timestamp_utc
price_usd_kwh
net_demand_mw
net_demand_signed
solar_fraction
```

`net_demand_signed` is:

```text
net_demand_mw / max(abs(net_demand_mw))
```

where the denominator is the maximum absolute net demand in the Pacific
May-2025 reference window. The result is clipped only to `[-1, 1]`. Negative
midday oversupply therefore remains negative and observable. The convex
quadratic grid-stress term uses `max(net_demand_signed, 0)`; real low/negative
LMP supplies the economic incentive to execute during negative-demand troughs.

The primary reward does not contain a ramp derivative. Evaluation compares raw
`net_demand_mw` against `net_demand_mw + grid_mw` using per-region one-hour and
three-hour maximum/p95 upward-ramp KPIs. A ramp-aware reward is future work only
if the first v2 policies worsen those independent physical measures.

## Equal-capacity proxy contract

Every Google workload curve is already CPU usage divided by its own source
cell's CPU capacity. Energy model v2 maps each utilization shape onto an
equal-sized 100 MW proxy DC, so every scenario site declares:

```text
rated_power_mw: 100.0
capacity: 1.0
memory_capacity: 1.0
```

Raw machine totals remain provenance/calibration metadata and do not rescale
destination capacity a second time.

The scenario permits unrestricted service and batch routing among all four
slots. This defines an optimistic upper bound; the data contract does not imply
latency, residency, network-capacity, or movement feasibility.

## Review gate

Before any PPO retraining:

1. validate source coverage and hashes;
2. verify no missing/duplicate timestamps;
3. graph the full reference month;
4. graph average UTC profiles for every shifted slot;
5. report price/net-demand peak, trough, and correlation;
6. pass `scripts/preflight_energy_model_v2.py`, including measured-tier
   conservation and spatial/batch Status Quo neutrality;
7. freeze the primary objective and its coefficient sensitivity;
8. document that MPC is future work, routing is unrestricted, deadlines are
   experimental control semantics, and energy generalization is out of scope;
9. run spatial and spatial+temporal QP bounds; and
10. obtain explicit user approval.

The source, unit-capacity, measured-tier, signed-demand, current-arrival
observation, and Status Quo neutrality checks pass. PPO training has not begun.
