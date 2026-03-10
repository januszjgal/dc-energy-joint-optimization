- Cell a: Machines -
  10001 machines
  CPU  — unique types: 6, range: [0.3867, 1.0000]
  Mem  — unique types: 6, range: [0.1667, 1.0000]

- Cell b: Machines -
  10047 machines
  CPU  — unique types: 5, range: [0.3867, 1.0000]
  Mem  — unique types: 5, range: [0.1667, 1.0000]

- Cell c: Machines -
  13245 machines
  CPU  — unique types: 6, range: [0.2593, 1.0000]
  Mem  — unique types: 6, range: [0.1667, 1.0000]

- Cell d: Machines -
  12576 machines
  CPU  — unique types: 5, range: [0.2593, 1.0000]
  Mem  — unique types: 4, range: [0.1667, 0.6670]

✅ 45869 total machines across all cells


- Cell a: Job Metadata -
  4763884 jobs — 205060 batch (4.3%), 4558824 service (95.7%)
  Completed: 1199850 — duration median=30s, mean=1391s

- Cell b: Job Metadata -
  1041498 jobs — 503599 batch (48.4%), 537899 service (51.6%)
  Completed: 534923 — duration median=207s, mean=3186s

- Cell c: Job Metadata -
  6453721 jobs — 836079 batch (13.0%), 5617642 service (87.0%)
  Completed: 4712488 — duration median=242s, mean=2090s

- Cell d: Job Metadata -
  1783250 jobs — 359262 batch (20.1%), 1423988 service (79.9%)
  Completed: 865097 — duration median=142s, mean=5153s

✅ Job metadata done!

==================================================
Cell a: Fitting batch job distributions
==================================================
  Total batch jobs: 205060
  Completed batch jobs: 102364

  Inter-arrival times:
    Best fit: expon (KS p=0.0000)
    Mean=26.2s, Median=7.6s

  Job durations:
    Best fit: expon (KS p=0.0000)
    Mean=2106s, Median=107s

  CPU request per task:
    Best fit: expon (KS p=0.0000)
    Mean=0.0096, Median=0.0060

  Memory request per task:
    Best fit: expon (KS p=0.0000)
    Mean=0.0068, Median=0.0047

  Tasks per job:
    Best fit: expon (KS p=0.0000)
    Mean=7.6, Median=1.0

  Workload mix: 4.3% batch

==================================================
Cell b: Fitting batch job distributions
==================================================
  Total batch jobs: 503599
  Completed batch jobs: 166830

  Inter-arrival times:
    Best fit: expon (KS p=0.0000)
    Mean=16.1s, Median=5.1s

  Job durations:
    Best fit: expon (KS p=0.0000)
    Mean=4790s, Median=236s

  CPU request per task:
    Best fit: expon (KS p=0.0000)
    Mean=0.0098, Median=0.0066

  Memory request per task:
    Best fit: expon (KS p=0.0000)
    Mean=0.0070, Median=0.0051

  Tasks per job:
    Best fit: expon (KS p=0.0000)
    Mean=30.1, Median=1.0

  Workload mix: 48.4% batch

==================================================
Cell c: Fitting batch job distributions
==================================================
  Total batch jobs: 836079
  Completed batch jobs: 445251

  Inter-arrival times:
    Best fit: expon (KS p=0.0000)
    Mean=6.0s, Median=2.2s

  Job durations:
    Best fit: expon (KS p=0.0000)
    Mean=5714s, Median=79s

  CPU request per task:
    Best fit: expon (KS p=0.0000)
    Mean=0.0104, Median=0.0054

  Memory request per task:
    Best fit: expon (KS p=0.0000)
    Mean=0.0081, Median=0.0052

  Tasks per job:
    Best fit: expon (KS p=0.0000)
    Mean=8.1, Median=1.0

  Workload mix: 13.0% batch

==================================================
Cell d: Fitting batch job distributions
==================================================
  Total batch jobs: 359262
  Completed batch jobs: 145446

  Inter-arrival times:
    Best fit: expon (KS p=0.0000)
    Mean=18.4s, Median=2.9s

  Job durations:
    Best fit: expon (KS p=0.0000)
    Mean=3626s, Median=154s

  CPU request per task:
    Best fit: expon (KS p=0.0000)
    Mean=0.0080, Median=0.0041

  Memory request per task:
    Best fit: expon (KS p=0.0000)
    Mean=0.0069, Median=0.0032

  Tasks per job:
    Best fit: expon (KS p=0.0000)
    Mean=6.5, Median=1.0

  Workload mix: 20.1% batch

✅ Distribution fitting done!



Saved data/workload_generator_params.json

Primary cell (a) generator parameters:
  Inter-arrival: expon (mean=26.2s)
  Duration:      expon (mean=2106s)
  CPU request:   expon (mean=0.0096)
  Tasks/job:     expon (mean=7.6)
  Batch fraction: 4.3%

Cross-cell validation:
  Cell a: batch=4.3%, dur_mean=2106s, iat_mean=26.2s
  Cell b: batch=48.4%, dur_mean=4790s, iat_mean=16.1s
  Cell c: batch=13.0%, dur_mean=5714s, iat_mean=6.0s
  Cell d: batch=20.1%, dur_mean=3626s, iat_mean=18.4s

✅ Workload generator params done!


EXTRACTION SUMMARY

Cell a:
  Utilization (5min)             8,929 rows
  Machines                      10,001 rows
  Jobs (all)                 4,763,884 rows
  Batch distributions              2.2 KB

Cell b:
  Utilization (5min)             8,929 rows
  Machines                      10,047 rows
  Jobs (all)                 1,041,498 rows
  Batch distributions              2.3 KB

Cell c:
  Utilization (5min)             8,929 rows
  Machines                      13,245 rows
  Jobs (all)                 6,453,721 rows
  Batch distributions              2.2 KB

Cell d:
  Utilization (5min)             8,929 rows
  Machines                      12,576 rows
  Jobs (all)                 1,783,250 rows
  Batch distributions              2.2 KB

Global files:
  Power model                      0.2 KB
  All machines                  45,869 rows
  Generator params                 3.3 KB
  Power scatter                  2,980 rows
  Summary plot                   271.0 KB


BENCHMARK PAPER MAPPING
-

Grange et al. (2018) — Batch scheduling + renewable awareness:
  → workload_generator_params.json  (generate synthetic batch jobs)
  → cells/cell_*.csv               (aggregate demand for capacity)
  → power_model_params.json        (energy cost model)
  → YOU ADD: solar trace from NREL/PVGIS + electricity prices from ComEd/IESO
  → Deadline formula: submit_time + duration × (1 + flexibility_factor)

Xu et al. (2020) — Self-adaptive brownout + batch deferral:
  → jobs/jobs_*.csv                (batch/service classification + resource requests)
  → cells/cell_*.csv               (utilization time series for brownout triggers)
  → power_model_params.json        (energy model)
  → YOU ADD: renewable energy trace + brown/green energy pricing

Haghshenas et al. (2022) — Infrastructure-aware heterogeneous scheduling:
  → machines/machines_all.csv      (heterogeneous server fleet)
  → jobs/jobs_*.csv                (heterogeneous workload mix)
  → cells/cell_*.csv               (cooling model input — utilization drives heat)
  → power_model_params.json        (per-server-type power model)
  → YOU ADD: cooling power model + tiered electricity rate structure