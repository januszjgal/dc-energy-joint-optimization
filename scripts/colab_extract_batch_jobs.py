"""
Google Colab script: Extract batch job data from Google ClusterData 2019.

Run this entire file as a single Colab cell (or split at the marked sections).
Downloads the output CSVs to your local machine automatically.

Prerequisites: A GCP project with BigQuery API enabled.
"""

# ============================================================
# CELL 1: Setup and authenticate
# ============================================================
# !pip install google-cloud-bigquery pyarrow db-dtypes

from google.colab import auth
auth.authenticate_user()

# >>> SET YOUR GCP PROJECT ID HERE <<<
PROJECT_ID = "your-gcp-project-id"  # <-- CHANGE THIS

from google.cloud import bigquery
client = bigquery.Client(project=PROJECT_ID)

# ============================================================
# CELL 2: Define extraction query
# ============================================================

CELLS = ["a", "b", "c", "d"]

def build_query(cell: str) -> str:
    """Build the BigQuery SQL for one cell.

    Extracts deferrable jobs with their timing, resource requests, and task
    counts. "Deferrable" = the Borg NO-SLO tiers: free (priority <= 99) and
    best-effort batch / beb (110-115) — both run without SLOs (Tirmazi et al.
    2020, "Borg: the Next Generation", EuroSys '20, §2). Strict beb alone is a
    negligible share of these cells' load, so the deferrable class is the union
    of the two no-SLO tiers. We classify by PRIORITY, not scheduling_class:
    scheduling_class (latency-sensitivity) is orthogonal to tier, and
    `scheduling_class <= 1` is dominated by latency-insensitive *production*
    (priority 200) jobs, which are SLO-bound and must not be deferred.
    Caps at 200k jobs per cell (more than enough for distribution fitting).
    """
    dataset = f"`google.com:google-cluster-data`.clusterdata_2019_{cell}"

    return f"""
    WITH batch_jobs AS (
      -- Get submit and terminal times for best-effort batch (beb) jobs.
      -- Event types: 4=EVICT, 5=FAIL, 6=FINISH, 7=KILL, 8=LOST (FINISH is 6).
      SELECT
        collection_id,
        MIN(IF(type = 0, time, NULL)) AS submit_time_us,
        -- Last terminal event (MAX time); MAX(type) prefers FINISH(6) as outcome
        MAX(IF(type IN (4, 5, 6, 7, 8), time, NULL)) AS end_time_us,
        MAX(IF(type IN (4, 5, 6, 7, 8), type, NULL)) AS terminal_type,
        ANY_VALUE(scheduling_class) AS scheduling_class,
        ANY_VALUE(priority) AS priority
      FROM {dataset}.collection_events
      WHERE (priority <= 99 OR priority BETWEEN 110 AND 115)  -- no-SLO tiers: free + beb; Tirmazi 2020 §2
        AND collection_type = 0            -- jobs (not alloc sets)
      GROUP BY collection_id
      HAVING submit_time_us IS NOT NULL
        AND end_time_us IS NOT NULL
        AND end_time_us > submit_time_us
    ),

    job_resources AS (
      -- Get per-job aggregated resource requests from instance SUBMIT events
      SELECT
        collection_id,
        COUNT(DISTINCT instance_index) AS num_tasks,
        AVG(resource_request.cpus) AS avg_cpu_request,
        AVG(resource_request.memory) AS avg_mem_request,
        SUM(resource_request.cpus) AS total_cpu_request,
        SUM(resource_request.memory) AS total_mem_request
      FROM {dataset}.instance_events
      WHERE type = 0                     -- SUBMIT events only
        AND collection_id IN (SELECT collection_id FROM batch_jobs)
      GROUP BY collection_id
    )

    SELECT
      bj.collection_id,
      bj.submit_time_us,
      bj.end_time_us,
      -- Duration in seconds (original is microseconds)
      (bj.end_time_us - bj.submit_time_us) / 1000000.0 AS duration_sec,
      bj.scheduling_class,
      bj.priority,
      bj.terminal_type,
      jr.num_tasks,
      jr.avg_cpu_request,
      jr.avg_mem_request,
      jr.total_cpu_request,
      jr.total_mem_request
    FROM batch_jobs bj
    JOIN job_resources jr USING (collection_id)
    ORDER BY bj.submit_time_us
    LIMIT 200000
    """


# ============================================================
# CELL 3: Run queries and save CSVs
# ============================================================
import pandas as pd

total_jobs_per_cell = {}
batch_jobs_per_cell = {}

for cell in CELLS:
    print(f"\n{'='*60}")
    print(f"Extracting cell {cell}...")
    print(f"{'='*60}")

    query = build_query(cell)
    df = client.query(query).to_dataframe()
    print(f"  Retrieved {len(df)} batch jobs")

    # Basic stats
    print(f"  Duration (sec):  median={df['duration_sec'].median():.1f}, "
          f"mean={df['duration_sec'].mean():.1f}, "
          f"p95={df['duration_sec'].quantile(0.95):.1f}")
    print(f"  CPU request:     median={df['avg_cpu_request'].median():.6f}, "
          f"mean={df['avg_cpu_request'].mean():.6f}")
    print(f"  Memory request:  median={df['avg_mem_request'].median():.6f}, "
          f"mean={df['avg_mem_request'].mean():.6f}")
    print(f"  Tasks per job:   median={df['num_tasks'].median():.0f}, "
          f"mean={df['num_tasks'].mean():.1f}, "
          f"max={df['num_tasks'].max()}")

    # Compute inter-arrival times
    sorted_submit = df['submit_time_us'].sort_values().values
    inter_arrivals_sec = pd.Series(
        (sorted_submit[1:] - sorted_submit[:-1]) / 1e6
    )
    print(f"  Inter-arrival (sec): median={inter_arrivals_sec.median():.2f}, "
          f"mean={inter_arrivals_sec.mean():.2f}")

    # Save to CSV (matching the column format the refit script expects)
    filename = f"batch_raw_{cell}.csv"
    df.to_csv(filename, index=False)
    print(f"  Saved {filename} ({len(df)} rows)")

    batch_jobs_per_cell[cell] = len(df)

    # Also get total job count for batch_fraction calculation
    dataset = f"`google.com:google-cluster-data`.clusterdata_2019_{cell}"
    total_query = f"""
    SELECT COUNT(DISTINCT collection_id) as total_jobs
    FROM {dataset}.collection_events
    WHERE collection_type = 0
    """
    total_df = client.query(total_query).to_dataframe()
    total_jobs_per_cell[cell] = int(total_df['total_jobs'].iloc[0])
    print(f"  Total jobs in cell: {total_jobs_per_cell[cell]}")
    print(f"  Batch fraction: {len(df) / total_jobs_per_cell[cell]:.5f}")


# ============================================================
# CELL 4: Print summary and download files
# ============================================================
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
for cell in CELLS:
    total = total_jobs_per_cell[cell]
    batch = batch_jobs_per_cell[cell]
    print(f"  Cell {cell}: {batch:,} batch / {total:,} total = "
          f"{batch/total:.4%} batch fraction")

# Download CSVs to your local machine
from google.colab import files
for cell in CELLS:
    files.download(f"batch_raw_{cell}.csv")

print("\nDone! Place the downloaded CSVs into data/jobs/ in the thesis repo.")
