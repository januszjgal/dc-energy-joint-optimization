-- Five-minute CPU work of every top-level usage row, cells a-h, counting short
-- windows in proportion to the part of the bucket they cover. Paste into the
-- BigQuery console and check the byte estimate before running. It scans the same
-- columns as the workload query in extract_cells.ipynb.
-- Saved result: results/usage_all_windows.csv, in this folder.
--
-- average_usage.cpus is a rate over the row's own window, which is typically 300 s
-- but shorter if the instance starts, stops, or is updated. A row's work in a bucket
-- is therefore cpus x (seconds of the window inside the bucket) / 300. A row that
-- crosses a bucket boundary gives each bucket the part it holds. This is the same
-- work that audit_300s_filter.sql measures, with its start-time bucketing made exact.
--
-- service_full and batch_full keep only full-length rows, the previous extraction,
-- so the result can be checked against the committed curves. The row counts show
-- how many windows are short, cross a boundary, or break the 300 s maximum.
-- Result, 2026-11-03: 49.88 billion rows, none crossing a boundary or over 300 s.

WITH usage AS (
  SELECT 'a' AS cell, start_time, end_time, collection_id, average_usage.cpus AS cpus
  FROM `google.com:google-cluster-data`.clusterdata_2019_a.instance_usage
  WHERE alloc_collection_id IS NULL OR alloc_collection_id = 0
  UNION ALL
  SELECT 'b', start_time, end_time, collection_id, average_usage.cpus
  FROM `google.com:google-cluster-data`.clusterdata_2019_b.instance_usage
  WHERE alloc_collection_id IS NULL OR alloc_collection_id = 0
  UNION ALL
  SELECT 'c', start_time, end_time, collection_id, average_usage.cpus
  FROM `google.com:google-cluster-data`.clusterdata_2019_c.instance_usage
  WHERE alloc_collection_id IS NULL OR alloc_collection_id = 0
  UNION ALL
  SELECT 'd', start_time, end_time, collection_id, average_usage.cpus
  FROM `google.com:google-cluster-data`.clusterdata_2019_d.instance_usage
  WHERE alloc_collection_id IS NULL OR alloc_collection_id = 0
  UNION ALL
  SELECT 'e', start_time, end_time, collection_id, average_usage.cpus
  FROM `google.com:google-cluster-data`.clusterdata_2019_e.instance_usage
  WHERE alloc_collection_id IS NULL OR alloc_collection_id = 0
  UNION ALL
  SELECT 'f', start_time, end_time, collection_id, average_usage.cpus
  FROM `google.com:google-cluster-data`.clusterdata_2019_f.instance_usage
  WHERE alloc_collection_id IS NULL OR alloc_collection_id = 0
  UNION ALL
  SELECT 'g', start_time, end_time, collection_id, average_usage.cpus
  FROM `google.com:google-cluster-data`.clusterdata_2019_g.instance_usage
  WHERE alloc_collection_id IS NULL OR alloc_collection_id = 0
  UNION ALL
  SELECT 'h', start_time, end_time, collection_id, average_usage.cpus
  FROM `google.com:google-cluster-data`.clusterdata_2019_h.instance_usage
  WHERE alloc_collection_id IS NULL OR alloc_collection_id = 0
),
prio AS (
  SELECT 'a' AS cell, collection_id, MAX(priority) AS priority
  FROM `google.com:google-cluster-data`.clusterdata_2019_a.collection_events
  WHERE type = 0 GROUP BY collection_id
  UNION ALL
  SELECT 'b', collection_id, MAX(priority)
  FROM `google.com:google-cluster-data`.clusterdata_2019_b.collection_events
  WHERE type = 0 GROUP BY collection_id
  UNION ALL
  SELECT 'c', collection_id, MAX(priority)
  FROM `google.com:google-cluster-data`.clusterdata_2019_c.collection_events
  WHERE type = 0 GROUP BY collection_id
  UNION ALL
  SELECT 'd', collection_id, MAX(priority)
  FROM `google.com:google-cluster-data`.clusterdata_2019_d.collection_events
  WHERE type = 0 GROUP BY collection_id
  UNION ALL
  SELECT 'e', collection_id, MAX(priority)
  FROM `google.com:google-cluster-data`.clusterdata_2019_e.collection_events
  WHERE type = 0 GROUP BY collection_id
  UNION ALL
  SELECT 'f', collection_id, MAX(priority)
  FROM `google.com:google-cluster-data`.clusterdata_2019_f.collection_events
  WHERE type = 0 GROUP BY collection_id
  UNION ALL
  SELECT 'g', collection_id, MAX(priority)
  FROM `google.com:google-cluster-data`.clusterdata_2019_g.collection_events
  WHERE type = 0 GROUP BY collection_id
  UNION ALL
  SELECT 'h', collection_id, MAX(priority)
  FROM `google.com:google-cluster-data`.clusterdata_2019_h.collection_events
  WHERE type = 0 GROUP BY collection_id
),
cap AS (
  -- The same capacity the workload query divides by: each machine's largest
  -- reported CPU capacity, summed over all machines of the cell.
  SELECT cell, SUM(cpu_cap) AS cpu_capacity
  FROM (
    SELECT 'a' AS cell, machine_id, MAX(capacity.cpus) AS cpu_cap
    FROM `google.com:google-cluster-data`.clusterdata_2019_a.machine_events GROUP BY machine_id
    UNION ALL
    SELECT 'b', machine_id, MAX(capacity.cpus)
    FROM `google.com:google-cluster-data`.clusterdata_2019_b.machine_events GROUP BY machine_id
    UNION ALL
    SELECT 'c', machine_id, MAX(capacity.cpus)
    FROM `google.com:google-cluster-data`.clusterdata_2019_c.machine_events GROUP BY machine_id
    UNION ALL
    SELECT 'd', machine_id, MAX(capacity.cpus)
    FROM `google.com:google-cluster-data`.clusterdata_2019_d.machine_events GROUP BY machine_id
    UNION ALL
    SELECT 'e', machine_id, MAX(capacity.cpus)
    FROM `google.com:google-cluster-data`.clusterdata_2019_e.machine_events GROUP BY machine_id
    UNION ALL
    SELECT 'f', machine_id, MAX(capacity.cpus)
    FROM `google.com:google-cluster-data`.clusterdata_2019_f.machine_events GROUP BY machine_id
    UNION ALL
    SELECT 'g', machine_id, MAX(capacity.cpus)
    FROM `google.com:google-cluster-data`.clusterdata_2019_g.machine_events GROUP BY machine_id
    UNION ALL
    SELECT 'h', machine_id, MAX(capacity.cpus)
    FROM `google.com:google-cluster-data`.clusterdata_2019_h.machine_events GROUP BY machine_id
  )
  GROUP BY cell
),
labeled AS (
  SELECT
    u.cell, u.start_time, u.end_time, u.cpus,
    IFNULL(p.priority <= 115, FALSE) AS is_batch,  -- no matching priority -> service
    CAST(FLOOR(u.start_time / 300000000) AS INT64) AS first_bucket
  FROM usage u
  LEFT JOIN prio p USING (cell, collection_id)
),
pieces AS (
  SELECT
    cell, is_batch, step,
    first_bucket + step AS time_bucket,
    end_time - start_time AS dur_us,
    IFNULL(cpus, 0)
      * GREATEST(0, LEAST(end_time, (first_bucket + step + 1) * 300000000)
                    - GREATEST(start_time, (first_bucket + step) * 300000000))
      / 300000000 AS work
  FROM labeled
  CROSS JOIN UNNEST(IF(end_time > (first_bucket + 1) * 300000000, [0, 1], [0])) AS step
)
SELECT
  p.cell,
  p.time_bucket,
  ROUND(c.cpu_capacity, 8)                                       AS cpu_capacity,
  ROUND(SUM(IF(NOT p.is_batch, p.work, 0)), 6)                   AS service_all,
  ROUND(SUM(IF(p.is_batch, p.work, 0)), 6)                       AS batch_all,
  ROUND(SUM(IF(NOT p.is_batch AND p.dur_us >= 300000000, p.work, 0)), 6) AS service_full,
  ROUND(SUM(IF(p.is_batch AND p.dur_us >= 300000000, p.work, 0)), 6)     AS batch_full,
  ROUND(SUM(IF(p.step = 1, p.work, 0)), 6)                       AS work_from_previous_bucket,
  COUNTIF(p.step = 0)                                            AS rows_all,
  COUNTIF(p.step = 0 AND p.dur_us >= 300000000)                  AS rows_full,
  COUNTIF(p.step = 1)                                            AS rows_crossing,
  COUNTIF(p.step = 0 AND p.dur_us > 300000000)                   AS rows_over_300s,
  COUNTIF(p.step = 0 AND p.dur_us <= 0)                          AS rows_nonpositive
FROM pieces p
JOIN cap c USING (cell)
GROUP BY p.cell, p.time_bucket, c.cpu_capacity
ORDER BY p.cell, p.time_bucket;
