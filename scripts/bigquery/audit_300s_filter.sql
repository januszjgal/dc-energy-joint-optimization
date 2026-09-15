-- Audit of the >= 300 s instance_usage filter used by extract_tier_curves.ipynb,
-- cells a-d. Paste into the BigQuery console and run each query separately.
-- Saved results: results/query1_filter_volume.csv (Query 1) and
-- results/query2_hourly_shape.csv (Query 2), next to this file.
-- Check the console's byte estimate first; both scan the same columns as the
-- tier-curve extraction.
--
-- CPU work of a row = average_usage.cpus x (end_time - start_time), in
-- NCU-seconds. Google defines average_usage over the window's duration, so
-- this is the row's executed CPU. Top-level filter and priority join are
-- identical to the extraction; unmatched priorities count as service.


-- ============================================================================
-- QUERY 1: how much the filter removes, by cell x class x instance kind.
-- Rows with class = 'ALL' / kind = 'ALL' are ROLLUP subtotals.
-- ============================================================================
WITH usage AS (
  SELECT 'a' AS cell, start_time, end_time, collection_id, alloc_collection_id,
         collection_type, average_usage.cpus AS cpus
  FROM `google.com:google-cluster-data`.clusterdata_2019_a.instance_usage
  UNION ALL
  SELECT 'b', start_time, end_time, collection_id, alloc_collection_id,
         collection_type, average_usage.cpus
  FROM `google.com:google-cluster-data`.clusterdata_2019_b.instance_usage
  UNION ALL
  SELECT 'c', start_time, end_time, collection_id, alloc_collection_id,
         collection_type, average_usage.cpus
  FROM `google.com:google-cluster-data`.clusterdata_2019_c.instance_usage
  UNION ALL
  SELECT 'd', start_time, end_time, collection_id, alloc_collection_id,
         collection_type, average_usage.cpus
  FROM `google.com:google-cluster-data`.clusterdata_2019_d.instance_usage
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
),
r AS (
  SELECT
    u.cell,
    CASE WHEN p.priority IS NULL THEN 'service_unmatched'
         WHEN p.priority <= 115 THEN 'batch'
         ELSE 'service' END AS class,
    -- collection_type: 0 = job (task), 1 = alloc set (alloc instance)
    CASE CAST(u.collection_type AS STRING)
         WHEN '0' THEN 'job_task'
         WHEN '1' THEN 'alloc_instance'
         ELSE 'other' END AS kind,
    u.end_time - u.start_time AS dur_us,
    MOD(u.start_time, 300000000) AS grid_offset_us,
    IFNULL(u.cpus, 0) * GREATEST(u.end_time - u.start_time, 0) / 1e6 AS work,
    (u.end_time - u.start_time) >= 300000000 AS kept
  FROM usage u
  LEFT JOIN prio p USING (cell, collection_id)
  WHERE u.alloc_collection_id IS NULL OR u.alloc_collection_id = 0
)
SELECT
  -- output aliases must differ from the grouping columns, or GROUP BY
  -- resolves to the alias (which contains GROUPING) and fails
  IF(GROUPING(cell) = 1, 'ALL', cell) AS cell_label,
  IF(GROUPING(class) = 1, 'ALL', class) AS class_label,
  IF(GROUPING(kind) = 1, 'ALL', kind) AS kind_label,

  COUNT(*)                                                AS rows_all,
  COUNTIF(NOT kept)                                       AS rows_dropped,
  ROUND(100 * SAFE_DIVIDE(COUNTIF(NOT kept), COUNT(*)), 2) AS pct_rows_dropped,

  ROUND(SUM(work) / 3600, 0)                              AS work_all_ncu_h,
  ROUND(SUM(IF(NOT kept, work, 0)) / 3600, 0)             AS work_dropped_ncu_h,
  ROUND(100 * SAFE_DIVIDE(SUM(IF(NOT kept, work, 0)), SUM(work)), 3)
                                                          AS pct_work_dropped,

  -- where the dropped work sits (percent of ALL work in the group)
  ROUND(100 * SAFE_DIVIDE(SUM(IF(dur_us < 60000000, work, 0)), SUM(work)), 3)
                                                          AS pct_work_in_lt60s,
  ROUND(100 * SAFE_DIVIDE(SUM(IF(dur_us >= 60000000 AND dur_us < 300000000,
                                 work, 0)), SUM(work)), 3)
                                                          AS pct_work_in_60_299s,

  -- sanity checks on the retained rows
  COUNTIF(dur_us <= 0)                                    AS rows_nonpositive_dur,
  COUNTIF(dur_us > 300000000)                             AS rows_longer_than_300s,
  -- share of retained work whose window does not start on the 5-min grid
  -- (such rows straddle two of the extraction's start_time buckets)
  ROUND(100 * SAFE_DIVIDE(SUM(IF(kept AND grid_offset_us != 0, work, 0)),
                          SUM(IF(kept, work, 0))), 2)     AS pct_kept_work_off_grid
FROM r
GROUP BY ROLLUP (cell, class, kind)
ORDER BY cell_label, class_label, kind_label;


-- ============================================================================
-- QUERY 2: does the filter change the hourly SHAPE (level, ramps, peak)?
-- Compares hourly retained work with hourly all-row work per cell x class.
-- Rows go to the hour of start_time, as in the extraction (windows are
-- <= 300 s, so at most 5 min spills into the next hour). The first and last
-- hour of each cell are dropped as partial.
-- To export the hourly series instead, replace the final SELECT with
--   SELECT * FROM hourly ORDER BY cell, class, hour
-- ============================================================================
WITH usage AS (
  SELECT 'a' AS cell, start_time, end_time, collection_id, alloc_collection_id,
         average_usage.cpus AS cpus
  FROM `google.com:google-cluster-data`.clusterdata_2019_a.instance_usage
  UNION ALL
  SELECT 'b', start_time, end_time, collection_id, alloc_collection_id,
         average_usage.cpus
  FROM `google.com:google-cluster-data`.clusterdata_2019_b.instance_usage
  UNION ALL
  SELECT 'c', start_time, end_time, collection_id, alloc_collection_id,
         average_usage.cpus
  FROM `google.com:google-cluster-data`.clusterdata_2019_c.instance_usage
  UNION ALL
  SELECT 'd', start_time, end_time, collection_id, alloc_collection_id,
         average_usage.cpus
  FROM `google.com:google-cluster-data`.clusterdata_2019_d.instance_usage
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
),
x AS (
  SELECT
    u.cell,
    IF(p.priority <= 115, 'batch', 'service') AS class,  -- NULL -> service
    DIV(u.start_time, 3600000000) AS hour,
    IFNULL(u.cpus, 0) * GREATEST(u.end_time - u.start_time, 0) / 1e6 AS work,
    (u.end_time - u.start_time) >= 300000000 AS kept
  FROM usage u
  LEFT JOIN prio p USING (cell, collection_id)
  WHERE u.alloc_collection_id IS NULL OR u.alloc_collection_id = 0
),
rolled AS (
  SELECT
    cell, hour,
    IF(GROUPING(class) = 1, 'total', class) AS class_label,
    GROUPING(hour) AS g_hour,
    SUM(work) / 3600 AS ncu_all,               -- mean NCUs over the hour
    SUM(IF(kept, work, 0)) / 3600 AS ncu_kept
  FROM x
  GROUP BY ROLLUP (cell, hour, class)
),
agg AS (
  -- keep cell x hour x class rows and cell x hour totals only
  SELECT cell, hour, class_label AS class, ncu_all, ncu_kept
  FROM rolled
  WHERE cell IS NOT NULL AND g_hour = 0
),
bounds AS (
  SELECT cell, MIN(hour) AS h0, MAX(hour) AS h1 FROM agg GROUP BY cell
),
hourly AS (
  SELECT
    a.cell, a.class, a.hour, a.ncu_all, a.ncu_kept,
    a.ncu_all  - LAG(a.ncu_all)  OVER w AS ramp_all,
    a.ncu_kept - LAG(a.ncu_kept) OVER w AS ramp_kept
  FROM agg a JOIN bounds b USING (cell)
  WHERE a.hour > b.h0 AND a.hour < b.h1
  WINDOW w AS (PARTITION BY a.cell, a.class ORDER BY a.hour)
)
SELECT
  cell, class,
  COUNT(*) AS n_hours,
  ROUND(100 * (1 - SUM(ncu_kept) / SUM(ncu_all)), 3) AS pct_work_dropped,

  -- hour-to-hour variation of the dropped share (uniform => shape preserved)
  ROUND(100 * APPROX_QUANTILES(1 - SAFE_DIVIDE(ncu_kept, ncu_all), 20)[OFFSET(1)], 3)
                                                          AS dropped_pct_p05,
  ROUND(100 * APPROX_QUANTILES(1 - SAFE_DIVIDE(ncu_kept, ncu_all), 20)[OFFSET(10)], 3)
                                                          AS dropped_pct_p50,
  ROUND(100 * APPROX_QUANTILES(1 - SAFE_DIVIDE(ncu_kept, ncu_all), 20)[OFFSET(19)], 3)
                                                          AS dropped_pct_p95,
  ROUND(100 * MAX(1 - SAFE_DIVIDE(ncu_kept, ncu_all)), 3) AS dropped_pct_max,

  ROUND(CORR(ncu_kept, ncu_all), 5)                       AS corr_level,
  ROUND(CORR(ramp_kept, ramp_all), 5)                     AS corr_1h_ramp,

  -- relative 1 h ramp size and peak-to-mean, retained vs all rows
  ROUND(AVG(ABS(ramp_kept)) / AVG(ncu_kept), 5)           AS rel_ramp_kept,
  ROUND(AVG(ABS(ramp_all))  / AVG(ncu_all), 5)            AS rel_ramp_all,
  ROUND(MAX(ncu_kept) / AVG(ncu_kept), 4)                 AS peak_to_mean_kept,
  ROUND(MAX(ncu_all)  / AVG(ncu_all), 4)                  AS peak_to_mean_all,
  ARRAY_AGG(hour ORDER BY ncu_kept DESC LIMIT 1)[OFFSET(0)] AS peak_hour_kept,
  ARRAY_AGG(hour ORDER BY ncu_all  DESC LIMIT 1)[OFFSET(0)] AS peak_hour_all
FROM hourly
GROUP BY cell, class
ORDER BY cell, class;
