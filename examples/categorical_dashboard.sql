-- =================================================================
-- Categorical Evaluation Dashboard Queries
-- =================================================================
--
-- HARD RULE: All dashboards and alerts must query
-- `categorical_results_latest`, not the raw `categorical_results`
-- table.  The raw table is append-only and will contain duplicate
-- rows from retries and overlapping scheduled runs.
--
-- The dedup view uses ROW_NUMBER() to keep only the latest
-- classification per (session_id, metric_name, prompt_version).
--
-- Prerequisites:
--   bq-agent-sdk categorical-views \
--     --project-id=PROJECT --dataset-id=DATASET
--
-- =================================================================


-- -----------------------------------------------------------------
-- 1. Category Trend Over Time (daily)
-- -----------------------------------------------------------------
-- Use in a stacked bar or line chart to track category shifts.

SELECT
  eval_date,
  metric_name,
  category,
  session_count
FROM `PROJECT.DATASET.categorical_daily_counts`
WHERE metric_name = 'tone'
ORDER BY eval_date, category;


-- -----------------------------------------------------------------
-- 2. Recent Session Monitoring (last 1 hour)
-- -----------------------------------------------------------------
-- Quick check for operational dashboards showing live evaluation
-- results.  Uses the hourly pre-aggregated view.

SELECT
  eval_hour,
  metric_name,
  category,
  session_count
FROM `PROJECT.DATASET.categorical_hourly_counts`
WHERE eval_hour >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR)
ORDER BY eval_hour DESC, metric_name, category;


-- -----------------------------------------------------------------
-- 3. Failure / Escalation Drill-Down
-- -----------------------------------------------------------------
-- Find sessions classified into a specific failure category for
-- root-cause investigation.

SELECT
  session_id,
  metric_name,
  category,
  justification,
  created_at
FROM `PROJECT.DATASET.categorical_results_latest`
WHERE category = 'escalation'
  AND metric_name = 'outcome'
ORDER BY created_at DESC
LIMIT 50;


-- -----------------------------------------------------------------
-- 4. Alerting Threshold Queries
-- -----------------------------------------------------------------
-- Example: alert if parse_error_rate exceeds 10% on any day.
-- Wire this into a scheduled query or monitoring tool.

SELECT
  eval_date,
  execution_mode,
  endpoint,
  parse_errors,
  total,
  parse_error_rate
FROM `PROJECT.DATASET.categorical_operational_metrics`
WHERE parse_error_rate > 0.10
  AND eval_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
ORDER BY eval_date DESC;


-- Example: alert if a specific category proportion changes sharply.
-- Compare today vs. the 7-day rolling average.

WITH daily AS (
  SELECT
    eval_date,
    category,
    session_count
  FROM `PROJECT.DATASET.categorical_daily_counts`
  WHERE metric_name = 'tone'
    AND eval_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 8 DAY)
),
with_avg AS (
  SELECT
    eval_date,
    category,
    session_count,
    AVG(session_count) OVER (
      PARTITION BY category
      ORDER BY eval_date
      ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
    ) AS rolling_avg_7d
  FROM daily
)
SELECT *
FROM with_avg
WHERE eval_date = CURRENT_DATE()
  AND session_count > rolling_avg_7d * 1.5  -- 50% spike threshold
ORDER BY category;


-- -----------------------------------------------------------------
-- 5. Prompt Version Comparison
-- -----------------------------------------------------------------
-- Compare category distributions across prompt versions to
-- evaluate prompt changes.

SELECT
  prompt_version,
  metric_name,
  category,
  COUNT(*) AS session_count,
  SAFE_DIVIDE(
    COUNT(*),
    SUM(COUNT(*)) OVER (PARTITION BY prompt_version, metric_name)
  ) AS category_pct
FROM `PROJECT.DATASET.categorical_results_latest`
WHERE prompt_version IS NOT NULL
GROUP BY 1, 2, 3
ORDER BY metric_name, prompt_version, category;


-- =================================================================
-- Scheduled Micro-Batch Pattern
-- =================================================================
--
-- The categorical evaluator supports narrow time windows for
-- near-real-time evaluation via `--last` + `--persist`:
--
--   bq-agent-sdk categorical-eval \
--     --project-id=PROJECT \
--     --dataset-id=DATASET \
--     --metrics-file=metrics.json \
--     --last=5m \
--     --persist \
--     --prompt-version=v2
--
-- Schedule this with cron (every 5 minutes):
--
--   */5 * * * * bq-agent-sdk categorical-eval \
--     --project-id=PROJECT \
--     --dataset-id=DATASET \
--     --metrics-file=/path/to/metrics.json \
--     --last=5m \
--     --persist \
--     --prompt-version=v2 \
--     >> /var/log/categorical-eval.log 2>&1
--
-- Overlapping windows or retries are safe: the dedup view
-- (`categorical_results_latest`) keeps only the latest row per
-- (session_id, metric_name, prompt_version) key, so dashboard
-- counts remain correct regardless of how many times a session
-- is evaluated.
--
-- For Cloud Run Jobs (wraps the same CLI in a container):
--
--   # 1. Build an image that installs bq-agent-sdk
--   # 2. Create a Cloud Run job that runs the CLI
--   gcloud run jobs create categorical-eval-job \
--     --image=IMAGE_URL \
--     --command="bq-agent-sdk" \
--     --args="categorical-eval,--project-id=PROJECT,--dataset-id=DATASET,--metrics-file=/config/metrics.json,--last=5m,--persist,--prompt-version=v2"
--
--   # 3. Schedule it with Cloud Scheduler
--   gcloud scheduler jobs create http categorical-eval-schedule \
--     --schedule="*/5 * * * *" \
--     --uri="https://REGION-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/PROJECT/jobs/categorical-eval-job:run" \
--     --http-method=POST \
--     --oauth-service-account-email=SA_EMAIL
--
-- =================================================================
