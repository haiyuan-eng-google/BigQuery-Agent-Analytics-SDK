-- AI.FORECAST vs ML.FORECAST — Side-by-Side Comparison
--
-- AI.FORECAST uses the built-in TimesFM model.  No model training
-- or provisioning is needed — just pass a subquery of historical data
-- and the function handles the rest.
--
-- ML.FORECAST requires a pre-trained ARIMA_PLUS model created via
-- CREATE MODEL ... OPTIONS(model_type='ARIMA_PLUS', ...).  The model
-- must be trained before ML.FORECAST can be called.
--
-- Both return prediction/confidence intervals
-- (prediction_interval_lower_bound, prediction_interval_upper_bound).

-- ================================================================
-- 1. AI.FORECAST — 24-hour hourly latency forecast (no model needed)
-- ================================================================
SELECT *
FROM AI.FORECAST(
  (
    SELECT
      TIMESTAMP_TRUNC(timestamp, HOUR) AS hour,
      AVG(CAST(JSON_EXTRACT_SCALAR(latency_ms, '$.total_ms') AS FLOAT64)) AS avg_latency
    FROM `my_project.agent_analytics.agent_events`
    WHERE event_type = 'LLM_RESPONSE'
      AND latency_ms IS NOT NULL
      AND timestamp > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
    GROUP BY hour
    HAVING avg_latency IS NOT NULL
  ),
  horizon => 24,
  confidence_level => 0.95,
  timestamp_col => 'hour',
  data_col => 'avg_latency'
);

-- ================================================================
-- 2. ML.FORECAST — same forecast using pre-trained ARIMA_PLUS model
-- ================================================================
-- Prerequisite: train the model first:
--   CREATE OR REPLACE MODEL `my_project.agent_analytics.latency_anomaly_model`
--   OPTIONS(model_type='ARIMA_PLUS', time_series_timestamp_col='hour',
--           time_series_data_col='avg_latency', auto_arima=TRUE,
--           data_frequency='HOURLY')
--   AS SELECT ... (same hourly aggregation);

SELECT *
FROM ML.FORECAST(
  MODEL `my_project.agent_analytics.latency_anomaly_model`,
  STRUCT(24 AS horizon, 0.95 AS confidence_level)
);

-- ================================================================
-- 3. Agreement check — compare forecast values and interval widths
-- ================================================================
WITH ai_forecast AS (
  SELECT
    time_series_timestamp AS ts,
    time_series_data AS forecast_value,
    prediction_interval_upper_bound - prediction_interval_lower_bound AS interval_width
  FROM AI.FORECAST(
    (
      SELECT
        TIMESTAMP_TRUNC(timestamp, HOUR) AS hour,
        AVG(CAST(JSON_EXTRACT_SCALAR(latency_ms, '$.total_ms') AS FLOAT64)) AS avg_latency
      FROM `my_project.agent_analytics.agent_events`
      WHERE event_type = 'LLM_RESPONSE'
        AND latency_ms IS NOT NULL
        AND timestamp > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
      GROUP BY hour
      HAVING avg_latency IS NOT NULL
    ),
    horizon => 24,
    confidence_level => 0.95,
    timestamp_col => 'hour',
    data_col => 'avg_latency'
  )
),
ml_forecast AS (
  SELECT
    forecast_timestamp AS ts,
    forecast_value,
    prediction_interval_upper_bound - prediction_interval_lower_bound AS interval_width
  FROM ML.FORECAST(
    MODEL `my_project.agent_analytics.latency_anomaly_model`,
    STRUCT(24 AS horizon, 0.95 AS confidence_level)
  )
)
SELECT
  a.ts,
  a.forecast_value AS ai_forecast,
  m.forecast_value AS ml_forecast,
  ABS(a.forecast_value - m.forecast_value) AS value_diff,
  a.interval_width AS ai_interval_width,
  m.interval_width AS ml_interval_width
FROM ai_forecast a
JOIN ml_forecast m ON a.ts = m.ts
ORDER BY a.ts;
