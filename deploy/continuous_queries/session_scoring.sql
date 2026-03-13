-- Copyright 2026 Google LLC
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

-- Continuous Query: Per-Session Quality Scoring
--
-- Computes real-time quality scores for each session as events arrive.
-- Tracks latency, error rate, and turn count metrics.
--
-- Prerequisites:
--   1. Enterprise reservation (see setup_reservation.md)
--   2. Sink table: PROJECT.DATASET.session_scores
--
-- Placeholders:
--   PROJECT — GCP project ID
--   DATASET — BigQuery dataset
--
-- Usage:
--   bq query --use_legacy_sql=false --continuous=true < session_scoring.sql

CREATE TABLE IF NOT EXISTS `PROJECT.DATASET.session_scores` (
  session_id STRING,
  event_count INT64,
  tool_calls INT64,
  tool_errors INT64,
  llm_calls INT64,
  error_rate FLOAT64,
  total_latency_ms FLOAT64,
  turn_count INT64,
  quality_score FLOAT64,
  scored_at TIMESTAMP
);

EXPORT DATA
OPTIONS (
  format = 'CLOUD_BIGTABLE',
  overwrite = false
)
AS
SELECT
  session_id,
  COUNT(*) AS event_count,
  COUNTIF(event_type IN ('TOOL_CALL', 'TOOL_COMPLETED')) AS tool_calls,
  COUNTIF(
    event_type = 'TOOL_COMPLETED'
    AND JSON_VALUE(content, '$.error_message') IS NOT NULL
  ) AS tool_errors,
  COUNTIF(event_type IN ('LLM_REQUEST', 'LLM_RESPONSE')) AS llm_calls,
  SAFE_DIVIDE(
    COUNTIF(
      event_type = 'TOOL_COMPLETED'
      AND JSON_VALUE(content, '$.error_message') IS NOT NULL
    ),
    NULLIF(
      COUNTIF(event_type IN ('TOOL_CALL', 'TOOL_COMPLETED')),
      0
    )
  ) AS error_rate,
  TIMESTAMP_DIFF(
    MAX(event_timestamp),
    MIN(event_timestamp),
    MILLISECOND
  ) AS total_latency_ms,
  COUNTIF(event_type = 'USER_INPUT') AS turn_count,
  -- Quality score: weighted combination (lower error rate + lower latency = higher score)
  GREATEST(0.0, LEAST(1.0,
    1.0
    - COALESCE(SAFE_DIVIDE(
        COUNTIF(
          event_type = 'TOOL_COMPLETED'
          AND JSON_VALUE(content, '$.error_message') IS NOT NULL
        ),
        NULLIF(
          COUNTIF(event_type IN ('TOOL_CALL', 'TOOL_COMPLETED')),
          0
        )
      ), 0.0) * 0.5
    - LEAST(
        TIMESTAMP_DIFF(MAX(event_timestamp), MIN(event_timestamp), MILLISECOND) / 10000.0,
        0.5
      )
  )) AS quality_score,
  CURRENT_TIMESTAMP() AS scored_at
FROM
  APPENDS(TABLE `PROJECT.DATASET.agent_events`)
GROUP BY
  session_id;
