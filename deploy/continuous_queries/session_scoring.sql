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

-- Continuous Query: Per-Event Session Metrics Sink
--
-- Writes per-event rows with session context to a sink table.
-- BigQuery continuous queries do not allow GROUP BY or aggregation,
-- so this emits one row per event.  Downstream dashboards or
-- scheduled queries can aggregate by session_id.
--
-- Prerequisites:
--   1. Enterprise reservation (see setup_reservation.md)
--   2. Sink table created separately (see CREATE TABLE below)
--
-- Placeholders:
--   PROJECT — GCP project ID
--   DATASET — BigQuery dataset
--
-- One-time setup:
--
--   CREATE TABLE IF NOT EXISTS `PROJECT.DATASET.session_events_scored` (
--     session_id STRING,
--     event_type STRING,
--     agent STRING,
--     timestamp TIMESTAMP,
--     is_tool_call BOOL,
--     is_tool_error BOOL,
--     is_llm_call BOOL,
--     is_user_turn BOOL,
--     is_error BOOL,
--     latency_total_ms FLOAT64,
--     ingested_at TIMESTAMP
--   );
--
-- Start the continuous query:
--   bq query --use_legacy_sql=false --continuous=true \
--     < session_scoring.sql

INSERT INTO `PROJECT.DATASET.session_events_scored`
SELECT
  session_id,
  event_type,
  agent,
  timestamp,
  event_type = 'TOOL_STARTING' AS is_tool_call,
  event_type = 'TOOL_ERROR' AS is_tool_error,
  event_type = 'LLM_REQUEST' AS is_llm_call,
  event_type = 'USER_MESSAGE_RECEIVED' AS is_user_turn,
  (ENDS_WITH(event_type, '_ERROR')
   OR error_message IS NOT NULL
   OR status = 'ERROR') AS is_error,
  CAST(
    JSON_VALUE(latency_ms, '$.total_ms') AS FLOAT64
  ) AS latency_total_ms,
  CURRENT_TIMESTAMP() AS ingested_at
FROM
  APPENDS(TABLE `PROJECT.DATASET.agent_events`);
