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

-- Continuous Query: Real-time Error Classification
--
-- Routes each error event through AI.GENERATE_TEXT for classification,
-- then writes to a sink table.  No GROUP BY or aggregation — each row
-- is processed independently, which is the continuous-query requirement.
--
-- NOTE: This query uses AI.GENERATE_TEXT (table-valued function) rather
-- than AI.GENERATE (scalar function) because continuous queries using
-- APPENDS() require the table-valued function form with a MODEL reference.
--
-- Prerequisites:
--   1. Enterprise reservation (see setup_reservation.md)
--   2. BQ connection with Vertex AI access
--   3. Sink table created separately (see CREATE TABLE below — run once,
--      NOT as part of the continuous query)
--
-- Placeholders:
--   PROJECT    — GCP project ID
--   DATASET    — BigQuery dataset
--   CONNECTION — BQ connection ID (e.g. PROJECT.REGION.my-connection)
--
-- One-time setup (run before starting the continuous query):
--
--   CREATE TABLE IF NOT EXISTS `PROJECT.DATASET.error_analysis` (
--     session_id STRING,
--     event_timestamp TIMESTAMP,
--     error_message STRING,
--     error_category STRING,
--     severity STRING,
--     suggested_action STRING,
--     analyzed_at TIMESTAMP
--   );
--
-- Start the continuous query:
--   bq query --use_legacy_sql=false --continuous=true \
--     < realtime_error_analysis.sql

INSERT INTO `PROJECT.DATASET.error_analysis`
SELECT
  e.session_id,
  e.timestamp AS event_timestamp,
  e.error_message,
  JSON_VALUE(classification, '$.category') AS error_category,
  JSON_VALUE(classification, '$.severity') AS severity,
  JSON_VALUE(classification, '$.action') AS suggested_action,
  CURRENT_TIMESTAMP() AS analyzed_at
FROM
  APPENDS(TABLE `PROJECT.DATASET.agent_events`) AS e,
  UNNEST([
    AI.GENERATE_TEXT(
      MODEL `CONNECTION`,
      CONCAT(
        'Classify this agent error into exactly one category ',
        '(configuration, authentication, rate_limit, data_quality, ',
        'timeout, internal, unknown) and severity (critical, high, ',
        'medium, low). Return JSON only: {"category": "...", ',
        '"severity": "...", "action": "..."}\n\nError: ',
        e.error_message
      )
    ).ml_generate_text_llm_result
  ]) AS classification
WHERE
  e.event_type = 'TOOL_ERROR'
  OR e.error_message IS NOT NULL
  OR e.status = 'ERROR';
