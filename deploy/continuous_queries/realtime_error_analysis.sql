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

-- Continuous Query: Real-time Error Analysis
--
-- Classifies tool errors using AI.GENERATE_TEXT and writes results
-- to an error_analysis sink table.
--
-- Prerequisites:
--   1. Enterprise reservation (see setup_reservation.md)
--   2. BQ connection with Vertex AI access
--   3. Sink table: PROJECT.DATASET.error_analysis
--
-- Placeholders:
--   PROJECT    — GCP project ID
--   DATASET    — BigQuery dataset
--   CONNECTION — BQ connection ID (e.g. PROJECT.REGION.my-connection)
--
-- Usage:
--   bq query --use_legacy_sql=false --continuous=true < realtime_error_analysis.sql

CREATE TABLE IF NOT EXISTS `PROJECT.DATASET.error_analysis` (
  session_id STRING,
  event_timestamp TIMESTAMP,
  error_message STRING,
  error_category STRING,
  severity STRING,
  suggested_action STRING,
  analyzed_at TIMESTAMP
);

EXPORT DATA
OPTIONS (
  format = 'CLOUD_BIGTABLE',
  overwrite = false
)
AS
SELECT
  e.session_id,
  e.event_timestamp,
  JSON_VALUE(e.content, '$.error_message') AS error_message,
  JSON_VALUE(
    AI.GENERATE_TEXT(
      MODEL `CONNECTION`,
      CONCAT(
        'Classify this agent error into exactly one category ',
        '(configuration, authentication, rate_limit, data_quality, ',
        'timeout, internal, unknown) and severity (critical, high, ',
        'medium, low). Return JSON: {"category": "...", "severity": "...", ',
        '"action": "..."}\n\nError: ',
        JSON_VALUE(e.content, '$.error_message')
      )
    ).ml_generate_text_llm_result,
    '$.category'
  ) AS error_category,
  JSON_VALUE(
    AI.GENERATE_TEXT(
      MODEL `CONNECTION`,
      CONCAT(
        'Classify this agent error into exactly one category ',
        '(configuration, authentication, rate_limit, data_quality, ',
        'timeout, internal, unknown) and severity (critical, high, ',
        'medium, low). Return JSON: {"category": "...", "severity": "...", ',
        '"action": "..."}\n\nError: ',
        JSON_VALUE(e.content, '$.error_message')
      )
    ).ml_generate_text_llm_result,
    '$.severity'
  ) AS severity,
  JSON_VALUE(
    AI.GENERATE_TEXT(
      MODEL `CONNECTION`,
      CONCAT(
        'Classify this agent error into exactly one category ',
        '(configuration, authentication, rate_limit, data_quality, ',
        'timeout, internal, unknown) and severity (critical, high, ',
        'medium, low). Return JSON: {"category": "...", "severity": "...", ',
        '"action": "..."}\n\nError: ',
        JSON_VALUE(e.content, '$.error_message')
      )
    ).ml_generate_text_llm_result,
    '$.action'
  ) AS suggested_action,
  CURRENT_TIMESTAMP() AS analyzed_at
FROM
  APPENDS(TABLE `PROJECT.DATASET.agent_events`) AS e
WHERE
  e.event_type = 'TOOL_COMPLETED'
  AND JSON_VALUE(e.content, '$.error_message') IS NOT NULL;
