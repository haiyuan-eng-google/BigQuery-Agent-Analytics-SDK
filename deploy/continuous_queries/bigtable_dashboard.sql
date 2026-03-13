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

-- Continuous Query: Session Metrics → Bigtable Dashboard
--
-- Streams per-session metrics to Bigtable for low-latency dashboard
-- reads (e.g. Grafana, Looker real-time panels).
--
-- Prerequisites:
--   1. Enterprise reservation (see setup_reservation.md)
--   2. Bigtable instance + table with column family "metrics"
--   3. BQ connection with Bigtable write access
--
-- Placeholders:
--   PROJECT      — GCP project ID
--   DATASET      — BigQuery dataset
--   BT_PROJECT   — Bigtable project (often same as PROJECT)
--   BT_INSTANCE  — Bigtable instance ID
--   BT_TABLE     — Bigtable table name
--
-- Usage:
--   bq query --use_legacy_sql=false --continuous=true < bigtable_dashboard.sql

EXPORT DATA
OPTIONS (
  format = 'CLOUD_BIGTABLE',
  overwrite = true,
  bigtable_options = """{
    "projectId": "BT_PROJECT",
    "instanceId": "BT_INSTANCE",
    "tableId": "BT_TABLE",
    "columnFamilies": [{
      "familyId": "metrics",
      "onlyReadLatest": true
    }]
  }"""
)
AS
SELECT
  session_id AS rowkey,
  COUNT(*) AS event_count,
  COUNTIF(event_type IN ('TOOL_CALL', 'TOOL_COMPLETED')) AS tool_calls,
  COUNTIF(
    event_type = 'TOOL_COMPLETED'
    AND JSON_VALUE(content, '$.error_message') IS NOT NULL
  ) AS tool_errors,
  COUNTIF(event_type IN ('LLM_REQUEST', 'LLM_RESPONSE')) AS llm_calls,
  COUNTIF(event_type = 'USER_INPUT') AS turn_count,
  TIMESTAMP_DIFF(
    MAX(event_timestamp),
    MIN(event_timestamp),
    MILLISECOND
  ) AS total_latency_ms,
  MAX(event_timestamp) AS last_event_at
FROM
  APPENDS(TABLE `PROJECT.DATASET.agent_events`)
GROUP BY
  session_id;
