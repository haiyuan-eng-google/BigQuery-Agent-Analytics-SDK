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

-- Continuous Query: Event Metrics → Bigtable Dashboard
--
-- Streams per-event metrics to Bigtable for low-latency dashboard
-- reads (e.g. Grafana, Looker real-time panels).  Each event is
-- written as a separate Bigtable row — no GROUP BY (continuous
-- queries do not allow aggregation).
--
-- Row key format: session_id#timestamp to enable range scans per
-- session.  Bigtable column family "metrics" stores the event data.
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
-- Start the continuous query:
--   bq query --use_legacy_sql=false --continuous=true \
--     < bigtable_dashboard.sql

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
  CONCAT(session_id, '#', CAST(timestamp AS STRING)) AS rowkey,
  session_id,
  event_type,
  agent,
  CAST(
    JSON_VALUE(latency_ms, '$.total_ms') AS FLOAT64
  ) AS latency_total_ms,
  event_type = 'TOOL_STARTING' AS is_tool_call,
  event_type = 'TOOL_ERROR' AS is_tool_error,
  event_type = 'LLM_REQUEST' AS is_llm_call,
  event_type = 'USER_MESSAGE_RECEIVED' AS is_user_turn,
  (ENDS_WITH(event_type, '_ERROR')
   OR error_message IS NOT NULL
   OR status = 'ERROR') AS is_error,
  timestamp
FROM
  APPENDS(TABLE `PROJECT.DATASET.agent_events`);
