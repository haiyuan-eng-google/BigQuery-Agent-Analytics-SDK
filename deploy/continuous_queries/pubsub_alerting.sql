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

-- Continuous Query: Critical Error → Pub/Sub Alerting
--
-- Publishes critical agent errors to a Pub/Sub topic for real-time
-- alerting (PagerDuty, Slack, etc.).
--
-- Prerequisites:
--   1. Enterprise reservation (see setup_reservation.md)
--   2. Pub/Sub topic: projects/PROJECT/topics/agent-alerts
--   3. BQ connection with Pub/Sub write access
--
-- Placeholders:
--   PROJECT — GCP project ID
--   DATASET — BigQuery dataset
--   TOPIC   — Pub/Sub topic path (e.g. projects/PROJECT/topics/agent-alerts)
--
-- Usage:
--   bq query --use_legacy_sql=false --continuous=true < pubsub_alerting.sql

EXPORT DATA
OPTIONS (
  format = 'CLOUD_PUBSUB',
  uri = 'TOPIC'
)
AS
SELECT
  TO_JSON_STRING(
    STRUCT(
      session_id,
      event_type,
      JSON_VALUE(content, '$.agent') AS agent,
      JSON_VALUE(content, '$.error_message') AS error_message,
      JSON_VALUE(content, '$.tool_name') AS tool_name,
      event_timestamp,
      'critical' AS severity
    )
  ) AS message
FROM
  APPENDS(TABLE `PROJECT.DATASET.agent_events`)
WHERE
  -- Alert on tool errors
  (event_type = 'TOOL_COMPLETED'
   AND JSON_VALUE(content, '$.error_message') IS NOT NULL)
  -- Alert on agent errors
  OR event_type = 'ERROR';
