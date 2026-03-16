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

-- Example: Event classification using Python UDF kernels.
--
-- Prerequisites:
--   Register UDFs from deploy/python_udf/register.sql
--
-- Replace PROJECT, DATASET, and UDF_DATASET with your values.


-- ------------------------------------------------------------------ --
-- 1. Classify errors and tool outcomes per event                       --
-- ------------------------------------------------------------------ --

SELECT
  session_id,
  event_type,
  timestamp,
  `PROJECT.UDF_DATASET.bqaa_is_error_event`(
    event_type, error_message, status
  ) AS is_error,
  `PROJECT.UDF_DATASET.bqaa_tool_outcome`(
    event_type, status
  ) AS tool_result
FROM
  `PROJECT.DATASET.agent_events`
WHERE
  event_type IN ('TOOL_STARTING', 'TOOL_COMPLETED', 'TOOL_ERROR')
ORDER BY
  session_id, timestamp;


-- ------------------------------------------------------------------ --
-- 2. Extract agent responses from LLM events                          --
-- ------------------------------------------------------------------ --
-- The content column is JSON-typed in the SDK schema.  The UDF expects
-- a plain STRING, so use CAST(content AS STRING) — not TO_JSON_STRING,
-- which would double-encode the payload and break key extraction.

SELECT
  session_id,
  timestamp,
  `PROJECT.UDF_DATASET.bqaa_extract_response_text`(
    CAST(content AS STRING)
  ) AS response_text
FROM
  `PROJECT.DATASET.agent_events`
WHERE
  event_type IN ('LLM_RESPONSE', 'AGENT_COMPLETED')
  AND content IS NOT NULL
ORDER BY
  session_id, timestamp;


-- ------------------------------------------------------------------ --
-- 3. Error rate per session using the UDF for classification           --
-- ------------------------------------------------------------------ --

SELECT
  session_id,
  COUNT(*) AS total_events,
  COUNTIF(
    `PROJECT.UDF_DATASET.bqaa_is_error_event`(
      event_type, error_message, status
    )
  ) AS error_events,
  COUNTIF(
    `PROJECT.UDF_DATASET.bqaa_tool_outcome`(event_type, status) = 'error'
  ) AS tool_errors,
  COUNTIF(
    `PROJECT.UDF_DATASET.bqaa_tool_outcome`(event_type, status) = 'success'
  ) AS tool_successes
FROM
  `PROJECT.DATASET.agent_events`
GROUP BY
  session_id
ORDER BY
  error_events DESC
LIMIT 50;
