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

-- Example: Session evaluation using Python UDF score kernels.
--
-- This example shows the SQL + UDF split pattern: SQL pre-aggregates
-- per-session metrics, then UDF score kernels compute 0.0-1.0 scores.
--
-- Prerequisites:
--   Register UDFs from deploy/python_udf/register.sql
--
-- Replace PROJECT, DATASET, and UDF_DATASET with your values.


-- ------------------------------------------------------------------ --
-- 1. Score all sessions on latency, error rate, and turn count         --
-- ------------------------------------------------------------------ --

-- Note: COALESCE guards are required because Python UDFs receive NULL
-- as None, which fails numeric comparisons.  The SDK evaluator path
-- catches these via exception handling (evaluators.py:211), but the
-- UDF kernels are pure functions with no such fallback.

WITH session_summary AS (
  SELECT
    session_id,
    COALESCE(AVG(
      CAST(
        JSON_VALUE(latency_ms, '$.total_ms') AS FLOAT64
      )
    ), 0.0) AS avg_latency_ms,
    COUNTIF(event_type = 'TOOL_STARTING') AS tool_calls,
    COUNTIF(event_type = 'TOOL_ERROR') AS tool_errors,
    COUNTIF(event_type = 'USER_MESSAGE_RECEIVED') AS turn_count
  FROM
    `PROJECT.DATASET.agent_events`
  GROUP BY
    session_id
)
SELECT
  session_id,
  avg_latency_ms,
  tool_calls,
  tool_errors,
  turn_count,
  `PROJECT.UDF_DATASET.bqaa_score_latency`(
    avg_latency_ms, 5000.0
  ) AS latency_score,
  `PROJECT.UDF_DATASET.bqaa_score_error_rate`(
    tool_calls, tool_errors, 0.1
  ) AS error_rate_score,
  `PROJECT.UDF_DATASET.bqaa_score_turn_count`(
    turn_count, 10
  ) AS turn_count_score
FROM
  session_summary
ORDER BY
  latency_score ASC
LIMIT 100;


-- ------------------------------------------------------------------ --
-- 2. Multi-metric pass/fail gate                                       --
-- ------------------------------------------------------------------ --
-- Sessions pass only if ALL scores are above 0.5.

WITH session_summary AS (
  SELECT
    session_id,
    COALESCE(AVG(
      CAST(
        JSON_VALUE(latency_ms, '$.total_ms') AS FLOAT64
      )
    ), 0.0) AS avg_latency_ms,
    COUNTIF(event_type = 'TOOL_STARTING') AS tool_calls,
    COUNTIF(event_type = 'TOOL_ERROR') AS tool_errors,
    COUNTIF(event_type = 'USER_MESSAGE_RECEIVED') AS turn_count
  FROM
    `PROJECT.DATASET.agent_events`
  GROUP BY
    session_id
),
scored AS (
  SELECT
    session_id,
    `PROJECT.UDF_DATASET.bqaa_score_latency`(
      avg_latency_ms, 5000.0
    ) AS latency_score,
    `PROJECT.UDF_DATASET.bqaa_score_error_rate`(
      tool_calls, tool_errors, 0.1
    ) AS error_rate_score,
    `PROJECT.UDF_DATASET.bqaa_score_turn_count`(
      turn_count, 10
    ) AS turn_count_score
  FROM
    session_summary
)
SELECT
  session_id,
  latency_score,
  error_rate_score,
  turn_count_score,
  (latency_score >= 0.5
   AND error_rate_score >= 0.5
   AND turn_count_score >= 0.5) AS passed
FROM
  scored
ORDER BY
  passed, session_id;


-- ------------------------------------------------------------------ --
-- 3. Cost evaluation with custom pricing                               --
-- ------------------------------------------------------------------ --

WITH session_tokens AS (
  SELECT
    session_id,
    SUM(COALESCE(
      CAST(JSON_VALUE(
        attributes, '$.usage_metadata.prompt_token_count'
      ) AS INT64), 0
    )) AS input_tokens,
    SUM(COALESCE(
      CAST(JSON_VALUE(
        attributes, '$.usage_metadata.candidates_token_count'
      ) AS INT64), 0
    )) AS output_tokens
  FROM
    `PROJECT.DATASET.agent_events`
  WHERE
    event_type = 'LLM_RESPONSE'
  GROUP BY
    session_id
)
SELECT
  session_id,
  input_tokens,
  output_tokens,
  `PROJECT.UDF_DATASET.bqaa_score_cost`(
    input_tokens, output_tokens,
    -- max $2.00 per session, Gemini 2.5 Flash pricing
    2.0, 0.00015, 0.0006
  ) AS cost_score
FROM
  session_tokens
ORDER BY
  cost_score ASC
LIMIT 50;
