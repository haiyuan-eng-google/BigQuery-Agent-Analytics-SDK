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

-- Example: All-in-one session evaluation with JSON STRING envelope.
--
-- bqaa_eval_summary_json computes all six score kernels in a single
-- UDF call and returns a JSON STRING with individual scores plus an
-- overall pass/fail flag.  Use JSON_VALUE() to extract fields.
--
-- Prerequisites:
--   Register UDFs from deploy/python_udf/register.sql
--
-- Replace PROJECT, DATASET, and UDF_DATASET with your values.


-- ------------------------------------------------------------------ --
-- 1. Complete session evaluation in one UDF call                       --
-- ------------------------------------------------------------------ --

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
    COUNTIF(event_type = 'USER_MESSAGE_RECEIVED') AS turn_count,
    SUM(COALESCE(
      CAST(JSON_VALUE(
        attributes, '$.usage_metadata.total_token_count'
      ) AS INT64),
      CAST(JSON_VALUE(
        content, '$.usage.total'
      ) AS INT64),
      COALESCE(
        CAST(JSON_VALUE(
          attributes, '$.input_tokens'
        ) AS INT64), 0
      ) + COALESCE(
        CAST(JSON_VALUE(
          attributes, '$.output_tokens'
        ) AS INT64), 0
      )
    )) AS total_tokens,
    COALESCE(AVG(
      CAST(
        JSON_VALUE(latency_ms, '$.time_to_first_token_ms') AS FLOAT64
      )
    ), 0.0) AS avg_ttft_ms,
    SUM(COALESCE(
      CAST(JSON_VALUE(
        attributes, '$.usage_metadata.prompt_token_count'
      ) AS INT64),
      CAST(JSON_VALUE(
        content, '$.usage.prompt'
      ) AS INT64),
      CAST(JSON_VALUE(
        attributes, '$.input_tokens'
      ) AS INT64)
    )) AS input_tokens,
    SUM(COALESCE(
      CAST(JSON_VALUE(
        attributes, '$.usage_metadata.candidates_token_count'
      ) AS INT64),
      CAST(JSON_VALUE(
        content, '$.usage.completion'
      ) AS INT64),
      CAST(JSON_VALUE(
        attributes, '$.output_tokens'
      ) AS INT64)
    )) AS output_tokens
  FROM
    `PROJECT.DATASET.agent_events`
  GROUP BY
    session_id
)
SELECT
  session_id,
  `PROJECT.UDF_DATASET.bqaa_eval_summary_json`(
    avg_latency_ms, tool_calls, tool_errors,
    turn_count, total_tokens, avg_ttft_ms,
    input_tokens, output_tokens,
    -- Thresholds
    5000.0,   -- threshold_ms (latency)
    0.1,      -- max_error_rate
    10,       -- max_turns
    50000,    -- max_tokens
    1000.0,   -- ttft_threshold_ms
    2.0,      -- max_cost_usd
    0.00015,  -- input_cost_per_1k
    0.0006    -- output_cost_per_1k
  ) AS eval_summary
FROM
  session_summary
ORDER BY
  session_id
LIMIT 100;


-- ------------------------------------------------------------------ --
-- 2. Extract individual scores from the JSON summary                   --
-- ------------------------------------------------------------------ --

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
    COUNTIF(event_type = 'USER_MESSAGE_RECEIVED') AS turn_count,
    SUM(COALESCE(
      CAST(JSON_VALUE(
        attributes, '$.usage_metadata.total_token_count'
      ) AS INT64),
      CAST(JSON_VALUE(
        content, '$.usage.total'
      ) AS INT64),
      COALESCE(
        CAST(JSON_VALUE(
          attributes, '$.input_tokens'
        ) AS INT64), 0
      ) + COALESCE(
        CAST(JSON_VALUE(
          attributes, '$.output_tokens'
        ) AS INT64), 0
      )
    )) AS total_tokens,
    COALESCE(AVG(
      CAST(
        JSON_VALUE(latency_ms, '$.time_to_first_token_ms') AS FLOAT64
      )
    ), 0.0) AS avg_ttft_ms,
    SUM(COALESCE(
      CAST(JSON_VALUE(
        attributes, '$.usage_metadata.prompt_token_count'
      ) AS INT64),
      CAST(JSON_VALUE(
        content, '$.usage.prompt'
      ) AS INT64),
      CAST(JSON_VALUE(
        attributes, '$.input_tokens'
      ) AS INT64)
    )) AS input_tokens,
    SUM(COALESCE(
      CAST(JSON_VALUE(
        attributes, '$.usage_metadata.candidates_token_count'
      ) AS INT64),
      CAST(JSON_VALUE(
        content, '$.usage.completion'
      ) AS INT64),
      CAST(JSON_VALUE(
        attributes, '$.output_tokens'
      ) AS INT64)
    )) AS output_tokens
  FROM
    `PROJECT.DATASET.agent_events`
  GROUP BY
    session_id
),
evaluated AS (
  SELECT
    session_id,
    `PROJECT.UDF_DATASET.bqaa_eval_summary_json`(
      avg_latency_ms, tool_calls, tool_errors,
      turn_count, total_tokens, avg_ttft_ms,
      input_tokens, output_tokens,
      5000.0, 0.1, 10, 50000, 1000.0,
      2.0, 0.00015, 0.0006
    ) AS summary
  FROM
    session_summary
)
SELECT
  session_id,
  CAST(JSON_VALUE(summary, '$.latency') AS FLOAT64) AS latency_score,
  CAST(JSON_VALUE(summary, '$.error_rate') AS FLOAT64) AS error_rate_score,
  CAST(JSON_VALUE(summary, '$.turn_count') AS FLOAT64) AS turn_count_score,
  CAST(JSON_VALUE(summary, '$.token_efficiency') AS FLOAT64) AS token_score,
  CAST(JSON_VALUE(summary, '$.ttft') AS FLOAT64) AS ttft_score,
  CAST(JSON_VALUE(summary, '$.cost') AS FLOAT64) AS cost_score,
  JSON_VALUE(summary, '$.passed') = 'true' AS passed
FROM
  evaluated
ORDER BY
  passed, session_id;


-- ------------------------------------------------------------------ --
-- 3. Event label normalization                                         --
-- ------------------------------------------------------------------ --
-- Normalize raw event_type values to high-level categories for
-- aggregate analysis.

SELECT
  `PROJECT.UDF_DATASET.bqaa_normalize_event_label`(
    event_type
  ) AS event_category,
  COUNT(*) AS event_count,
  COUNT(DISTINCT session_id) AS session_count
FROM
  `PROJECT.DATASET.agent_events`
GROUP BY
  event_category
ORDER BY
  event_count DESC;
