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

-- Example: Vectorized UDFs for batch scoring and event normalization.
--
-- Vectorized UDFs (OPTIONS vectorized = true) process rows in batches
-- using numpy/pandas, which is faster than row-by-row scalar UDFs on
-- large result sets.  The SQL call-site is identical to scalar UDFs —
-- BigQuery handles the batching transparently.
--
-- Prerequisites:
--   Register UDFs from deploy/python_udf/register.sql
--
-- Replace PROJECT, DATASET, and UDF_DATASET with your values.


-- ------------------------------------------------------------------ --
-- 1. Batch latency scoring (vectorized)                               --
-- ------------------------------------------------------------------ --
-- Same query as the scalar example, but uses the _batch variant.
-- On tables with >10K sessions, expect 2-5x throughput improvement.

WITH session_summary AS (
  SELECT
    session_id,
    COALESCE(AVG(
      CAST(
        JSON_VALUE(latency_ms, '$.total_ms') AS FLOAT64
      )
    ), 0.0) AS avg_latency_ms
  FROM
    `PROJECT.DATASET.agent_events`
  GROUP BY
    session_id
)
SELECT
  session_id,
  avg_latency_ms,
  `PROJECT.UDF_DATASET.bqaa_score_latency_batch`(
    avg_latency_ms, 5000.0
  ) AS latency_score
FROM
  session_summary
ORDER BY
  latency_score ASC
LIMIT 100;


-- ------------------------------------------------------------------ --
-- 2. Batch cost scoring (vectorized)                                  --
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
  `PROJECT.UDF_DATASET.bqaa_score_cost_batch`(
    input_tokens, output_tokens,
    2.0, 0.00015, 0.0006
  ) AS cost_score
FROM
  session_tokens
ORDER BY
  cost_score ASC
LIMIT 100;


-- ------------------------------------------------------------------ --
-- 3. Event label normalization (vectorized)                           --
-- ------------------------------------------------------------------ --
-- Normalize raw event_type values to high-level categories for
-- aggregate analysis.  Vectorized processing handles the string
-- mapping efficiently across all rows.

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


-- ------------------------------------------------------------------ --
-- 4. Combined: batch error-rate scoring + label normalization         --
-- ------------------------------------------------------------------ --

WITH session_errors AS (
  SELECT
    session_id,
    COUNTIF(event_type = 'TOOL_STARTING') AS tool_calls,
    COUNTIF(event_type = 'TOOL_ERROR') AS tool_errors
  FROM
    `PROJECT.DATASET.agent_events`
  GROUP BY
    session_id
)
SELECT
  session_id,
  tool_calls,
  tool_errors,
  `PROJECT.UDF_DATASET.bqaa_score_error_rate_batch`(
    tool_calls, tool_errors, 0.1
  ) AS error_rate_score
FROM
  session_errors
ORDER BY
  error_rate_score ASC
LIMIT 100;
