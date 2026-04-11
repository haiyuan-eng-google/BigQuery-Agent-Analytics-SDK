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

-- ai_classify_side_by_side.sql
--
-- Side-by-side validation queries for AI.CLASSIFY vs AI.GENERATE
-- categorical evaluation results.  Run these against the persistence
-- table after executing `categorical-eval` with both
-- `--no-include-justification` (AI.CLASSIFY) and the default
-- `--include-justification` (AI.GENERATE).
--
-- Replace {project}, {dataset}, and {results_table} with your values.

-- 1. Agreement rate between AI.CLASSIFY and AI.GENERATE
--    How often do both execution modes pick the same category?
SELECT
  metric_name,
  COUNTIF(c.category = g.category) AS agree,
  COUNT(*) AS total,
  SAFE_DIVIDE(COUNTIF(c.category = g.category), COUNT(*)) AS agreement_rate
FROM
  `{project}.{dataset}.{results_table}` AS c
JOIN
  `{project}.{dataset}.{results_table}` AS g
  ON c.session_id = g.session_id
  AND c.metric_name = g.metric_name
WHERE
  c.execution_mode = 'ai_classify'
  AND g.execution_mode = 'ai_generate'
GROUP BY metric_name
ORDER BY metric_name;

-- 2. Disagreement details with justifications from AI.GENERATE
SELECT
  c.session_id,
  c.metric_name,
  c.category AS classify_category,
  g.category AS generate_category,
  g.justification
FROM
  `{project}.{dataset}.{results_table}` AS c
JOIN
  `{project}.{dataset}.{results_table}` AS g
  ON c.session_id = g.session_id
  AND c.metric_name = g.metric_name
WHERE
  c.execution_mode = 'ai_classify'
  AND g.execution_mode = 'ai_generate'
  AND c.category != g.category
ORDER BY c.metric_name, c.session_id;

-- 3. Error rate comparison
--    AI.CLASSIFY NULLs (execution failure) vs AI.GENERATE parse errors.
SELECT
  execution_mode,
  metric_name,
  COUNTIF(parse_error) AS parse_errors,
  COUNTIF(NOT passed_validation AND NOT parse_error) AS null_failures,
  COUNT(*) AS total,
  SAFE_DIVIDE(
    COUNTIF(NOT passed_validation),
    COUNT(*)
  ) AS error_rate
FROM
  `{project}.{dataset}.{results_table}`
WHERE
  execution_mode IN ('ai_classify', 'ai_generate')
GROUP BY execution_mode, metric_name
ORDER BY execution_mode, metric_name;

-- 4. Category distribution pivot by execution mode
SELECT
  metric_name,
  category,
  COUNTIF(execution_mode = 'ai_classify') AS classify_count,
  COUNTIF(execution_mode = 'ai_generate') AS generate_count
FROM
  `{project}.{dataset}.{results_table}`
WHERE
  execution_mode IN ('ai_classify', 'ai_generate')
  AND category IS NOT NULL
GROUP BY metric_name, category
ORDER BY metric_name, category;
