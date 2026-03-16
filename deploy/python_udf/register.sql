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

-- BigQuery Python UDF registration for Agent Analytics SDK.
--
-- This file registers all Tier 1 (event semantics) and Tier 2 (score
-- kernel) UDFs.  Each UDF inlines its kernel body so there are no
-- external dependencies — no pip install, no Cloud Function.
--
-- Prerequisites:
--   - BigQuery Python UDF support enabled (Preview)
--   - A dataset to host the UDFs
--
-- Replace PROJECT and UDF_DATASET with your values.
-- UDFs are region-scoped; create them in each region where your
-- data lives, or use dataset replication for utility datasets.
--
-- To generate this file programmatically:
--   python -c "
--     from bigquery_agent_analytics.udf_sql_templates import generate_all_udfs
--     print(generate_all_udfs('PROJECT', 'UDF_DATASET'))
--   "


-- ------------------------------------------------------------------ --
-- Tier 1: Event Semantics                                              --
-- ------------------------------------------------------------------ --

-- Returns TRUE when the event represents an error.
CREATE OR REPLACE FUNCTION `PROJECT.UDF_DATASET.bqaa_is_error_event`(
  event_type STRING, error_message STRING, status STRING
)
RETURNS BOOL
LANGUAGE python
OPTIONS (
  entry_point = 'bqaa_is_error_event',
  runtime_version = 'python-3.11',
  description = """Returns TRUE when the event represents an error."""
)
AS r"""
    def bqaa_is_error_event(event_type, error_message, status):
        return (
            event_type.endswith("_ERROR")
            or error_message is not None
            or status == "ERROR"
        )
""";

-- Returns a canonical tool outcome: 'success', 'error', or 'in_progress'.
CREATE OR REPLACE FUNCTION `PROJECT.UDF_DATASET.bqaa_tool_outcome`(
  event_type STRING, status STRING
)
RETURNS STRING
LANGUAGE python
OPTIONS (
  entry_point = 'bqaa_tool_outcome',
  runtime_version = 'python-3.11',
  description = """Returns a canonical tool outcome: 'success', 'error', or 'in_progress'."""
)
AS r"""
    def bqaa_tool_outcome(event_type, status):
        if event_type == "TOOL_ERROR" or status == "ERROR":
            return "error"
        if event_type == "TOOL_COMPLETED":
            return "success"
        return "in_progress"
""";

-- Extracts user-visible response text from a JSON content string.
CREATE OR REPLACE FUNCTION `PROJECT.UDF_DATASET.bqaa_extract_response_text`(
  content_json STRING
)
RETURNS STRING
LANGUAGE python
OPTIONS (
  entry_point = 'bqaa_extract_response_text',
  runtime_version = 'python-3.11',
  description = """Extracts user-visible response text from a JSON content string."""
)
AS r"""
    import json

    def bqaa_extract_response_text(content_json):
        if not content_json:
            return None
        try:
            content = json.loads(content_json)
        except (json.JSONDecodeError, TypeError):
            return str(content_json) if content_json else None
        if not isinstance(content, dict):
            return str(content) if content else None
        return (
            content.get("response")
            or content.get("text_summary")
            or content.get("text")
            or content.get("raw")
            or None
        )
""";


-- ------------------------------------------------------------------ --
-- Tier 2: Score Kernels                                                --
-- ------------------------------------------------------------------ --

-- Score average latency against a threshold (0.0-1.0).
CREATE OR REPLACE FUNCTION `PROJECT.UDF_DATASET.bqaa_score_latency`(
  avg_latency_ms FLOAT64, threshold_ms FLOAT64
)
RETURNS FLOAT64
LANGUAGE python
OPTIONS (
  entry_point = 'bqaa_score_latency',
  runtime_version = 'python-3.11',
  description = """Score average latency against a threshold (0.0-1.0)."""
)
AS r"""
    def bqaa_score_latency(avg_latency_ms, threshold_ms):
        if avg_latency_ms <= 0:
            return 1.0
        if avg_latency_ms >= threshold_ms:
            return 0.0
        return 1.0 - (avg_latency_ms / threshold_ms)
""";

-- Score tool error rate against a threshold (0.0-1.0).
CREATE OR REPLACE FUNCTION `PROJECT.UDF_DATASET.bqaa_score_error_rate`(
  tool_calls INT64, tool_errors INT64, max_error_rate FLOAT64
)
RETURNS FLOAT64
LANGUAGE python
OPTIONS (
  entry_point = 'bqaa_score_error_rate',
  runtime_version = 'python-3.11',
  description = """Score tool error rate against a threshold (0.0-1.0)."""
)
AS r"""
    def bqaa_score_error_rate(tool_calls, tool_errors, max_error_rate):
        if tool_calls <= 0:
            return 1.0
        rate = tool_errors / tool_calls
        if rate >= max_error_rate:
            return 0.0
        return 1.0 - (rate / max_error_rate)
""";

-- Score turn count against a maximum (0.0-1.0).
CREATE OR REPLACE FUNCTION `PROJECT.UDF_DATASET.bqaa_score_turn_count`(
  turn_count INT64, max_turns INT64
)
RETURNS FLOAT64
LANGUAGE python
OPTIONS (
  entry_point = 'bqaa_score_turn_count',
  runtime_version = 'python-3.11',
  description = """Score turn count against a maximum (0.0-1.0)."""
)
AS r"""
    def bqaa_score_turn_count(turn_count, max_turns):
        if turn_count <= 0:
            return 1.0
        if turn_count >= max_turns:
            return 0.0
        return 1.0 - (turn_count / max_turns)
""";

-- Score total token usage against a maximum (0.0-1.0).
CREATE OR REPLACE FUNCTION `PROJECT.UDF_DATASET.bqaa_score_token_efficiency`(
  total_tokens INT64, max_tokens INT64
)
RETURNS FLOAT64
LANGUAGE python
OPTIONS (
  entry_point = 'bqaa_score_token_efficiency',
  runtime_version = 'python-3.11',
  description = """Score total token usage against a maximum (0.0-1.0)."""
)
AS r"""
    def bqaa_score_token_efficiency(total_tokens, max_tokens):
        if total_tokens <= 0:
            return 1.0
        if total_tokens >= max_tokens:
            return 0.0
        return 1.0 - (total_tokens / max_tokens)
""";

-- Score average time-to-first-token against a threshold (0.0-1.0).
CREATE OR REPLACE FUNCTION `PROJECT.UDF_DATASET.bqaa_score_ttft`(
  avg_ttft_ms FLOAT64, threshold_ms FLOAT64
)
RETURNS FLOAT64
LANGUAGE python
OPTIONS (
  entry_point = 'bqaa_score_ttft',
  runtime_version = 'python-3.11',
  description = """Score average time-to-first-token against a threshold (0.0-1.0)."""
)
AS r"""
    def bqaa_score_ttft(avg_ttft_ms, threshold_ms):
        if avg_ttft_ms <= 0:
            return 1.0
        if avg_ttft_ms >= threshold_ms:
            return 0.0
        return 1.0 - (avg_ttft_ms / threshold_ms)
""";

-- Score estimated session cost against a maximum (0.0-1.0).
CREATE OR REPLACE FUNCTION `PROJECT.UDF_DATASET.bqaa_score_cost`(
  input_tokens INT64, output_tokens INT64,
  max_cost_usd FLOAT64,
  input_cost_per_1k FLOAT64, output_cost_per_1k FLOAT64
)
RETURNS FLOAT64
LANGUAGE python
OPTIONS (
  entry_point = 'bqaa_score_cost',
  runtime_version = 'python-3.11',
  description = """Score estimated session cost against a maximum (0.0-1.0)."""
)
AS r"""
    def bqaa_score_cost(input_tokens, output_tokens, max_cost_usd,
                        input_cost_per_1k, output_cost_per_1k):
        cost = ((input_tokens / 1000) * input_cost_per_1k
                + (output_tokens / 1000) * output_cost_per_1k)
        if cost <= 0:
            return 1.0
        if cost >= max_cost_usd:
            return 0.0
        return 1.0 - (cost / max_cost_usd)
""";
