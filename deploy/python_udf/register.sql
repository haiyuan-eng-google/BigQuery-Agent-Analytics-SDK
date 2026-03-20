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
-- This file registers Tier 1 (event semantics), Tier 2 (score kernel),
-- and Tier 4 (STRING envelope) UDFs.  Each UDF inlines its
-- kernel body so there are no
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


-- ------------------------------------------------------------------ --
-- Tier 1b: Event Label Normalization                                   --
-- ------------------------------------------------------------------ --

-- Normalize event_type to a high-level category (llm, tool, user, agent, other).
CREATE OR REPLACE FUNCTION `PROJECT.UDF_DATASET.bqaa_normalize_event_label`(
  event_type STRING
)
RETURNS STRING
LANGUAGE python
OPTIONS (
  entry_point = 'bqaa_normalize_event_label',
  runtime_version = 'python-3.11',
  description = """Normalize event_type to a high-level category (llm, tool, user, agent, other)."""
)
AS r"""
_EVENT_LABEL_MAP = {
    "LLM_REQUEST": "llm",
    "LLM_RESPONSE": "llm",
    "TOOL_STARTING": "tool",
    "TOOL_COMPLETED": "tool",
    "TOOL_ERROR": "tool_error",
    "USER_MESSAGE_RECEIVED": "user",
    "AGENT_COMPLETED": "agent",
}

def bqaa_normalize_event_label(event_type):
    return _EVENT_LABEL_MAP.get(event_type, "other")
""";


-- ------------------------------------------------------------------ --
-- Tier 4: STRING Envelope UDFs                                         --
-- ------------------------------------------------------------------ --

-- Compute all six scores and return a JSON STRING summary.
CREATE OR REPLACE FUNCTION `PROJECT.UDF_DATASET.bqaa_eval_summary_json`(
  avg_latency_ms FLOAT64,
  tool_calls INT64, tool_errors INT64,
  turn_count INT64, total_tokens INT64,
  avg_ttft_ms FLOAT64,
  input_tokens INT64, output_tokens INT64,
  threshold_ms FLOAT64,
  max_error_rate FLOAT64,
  max_turns INT64, max_tokens INT64,
  ttft_threshold_ms FLOAT64,
  max_cost_usd FLOAT64,
  input_cost_per_1k FLOAT64,
  output_cost_per_1k FLOAT64
)
RETURNS STRING
LANGUAGE python
OPTIONS (
  entry_point = 'bqaa_eval_summary_json',
  runtime_version = 'python-3.11',
  description = """Compute all six scores and return a JSON STRING summary."""
)
AS r"""
import json

def _score_latency(avg, thresh):
    if avg <= 0:
        return 1.0
    if avg >= thresh:
        return 0.0
    return 1.0 - (avg / thresh)

def _score_error_rate(calls, errors, max_rate):
    if calls <= 0:
        return 1.0
    rate = errors / calls
    if rate >= max_rate:
        return 0.0
    return 1.0 - (rate / max_rate)

def _score_turn_count(turns, max_t):
    if turns <= 0:
        return 1.0
    if turns >= max_t:
        return 0.0
    return 1.0 - (turns / max_t)

def _score_token_efficiency(tokens, max_t):
    if tokens <= 0:
        return 1.0
    if tokens >= max_t:
        return 0.0
    return 1.0 - (tokens / max_t)

def _score_ttft(avg, thresh):
    if avg <= 0:
        return 1.0
    if avg >= thresh:
        return 0.0
    return 1.0 - (avg / thresh)

def _score_cost(inp, out, max_c, ic, oc):
    cost = (inp / 1000) * ic + (out / 1000) * oc
    if cost <= 0:
        return 1.0
    if cost >= max_c:
        return 0.0
    return 1.0 - (cost / max_c)

def bqaa_eval_summary_json(
        avg_latency_ms, tool_calls, tool_errors,
        turn_count, total_tokens, avg_ttft_ms,
        input_tokens, output_tokens,
        threshold_ms, max_error_rate,
        max_turns, max_tokens, ttft_threshold_ms,
        max_cost_usd, input_cost_per_1k, output_cost_per_1k):
    scores = {
        "latency": _score_latency(avg_latency_ms, threshold_ms),
        "error_rate": _score_error_rate(
            tool_calls, tool_errors, max_error_rate),
        "turn_count": _score_turn_count(turn_count, max_turns),
        "token_efficiency": _score_token_efficiency(
            total_tokens, max_tokens),
        "ttft": _score_ttft(avg_ttft_ms, ttft_threshold_ms),
        "cost": _score_cost(
            input_tokens, output_tokens,
            max_cost_usd, input_cost_per_1k, output_cost_per_1k),
    }
    scores["passed"] = all(v >= 0.5 for v in scores.values())
    return json.dumps(scores)
""";
