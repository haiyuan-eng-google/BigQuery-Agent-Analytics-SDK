# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SQL template generation for BigQuery Python UDFs.

Generates ``CREATE OR REPLACE FUNCTION`` DDL for registering SDK
analytical kernels as BigQuery Python UDFs.  Each UDF inlines the
kernel body from :mod:`udf_kernels` so that the deployed function
has zero runtime dependencies — no ``pip install`` required.

Usage::

    from bigquery_agent_analytics.udf_sql_templates import (
        generate_all_udfs,
        generate_udf,
    )

    # All UDFs for a dataset
    sql = generate_all_udfs("my-project", "analytics")

    # Single UDF
    sql = generate_udf("bqaa_score_latency", "my-project", "analytics")
"""

from __future__ import annotations

from dataclasses import dataclass
import textwrap
from typing import Optional


@dataclass(frozen=True)
class _UdfSpec:
  """Specification for a single Python UDF."""

  name: str
  params: str
  return_type: str
  body: str
  description: str


# ------------------------------------------------------------------ #
# Tier 1: Event Semantics UDFs                                        #
# ------------------------------------------------------------------ #

_IS_ERROR_EVENT = _UdfSpec(
    name="bqaa_is_error_event",
    params="event_type STRING, error_message STRING, status STRING",
    return_type="BOOL",
    description="Returns TRUE when the event represents an error.",
    body=textwrap.dedent(
        """\
        def bqaa_is_error_event(event_type, error_message, status):
            return (
                event_type.endswith("_ERROR")
                or error_message is not None
                or status == "ERROR"
            )
    """
    ),
)

_TOOL_OUTCOME = _UdfSpec(
    name="bqaa_tool_outcome",
    params="event_type STRING, status STRING",
    return_type="STRING",
    description=(
        "Returns a canonical tool outcome: "
        "'success', 'error', or 'in_progress'."
    ),
    body=textwrap.dedent(
        """\
        def bqaa_tool_outcome(event_type, status):
            if event_type == "TOOL_ERROR" or status == "ERROR":
                return "error"
            if event_type == "TOOL_COMPLETED":
                return "success"
            return "in_progress"
    """
    ),
)

_EXTRACT_RESPONSE_TEXT = _UdfSpec(
    name="bqaa_extract_response_text",
    params="content_json STRING",
    return_type="STRING",
    description=(
        "Extracts user-visible response text from a JSON content string."
    ),
    body=textwrap.dedent(
        """\
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
    """
    ),
)

# ------------------------------------------------------------------ #
# Tier 2: Score Kernel UDFs                                            #
# ------------------------------------------------------------------ #

_SCORE_LATENCY = _UdfSpec(
    name="bqaa_score_latency",
    params="avg_latency_ms FLOAT64, threshold_ms FLOAT64",
    return_type="FLOAT64",
    description="Score average latency against a threshold (0.0-1.0).",
    body=textwrap.dedent(
        """\
        def bqaa_score_latency(avg_latency_ms, threshold_ms):
            if avg_latency_ms <= 0:
                return 1.0
            if avg_latency_ms >= threshold_ms:
                return 0.0
            return 1.0 - (avg_latency_ms / threshold_ms)
    """
    ),
)

_SCORE_ERROR_RATE = _UdfSpec(
    name="bqaa_score_error_rate",
    params="tool_calls INT64, tool_errors INT64, max_error_rate FLOAT64",
    return_type="FLOAT64",
    description="Score tool error rate against a threshold (0.0-1.0).",
    body=textwrap.dedent(
        """\
        def bqaa_score_error_rate(tool_calls, tool_errors, max_error_rate):
            if tool_calls <= 0:
                return 1.0
            rate = tool_errors / tool_calls
            if rate >= max_error_rate:
                return 0.0
            return 1.0 - (rate / max_error_rate)
    """
    ),
)

_SCORE_TURN_COUNT = _UdfSpec(
    name="bqaa_score_turn_count",
    params="turn_count INT64, max_turns INT64",
    return_type="FLOAT64",
    description="Score turn count against a maximum (0.0-1.0).",
    body=textwrap.dedent(
        """\
        def bqaa_score_turn_count(turn_count, max_turns):
            if turn_count <= 0:
                return 1.0
            if turn_count >= max_turns:
                return 0.0
            return 1.0 - (turn_count / max_turns)
    """
    ),
)

_SCORE_TOKEN_EFFICIENCY = _UdfSpec(
    name="bqaa_score_token_efficiency",
    params="total_tokens INT64, max_tokens INT64",
    return_type="FLOAT64",
    description="Score total token usage against a maximum (0.0-1.0).",
    body=textwrap.dedent(
        """\
        def bqaa_score_token_efficiency(total_tokens, max_tokens):
            if total_tokens <= 0:
                return 1.0
            if total_tokens >= max_tokens:
                return 0.0
            return 1.0 - (total_tokens / max_tokens)
    """
    ),
)

_SCORE_TTFT = _UdfSpec(
    name="bqaa_score_ttft",
    params="avg_ttft_ms FLOAT64, threshold_ms FLOAT64",
    return_type="FLOAT64",
    description=(
        "Score average time-to-first-token against a threshold (0.0-1.0)."
    ),
    body=textwrap.dedent(
        """\
        def bqaa_score_ttft(avg_ttft_ms, threshold_ms):
            if avg_ttft_ms <= 0:
                return 1.0
            if avg_ttft_ms >= threshold_ms:
                return 0.0
            return 1.0 - (avg_ttft_ms / threshold_ms)
    """
    ),
)

_SCORE_COST = _UdfSpec(
    name="bqaa_score_cost",
    params=(
        "input_tokens INT64, output_tokens INT64,"
        " max_cost_usd FLOAT64,"
        " input_cost_per_1k FLOAT64, output_cost_per_1k FLOAT64"
    ),
    return_type="FLOAT64",
    description="Score estimated session cost against a maximum (0.0-1.0).",
    body=textwrap.dedent(
        """\
        def bqaa_score_cost(input_tokens, output_tokens, max_cost_usd,
                            input_cost_per_1k, output_cost_per_1k):
            cost = ((input_tokens / 1000) * input_cost_per_1k
                    + (output_tokens / 1000) * output_cost_per_1k)
            if cost <= 0:
                return 1.0
            if cost >= max_cost_usd:
                return 0.0
            return 1.0 - (cost / max_cost_usd)
    """
    ),
)

# ------------------------------------------------------------------ #
# Registry                                                             #
# ------------------------------------------------------------------ #

ALL_UDFS: list[_UdfSpec] = [
    # Tier 1: event semantics
    _IS_ERROR_EVENT,
    _TOOL_OUTCOME,
    _EXTRACT_RESPONSE_TEXT,
    # Tier 2: score kernels
    _SCORE_LATENCY,
    _SCORE_ERROR_RATE,
    _SCORE_TURN_COUNT,
    _SCORE_TOKEN_EFFICIENCY,
    _SCORE_TTFT,
    _SCORE_COST,
]

UDF_NAMES: list[str] = [u.name for u in ALL_UDFS]


def _render_udf(
    spec: _UdfSpec,
    project: str,
    dataset: str,
) -> str:
  """Render CREATE OR REPLACE FUNCTION DDL for a single UDF."""
  # Indent body for the OPTIONS block
  indented = textwrap.indent(spec.body.rstrip(), "    ")
  return (
      f"-- {spec.description}\n"
      f"CREATE OR REPLACE FUNCTION"
      f" `{project}.{dataset}.{spec.name}`(\n"
      f"  {spec.params}\n"
      f")\n"
      f"RETURNS {spec.return_type}\n"
      f"LANGUAGE python\n"
      f"OPTIONS (\n"
      f"  entry_point = '{spec.name}',\n"
      f"  runtime_version = 'python-3.11',\n"
      f'  description = """{spec.description}"""\n'
      f")\n"
      f'AS r"""\n'
      f"{indented}\n"
      f'""";'
  )


def generate_udf(
    name: str,
    project: str,
    dataset: str,
) -> str:
  """Generate CREATE FUNCTION DDL for a single UDF by name.

  Args:
      name: UDF name (e.g. ``"bqaa_score_latency"``).
      project: GCP project ID.
      dataset: BigQuery dataset for the UDF.

  Returns:
      SQL DDL string.

  Raises:
      ValueError: If the name is not a known UDF.
  """
  for spec in ALL_UDFS:
    if spec.name == name:
      return _render_udf(spec, project, dataset)
  raise ValueError(f"Unknown UDF: {name!r}. Known UDFs: {UDF_NAMES}")


def generate_all_udfs(
    project: str,
    dataset: str,
    separator: str = "\n\n",
) -> str:
  """Generate CREATE FUNCTION DDL for all UDFs.

  Args:
      project: GCP project ID.
      dataset: BigQuery dataset for the UDFs.
      separator: String between DDL statements.

  Returns:
      Concatenated SQL DDL for all UDFs.
  """
  parts = [_render_udf(spec, project, dataset) for spec in ALL_UDFS]
  return separator.join(parts)


def list_udfs() -> list[dict[str, str]]:
  """Return metadata for all available UDFs.

  Returns:
      List of dicts with ``name``, ``params``, ``return_type``,
      and ``description`` keys.
  """
  return [
      {
          "name": spec.name,
          "params": spec.params,
          "return_type": spec.return_type,
          "description": spec.description,
      }
      for spec in ALL_UDFS
  ]
