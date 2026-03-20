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

"""Categorical evaluation engine for BigQuery Agent Analytics SDK.

Classifies agent sessions into user-defined categories using BigQuery's
native ``AI.GENERATE``. Unlike the numeric ``CodeEvaluator`` and
``LLMAsJudge`` report paths, this module returns label-valued results
with strict category validation.

Example usage::

    from bigquery_agent_analytics.categorical_evaluator import (
        CategoricalEvaluationConfig,
        CategoricalMetricCategory,
        CategoricalMetricDefinition,
    )

    config = CategoricalEvaluationConfig(
        metrics=[
            CategoricalMetricDefinition(
                name="tone",
                definition="Overall tone of the conversation.",
                categories=[
                    CategoricalMetricCategory(
                        name="positive",
                        definition="User is satisfied.",
                    ),
                    CategoricalMetricCategory(
                        name="negative",
                        definition="User is frustrated.",
                    ),
                    CategoricalMetricCategory(
                        name="neutral",
                        definition="No strong sentiment.",
                    ),
                ],
            ),
        ],
    )

    report = client.evaluate_categorical(config=config)
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from datetime import timezone
import json
import logging
from typing import Any, Optional

from pydantic import BaseModel
from pydantic import Field

logger = logging.getLogger("bigquery_agent_analytics." + __name__)

DEFAULT_ENDPOINT = "gemini-2.5-flash"


# ------------------------------------------------------------------ #
# Configuration Models                                                 #
# ------------------------------------------------------------------ #


class CategoricalMetricCategory(BaseModel):
  """A single allowed category for a categorical metric."""

  name: str = Field(description="Category label.")
  definition: str = Field(description="What this category means.")


class CategoricalMetricDefinition(BaseModel):
  """Definition of one categorical metric to evaluate."""

  name: str = Field(description="Metric name.")
  definition: str = Field(description="What this metric measures.")
  categories: list[CategoricalMetricCategory] = Field(
      description="Allowed categories for this metric.",
  )
  required: bool = Field(
      default=True,
      description="Whether this metric must be classified.",
  )


class CategoricalEvaluationConfig(BaseModel):
  """Configuration for a categorical evaluation run."""

  metrics: list[CategoricalMetricDefinition] = Field(
      description="Metrics to evaluate.",
  )
  endpoint: str = Field(
      default=DEFAULT_ENDPOINT,
      description="Model endpoint for classification.",
  )
  temperature: float = Field(
      default=0.0,
      description="Sampling temperature.",
  )
  persist_results: bool = Field(
      default=False,
      description="Write results to BigQuery.",
  )
  results_table: Optional[str] = Field(
      default=None,
      description="Destination table for results.",
  )
  include_justification: bool = Field(
      default=True,
      description="Include justification in output.",
  )
  prompt_version: Optional[str] = Field(
      default=None,
      description="Tracks prompt version for reproducibility.",
  )


# ------------------------------------------------------------------ #
# Result Models                                                        #
# ------------------------------------------------------------------ #


class CategoricalMetricResult(BaseModel):
  """Classification result for a single metric on a single session."""

  metric_name: str
  category: Optional[str] = None
  passed_validation: bool = True
  justification: Optional[str] = None
  raw_response: Optional[str] = None
  parse_error: bool = False


class CategoricalSessionResult(BaseModel):
  """Classification results for all metrics on a single session."""

  session_id: str
  metrics: list[CategoricalMetricResult] = Field(default_factory=list)
  details: dict[str, Any] = Field(default_factory=dict)


class CategoricalEvaluationReport(BaseModel):
  """Aggregate report from a categorical evaluation run."""

  dataset: str = Field(description="Dataset or filter description.")
  evaluator_name: str = "categorical_evaluator"
  total_sessions: int = 0
  category_distributions: dict[str, dict[str, int]] = Field(
      default_factory=dict,
      description="Maps metric_name -> {category -> count}.",
  )
  details: dict[str, Any] = Field(default_factory=dict)
  session_results: list[CategoricalSessionResult] = Field(
      default_factory=list,
  )
  created_at: datetime = Field(
      default_factory=lambda: datetime.now(timezone.utc),
  )

  def summary(self) -> str:
    """Returns a human-readable summary."""
    lines = [
        f"Categorical Evaluation Report: {self.evaluator_name}",
        f"  Dataset: {self.dataset}",
        f"  Sessions: {self.total_sessions}",
    ]
    parse_errors = self.details.get("parse_errors", 0)
    if parse_errors:
      lines.append(
          f"  Parse errors: {parse_errors}"
          f" ({self.details.get('parse_error_rate', 0):.1%})"
      )
    if self.category_distributions:
      lines.append("  Category Distributions:")
      for metric, dist in sorted(self.category_distributions.items()):
        lines.append(f"    {metric}:")
        for cat, count in sorted(dist.items(), key=lambda x: -x[1]):
          lines.append(f"      {cat}: {count}")
    return "\n".join(lines)


# ------------------------------------------------------------------ #
# SQL Template                                                         #
# ------------------------------------------------------------------ #

CATEGORICAL_AI_GENERATE_QUERY = """\
WITH session_transcripts AS (
  SELECT
    session_id,
    STRING_AGG(
      CONCAT(
        event_type,
        COALESCE(CONCAT(' [', agent, ']'), ''),
        ': ',
        COALESCE(
          JSON_VALUE(content, '$.text_summary'),
          JSON_VALUE(content, '$.response'),
          JSON_VALUE(content, '$.tool'),
          ''
        )
      ),
      '\\n' ORDER BY timestamp
    ) AS transcript
  FROM `{project}.{dataset}.{table}`
  WHERE {where}
  GROUP BY session_id
  HAVING LENGTH(transcript) > 10
  LIMIT @trace_limit
)
SELECT
  session_id,
  transcript,
  (AI.GENERATE(
    CONCAT(
      @categorical_prompt,
      '\\n\\nTranscript:\\n', transcript
    ),
    endpoint => '{endpoint}',
    model_params => JSON '{{"generationConfig": {{"temperature": {temperature}, "maxOutputTokens": 1024}}}}',
    output_schema => 'classifications STRING'
  )).classifications AS classifications
FROM session_transcripts
"""


# ------------------------------------------------------------------ #
# Prompt Builder                                                       #
# ------------------------------------------------------------------ #


def build_categorical_prompt(
    config: CategoricalEvaluationConfig,
) -> str:
  """Builds the classification prompt from metric definitions.

  Args:
      config: Categorical evaluation configuration.

  Returns:
      Prompt string instructing the model to classify the session.
  """
  lines = [
      "You are classifying an agent conversation session.",
      "For each metric below, choose exactly one category from the"
      " allowed set.",
      "Do not invent categories or return free-form labels.",
      "",
  ]

  for metric in config.metrics:
    lines.append(f"## Metric: {metric.name}")
    lines.append(f"Definition: {metric.definition}")
    lines.append("Allowed categories:")
    for cat in metric.categories:
      lines.append(f"  - {cat.name}: {cat.definition}")
    lines.append("")

  if config.include_justification:
    justification_note = (
        'For each metric, include a brief "justification" string'
        " explaining your choice."
    )
  else:
    justification_note = (
        'Do not include a "justification" field in your response.'
    )

  lines.extend(
      [
          justification_note,
          "",
          "Respond with ONLY a valid JSON array. Each element must have:",
          '  - "metric_name": the metric name exactly as shown above',
          '  - "category": one of the allowed categories exactly as shown above',
      ]
  )
  if config.include_justification:
    lines.append('  - "justification": a brief explanation')

  lines.extend(
      [
          "",
          "Example output format:",
      ]
  )
  example = []
  for metric in config.metrics:
    entry: dict[str, str] = {
        "metric_name": metric.name,
        "category": metric.categories[0].name,
    }
    if config.include_justification:
      entry["justification"] = "..."
    example.append(entry)
  lines.append(json.dumps(example, indent=2))

  return "\n".join(lines)


# ------------------------------------------------------------------ #
# Parsing and Validation                                               #
# ------------------------------------------------------------------ #


def _build_category_lookup(
    config: CategoricalEvaluationConfig,
) -> dict[str, dict[str, str]]:
  """Builds a case-insensitive category lookup from config.

  Returns:
      ``{metric_name: {lower_cat_name: canonical_cat_name, ...}, ...}``
  """
  lookup: dict[str, dict[str, str]] = {}
  for metric in config.metrics:
    lookup[metric.name] = {
        cat.name.lower().strip(): cat.name for cat in metric.categories
    }
  return lookup


def parse_classifications(
    raw_json: Optional[str],
    config: CategoricalEvaluationConfig,
) -> list[CategoricalMetricResult]:
  """Parses the JSON STRING envelope and validates categories.

  Args:
      raw_json: Raw JSON string from the ``classifications`` column.
      config: Evaluation config with metric definitions.

  Returns:
      One ``CategoricalMetricResult`` per configured metric.
  """
  lookup = _build_category_lookup(config)
  required_metrics = {m.name for m in config.metrics if m.required}
  all_metrics = {m.name for m in config.metrics}

  if not raw_json or not raw_json.strip():
    return [
        CategoricalMetricResult(
            metric_name=m.name,
            parse_error=True,
            passed_validation=False,
            raw_response=raw_json,
        )
        for m in config.metrics
    ]

  try:
    parsed = json.loads(raw_json)
  except (json.JSONDecodeError, TypeError):
    return [
        CategoricalMetricResult(
            metric_name=m.name,
            parse_error=True,
            passed_validation=False,
            raw_response=raw_json,
        )
        for m in config.metrics
    ]

  if not isinstance(parsed, list):
    parsed = [parsed]

  results_by_metric: dict[str, CategoricalMetricResult] = {}

  for entry in parsed:
    if not isinstance(entry, dict):
      continue

    metric_name = entry.get("metric_name", "")
    if metric_name not in all_metrics:
      continue

    # Duplicate metric entries are malformed — the prompt asks for
    # exactly one category per metric.  Flag as a parse error.
    if metric_name in results_by_metric:
      results_by_metric[metric_name] = CategoricalMetricResult(
          metric_name=metric_name,
          passed_validation=False,
          parse_error=True,
          raw_response=raw_json,
      )
      continue

    raw_category = str(entry.get("category", "")).lower().strip()
    canonical = lookup.get(metric_name, {}).get(raw_category)

    if canonical is not None:
      results_by_metric[metric_name] = CategoricalMetricResult(
          metric_name=metric_name,
          category=canonical,
          passed_validation=True,
          justification=entry.get("justification"),
          raw_response=raw_json,
      )
    else:
      results_by_metric[metric_name] = CategoricalMetricResult(
          metric_name=metric_name,
          category=entry.get("category"),
          passed_validation=False,
          parse_error=True,
          justification=entry.get("justification"),
          raw_response=raw_json,
      )

  # Fill in missing metrics.
  for metric in config.metrics:
    if metric.name not in results_by_metric:
      results_by_metric[metric.name] = CategoricalMetricResult(
          metric_name=metric.name,
          parse_error=metric.name in required_metrics,
          passed_validation=metric.name not in required_metrics,
          raw_response=raw_json,
      )

  return [results_by_metric[m.name] for m in config.metrics]


def parse_categorical_row(
    session_id: str,
    row: dict[str, Any],
    config: CategoricalEvaluationConfig,
) -> CategoricalSessionResult:
  """Parses a BigQuery result row into a CategoricalSessionResult.

  Args:
      session_id: The session ID.
      row: Dict from ``dict(bigquery_row)`` containing at least
          a ``classifications`` STRING column.
      config: Evaluation config with metric definitions.

  Returns:
      CategoricalSessionResult with validated metric results.
  """
  raw = row.get("classifications")
  metrics = parse_classifications(raw, config)
  return CategoricalSessionResult(
      session_id=session_id,
      metrics=metrics,
  )


# ------------------------------------------------------------------ #
# Report Builder                                                       #
# ------------------------------------------------------------------ #


def build_categorical_report(
    dataset: str,
    session_results: list[CategoricalSessionResult],
    config: CategoricalEvaluationConfig,
) -> CategoricalEvaluationReport:
  """Builds an aggregate report from per-session results.

  Args:
      dataset: Dataset description for the report.
      session_results: Per-session classification results.
      config: Evaluation config.

  Returns:
      CategoricalEvaluationReport with distributions and details.
  """
  distributions: dict[str, Counter] = {
      m.name: Counter() for m in config.metrics
  }
  parse_error_count = 0

  for sr in session_results:
    for mr in sr.metrics:
      if mr.parse_error:
        parse_error_count += 1
      if mr.category is not None:
        distributions[mr.metric_name][mr.category] += 1

  total_classifications = len(session_results) * len(config.metrics)
  parse_error_rate = (
      parse_error_count / total_classifications
      if total_classifications > 0
      else 0.0
  )

  return CategoricalEvaluationReport(
      dataset=dataset,
      total_sessions=len(session_results),
      category_distributions={
          name: dict(counter) for name, counter in distributions.items()
      },
      details={
          "parse_errors": parse_error_count,
          "parse_error_rate": parse_error_rate,
      },
      session_results=session_results,
  )
