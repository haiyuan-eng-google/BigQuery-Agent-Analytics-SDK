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

"""Tests for the categorical evaluator module."""

import json

import pytest

from bigquery_agent_analytics.categorical_evaluator import build_categorical_prompt
from bigquery_agent_analytics.categorical_evaluator import build_categorical_report
from bigquery_agent_analytics.categorical_evaluator import CATEGORICAL_AI_GENERATE_QUERY
from bigquery_agent_analytics.categorical_evaluator import CategoricalEvaluationConfig
from bigquery_agent_analytics.categorical_evaluator import CategoricalEvaluationReport
from bigquery_agent_analytics.categorical_evaluator import CategoricalMetricCategory
from bigquery_agent_analytics.categorical_evaluator import CategoricalMetricDefinition
from bigquery_agent_analytics.categorical_evaluator import CategoricalMetricResult
from bigquery_agent_analytics.categorical_evaluator import CategoricalSessionResult
from bigquery_agent_analytics.categorical_evaluator import parse_categorical_row
from bigquery_agent_analytics.categorical_evaluator import parse_classifications

# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #


def _make_config(include_justification=True):
  """Builds a two-metric config for testing."""
  return CategoricalEvaluationConfig(
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
          CategoricalMetricDefinition(
              name="safety",
              definition="Whether the response is safe.",
              categories=[
                  CategoricalMetricCategory(
                      name="safe",
                      definition="Response is safe.",
                  ),
                  CategoricalMetricCategory(
                      name="unsafe",
                      definition="Response contains unsafe content.",
                  ),
              ],
          ),
      ],
      include_justification=include_justification,
  )


# ------------------------------------------------------------------ #
# Model Tests                                                          #
# ------------------------------------------------------------------ #


class TestCategoricalModels:
  """Tests for Pydantic config and result models."""

  def test_metric_category_fields(self):
    cat = CategoricalMetricCategory(name="good", definition="It is good.")
    assert cat.name == "good"
    assert cat.definition == "It is good."

  def test_metric_definition_defaults(self):
    defn = CategoricalMetricDefinition(
        name="tone",
        definition="Tone.",
        categories=[
            CategoricalMetricCategory(name="a", definition="A."),
        ],
    )
    assert defn.required is True

  def test_config_defaults(self):
    config = _make_config()
    assert config.endpoint == "gemini-2.5-flash"
    assert config.temperature == 0.0
    assert config.persist_results is False
    assert config.include_justification is True
    assert config.prompt_version is None
    assert config.results_table is None

  def test_metric_result_defaults(self):
    result = CategoricalMetricResult(metric_name="tone")
    assert result.category is None
    assert result.passed_validation is True
    assert result.parse_error is False
    assert result.justification is None
    assert result.raw_response is None

  def test_session_result_defaults(self):
    sr = CategoricalSessionResult(session_id="s1")
    assert sr.metrics == []
    assert sr.details == {}

  def test_report_defaults(self):
    report = CategoricalEvaluationReport(dataset="test")
    assert report.total_sessions == 0
    assert report.evaluator_name == "categorical_evaluator"
    assert report.category_distributions == {}
    assert report.details == {}
    assert report.session_results == []
    assert report.created_at is not None


# ------------------------------------------------------------------ #
# Prompt Builder Tests                                                 #
# ------------------------------------------------------------------ #


class TestBuildCategoricalPrompt:
  """Tests for build_categorical_prompt."""

  def test_includes_metric_names(self):
    prompt = build_categorical_prompt(_make_config())
    assert "tone" in prompt
    assert "safety" in prompt

  def test_includes_category_names(self):
    prompt = build_categorical_prompt(_make_config())
    assert "positive" in prompt
    assert "negative" in prompt
    assert "neutral" in prompt
    assert "safe" in prompt
    assert "unsafe" in prompt

  def test_includes_definitions(self):
    prompt = build_categorical_prompt(_make_config())
    assert "User is satisfied" in prompt
    assert "Whether the response is safe" in prompt

  def test_includes_json_format_instruction(self):
    prompt = build_categorical_prompt(_make_config())
    assert "JSON array" in prompt
    assert "metric_name" in prompt
    assert "category" in prompt

  def test_includes_example(self):
    prompt = build_categorical_prompt(_make_config())
    # The example should be valid JSON.
    example_start = prompt.rfind("[")
    example_end = prompt.rfind("]") + 1
    example = json.loads(prompt[example_start:example_end])
    assert len(example) == 2
    assert example[0]["metric_name"] == "tone"

  def test_no_justification(self):
    prompt = build_categorical_prompt(_make_config(include_justification=False))
    assert "Do not include" in prompt
    # The output spec after the instruction lines should not list
    # justification as a required field.
    after_spec = prompt.split("Each element must have:")[1]
    spec_lines = after_spec.split("Example")[0]
    assert '"justification"' not in spec_lines


# ------------------------------------------------------------------ #
# Parse Classifications Tests                                          #
# ------------------------------------------------------------------ #


class TestParseClassifications:
  """Tests for parse_classifications."""

  def test_valid_json(self):
    config = _make_config()
    raw = json.dumps(
        [
            {
                "metric_name": "tone",
                "category": "positive",
                "justification": "kind",
            },
            {
                "metric_name": "safety",
                "category": "safe",
                "justification": "ok",
            },
        ]
    )
    results = parse_classifications(raw, config)
    assert len(results) == 2
    assert results[0].metric_name == "tone"
    assert results[0].category == "positive"
    assert results[0].passed_validation is True
    assert results[0].parse_error is False
    assert results[0].justification == "kind"
    assert results[1].metric_name == "safety"
    assert results[1].category == "safe"

  def test_invalid_category(self):
    config = _make_config()
    raw = json.dumps(
        [
            {"metric_name": "tone", "category": "unknown_val"},
            {"metric_name": "safety", "category": "safe"},
        ]
    )
    results = parse_classifications(raw, config)
    tone = results[0]
    assert tone.parse_error is True
    assert tone.passed_validation is False
    safety = results[1]
    assert safety.parse_error is False
    assert safety.passed_validation is True

  def test_missing_metric(self):
    config = _make_config()
    raw = json.dumps(
        [
            {"metric_name": "tone", "category": "positive"},
        ]
    )
    results = parse_classifications(raw, config)
    assert len(results) == 2
    safety = results[1]
    assert safety.metric_name == "safety"
    assert safety.parse_error is True
    assert safety.passed_validation is False

  def test_malformed_json(self):
    config = _make_config()
    results = parse_classifications("not json at all", config)
    assert len(results) == 2
    assert all(r.parse_error is True for r in results)
    assert all(r.passed_validation is False for r in results)

  def test_empty_input(self):
    config = _make_config()
    results = parse_classifications("", config)
    assert len(results) == 2
    assert all(r.parse_error is True for r in results)

  def test_none_input(self):
    config = _make_config()
    results = parse_classifications(None, config)
    assert len(results) == 2
    assert all(r.parse_error is True for r in results)

  def test_case_insensitive(self):
    config = _make_config()
    raw = json.dumps(
        [
            {"metric_name": "tone", "category": "POSITIVE"},
            {"metric_name": "safety", "category": "Safe"},
        ]
    )
    results = parse_classifications(raw, config)
    assert results[0].category == "positive"
    assert results[0].passed_validation is True
    assert results[1].category == "safe"
    assert results[1].passed_validation is True

  def test_extra_whitespace(self):
    config = _make_config()
    raw = json.dumps(
        [
            {"metric_name": "tone", "category": "  positive  "},
            {"metric_name": "safety", "category": "safe"},
        ]
    )
    results = parse_classifications(raw, config)
    assert results[0].category == "positive"
    assert results[0].passed_validation is True

  def test_unknown_metric_ignored(self):
    config = _make_config()
    raw = json.dumps(
        [
            {"metric_name": "tone", "category": "positive"},
            {"metric_name": "safety", "category": "safe"},
            {"metric_name": "bogus", "category": "whatever"},
        ]
    )
    results = parse_classifications(raw, config)
    assert len(results) == 2

  def test_duplicate_metric_flagged_as_error(self):
    config = _make_config()
    raw = json.dumps(
        [
            {"metric_name": "tone", "category": "positive"},
            {"metric_name": "tone", "category": "negative"},
            {"metric_name": "safety", "category": "safe"},
        ]
    )
    results = parse_classifications(raw, config)
    tone = results[0]
    assert tone.parse_error is True
    assert tone.passed_validation is False
    # The duplicate should wipe the category — it's ambiguous.
    assert tone.category is None
    # safety should be unaffected.
    safety = results[1]
    assert safety.category == "safe"
    assert safety.passed_validation is True

  def test_single_object_not_array(self):
    config = CategoricalEvaluationConfig(
        metrics=[
            CategoricalMetricDefinition(
                name="tone",
                definition="Tone.",
                categories=[
                    CategoricalMetricCategory(
                        name="positive",
                        definition="Good.",
                    ),
                ],
            ),
        ],
    )
    raw = json.dumps({"metric_name": "tone", "category": "positive"})
    results = parse_classifications(raw, config)
    assert len(results) == 1
    assert results[0].category == "positive"


# ------------------------------------------------------------------ #
# Parse Row Tests                                                      #
# ------------------------------------------------------------------ #


class TestParseCategoricalRow:
  """Tests for parse_categorical_row."""

  def test_valid_row(self):
    config = _make_config()
    raw = json.dumps(
        [
            {"metric_name": "tone", "category": "positive"},
            {"metric_name": "safety", "category": "safe"},
        ]
    )
    row = {
        "session_id": "s1",
        "transcript": "some text",
        "classifications": raw,
    }
    result = parse_categorical_row("s1", row, config)
    assert result.session_id == "s1"
    assert len(result.metrics) == 2
    assert result.metrics[0].category == "positive"
    assert result.metrics[1].category == "safe"

  def test_missing_classifications_column(self):
    config = _make_config()
    row = {"session_id": "s1", "transcript": "text"}
    result = parse_categorical_row("s1", row, config)
    assert len(result.metrics) == 2
    assert all(m.parse_error is True for m in result.metrics)


# ------------------------------------------------------------------ #
# Report Builder Tests                                                 #
# ------------------------------------------------------------------ #


class TestBuildCategoricalReport:
  """Tests for build_categorical_report."""

  def test_aggregation(self):
    config = _make_config()
    sessions = [
        CategoricalSessionResult(
            session_id="s1",
            metrics=[
                CategoricalMetricResult(
                    metric_name="tone", category="positive"
                ),
                CategoricalMetricResult(metric_name="safety", category="safe"),
            ],
        ),
        CategoricalSessionResult(
            session_id="s2",
            metrics=[
                CategoricalMetricResult(
                    metric_name="tone", category="positive"
                ),
                CategoricalMetricResult(
                    metric_name="safety", category="unsafe"
                ),
            ],
        ),
        CategoricalSessionResult(
            session_id="s3",
            metrics=[
                CategoricalMetricResult(
                    metric_name="tone", category="negative"
                ),
                CategoricalMetricResult(metric_name="safety", category="safe"),
            ],
        ),
    ]

    report = build_categorical_report("test_ds", sessions, config)
    assert report.total_sessions == 3
    assert report.category_distributions["tone"]["positive"] == 2
    assert report.category_distributions["tone"]["negative"] == 1
    assert report.category_distributions["safety"]["safe"] == 2
    assert report.category_distributions["safety"]["unsafe"] == 1
    assert report.details["parse_errors"] == 0
    assert report.details["parse_error_rate"] == 0.0

  def test_parse_error_counting(self):
    config = _make_config()
    sessions = [
        CategoricalSessionResult(
            session_id="s1",
            metrics=[
                CategoricalMetricResult(
                    metric_name="tone",
                    category="positive",
                ),
                CategoricalMetricResult(
                    metric_name="safety",
                    parse_error=True,
                    passed_validation=False,
                ),
            ],
        ),
    ]
    report = build_categorical_report("test_ds", sessions, config)
    assert report.details["parse_errors"] == 1
    # 1 error out of 2 total classifications.
    assert report.details["parse_error_rate"] == 0.5

  def test_empty_sessions(self):
    config = _make_config()
    report = build_categorical_report("test_ds", [], config)
    assert report.total_sessions == 0
    assert report.details["parse_errors"] == 0
    assert report.details["parse_error_rate"] == 0.0

  def test_summary(self):
    config = _make_config()
    sessions = [
        CategoricalSessionResult(
            session_id="s1",
            metrics=[
                CategoricalMetricResult(
                    metric_name="tone", category="positive"
                ),
                CategoricalMetricResult(metric_name="safety", category="safe"),
            ],
        ),
    ]
    report = build_categorical_report("test_ds", sessions, config)
    text = report.summary()
    assert "categorical_evaluator" in text
    assert "tone" in text
    assert "positive" in text


# ------------------------------------------------------------------ #
# SQL Template Tests                                                   #
# ------------------------------------------------------------------ #


class TestCategoricalAIGenerateQuery:
  """Tests for the SQL template constant."""

  def test_contains_ai_generate(self):
    assert "AI.GENERATE" in CATEGORICAL_AI_GENERATE_QUERY

  def test_contains_output_schema(self):
    assert "output_schema" in CATEGORICAL_AI_GENERATE_QUERY

  def test_contains_classifications_string(self):
    assert "classifications STRING" in CATEGORICAL_AI_GENERATE_QUERY

  def test_contains_endpoint_placeholder(self):
    assert "{endpoint}" in CATEGORICAL_AI_GENERATE_QUERY

  def test_does_not_use_legacy_ml_generate(self):
    assert "ML.GENERATE_TEXT" not in CATEGORICAL_AI_GENERATE_QUERY

  def test_scalar_function_shape(self):
    """AI.GENERATE is a scalar function — prompt is a positional arg,
    result is accessed via .classifications on the returned STRUCT."""
    assert ")).classifications" in CATEGORICAL_AI_GENERATE_QUERY

  def test_generation_config_format(self):
    """model_params must use GenerateContent API format."""
    assert "generationConfig" in CATEGORICAL_AI_GENERATE_QUERY
    assert "maxOutputTokens" in CATEGORICAL_AI_GENERATE_QUERY

  def test_not_table_valued(self):
    """Must NOT use the table-valued FROM ... AI.GENERATE(...) AS result
    syntax — that form does not exist in BigQuery."""
    assert "FROM session_transcripts," not in CATEGORICAL_AI_GENERATE_QUERY
    assert ") AS result" not in CATEGORICAL_AI_GENERATE_QUERY

  def test_format_succeeds(self):
    formatted = CATEGORICAL_AI_GENERATE_QUERY.format(
        project="p",
        dataset="d",
        table="t",
        where="1=1",
        endpoint="gemini-2.5-flash",
        temperature=0.0,
    )
    assert "p.d.t" in formatted
    assert "gemini-2.5-flash" in formatted
