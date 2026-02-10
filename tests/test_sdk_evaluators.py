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

"""Tests for the SDK evaluators module."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from bigquery_agent_analytics.evaluators import _parse_json_from_text
from bigquery_agent_analytics.evaluators import AI_GENERATE_JUDGE_BATCH_QUERY
from bigquery_agent_analytics.evaluators import CodeEvaluator
from bigquery_agent_analytics.evaluators import DEFAULT_ENDPOINT
from bigquery_agent_analytics.evaluators import EvaluationReport
from bigquery_agent_analytics.evaluators import LLM_JUDGE_BATCH_QUERY
from bigquery_agent_analytics.evaluators import LLMAsJudge
from bigquery_agent_analytics.evaluators import SESSION_SUMMARY_QUERY
from bigquery_agent_analytics.evaluators import SessionScore
import pytest


class TestCodeEvaluator:
  """Tests for CodeEvaluator class."""

  def test_custom_metric(self):
    evaluator = CodeEvaluator(name="test")
    evaluator.add_metric(
        name="custom",
        fn=lambda s: 0.8,
        threshold=0.5,
    )

    summary = {"session_id": "s1", "tool_calls": 5}
    score = evaluator.evaluate_session(summary)

    assert score.session_id == "s1"
    assert score.scores["custom"] == 0.8
    assert score.passed is True

  def test_custom_metric_fail(self):
    evaluator = CodeEvaluator(name="test")
    evaluator.add_metric(
        name="custom",
        fn=lambda s: 0.2,
        threshold=0.5,
    )

    score = evaluator.evaluate_session({"session_id": "s1"})

    assert score.scores["custom"] == 0.2
    assert score.passed is False

  def test_metric_exception_handled(self):
    evaluator = CodeEvaluator(name="test")
    evaluator.add_metric(
        name="broken",
        fn=lambda s: 1 / 0,
        threshold=0.5,
    )

    score = evaluator.evaluate_session({"session_id": "s1"})

    assert score.scores["broken"] == 0.0
    assert score.passed is False

  def test_metric_clamping(self):
    evaluator = CodeEvaluator(name="test")
    evaluator.add_metric(
        name="over",
        fn=lambda s: 1.5,
        threshold=0.5,
    )
    evaluator.add_metric(
        name="under",
        fn=lambda s: -0.5,
        threshold=0.0,
    )

    score = evaluator.evaluate_session({"session_id": "s1"})

    assert score.scores["over"] == 1.0
    assert score.scores["under"] == 0.0

  def test_chaining(self):
    evaluator = (
        CodeEvaluator(name="chain")
        .add_metric("a", lambda s: 0.9)
        .add_metric("b", lambda s: 0.7)
    )
    score = evaluator.evaluate_session({"session_id": "s1"})
    assert "a" in score.scores
    assert "b" in score.scores


class TestCodeEvaluatorPrebuilt:
  """Tests for pre-built CodeEvaluator factories."""

  def test_latency_pass(self):
    evaluator = CodeEvaluator.latency(threshold_ms=5000)
    score = evaluator.evaluate_session({
        "session_id": "s1",
        "avg_latency_ms": 2000,
    })
    assert score.passed is True
    assert score.scores["latency"] > 0.5

  def test_latency_fail(self):
    evaluator = CodeEvaluator.latency(threshold_ms=1000)
    score = evaluator.evaluate_session({
        "session_id": "s1",
        "avg_latency_ms": 2000,
    })
    assert score.passed is False
    assert score.scores["latency"] == 0.0

  def test_latency_zero(self):
    evaluator = CodeEvaluator.latency(threshold_ms=5000)
    score = evaluator.evaluate_session({
        "session_id": "s1",
        "avg_latency_ms": 0,
    })
    assert score.scores["latency"] == 1.0

  def test_turn_count_pass(self):
    evaluator = CodeEvaluator.turn_count(max_turns=10)
    score = evaluator.evaluate_session({
        "session_id": "s1",
        "turn_count": 3,
    })
    assert score.passed is True
    assert score.scores["turn_count"] > 0.5

  def test_turn_count_fail(self):
    evaluator = CodeEvaluator.turn_count(max_turns=5)
    score = evaluator.evaluate_session({
        "session_id": "s1",
        "turn_count": 8,
    })
    assert score.passed is False

  def test_error_rate_pass(self):
    evaluator = CodeEvaluator.error_rate(max_error_rate=0.1)
    score = evaluator.evaluate_session({
        "session_id": "s1",
        "tool_calls": 20,
        "tool_errors": 1,
    })
    assert score.passed is True

  def test_error_rate_fail(self):
    evaluator = CodeEvaluator.error_rate(max_error_rate=0.1)
    score = evaluator.evaluate_session({
        "session_id": "s1",
        "tool_calls": 10,
        "tool_errors": 5,
    })
    assert score.passed is False

  def test_error_rate_no_calls(self):
    evaluator = CodeEvaluator.error_rate(max_error_rate=0.1)
    score = evaluator.evaluate_session({
        "session_id": "s1",
        "tool_calls": 0,
        "tool_errors": 0,
    })
    assert score.scores["error_rate"] == 1.0


class TestLLMAsJudgePrebuilt:
  """Tests for pre-built LLMAsJudge factories."""

  def test_correctness_factory(self):
    judge = LLMAsJudge.correctness(threshold=0.7)
    assert judge.name == "correctness_judge"
    assert len(judge._criteria) == 1
    assert judge._criteria[0].name == "correctness"
    assert judge._criteria[0].threshold == 0.7

  def test_hallucination_factory(self):
    judge = LLMAsJudge.hallucination()
    assert judge.name == "hallucination_judge"
    assert judge._criteria[0].name == "faithfulness"

  def test_sentiment_factory(self):
    judge = LLMAsJudge.sentiment()
    assert judge.name == "sentiment_judge"
    assert judge._criteria[0].name == "sentiment"

  def test_custom_criterion(self):
    judge = LLMAsJudge(name="custom")
    judge.add_criterion(
        name="helpfulness",
        prompt_template="Rate helpfulness: {trace_text} {final_response}",
        score_key="helpfulness",
        threshold=0.6,
    )
    assert len(judge._criteria) == 1
    assert judge._criteria[0].name == "helpfulness"

  def test_chaining(self):
    judge = (
        LLMAsJudge(name="multi")
        .add_criterion("a", "p1 {trace_text} {final_response}", "a")
        .add_criterion("b", "p2 {trace_text} {final_response}", "b")
    )
    assert len(judge._criteria) == 2


class TestEvaluationReport:
  """Tests for EvaluationReport class."""

  def test_pass_rate(self):
    report = EvaluationReport(
        dataset="test",
        evaluator_name="test_eval",
        total_sessions=10,
        passed_sessions=7,
        failed_sessions=3,
    )
    assert report.pass_rate == 0.7

  def test_pass_rate_zero(self):
    report = EvaluationReport(
        dataset="test",
        evaluator_name="test_eval",
        total_sessions=0,
        passed_sessions=0,
        failed_sessions=0,
    )
    assert report.pass_rate == 0.0

  def test_summary(self):
    report = EvaluationReport(
        dataset="test",
        evaluator_name="latency_evaluator",
        total_sessions=5,
        passed_sessions=3,
        failed_sessions=2,
        aggregate_scores={"latency": 0.65},
    )
    text = report.summary()
    assert "latency_evaluator" in text
    assert "5" in text
    assert "60%" in text
    assert "0.650" in text


class TestParseJson:
  """Tests for JSON parsing helper."""

  def test_plain_json(self):
    result = _parse_json_from_text('{"score": 8, "justification": "good"}')
    assert result["score"] == 8

  def test_json_in_code_block(self):
    result = _parse_json_from_text('```json\n{"score": 7}\n```')
    assert result["score"] == 7

  def test_json_with_surrounding_text(self):
    result = _parse_json_from_text(
        'Here is my analysis: {"score": 9} That is all.'
    )
    assert result["score"] == 9

  def test_empty_input(self):
    assert _parse_json_from_text("") is None
    assert _parse_json_from_text(None) is None

  def test_no_json(self):
    assert _parse_json_from_text("no json here") is None


class TestDefaultEndpoint:
  """Tests for DEFAULT_ENDPOINT constant."""

  def test_default_endpoint_value(self):
    assert DEFAULT_ENDPOINT == "gemini-2.5-flash"


class TestAIGenerateJudgeBatchQuery:
  """Tests for the AI.GENERATE judge batch query template."""

  def test_contains_ai_generate(self):
    assert "AI.GENERATE" in AI_GENERATE_JUDGE_BATCH_QUERY

  def test_contains_output_schema(self):
    assert "output_schema" in AI_GENERATE_JUDGE_BATCH_QUERY

  def test_contains_endpoint_placeholder(self):
    assert "{endpoint}" in AI_GENERATE_JUDGE_BATCH_QUERY

  def test_contains_score_and_justification(self):
    assert "score INT64" in AI_GENERATE_JUDGE_BATCH_QUERY
    assert "justification STRING" in AI_GENERATE_JUDGE_BATCH_QUERY

  def test_does_not_contain_ml_generate_text(self):
    assert "ML.GENERATE_TEXT" not in AI_GENERATE_JUDGE_BATCH_QUERY

  def test_legacy_template_uses_ml_generate_text(self):
    assert "ML.GENERATE_TEXT" in LLM_JUDGE_BATCH_QUERY
    assert "ml_generate_text_result" in LLM_JUDGE_BATCH_QUERY


class TestSessionSummaryQuery:
  """Tests for SESSION_SUMMARY_QUERY token fields."""

  def test_contains_input_tokens(self):
    assert "input_tokens" in SESSION_SUMMARY_QUERY

  def test_contains_output_tokens(self):
    assert "output_tokens" in SESSION_SUMMARY_QUERY

  def test_contains_total_tokens(self):
    assert "total_tokens" in SESSION_SUMMARY_QUERY


class TestTokenEfficiencyPrebuilt:
  """Tests for CodeEvaluator.token_efficiency() preset."""

  def test_zero_tokens(self):
    evaluator = CodeEvaluator.token_efficiency(max_tokens=50000)
    score = evaluator.evaluate_session({
        "session_id": "s1",
        "total_tokens": 0,
    })
    assert score.scores["token_efficiency"] == 1.0
    assert score.passed is True

  def test_under_budget(self):
    evaluator = CodeEvaluator.token_efficiency(max_tokens=50000)
    score = evaluator.evaluate_session({
        "session_id": "s1",
        "total_tokens": 10000,
    })
    # 1.0 - 10000/50000 = 0.8
    assert score.scores["token_efficiency"] == pytest.approx(0.8)
    assert score.passed is True

  def test_over_budget(self):
    evaluator = CodeEvaluator.token_efficiency(max_tokens=50000)
    score = evaluator.evaluate_session({
        "session_id": "s1",
        "total_tokens": 60000,
    })
    assert score.scores["token_efficiency"] == 0.0
    assert score.passed is False

  def test_exactly_at_budget(self):
    evaluator = CodeEvaluator.token_efficiency(max_tokens=50000)
    score = evaluator.evaluate_session({
        "session_id": "s1",
        "total_tokens": 50000,
    })
    assert score.scores["token_efficiency"] == 0.0
    assert score.passed is False


class TestCostPerSessionPrebuilt:
  """Tests for CodeEvaluator.cost_per_session() preset."""

  def test_zero_tokens(self):
    evaluator = CodeEvaluator.cost_per_session(max_cost_usd=1.0)
    score = evaluator.evaluate_session({
        "session_id": "s1",
        "input_tokens": 0,
        "output_tokens": 0,
    })
    assert score.scores["cost"] == 1.0
    assert score.passed is True

  def test_under_budget(self):
    evaluator = CodeEvaluator.cost_per_session(
        max_cost_usd=1.0,
        input_cost_per_1k=0.001,
        output_cost_per_1k=0.002,
    )
    # Cost = (10000/1000)*0.001 + (5000/1000)*0.002
    #      = 10*0.001 + 5*0.002 = 0.01 + 0.01 = 0.02
    score = evaluator.evaluate_session({
        "session_id": "s1",
        "input_tokens": 10000,
        "output_tokens": 5000,
    })
    assert score.scores["cost"] == pytest.approx(0.98)
    assert score.passed is True

  def test_over_budget(self):
    evaluator = CodeEvaluator.cost_per_session(
        max_cost_usd=0.01,
        input_cost_per_1k=1.0,
        output_cost_per_1k=1.0,
    )
    # Cost = (1000/1000)*1.0 + (1000/1000)*1.0 = 2.0
    score = evaluator.evaluate_session({
        "session_id": "s1",
        "input_tokens": 1000,
        "output_tokens": 1000,
    })
    assert score.scores["cost"] == 0.0
    assert score.passed is False

  def test_missing_tokens_defaults_to_zero(self):
    evaluator = CodeEvaluator.cost_per_session(max_cost_usd=1.0)
    score = evaluator.evaluate_session({
        "session_id": "s1",
    })
    assert score.scores["cost"] == 1.0
