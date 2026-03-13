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

"""Tests for the bq-agent-sdk CLI."""

from datetime import datetime
from datetime import timezone
import json
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from bigquery_agent_analytics.cli import app
from bigquery_agent_analytics.evaluators import EvaluationReport
from bigquery_agent_analytics.evaluators import LLMAsJudge
from bigquery_agent_analytics.evaluators import SessionScore
from bigquery_agent_analytics.feedback import DriftReport
from bigquery_agent_analytics.feedback import QuestionDistribution
from bigquery_agent_analytics.insights import AggregatedInsights
from bigquery_agent_analytics.insights import InsightsReport
from bigquery_agent_analytics.insights import SessionMetadata
from bigquery_agent_analytics.trace import Span
from bigquery_agent_analytics.trace import Trace

runner = CliRunner()

_NOW = datetime(2026, 3, 12, 10, 0, 0, tzinfo=timezone.utc)


def _mock_trace():
  return Trace(
      trace_id="t1",
      session_id="s1",
      spans=[
          Span(
              event_type="LLM_REQUEST",
              agent="bot",
              timestamp=_NOW,
              content={},
              attributes={},
          )
      ],
      start_time=_NOW,
      end_time=_NOW,
      total_latency_ms=200.0,
  )


def _mock_report(passed, total):
  return EvaluationReport(
      dataset="test",
      evaluator_name="latency",
      total_sessions=total,
      passed_sessions=passed,
      failed_sessions=total - passed,
      created_at=_NOW,
      session_scores=[
          SessionScore(
              session_id=f"s{i}",
              scores={"latency": 0.9 if i < passed else 0.3},
              passed=i < passed,
          )
          for i in range(total)
      ],
  )


# ------------------------------------------------------------------ #
# doctor                                                               #
# ------------------------------------------------------------------ #


class TestDoctor:

  @patch("bigquery_agent_analytics.cli._build_client")
  def test_doctor_json(self, mock_build):
    client = MagicMock()
    client.doctor.return_value = {
        "status": "OK",
        "event_count": 100,
    }
    mock_build.return_value = client

    result = runner.invoke(
        app,
        [
            "doctor",
            "--project-id=proj",
            "--dataset-id=ds",
        ],
    )
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert parsed["status"] == "OK"
    assert parsed["event_count"] == 100

  @patch("bigquery_agent_analytics.cli._build_client")
  def test_doctor_error_exit_2(self, mock_build):
    mock_build.side_effect = RuntimeError("connection failed")
    result = runner.invoke(
        app,
        [
            "doctor",
            "--project-id=proj",
            "--dataset-id=ds",
        ],
    )
    assert result.exit_code == 2


# ------------------------------------------------------------------ #
# get-trace                                                            #
# ------------------------------------------------------------------ #


class TestGetTrace:

  @patch("bigquery_agent_analytics.cli._build_client")
  def test_get_trace_by_session_id(self, mock_build):
    client = MagicMock()
    client.get_session_trace.return_value = _mock_trace()
    mock_build.return_value = client

    result = runner.invoke(
        app,
        [
            "get-trace",
            "--project-id=proj",
            "--dataset-id=ds",
            "--session-id=s1",
        ],
    )
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert parsed["trace_id"] == "t1"
    assert parsed["session_id"] == "s1"
    client.get_session_trace.assert_called_once_with("s1")

  @patch("bigquery_agent_analytics.cli._build_client")
  def test_get_trace_by_trace_id(self, mock_build):
    client = MagicMock()
    client.get_trace.return_value = _mock_trace()
    mock_build.return_value = client

    result = runner.invoke(
        app,
        [
            "get-trace",
            "--project-id=proj",
            "--dataset-id=ds",
            "--trace-id=t1",
        ],
    )
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert parsed["trace_id"] == "t1"
    client.get_trace.assert_called_once_with("t1")

  def test_get_trace_missing_id_exit_2(self):
    result = runner.invoke(
        app,
        [
            "get-trace",
            "--project-id=proj",
            "--dataset-id=ds",
        ],
    )
    assert result.exit_code == 2

  @patch("bigquery_agent_analytics.cli._build_client")
  def test_get_trace_text_format(self, mock_build):
    client = MagicMock()
    client.get_session_trace.return_value = _mock_trace()
    mock_build.return_value = client

    result = runner.invoke(
        app,
        [
            "get-trace",
            "--project-id=proj",
            "--dataset-id=ds",
            "--session-id=s1",
            "--format=text",
        ],
    )
    assert result.exit_code == 0
    # Text format for Trace uses render() which includes trace_id
    assert "t1" in result.output


# ------------------------------------------------------------------ #
# evaluate                                                             #
# ------------------------------------------------------------------ #


class TestEvaluate:

  @patch("bigquery_agent_analytics.cli._build_client")
  def test_evaluate_latency_pass(self, mock_build):
    client = MagicMock()
    client.evaluate.return_value = _mock_report(10, 10)
    mock_build.return_value = client

    result = runner.invoke(
        app,
        [
            "evaluate",
            "--project-id=proj",
            "--dataset-id=ds",
            "--evaluator=latency",
            "--threshold=5000",
        ],
    )
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert parsed["total_sessions"] == 10
    assert parsed["passed_sessions"] == 10

  @patch("bigquery_agent_analytics.cli._build_client")
  def test_evaluate_exit_code_on_failure(self, mock_build):
    client = MagicMock()
    client.evaluate.return_value = _mock_report(7, 10)
    mock_build.return_value = client

    result = runner.invoke(
        app,
        [
            "evaluate",
            "--project-id=proj",
            "--dataset-id=ds",
            "--evaluator=latency",
            "--exit-code",
        ],
    )
    assert result.exit_code == 1

  @patch("bigquery_agent_analytics.cli._build_client")
  def test_evaluate_exit_code_on_pass(self, mock_build):
    client = MagicMock()
    client.evaluate.return_value = _mock_report(10, 10)
    mock_build.return_value = client

    result = runner.invoke(
        app,
        [
            "evaluate",
            "--project-id=proj",
            "--dataset-id=ds",
            "--evaluator=latency",
            "--exit-code",
        ],
    )
    assert result.exit_code == 0

  @patch("bigquery_agent_analytics.cli._build_client")
  def test_evaluate_with_filter_args(self, mock_build):
    client = MagicMock()
    client.evaluate.return_value = _mock_report(5, 5)
    mock_build.return_value = client

    result = runner.invoke(
        app,
        [
            "evaluate",
            "--project-id=proj",
            "--dataset-id=ds",
            "--evaluator=error_rate",
            "--threshold=0.1",
            "--agent-id=bot",
            "--last=1h",
            "--limit=50",
        ],
    )
    assert result.exit_code == 0
    # Verify the filter was passed
    call_kwargs = client.evaluate.call_args
    filters = call_kwargs.kwargs.get("filters") or call_kwargs[1].get("filters")
    assert filters.agent_id == "bot"
    assert filters.limit == 50
    assert filters.start_time is not None

  @patch("bigquery_agent_analytics.cli._build_client")
  def test_evaluate_llm_judge(self, mock_build):
    client = MagicMock()
    client.evaluate.return_value = _mock_report(8, 10)
    mock_build.return_value = client

    result = runner.invoke(
        app,
        [
            "evaluate",
            "--project-id=proj",
            "--dataset-id=ds",
            "--evaluator=llm-judge",
            "--criterion=correctness",
            "--threshold=0.7",
        ],
    )
    assert result.exit_code == 0
    call_args = client.evaluate.call_args
    ev = call_args.kwargs.get("evaluator") or call_args[1].get("evaluator")
    assert isinstance(ev, LLMAsJudge)

  @patch("bigquery_agent_analytics.cli._build_client")
  def test_evaluate_text_format(self, mock_build):
    client = MagicMock()
    client.evaluate.return_value = _mock_report(10, 10)
    mock_build.return_value = client

    result = runner.invoke(
        app,
        [
            "evaluate",
            "--project-id=proj",
            "--dataset-id=ds",
            "--format=text",
        ],
    )
    assert result.exit_code == 0
    # Text format uses .summary() which mentions evaluator name
    assert "latency" in result.output

  def test_evaluate_unknown_evaluator(self):
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--project-id=proj",
            "--dataset-id=ds",
            "--evaluator=bogus",
        ],
    )
    assert result.exit_code == 2

  def test_evaluate_unknown_criterion(self):
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--project-id=proj",
            "--dataset-id=ds",
            "--evaluator=llm-judge",
            "--criterion=bogus",
        ],
    )
    assert result.exit_code == 2

  @patch("bigquery_agent_analytics.cli._build_client")
  def test_evaluate_strict_flag(self, mock_build):
    client = MagicMock()
    client.evaluate.return_value = _mock_report(10, 10)
    mock_build.return_value = client

    result = runner.invoke(
        app,
        [
            "evaluate",
            "--project-id=proj",
            "--dataset-id=ds",
            "--strict",
        ],
    )
    assert result.exit_code == 0
    call_kwargs = client.evaluate.call_args
    assert call_kwargs.kwargs.get("strict") is True

  @patch("bigquery_agent_analytics.cli._build_client")
  def test_evaluate_infra_error_exit_2(self, mock_build):
    client = MagicMock()
    client.evaluate.side_effect = RuntimeError("BQ timeout")
    mock_build.return_value = client

    result = runner.invoke(
        app,
        [
            "evaluate",
            "--project-id=proj",
            "--dataset-id=ds",
        ],
    )
    assert result.exit_code == 2


# ------------------------------------------------------------------ #
# env var fallback                                                     #
# ------------------------------------------------------------------ #


class TestEnvVars:

  @patch("bigquery_agent_analytics.cli._build_client")
  def test_env_var_project_and_dataset(self, mock_build):
    client = MagicMock()
    client.doctor.return_value = {"status": "OK"}
    mock_build.return_value = client

    result = runner.invoke(
        app,
        ["doctor"],
        env={
            "BQ_AGENT_PROJECT": "env-proj",
            "BQ_AGENT_DATASET": "env-ds",
        },
    )
    assert result.exit_code == 0
    mock_build.assert_called_once()
    call_args = mock_build.call_args
    assert call_args[1].get("project_id") or call_args[0][0] in ("env-proj",)


# ------------------------------------------------------------------ #
# all evaluator types                                                  #
# ------------------------------------------------------------------ #


class TestAllEvaluators:

  @pytest.mark.parametrize(
      "name",
      [
          "latency",
          "error_rate",
          "turn_count",
          "token_efficiency",
          "ttft",
          "cost",
      ],
  )
  @patch("bigquery_agent_analytics.cli._build_client")
  def test_code_evaluator(self, mock_build, name):
    client = MagicMock()
    client.evaluate.return_value = _mock_report(10, 10)
    mock_build.return_value = client

    result = runner.invoke(
        app,
        [
            "evaluate",
            "--project-id=proj",
            "--dataset-id=ds",
            f"--evaluator={name}",
            "--threshold=100",
        ],
    )
    assert result.exit_code == 0

  @pytest.mark.parametrize(
      "criterion",
      ["correctness", "hallucination", "sentiment"],
  )
  @patch("bigquery_agent_analytics.cli._build_client")
  def test_llm_judge_criteria(self, mock_build, criterion):
    client = MagicMock()
    client.evaluate.return_value = _mock_report(10, 10)
    mock_build.return_value = client

    result = runner.invoke(
        app,
        [
            "evaluate",
            "--project-id=proj",
            "--dataset-id=ds",
            "--evaluator=llm-judge",
            f"--criterion={criterion}",
            "--threshold=0.5",
        ],
    )
    assert result.exit_code == 0


# ------------------------------------------------------------------ #
# default thresholds                                                   #
# ------------------------------------------------------------------ #


class TestDefaultThresholds:

  @pytest.mark.parametrize(
      "name",
      [
          "latency",
          "error_rate",
          "turn_count",
          "token_efficiency",
          "ttft",
          "cost",
      ],
  )
  @patch("bigquery_agent_analytics.cli._build_client")
  def test_code_evaluator_uses_sdk_default(self, mock_build, name):
    """Omitting --threshold should use the SDK's built-in default."""
    client = MagicMock()
    client.evaluate.return_value = _mock_report(10, 10)
    mock_build.return_value = client

    result = runner.invoke(
        app,
        [
            "evaluate",
            "--project-id=proj",
            "--dataset-id=ds",
            f"--evaluator={name}",
        ],
    )
    assert result.exit_code == 0

  @pytest.mark.parametrize(
      "criterion",
      ["correctness", "hallucination", "sentiment"],
  )
  @patch("bigquery_agent_analytics.cli._build_client")
  def test_llm_judge_uses_sdk_default(self, mock_build, criterion):
    """Omitting --threshold for llm-judge should use 0.5, not 5000."""
    client = MagicMock()
    client.evaluate.return_value = _mock_report(10, 10)
    mock_build.return_value = client

    result = runner.invoke(
        app,
        [
            "evaluate",
            "--project-id=proj",
            "--dataset-id=ds",
            "--evaluator=llm-judge",
            f"--criterion={criterion}",
        ],
    )
    assert result.exit_code == 0


# ------------------------------------------------------------------ #
# Helpers for v1.1 commands                                            #
# ------------------------------------------------------------------ #


def _mock_insights():
  meta = SessionMetadata(
      session_id="s1",
      event_count=10,
      tool_calls=3,
      tool_errors=0,
      llm_calls=5,
      turn_count=2,
      total_latency_ms=1200.0,
      avg_latency_ms=600.0,
      agents_used=["bot"],
      tools_used=["search"],
      has_error=False,
      hitl_events=0,
      state_changes=0,
      start_time=_NOW,
      end_time=_NOW,
  )
  return InsightsReport(
      created_at=_NOW,
      session_metadata=[meta],
      aggregated=AggregatedInsights(
          total_sessions=1,
          success_rate=1.0,
          avg_effectiveness=0.9,
          avg_latency_ms=1200.0,
          avg_turns=2.0,
          error_rate=0.0,
      ),
      executive_summary="All good.",
  )


def _mock_drift():
  return DriftReport(
      coverage_percentage=0.85,
      total_golden=100,
      total_production=200,
  )


def _mock_distribution():
  return QuestionDistribution(total_questions=50)


# ------------------------------------------------------------------ #
# insights                                                             #
# ------------------------------------------------------------------ #


class TestInsights:

  @patch("bigquery_agent_analytics.cli._build_client")
  def test_insights_json(self, mock_build):
    client = MagicMock()
    client.insights.return_value = _mock_insights()
    mock_build.return_value = client

    result = runner.invoke(
        app,
        [
            "insights",
            "--project-id=proj",
            "--dataset-id=ds",
        ],
    )
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert "aggregated" in parsed

  @patch("bigquery_agent_analytics.cli._build_client")
  def test_insights_with_filters(self, mock_build):
    client = MagicMock()
    client.insights.return_value = _mock_insights()
    mock_build.return_value = client

    result = runner.invoke(
        app,
        [
            "insights",
            "--project-id=proj",
            "--dataset-id=ds",
            "--agent-id=bot",
            "--last=1h",
            "--max-sessions=25",
        ],
    )
    assert result.exit_code == 0
    call_kwargs = client.insights.call_args[1]
    assert call_kwargs["filters"].agent_id == "bot"
    assert call_kwargs["config"].max_sessions == 25

  @patch("bigquery_agent_analytics.cli._build_client")
  def test_insights_error_exit_2(self, mock_build):
    client = MagicMock()
    client.insights.side_effect = RuntimeError("fail")
    mock_build.return_value = client

    result = runner.invoke(
        app,
        [
            "insights",
            "--project-id=proj",
            "--dataset-id=ds",
        ],
    )
    assert result.exit_code == 2


# ------------------------------------------------------------------ #
# drift                                                                #
# ------------------------------------------------------------------ #


class TestDrift:

  @patch("bigquery_agent_analytics.cli._build_client")
  def test_drift_json(self, mock_build):
    client = MagicMock()
    client.drift_detection.return_value = _mock_drift()
    mock_build.return_value = client

    result = runner.invoke(
        app,
        [
            "drift",
            "--project-id=proj",
            "--dataset-id=ds",
            "--golden-dataset=golden",
        ],
    )
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert parsed["coverage_percentage"] == 0.85

  @patch("bigquery_agent_analytics.cli._build_client")
  def test_drift_with_filters(self, mock_build):
    client = MagicMock()
    client.drift_detection.return_value = _mock_drift()
    mock_build.return_value = client

    result = runner.invoke(
        app,
        [
            "drift",
            "--project-id=proj",
            "--dataset-id=ds",
            "--golden-dataset=golden",
            "--agent-id=bot",
            "--last=24h",
        ],
    )
    assert result.exit_code == 0
    call_kwargs = client.drift_detection.call_args[1]
    assert call_kwargs["golden_dataset"] == "golden"
    assert call_kwargs["filters"].agent_id == "bot"

  def test_drift_missing_golden_dataset_exit_2(self):
    result = runner.invoke(
        app,
        [
            "drift",
            "--project-id=proj",
            "--dataset-id=ds",
        ],
    )
    assert result.exit_code == 2

  @patch("bigquery_agent_analytics.cli._build_client")
  def test_drift_error_exit_2(self, mock_build):
    client = MagicMock()
    client.drift_detection.side_effect = RuntimeError("fail")
    mock_build.return_value = client

    result = runner.invoke(
        app,
        [
            "drift",
            "--project-id=proj",
            "--dataset-id=ds",
            "--golden-dataset=golden",
        ],
    )
    assert result.exit_code == 2


# ------------------------------------------------------------------ #
# distribution                                                         #
# ------------------------------------------------------------------ #


class TestDistribution:

  @patch("bigquery_agent_analytics.cli._build_client")
  def test_distribution_json(self, mock_build):
    client = MagicMock()
    client.deep_analysis.return_value = _mock_distribution()
    mock_build.return_value = client

    result = runner.invoke(
        app,
        [
            "distribution",
            "--project-id=proj",
            "--dataset-id=ds",
        ],
    )
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert parsed["total_questions"] == 50

  @patch("bigquery_agent_analytics.cli._build_client")
  def test_distribution_with_options(self, mock_build):
    client = MagicMock()
    client.deep_analysis.return_value = _mock_distribution()
    mock_build.return_value = client

    result = runner.invoke(
        app,
        [
            "distribution",
            "--project-id=proj",
            "--dataset-id=ds",
            "--mode=frequently_asked",
            "--top-k=10",
        ],
    )
    assert result.exit_code == 0
    call_kwargs = client.deep_analysis.call_args[1]
    assert call_kwargs["configuration"].mode == "frequently_asked"
    assert call_kwargs["configuration"].top_k == 10

  @patch("bigquery_agent_analytics.cli._build_client")
  def test_distribution_error_exit_2(self, mock_build):
    client = MagicMock()
    client.deep_analysis.side_effect = RuntimeError("fail")
    mock_build.return_value = client

    result = runner.invoke(
        app,
        [
            "distribution",
            "--project-id=proj",
            "--dataset-id=ds",
        ],
    )
    assert result.exit_code == 2


# ------------------------------------------------------------------ #
# hitl-metrics                                                         #
# ------------------------------------------------------------------ #


class TestHitlMetrics:

  @patch("bigquery_agent_analytics.cli._build_client")
  def test_hitl_metrics_json(self, mock_build):
    client = MagicMock()
    client.hitl_metrics.return_value = {
        "total_hitl_events": 15,
        "sessions_with_hitl": 3,
    }
    mock_build.return_value = client

    result = runner.invoke(
        app,
        [
            "hitl-metrics",
            "--project-id=proj",
            "--dataset-id=ds",
        ],
    )
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert parsed["total_hitl_events"] == 15

  @patch("bigquery_agent_analytics.cli._build_client")
  def test_hitl_metrics_with_filters(self, mock_build):
    client = MagicMock()
    client.hitl_metrics.return_value = {"total_hitl_events": 5}
    mock_build.return_value = client

    result = runner.invoke(
        app,
        [
            "hitl-metrics",
            "--project-id=proj",
            "--dataset-id=ds",
            "--agent-id=bot",
            "--last=7d",
        ],
    )
    assert result.exit_code == 0
    call_kwargs = client.hitl_metrics.call_args[1]
    assert call_kwargs["filters"].agent_id == "bot"

  @patch("bigquery_agent_analytics.cli._build_client")
  def test_hitl_metrics_error_exit_2(self, mock_build):
    client = MagicMock()
    client.hitl_metrics.side_effect = RuntimeError("fail")
    mock_build.return_value = client

    result = runner.invoke(
        app,
        [
            "hitl-metrics",
            "--project-id=proj",
            "--dataset-id=ds",
        ],
    )
    assert result.exit_code == 2


# ------------------------------------------------------------------ #
# list-traces                                                          #
# ------------------------------------------------------------------ #


class TestListTraces:

  @patch("bigquery_agent_analytics.cli._build_client")
  def test_list_traces_json(self, mock_build):
    client = MagicMock()
    client.list_traces.return_value = [_mock_trace(), _mock_trace()]
    mock_build.return_value = client

    result = runner.invoke(
        app,
        [
            "list-traces",
            "--project-id=proj",
            "--dataset-id=ds",
        ],
    )
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert isinstance(parsed, list)
    assert len(parsed) == 2

  @patch("bigquery_agent_analytics.cli._build_client")
  def test_list_traces_with_filters(self, mock_build):
    client = MagicMock()
    client.list_traces.return_value = [_mock_trace()]
    mock_build.return_value = client

    result = runner.invoke(
        app,
        [
            "list-traces",
            "--project-id=proj",
            "--dataset-id=ds",
            "--session-id=s1",
            "--agent-id=bot",
            "--last=1h",
            "--limit=10",
        ],
    )
    assert result.exit_code == 0
    call_kwargs = client.list_traces.call_args[1]
    filters = call_kwargs["filter_criteria"]
    assert filters.agent_id == "bot"
    assert filters.limit == 10
    assert filters.session_ids == ["s1"]

  @patch("bigquery_agent_analytics.cli._build_client")
  def test_list_traces_error_exit_2(self, mock_build):
    client = MagicMock()
    client.list_traces.side_effect = RuntimeError("fail")
    mock_build.return_value = client

    result = runner.invoke(
        app,
        [
            "list-traces",
            "--project-id=proj",
            "--dataset-id=ds",
        ],
    )
    assert result.exit_code == 2


# ------------------------------------------------------------------ #
# views create-all / views create                                      #
# ------------------------------------------------------------------ #


class TestViews:

  @patch("bigquery_agent_analytics.cli._build_view_manager")
  def test_views_create_all(self, mock_build_vm):
    vm = MagicMock()
    vm.create_all_views.return_value = {
        "LLM_REQUEST": "adk_llm_request",
        "TOOL_CALL": "adk_tool_call",
    }
    mock_build_vm.return_value = vm

    result = runner.invoke(
        app,
        [
            "views",
            "create-all",
            "--project-id=proj",
            "--dataset-id=ds",
        ],
    )
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert parsed["LLM_REQUEST"] == "adk_llm_request"

  @patch("bigquery_agent_analytics.cli._build_view_manager")
  def test_views_create_all_with_prefix(self, mock_build_vm):
    vm = MagicMock()
    vm.create_all_views.return_value = {}
    mock_build_vm.return_value = vm

    result = runner.invoke(
        app,
        [
            "views",
            "create-all",
            "--project-id=proj",
            "--dataset-id=ds",
            "--prefix=custom_",
        ],
    )
    assert result.exit_code == 0
    mock_build_vm.assert_called_once()
    call_args = mock_build_vm.call_args
    assert (
        call_args[0][3] == "custom_" or call_args[1].get("prefix") == "custom_"
    )

  @patch("bigquery_agent_analytics.cli._build_view_manager")
  def test_views_create_single(self, mock_build_vm):
    vm = MagicMock()
    mock_build_vm.return_value = vm

    result = runner.invoke(
        app,
        [
            "views",
            "create",
            "LLM_REQUEST",
            "--project-id=proj",
            "--dataset-id=ds",
        ],
    )
    assert result.exit_code == 0
    vm.create_view.assert_called_once_with("LLM_REQUEST")
    assert "LLM_REQUEST" in result.output

  @patch("bigquery_agent_analytics.cli._build_view_manager")
  def test_views_create_all_error_exit_2(self, mock_build_vm):
    vm = MagicMock()
    vm.create_all_views.side_effect = RuntimeError("fail")
    mock_build_vm.return_value = vm

    result = runner.invoke(
        app,
        [
            "views",
            "create-all",
            "--project-id=proj",
            "--dataset-id=ds",
        ],
    )
    assert result.exit_code == 2

  @patch("bigquery_agent_analytics.cli._build_view_manager")
  def test_views_create_error_exit_2(self, mock_build_vm):
    vm = MagicMock()
    vm.create_view.side_effect = RuntimeError("fail")
    mock_build_vm.return_value = vm

    result = runner.invoke(
        app,
        [
            "views",
            "create",
            "BOGUS",
            "--project-id=proj",
            "--dataset-id=ds",
        ],
    )
    assert result.exit_code == 2
