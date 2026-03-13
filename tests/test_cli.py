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
