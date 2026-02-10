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

"""Tests for the SDK Client class."""

from datetime import datetime
from datetime import timezone
from unittest.mock import MagicMock
from unittest.mock import patch

from bigquery_agent_analytics.client import Client
from bigquery_agent_analytics.evaluators import CodeEvaluator
from bigquery_agent_analytics.evaluators import EvaluationReport
from bigquery_agent_analytics.trace import TraceFilter
import pytest


def _mock_bq_client():
  """Creates a mock BigQuery client."""
  return MagicMock()


def _make_event_rows(n=3, session_id="sess-1"):
  """Creates mock BigQuery result rows for events."""
  ts = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
  events = [
      {
          "event_type": "USER_MESSAGE_RECEIVED",
          "agent": None,
          "timestamp": ts,
          "session_id": session_id,
          "invocation_id": "inv-1",
          "user_id": "user-1",
          "trace_id": "trace-1",
          "span_id": "s1",
          "parent_span_id": None,
          "content": '{"text_summary": "Hello"}',
          "content_parts": [],
          "attributes": "{}",
          "latency_ms": None,
          "status": "OK",
          "error_message": None,
          "is_truncated": False,
      },
      {
          "event_type": "TOOL_STARTING",
          "agent": "my_agent",
          "timestamp": ts,
          "session_id": session_id,
          "invocation_id": "inv-1",
          "user_id": "user-1",
          "trace_id": "trace-1",
          "span_id": "s2",
          "parent_span_id": "s1",
          "content": '{"tool": "search", "args": {"q": "hi"}}',
          "content_parts": [],
          "attributes": "{}",
          "latency_ms": None,
          "status": "OK",
          "error_message": None,
          "is_truncated": False,
      },
      {
          "event_type": "AGENT_COMPLETED",
          "agent": "my_agent",
          "timestamp": ts,
          "session_id": session_id,
          "invocation_id": "inv-1",
          "user_id": "user-1",
          "trace_id": "trace-1",
          "span_id": "s3",
          "parent_span_id": "s1",
          "content": '{"response": "Hello!"}',
          "content_parts": [],
          "attributes": "{}",
          "latency_ms": '{"total_ms": 250}',
          "status": "OK",
          "error_message": None,
          "is_truncated": False,
      },
  ]
  return events[:n]


def _make_mock_row(data):
  """Creates a mock BQ row that supports dict(row) and row.get()."""
  mock = MagicMock()
  mock.__iter__ = MagicMock(return_value=iter(data.items()))
  mock.get = data.get
  mock.keys = data.keys
  mock.values = data.values
  mock.items = data.items
  # Support dict() conversion
  mock.__getitem__ = lambda self, k: data[k]
  return mock


class TestClientInit:
  """Tests for Client initialization."""

  def test_init_skip_verify(self):
    mock_bq = _mock_bq_client()
    client = Client(
        project_id="proj",
        dataset_id="ds",
        table_id="events",
        verify_schema=False,
        bq_client=mock_bq,
    )
    assert client.project_id == "proj"
    assert client.dataset_id == "ds"
    assert client.table_id == "events"
    mock_bq.query.assert_not_called()

  def test_init_with_verify(self):
    mock_bq = _mock_bq_client()
    # Mock schema query result
    mock_rows = [
        _make_mock_row({
            "column_name": "timestamp",
            "data_type": "TIMESTAMP",
        }),
        _make_mock_row({
            "column_name": "event_type",
            "data_type": "STRING",
        }),
        _make_mock_row({
            "column_name": "session_id",
            "data_type": "STRING",
        }),
        _make_mock_row({
            "column_name": "content",
            "data_type": "JSON",
        }),
    ]
    mock_job = MagicMock()
    mock_job.result.return_value = mock_rows
    mock_bq.query.return_value = mock_job

    client = Client(
        project_id="proj",
        dataset_id="ds",
        verify_schema=True,
        bq_client=mock_bq,
    )
    mock_bq.query.assert_called_once()


class TestClientGetTrace:
  """Tests for Client.get_trace()."""

  def test_get_trace_success(self):
    mock_bq = _mock_bq_client()
    rows = [_make_mock_row(r) for r in _make_event_rows()]
    mock_job = MagicMock()
    mock_job.result.return_value = rows
    mock_bq.query.return_value = mock_job

    client = Client(
        project_id="proj",
        dataset_id="ds",
        verify_schema=False,
        bq_client=mock_bq,
    )
    trace = client.get_trace("sess-1")

    assert trace.session_id == "sess-1"
    assert trace.trace_id == "trace-1"
    assert len(trace.spans) == 3
    assert trace.user_id == "user-1"

  def test_get_trace_not_found(self):
    mock_bq = _mock_bq_client()
    mock_job = MagicMock()
    mock_job.result.return_value = []
    mock_bq.query.return_value = mock_job

    client = Client(
        project_id="proj",
        dataset_id="ds",
        verify_schema=False,
        bq_client=mock_bq,
    )

    with pytest.raises(ValueError, match="No events found"):
      client.get_trace("nonexistent")


class TestClientListTraces:
  """Tests for Client.list_traces()."""

  def test_list_traces_groups_by_session(self):
    mock_bq = _mock_bq_client()
    rows_s1 = [_make_mock_row(r) for r in _make_event_rows(2, "s1")]
    rows_s2 = [_make_mock_row(r) for r in _make_event_rows(2, "s2")]
    mock_job = MagicMock()
    mock_job.result.return_value = rows_s1 + rows_s2
    mock_bq.query.return_value = mock_job

    client = Client(
        project_id="proj",
        dataset_id="ds",
        verify_schema=False,
        bq_client=mock_bq,
    )
    traces = client.list_traces()

    assert len(traces) == 2
    session_ids = {t.session_id for t in traces}
    assert "s1" in session_ids
    assert "s2" in session_ids

  def test_list_traces_with_filter(self):
    mock_bq = _mock_bq_client()
    mock_job = MagicMock()
    mock_job.result.return_value = []
    mock_bq.query.return_value = mock_job

    client = Client(
        project_id="proj",
        dataset_id="ds",
        verify_schema=False,
        bq_client=mock_bq,
    )
    filt = TraceFilter(agent_id="my_agent", has_error=True)
    traces = client.list_traces(filter_criteria=filt)

    assert traces == []
    # Verify query was called
    mock_bq.query.assert_called_once()


class TestClientEvaluate:
  """Tests for Client.evaluate()."""

  def test_evaluate_code_evaluator(self):
    mock_bq = _mock_bq_client()
    # Mock session summary results
    summary_rows = [
        _make_mock_row({
            "session_id": "s1",
            "total_events": 10,
            "tool_calls": 3,
            "tool_errors": 0,
            "llm_calls": 2,
            "avg_latency_ms": 1500.0,
            "max_latency_ms": 3000.0,
            "total_latency_ms": 5000.0,
            "turn_count": 2,
            "has_error": False,
        }),
        _make_mock_row({
            "session_id": "s2",
            "total_events": 20,
            "tool_calls": 5,
            "tool_errors": 3,
            "llm_calls": 4,
            "avg_latency_ms": 8000.0,
            "max_latency_ms": 15000.0,
            "total_latency_ms": 40000.0,
            "turn_count": 6,
            "has_error": True,
        }),
    ]
    mock_job = MagicMock()
    mock_job.result.return_value = summary_rows
    mock_bq.query.return_value = mock_job

    client = Client(
        project_id="proj",
        dataset_id="ds",
        verify_schema=False,
        bq_client=mock_bq,
    )

    evaluator = CodeEvaluator.latency(threshold_ms=5000)
    report = client.evaluate(evaluator=evaluator)

    assert isinstance(report, EvaluationReport)
    assert report.total_sessions == 2
    assert report.evaluator_name == "latency_evaluator"
    # s1 should pass (1500ms < 5000ms), s2 should fail
    assert report.passed_sessions >= 1

  def test_evaluate_invalid_evaluator(self):
    mock_bq = _mock_bq_client()
    client = Client(
        project_id="proj",
        dataset_id="ds",
        verify_schema=False,
        bq_client=mock_bq,
    )

    with pytest.raises(TypeError, match="Unsupported"):
      client.evaluate(evaluator="not_an_evaluator")


class TestClientEndpointInit:
  """Tests for Client endpoint and connection_id params."""

  def test_default_endpoint(self):
    mock_bq = _mock_bq_client()
    client = Client(
        project_id="proj",
        dataset_id="ds",
        verify_schema=False,
        bq_client=mock_bq,
    )
    assert client.endpoint == "gemini-2.5-flash"
    assert client.connection_id is None

  def test_custom_endpoint(self):
    mock_bq = _mock_bq_client()
    client = Client(
        project_id="proj",
        dataset_id="ds",
        verify_schema=False,
        bq_client=mock_bq,
        endpoint="gemini-2.5-pro",
    )
    assert client.endpoint == "gemini-2.5-pro"

  def test_custom_connection_id(self):
    mock_bq = _mock_bq_client()
    client = Client(
        project_id="proj",
        dataset_id="ds",
        verify_schema=False,
        bq_client=mock_bq,
        connection_id="us.my-connection",
    )
    assert client.connection_id == "us.my-connection"

  def test_legacy_model_ref(self):
    mock_bq = _mock_bq_client()
    client = Client(
        project_id="proj",
        dataset_id="ds",
        verify_schema=False,
        bq_client=mock_bq,
        endpoint="proj.ds.my_model",
    )
    assert client.endpoint == "proj.ds.my_model"
    assert client._is_legacy_model_ref(client.endpoint)


class TestIsLegacyModelRef:
  """Tests for Client._is_legacy_model_ref()."""

  def test_endpoint_name(self):
    assert not Client._is_legacy_model_ref("gemini-2.5-flash")

  def test_endpoint_with_one_dot(self):
    assert not Client._is_legacy_model_ref("gemini-2.5-pro")

  def test_legacy_model_two_dots(self):
    assert Client._is_legacy_model_ref("p.d.model_name")

  def test_legacy_model_three_dots(self):
    assert Client._is_legacy_model_ref("a.b.c.d")


class TestAIGenerateJudge:
  """Tests for Client._ai_generate_judge()."""

  def test_ai_generate_judge_typed_columns(self):
    mock_bq = _mock_bq_client()
    mock_rows = [
        _make_mock_row({
            "session_id": "s1",
            "trace_text": "USER: hi",
            "final_response": "hello",
            "score": 8,
            "justification": "Good response",
        }),
        _make_mock_row({
            "session_id": "s2",
            "trace_text": "USER: bye",
            "final_response": "goodbye",
            "score": 3,
            "justification": "Incomplete",
        }),
    ]
    mock_job = MagicMock()
    mock_job.result.return_value = mock_rows
    mock_bq.query.return_value = mock_job

    client = Client(
        project_id="proj",
        dataset_id="ds",
        verify_schema=False,
        bq_client=mock_bq,
    )

    from bigquery_agent_analytics.evaluators import _JudgeCriterion
    from bigquery_agent_analytics.evaluators import LLMAsJudge

    evaluator = LLMAsJudge.correctness(threshold=0.5)
    criterion = evaluator._criteria[0]

    report = client._ai_generate_judge(
        evaluator,
        criterion,
        "agent_events_v2",
        "TRUE",
        [],
    )
    assert report.total_sessions == 2
    assert report.session_scores[0].scores["correctness"] == 0.8
    assert report.session_scores[1].scores["correctness"] == 0.3
    assert report.session_scores[0].passed is True
    assert report.session_scores[1].passed is False

  def test_fallback_chain_tries_ai_generate_first(self):
    """Verify _evaluate_llm_judge tries AI.GENERATE first."""
    mock_bq = _mock_bq_client()
    mock_rows = [
        _make_mock_row({
            "session_id": "s1",
            "trace_text": "USER: hi",
            "final_response": "hello",
            "score": 7,
            "justification": "OK",
        }),
    ]
    mock_job = MagicMock()
    mock_job.result.return_value = mock_rows
    mock_bq.query.return_value = mock_job

    client = Client(
        project_id="proj",
        dataset_id="ds",
        verify_schema=False,
        bq_client=mock_bq,
    )
    from bigquery_agent_analytics.evaluators import LLMAsJudge

    evaluator = LLMAsJudge.correctness()
    report = client._evaluate_llm_judge(
        evaluator,
        "agent_events_v2",
        "TRUE",
        [],
    )
    # Should have gotten a result from AI.GENERATE path
    assert report.total_sessions == 1
    # Verify AI.GENERATE query was used (contains endpoint)
    call_args = mock_bq.query.call_args
    query_str = call_args[0][0]
    assert "AI.GENERATE" in query_str
