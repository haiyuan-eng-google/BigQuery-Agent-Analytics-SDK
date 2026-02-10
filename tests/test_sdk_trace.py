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

"""Tests for the SDK trace module."""

from datetime import datetime
from datetime import timezone

from bigquery_agent_analytics.trace import ContentPart
from bigquery_agent_analytics.trace import Span
from bigquery_agent_analytics.trace import Trace
from bigquery_agent_analytics.trace import TraceFilter
import pytest


class TestSpan:
  """Tests for Span class."""

  def test_from_bigquery_row_basic(self):
    row = {
        "event_type": "TOOL_STARTING",
        "agent": "my_agent",
        "timestamp": datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        "content": '{"tool": "search", "args": {"q": "test"}}',
        "attributes": '{"model": "gemini"}',
        "span_id": "span-1",
        "parent_span_id": "parent-1",
        "status": "OK",
        "session_id": "sess-1",
    }

    span = Span.from_bigquery_row(row)

    assert span.event_type == "TOOL_STARTING"
    assert span.agent == "my_agent"
    assert span.content["tool"] == "search"
    assert span.attributes["model"] == "gemini"
    assert span.span_id == "span-1"
    assert span.parent_span_id == "parent-1"
    assert span.status == "OK"
    assert span.session_id == "sess-1"

  def test_from_bigquery_row_json_latency(self):
    row = {
        "event_type": "LLM_RESPONSE",
        "agent": "agent",
        "timestamp": datetime.now(timezone.utc),
        "content": None,
        "attributes": None,
        "latency_ms": '{"total_ms": 450}',
        "status": "OK",
    }
    span = Span.from_bigquery_row(row)
    assert span.latency_ms == 450

  def test_from_bigquery_row_dict_latency(self):
    row = {
        "event_type": "LLM_RESPONSE",
        "agent": "agent",
        "timestamp": datetime.now(timezone.utc),
        "content": None,
        "attributes": None,
        "latency_ms": {"total_ms": 200},
        "status": "OK",
    }
    span = Span.from_bigquery_row(row)
    assert span.latency_ms == 200

  def test_from_bigquery_row_with_content_parts(self):
    row = {
        "event_type": "LLM_RESPONSE",
        "agent": "agent",
        "timestamp": datetime.now(timezone.utc),
        "content": "{}",
        "attributes": "{}",
        "content_parts": [{
            "mime_type": "image/png",
            "uri": "gs://bucket/img.png",
            "text": None,
            "storage_mode": "GCS_REFERENCE",
        }],
        "status": "OK",
    }
    span = Span.from_bigquery_row(row)
    assert len(span.content_parts) == 1
    assert span.content_parts[0].mime_type == "image/png"
    assert span.content_parts[0].uri == "gs://bucket/img.png"

  def test_label_tool_event(self):
    span = Span(
        event_type="TOOL_STARTING",
        agent="agent",
        timestamp=datetime.now(timezone.utc),
        content={"tool": "search_web"},
    )
    assert "TOOL_STARTING" in span.label
    assert "(search_web)" in span.label

  def test_label_error_event(self):
    span = Span(
        event_type="LLM_ERROR",
        agent="agent",
        timestamp=datetime.now(timezone.utc),
        status="ERROR",
    )
    assert "ERROR" in span.label

  def test_summary_with_text(self):
    span = Span(
        event_type="USER_MESSAGE_RECEIVED",
        agent=None,
        timestamp=datetime.now(timezone.utc),
        content={"text_summary": "What is the weather?"},
    )
    assert span.summary == "What is the weather?"

  def test_summary_truncation(self):
    long_text = "x" * 200
    span = Span(
        event_type="LLM_RESPONSE",
        agent="agent",
        timestamp=datetime.now(timezone.utc),
        content={"text_summary": long_text},
    )
    assert len(span.summary) == 120
    assert span.summary.endswith("...")

  def test_summary_from_error_message(self):
    span = Span(
        event_type="TOOL_ERROR",
        agent="agent",
        timestamp=datetime.now(timezone.utc),
        error_message="Connection refused",
        status="ERROR",
    )
    assert span.summary == "Connection refused"

  def test_summary_from_content_parts(self):
    span = Span(
        event_type="LLM_RESPONSE",
        agent="agent",
        timestamp=datetime.now(timezone.utc),
        content={},
        content_parts=[
            ContentPart(
                mime_type="image/png",
                uri="gs://bucket/image.png",
            )
        ],
    )
    assert "image/png" in span.summary
    assert "gs://bucket/image.png" in span.summary


class TestTrace:
  """Tests for Trace class."""

  def _make_spans(self):
    ts = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    return [
        Span(
            event_type="USER_MESSAGE_RECEIVED",
            agent=None,
            timestamp=ts,
            content={"text_summary": "Hello"},
            span_id="s1",
        ),
        Span(
            event_type="AGENT_STARTING",
            agent="my_agent",
            timestamp=ts,
            span_id="s2",
            parent_span_id="s1",
        ),
        Span(
            event_type="TOOL_STARTING",
            agent="my_agent",
            timestamp=ts,
            content={"tool": "search", "args": {"q": "hi"}},
            span_id="s3",
            parent_span_id="s2",
        ),
        Span(
            event_type="TOOL_COMPLETED",
            agent="my_agent",
            timestamp=ts,
            content={"tool": "search", "result": {"data": 1}},
            span_id="s4",
            parent_span_id="s2",
            latency_ms=100,
            status="OK",
        ),
        Span(
            event_type="AGENT_COMPLETED",
            agent="my_agent",
            timestamp=ts,
            content={"response": "Hi there!"},
            span_id="s5",
            parent_span_id="s1",
        ),
    ]

  def test_build_tree(self):
    trace = Trace(
        trace_id="t1",
        session_id="sess-1",
        spans=self._make_spans(),
    )
    roots = trace._build_tree()
    assert len(roots) == 1
    assert roots[0].span_id == "s1"
    assert len(roots[0].children) >= 1

  def test_render_tree(self):
    trace = Trace(
        trace_id="t1",
        session_id="sess-1",
        spans=self._make_spans(),
        total_latency_ms=500,
    )
    output = trace.render()
    assert "t1" in output
    assert "sess-1" in output
    assert "USER_MESSAGE_RECEIVED" in output
    assert "TOOL_STARTING" in output

  def test_render_flat_no_span_ids(self):
    ts = datetime.now(timezone.utc)
    spans = [
        Span(
            event_type="USER_MESSAGE_RECEIVED",
            agent=None,
            timestamp=ts,
            content={"text_summary": "Hello"},
        ),
        Span(
            event_type="AGENT_COMPLETED",
            agent="agent",
            timestamp=ts,
            content={"response": "Goodbye"},
        ),
    ]
    trace = Trace(
        trace_id="t2",
        session_id="sess-2",
        spans=spans,
    )
    output = trace.render()
    assert "USER_MESSAGE_RECEIVED" in output
    assert "AGENT_COMPLETED" in output

  def test_tool_calls_extraction(self):
    trace = Trace(
        trace_id="t1",
        session_id="sess-1",
        spans=self._make_spans(),
    )
    calls = trace.tool_calls
    assert len(calls) == 1
    assert calls[0]["tool_name"] == "search"

  def test_final_response(self):
    trace = Trace(
        trace_id="t1",
        session_id="sess-1",
        spans=self._make_spans(),
    )
    assert trace.final_response == "Hi there!"

  def test_error_spans(self):
    ts = datetime.now(timezone.utc)
    spans = [
        Span(
            event_type="TOOL_ERROR",
            agent="agent",
            timestamp=ts,
            status="ERROR",
            error_message="Timeout",
        ),
        Span(
            event_type="LLM_RESPONSE",
            agent="agent",
            timestamp=ts,
            status="OK",
        ),
    ]
    trace = Trace(
        trace_id="t",
        session_id="s",
        spans=spans,
    )
    assert len(trace.error_spans) == 1
    assert trace.error_spans[0].event_type == "TOOL_ERROR"


class TestTraceFilter:
  """Tests for TraceFilter class."""

  def test_empty_filter(self):
    filt = TraceFilter()
    where, params = filt.to_sql_conditions()
    assert where == "TRUE"
    assert len(params) == 1  # trace_limit

  def test_time_range_filter(self):
    filt = TraceFilter(
        start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_time=datetime(2024, 12, 31, tzinfo=timezone.utc),
    )
    where, params = filt.to_sql_conditions()
    assert "timestamp >= @start_time" in where
    assert "timestamp <= @end_time" in where
    assert len(params) == 3  # start, end, limit

  def test_agent_filter(self):
    filt = TraceFilter(agent_id="my_agent")
    where, _ = filt.to_sql_conditions()
    assert "agent = @agent_id" in where

  def test_error_filter(self):
    filt = TraceFilter(has_error=True)
    where, _ = filt.to_sql_conditions()
    assert "status = 'ERROR'" in where

  def test_session_ids_filter(self):
    filt = TraceFilter(session_ids=["s1", "s2"])
    where, _ = filt.to_sql_conditions()
    assert "session_id IN UNNEST(@session_ids)" in where

  def test_latency_filter(self):
    filt = TraceFilter(min_latency_ms=100, max_latency_ms=5000)
    where, _ = filt.to_sql_conditions()
    assert "@min_latency_ms" in where
    assert "@max_latency_ms" in where

  def test_combined_filters(self):
    filt = TraceFilter(
        agent_id="agent",
        has_error=True,
        user_id="user-1",
    )
    where, _ = filt.to_sql_conditions()
    assert " AND " in where
    assert "agent = @agent_id" in where
    assert "status = 'ERROR'" in where
    assert "user_id = @user_id" in where
