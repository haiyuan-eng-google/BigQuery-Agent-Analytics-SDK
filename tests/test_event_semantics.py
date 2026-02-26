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

"""Tests for event_semantics module."""

from bigquery_agent_analytics.event_semantics import ALL_KNOWN_EVENT_TYPES
from bigquery_agent_analytics.event_semantics import EVENT_FAMILIES
from bigquery_agent_analytics.event_semantics import extract_response_text
from bigquery_agent_analytics.event_semantics import is_error_event
from bigquery_agent_analytics.event_semantics import is_hitl_event
from bigquery_agent_analytics.event_semantics import is_tool_event
from bigquery_agent_analytics.event_semantics import tool_outcome


class TestIsErrorEvent:
  """Tests for is_error_event()."""

  def test_error_event_type(self):
    assert is_error_event("LLM_ERROR") is True
    assert is_error_event("TOOL_ERROR") is True

  def test_error_message_set(self):
    assert is_error_event("TOOL_COMPLETED", error_message="fail") is True

  def test_status_error(self):
    assert is_error_event("TOOL_COMPLETED", status="ERROR") is True

  def test_no_error(self):
    assert is_error_event("TOOL_COMPLETED") is False
    assert is_error_event("LLM_RESPONSE") is False

  def test_combined(self):
    assert (
        is_error_event("TOOL_ERROR", error_message="x", status="ERROR") is True
    )


class TestExtractResponseText:
  """Tests for extract_response_text()."""

  def test_response_key(self):
    assert extract_response_text({"response": "hello"}) == "hello"

  def test_text_summary_key(self):
    assert extract_response_text({"text_summary": "hi"}) == "hi"

  def test_text_key(self):
    assert extract_response_text({"text": "hey"}) == "hey"

  def test_priority_order(self):
    content = {"text_summary": "a", "response": "b"}
    assert extract_response_text(content) == "b"

  def test_empty_dict(self):
    assert extract_response_text({}) is None

  def test_none_content(self):
    assert extract_response_text(None) is None

  def test_string_content(self):
    assert extract_response_text("raw text") == "raw text"


class TestIsToolEvent:
  """Tests for is_tool_event()."""

  def test_tool_events(self):
    assert is_tool_event("TOOL_STARTING") is True
    assert is_tool_event("TOOL_COMPLETED") is True
    assert is_tool_event("TOOL_ERROR") is True

  def test_non_tool_events(self):
    assert is_tool_event("LLM_REQUEST") is False
    assert is_tool_event("AGENT_COMPLETED") is False


class TestToolOutcome:
  """Tests for tool_outcome()."""

  def test_completed(self):
    assert tool_outcome("TOOL_COMPLETED") == "success"

  def test_error_type(self):
    assert tool_outcome("TOOL_ERROR") == "error"

  def test_error_status(self):
    assert tool_outcome("TOOL_COMPLETED", status="ERROR") == "error"

  def test_starting(self):
    assert tool_outcome("TOOL_STARTING") == "in_progress"


class TestIsHitlEvent:
  """Tests for is_hitl_event()."""

  def test_hitl_events(self):
    assert is_hitl_event("HITL_CONFIRMATION_REQUEST") is True
    assert is_hitl_event("HITL_INPUT_REQUEST_COMPLETED") is True

  def test_non_hitl(self):
    assert is_hitl_event("LLM_REQUEST") is False


class TestEventFamilies:
  """Tests for EVENT_FAMILIES and ALL_KNOWN_EVENT_TYPES."""

  def test_all_families_present(self):
    assert "user" in EVENT_FAMILIES
    assert "llm" in EVENT_FAMILIES
    assert "tool" in EVENT_FAMILIES
    assert "hitl" in EVENT_FAMILIES
    assert "state" in EVENT_FAMILIES

  def test_all_known_types_consistent(self):
    total = sum(len(v) for v in EVENT_FAMILIES.values())
    assert len(ALL_KNOWN_EVENT_TYPES) == total
