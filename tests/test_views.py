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

"""Tests for the ViewManager and event-specific view generation."""

from unittest import mock

import pytest

from bigquery_agent_analytics.views import _EVENT_VIEW_DEFS
from bigquery_agent_analytics.views import ViewManager


PROJECT = "test-project"
DATASET = "analytics"
TABLE = "agent_events"


@pytest.fixture
def vm():
  return ViewManager(
      project_id=PROJECT,
      dataset_id=DATASET,
      table_id=TABLE,
      bq_client=mock.MagicMock(),
  )


class TestViewManager:

  def test_available_event_types(self, vm):
    types = vm.available_event_types
    assert "LLM_REQUEST" in types
    assert "TOOL_STARTING" in types
    assert "TOOL_COMPLETED" in types
    assert "TOOL_ERROR" in types
    assert "STATE_DELTA" in types
    assert len(types) == len(_EVENT_VIEW_DEFS)

  def test_get_view_name(self, vm):
    assert vm.get_view_name("LLM_REQUEST") == "adk_llm_requests"
    assert vm.get_view_name("TOOL_STARTING") == "adk_tool_starts"

  def test_get_view_sql_contains_event_filter(self, vm):
    sql = vm.get_view_sql("LLM_REQUEST")
    assert "WHERE event_type = 'LLM_REQUEST'" in sql
    assert "CREATE OR REPLACE VIEW" in sql
    assert f"`{PROJECT}.{DATASET}." in sql

  def test_get_view_sql_has_standard_headers(self, vm):
    sql = vm.get_view_sql("TOOL_STARTING")
    for header in [
        "timestamp", "agent", "session_id",
        "invocation_id", "span_id",
    ]:
      assert header in sql

  def test_get_view_sql_llm_request_columns(self, vm):
    sql = vm.get_view_sql("LLM_REQUEST")
    assert "model" in sql
    assert "model_version" in sql
    assert "llm_config" in sql

  def test_get_view_sql_tool_starting_columns(self, vm):
    sql = vm.get_view_sql("TOOL_STARTING")
    assert "tool_name" in sql
    assert "tool_origin" in sql
    assert "tool_args" in sql

  def test_get_view_sql_tool_completed_columns(self, vm):
    sql = vm.get_view_sql("TOOL_COMPLETED")
    assert "tool_name" in sql
    assert "tool_result" in sql
    assert "total_ms" in sql

  def test_get_view_sql_llm_response_tokens(self, vm):
    sql = vm.get_view_sql("LLM_RESPONSE")
    assert "prompt_tokens" in sql
    assert "candidate_tokens" in sql
    assert "total_tokens" in sql

  def test_get_view_sql_unknown_event_raises(self, vm):
    with pytest.raises(KeyError, match="Unknown event_type"):
      vm.get_view_sql("NONEXISTENT_TYPE")

  def test_create_view_executes_sql(self, vm):
    vm.create_view("LLM_REQUEST")
    vm.bq_client.query.assert_called_once()
    sql = vm.bq_client.query.call_args[0][0]
    assert "LLM_REQUEST" in sql
    vm.bq_client.query.return_value.result.assert_called_once()

  def test_create_all_views(self, vm):
    created = vm.create_all_views()
    assert len(created) == len(_EVENT_VIEW_DEFS)
    assert vm.bq_client.query.call_count == len(_EVENT_VIEW_DEFS)

  def test_create_all_views_handles_errors(self, vm):
    vm.bq_client.query.side_effect = Exception("BQ error")
    created = vm.create_all_views()
    assert len(created) == 0

  def test_custom_prefix(self):
    vm = ViewManager(
        project_id=PROJECT,
        dataset_id=DATASET,
        view_prefix="custom_",
        bq_client=mock.MagicMock(),
    )
    assert vm.get_view_name("LLM_REQUEST") == "custom_llm_requests"
    sql = vm.get_view_sql("LLM_REQUEST")
    assert "custom_llm_requests" in sql

  def test_all_event_defs_produce_valid_sql(self, vm):
    """Every defined event type produces SQL without errors."""
    for event_type in _EVENT_VIEW_DEFS:
      sql = vm.get_view_sql(event_type)
      assert "CREATE OR REPLACE VIEW" in sql
      assert f"event_type = '{event_type}'" in sql
