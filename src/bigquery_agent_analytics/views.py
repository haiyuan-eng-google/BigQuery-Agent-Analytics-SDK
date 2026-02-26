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

"""Event-specific BigQuery views for agent analytics.

Creates standard (non-materialized) BigQuery views that unnest the
generic ``agent_events`` table into per-event-type views with typed
columns.  Every view retains the standard identity headers:
``timestamp``, ``agent``, ``session_id``, ``invocation_id``.

Example usage::

    from bigquery_agent_analytics.views import ViewManager

    vm = ViewManager(
        project_id="my-project",
        dataset_id="analytics",
        table_id="agent_events",
    )
    vm.create_all_views()              # create all per-event views
    vm.create_view("LLM_REQUEST")      # create a single view
    print(vm.get_view_sql("TOOL_CALL"))  # inspect SQL without creating
"""

from __future__ import annotations

import logging
from typing import Optional

from google.cloud import bigquery

logger = logging.getLogger("bigquery_agent_analytics." + __name__)

# ------------------------------------------------------------------ #
# Standard header columns included in every view                       #
# ------------------------------------------------------------------ #

_STANDARD_HEADERS = """\
  timestamp,
  agent,
  session_id,
  invocation_id,
  user_id,
  trace_id,
  span_id,
  parent_span_id,
  status,
  error_message,
  is_truncated"""

# ------------------------------------------------------------------ #
# Per-event-type column definitions                                    #
# ------------------------------------------------------------------ #
# Each entry maps an event_type string to a tuple of
# (view_suffix, extra_columns_sql).  ``extra_columns_sql`` extracts
# event-specific fields from the ``content`` and ``attributes`` JSON
# columns into typed top-level columns.

_EVENT_VIEW_DEFS: dict[str, tuple[str, str]] = {
    "LLM_REQUEST": (
        "llm_requests",
        """\
  JSON_EXTRACT_SCALAR(attributes, '$.model') AS model,
  JSON_EXTRACT_SCALAR(attributes, '$.model_version') AS model_version,
  JSON_EXTRACT(attributes, '$.llm_config') AS llm_config,
  JSON_EXTRACT(attributes, '$.tools') AS tools,
  content""",
    ),
    "LLM_RESPONSE": (
        "llm_responses",
        """\
  JSON_EXTRACT_SCALAR(attributes, '$.model') AS model,
  JSON_EXTRACT_SCALAR(
    attributes, '$.usage_metadata.prompt_token_count'
  ) AS prompt_tokens,
  JSON_EXTRACT_SCALAR(
    attributes, '$.usage_metadata.candidates_token_count'
  ) AS candidate_tokens,
  JSON_EXTRACT_SCALAR(
    attributes, '$.usage_metadata.total_token_count'
  ) AS total_tokens,
  JSON_EXTRACT_SCALAR(latency_ms, '$.total_ms') AS total_ms,
  JSON_EXTRACT_SCALAR(
    latency_ms, '$.time_to_first_token_ms'
  ) AS time_to_first_token_ms,
  content""",
    ),
    "LLM_ERROR": (
        "llm_errors",
        """\
  JSON_EXTRACT_SCALAR(attributes, '$.model') AS model,
  error_message,
  JSON_EXTRACT_SCALAR(latency_ms, '$.total_ms') AS total_ms,
  content""",
    ),
    "TOOL_STARTING": (
        "tool_starts",
        """\
  JSON_EXTRACT_SCALAR(content, '$.tool') AS tool_name,
  JSON_EXTRACT_SCALAR(content, '$.tool_origin') AS tool_origin,
  JSON_EXTRACT(content, '$.args') AS tool_args""",
    ),
    "TOOL_COMPLETED": (
        "tool_completions",
        """\
  JSON_EXTRACT_SCALAR(content, '$.tool') AS tool_name,
  JSON_EXTRACT_SCALAR(content, '$.tool_origin') AS tool_origin,
  JSON_EXTRACT(content, '$.result') AS tool_result,
  JSON_EXTRACT_SCALAR(latency_ms, '$.total_ms') AS total_ms""",
    ),
    "TOOL_ERROR": (
        "tool_errors",
        """\
  JSON_EXTRACT_SCALAR(content, '$.tool') AS tool_name,
  JSON_EXTRACT_SCALAR(content, '$.tool_origin') AS tool_origin,
  JSON_EXTRACT(content, '$.args') AS tool_args,
  error_message,
  JSON_EXTRACT_SCALAR(latency_ms, '$.total_ms') AS total_ms""",
    ),
    "USER_MESSAGE_RECEIVED": (
        "user_messages",
        """\
  JSON_EXTRACT_SCALAR(content, '$.text_summary') AS text_summary,
  content""",
    ),
    "AGENT_STARTING": (
        "agent_starts",
        """\
  JSON_EXTRACT_SCALAR(
    attributes, '$.root_agent_name'
  ) AS root_agent_name,
  content""",
    ),
    "AGENT_COMPLETED": (
        "agent_completions",
        """\
  JSON_EXTRACT_SCALAR(
    attributes, '$.root_agent_name'
  ) AS root_agent_name,
  JSON_EXTRACT_SCALAR(latency_ms, '$.total_ms') AS total_ms,
  content""",
    ),
    "INVOCATION_STARTING": (
        "invocation_starts",
        """\
  JSON_EXTRACT_SCALAR(
    attributes, '$.root_agent_name'
  ) AS root_agent_name,
  content""",
    ),
    "INVOCATION_COMPLETED": (
        "invocation_completions",
        """\
  JSON_EXTRACT_SCALAR(
    attributes, '$.root_agent_name'
  ) AS root_agent_name,
  JSON_EXTRACT_SCALAR(latency_ms, '$.total_ms') AS total_ms,
  content""",
    ),
    "STATE_DELTA": (
        "state_deltas",
        """\
  JSON_EXTRACT(attributes, '$.state_delta') AS state_delta,
  content""",
    ),
    "HITL_CREDENTIAL_REQUEST": (
        "hitl_credential_requests",
        """\
  JSON_EXTRACT_SCALAR(content, '$.tool') AS tool_name,
  JSON_EXTRACT(content, '$.args') AS tool_args,
  content""",
    ),
    "HITL_CONFIRMATION_REQUEST": (
        "hitl_confirmation_requests",
        """\
  JSON_EXTRACT_SCALAR(content, '$.tool') AS tool_name,
  JSON_EXTRACT(content, '$.args') AS tool_args,
  content""",
    ),
    "HITL_INPUT_REQUEST": (
        "hitl_input_requests",
        """\
  JSON_EXTRACT_SCALAR(content, '$.tool') AS tool_name,
  JSON_EXTRACT(content, '$.args') AS tool_args,
  content""",
    ),
    "HITL_CREDENTIAL_REQUEST_COMPLETED": (
        "hitl_credential_completions",
        """\
  JSON_EXTRACT_SCALAR(content, '$.tool') AS tool_name,
  JSON_EXTRACT(content, '$.result') AS tool_result,
  content""",
    ),
    "HITL_CONFIRMATION_REQUEST_COMPLETED": (
        "hitl_confirmation_completions",
        """\
  JSON_EXTRACT_SCALAR(content, '$.tool') AS tool_name,
  JSON_EXTRACT(content, '$.result') AS tool_result,
  content""",
    ),
    "HITL_INPUT_REQUEST_COMPLETED": (
        "hitl_input_completions",
        """\
  JSON_EXTRACT_SCALAR(content, '$.tool') AS tool_name,
  JSON_EXTRACT(content, '$.result') AS tool_result,
  content""",
    ),
}

# ------------------------------------------------------------------ #
# View Template                                                        #
# ------------------------------------------------------------------ #

_VIEW_SQL_TEMPLATE = """\
CREATE OR REPLACE VIEW `{project}.{dataset}.{view_name}` AS
SELECT
{standard_headers},
{extra_columns}
FROM `{project}.{dataset}.{table}`
WHERE event_type = '{event_type}'
"""


def _build_view_sql(
    project: str,
    dataset: str,
    table: str,
    event_type: str,
    view_name: str,
    extra_columns: str,
) -> str:
  """Builds the CREATE OR REPLACE VIEW SQL for one event type."""
  return _VIEW_SQL_TEMPLATE.format(
      project=project,
      dataset=dataset,
      table=table,
      view_name=view_name,
      event_type=event_type,
      standard_headers=_STANDARD_HEADERS,
      extra_columns=extra_columns,
  )


# ------------------------------------------------------------------ #
# ViewManager                                                          #
# ------------------------------------------------------------------ #


class ViewManager:
  """Manages per-event-type BigQuery views over the agent events table.

  Args:
      project_id: Google Cloud project ID.
      dataset_id: BigQuery dataset containing agent events.
      table_id: Source table name (default ``agent_events``).
      view_prefix: Optional prefix for view names (e.g. ``"adk_"``).
      bq_client: Optional pre-configured BigQuery client.
  """

  def __init__(
      self,
      project_id: str,
      dataset_id: str,
      table_id: str = "agent_events",
      view_prefix: str = "adk_",
      bq_client: Optional[bigquery.Client] = None,
  ) -> None:
    self.project_id = project_id
    self.dataset_id = dataset_id
    self.table_id = table_id
    self.view_prefix = view_prefix
    self._bq_client = bq_client

  @property
  def bq_client(self) -> bigquery.Client:
    if self._bq_client is None:
      self._bq_client = bigquery.Client(project=self.project_id)
    return self._bq_client

  @property
  def available_event_types(self) -> list[str]:
    """Returns the list of event types with view definitions."""
    return sorted(_EVENT_VIEW_DEFS.keys())

  def get_view_name(self, event_type: str) -> str:
    """Returns the fully-qualified view name for an event type."""
    suffix = _EVENT_VIEW_DEFS[event_type][0]
    return f"{self.view_prefix}{suffix}"

  def get_view_sql(self, event_type: str) -> str:
    """Returns the SQL for a single event-type view.

    Args:
        event_type: One of the supported event type strings.

    Returns:
        The CREATE OR REPLACE VIEW SQL statement.

    Raises:
        KeyError: If the event_type is not recognized.
    """
    if event_type not in _EVENT_VIEW_DEFS:
      raise KeyError(
          f"Unknown event_type '{event_type}'. "
          f"Available: {self.available_event_types}"
      )
    suffix, extra_columns = _EVENT_VIEW_DEFS[event_type]
    view_name = f"{self.view_prefix}{suffix}"
    return _build_view_sql(
        project=self.project_id,
        dataset=self.dataset_id,
        table=self.table_id,
        event_type=event_type,
        view_name=view_name,
        extra_columns=extra_columns,
    )

  def create_view(self, event_type: str) -> None:
    """Creates (or replaces) the view for one event type.

    Args:
        event_type: The event type to create a view for.
    """
    sql = self.get_view_sql(event_type)
    view_name = self.get_view_name(event_type)
    logger.info(
        "Creating view %s.%s.%s", self.project_id, self.dataset_id, view_name
    )
    self.bq_client.query(sql).result()
    logger.info("View %s created successfully.", view_name)

  def create_all_views(self) -> dict[str, str]:
    """Creates views for all supported event types.

    Returns:
        A dict mapping event_type to view name for each created view.
    """
    created = {}
    for event_type in _EVENT_VIEW_DEFS:
      try:
        self.create_view(event_type)
        created[event_type] = self.get_view_name(event_type)
      except Exception as e:
        logger.error(
            "Failed to create view for %s: %s",
            event_type,
            e,
            exc_info=True,
        )
    return created
