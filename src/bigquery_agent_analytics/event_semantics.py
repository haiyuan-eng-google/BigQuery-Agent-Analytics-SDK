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

"""Canonical event semantic layer for BigQuery Agent Analytics SDK.

Centralizes the logic for interpreting ADK plugin events so that
every module (evaluators, memory, insights, trace) uses consistent
definitions for "final response", "error event", "tool outcome",
etc.  Import helpers from this module instead of re-implementing
event-type checks in each module.

Example usage::

    from bigquery_agent_analytics.event_semantics import (
        is_error_event,
        extract_response_text,
    )

    for span in trace.spans:
        if is_error_event(span.event_type, span.error_message,
                          span.status):
            print("Error:", span.error_message)
"""

from __future__ import annotations

from typing import Any, Optional

# ------------------------------------------------------------------ #
# Error Detection                                                      #
# ------------------------------------------------------------------ #


def is_error_event(
    event_type: str,
    error_message: Optional[str] = None,
    status: str = "OK",
) -> bool:
  """Returns True when the event represents an error.

  Uses the canonical predicate aligned with the ADK plugin:
  the event type ends with ``_ERROR``, the ``error_message``
  column is populated, or the ``status`` column is ``'ERROR'``.

  Args:
      event_type: The event_type column value.
      error_message: The error_message column value.
      status: The status column value.
  """
  return (
      event_type.endswith("_ERROR")
      or error_message is not None
      or status == "ERROR"
  )


# SQL fragment for use in BigQuery WHERE clauses.
ERROR_SQL_PREDICATE = (
    "(ENDS_WITH(event_type, '_ERROR')"
    " OR error_message IS NOT NULL"
    " OR status = 'ERROR')"
)

# Negated version for filtering out errors.
NO_ERROR_SQL_PREDICATE = (
    "NOT ENDS_WITH(event_type, '_ERROR')"
    " AND error_message IS NULL"
    " AND status != 'ERROR'"
)


# ------------------------------------------------------------------ #
# Response Extraction                                                  #
# ------------------------------------------------------------------ #


def extract_response_text(content: dict[str, Any]) -> Optional[str]:
  """Extracts user-visible response text from a content dict.

  Checks keys in priority order: ``response``, ``text_summary``,
  ``text``, ``raw``.

  Args:
      content: The parsed ``content`` JSON column.

  Returns:
      The response text or ``None``.
  """
  if not isinstance(content, dict):
    return str(content) if content else None
  return (
      content.get("response")
      or content.get("text_summary")
      or content.get("text")
      or content.get("raw")
      or None
  )


# Event types that carry the final agent response, in priority order.
RESPONSE_EVENT_TYPES = ("LLM_RESPONSE", "AGENT_COMPLETED")


# ------------------------------------------------------------------ #
# Tool Outcome                                                         #
# ------------------------------------------------------------------ #


def is_tool_event(event_type: str) -> bool:
  """Returns True for tool-related event types."""
  return event_type in (
      "TOOL_STARTING",
      "TOOL_COMPLETED",
      "TOOL_ERROR",
  )


def tool_outcome(event_type: str, status: str = "OK") -> str:
  """Returns a canonical tool outcome string.

  Args:
      event_type: The event type.
      status: The status column.

  Returns:
      One of ``"success"``, ``"error"``, or ``"in_progress"``.
  """
  if event_type == "TOOL_ERROR" or status == "ERROR":
    return "error"
  if event_type == "TOOL_COMPLETED":
    return "success"
  return "in_progress"


# ------------------------------------------------------------------ #
# Event Classification                                                 #
# ------------------------------------------------------------------ #


def is_hitl_event(event_type: str) -> bool:
  """Returns True for Human-in-the-Loop event types."""
  return event_type.startswith("HITL_")


def is_hitl_completed(event_type: str) -> bool:
  """Returns True for completed HITL events."""
  return event_type.startswith("HITL_") and event_type.endswith("_COMPLETED")


# All event types known to the SDK, grouped by family.
EVENT_FAMILIES = {
    "user": ["USER_MESSAGE_RECEIVED"],
    "invocation": [
        "INVOCATION_STARTING",
        "INVOCATION_COMPLETED",
    ],
    "agent": ["AGENT_STARTING", "AGENT_COMPLETED"],
    "llm": ["LLM_REQUEST", "LLM_RESPONSE", "LLM_ERROR"],
    "tool": [
        "TOOL_STARTING",
        "TOOL_COMPLETED",
        "TOOL_ERROR",
    ],
    "state": ["STATE_DELTA"],
    "hitl": [
        "HITL_CONFIRMATION_REQUEST",
        "HITL_CONFIRMATION_REQUEST_COMPLETED",
        "HITL_CREDENTIAL_REQUEST",
        "HITL_CREDENTIAL_REQUEST_COMPLETED",
        "HITL_INPUT_REQUEST",
        "HITL_INPUT_REQUEST_COMPLETED",
    ],
}

ALL_KNOWN_EVENT_TYPES = [
    et for family in EVENT_FAMILIES.values() for et in family
]
