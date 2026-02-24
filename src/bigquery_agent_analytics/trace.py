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

"""Trace reconstruction and visualization for BigQuery Agent Analytics SDK.

This module provides the Trace and Span objects that allow developers
to reconstruct and visualize agent conversation traces stored in
BigQuery. The key feature is ``trace.render()`` which generates a
hierarchical DAG view of the agent's reasoning steps.

Example usage::

    client = Client(project_id="my-project", dataset_id="analytics")
    trace = client.get_trace("trace-123")
    trace.render()  # Prints hierarchical DAG in notebook/terminal
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timezone
from enum import Enum
import json
import logging
from typing import Any
from typing import Optional

logger = logging.getLogger("bigquery_agent_analytics." + __name__)


class EventType(Enum):
  """Standard event types logged by the analytics plugin."""

  USER_MESSAGE_RECEIVED = "USER_MESSAGE_RECEIVED"
  INVOCATION_STARTING = "INVOCATION_STARTING"
  INVOCATION_COMPLETED = "INVOCATION_COMPLETED"
  AGENT_STARTING = "AGENT_STARTING"
  AGENT_COMPLETED = "AGENT_COMPLETED"
  LLM_REQUEST = "LLM_REQUEST"
  LLM_RESPONSE = "LLM_RESPONSE"
  LLM_ERROR = "LLM_ERROR"
  TOOL_STARTING = "TOOL_STARTING"
  TOOL_COMPLETED = "TOOL_COMPLETED"
  TOOL_ERROR = "TOOL_ERROR"
  STATE_DELTA = "STATE_DELTA"
  HITL_CONFIRMATION_REQUEST = "HITL_CONFIRMATION_REQUEST"
  HITL_CREDENTIAL_REQUEST = "HITL_CREDENTIAL_REQUEST"
  HITL_INPUT_REQUEST = "HITL_INPUT_REQUEST"
  HITL_CONFIRMATION_REQUEST_COMPLETED = (
      "HITL_CONFIRMATION_REQUEST_COMPLETED"
  )
  HITL_CREDENTIAL_REQUEST_COMPLETED = (
      "HITL_CREDENTIAL_REQUEST_COMPLETED"
  )
  HITL_INPUT_REQUEST_COMPLETED = (
      "HITL_INPUT_REQUEST_COMPLETED"
  )


@dataclass
class ContentPart:
  """A single part of multimodal content."""

  mime_type: Optional[str] = None
  text: Optional[str] = None
  uri: Optional[str] = None
  storage_mode: Optional[str] = None


@dataclass
class Span:
  """Represents a single span (event) in a trace.

  Spans form a tree structure via ``parent_span_id`` references.
  """

  event_type: str
  agent: Optional[str]
  timestamp: datetime
  content: dict[str, Any] = field(default_factory=dict)
  attributes: dict[str, Any] = field(default_factory=dict)
  span_id: Optional[str] = None
  parent_span_id: Optional[str] = None
  latency_ms: Optional[float] = None
  status: str = "OK"
  error_message: Optional[str] = None
  content_parts: list[ContentPart] = field(default_factory=list)
  children: list[Span] = field(default_factory=list)
  session_id: Optional[str] = None
  invocation_id: Optional[str] = None
  user_id: Optional[str] = None

  @classmethod
  def from_bigquery_row(cls, row: dict[str, Any]) -> Span:
    """Creates a Span from a BigQuery row dictionary."""
    content = row.get("content")
    if isinstance(content, str):
      try:
        content = json.loads(content)
      except (json.JSONDecodeError, TypeError):
        content = {"raw": content}
    elif content is None:
      content = {}

    attributes = row.get("attributes")
    if isinstance(attributes, str):
      try:
        attributes = json.loads(attributes)
      except (json.JSONDecodeError, TypeError):
        attributes = {}
    elif attributes is None:
      attributes = {}

    latency_ms = row.get("latency_ms")
    if isinstance(latency_ms, str):
      try:
        latency_data = json.loads(latency_ms)
        latency_ms = latency_data.get("total_ms")
      except (json.JSONDecodeError, TypeError):
        latency_ms = None
    elif isinstance(latency_ms, dict):
      latency_ms = latency_ms.get("total_ms")

    parts_raw = row.get("content_parts", [])
    content_parts = []
    if parts_raw:
      for p in parts_raw:
        content_parts.append(
            ContentPart(
                mime_type=p.get("mime_type"),
                text=p.get("text"),
                uri=p.get("uri"),
                storage_mode=p.get("storage_mode"),
            )
        )

    return cls(
        event_type=row.get("event_type", "UNKNOWN"),
        agent=row.get("agent"),
        timestamp=row.get("timestamp", datetime.now(timezone.utc)),
        content=content,
        attributes=attributes,
        span_id=row.get("span_id"),
        parent_span_id=row.get("parent_span_id"),
        latency_ms=latency_ms,
        status=row.get("status", "OK"),
        error_message=row.get("error_message"),
        content_parts=content_parts,
        session_id=row.get("session_id"),
        invocation_id=row.get("invocation_id"),
        user_id=row.get("user_id"),
    )

  @property
  def is_error(self) -> bool:
    """Returns True if this span has ERROR status."""
    return self.status == "ERROR"

  @property
  def subtree_has_error(self) -> bool:
    """Returns True if this span or any descendant has an error."""
    if self.is_error:
      return True
    return any(c.subtree_has_error for c in self.children)

  @property
  def failure_context(self) -> Optional[str]:
    """Returns a concise failure description if this span errored.

    Combines the event_type, tool name (if applicable), and the
    error_message into a single string for quick debugging.
    """
    if not self.is_error:
      return None
    parts = [self.event_type]
    tool = self.content.get("tool")
    if tool:
      parts.append(f"tool={tool}")
    if self.error_message:
      parts.append(self.error_message[:200])
    return " | ".join(parts)

  @property
  def label(self) -> str:
    """Returns a human-readable label for this span."""
    parts = [self.event_type]
    if self.agent:
      parts.append(f"[{self.agent}]")

    # Add contextual detail
    if self.event_type in ("TOOL_STARTING", "TOOL_COMPLETED", "TOOL_ERROR"):
      tool = self.content.get("tool", "")
      if tool:
        parts.append(f"({tool})")
    elif self.event_type == "LLM_REQUEST":
      model = self.attributes.get("model", "")
      if model:
        parts.append(f"({model})")

    if self.status == "ERROR":
      parts.append("ERROR")

    return " ".join(parts)

  @property
  def summary(self) -> str:
    """Returns a brief content summary for display."""
    if self.error_message:
      return self.error_message[:120]

    text = self.content.get("text_summary") or ""
    if not text:
      text = self.content.get("response") or ""
    if not text:
      text = self.content.get("text") or ""
    if not text and self.content_parts:
      for p in self.content_parts:
        if p.text:
          text = p.text
          break
        if p.uri:
          text = f"[{p.mime_type or 'file'}] {p.uri}"
          break

    if len(text) > 120:
      return text[:117] + "..."
    return text


@dataclass
class TraceFilter:
  """Filtering criteria for listing traces.

  All fields are optional. When multiple fields are set they
  are combined with AND logic.
  """

  start_time: Optional[datetime] = None
  end_time: Optional[datetime] = None
  agent_id: Optional[str] = None
  user_id: Optional[str] = None
  session_ids: Optional[list[str]] = None
  experiment_id: Optional[str] = None
  has_error: Optional[bool] = None
  error_type: Optional[str] = None
  custom_labels: Optional[dict[str, str]] = None
  min_latency_ms: Optional[float] = None
  max_latency_ms: Optional[float] = None
  event_types: Optional[list[str]] = None
  limit: int = 100

  def to_sql_conditions(self) -> tuple[str, list]:
    """Converts filter to SQL WHERE clauses and query parameters.

    Returns:
        Tuple of (SQL conditions string, list of BQ query params).
    """
    from google.cloud import bigquery

    conditions = []
    params = []

    if self.start_time:
      conditions.append("timestamp >= @start_time")
      params.append(
          bigquery.ScalarQueryParameter(
              "start_time",
              "TIMESTAMP",
              self.start_time,
          )
      )
    if self.end_time:
      conditions.append("timestamp <= @end_time")
      params.append(
          bigquery.ScalarQueryParameter(
              "end_time",
              "TIMESTAMP",
              self.end_time,
          )
      )
    if self.agent_id:
      conditions.append("agent = @agent_id")
      params.append(
          bigquery.ScalarQueryParameter(
              "agent_id",
              "STRING",
              self.agent_id,
          )
      )
    if self.user_id:
      conditions.append("user_id = @user_id")
      params.append(
          bigquery.ScalarQueryParameter(
              "user_id",
              "STRING",
              self.user_id,
          )
      )
    if self.session_ids:
      conditions.append("session_id IN UNNEST(@session_ids)")
      params.append(
          bigquery.ArrayQueryParameter(
              "session_ids",
              "STRING",
              self.session_ids,
          )
      )
    if self.has_error is True:
      conditions.append("status = 'ERROR'")
    elif self.has_error is False:
      conditions.append("status != 'ERROR'")
    if self.error_type:
      conditions.append("error_message LIKE @error_type")
      params.append(
          bigquery.ScalarQueryParameter(
              "error_type",
              "STRING",
              f"%{self.error_type}%",
          )
      )
    if self.min_latency_ms is not None:
      conditions.append(
          "CAST(JSON_EXTRACT_SCALAR(latency_ms, '$.total_ms')"
          " AS FLOAT64) >= @min_latency_ms"
      )
      params.append(
          bigquery.ScalarQueryParameter(
              "min_latency_ms",
              "FLOAT64",
              self.min_latency_ms,
          )
      )
    if self.max_latency_ms is not None:
      conditions.append(
          "CAST(JSON_EXTRACT_SCALAR(latency_ms, '$.total_ms')"
          " AS FLOAT64) <= @max_latency_ms"
      )
      params.append(
          bigquery.ScalarQueryParameter(
              "max_latency_ms",
              "FLOAT64",
              self.max_latency_ms,
          )
      )
    if self.event_types:
      conditions.append("event_type IN UNNEST(@event_types)")
      params.append(
          bigquery.ArrayQueryParameter(
              "event_types",
              "STRING",
              self.event_types,
          )
      )

    params.append(
        bigquery.ScalarQueryParameter(
            "trace_limit",
            "INT64",
            self.limit,
        )
    )

    where = " AND ".join(conditions) if conditions else "TRUE"
    return where, params


@dataclass
class Trace:
  """A complete agent trace for a session.

  Contains all spans (events) for the session and provides
  visualization via the :meth:`render` method.
  """

  trace_id: str
  session_id: str
  spans: list[Span] = field(default_factory=list)
  user_id: Optional[str] = None
  start_time: Optional[datetime] = None
  end_time: Optional[datetime] = None
  total_latency_ms: Optional[float] = None

  def _build_tree(self) -> list[Span]:
    """Builds a tree of spans using parent_span_id relationships."""
    by_id: dict[str, Span] = {}
    for span in self.spans:
      if span.span_id:
        by_id[span.span_id] = span
      span.children = []

    roots: list[Span] = []
    for span in self.spans:
      parent = span.parent_span_id
      if parent and parent in by_id:
        by_id[parent].children.append(span)
      else:
        roots.append(span)

    return roots

  def render(self, format: str = "tree") -> str:
    """Renders the trace as a hierarchical DAG view.

    This generates a tree representation of the agent's
    reasoning steps:
    ``User Input -> Agent Thought -> Tool Call -> Response``

    Multimodal content parts show their MIME type and URI.

    Args:
        format: Render format. Currently supports "tree".

    Returns:
        A string containing the rendered trace. Also printed
        to stdout for notebook/terminal use.
    """
    roots = self._build_tree()
    lines: list[str] = []

    header = f"Trace: {self.trace_id}"
    if self.session_id:
      header += f" | Session: {self.session_id}"
    if self.total_latency_ms is not None:
      header += f" | {self.total_latency_ms:.0f}ms"
    lines.append(header)
    lines.append("=" * len(header))

    if not roots:
      # Flat rendering when no span IDs exist
      for span in self.spans:
        self._render_flat_span(span, lines)
    else:
      for root in roots:
        self._render_span(root, lines, prefix="", is_last=True)

    output = "\n".join(lines)
    print(output)
    return output

  def _render_span(
      self,
      span: Span,
      lines: list[str],
      prefix: str,
      is_last: bool,
  ) -> None:
    """Recursively renders a span and its children as a tree."""
    connector = "\u2514\u2500 " if is_last else "\u251c\u2500 "

    if span.is_error:
      status_icon = "\u2717"
    elif span.subtree_has_error:
      # Propagate error visibility: mark parents whose subtree
      # contains an error so the failure is visible at every level.
      status_icon = "\u26a0"
    else:
      status_icon = "\u2713"

    latency_str = ""
    if span.latency_ms is not None:
      latency_str = f" ({span.latency_ms:.0f}ms)"

    line = f"{prefix}{connector}[{status_icon}] {span.label}"
    line += latency_str

    summary = span.summary
    if summary:
      line += f" - {summary}"

    lines.append(line)

    # Multimodal content parts
    child_prefix = prefix + ("   " if is_last else "\u2502  ")
    for part in span.content_parts:
      if part.uri:
        lines.append(
            f"{child_prefix}   [{part.mime_type or 'file'}] {part.uri}"
        )

    for i, child in enumerate(span.children):
      self._render_span(
          child,
          lines,
          child_prefix,
          is_last=(i == len(span.children) - 1),
      )

  def _render_flat_span(self, span: Span, lines: list[str]) -> None:
    """Renders a single span without tree structure."""
    status_icon = "\u2717" if span.status == "ERROR" else "\u2713"
    latency = ""
    if span.latency_ms is not None:
      latency = f" ({span.latency_ms:.0f}ms)"

    summary = span.summary
    detail = f" - {summary}" if summary else ""
    lines.append(f"  [{status_icon}] {span.label}{latency}{detail}")

  @property
  def tool_calls(self) -> list[dict[str, Any]]:
    """Extracts tool calls from the trace."""
    calls = []
    starts: dict[str, Span] = {}

    for span in self.spans:
      if span.event_type == "TOOL_STARTING":
        key = span.span_id or span.content.get("tool", "")
        starts[key] = span
      elif span.event_type in ("TOOL_COMPLETED", "TOOL_ERROR"):
        key = span.span_id or span.content.get("tool", "")
        start = starts.pop(key, None)
        calls.append({
            "tool_name": span.content.get("tool", "unknown"),
            "args": start.content.get("args", {}) if start else {},
            "result": span.content.get("result"),
            "status": span.status,
            "error": span.error_message,
            "latency_ms": span.latency_ms,
        })

    return calls

  @property
  def final_response(self) -> Optional[str]:
    """Extracts the final agent response text."""
    for span in reversed(self.spans):
      if span.event_type == "AGENT_COMPLETED":
        c = span.content
        if isinstance(c, dict):
          result = c.get("response") or c.get("text_summary")
          if result:
            return result
        elif c:
          return str(c)

    for span in reversed(self.spans):
      if span.event_type == "LLM_RESPONSE":
        c = span.content
        if isinstance(c, dict):
          result = c.get("response")
          if result:
            return result
        elif c:
          return str(c)
    return None

  @property
  def error_spans(self) -> list[Span]:
    """Returns all spans with ERROR status."""
    return [s for s in self.spans if s.status == "ERROR"]

  def errors(self) -> list[dict[str, Any]]:
    """Returns error spans with full failure context.

    Each entry contains the span's event_type, agent, tool name,
    error_message, latency, and span_id for easy debugging.

    Returns:
        List of dicts describing each error.
    """
    results = []
    for span in self.spans:
      if span.is_error:
        entry: dict[str, Any] = {
            "event_type": span.event_type,
            "agent": span.agent,
            "span_id": span.span_id,
            "error_message": span.error_message,
            "failure_context": span.failure_context,
            "latency_ms": span.latency_ms,
            "timestamp": span.timestamp,
        }
        tool = span.content.get("tool")
        if tool:
          entry["tool"] = tool
        origin = span.content.get("tool_origin")
        if origin:
          entry["tool_origin"] = origin
        results.append(entry)
    return results
