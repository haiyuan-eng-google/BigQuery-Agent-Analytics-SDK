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

"""Tests for the output formatter."""

from datetime import datetime
from datetime import timezone
import json

import pytest

from bigquery_agent_analytics.evaluators import EvaluationReport
from bigquery_agent_analytics.formatter import format_output
from bigquery_agent_analytics.trace import Span
from bigquery_agent_analytics.trace import Trace

_NOW = datetime(2026, 3, 12, 10, 0, 0, tzinfo=timezone.utc)


class TestFormatJson:

  def test_dict(self):
    out = format_output({"key": "val"}, "json")
    parsed = json.loads(out)
    assert parsed == {"key": "val"}

  def test_trace(self):
    trace = Trace(
        trace_id="t1",
        session_id="s1",
        spans=[
            Span(
                event_type="LLM_REQUEST",
                agent=None,
                timestamp=_NOW,
                content={},
                attributes={},
            )
        ],
        start_time=_NOW,
    )
    out = format_output(trace, "json")
    parsed = json.loads(out)
    assert parsed["trace_id"] == "t1"
    assert isinstance(parsed["start_time"], str)

  def test_datetime_in_dict(self):
    out = format_output({"ts": _NOW}, "json")
    parsed = json.loads(out)
    assert "2026-03-12" in parsed["ts"]

  def test_indented(self):
    out = format_output({"a": 1}, "json")
    assert "\n" in out  # indent=2 produces multiline


class TestFormatText:

  def test_uses_summary(self):
    report = EvaluationReport(
        dataset="test",
        evaluator_name="latency",
        total_sessions=10,
        passed_sessions=8,
        created_at=_NOW,
    )
    out = format_output(report, "text")
    # .summary() includes the evaluator name
    assert "latency" in out

  def test_fallback_to_json(self):
    out = format_output({"plain": "dict"}, "text")
    parsed = json.loads(out)
    assert parsed["plain"] == "dict"


class TestFormatTable:

  def test_list_of_dicts(self):
    data = [
        {"name": "alice", "score": 0.9},
        {"name": "bob", "score": 0.7},
    ]
    out = format_output(data, "table")
    lines = out.strip().split("\n")
    assert len(lines) == 4  # header + separator + 2 rows
    assert "name" in lines[0]
    assert "score" in lines[0]
    assert "---" in lines[1]
    assert "alice" in lines[2]

  def test_flat_dict(self):
    out = format_output({"key": "val", "num": 42}, "table")
    assert "key" in out
    assert "val" in out

  def test_empty_list(self):
    out = format_output([], "table")
    assert out == "[]"


class TestFormatUnknown:

  def test_raises_on_bad_format(self):
    with pytest.raises(ValueError, match="Unknown format"):
      format_output({}, "csv")
