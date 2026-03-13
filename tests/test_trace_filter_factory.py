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

"""Tests for TraceFilter.from_cli_args() factory method."""

from datetime import datetime
from datetime import timedelta
from datetime import timezone

import pytest

from bigquery_agent_analytics.trace import TraceFilter


class TestTraceFilterFromCliArgs:

  def test_last_1h(self):
    tf = TraceFilter.from_cli_args(last="1h")
    assert tf.start_time is not None
    expected = datetime.now(timezone.utc) - timedelta(hours=1)
    delta = abs((tf.start_time - expected).total_seconds())
    assert delta < 5, f"start_time off by {delta}s"

  def test_last_7d(self):
    tf = TraceFilter.from_cli_args(last="7d")
    expected = datetime.now(timezone.utc) - timedelta(days=7)
    delta = abs((tf.start_time - expected).total_seconds())
    assert delta < 5

  def test_last_30m(self):
    tf = TraceFilter.from_cli_args(last="30m")
    expected = datetime.now(timezone.utc) - timedelta(minutes=30)
    delta = abs((tf.start_time - expected).total_seconds())
    assert delta < 5

  def test_last_24h(self):
    tf = TraceFilter.from_cli_args(last="24h")
    expected = datetime.now(timezone.utc) - timedelta(hours=24)
    delta = abs((tf.start_time - expected).total_seconds())
    assert delta < 5

  def test_agent_id(self):
    tf = TraceFilter.from_cli_args(agent_id="support_bot")
    assert tf.agent_id == "support_bot"
    assert tf.start_time is None

  def test_session_id_wraps_in_list(self):
    tf = TraceFilter.from_cli_args(session_id="sess-001")
    assert tf.session_ids == ["sess-001"]

  def test_session_id_none_leaves_none(self):
    tf = TraceFilter.from_cli_args()
    assert tf.session_ids is None

  def test_user_id(self):
    tf = TraceFilter.from_cli_args(user_id="u42")
    assert tf.user_id == "u42"

  def test_has_error(self):
    tf = TraceFilter.from_cli_args(has_error=True)
    assert tf.has_error is True

  def test_custom_limit(self):
    tf = TraceFilter.from_cli_args(limit=50)
    assert tf.limit == 50

  def test_combined(self):
    tf = TraceFilter.from_cli_args(
        last="7d",
        agent_id="bot",
        session_id="s1",
        limit=50,
    )
    assert tf.start_time is not None
    assert tf.agent_id == "bot"
    assert tf.session_ids == ["s1"]
    assert tf.limit == 50

  def test_defaults(self):
    tf = TraceFilter.from_cli_args()
    assert tf.start_time is None
    assert tf.end_time is None
    assert tf.agent_id is None
    assert tf.user_id is None
    assert tf.session_ids is None
    assert tf.has_error is None
    assert tf.limit == 100

  def test_invalid_last_raises(self):
    with pytest.raises(ValueError, match="Invalid time window"):
      TraceFilter.from_cli_args(last="foo")

  def test_invalid_unit_raises(self):
    with pytest.raises(ValueError, match="Invalid time window"):
      TraceFilter.from_cli_args(last="1w")

  def test_empty_string_raises(self):
    with pytest.raises(ValueError, match="Invalid time window"):
      TraceFilter.from_cli_args(last="")

  def test_case_insensitive(self):
    tf = TraceFilter.from_cli_args(last="1H")
    assert tf.start_time is not None

  def test_whitespace_stripped(self):
    tf = TraceFilter.from_cli_args(last=" 2d ")
    assert tf.start_time is not None
