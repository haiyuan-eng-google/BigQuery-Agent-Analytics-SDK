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

"""Tests for the eval_validator module."""

import pytest

from bigquery_agent_analytics.eval_suite import EvalCategory
from bigquery_agent_analytics.eval_suite import EvalSuite
from bigquery_agent_analytics.eval_suite import EvalTaskDef
from bigquery_agent_analytics.eval_validator import EvalValidator
from bigquery_agent_analytics.eval_validator import ValidationWarning


def _make_task(
    task_id="t1",
    session_id="s1",
    is_positive=True,
    expected_trajectory=None,
    expected_response=None,
    thresholds=None,
):
  return EvalTaskDef(
      task_id=task_id,
      session_id=session_id,
      description=f"Task {task_id}",
      is_positive_case=is_positive,
      expected_trajectory=expected_trajectory,
      expected_response=expected_response,
      thresholds=thresholds or {},
  )


# ------------------------------------------------------------------ #
# Tests for check_ambiguity                                            #
# ------------------------------------------------------------------ #


class TestCheckAmbiguity:
  """Tests for ambiguity checking."""

  def test_ambiguous_task(self):
    tasks = [_make_task("t1")]  # No trajectory or response
    warnings = EvalValidator.check_ambiguity(tasks)
    assert len(warnings) == 1
    assert warnings[0].check_name == "ambiguity"
    assert warnings[0].task_id == "t1"

  def test_non_ambiguous_with_trajectory(self):
    tasks = [
        _make_task(
            "t1",
            expected_trajectory=[{"tool_name": "search", "args": {}}],
        )
    ]
    warnings = EvalValidator.check_ambiguity(tasks)
    assert len(warnings) == 0

  def test_non_ambiguous_with_response(self):
    tasks = [_make_task("t1", expected_response="hello")]
    warnings = EvalValidator.check_ambiguity(tasks)
    assert len(warnings) == 0

  def test_empty_list(self):
    warnings = EvalValidator.check_ambiguity([])
    assert len(warnings) == 0


# ------------------------------------------------------------------ #
# Tests for check_balance                                              #
# ------------------------------------------------------------------ #


class TestCheckBalance:
  """Tests for balance checking."""

  def test_balanced(self):
    tasks = [
        _make_task(
            f"p{i}",
            session_id=f"sp{i}",
            is_positive=True,
        )
        for i in range(5)
    ] + [
        _make_task(
            f"n{i}",
            session_id=f"sn{i}",
            is_positive=False,
        )
        for i in range(5)
    ]
    warnings = EvalValidator.check_balance(tasks)
    assert len(warnings) == 0

  def test_too_many_positive(self):
    tasks = [
        _make_task(
            f"p{i}",
            session_id=f"sp{i}",
            is_positive=True,
        )
        for i in range(8)
    ] + [_make_task("n0", session_id="sn0", is_positive=False)]
    warnings = EvalValidator.check_balance(tasks)
    assert len(warnings) == 1
    assert "High positive" in warnings[0].message

  def test_too_many_negative(self):
    tasks = [_make_task("p0", session_id="sp0", is_positive=True)] + [
        _make_task(
            f"n{i}",
            session_id=f"sn{i}",
            is_positive=False,
        )
        for i in range(8)
    ]
    warnings = EvalValidator.check_balance(tasks)
    assert len(warnings) == 1
    assert "Low positive" in warnings[0].message

  def test_empty(self):
    warnings = EvalValidator.check_balance([])
    assert len(warnings) == 0


# ------------------------------------------------------------------ #
# Tests for check_threshold_consistency                                #
# ------------------------------------------------------------------ #


class TestCheckThresholdConsistency:
  """Tests for threshold consistency checking."""

  def test_zero_threshold(self):
    tasks = [_make_task("t1", thresholds={"accuracy": 0.0})]
    warnings = EvalValidator.check_threshold_consistency(tasks)
    assert len(warnings) == 1
    assert "0.0" in warnings[0].message

  def test_one_threshold(self):
    tasks = [_make_task("t1", thresholds={"accuracy": 1.0})]
    warnings = EvalValidator.check_threshold_consistency(tasks)
    assert len(warnings) == 1
    assert "1.0" in warnings[0].message

  def test_normal_threshold(self):
    tasks = [_make_task("t1", thresholds={"accuracy": 0.7})]
    warnings = EvalValidator.check_threshold_consistency(tasks)
    assert len(warnings) == 0

  def test_no_thresholds(self):
    tasks = [_make_task("t1")]
    warnings = EvalValidator.check_threshold_consistency(tasks)
    assert len(warnings) == 0


# ------------------------------------------------------------------ #
# Tests for check_duplicate_sessions                                   #
# ------------------------------------------------------------------ #


class TestCheckDuplicateSessions:
  """Tests for duplicate session checking."""

  def test_duplicates(self):
    tasks = [
        _make_task("t1", session_id="s1"),
        _make_task("t2", session_id="s1"),
    ]
    warnings = EvalValidator.check_duplicate_sessions(tasks)
    assert len(warnings) == 2
    assert all(w.check_name == "duplicate_sessions" for w in warnings)

  def test_no_duplicates(self):
    tasks = [
        _make_task("t1", session_id="s1"),
        _make_task("t2", session_id="s2"),
    ]
    warnings = EvalValidator.check_duplicate_sessions(tasks)
    assert len(warnings) == 0

  def test_empty(self):
    warnings = EvalValidator.check_duplicate_sessions([])
    assert len(warnings) == 0


# ------------------------------------------------------------------ #
# Tests for check_saturation                                           #
# ------------------------------------------------------------------ #


class TestCheckSaturation:
  """Tests for saturation checking."""

  def test_saturated(self):
    history = {"t1": [True] * 5}
    warnings = EvalValidator.check_saturation(history, min_runs=5)
    assert len(warnings) == 1
    assert warnings[0].task_id == "t1"
    assert warnings[0].check_name == "saturation"

  def test_not_saturated(self):
    history = {"t1": [True, True, False, True, True]}
    warnings = EvalValidator.check_saturation(history, min_runs=5)
    assert len(warnings) == 0

  def test_not_enough_runs(self):
    history = {"t1": [True, True, True]}
    warnings = EvalValidator.check_saturation(history, min_runs=5)
    assert len(warnings) == 0

  def test_empty_history(self):
    warnings = EvalValidator.check_saturation({})
    assert len(warnings) == 0


# ------------------------------------------------------------------ #
# Tests for validate_suite                                             #
# ------------------------------------------------------------------ #


class TestValidateSuite:
  """Tests for the full validate_suite method."""

  def test_aggregates_all_checks(self):
    suite = EvalSuite(name="test")
    # Ambiguous (no trajectory or response)
    suite.add_task(_make_task("t1"))
    # Duplicate session
    suite.add_task(_make_task("t2", session_id="s1"))
    # Bad threshold
    suite.add_task(
        _make_task(
            "t3",
            session_id="s3",
            thresholds={"m": 0.0},
        )
    )

    warnings = EvalValidator.validate_suite(suite)

    check_names = {w.check_name for w in warnings}
    assert "ambiguity" in check_names
    assert "duplicate_sessions" in check_names
    assert "threshold_consistency" in check_names

  def test_empty_suite(self):
    suite = EvalSuite(name="empty")
    warnings = EvalValidator.validate_suite(suite)
    # Empty suite should not crash
    assert isinstance(warnings, list)

  def test_with_pass_history(self):
    suite = EvalSuite(name="test")
    suite.add_task(
        _make_task(
            "t1",
            expected_trajectory=[{"tool_name": "a", "args": {}}],
        )
    )

    history = {"t1": [True] * 5}
    warnings = EvalValidator.validate_suite(suite, pass_history=history)

    sat_warnings = [w for w in warnings if w.check_name == "saturation"]
    assert len(sat_warnings) == 1

  def test_single_task(self):
    suite = EvalSuite(name="test")
    suite.add_task(
        _make_task(
            "t1",
            expected_response="ok",
            is_positive=True,
        )
    )
    warnings = EvalValidator.validate_suite(suite)
    # Single positive task => high balance ratio warning
    balance = [w for w in warnings if w.check_name == "balance"]
    assert len(balance) == 1

  def test_all_same_session(self):
    suite = EvalSuite(name="test")
    for i in range(3):
      suite.add_task(_make_task(f"t{i}", session_id="same_session"))

    warnings = EvalValidator.validate_suite(suite)
    dup_warnings = [w for w in warnings if w.check_name == "duplicate_sessions"]
    assert len(dup_warnings) == 3
