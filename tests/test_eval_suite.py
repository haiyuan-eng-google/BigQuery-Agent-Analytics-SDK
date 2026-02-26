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

"""Tests for the eval_suite module."""

import pytest

from bigquery_agent_analytics.eval_suite import EvalCategory
from bigquery_agent_analytics.eval_suite import EvalSuite
from bigquery_agent_analytics.eval_suite import EvalTaskDef
from bigquery_agent_analytics.eval_suite import SuiteHealth


def _make_task(
    task_id="t1",
    session_id="s1",
    category=EvalCategory.CAPABILITY,
    tags=None,
    is_positive=True,
    expected_trajectory=None,
    expected_response=None,
    thresholds=None,
):
  return EvalTaskDef(
      task_id=task_id,
      session_id=session_id,
      description=f"Task {task_id}",
      category=category,
      tags=tags or [],
      is_positive_case=is_positive,
      expected_trajectory=expected_trajectory,
      expected_response=expected_response,
      thresholds=thresholds or {},
  )


# ------------------------------------------------------------------ #
# Tests for add/remove/get_tasks                                       #
# ------------------------------------------------------------------ #


class TestEvalSuiteBasic:
  """Tests for basic suite operations."""

  def test_add_task(self):
    suite = EvalSuite(name="test")
    suite.add_task(_make_task("t1"))
    assert len(suite.get_tasks()) == 1

  def test_add_duplicate_raises(self):
    suite = EvalSuite(name="test")
    suite.add_task(_make_task("t1"))
    with pytest.raises(ValueError, match="already exists"):
      suite.add_task(_make_task("t1"))

  def test_remove_task(self):
    suite = EvalSuite(name="test")
    suite.add_task(_make_task("t1"))
    suite.remove_task("t1")
    assert len(suite.get_tasks()) == 0

  def test_remove_missing_raises(self):
    suite = EvalSuite(name="test")
    with pytest.raises(KeyError, match="not found"):
      suite.remove_task("nonexistent")

  def test_get_tasks_filter_category(self):
    suite = EvalSuite(name="test")
    suite.add_task(_make_task("t1", category=EvalCategory.CAPABILITY))
    suite.add_task(
        _make_task(
            "t2",
            session_id="s2",
            category=EvalCategory.REGRESSION,
        )
    )

    cap = suite.get_tasks(category=EvalCategory.CAPABILITY)
    reg = suite.get_tasks(category=EvalCategory.REGRESSION)

    assert len(cap) == 1
    assert cap[0].task_id == "t1"
    assert len(reg) == 1
    assert reg[0].task_id == "t2"

  def test_get_tasks_filter_tags(self):
    suite = EvalSuite(name="test")
    suite.add_task(_make_task("t1", tags=["search", "v1"]))
    suite.add_task(_make_task("t2", session_id="s2", tags=["search", "v2"]))
    suite.add_task(_make_task("t3", session_id="s3", tags=["chat"]))

    search_tasks = suite.get_tasks(tags=["search"])
    assert len(search_tasks) == 2

    v1_search = suite.get_tasks(tags=["search", "v1"])
    assert len(v1_search) == 1
    assert v1_search[0].task_id == "t1"

  def test_init_with_tasks(self):
    tasks = [
        _make_task("t1"),
        _make_task("t2", session_id="s2"),
    ]
    suite = EvalSuite(name="test", tasks=tasks)
    assert len(suite.get_tasks()) == 2


# ------------------------------------------------------------------ #
# Tests for check_health                                               #
# ------------------------------------------------------------------ #


class TestEvalSuiteHealth:
  """Tests for suite health checking."""

  def test_empty_suite(self):
    suite = EvalSuite(name="empty")
    health = suite.check_health()
    assert health.total_tasks == 0
    assert "empty" in health.warnings[0].lower()

  def test_balanced_suite(self):
    suite = EvalSuite(name="test")
    # 5 positive, 3 negative = 62.5% positive (in range)
    for i in range(5):
      suite.add_task(
          _make_task(
              f"p{i}",
              session_id=f"sp{i}",
              is_positive=True,
          )
      )
    for i in range(3):
      suite.add_task(
          _make_task(
              f"n{i}",
              session_id=f"sn{i}",
              is_positive=False,
          )
      )

    health = suite.check_health()
    assert health.total_tasks == 8
    assert health.positive_cases == 5
    assert health.negative_cases == 3
    assert 0.3 <= health.balance_ratio <= 0.7

  def test_imbalanced_high(self):
    suite = EvalSuite(name="test")
    for i in range(9):
      suite.add_task(
          _make_task(
              f"p{i}",
              session_id=f"sp{i}",
              is_positive=True,
          )
      )
    suite.add_task(
        _make_task(
            "n0",
            session_id="sn0",
            is_positive=False,
        )
    )

    health = suite.check_health()
    assert any("High positive" in w for w in health.warnings)

  def test_imbalanced_low(self):
    suite = EvalSuite(name="test")
    suite.add_task(_make_task("p0", session_id="sp0", is_positive=True))
    for i in range(9):
      suite.add_task(
          _make_task(
              f"n{i}",
              session_id=f"sn{i}",
              is_positive=False,
          )
      )

    health = suite.check_health()
    assert any("Low positive" in w for w in health.warnings)

  def test_saturation_detection(self):
    suite = EvalSuite(name="test")
    suite.add_task(_make_task("t1"))
    suite.add_task(_make_task("t2", session_id="s2"))

    history = {
        "t1": [True, True, True, True, True],
        "t2": [True, True, False, True, True],
    }

    health = suite.check_health(pass_history=history)
    assert "t1" in health.saturated_task_ids
    assert "t2" not in health.saturated_task_ids

  def test_missing_expectations_warning(self):
    suite = EvalSuite(name="test")
    suite.add_task(_make_task("t1"))  # No trajectory or response

    health = suite.check_health()
    assert any("missing" in w.lower() for w in health.warnings)


# ------------------------------------------------------------------ #
# Tests for graduation                                                 #
# ------------------------------------------------------------------ #


class TestEvalSuiteGraduation:
  """Tests for task graduation."""

  def test_graduate_to_regression(self):
    suite = EvalSuite(name="test")
    suite.add_task(_make_task("t1"))
    suite.graduate_to_regression("t1")

    tasks = suite.get_tasks(category=EvalCategory.REGRESSION)
    assert len(tasks) == 1
    assert tasks[0].task_id == "t1"

  def test_graduate_missing_raises(self):
    suite = EvalSuite(name="test")
    with pytest.raises(KeyError):
      suite.graduate_to_regression("nonexistent")

  def test_graduate_already_regression_raises(self):
    suite = EvalSuite(name="test")
    suite.add_task(_make_task("t1", category=EvalCategory.REGRESSION))
    with pytest.raises(ValueError, match="already"):
      suite.graduate_to_regression("t1")

  def test_auto_graduate(self):
    suite = EvalSuite(name="test")
    suite.add_task(_make_task("t1"))
    suite.add_task(_make_task("t2", session_id="s2"))
    suite.add_task(_make_task("t3", session_id="s3"))

    history = {
        "t1": [True] * 10,  # Should graduate
        "t2": [True] * 8 + [False, True],  # Not all pass
        "t3": [True] * 5,  # Not enough runs
    }

    graduated = suite.auto_graduate(history, threshold_runs=10)
    assert graduated == ["t1"]

    # Verify t1 is now regression
    t1_tasks = suite.get_tasks(category=EvalCategory.REGRESSION)
    assert len(t1_tasks) == 1

  def test_auto_graduate_skips_regression(self):
    suite = EvalSuite(name="test")
    suite.add_task(_make_task("t1", category=EvalCategory.REGRESSION))

    history = {"t1": [True] * 10}
    graduated = suite.auto_graduate(history, threshold_runs=10)
    assert graduated == []


# ------------------------------------------------------------------ #
# Tests for serialization                                              #
# ------------------------------------------------------------------ #


class TestEvalSuiteSerialization:
  """Tests for JSON serialization."""

  def test_round_trip(self):
    suite = EvalSuite(name="my_suite")
    suite.add_task(
        _make_task(
            "t1",
            tags=["search"],
            expected_trajectory=[{"tool_name": "search", "args": {}}],
        )
    )
    suite.add_task(
        _make_task(
            "t2",
            session_id="s2",
            category=EvalCategory.REGRESSION,
            expected_response="hello",
        )
    )

    json_str = suite.to_json()
    restored = EvalSuite.from_json(json_str)

    assert restored.name == "my_suite"
    tasks = restored.get_tasks()
    assert len(tasks) == 2

    t1 = [t for t in tasks if t.task_id == "t1"][0]
    assert t1.tags == ["search"]
    assert t1.expected_trajectory == [{"tool_name": "search", "args": {}}]

    t2 = [t for t in tasks if t.task_id == "t2"][0]
    assert t2.category == EvalCategory.REGRESSION
    assert t2.expected_response == "hello"


# ------------------------------------------------------------------ #
# Tests for to_eval_dataset                                            #
# ------------------------------------------------------------------ #


class TestEvalSuiteDataset:
  """Tests for to_eval_dataset."""

  def test_basic(self):
    suite = EvalSuite(name="test")
    suite.add_task(
        _make_task(
            "t1",
            expected_trajectory=[{"tool_name": "search", "args": {}}],
            expected_response="done",
        )
    )

    dataset = suite.to_eval_dataset()
    assert len(dataset) == 1
    assert dataset[0]["session_id"] == "s1"
    assert "expected_trajectory" in dataset[0]
    assert "expected_response" in dataset[0]

  def test_filter_by_category(self):
    suite = EvalSuite(name="test")
    suite.add_task(_make_task("t1", category=EvalCategory.CAPABILITY))
    suite.add_task(
        _make_task(
            "t2",
            session_id="s2",
            category=EvalCategory.REGRESSION,
        )
    )

    cap_dataset = suite.to_eval_dataset(category=EvalCategory.CAPABILITY)
    assert len(cap_dataset) == 1
    assert cap_dataset[0]["session_id"] == "s1"

  def test_omits_none_fields(self):
    suite = EvalSuite(name="test")
    suite.add_task(_make_task("t1"))  # No trajectory or response

    dataset = suite.to_eval_dataset()
    assert "expected_trajectory" not in dataset[0]
    assert "expected_response" not in dataset[0]
