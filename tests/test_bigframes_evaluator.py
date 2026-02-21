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

"""Tests for BigFrames evaluator."""

import sys
from unittest.mock import MagicMock
from unittest.mock import patch

from bigquery_agent_analytics.bigframes_evaluator import BigFramesEvaluator
import pytest


class TestBigFramesEvaluatorInit:
  """Tests for BigFramesEvaluator initialization."""

  def test_default_endpoint(self):
    ev = BigFramesEvaluator(
        project_id="proj",
        dataset_id="ds",
    )
    assert ev.endpoint == "gemini-2.5-flash"
    assert ev.connection_id is None
    assert ev.table_id == "agent_events"

  def test_custom_endpoint(self):
    ev = BigFramesEvaluator(
        project_id="proj",
        dataset_id="ds",
        endpoint="gemini-2.5-pro",
    )
    assert ev.endpoint == "gemini-2.5-pro"

  def test_custom_connection(self):
    ev = BigFramesEvaluator(
        project_id="proj",
        dataset_id="ds",
        connection_id="us.my-conn",
    )
    assert ev.connection_id == "us.my-conn"

  def test_custom_table(self):
    ev = BigFramesEvaluator(
        project_id="proj",
        dataset_id="ds",
        table_id="custom_events",
    )
    assert ev.table_id == "custom_events"


def _install_mock_bigframes():
  """Creates mock bigframes modules in sys.modules.

  The parent ``bigframes`` mock must have ``pandas`` and ``bigquery``
  attributes wired to the child mocks, because ``import a.b as c``
  resolves ``c`` via ``getattr(a, 'b')``.
  """
  mock_bpd = MagicMock()
  mock_bbq = MagicMock()
  mock_ai = MagicMock()
  mock_bbq.ai = mock_ai

  mock_bf = MagicMock()
  mock_bf.pandas = mock_bpd
  mock_bf.bigquery = mock_bbq

  modules = {
      "bigframes": mock_bf,
      "bigframes.pandas": mock_bpd,
      "bigframes.bigquery": mock_bbq,
      "bigframes.bigquery.ai": mock_ai,
  }
  return modules, mock_bpd, mock_bbq


class TestBigFramesEvaluatorEvaluate:
  """Tests for BigFramesEvaluator.evaluate_sessions()."""

  def test_evaluate_sessions(self):
    """Test evaluate_sessions calls bigframes correctly."""
    modules, mock_bpd, mock_bbq = _install_mock_bigframes()

    mock_df = MagicMock()
    mock_df.__getitem__ = MagicMock(return_value=MagicMock())
    mock_df.__setitem__ = MagicMock()

    mock_result = MagicMock()
    mock_result.__getitem__ = MagicMock(
        return_value=mock_result,
    )
    mock_result.__setitem__ = MagicMock()

    mock_bpd.read_gbq = MagicMock(return_value=mock_df)
    mock_bbq.ai.generate = MagicMock(
        return_value=mock_result,
    )

    with patch.dict(sys.modules, modules):
      ev = BigFramesEvaluator(
          project_id="proj",
          dataset_id="ds",
      )
      ev.evaluate_sessions(max_sessions=10)

      mock_bpd.read_gbq.assert_called_once()
      mock_bbq.ai.generate.assert_called_once()
      # Verify output_schema was passed
      gen_kwargs = mock_bbq.ai.generate.call_args
      assert "output_schema" in gen_kwargs.kwargs
      schema = gen_kwargs.kwargs["output_schema"]
      assert "score" in schema
      assert "justification" in schema


class TestBigFramesEvaluatorFacets:
  """Tests for BigFramesEvaluator.extract_facets()."""

  def test_extract_facets(self):
    """Test extract_facets calls bigframes correctly."""
    modules, mock_bpd, mock_bbq = _install_mock_bigframes()

    mock_df = MagicMock()
    mock_df.__getitem__ = MagicMock(return_value=MagicMock())
    mock_df.__setitem__ = MagicMock()

    mock_result = MagicMock()
    mock_result.__getitem__ = MagicMock(
        return_value=mock_result,
    )
    mock_result.__setitem__ = MagicMock()

    mock_bpd.read_gbq = MagicMock(return_value=mock_df)
    mock_bbq.ai.generate = MagicMock(
        return_value=mock_result,
    )

    with patch.dict(sys.modules, modules):
      ev = BigFramesEvaluator(
          project_id="proj",
          dataset_id="ds",
      )
      ev.extract_facets(session_ids=["s1", "s2"])

      mock_bpd.read_gbq.assert_called_once()
      mock_bbq.ai.generate.assert_called_once()
      # Verify facet output_schema
      gen_kwargs = mock_bbq.ai.generate.call_args
      assert "output_schema" in gen_kwargs.kwargs
      schema = gen_kwargs.kwargs["output_schema"]
      assert "goal_categories" in schema
      assert "outcome" in schema
      assert "satisfaction" in schema

  def test_extract_facets_no_session_ids(self):
    """Test extract_facets without specific session IDs."""
    modules, mock_bpd, mock_bbq = _install_mock_bigframes()

    mock_df = MagicMock()
    mock_df.__getitem__ = MagicMock(return_value=MagicMock())
    mock_df.__setitem__ = MagicMock()

    mock_result = MagicMock()
    mock_result.__getitem__ = MagicMock(
        return_value=mock_result,
    )
    mock_result.__setitem__ = MagicMock()

    mock_bpd.read_gbq = MagicMock(return_value=mock_df)
    mock_bbq.ai.generate = MagicMock(
        return_value=mock_result,
    )

    with patch.dict(sys.modules, modules):
      ev = BigFramesEvaluator(
          project_id="proj",
          dataset_id="ds",
      )
      ev.extract_facets(max_sessions=25)

      # Verify read_gbq was called with WHERE TRUE
      call_args = mock_bpd.read_gbq.call_args[0][0]
      assert "TRUE" in call_args
