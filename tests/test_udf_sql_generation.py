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

"""Tests for Python UDF SQL generation."""

import numpy as np
import pandas as pd
import pytest

from bigquery_agent_analytics.udf_sql_templates import ALL_UDFS
from bigquery_agent_analytics.udf_sql_templates import generate_all_udfs
from bigquery_agent_analytics.udf_sql_templates import generate_udf
from bigquery_agent_analytics.udf_sql_templates import list_udfs
from bigquery_agent_analytics.udf_sql_templates import UDF_NAMES

PROJECT = "test-project"
DATASET = "analytics"

TOTAL_UDFS = 13  # 3 Tier 1 + 6 Tier 2 + 4 Tier 3


# ------------------------------------------------------------------ #
# Registry                                                             #
# ------------------------------------------------------------------ #


class TestRegistry:

  def test_all_udfs_count(self):
    assert len(ALL_UDFS) == TOTAL_UDFS

  def test_udf_names_count(self):
    assert len(UDF_NAMES) == TOTAL_UDFS

  def test_names_unique(self):
    assert len(set(UDF_NAMES)) == len(UDF_NAMES)

  def test_all_names_prefixed(self):
    for name in UDF_NAMES:
      assert name.startswith("bqaa_"), f"{name} missing bqaa_ prefix"

  def test_tier1_event_semantics(self):
    tier1 = [
        "bqaa_is_error_event",
        "bqaa_tool_outcome",
        "bqaa_extract_response_text",
    ]
    for name in tier1:
      assert name in UDF_NAMES, f"Missing Tier 1 UDF: {name}"

  def test_tier2_score_kernels(self):
    tier2 = [
        "bqaa_score_latency",
        "bqaa_score_error_rate",
        "bqaa_score_turn_count",
        "bqaa_score_token_efficiency",
        "bqaa_score_ttft",
        "bqaa_score_cost",
    ]
    for name in tier2:
      assert name in UDF_NAMES, f"Missing Tier 2 UDF: {name}"

  def test_tier3_vectorized(self):
    tier3 = [
        "bqaa_score_latency_batch",
        "bqaa_score_error_rate_batch",
        "bqaa_score_cost_batch",
        "bqaa_normalize_event_label",
    ]
    for name in tier3:
      assert name in UDF_NAMES, f"Missing Tier 3 UDF: {name}"

  def test_vectorized_flags(self):
    vectorized_names = {
        "bqaa_score_latency_batch",
        "bqaa_score_error_rate_batch",
        "bqaa_score_cost_batch",
        "bqaa_normalize_event_label",
    }
    for spec in ALL_UDFS:
      if spec.name in vectorized_names:
        assert spec.vectorized, f"{spec.name} should be vectorized"
      else:
        assert not spec.vectorized, f"{spec.name} should not be vectorized"


# ------------------------------------------------------------------ #
# generate_udf                                                        #
# ------------------------------------------------------------------ #


class TestGenerateUdf:

  def test_unknown_name_raises(self):
    with pytest.raises(ValueError, match="bogus"):
      generate_udf("bogus", PROJECT, DATASET)

  @pytest.mark.parametrize("name", UDF_NAMES)
  def test_generates_valid_ddl(self, name):
    sql = generate_udf(name, PROJECT, DATASET)
    assert "CREATE OR REPLACE FUNCTION" in sql
    assert f"`{PROJECT}.{DATASET}.{name}`" in sql
    assert "LANGUAGE python" in sql
    assert "RETURNS" in sql

  @pytest.mark.parametrize("name", UDF_NAMES)
  def test_has_entry_point(self, name):
    sql = generate_udf(name, PROJECT, DATASET)
    assert f"entry_point = '{name}'" in sql

  @pytest.mark.parametrize("name", UDF_NAMES)
  def test_has_runtime_version(self, name):
    sql = generate_udf(name, PROJECT, DATASET)
    assert "runtime_version = 'python-3.11'" in sql

  @pytest.mark.parametrize("name", UDF_NAMES)
  def test_has_description(self, name):
    sql = generate_udf(name, PROJECT, DATASET)
    assert "description =" in sql

  @pytest.mark.parametrize("name", UDF_NAMES)
  def test_body_contains_def(self, name):
    sql = generate_udf(name, PROJECT, DATASET)
    assert f"def {name}(" in sql

  def test_latency_return_type(self):
    sql = generate_udf("bqaa_score_latency", PROJECT, DATASET)
    assert "RETURNS FLOAT64" in sql

  def test_is_error_return_type(self):
    sql = generate_udf("bqaa_is_error_event", PROJECT, DATASET)
    assert "RETURNS BOOL" in sql

  def test_tool_outcome_return_type(self):
    sql = generate_udf("bqaa_tool_outcome", PROJECT, DATASET)
    assert "RETURNS STRING" in sql

  def test_extract_response_return_type(self):
    sql = generate_udf("bqaa_extract_response_text", PROJECT, DATASET)
    assert "RETURNS STRING" in sql

  def test_cost_has_five_params(self):
    sql = generate_udf("bqaa_score_cost", PROJECT, DATASET)
    assert "input_tokens INT64" in sql
    assert "output_tokens INT64" in sql
    assert "max_cost_usd FLOAT64" in sql
    assert "input_cost_per_1k FLOAT64" in sql
    assert "output_cost_per_1k FLOAT64" in sql


# ------------------------------------------------------------------ #
# Vectorized DDL structure                                             #
# ------------------------------------------------------------------ #

VECTORIZED_NAMES = [
    "bqaa_score_latency_batch",
    "bqaa_score_error_rate_batch",
    "bqaa_score_cost_batch",
    "bqaa_normalize_event_label",
]
SCALAR_NAMES = [n for n in UDF_NAMES if n not in VECTORIZED_NAMES]


class TestVectorizedDdl:

  @pytest.mark.parametrize("name", VECTORIZED_NAMES)
  def test_has_vectorized_option(self, name):
    sql = generate_udf(name, PROJECT, DATASET)
    assert "vectorized = true" in sql

  @pytest.mark.parametrize("name", SCALAR_NAMES)
  def test_scalar_has_no_vectorized_option(self, name):
    sql = generate_udf(name, PROJECT, DATASET)
    assert "vectorized" not in sql

  def test_latency_batch_return_type(self):
    sql = generate_udf("bqaa_score_latency_batch", PROJECT, DATASET)
    assert "RETURNS FLOAT64" in sql

  def test_error_rate_batch_return_type(self):
    sql = generate_udf("bqaa_score_error_rate_batch", PROJECT, DATASET)
    assert "RETURNS FLOAT64" in sql

  def test_cost_batch_return_type(self):
    sql = generate_udf("bqaa_score_cost_batch", PROJECT, DATASET)
    assert "RETURNS FLOAT64" in sql

  def test_normalize_label_return_type(self):
    sql = generate_udf("bqaa_normalize_event_label", PROJECT, DATASET)
    assert "RETURNS STRING" in sql

  def test_latency_batch_uses_numpy(self):
    sql = generate_udf("bqaa_score_latency_batch", PROJECT, DATASET)
    assert "import numpy" in sql

  def test_normalize_label_uses_map(self):
    sql = generate_udf("bqaa_normalize_event_label", PROJECT, DATASET)
    assert ".map(" in sql


# ------------------------------------------------------------------ #
# generate_all_udfs                                                    #
# ------------------------------------------------------------------ #


class TestGenerateAllUdfs:

  def test_contains_all_names(self):
    sql = generate_all_udfs(PROJECT, DATASET)
    for name in UDF_NAMES:
      assert f".{name}`" in sql, f"Missing UDF: {name}"

  def test_total_create_statements(self):
    sql = generate_all_udfs(PROJECT, DATASET)
    assert sql.count("CREATE OR REPLACE FUNCTION") == TOTAL_UDFS

  def test_custom_separator(self):
    sql = generate_all_udfs(PROJECT, DATASET, separator="\n---\n")
    assert "---" in sql

  def test_different_project_dataset(self):
    sql = generate_all_udfs("my-proj", "my_ds")
    assert "`my-proj.my_ds." in sql
    assert PROJECT not in sql


# ------------------------------------------------------------------ #
# list_udfs                                                            #
# ------------------------------------------------------------------ #


class TestListUdfs:

  def test_returns_list(self):
    result = list_udfs()
    assert isinstance(result, list)
    assert len(result) == TOTAL_UDFS

  def test_dict_keys(self):
    for entry in list_udfs():
      assert "name" in entry
      assert "params" in entry
      assert "return_type" in entry
      assert "description" in entry
      assert "vectorized" in entry

  def test_names_match_registry(self):
    names = [e["name"] for e in list_udfs()]
    assert names == UDF_NAMES

  def test_vectorized_entries(self):
    vectorized = [e for e in list_udfs() if e["vectorized"]]
    assert len(vectorized) == 4


# ------------------------------------------------------------------ #
# Body parity: inline bodies must match udf_kernels logic              #
# ------------------------------------------------------------------ #


class TestBodyParity:
  """Verify that the inlined UDF bodies produce the same results
  as the shared udf_kernels functions by exec'ing the body and
  calling the function."""

  def _exec_udf(self, name):
    """Extract and exec the UDF body, return the callable."""
    import textwrap

    sql = generate_udf(name, PROJECT, DATASET)
    # Extract body between AS r""" and """;
    start = sql.index('AS r"""') + len('AS r"""')
    end = sql.index('""";')
    body = textwrap.dedent(sql[start:end])
    ns = {}
    exec(body, ns)  # noqa: S102
    return ns[name]

  def test_is_error_event_parity(self):
    from bigquery_agent_analytics.udf_kernels import is_error_event

    fn = self._exec_udf("bqaa_is_error_event")
    cases = [
        ("TOOL_ERROR", None, "OK"),
        ("LLM_REQUEST", None, "OK"),
        ("LLM_REQUEST", "oops", "OK"),
        ("LLM_REQUEST", None, "ERROR"),
    ]
    for et, em, st in cases:
      assert fn(et, em, st) == is_error_event(et, em, st)

  def test_tool_outcome_parity(self):
    from bigquery_agent_analytics.udf_kernels import tool_outcome

    fn = self._exec_udf("bqaa_tool_outcome")
    cases = [
        ("TOOL_COMPLETED", "OK"),
        ("TOOL_ERROR", "OK"),
        ("TOOL_STARTING", "OK"),
        ("TOOL_COMPLETED", "ERROR"),
    ]
    for et, st in cases:
      assert fn(et, st) == tool_outcome(et, st)

  def test_extract_response_text_parity(self):
    from bigquery_agent_analytics.udf_kernels import extract_response_text

    fn = self._exec_udf("bqaa_extract_response_text")
    cases = [
        '{"response": "hello"}',
        '{"text_summary": "s"}',
        '{"text": "t"}',
        '{"raw": "r"}',
        "{}",
        None,
        "",
        "not json",
    ]
    for c in cases:
      assert fn(c) == extract_response_text(c), f"Mismatch: {c}"

  def test_score_latency_parity(self):
    from bigquery_agent_analytics.udf_kernels import score_latency

    fn = self._exec_udf("bqaa_score_latency")
    for avg, thresh in [(0, 5000), (2500, 5000), (5000, 5000), (10000, 5000)]:
      assert fn(avg, thresh) == pytest.approx(score_latency(avg, thresh))

  def test_score_error_rate_parity(self):
    from bigquery_agent_analytics.udf_kernels import score_error_rate

    fn = self._exec_udf("bqaa_score_error_rate")
    for c, e, m in [(0, 0, 0.1), (10, 0, 0.1), (10, 1, 0.1), (100, 5, 0.1)]:
      assert fn(c, e, m) == pytest.approx(score_error_rate(c, e, m))

  def test_score_turn_count_parity(self):
    from bigquery_agent_analytics.udf_kernels import score_turn_count

    fn = self._exec_udf("bqaa_score_turn_count")
    for t, m in [(0, 10), (5, 10), (10, 10), (20, 10)]:
      assert fn(t, m) == pytest.approx(score_turn_count(t, m))

  def test_score_token_efficiency_parity(self):
    from bigquery_agent_analytics.udf_kernels import score_token_efficiency

    fn = self._exec_udf("bqaa_score_token_efficiency")
    for t, m in [(0, 50000), (25000, 50000), (50000, 50000)]:
      assert fn(t, m) == pytest.approx(score_token_efficiency(t, m))

  def test_score_ttft_parity(self):
    from bigquery_agent_analytics.udf_kernels import score_ttft

    fn = self._exec_udf("bqaa_score_ttft")
    for a, t in [(0, 1000), (500, 1000), (1000, 1000)]:
      assert fn(a, t) == pytest.approx(score_ttft(a, t))

  def test_score_cost_parity(self):
    from bigquery_agent_analytics.udf_kernels import score_cost

    fn = self._exec_udf("bqaa_score_cost")
    cases = [
        (0, 0, 1.0, 0.00025, 0.00125),
        (10000, 10000, 1.0, 0.00025, 0.00125),
        (1000, 1000, 0.01, 0.001, 0.002),
    ]
    for args in cases:
      assert fn(*args) == pytest.approx(score_cost(*args))


# ------------------------------------------------------------------ #
# Vectorized body parity: vectorized bodies must match scalar kernels  #
# ------------------------------------------------------------------ #


class TestVectorizedBodyParity:
  """Verify that the vectorized UDF bodies produce element-wise
  identical results to the scalar udf_kernels functions."""

  def _exec_vectorized_udf(self, name):
    """Extract and exec the vectorized UDF body, return the callable."""
    import textwrap

    sql = generate_udf(name, PROJECT, DATASET)
    start = sql.index('AS r"""') + len('AS r"""')
    end = sql.index('""";')
    body = textwrap.dedent(sql[start:end])
    ns = {"np": np, "pd": pd}
    exec(body, ns)  # noqa: S102
    return ns[name]

  def test_score_latency_batch_parity(self):
    from bigquery_agent_analytics.udf_kernels import score_latency

    fn = self._exec_vectorized_udf("bqaa_score_latency_batch")
    avgs = pd.Series([0, 2500, 5000, 10000, -1], dtype=float)
    thresholds = pd.Series([5000.0] * 5)
    result = fn(avgs, thresholds)
    for i, (a, t) in enumerate(zip(avgs, thresholds)):
      assert result[i] == pytest.approx(
          score_latency(a, t)
      ), f"Mismatch at index {i}: avg={a}, thresh={t}"

  def test_score_error_rate_batch_parity(self):
    from bigquery_agent_analytics.udf_kernels import score_error_rate

    fn = self._exec_vectorized_udf("bqaa_score_error_rate_batch")
    calls = pd.Series([0, 10, 10, 100, 5], dtype=np.int64)
    errors = pd.Series([0, 0, 1, 5, 5], dtype=np.int64)
    max_rates = pd.Series([0.1] * 5)
    result = fn(calls, errors, max_rates)
    for i in range(len(calls)):
      assert result[i] == pytest.approx(
          score_error_rate(calls[i], errors[i], max_rates[i])
      ), f"Mismatch at index {i}"

  def test_score_cost_batch_parity(self):
    from bigquery_agent_analytics.udf_kernels import score_cost

    fn = self._exec_vectorized_udf("bqaa_score_cost_batch")
    inp = pd.Series([0, 10000, 1000], dtype=np.int64)
    out = pd.Series([0, 10000, 1000], dtype=np.int64)
    max_cost = pd.Series([1.0, 1.0, 0.01])
    icost = pd.Series([0.00025, 0.00025, 0.001])
    ocost = pd.Series([0.00125, 0.00125, 0.002])
    result = fn(inp, out, max_cost, icost, ocost)
    for i in range(len(inp)):
      expected = score_cost(
          inp[i],
          out[i],
          max_cost[i],
          icost[i],
          ocost[i],
      )
      assert result[i] == pytest.approx(expected), f"Mismatch at index {i}"

  def test_normalize_event_label_parity(self):
    from bigquery_agent_analytics.udf_kernels import normalize_event_label

    fn = self._exec_vectorized_udf("bqaa_normalize_event_label")
    events = pd.Series(
        [
            "LLM_REQUEST",
            "LLM_RESPONSE",
            "TOOL_STARTING",
            "TOOL_COMPLETED",
            "TOOL_ERROR",
            "USER_MESSAGE_RECEIVED",
            "AGENT_COMPLETED",
            "UNKNOWN_EVENT",
        ]
    )
    result = fn(events)
    for i, ev in enumerate(events):
      assert result[i] == normalize_event_label(ev), f"Mismatch for {ev}"
