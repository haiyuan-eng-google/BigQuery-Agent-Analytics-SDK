# Copyright 2025 Google LLC
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

"""Tests for the context_graph module."""

from datetime import datetime
from datetime import timezone
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from bigquery_agent_analytics.context_graph import BizNode
from bigquery_agent_analytics.context_graph import ContextGraphConfig
from bigquery_agent_analytics.context_graph import ContextGraphManager
from bigquery_agent_analytics.context_graph import WorldChangeAlert
from bigquery_agent_analytics.context_graph import WorldChangeReport


# ------------------------------------------------------------------ #
# Data Model Tests                                                     #
# ------------------------------------------------------------------ #


class TestBizNode:
  """Tests for BizNode dataclass."""

  def test_creation(self):
    node = BizNode(
        span_id="span-1",
        session_id="sess-1",
        node_type="Product",
        node_value="Yahoo Homepage",
    )
    assert node.node_type == "Product"
    assert node.node_value == "Yahoo Homepage"
    assert node.confidence == 1.0
    assert node.metadata == {}

  def test_with_confidence(self):
    node = BizNode(
        span_id="span-1",
        session_id="sess-1",
        node_type="Targeting",
        node_value="Millennials",
        confidence=0.92,
        metadata={"source": "brief"},
    )
    assert node.confidence == 0.92
    assert node.metadata["source"] == "brief"


class TestWorldChangeAlert:
  """Tests for WorldChangeAlert model."""

  def test_creation(self):
    alert = WorldChangeAlert(
        biz_node="Yahoo Homepage",
        original_state="Product: Yahoo Homepage",
        current_state="unavailable",
        drift_type="inventory_depleted",
        severity=0.9,
    )
    assert alert.biz_node == "Yahoo Homepage"
    assert alert.drift_type == "inventory_depleted"
    assert alert.severity == 0.9
    assert alert.recommendation == "Review before approving."


class TestWorldChangeReport:
  """Tests for WorldChangeReport model."""

  def test_safe_report(self):
    report = WorldChangeReport(
        session_id="sess-1",
        total_entities_checked=5,
        stale_entities=0,
        is_safe_to_approve=True,
    )
    assert report.is_safe_to_approve
    assert report.stale_entities == 0
    assert "Safe to approve  : True" in report.summary()

  def test_unsafe_report(self):
    alert = WorldChangeAlert(
        biz_node="Yahoo Homepage",
        original_state="Product: Yahoo Homepage",
        current_state="sold_out",
        drift_type="inventory_depleted",
        severity=0.95,
    )
    report = WorldChangeReport(
        session_id="sess-1",
        alerts=[alert],
        total_entities_checked=3,
        stale_entities=1,
        is_safe_to_approve=False,
    )
    assert not report.is_safe_to_approve
    assert report.stale_entities == 1
    summary = report.summary()
    assert "inventory_depleted" in summary
    assert "Yahoo Homepage" in summary

  def test_summary_format(self):
    report = WorldChangeReport(
        session_id="sess-42",
        total_entities_checked=10,
        stale_entities=2,
        is_safe_to_approve=False,
        alerts=[
            WorldChangeAlert(
                biz_node="Product A",
                original_state="available",
                current_state="depleted",
                drift_type="unavailable",
                severity=0.8,
            ),
            WorldChangeAlert(
                biz_node="Product B",
                original_state="$50",
                current_state="$75",
                drift_type="price_changed",
                severity=0.6,
            ),
        ],
    )
    summary = report.summary()
    assert "sess-42" in summary
    assert "Entities checked : 10" in summary
    assert "Stale entities   : 2" in summary
    assert "Product A" in summary
    assert "Product B" in summary


class TestContextGraphConfig:
  """Tests for ContextGraphConfig model."""

  def test_defaults(self):
    config = ContextGraphConfig()
    assert config.biz_nodes_table == "extracted_biz_nodes"
    assert config.cross_links_table == "context_cross_links"
    assert config.graph_name == "agent_context_graph"
    assert config.max_hops == 20
    assert "Product" in config.entity_types

  def test_custom_config(self):
    config = ContextGraphConfig(
        graph_name="adcp_graph",
        entity_types=["Ad", "Inventory"],
        max_hops=10,
    )
    assert config.graph_name == "adcp_graph"
    assert config.entity_types == ["Ad", "Inventory"]
    assert config.max_hops == 10


# ------------------------------------------------------------------ #
# ContextGraphManager Tests                                            #
# ------------------------------------------------------------------ #


class TestContextGraphManager:
  """Tests for ContextGraphManager."""

  def _make_manager(self, mock_client=None):
    return ContextGraphManager(
        project_id="test-project",
        dataset_id="test_dataset",
        table_id="agent_events",
        client=mock_client or MagicMock(),
    )

  def test_resolve_endpoint_short_name(self):
    mgr = self._make_manager()
    ep = mgr._resolve_endpoint()
    assert ep == (
        "https://aiplatform.googleapis.com/v1/projects/"
        "test-project/locations/global/publishers/google/"
        "models/gemini-2.5-flash"
    )

  def test_resolve_endpoint_full_url(self):
    mgr = self._make_manager()
    mgr.config = ContextGraphConfig(
        endpoint="https://aiplatform.googleapis.com/v1/projects/p/locations/global/publishers/google/models/gemini-3-flash-preview"
    )
    ep = mgr._resolve_endpoint()
    assert ep.startswith("https://")
    assert "gemini-3-flash-preview" in ep

  def test_get_property_graph_ddl(self):
    mgr = self._make_manager()
    ddl = mgr.get_property_graph_ddl()
    assert "CREATE OR REPLACE PROPERTY GRAPH" in ddl
    assert "test-project" in ddl
    assert "test_dataset" in ddl
    assert "agent_events" in ddl
    assert "TechNode" in ddl
    assert "BizNode" in ddl
    assert "Caused" in ddl
    assert "Evaluated" in ddl
    # P1: composite keys
    assert "KEY (biz_node_id)" in ddl
    assert "KEY (link_id)" in ddl

  def test_get_property_graph_ddl_custom_name(self):
    mgr = self._make_manager()
    ddl = mgr.get_property_graph_ddl(graph_name="my_graph")
    assert "my_graph" in ddl

  def test_get_reasoning_chain_gql(self):
    mgr = self._make_manager()
    gql = mgr.get_reasoning_chain_gql(
        decision_event_type="HITL_CONFIRMATION_REQUEST_COMPLETED",
        biz_entity="Yahoo Homepage",
    )
    assert "GRAPH" in gql
    assert "MATCH" in gql
    assert "Caused" in gql
    assert "Evaluated" in gql
    # P2: biz_entity is parameterized, not interpolated
    assert "@biz_entity" in gql
    assert "Yahoo Homepage" not in gql

  def test_get_reasoning_chain_gql_no_entity(self):
    mgr = self._make_manager()
    gql = mgr.get_reasoning_chain_gql(
        decision_event_type="AGENT_COMPLETED",
    )
    assert "GRAPH" in gql
    assert "@biz_entity" not in gql

  def test_get_causal_chain_gql(self):
    mgr = self._make_manager()
    gql = mgr.get_causal_chain_gql(session_id="sess-1")
    assert "GRAPH" in gql
    assert "MATCH" in gql
    assert "USER_MESSAGE_RECEIVED" in gql

  def test_create_property_graph_success(self):
    mock_client = MagicMock()
    mock_job = MagicMock()
    mock_client.query.return_value = mock_job
    mgr = self._make_manager(mock_client)

    result = mgr.create_property_graph()
    assert result is True
    mock_client.query.assert_called_once()
    mock_job.result.assert_called_once()

  def test_create_property_graph_failure(self):
    mock_client = MagicMock()
    mock_client.query.side_effect = Exception("BigQuery error")
    mgr = self._make_manager(mock_client)

    result = mgr.create_property_graph()
    assert result is False

  def test_store_biz_nodes_empty(self):
    mgr = self._make_manager()
    assert mgr.store_biz_nodes([]) is True

  def test_store_biz_nodes_success(self):
    mock_client = MagicMock()
    mock_job = MagicMock()
    mock_client.query.return_value = mock_job
    mock_client.insert_rows_json.return_value = []
    mgr = self._make_manager(mock_client)

    nodes = [
        BizNode(
            span_id="s1",
            session_id="sess-1",
            node_type="Product",
            node_value="Homepage",
        ),
    ]
    result = mgr.store_biz_nodes(nodes)
    assert result is True
    mock_client.insert_rows_json.assert_called_once()

  def test_store_biz_nodes_insert_error(self):
    mock_client = MagicMock()
    mock_job = MagicMock()
    mock_client.query.return_value = mock_job
    mock_client.insert_rows_json.return_value = [
        {"errors": ["insert failed"]}
    ]
    mgr = self._make_manager(mock_client)

    nodes = [
        BizNode(
            span_id="s1",
            session_id="sess-1",
            node_type="Product",
            node_value="Homepage",
        ),
    ]
    result = mgr.store_biz_nodes(nodes)
    assert result is False

  def test_detect_world_changes_no_drift(self):
    mock_client = MagicMock()
    mock_job = MagicMock()
    mock_job.result.return_value = []
    mock_client.query.return_value = mock_job
    mgr = self._make_manager(mock_client)

    report = mgr.detect_world_changes(session_id="sess-1")
    assert report.is_safe_to_approve
    assert report.stale_entities == 0
    assert len(report.alerts) == 0

  def test_detect_world_changes_with_drift(self):
    mock_client = MagicMock()
    mock_job = MagicMock()
    # Simulate returned biz nodes with evaluated_at timestamps
    mock_job.result.return_value = [
        {
            "span_id": "s1",
            "node_type": "Product",
            "node_value": "Yahoo Homepage",
            "confidence": 0.95,
            "evaluated_at": datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc),
        },
        {
            "span_id": "s2",
            "node_type": "Targeting",
            "node_value": "Millennials",
            "confidence": 0.90,
            "evaluated_at": datetime(2025, 6, 1, 12, 1, tzinfo=timezone.utc),
        },
    ]
    mock_client.query.return_value = mock_job
    mgr = self._make_manager(mock_client)

    def check_state(node):
      # Verify evaluated_at timestamp is passed through
      assert node.evaluated_at is not None
      if node.node_value == "Yahoo Homepage":
        return {
            "available": False,
            "current_value": "sold_out",
            "drift_type": "inventory_depleted",
            "severity": 0.95,
        }
      return {"available": True, "current_value": node.node_value}

    report = mgr.detect_world_changes(
        session_id="sess-1",
        current_state_fn=check_state,
    )
    assert not report.is_safe_to_approve
    assert report.stale_entities == 1
    assert len(report.alerts) == 1
    assert report.alerts[0].biz_node == "Yahoo Homepage"
    assert report.alerts[0].drift_type == "inventory_depleted"

  def test_detect_world_changes_fn_exception(self):
    mock_client = MagicMock()
    mock_job = MagicMock()
    mock_job.result.return_value = [
        {
            "span_id": "s1",
            "node_type": "Product",
            "node_value": "Test",
            "confidence": 1.0,
            "evaluated_at": datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc),
        },
    ]
    mock_client.query.return_value = mock_job
    mgr = self._make_manager(mock_client)

    def bad_fn(node):
      raise RuntimeError("API failure")

    report = mgr.detect_world_changes(
        session_id="sess-1",
        current_state_fn=bad_fn,
    )
    # Should not crash; entity is skipped
    assert report.is_safe_to_approve
    assert report.stale_entities == 0

  def test_create_cross_links_success(self):
    mock_client = MagicMock()
    mock_job = MagicMock()
    mock_client.query.return_value = mock_job
    mgr = self._make_manager(mock_client)

    result = mgr.create_cross_links(["sess-1"])
    assert result is True
    # create table + delete old links + insert new links
    assert mock_client.query.call_count == 3

  def test_create_cross_links_failure(self):
    mock_client = MagicMock()
    mock_client.query.side_effect = Exception("fail")
    mgr = self._make_manager(mock_client)

    result = mgr.create_cross_links(["sess-1"])
    assert result is False

  def test_build_context_graph(self):
    mock_client = MagicMock()
    mock_job = MagicMock()
    mock_job.result.return_value = []
    mock_client.query.return_value = mock_job
    mgr = self._make_manager(mock_client)

    results = mgr.build_context_graph(
        session_ids=["sess-1"],
        use_ai_generate=False,
    )
    assert "biz_nodes_count" in results
    assert "cross_links_created" in results
    assert "property_graph_created" in results

  def test_explain_decision_failure(self):
    mock_client = MagicMock()
    mock_client.query.side_effect = Exception("GQL error")
    mgr = self._make_manager(mock_client)

    result = mgr.explain_decision(
        biz_entity="Yahoo Homepage",
    )
    assert result == []

  def test_traverse_causal_chain_failure(self):
    mock_client = MagicMock()
    mock_client.query.side_effect = Exception("GQL error")
    mgr = self._make_manager(mock_client)

    result = mgr.traverse_causal_chain(session_id="sess-1")
    assert result == []

  def test_extract_query_has_output_schema(self):
    mgr = self._make_manager()
    from bigquery_agent_analytics.context_graph import (
        _BIZ_NODE_OUTPUT_SCHEMA,
        _EXTRACT_BIZ_NODES_QUERY,
    )
    assert "output_schema =>" in _EXTRACT_BIZ_NODES_QUERY
    assert "entity_type" in _BIZ_NODE_OUTPUT_SCHEMA
    assert "entity_value" in _BIZ_NODE_OUTPUT_SCHEMA
    assert "confidence" in _BIZ_NODE_OUTPUT_SCHEMA

  def test_property_graph_ddl_has_artifact_uri(self):
    mgr = self._make_manager()
    ddl = mgr.get_property_graph_ddl()
    assert "artifact_uri" in ddl

  def test_property_graph_ddl_evaluated_has_properties(self):
    mgr = self._make_manager()
    ddl = mgr.get_property_graph_ddl()
    assert "link_type" in ddl
    assert "created_at" in ddl

  def test_reconstruct_trace_gql_success(self):
    mock_client = MagicMock()
    mock_job = MagicMock()
    mock_job.result.return_value = [
        {
            "parent_span_id": "s1",
            "parent_event_type": "USER_MESSAGE_RECEIVED",
            "parent_agent": "root",
            "parent_timestamp": datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc),
            "session_id": "sess-1",
            "parent_invocation_id": "inv-1",
            "parent_content": {},
            "parent_latency_ms": None,
            "parent_status": "OK",
            "parent_error_message": None,
            "child_span_id": "s2",
            "child_event_type": "LLM_REQUEST",
            "child_agent": "root",
            "child_timestamp": datetime(2025, 6, 1, 12, 1, tzinfo=timezone.utc),
            "child_invocation_id": "inv-1",
            "child_content": {},
            "child_latency_ms": 500,
            "child_status": "OK",
            "child_error_message": None,
        },
    ]
    mock_client.query.return_value = mock_job
    mgr = self._make_manager(mock_client)

    rows = mgr.reconstruct_trace_gql(session_id="sess-1")
    assert len(rows) == 1
    assert rows[0]["parent_span_id"] == "s1"
    assert rows[0]["child_span_id"] == "s2"

  def test_reconstruct_trace_gql_failure(self):
    mock_client = MagicMock()
    mock_client.query.side_effect = Exception("GQL error")
    mgr = self._make_manager(mock_client)

    result = mgr.reconstruct_trace_gql(session_id="sess-1")
    assert result == []

  def test_biz_node_has_evaluated_at_and_artifact_uri(self):
    node = BizNode(
        span_id="s1",
        session_id="sess-1",
        node_type="Product",
        node_value="Yahoo Homepage",
        evaluated_at=datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc),
        artifact_uri="gs://bucket/path/file.json",
    )
    assert node.evaluated_at is not None
    assert node.artifact_uri == "gs://bucket/path/file.json"

  def test_detect_world_changes_passes_evaluated_at(self):
    mock_client = MagicMock()
    mock_job = MagicMock()
    eval_time = datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc)
    mock_job.result.return_value = [
        {
            "span_id": "s1",
            "node_type": "Product",
            "node_value": "Test",
            "confidence": 1.0,
            "evaluated_at": eval_time,
        },
    ]
    mock_client.query.return_value = mock_job
    mgr = self._make_manager(mock_client)

    received_timestamps = []

    def check_fn(node):
      received_timestamps.append(node.evaluated_at)
      return {"available": True, "current_value": node.node_value}

    mgr.detect_world_changes(
        session_id="sess-1",
        current_state_fn=check_fn,
    )
    assert len(received_timestamps) == 1
    assert received_timestamps[0] == eval_time

  def test_get_biz_nodes_returns_artifact_uri(self):
    mock_client = MagicMock()
    mock_job = MagicMock()
    mock_job.result.return_value = [
        {
            "biz_node_id": "s1:Product:Yahoo",
            "span_id": "s1",
            "session_id": "sess-1",
            "node_type": "Product",
            "node_value": "Yahoo",
            "confidence": 0.95,
            "artifact_uri": "gs://bucket/output.json",
        },
    ]
    mock_client.query.return_value = mock_job
    mgr = self._make_manager(mock_client)

    nodes = mgr.get_biz_nodes_for_session("sess-1")
    assert len(nodes) == 1
    assert nodes[0].artifact_uri == "gs://bucket/output.json"

  def test_read_biz_nodes_returns_artifact_uri(self):
    mock_client = MagicMock()
    mock_job = MagicMock()
    mock_job.result.return_value = [
        {
            "span_id": "s1",
            "session_id": "sess-1",
            "node_type": "Product",
            "node_value": "Yahoo",
            "confidence": 0.95,
            "artifact_uri": "gs://bucket/file.pdf",
        },
    ]
    mock_client.query.return_value = mock_job
    mgr = self._make_manager(mock_client)

    nodes = mgr._read_biz_nodes(["sess-1"])
    assert len(nodes) == 1
    assert nodes[0].artifact_uri == "gs://bucket/file.pdf"

  def test_cross_link_id_uses_biz_node_id(self):
    from bigquery_agent_analytics.context_graph import (
        _INSERT_CROSS_LINKS_QUERY,
    )
    assert "b.biz_node_id AS link_id" in _INSERT_CROSS_LINKS_QUERY

  def test_merge_deletes_stale_biz_nodes(self):
    from bigquery_agent_analytics.context_graph import (
        _EXTRACT_BIZ_NODES_QUERY,
    )
    assert "WHEN NOT MATCHED BY SOURCE" in _EXTRACT_BIZ_NODES_QUERY
    assert "DELETE" in _EXTRACT_BIZ_NODES_QUERY


# ------------------------------------------------------------------ #
# Client integration test                                              #
# ------------------------------------------------------------------ #


class TestClientContextGraph:
  """Tests for Client.context_graph() factory method."""

  def test_context_graph_returns_manager(self):
    with patch(
        "bigquery_agent_analytics.client.bigquery.Client"
    ):
      from bigquery_agent_analytics.client import Client

      # Patch schema verification
      with patch.object(Client, "_verify_schema"):
        client = Client(
            project_id="p",
            dataset_id="d",
        )
        mgr = client.context_graph()
        assert isinstance(mgr, ContextGraphManager)
        assert mgr.project_id == "p"
        assert mgr.dataset_id == "d"

  def test_context_graph_with_config(self):
    with patch(
        "bigquery_agent_analytics.client.bigquery.Client"
    ):
      from bigquery_agent_analytics.client import Client

      with patch.object(Client, "_verify_schema"):
        client = Client(
            project_id="p",
            dataset_id="d",
        )
        cfg = ContextGraphConfig(graph_name="custom_graph")
        mgr = client.context_graph(config=cfg)
        assert mgr.config.graph_name == "custom_graph"

  def test_get_session_trace_gql_fallback_on_empty(self):
    """GQL with no edges falls back to flat get_session_trace."""
    with patch(
        "bigquery_agent_analytics.client.bigquery.Client"
    ):
      from bigquery_agent_analytics.client import Client
      from bigquery_agent_analytics.trace import Trace

      with patch.object(Client, "_verify_schema"):
        client = Client(
            project_id="p",
            dataset_id="d",
        )
        # GQL returns empty
        with patch.object(
            ContextGraphManager, "reconstruct_trace_gql",
            return_value=[],
        ):
          mock_trace = Trace(
              trace_id="t1", session_id="sess-1", spans=[]
          )
          with patch.object(
              Client, "get_session_trace",
              return_value=mock_trace,
          ) as mock_flat:
            result = client.get_session_trace_gql(
                session_id="sess-1"
            )
            mock_flat.assert_called_once_with("sess-1")
            assert result.session_id == "sess-1"

  def test_get_session_trace_gql_merges_isolated_events(self):
    """GQL edges + flat SQL merge captures isolated events."""
    with patch(
        "bigquery_agent_analytics.client.bigquery.Client"
    ):
      from bigquery_agent_analytics.client import Client
      from bigquery_agent_analytics.trace import Span, Trace

      with patch.object(Client, "_verify_schema"):
        client = Client(
            project_id="p",
            dataset_id="d",
        )
        ts = datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc)
        # GQL returns one edge pair (s1 -> s2)
        gql_rows = [
            {
                "parent_span_id": "s1",
                "parent_event_type": "USER_MESSAGE_RECEIVED",
                "parent_agent": "root",
                "parent_timestamp": ts,
                "session_id": "sess-1",
                "parent_invocation_id": "inv-1",
                "parent_content": {},
                "parent_latency_ms": None,
                "parent_status": "OK",
                "parent_error_message": None,
                "child_span_id": "s2",
                "child_event_type": "LLM_REQUEST",
                "child_agent": "root",
                "child_timestamp": ts,
                "child_invocation_id": "inv-1",
                "child_content": {},
                "child_latency_ms": 500,
                "child_status": "OK",
                "child_error_message": None,
            },
        ]
        # Flat trace has s1, s2, and an isolated s3
        flat_spans = [
            Span(event_type="USER_MESSAGE_RECEIVED",
                 agent="root", timestamp=ts, span_id="s1"),
            Span(event_type="LLM_REQUEST",
                 agent="root", timestamp=ts, span_id="s2"),
            Span(event_type="STATE_DELTA",
                 agent="root", timestamp=ts, span_id="s3"),
        ]
        flat_trace = Trace(
            trace_id="t1", session_id="sess-1", spans=flat_spans,
        )
        with patch.object(
            ContextGraphManager, "reconstruct_trace_gql",
            return_value=gql_rows,
        ):
          with patch.object(
              Client, "get_session_trace",
              return_value=flat_trace,
          ):
            result = client.get_session_trace_gql(
                session_id="sess-1"
            )
            span_ids = {s.span_id for s in result.spans}
            # All three spans present: s1, s2 from GQL + s3 from flat
            assert "s1" in span_ids
            assert "s2" in span_ids
            assert "s3" in span_ids
            assert len(result.spans) == 3
