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

"""Tests for ontology_graph — OntologyGraphManager extraction engine."""

from __future__ import annotations

import json
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from bigquery_agent_analytics.ontology_graph import _build_edge_node_ref
from bigquery_agent_analytics.ontology_graph import _hydrate_graph
from bigquery_agent_analytics.ontology_graph import OntologyGraphManager
from bigquery_agent_analytics.ontology_models import BindingSpec
from bigquery_agent_analytics.ontology_models import EntitySpec
from bigquery_agent_analytics.ontology_models import GraphSpec
from bigquery_agent_analytics.ontology_models import KeySpec
from bigquery_agent_analytics.ontology_models import PropertySpec
from bigquery_agent_analytics.ontology_models import RelationshipSpec

# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #


def _make_entity(name, props=None, keys=None):
  props = props or [PropertySpec(name="eid", type="string")]
  keys = keys or ["eid"]
  return EntitySpec(
      name=name,
      binding=BindingSpec(source="p.d.t"),
      keys=KeySpec(primary=keys),
      properties=props,
      labels=[name],
  )


def _simple_spec():
  """Two entities, one relationship."""
  a = _make_entity(
      "Alpha",
      props=[
          PropertySpec(name="alpha_id", type="string"),
          PropertySpec(name="score", type="double"),
      ],
      keys=["alpha_id"],
  )
  b = _make_entity(
      "Beta",
      props=[
          PropertySpec(name="beta_id", type="string"),
          PropertySpec(name="active", type="bool"),
      ],
      keys=["beta_id"],
  )
  rel = RelationshipSpec(
      name="AlphaToBeta",
      from_entity="Alpha",
      to_entity="Beta",
      binding=BindingSpec(
          source="p.d.edges",
          from_columns=["alpha_id"],
          to_columns=["beta_id"],
      ),
      properties=[PropertySpec(name="weight", type="double")],
  )
  return GraphSpec(
      name="test_graph",
      entities=[a, b],
      relationships=[rel],
  )


def _mock_bq_client():
  return MagicMock()


# ------------------------------------------------------------------ #
# OntologyGraphManager.__init__                                        #
# ------------------------------------------------------------------ #


class TestOntologyGraphManagerInit:

  def test_basic_init(self):
    mgr = OntologyGraphManager(
        project_id="proj",
        dataset_id="ds",
        spec=_simple_spec(),
        bq_client=_mock_bq_client(),
    )
    assert mgr.project_id == "proj"
    assert mgr.dataset_id == "ds"
    assert mgr.table_id == "agent_events"
    assert mgr.endpoint == "gemini-2.5-flash"
    assert mgr.location is None

  def test_custom_table_and_endpoint(self):
    mgr = OntologyGraphManager(
        project_id="proj",
        dataset_id="ds",
        spec=_simple_spec(),
        table_id="custom_events",
        endpoint="gemini-2.5-pro",
        bq_client=_mock_bq_client(),
    )
    assert mgr.table_id == "custom_events"
    assert mgr.endpoint == "gemini-2.5-pro"


# ------------------------------------------------------------------ #
# Lazy BQ Client                                                       #
# ------------------------------------------------------------------ #


class TestLazyBqClient:

  def test_uses_provided_client(self):
    client = _mock_bq_client()
    mgr = OntologyGraphManager(
        project_id="proj",
        dataset_id="ds",
        spec=_simple_spec(),
        bq_client=client,
    )
    assert mgr.bq_client is client

  @patch("bigquery_agent_analytics.ontology_graph.bigquery.Client")
  def test_creates_client_when_none(self, mock_client_cls):
    mock_client_cls.return_value = MagicMock()
    mgr = OntologyGraphManager(
        project_id="proj",
        dataset_id="ds",
        spec=_simple_spec(),
    )
    _ = mgr.bq_client
    mock_client_cls.assert_called_once_with(project="proj")

  @patch("bigquery_agent_analytics.ontology_graph.bigquery.Client")
  def test_creates_client_with_location(self, mock_client_cls):
    mock_client_cls.return_value = MagicMock()
    mgr = OntologyGraphManager(
        project_id="proj",
        dataset_id="ds",
        spec=_simple_spec(),
        location="us-east4",
    )
    _ = mgr.bq_client
    mock_client_cls.assert_called_once_with(project="proj", location="us-east4")

  @patch("bigquery_agent_analytics.ontology_graph.bigquery.Client")
  def test_lazy_client_cached(self, mock_client_cls):
    mock_client_cls.return_value = MagicMock()
    mgr = OntologyGraphManager(
        project_id="proj",
        dataset_id="ds",
        spec=_simple_spec(),
    )
    c1 = mgr.bq_client
    c2 = mgr.bq_client
    assert c1 is c2
    mock_client_cls.assert_called_once()


# ------------------------------------------------------------------ #
# Endpoint Resolution                                                  #
# ------------------------------------------------------------------ #


class TestResolveEndpoint:

  def test_short_name_becomes_vertex_url(self):
    mgr = OntologyGraphManager(
        project_id="proj",
        dataset_id="ds",
        spec=_simple_spec(),
        endpoint="gemini-2.5-flash",
        bq_client=_mock_bq_client(),
    )
    url = mgr._resolve_endpoint()
    assert url.startswith("https://aiplatform.googleapis.com/v1/projects/")
    assert "proj" in url
    assert "gemini-2.5-flash" in url

  def test_full_url_passthrough(self):
    custom_url = "https://custom.endpoint.com/v1/model"
    mgr = OntologyGraphManager(
        project_id="proj",
        dataset_id="ds",
        spec=_simple_spec(),
        endpoint=custom_url,
        bq_client=_mock_bq_client(),
    )
    assert mgr._resolve_endpoint() == custom_url

  def test_legacy_bq_ml_ref_raises(self):
    mgr = OntologyGraphManager(
        project_id="proj",
        dataset_id="ds",
        spec=_simple_spec(),
        endpoint="project.dataset.model",
        bq_client=_mock_bq_client(),
    )
    with pytest.raises(ValueError, match="Legacy BQ ML model reference"):
      mgr._resolve_endpoint()


# ------------------------------------------------------------------ #
# get_extraction_sql                                                   #
# ------------------------------------------------------------------ #


class TestGetExtractionSql:

  def test_returns_sql_string(self):
    mgr = OntologyGraphManager(
        project_id="proj",
        dataset_id="ds",
        spec=_simple_spec(),
        bq_client=_mock_bq_client(),
    )
    sql = mgr.get_extraction_sql()
    assert "AI.GENERATE" in sql
    assert "proj.ds.agent_events" in sql
    assert "@session_ids" in sql

  def test_sql_contains_endpoint(self):
    mgr = OntologyGraphManager(
        project_id="proj",
        dataset_id="ds",
        spec=_simple_spec(),
        bq_client=_mock_bq_client(),
    )
    sql = mgr.get_extraction_sql()
    assert "aiplatform.googleapis.com" in sql
    assert "gemini-2.5-flash" in sql

  def test_sql_contains_output_schema(self):
    mgr = OntologyGraphManager(
        project_id="proj",
        dataset_id="ds",
        spec=_simple_spec(),
        bq_client=_mock_bq_client(),
    )
    sql = mgr.get_extraction_sql()
    assert "output_schema" in sql

  def test_sql_uses_custom_table(self):
    mgr = OntologyGraphManager(
        project_id="proj",
        dataset_id="ds",
        spec=_simple_spec(),
        table_id="my_events",
        bq_client=_mock_bq_client(),
    )
    sql = mgr.get_extraction_sql()
    assert "proj.ds.my_events" in sql

  def test_sql_escapes_single_quotes(self):
    mgr = OntologyGraphManager(
        project_id="proj",
        dataset_id="ds",
        spec=_simple_spec(),
        bq_client=_mock_bq_client(),
    )
    sql = mgr.get_extraction_sql()
    # The prompt and schema are wrapped in single-quoted SQL strings,
    # so any internal single quotes must be escaped.
    assert "endpoint =>" in sql


# ------------------------------------------------------------------ #
# _hydrate_graph                                                       #
# ------------------------------------------------------------------ #


class TestHydrateGraph:

  def test_empty_rows(self):
    graph = _hydrate_graph(_simple_spec(), [])
    assert graph.name == "test_graph"
    assert len(graph.nodes) == 0
    assert len(graph.edges) == 0

  def test_single_node(self):
    rows = [
        {
            "span_id": "s1",
            "session_id": "sess1",
            "graph_json": json.dumps(
                {
                    "nodes": [
                        {
                            "entity_name": "Alpha",
                            "alpha_id": "a1",
                            "score": 0.9,
                        }
                    ],
                    "edges": [],
                }
            ),
        }
    ]
    graph = _hydrate_graph(_simple_spec(), rows)
    assert len(graph.nodes) == 1
    node = graph.nodes[0]
    assert node.node_id == "sess1:s1:Alpha:0"
    assert node.entity_name == "Alpha"
    assert node.labels == ["Alpha"]
    assert len(node.properties) == 2  # alpha_id, score

  def test_node_properties_extracted(self):
    rows = [
        {
            "span_id": "s1",
            "session_id": "sess1",
            "graph_json": json.dumps(
                {
                    "nodes": [
                        {
                            "entity_name": "Alpha",
                            "alpha_id": "a1",
                            "score": 0.9,
                        }
                    ],
                    "edges": [],
                }
            ),
        }
    ]
    graph = _hydrate_graph(_simple_spec(), rows)
    prop_map = {p.name: p.value for p in graph.nodes[0].properties}
    assert prop_map["alpha_id"] == "a1"
    assert prop_map["score"] == 0.9

  def test_entity_name_excluded_from_properties(self):
    rows = [
        {
            "span_id": "s1",
            "session_id": "sess1",
            "graph_json": json.dumps(
                {
                    "nodes": [{"entity_name": "Alpha", "alpha_id": "a1"}],
                    "edges": [],
                }
            ),
        }
    ]
    graph = _hydrate_graph(_simple_spec(), rows)
    prop_names = [p.name for p in graph.nodes[0].properties]
    assert "entity_name" not in prop_names

  def test_multiple_nodes_across_rows(self):
    rows = [
        {
            "span_id": "s1",
            "session_id": "sess1",
            "graph_json": json.dumps(
                {
                    "nodes": [{"entity_name": "Alpha", "alpha_id": "a1"}],
                    "edges": [],
                }
            ),
        },
        {
            "span_id": "s2",
            "session_id": "sess1",
            "graph_json": json.dumps(
                {
                    "nodes": [{"entity_name": "Beta", "beta_id": "b1"}],
                    "edges": [],
                }
            ),
        },
    ]
    graph = _hydrate_graph(_simple_spec(), rows)
    assert len(graph.nodes) == 2
    assert graph.nodes[0].node_id == "sess1:s1:Alpha:0"
    assert graph.nodes[1].node_id == "sess1:s2:Beta:0"

  def test_multiple_nodes_in_single_row(self):
    rows = [
        {
            "span_id": "s1",
            "session_id": "sess1",
            "graph_json": json.dumps(
                {
                    "nodes": [
                        {"entity_name": "Alpha", "alpha_id": "a1"},
                        {"entity_name": "Alpha", "alpha_id": "a2"},
                    ],
                    "edges": [],
                }
            ),
        }
    ]
    graph = _hydrate_graph(_simple_spec(), rows)
    assert len(graph.nodes) == 2
    assert graph.nodes[0].node_id == "sess1:s1:Alpha:0"
    assert graph.nodes[1].node_id == "sess1:s1:Alpha:1"

  def test_edge_hydration(self):
    rows = [
        {
            "span_id": "s1",
            "session_id": "sess1",
            "graph_json": json.dumps(
                {
                    "nodes": [],
                    "edges": [
                        {
                            "relationship_name": "AlphaToBeta",
                            "from_entity_name": "Alpha",
                            "to_entity_name": "Beta",
                            "from_keys": {"alpha_id": "a1"},
                            "to_keys": {"beta_id": "b1"},
                            "weight": 0.5,
                        }
                    ],
                }
            ),
        }
    ]
    graph = _hydrate_graph(_simple_spec(), rows)
    assert len(graph.edges) == 1
    edge = graph.edges[0]
    assert edge.edge_id == "sess1:s1:AlphaToBeta:0"
    assert edge.relationship_name == "AlphaToBeta"

  def test_edge_from_to_node_refs(self):
    rows = [
        {
            "span_id": "s1",
            "session_id": "sess1",
            "graph_json": json.dumps(
                {
                    "nodes": [],
                    "edges": [
                        {
                            "relationship_name": "AlphaToBeta",
                            "from_entity_name": "Alpha",
                            "to_entity_name": "Beta",
                            "from_keys": {"alpha_id": "a1"},
                            "to_keys": {"beta_id": "b1"},
                        }
                    ],
                }
            ),
        }
    ]
    graph = _hydrate_graph(_simple_spec(), rows)
    edge = graph.edges[0]
    assert edge.from_node_id == "sess1:s1:Alpha:alpha_id=a1"
    assert edge.to_node_id == "sess1:s1:Beta:beta_id=b1"

  def test_edge_properties_extracted(self):
    rows = [
        {
            "span_id": "s1",
            "session_id": "sess1",
            "graph_json": json.dumps(
                {
                    "nodes": [],
                    "edges": [
                        {
                            "relationship_name": "AlphaToBeta",
                            "from_entity_name": "Alpha",
                            "to_entity_name": "Beta",
                            "from_keys": {"alpha_id": "a1"},
                            "to_keys": {"beta_id": "b1"},
                            "weight": 0.75,
                        }
                    ],
                }
            ),
        }
    ]
    graph = _hydrate_graph(_simple_spec(), rows)
    prop_map = {p.name: p.value for p in graph.edges[0].properties}
    assert prop_map["weight"] == 0.75

  def test_edge_structural_fields_excluded_from_properties(self):
    rows = [
        {
            "span_id": "s1",
            "session_id": "sess1",
            "graph_json": json.dumps(
                {
                    "nodes": [],
                    "edges": [
                        {
                            "relationship_name": "AlphaToBeta",
                            "from_entity_name": "Alpha",
                            "to_entity_name": "Beta",
                            "from_keys": {},
                            "to_keys": {},
                        }
                    ],
                }
            ),
        }
    ]
    graph = _hydrate_graph(_simple_spec(), rows)
    prop_names = [p.name for p in graph.edges[0].properties]
    for skip in [
        "relationship_name",
        "from_entity_name",
        "to_entity_name",
        "from_keys",
        "to_keys",
    ]:
      assert skip not in prop_names

  def test_empty_graph_json_skipped(self):
    rows = [
        {"span_id": "s1", "session_id": "sess1", "graph_json": ""},
        {"span_id": "s2", "session_id": "sess1", "graph_json": None},
    ]
    graph = _hydrate_graph(_simple_spec(), rows)
    assert len(graph.nodes) == 0
    assert len(graph.edges) == 0

  def test_invalid_json_skipped(self):
    rows = [
        {
            "span_id": "s1",
            "session_id": "sess1",
            "graph_json": "not valid json{",
        }
    ]
    graph = _hydrate_graph(_simple_spec(), rows)
    assert len(graph.nodes) == 0

  def test_non_dict_json_skipped(self):
    rows = [
        {
            "span_id": "s1",
            "session_id": "sess1",
            "graph_json": json.dumps([1, 2, 3]),
        }
    ]
    graph = _hydrate_graph(_simple_spec(), rows)
    assert len(graph.nodes) == 0

  def test_unknown_entity_uses_name_as_label(self):
    rows = [
        {
            "span_id": "s1",
            "session_id": "sess1",
            "graph_json": json.dumps(
                {
                    "nodes": [{"entity_name": "Unknown", "x": 1}],
                    "edges": [],
                }
            ),
        }
    ]
    graph = _hydrate_graph(_simple_spec(), rows)
    assert graph.nodes[0].labels == ["Unknown"]

  def test_labels_from_spec_entity(self):
    """Labels are resolved from the spec's entity definition."""
    spec = _simple_spec()
    # Manually set multi-labels to simulate extends.
    spec.entities[0].labels = ["Alpha", "BaseEntity"]
    rows = [
        {
            "span_id": "s1",
            "session_id": "sess1",
            "graph_json": json.dumps(
                {
                    "nodes": [{"entity_name": "Alpha", "alpha_id": "a1"}],
                    "edges": [],
                }
            ),
        }
    ]
    graph = _hydrate_graph(spec, rows)
    assert graph.nodes[0].labels == ["Alpha", "BaseEntity"]


# ------------------------------------------------------------------ #
# _build_edge_node_ref                                                 #
# ------------------------------------------------------------------ #


class TestBuildEdgeNodeRef:

  def test_single_key(self):
    raw_edge = {
        "from_entity_name": "Alpha",
        "from_keys": {"alpha_id": "a1"},
    }
    ref = _build_edge_node_ref(raw_edge, "from", "sess1", "s1")
    assert ref == "sess1:s1:Alpha:alpha_id=a1"

  def test_composite_keys_sorted(self):
    raw_edge = {
        "from_entity_name": "Multi",
        "from_keys": {"k2": 42, "k1": "abc"},
    }
    ref = _build_edge_node_ref(raw_edge, "from", "sess1", "s1")
    assert ref == "sess1:s1:Multi:k1=abc,k2=42"

  def test_empty_keys_object(self):
    raw_edge = {
        "to_entity_name": "Beta",
        "to_keys": {},
    }
    ref = _build_edge_node_ref(raw_edge, "to", "sess1", "s1")
    assert ref == "sess1:s1:Beta:unknown"

  def test_missing_keys_field(self):
    raw_edge = {
        "to_entity_name": "Beta",
    }
    ref = _build_edge_node_ref(raw_edge, "to", "sess1", "s1")
    assert ref == "sess1:s1:Beta:unknown"

  def test_to_direction(self):
    raw_edge = {
        "to_entity_name": "Beta",
        "to_keys": {"beta_id": "b1"},
    }
    ref = _build_edge_node_ref(raw_edge, "to", "sess1", "s1")
    assert ref == "sess1:s1:Beta:beta_id=b1"


# ------------------------------------------------------------------ #
# extract_graph (mocked BQ)                                           #
# ------------------------------------------------------------------ #


class TestExtractGraph:

  def _make_manager(self, mock_client=None):
    return OntologyGraphManager(
        project_id="proj",
        dataset_id="ds",
        spec=_simple_spec(),
        bq_client=mock_client or _mock_bq_client(),
    )

  def test_ai_generate_path(self):
    mock_client = _mock_bq_client()
    mock_job = MagicMock()
    mock_job.result.return_value = [
        {
            "span_id": "s1",
            "session_id": "sess1",
            "graph_json": json.dumps(
                {
                    "nodes": [{"entity_name": "Alpha", "alpha_id": "a1"}],
                    "edges": [],
                }
            ),
        }
    ]
    mock_client.query.return_value = mock_job

    mgr = self._make_manager(mock_client)
    graph = mgr.extract_graph(session_ids=["sess1"])

    assert len(graph.nodes) == 1
    assert graph.nodes[0].entity_name == "Alpha"
    mock_client.query.assert_called_once()

  def test_ai_generate_failure_returns_empty_graph(self):
    mock_client = _mock_bq_client()
    mock_client.query.side_effect = Exception("BQ error")

    mgr = self._make_manager(mock_client)
    graph = mgr.extract_graph(session_ids=["sess1"])

    assert graph.name == "test_graph"
    assert len(graph.nodes) == 0
    assert len(graph.edges) == 0

  def test_payload_fallback_path(self):
    mock_client = _mock_bq_client()
    mock_job = MagicMock()
    # Simulate BQ Row objects with .get() interface.
    mock_row = MagicMock()
    mock_row.get.side_effect = lambda k, default="": {
        "session_id": "sess1",
        "span_id": "s1",
        "payload_text": "some text",
    }.get(k, default)
    mock_job.result.return_value = [mock_row]
    mock_client.query.return_value = mock_job

    mgr = self._make_manager(mock_client)
    graph = mgr.extract_graph(session_ids=["sess1"], use_ai_generate=False)

    assert len(graph.nodes) == 1
    assert graph.nodes[0].entity_name == "raw_payload"

  def test_payload_failure_returns_empty_graph(self):
    mock_client = _mock_bq_client()
    mock_client.query.side_effect = Exception("BQ error")

    mgr = self._make_manager(mock_client)
    graph = mgr.extract_graph(session_ids=["sess1"], use_ai_generate=False)

    assert graph.name == "test_graph"
    assert len(graph.nodes) == 0

  def test_session_ids_passed_as_query_parameter(self):
    mock_client = _mock_bq_client()
    mock_job = MagicMock()
    mock_job.result.return_value = []
    mock_client.query.return_value = mock_job

    mgr = self._make_manager(mock_client)
    mgr.extract_graph(session_ids=["sess1", "sess2"])

    call_args = mock_client.query.call_args
    job_config = call_args[1].get("job_config") or call_args[0][1]
    params = job_config.query_parameters
    assert len(params) == 1
    assert params[0].name == "session_ids"
    assert params[0].values == ["sess1", "sess2"]
