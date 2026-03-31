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

"""Tests for ontology_materializer — table DDL + routing + persistence."""

from __future__ import annotations

import os
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from bigquery_agent_analytics.ontology_materializer import _ddl_type
from bigquery_agent_analytics.ontology_materializer import _parse_key_segment
from bigquery_agent_analytics.ontology_materializer import _route_edge
from bigquery_agent_analytics.ontology_materializer import _route_node
from bigquery_agent_analytics.ontology_materializer import compile_entity_ddl
from bigquery_agent_analytics.ontology_materializer import compile_relationship_ddl
from bigquery_agent_analytics.ontology_materializer import OntologyMaterializer
from bigquery_agent_analytics.ontology_models import BindingSpec
from bigquery_agent_analytics.ontology_models import EntitySpec
from bigquery_agent_analytics.ontology_models import ExtractedEdge
from bigquery_agent_analytics.ontology_models import ExtractedGraph
from bigquery_agent_analytics.ontology_models import ExtractedNode
from bigquery_agent_analytics.ontology_models import ExtractedProperty
from bigquery_agent_analytics.ontology_models import GraphSpec
from bigquery_agent_analytics.ontology_models import KeySpec
from bigquery_agent_analytics.ontology_models import load_graph_spec
from bigquery_agent_analytics.ontology_models import PropertySpec
from bigquery_agent_analytics.ontology_models import RelationshipSpec

# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #


def _make_entity(name, props=None, keys=None, source="p.d.t"):
  props = props or [PropertySpec(name="eid", type="string")]
  keys = keys or ["eid"]
  return EntitySpec(
      name=name,
      binding=BindingSpec(source=source),
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
      source="p.d.alpha_table",
  )
  b = _make_entity(
      "Beta",
      props=[
          PropertySpec(name="beta_id", type="string"),
          PropertySpec(name="active", type="bool"),
      ],
      keys=["beta_id"],
      source="p.d.beta_table",
  )
  rel = RelationshipSpec(
      name="AlphaToBeta",
      from_entity="Alpha",
      to_entity="Beta",
      binding=BindingSpec(
          source="p.d.alpha_beta_edges",
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
# Type Mapping                                                         #
# ------------------------------------------------------------------ #


class TestDdlTypeMapping:

  def test_string(self):
    assert _ddl_type("string") == "STRING"

  def test_int64(self):
    assert _ddl_type("int64") == "INT64"

  def test_double(self):
    assert _ddl_type("double") == "FLOAT64"

  def test_bool(self):
    assert _ddl_type("bool") == "BOOL"

  def test_timestamp(self):
    assert _ddl_type("timestamp") == "TIMESTAMP"

  def test_date(self):
    assert _ddl_type("date") == "DATE"

  def test_case_insensitive(self):
    assert _ddl_type("String") == "STRING"
    assert _ddl_type("INT64") == "INT64"

  def test_unknown_type_raises(self):
    with pytest.raises(ValueError, match="Unsupported property type"):
      _ddl_type("array<string>")


# ------------------------------------------------------------------ #
# DDL Generation                                                       #
# ------------------------------------------------------------------ #


class TestCompileEntityDdl:

  def test_creates_table(self):
    entity = _make_entity(
        "Alpha",
        props=[
            PropertySpec(name="alpha_id", type="string"),
            PropertySpec(name="score", type="double"),
        ],
        source="p.d.alpha_table",
    )
    ddl = compile_entity_ddl(entity, "proj", "ds")
    assert "CREATE TABLE IF NOT EXISTS" in ddl
    assert "`p.d.alpha_table`" in ddl

  def test_column_types(self):
    entity = _make_entity(
        "Alpha",
        props=[
            PropertySpec(name="alpha_id", type="string"),
            PropertySpec(name="score", type="double"),
            PropertySpec(name="count", type="int64"),
            PropertySpec(name="active", type="bool"),
        ],
        source="p.d.t",
    )
    ddl = compile_entity_ddl(entity, "proj", "ds")
    assert "alpha_id STRING" in ddl
    assert "score FLOAT64" in ddl
    assert "count INT64" in ddl
    assert "active BOOL" in ddl

  def test_metadata_columns(self):
    entity = _make_entity("A", source="p.d.t")
    ddl = compile_entity_ddl(entity, "proj", "ds")
    assert "session_id STRING" in ddl
    assert "extracted_at TIMESTAMP" in ddl

  def test_short_source_prefixed(self):
    """If binding.source has <2 dots, prefix with project.dataset."""
    entity = _make_entity("A", source="my_table")
    ddl = compile_entity_ddl(entity, "proj", "ds")
    assert "`proj.ds.my_table`" in ddl

  def test_fully_qualified_source_used_as_is(self):
    entity = _make_entity("A", source="other_proj.other_ds.t")
    ddl = compile_entity_ddl(entity, "proj", "ds")
    assert "`other_proj.other_ds.t`" in ddl


class TestCompileRelationshipDdl:

  def test_creates_table(self):
    ddl = compile_relationship_ddl(
        _simple_spec().relationships[0], _simple_spec(), "proj", "ds"
    )
    assert "CREATE TABLE IF NOT EXISTS" in ddl
    assert "`p.d.alpha_beta_edges`" in ddl

  def test_foreign_key_columns(self):
    ddl = compile_relationship_ddl(
        _simple_spec().relationships[0], _simple_spec(), "proj", "ds"
    )
    assert "alpha_id STRING" in ddl
    assert "beta_id STRING" in ddl

  def test_relationship_properties(self):
    ddl = compile_relationship_ddl(
        _simple_spec().relationships[0], _simple_spec(), "proj", "ds"
    )
    assert "weight FLOAT64" in ddl

  def test_metadata_columns(self):
    ddl = compile_relationship_ddl(
        _simple_spec().relationships[0], _simple_spec(), "proj", "ds"
    )
    assert "session_id STRING" in ddl
    assert "extracted_at TIMESTAMP" in ddl

  def test_composite_keys(self):
    entity = _make_entity(
        "Multi",
        props=[
            PropertySpec(name="k1", type="string"),
            PropertySpec(name="k2", type="int64"),
        ],
        keys=["k1", "k2"],
        source="p.d.multi",
    )
    other = _make_entity("Other", source="p.d.other")
    rel = RelationshipSpec(
        name="R",
        from_entity="Multi",
        to_entity="Other",
        binding=BindingSpec(
            source="p.d.edges",
            from_columns=["k1", "k2"],
            to_columns=["eid"],
        ),
    )
    spec = GraphSpec(name="g", entities=[entity, other], relationships=[rel])
    ddl = compile_relationship_ddl(rel, spec, "proj", "ds")
    assert "k1 STRING" in ddl
    assert "k2 INT64" in ddl
    assert "eid STRING" in ddl

  def test_demo_yaml_ddl(self):
    """The real YMGO spec produces valid DDL."""
    demo_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "examples",
        "ymgo_graph_spec.yaml",
    )
    spec = load_graph_spec(demo_path, env="p.d")
    for entity in spec.entities:
      ddl = compile_entity_ddl(entity, "proj", "ds")
      assert "CREATE TABLE IF NOT EXISTS" in ddl
    for rel in spec.relationships:
      ddl = compile_relationship_ddl(rel, spec, "proj", "ds")
      assert "CREATE TABLE IF NOT EXISTS" in ddl


# ------------------------------------------------------------------ #
# _parse_key_segment                                                   #
# ------------------------------------------------------------------ #


class TestParseKeySegment:

  def test_single_key(self):
    result = _parse_key_segment("sess1:Alpha:alpha_id=a1")
    assert result == {"alpha_id": "a1"}

  def test_composite_keys(self):
    result = _parse_key_segment("sess1:Multi:k1=abc,k2=42")
    assert result == {"k1": "abc", "k2": "42"}

  def test_index_fallback_returns_empty(self):
    result = _parse_key_segment("sess1:Alpha:0")
    assert result == {}

  def test_unknown_fallback_returns_empty(self):
    result = _parse_key_segment("sess1:Alpha:unknown")
    assert result == {}

  def test_short_id_returns_empty(self):
    result = _parse_key_segment("sess1")
    assert result == {}

  def test_value_with_equals(self):
    result = _parse_key_segment("sess1:E:key=a=b")
    assert result == {"key": "a=b"}


# ------------------------------------------------------------------ #
# _route_node                                                          #
# ------------------------------------------------------------------ #


class TestRouteNode:

  def test_basic_routing(self):
    entity = _simple_spec().entities[0]  # Alpha
    node = ExtractedNode(
        node_id="sess1:Alpha:alpha_id=a1",
        entity_name="Alpha",
        labels=["Alpha"],
        properties=[
            ExtractedProperty(name="alpha_id", value="a1"),
            ExtractedProperty(name="score", value=0.9),
        ],
    )
    row = _route_node(node, entity, "sess1")
    assert row["alpha_id"] == "a1"
    assert row["score"] == 0.9
    assert row["session_id"] == "sess1"


# ------------------------------------------------------------------ #
# _route_edge                                                          #
# ------------------------------------------------------------------ #


class TestRouteEdge:

  def test_basic_routing(self):
    spec = _simple_spec()
    rel = spec.relationships[0]  # AlphaToBeta
    edge = ExtractedEdge(
        edge_id="sess1:AlphaToBeta:0",
        relationship_name="AlphaToBeta",
        from_node_id="sess1:Alpha:alpha_id=a1",
        to_node_id="sess1:Beta:beta_id=b1",
        properties=[
            ExtractedProperty(name="weight", value=0.75),
        ],
    )
    row = _route_edge(edge, rel, spec, "sess1")
    assert row["alpha_id"] == "a1"
    assert row["beta_id"] == "b1"
    assert row["weight"] == 0.75
    assert row["session_id"] == "sess1"

  def test_composite_key_routing(self):
    entity = _make_entity(
        "Multi",
        props=[
            PropertySpec(name="k1", type="string"),
            PropertySpec(name="k2", type="int64"),
        ],
        keys=["k1", "k2"],
        source="p.d.multi",
    )
    other = _make_entity("Other", source="p.d.other")
    rel = RelationshipSpec(
        name="R",
        from_entity="Multi",
        to_entity="Other",
        binding=BindingSpec(
            source="p.d.edges",
            from_columns=["k1", "k2"],
            to_columns=["eid"],
        ),
    )
    spec = GraphSpec(name="g", entities=[entity, other], relationships=[rel])
    edge = ExtractedEdge(
        edge_id="sess1:R:0",
        relationship_name="R",
        from_node_id="sess1:Multi:k1=abc,k2=42",
        to_node_id="sess1:Other:eid=o1",
        properties=[],
    )
    row = _route_edge(edge, rel, spec, "sess1")
    assert row["k1"] == "abc"
    assert row["k2"] == "42"
    assert row["eid"] == "o1"


# ------------------------------------------------------------------ #
# OntologyMaterializer                                                 #
# ------------------------------------------------------------------ #


class TestOntologyMaterializerInit:

  def test_basic_init(self):
    mat = OntologyMaterializer(
        project_id="proj",
        dataset_id="ds",
        spec=_simple_spec(),
        bq_client=_mock_bq_client(),
    )
    assert mat.project_id == "proj"
    assert mat.dataset_id == "ds"

  @patch("bigquery_agent_analytics.ontology_materializer.bigquery.Client")
  def test_lazy_client(self, mock_client_cls):
    mock_client_cls.return_value = MagicMock()
    mat = OntologyMaterializer(
        project_id="proj",
        dataset_id="ds",
        spec=_simple_spec(),
    )
    _ = mat.bq_client
    mock_client_cls.assert_called_once_with(project="proj")


class TestGetDdl:

  def test_get_entity_ddl(self):
    mat = OntologyMaterializer(
        project_id="proj",
        dataset_id="ds",
        spec=_simple_spec(),
        bq_client=_mock_bq_client(),
    )
    ddl = mat.get_entity_ddl("Alpha")
    assert "CREATE TABLE IF NOT EXISTS" in ddl
    assert "alpha_id STRING" in ddl

  def test_get_entity_ddl_unknown_raises(self):
    mat = OntologyMaterializer(
        project_id="proj",
        dataset_id="ds",
        spec=_simple_spec(),
        bq_client=_mock_bq_client(),
    )
    with pytest.raises(ValueError, match="Entity.*NotHere.*not found"):
      mat.get_entity_ddl("NotHere")

  def test_get_relationship_ddl(self):
    mat = OntologyMaterializer(
        project_id="proj",
        dataset_id="ds",
        spec=_simple_spec(),
        bq_client=_mock_bq_client(),
    )
    ddl = mat.get_relationship_ddl("AlphaToBeta")
    assert "CREATE TABLE IF NOT EXISTS" in ddl
    assert "weight FLOAT64" in ddl

  def test_get_relationship_ddl_unknown_raises(self):
    mat = OntologyMaterializer(
        project_id="proj",
        dataset_id="ds",
        spec=_simple_spec(),
        bq_client=_mock_bq_client(),
    )
    with pytest.raises(ValueError, match="Relationship.*NotHere.*not found"):
      mat.get_relationship_ddl("NotHere")

  def test_get_all_ddl(self):
    mat = OntologyMaterializer(
        project_id="proj",
        dataset_id="ds",
        spec=_simple_spec(),
        bq_client=_mock_bq_client(),
    )
    all_ddl = mat.get_all_ddl()
    assert "Alpha" in all_ddl
    assert "Beta" in all_ddl
    assert "AlphaToBeta" in all_ddl
    assert len(all_ddl) == 3


class TestCreateTables:

  def test_creates_all_tables(self):
    mock_client = _mock_bq_client()
    mock_job = MagicMock()
    mock_job.result.return_value = None
    mock_client.query.return_value = mock_job

    mat = OntologyMaterializer(
        project_id="proj",
        dataset_id="ds",
        spec=_simple_spec(),
        bq_client=mock_client,
    )
    result = mat.create_tables()
    assert "Alpha" in result
    assert "Beta" in result
    assert "AlphaToBeta" in result
    # 3 DDL queries: 2 entities + 1 relationship.
    assert mock_client.query.call_count == 3

  def test_partial_failure_continues(self):
    mock_client = _mock_bq_client()
    call_count = 0

    def side_effect(sql):
      nonlocal call_count
      call_count += 1
      if call_count == 1:
        raise Exception("Table creation failed")
      mock_job = MagicMock()
      mock_job.result.return_value = None
      return mock_job

    mock_client.query.side_effect = side_effect

    mat = OntologyMaterializer(
        project_id="proj",
        dataset_id="ds",
        spec=_simple_spec(),
        bq_client=mock_client,
    )
    result = mat.create_tables()
    # First entity fails, but Beta + AlphaToBeta succeed.
    assert len(result) == 2


class TestMaterialize:

  def _make_graph(self):
    return ExtractedGraph(
        name="test_graph",
        nodes=[
            ExtractedNode(
                node_id="sess1:Alpha:alpha_id=a1",
                entity_name="Alpha",
                labels=["Alpha"],
                properties=[
                    ExtractedProperty(name="alpha_id", value="a1"),
                    ExtractedProperty(name="score", value=0.9),
                ],
            ),
            ExtractedNode(
                node_id="sess1:Beta:beta_id=b1",
                entity_name="Beta",
                labels=["Beta"],
                properties=[
                    ExtractedProperty(name="beta_id", value="b1"),
                    ExtractedProperty(name="active", value=True),
                ],
            ),
        ],
        edges=[
            ExtractedEdge(
                edge_id="sess1:AlphaToBeta:0",
                relationship_name="AlphaToBeta",
                from_node_id="sess1:Alpha:alpha_id=a1",
                to_node_id="sess1:Beta:beta_id=b1",
                properties=[
                    ExtractedProperty(name="weight", value=0.75),
                ],
            ),
        ],
    )

  def test_materialize_calls_delete_then_insert(self):
    mock_client = _mock_bq_client()
    mock_job = MagicMock()
    mock_job.result.return_value = None
    mock_client.query.return_value = mock_job
    mock_client.insert_rows_json.return_value = []

    mat = OntologyMaterializer(
        project_id="proj",
        dataset_id="ds",
        spec=_simple_spec(),
        bq_client=mock_client,
    )
    result = mat.materialize(self._make_graph(), session_ids=["sess1"])

    assert result["Alpha"] == 1
    assert result["Beta"] == 1
    assert result["AlphaToBeta"] == 1

    # Delete queries: 1 per entity + 1 per relationship = 3.
    assert mock_client.query.call_count == 3
    # Insert calls: 1 per entity + 1 per relationship = 3.
    assert mock_client.insert_rows_json.call_count == 3

  def test_inserted_node_rows_have_correct_shape(self):
    mock_client = _mock_bq_client()
    mock_job = MagicMock()
    mock_job.result.return_value = None
    mock_client.query.return_value = mock_job
    mock_client.insert_rows_json.return_value = []

    mat = OntologyMaterializer(
        project_id="proj",
        dataset_id="ds",
        spec=_simple_spec(),
        bq_client=mock_client,
    )
    mat.materialize(self._make_graph(), session_ids=["sess1"])

    # Find the Alpha insert call.
    alpha_call = None
    for call in mock_client.insert_rows_json.call_args_list:
      table_ref = call[0][0]
      if "alpha_table" in table_ref:
        alpha_call = call
        break

    assert alpha_call is not None
    rows = alpha_call[0][1]
    assert len(rows) == 1
    assert rows[0]["alpha_id"] == "a1"
    assert rows[0]["score"] == 0.9
    assert rows[0]["session_id"] == "sess1"

  def test_inserted_edge_rows_have_foreign_keys(self):
    mock_client = _mock_bq_client()
    mock_job = MagicMock()
    mock_job.result.return_value = None
    mock_client.query.return_value = mock_job
    mock_client.insert_rows_json.return_value = []

    mat = OntologyMaterializer(
        project_id="proj",
        dataset_id="ds",
        spec=_simple_spec(),
        bq_client=mock_client,
    )
    mat.materialize(self._make_graph(), session_ids=["sess1"])

    # Find the edge insert call.
    edge_call = None
    for call in mock_client.insert_rows_json.call_args_list:
      table_ref = call[0][0]
      if "alpha_beta_edges" in table_ref:
        edge_call = call
        break

    assert edge_call is not None
    rows = edge_call[0][1]
    assert len(rows) == 1
    assert rows[0]["alpha_id"] == "a1"
    assert rows[0]["beta_id"] == "b1"
    assert rows[0]["weight"] == 0.75
    assert rows[0]["session_id"] == "sess1"

  def test_unknown_entity_nodes_skipped(self):
    mock_client = _mock_bq_client()
    mock_job = MagicMock()
    mock_job.result.return_value = None
    mock_client.query.return_value = mock_job
    mock_client.insert_rows_json.return_value = []

    graph = ExtractedGraph(
        name="test",
        nodes=[
            ExtractedNode(
                node_id="sess1:Unknown:0",
                entity_name="Unknown",
                labels=["Unknown"],
                properties=[ExtractedProperty(name="x", value=1)],
            ),
        ],
    )
    mat = OntologyMaterializer(
        project_id="proj",
        dataset_id="ds",
        spec=_simple_spec(),
        bq_client=mock_client,
    )
    result = mat.materialize(graph, session_ids=["sess1"])
    assert result == {}

  def test_empty_graph(self):
    mock_client = _mock_bq_client()
    mat = OntologyMaterializer(
        project_id="proj",
        dataset_id="ds",
        spec=_simple_spec(),
        bq_client=mock_client,
    )
    result = mat.materialize(
        ExtractedGraph(name="empty"), session_ids=["sess1"]
    )
    assert result == {}

  def test_insert_error_does_not_crash(self):
    mock_client = _mock_bq_client()
    mock_job = MagicMock()
    mock_job.result.return_value = None
    mock_client.query.return_value = mock_job
    mock_client.insert_rows_json.side_effect = Exception("Insert failed")

    mat = OntologyMaterializer(
        project_id="proj",
        dataset_id="ds",
        spec=_simple_spec(),
        bq_client=mock_client,
    )
    result = mat.materialize(self._make_graph(), session_ids=["sess1"])
    # No entries in result because inserts failed.
    assert result == {}
