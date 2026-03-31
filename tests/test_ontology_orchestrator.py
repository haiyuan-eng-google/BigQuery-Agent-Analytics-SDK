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

"""Tests for ontology_orchestrator — showcase GQL + end-to-end pipeline."""

from __future__ import annotations

import os
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

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
from bigquery_agent_analytics.ontology_orchestrator import _short_alias
from bigquery_agent_analytics.ontology_orchestrator import build_ontology_graph
from bigquery_agent_analytics.ontology_orchestrator import compile_showcase_gql

# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

_DEMO_SPEC_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "examples",
    "ymgo_graph_spec.yaml",
)


def _make_entity(name, props=None, keys=None, source="p.d.t", labels=None):
  props = props or [PropertySpec(name="eid", type="string")]
  keys = keys or ["eid"]
  labels = labels or [name]
  return EntitySpec(
      name=name,
      binding=BindingSpec(source=source),
      keys=KeySpec(primary=keys),
      properties=props,
      labels=labels,
  )


def _simple_spec():
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


# ------------------------------------------------------------------ #
# _short_alias                                                         #
# ------------------------------------------------------------------ #


class TestShortAlias:

  def test_namespaced_name(self):
    assert _short_alias("mako_DecisionPoint") == "dp"

  def test_namespaced_multi_segment(self):
    assert _short_alias("sup_YahooAdUnit") == "yau"

  def test_camel_case(self):
    assert _short_alias("CandidateEdge") == "ce"

  def test_simple_name(self):
    assert _short_alias("Alpha") == "a"

  def test_lowercase_name(self):
    assert _short_alias("alpha") == "al"

  def test_prefix(self):
    assert _short_alias("CandidateEdge", prefix="e") == "ece"


# ------------------------------------------------------------------ #
# compile_showcase_gql                                                 #
# ------------------------------------------------------------------ #


class TestCompileShowcaseGql:

  def test_basic_gql(self):
    gql = compile_showcase_gql(_simple_spec(), "proj", "ds")
    assert "GRAPH `proj.ds.test_graph`" in gql
    assert "MATCH" in gql
    assert ":Alpha" in gql
    assert ":AlphaToBeta" in gql
    assert ":Beta" in gql
    assert "RETURN" in gql
    assert "LIMIT @result_limit" in gql

  def test_session_filter(self):
    gql = compile_showcase_gql(_simple_spec(), "proj", "ds")
    assert "WHERE" in gql
    assert "session_id = @session_id" in gql

  def test_no_session_filter(self):
    gql = compile_showcase_gql(
        _simple_spec(), "proj", "ds", session_filter=False
    )
    assert "WHERE" not in gql

  def test_custom_graph_name(self):
    gql = compile_showcase_gql(
        _simple_spec(), "proj", "ds", graph_name="custom"
    )
    assert "GRAPH `proj.ds.custom`" in gql

  def test_specific_relationship(self):
    gql = compile_showcase_gql(
        _simple_spec(), "proj", "ds", relationship_name="AlphaToBeta"
    )
    assert ":AlphaToBeta" in gql

  def test_unknown_relationship_raises(self):
    with pytest.raises(ValueError, match="Relationship.*NotHere.*not found"):
      compile_showcase_gql(
          _simple_spec(), "proj", "ds", relationship_name="NotHere"
      )

  def test_no_relationships_raises(self):
    spec = GraphSpec(
        name="no_rels",
        entities=[_make_entity("A")],
        relationships=[],
    )
    with pytest.raises(ValueError, match="no relationships"):
      compile_showcase_gql(spec, "proj", "ds")

  def test_return_columns_include_properties(self):
    gql = compile_showcase_gql(_simple_spec(), "proj", "ds")
    # Source entity properties.
    assert "src_alpha_id" in gql
    assert "src_score" in gql
    # Edge properties.
    assert "weight" in gql
    # Destination entity properties.
    assert "dst_beta_id" in gql
    assert "dst_active" in gql

  def test_demo_yaml_gql(self):
    """The real YMGO spec produces valid GQL structure."""
    spec = load_graph_spec(_DEMO_SPEC_PATH, env="p.d")
    gql = compile_showcase_gql(spec, "proj", "ds")
    assert "GRAPH `proj.ds.YMGO_Context_Graph_V3`" in gql
    assert ":mako_DecisionPoint" in gql
    assert ":CandidateEdge" in gql
    assert ":sup_YahooAdUnit" in gql

  def test_demo_yaml_second_relationship(self):
    spec = load_graph_spec(_DEMO_SPEC_PATH, env="p.d")
    gql = compile_showcase_gql(
        spec, "proj", "ds", relationship_name="ForCandidate"
    )
    assert ":mako_RejectionReason" in gql
    assert ":ForCandidate" in gql
    assert ":sup_YahooAdUnit" in gql

  def test_alias_collision_avoided(self):
    """Two entities with same alias prefix get disambiguated."""
    a = _make_entity(
        "SameAlias",
        props=[PropertySpec(name="a_id", type="string")],
        keys=["a_id"],
        source="p.d.a",
    )
    b = _make_entity(
        "SameAlias2",
        props=[PropertySpec(name="b_id", type="string")],
        keys=["b_id"],
        source="p.d.b",
        labels=["SameAlias2"],
    )
    rel = RelationshipSpec(
        name="R",
        from_entity="SameAlias",
        to_entity="SameAlias2",
        binding=BindingSpec(
            source="p.d.edges",
            from_columns=["a_id"],
            to_columns=["b_id"],
        ),
    )
    spec = GraphSpec(name="g", entities=[a, b], relationships=[rel])
    # Should not raise — aliases are disambiguated.
    gql = compile_showcase_gql(spec, "proj", "ds")
    assert "MATCH" in gql


# ------------------------------------------------------------------ #
# build_ontology_graph                                                 #
# ------------------------------------------------------------------ #


class TestBuildOntologyGraph:

  @patch(
      "bigquery_agent_analytics.ontology_property_graph"
      ".OntologyPropertyGraphCompiler"
  )
  @patch("bigquery_agent_analytics.ontology_materializer.OntologyMaterializer")
  @patch("bigquery_agent_analytics.ontology_graph.OntologyGraphManager")
  def test_full_pipeline(self, mock_mgr_cls, mock_mat_cls, mock_pg_cls):
    # Mock extractor.
    mock_extractor = MagicMock()
    mock_extractor.extract_graph.return_value = ExtractedGraph(
        name="test",
        nodes=[
            ExtractedNode(
                node_id="sess1:mako_DecisionPoint:decision_id=d1",
                entity_name="mako_DecisionPoint",
                labels=["mako_DecisionPoint"],
                properties=[
                    ExtractedProperty(name="decision_id", value="d1"),
                ],
            ),
        ],
        edges=[],
    )
    mock_mgr_cls.return_value = mock_extractor

    # Mock materializer.
    mock_materializer = MagicMock()
    mock_materializer.create_tables.return_value = {
        "mako_DecisionPoint": "p.d.decision_points"
    }
    mock_materializer.materialize.return_value = {"mako_DecisionPoint": 1}
    mock_mat_cls.return_value = mock_materializer

    # Mock property graph compiler.
    mock_compiler = MagicMock()
    mock_compiler.create_property_graph.return_value = True
    mock_pg_cls.return_value = mock_compiler

    result = build_ontology_graph(
        session_ids=["sess1"],
        spec_path=_DEMO_SPEC_PATH,
        project_id="proj",
        dataset_id="ds",
        env="p.d",
    )

    assert result["graph_name"] == "YMGO_Context_Graph_V3"
    assert result["graph_ref"] == "proj.ds.YMGO_Context_Graph_V3"
    assert result["property_graph_created"] is True
    assert result["rows_materialized"]["mako_DecisionPoint"] == 1
    assert len(result["graph"].nodes) == 1
    assert result["spec"].name == "YMGO_Context_Graph_V3"

    # Verify pipeline order.
    mock_extractor.extract_graph.assert_called_once()
    mock_materializer.create_tables.assert_called_once()
    mock_materializer.materialize.assert_called_once()
    mock_compiler.create_property_graph.assert_called_once()

  @patch(
      "bigquery_agent_analytics.ontology_property_graph"
      ".OntologyPropertyGraphCompiler"
  )
  @patch("bigquery_agent_analytics.ontology_materializer.OntologyMaterializer")
  @patch("bigquery_agent_analytics.ontology_graph.OntologyGraphManager")
  def test_custom_graph_name(self, mock_mgr_cls, mock_mat_cls, mock_pg_cls):
    mock_mgr_cls.return_value.extract_graph.return_value = ExtractedGraph(
        name="test"
    )
    mock_mat_cls.return_value.create_tables.return_value = {}
    mock_mat_cls.return_value.materialize.return_value = {}
    mock_pg_cls.return_value.create_property_graph.return_value = True

    result = build_ontology_graph(
        session_ids=["sess1"],
        spec_path=_DEMO_SPEC_PATH,
        project_id="proj",
        dataset_id="ds",
        env="p.d",
        graph_name="custom_graph",
    )

    assert result["graph_name"] == "custom_graph"
    assert result["graph_ref"] == "proj.ds.custom_graph"
    mock_pg_cls.return_value.create_property_graph.assert_called_once_with(
        graph_name="custom_graph"
    )

  @patch(
      "bigquery_agent_analytics.ontology_property_graph"
      ".OntologyPropertyGraphCompiler"
  )
  @patch("bigquery_agent_analytics.ontology_materializer.OntologyMaterializer")
  @patch("bigquery_agent_analytics.ontology_graph.OntologyGraphManager")
  def test_ai_generate_flag_passed(
      self, mock_mgr_cls, mock_mat_cls, mock_pg_cls
  ):
    mock_mgr_cls.return_value.extract_graph.return_value = ExtractedGraph(
        name="test"
    )
    mock_mat_cls.return_value.create_tables.return_value = {}
    mock_mat_cls.return_value.materialize.return_value = {}
    mock_pg_cls.return_value.create_property_graph.return_value = True

    build_ontology_graph(
        session_ids=["sess1"],
        spec_path=_DEMO_SPEC_PATH,
        project_id="proj",
        dataset_id="ds",
        env="p.d",
        use_ai_generate=False,
    )

    mock_mgr_cls.return_value.extract_graph.assert_called_once_with(
        session_ids=["sess1"],
        use_ai_generate=False,
    )
