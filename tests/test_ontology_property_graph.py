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

"""Tests for ontology_property_graph — Property Graph DDL transpiler."""

from __future__ import annotations

import os
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from bigquery_agent_analytics.ontology_models import BindingSpec
from bigquery_agent_analytics.ontology_models import EntitySpec
from bigquery_agent_analytics.ontology_models import GraphSpec
from bigquery_agent_analytics.ontology_models import KeySpec
from bigquery_agent_analytics.ontology_models import load_graph_spec
from bigquery_agent_analytics.ontology_models import PropertySpec
from bigquery_agent_analytics.ontology_models import RelationshipSpec
from bigquery_agent_analytics.ontology_property_graph import compile_edge_table_clause
from bigquery_agent_analytics.ontology_property_graph import compile_node_table_clause
from bigquery_agent_analytics.ontology_property_graph import compile_property_graph_ddl
from bigquery_agent_analytics.ontology_property_graph import OntologyPropertyGraphCompiler

# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #


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
  """Two entities, one relationship — matches the materializer test spec."""
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
# compile_node_table_clause                                            #
# ------------------------------------------------------------------ #


class TestCompileNodeTableClause:

  def test_basic_entity(self):
    entity = _make_entity(
        "Alpha",
        props=[
            PropertySpec(name="alpha_id", type="string"),
            PropertySpec(name="score", type="double"),
        ],
        keys=["alpha_id"],
        source="p.d.alpha_table",
    )
    clause = compile_node_table_clause(entity, "proj", "ds")
    assert "`p.d.alpha_table` AS Alpha" in clause
    # session_id is part of the node KEY for multi-session safety.
    assert "KEY (alpha_id, session_id)" in clause
    assert "LABEL Alpha" in clause
    # score is a non-key property.
    assert "score" in clause
    # Primary key and session_id should NOT be in PROPERTIES.
    props_section = clause.split("PROPERTIES")[1]
    assert "alpha_id" not in props_section
    assert "session_id" not in props_section
    # extracted_at metadata is present.
    assert "extracted_at" in clause

  def test_composite_keys(self):
    entity = _make_entity(
        "Multi",
        props=[
            PropertySpec(name="k1", type="string"),
            PropertySpec(name="k2", type="int64"),
            PropertySpec(name="val", type="string"),
        ],
        keys=["k1", "k2"],
        source="p.d.multi",
    )
    clause = compile_node_table_clause(entity, "proj", "ds")
    assert "KEY (k1, k2, session_id)" in clause
    # val should be in PROPERTIES but k1, k2, session_id should not.
    props_section = clause.split("PROPERTIES")[1]
    assert "val" in props_section
    assert "k1" not in props_section
    assert "k2" not in props_section
    assert "session_id" not in props_section

  def test_multiple_labels(self):
    """Entity with extends gets multiple LABEL lines."""
    entity = _make_entity(
        "sup_YahooAdUnit",
        props=[PropertySpec(name="adUnitId", type="string")],
        keys=["adUnitId"],
        source="p.d.yahoo_ad_units",
        labels=["sup_YahooAdUnit", "mako_Candidate"],
    )
    clause = compile_node_table_clause(entity, "proj", "ds")
    assert "LABEL sup_YahooAdUnit" in clause
    assert "LABEL mako_Candidate" in clause

  def test_short_source_prefixed(self):
    entity = _make_entity("A", source="my_table")
    clause = compile_node_table_clause(entity, "proj", "ds")
    assert "`proj.ds.my_table` AS A" in clause

  def test_fully_qualified_source_used_as_is(self):
    entity = _make_entity("A", source="other.ds.t")
    clause = compile_node_table_clause(entity, "proj", "ds")
    assert "`other.ds.t` AS A" in clause


# ------------------------------------------------------------------ #
# compile_edge_table_clause                                            #
# ------------------------------------------------------------------ #


class TestCompileEdgeTableClause:

  def test_basic_relationship(self):
    spec = _simple_spec()
    rel = spec.relationships[0]
    clause = compile_edge_table_clause(rel, spec, "proj", "ds")
    assert "`p.d.alpha_beta_edges` AS AlphaToBeta" in clause
    assert "KEY (alpha_id, beta_id, session_id)" in clause
    assert (
        "SOURCE KEY (alpha_id, session_id) "
        "REFERENCES Alpha (alpha_id, session_id)"
    ) in clause
    assert (
        "DESTINATION KEY (beta_id, session_id) "
        "REFERENCES Beta (beta_id, session_id)"
    ) in clause
    assert "LABEL AlphaToBeta" in clause
    assert "weight" in clause
    # session_id is now in KEY, not in PROPERTIES.
    props_section = clause.split("PROPERTIES")[1]
    assert "session_id" not in props_section
    assert "extracted_at" in clause

  def test_edge_key_deduplicates_overlapping_columns(self):
    """If from_columns and to_columns share a column, edge KEY deduplicates."""
    a = _make_entity(
        "A",
        props=[PropertySpec(name="shared_id", type="string")],
        keys=["shared_id"],
        source="p.d.a",
    )
    b = _make_entity(
        "B",
        props=[PropertySpec(name="shared_id", type="string")],
        keys=["shared_id"],
        source="p.d.b",
    )
    rel = RelationshipSpec(
        name="SelfRef",
        from_entity="A",
        to_entity="B",
        binding=BindingSpec(
            source="p.d.self_edges",
            from_columns=["shared_id"],
            to_columns=["shared_id"],
        ),
    )
    spec = GraphSpec(name="g", entities=[a, b], relationships=[rel])
    clause = compile_edge_table_clause(rel, spec, "proj", "ds")
    # KEY should not duplicate shared_id; session_id is appended.
    assert "KEY (shared_id, session_id)" in clause

  def test_composite_foreign_keys(self):
    src = _make_entity(
        "Src",
        props=[
            PropertySpec(name="k1", type="string"),
            PropertySpec(name="k2", type="int64"),
        ],
        keys=["k1", "k2"],
        source="p.d.src",
    )
    tgt = _make_entity("Tgt", source="p.d.tgt")
    rel = RelationshipSpec(
        name="R",
        from_entity="Src",
        to_entity="Tgt",
        binding=BindingSpec(
            source="p.d.edges",
            from_columns=["k1", "k2"],
            to_columns=["eid"],
        ),
    )
    spec = GraphSpec(name="g", entities=[src, tgt], relationships=[rel])
    clause = compile_edge_table_clause(rel, spec, "proj", "ds")
    assert "KEY (k1, k2, eid, session_id)" in clause
    assert (
        "SOURCE KEY (k1, k2, session_id) REFERENCES Src (k1, k2, session_id)"
    ) in clause
    assert (
        "DESTINATION KEY (eid, session_id) REFERENCES Tgt (eid, session_id)"
    ) in clause

  def test_default_columns_when_binding_omits(self):
    """When from_columns/to_columns are not set, defaults to entity PKs."""
    a = _make_entity(
        "A",
        props=[PropertySpec(name="a_id", type="string")],
        keys=["a_id"],
        source="p.d.a",
    )
    b = _make_entity(
        "B",
        props=[PropertySpec(name="b_id", type="string")],
        keys=["b_id"],
        source="p.d.b",
    )
    rel = RelationshipSpec(
        name="R",
        from_entity="A",
        to_entity="B",
        binding=BindingSpec(source="p.d.edges"),
    )
    spec = GraphSpec(name="g", entities=[a, b], relationships=[rel])
    clause = compile_edge_table_clause(rel, spec, "proj", "ds")
    assert (
        "SOURCE KEY (a_id, session_id) REFERENCES A (a_id, session_id)"
    ) in clause
    assert (
        "DESTINATION KEY (b_id, session_id) REFERENCES B (b_id, session_id)"
    ) in clause

  def test_edge_properties_exclude_key_columns(self):
    """Edge properties should not include columns already in KEY."""
    a = _make_entity(
        "A",
        props=[PropertySpec(name="a_id", type="string")],
        keys=["a_id"],
        source="p.d.a",
    )
    b = _make_entity(
        "B",
        props=[PropertySpec(name="b_id", type="string")],
        keys=["b_id"],
        source="p.d.b",
    )
    rel = RelationshipSpec(
        name="R",
        from_entity="A",
        to_entity="B",
        binding=BindingSpec(
            source="p.d.edges",
            from_columns=["a_id"],
            to_columns=["b_id"],
        ),
        properties=[
            PropertySpec(name="a_id", type="string"),
            PropertySpec(name="extra", type="string"),
        ],
    )
    spec = GraphSpec(name="g", entities=[a, b], relationships=[rel])
    clause = compile_edge_table_clause(rel, spec, "proj", "ds")
    props_section = clause.split("PROPERTIES")[1]
    # a_id is in the KEY, so it should be excluded from PROPERTIES.
    assert "a_id" not in props_section
    assert "extra" in props_section

  def test_subset_from_columns_raises(self):
    """from_columns that are a subset of the entity PK are rejected."""
    src = _make_entity(
        "Src",
        props=[
            PropertySpec(name="k1", type="string"),
            PropertySpec(name="k2", type="int64"),
        ],
        keys=["k1", "k2"],
        source="p.d.src",
    )
    tgt = _make_entity("Tgt", source="p.d.tgt")
    rel = RelationshipSpec(
        name="R",
        from_entity="Src",
        to_entity="Tgt",
        binding=BindingSpec(
            source="p.d.edges",
            from_columns=["k2"],
            to_columns=["eid"],
        ),
    )
    spec = GraphSpec(name="g", entities=[src, tgt], relationships=[rel])
    with pytest.raises(
        ValueError, match="from_columns.*do not match.*primary key"
    ):
      compile_edge_table_clause(rel, spec, "proj", "ds")

  def test_subset_to_columns_raises(self):
    """to_columns that are a subset of the entity PK are rejected."""
    src = _make_entity("Src", source="p.d.src")
    tgt = _make_entity(
        "Tgt",
        props=[
            PropertySpec(name="t1", type="string"),
            PropertySpec(name="t2", type="string"),
        ],
        keys=["t1", "t2"],
        source="p.d.tgt",
    )
    rel = RelationshipSpec(
        name="R",
        from_entity="Src",
        to_entity="Tgt",
        binding=BindingSpec(
            source="p.d.edges",
            from_columns=["eid"],
            to_columns=["t1"],
        ),
    )
    spec = GraphSpec(name="g", entities=[src, tgt], relationships=[rel])
    with pytest.raises(
        ValueError, match="to_columns.*do not match.*primary key"
    ):
      compile_edge_table_clause(rel, spec, "proj", "ds")

  def test_full_pk_binding_accepted(self):
    """from_columns == entity PK passes validation."""
    src = _make_entity(
        "Src",
        props=[
            PropertySpec(name="k1", type="string"),
            PropertySpec(name="k2", type="int64"),
        ],
        keys=["k1", "k2"],
        source="p.d.src",
    )
    tgt = _make_entity("Tgt", source="p.d.tgt")
    rel = RelationshipSpec(
        name="R",
        from_entity="Src",
        to_entity="Tgt",
        binding=BindingSpec(
            source="p.d.edges",
            from_columns=["k1", "k2"],
            to_columns=["eid"],
        ),
    )
    spec = GraphSpec(name="g", entities=[src, tgt], relationships=[rel])
    clause = compile_edge_table_clause(rel, spec, "proj", "ds")
    assert (
        "SOURCE KEY (k1, k2, session_id) " "REFERENCES Src (k1, k2, session_id)"
    ) in clause


# ------------------------------------------------------------------ #
# compile_property_graph_ddl                                           #
# ------------------------------------------------------------------ #


class TestCompilePropertyGraphDdl:

  def test_full_ddl(self):
    ddl = compile_property_graph_ddl(_simple_spec(), "proj", "ds")
    assert "CREATE OR REPLACE PROPERTY GRAPH" in ddl
    assert "`proj.ds.test_graph`" in ddl
    assert "NODE TABLES" in ddl
    assert "EDGE TABLES" in ddl
    # Both entities present.
    assert "AS Alpha" in ddl
    assert "AS Beta" in ddl
    # Relationship present.
    assert "AS AlphaToBeta" in ddl

  def test_custom_graph_name(self):
    ddl = compile_property_graph_ddl(
        _simple_spec(), "proj", "ds", graph_name="custom_name"
    )
    assert "`proj.ds.custom_name`" in ddl

  def test_no_relationships(self):
    """Spec with entities but no relationships omits EDGE TABLES block."""
    spec = GraphSpec(
        name="nodes_only",
        entities=[_make_entity("A", source="p.d.a")],
        relationships=[],
    )
    ddl = compile_property_graph_ddl(spec, "proj", "ds")
    assert "NODE TABLES" in ddl
    assert "EDGE TABLES" not in ddl

  def test_no_entities_raises(self):
    spec = GraphSpec(name="empty", entities=[], relationships=[])
    with pytest.raises(ValueError, match="no entities"):
      compile_property_graph_ddl(spec, "proj", "ds")

  def test_demo_yaml_generates_valid_ddl(self):
    """The real YMGO spec produces valid DDL structure."""
    demo_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "examples",
        "ymgo_graph_spec.yaml",
    )
    spec = load_graph_spec(demo_path, env="p.d")
    ddl = compile_property_graph_ddl(spec, "proj", "ds")
    assert "CREATE OR REPLACE PROPERTY GRAPH" in ddl
    # All 3 entities.
    assert "AS mako_DecisionPoint" in ddl
    assert "AS sup_YahooAdUnit" in ddl
    assert "AS mako_RejectionReason" in ddl
    # Label inheritance.
    assert "LABEL mako_Candidate" in ddl
    # Both relationships.
    assert "AS CandidateEdge" in ddl
    assert "AS ForCandidate" in ddl
    # FK references include session_id for multi-session safety.
    assert "REFERENCES mako_DecisionPoint (decision_id, session_id)" in ddl
    assert "REFERENCES sup_YahooAdUnit (adUnitId, session_id)" in ddl
    assert "REFERENCES mako_RejectionReason (rejection_id, session_id)" in ddl

  def test_multiple_node_tables_separated_by_commas(self):
    ddl = compile_property_graph_ddl(_simple_spec(), "proj", "ds")
    # Node clauses should be comma-separated within NODE TABLES block.
    node_block = ddl.split("NODE TABLES")[1].split("EDGE TABLES")[0]
    # Two node entries → one comma separator.
    assert node_block.count("AS Alpha") == 1
    assert node_block.count("AS Beta") == 1


# ------------------------------------------------------------------ #
# OntologyPropertyGraphCompiler                                        #
# ------------------------------------------------------------------ #


class TestCompilerInit:

  def test_basic_init(self):
    compiler = OntologyPropertyGraphCompiler(
        project_id="proj",
        dataset_id="ds",
        spec=_simple_spec(),
        bq_client=_mock_bq_client(),
    )
    assert compiler.project_id == "proj"
    assert compiler.dataset_id == "ds"

  @patch("bigquery_agent_analytics.ontology_property_graph.bigquery.Client")
  def test_lazy_client(self, mock_client_cls):
    mock_client_cls.return_value = MagicMock()
    compiler = OntologyPropertyGraphCompiler(
        project_id="proj",
        dataset_id="ds",
        spec=_simple_spec(),
    )
    _ = compiler.bq_client
    mock_client_cls.assert_called_once_with(project="proj")


class TestCompilerGetDdl:

  def test_get_ddl(self):
    compiler = OntologyPropertyGraphCompiler(
        project_id="proj",
        dataset_id="ds",
        spec=_simple_spec(),
        bq_client=_mock_bq_client(),
    )
    ddl = compiler.get_ddl()
    assert "CREATE OR REPLACE PROPERTY GRAPH" in ddl
    assert "`proj.ds.test_graph`" in ddl

  def test_get_ddl_custom_name(self):
    compiler = OntologyPropertyGraphCompiler(
        project_id="proj",
        dataset_id="ds",
        spec=_simple_spec(),
        bq_client=_mock_bq_client(),
    )
    ddl = compiler.get_ddl(graph_name="custom")
    assert "`proj.ds.custom`" in ddl


class TestCompilerGetClauses:

  def test_get_node_table_clause(self):
    compiler = OntologyPropertyGraphCompiler(
        project_id="proj",
        dataset_id="ds",
        spec=_simple_spec(),
        bq_client=_mock_bq_client(),
    )
    clause = compiler.get_node_table_clause("Alpha")
    assert "AS Alpha" in clause
    assert "KEY (alpha_id, session_id)" in clause

  def test_get_node_table_clause_unknown_raises(self):
    compiler = OntologyPropertyGraphCompiler(
        project_id="proj",
        dataset_id="ds",
        spec=_simple_spec(),
        bq_client=_mock_bq_client(),
    )
    with pytest.raises(ValueError, match="Entity.*NotHere.*not found"):
      compiler.get_node_table_clause("NotHere")

  def test_get_edge_table_clause(self):
    compiler = OntologyPropertyGraphCompiler(
        project_id="proj",
        dataset_id="ds",
        spec=_simple_spec(),
        bq_client=_mock_bq_client(),
    )
    clause = compiler.get_edge_table_clause("AlphaToBeta")
    assert "AS AlphaToBeta" in clause
    assert "SOURCE KEY" in clause

  def test_get_edge_table_clause_unknown_raises(self):
    compiler = OntologyPropertyGraphCompiler(
        project_id="proj",
        dataset_id="ds",
        spec=_simple_spec(),
        bq_client=_mock_bq_client(),
    )
    with pytest.raises(ValueError, match="Relationship.*NotHere.*not found"):
      compiler.get_edge_table_clause("NotHere")


class TestCompilerCreatePropertyGraph:

  def test_create_success(self):
    mock_client = _mock_bq_client()
    mock_job = MagicMock()
    mock_job.result.return_value = None
    mock_client.query.return_value = mock_job

    compiler = OntologyPropertyGraphCompiler(
        project_id="proj",
        dataset_id="ds",
        spec=_simple_spec(),
        bq_client=mock_client,
    )
    result = compiler.create_property_graph()
    assert result is True
    mock_client.query.assert_called_once()
    ddl = mock_client.query.call_args[0][0]
    assert "CREATE OR REPLACE PROPERTY GRAPH" in ddl

  def test_create_failure_returns_false(self):
    mock_client = _mock_bq_client()
    mock_client.query.side_effect = Exception("BQ error")

    compiler = OntologyPropertyGraphCompiler(
        project_id="proj",
        dataset_id="ds",
        spec=_simple_spec(),
        bq_client=mock_client,
    )
    result = compiler.create_property_graph()
    assert result is False

  def test_create_with_custom_name(self):
    mock_client = _mock_bq_client()
    mock_job = MagicMock()
    mock_job.result.return_value = None
    mock_client.query.return_value = mock_job

    compiler = OntologyPropertyGraphCompiler(
        project_id="proj",
        dataset_id="ds",
        spec=_simple_spec(),
        bq_client=mock_client,
    )
    compiler.create_property_graph(graph_name="my_graph")
    ddl = mock_client.query.call_args[0][0]
    assert "`proj.ds.my_graph`" in ddl
