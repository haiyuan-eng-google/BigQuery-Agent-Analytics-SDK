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

"""Tests for ontology_models — YAML spec + extracted graph models."""

from __future__ import annotations

import os
import textwrap

import pytest

from bigquery_agent_analytics.ontology_models import _resolve_inheritance
from bigquery_agent_analytics.ontology_models import _validate_graph_spec
from bigquery_agent_analytics.ontology_models import BindingSpec
from bigquery_agent_analytics.ontology_models import EntitySpec
from bigquery_agent_analytics.ontology_models import ExtractedEdge
from bigquery_agent_analytics.ontology_models import ExtractedGraph
from bigquery_agent_analytics.ontology_models import ExtractedNode
from bigquery_agent_analytics.ontology_models import ExtractedProperty
from bigquery_agent_analytics.ontology_models import GraphSpec
from bigquery_agent_analytics.ontology_models import KeySpec
from bigquery_agent_analytics.ontology_models import load_graph_spec
from bigquery_agent_analytics.ontology_models import load_graph_spec_from_string
from bigquery_agent_analytics.ontology_models import PropertySpec
from bigquery_agent_analytics.ontology_models import RelationshipSpec

# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

_MINIMAL_YAML = textwrap.dedent(
    """\
  graph:
    name: test_graph
    entities:
      - name: Foo
        binding:
          source: proj.ds.foo_table
        keys:
          primary: [foo_id]
        properties:
          - name: foo_id
            type: string
    relationships: []
"""
)

_TWO_ENTITY_YAML = textwrap.dedent(
    """\
  graph:
    name: test_graph
    entities:
      - name: Alpha
        binding:
          source: proj.ds.alpha
        keys:
          primary: [alpha_id]
        properties:
          - name: alpha_id
            type: string
      - name: Beta
        binding:
          source: proj.ds.beta
        keys:
          primary: [beta_id]
        properties:
          - name: beta_id
            type: string
    relationships:
      - name: AlphaToBeta
        from_entity: Alpha
        to_entity: Beta
        binding:
          source: proj.ds.alpha_beta_edges
          from_columns: [alpha_id]
          to_columns: [beta_id]
"""
)


def _make_entity(
    name="E",
    binding_source="p.d.t",
    keys=None,
    properties=None,
    extends=None,
):
  """Shortcut for building an EntitySpec in tests."""
  keys = keys or ["eid"]
  properties = properties or [PropertySpec(name="eid", type="string")]
  return EntitySpec(
      name=name,
      binding=BindingSpec(source=binding_source),
      keys=KeySpec(primary=keys),
      properties=properties,
      extends=extends,
  )


# ------------------------------------------------------------------ #
# Spec Model Construction                                             #
# ------------------------------------------------------------------ #


class TestSpecModels:

  def test_property_spec_fields(self):
    ps = PropertySpec(name="score", type="double", description="A score.")
    assert ps.name == "score"
    assert ps.type == "double"
    assert ps.description == "A score."

  def test_property_spec_defaults(self):
    ps = PropertySpec(name="x", type="string")
    assert ps.description == ""

  def test_key_spec(self):
    ks = KeySpec(primary=["a", "b"])
    assert ks.primary == ["a", "b"]

  def test_binding_spec_minimal(self):
    bs = BindingSpec(source="proj.ds.table")
    assert bs.source == "proj.ds.table"
    assert bs.from_columns is None
    assert bs.to_columns is None

  def test_binding_spec_with_join_columns(self):
    bs = BindingSpec(
        source="proj.ds.edges",
        from_columns=["a_id"],
        to_columns=["b_id"],
    )
    assert bs.from_columns == ["a_id"]
    assert bs.to_columns == ["b_id"]

  def test_entity_spec_minimal(self):
    e = _make_entity()
    assert e.name == "E"
    assert e.extends is None
    assert e.labels == []

  def test_entity_spec_with_extends(self):
    e = _make_entity(extends="Parent")
    assert e.extends == "Parent"

  def test_relationship_spec(self):
    r = RelationshipSpec(
        name="Edge",
        from_entity="A",
        to_entity="B",
        binding=BindingSpec(
            source="p.d.edges",
            from_columns=["a_id"],
            to_columns=["b_id"],
        ),
        properties=[PropertySpec(name="weight", type="double")],
    )
    assert r.name == "Edge"
    assert r.from_entity == "A"
    assert len(r.properties) == 1

  def test_graph_spec_empty(self):
    gs = GraphSpec(name="empty")
    assert gs.entities == []
    assert gs.relationships == []


# ------------------------------------------------------------------ #
# Extracted Models                                                     #
# ------------------------------------------------------------------ #


class TestExtractedModels:

  def test_extracted_property(self):
    ep = ExtractedProperty(name="score", value=0.95)
    assert ep.value == 0.95

  def test_extracted_node(self):
    en = ExtractedNode(
        node_id="n1",
        entity_name="Foo",
        labels=["Foo"],
        properties=[ExtractedProperty(name="x", value="hello")],
    )
    assert en.node_id == "n1"
    assert len(en.properties) == 1

  def test_extracted_edge(self):
    ee = ExtractedEdge(
        edge_id="e1",
        relationship_name="Link",
        from_node_id="n1",
        to_node_id="n2",
    )
    assert ee.from_node_id == "n1"

  def test_extracted_graph_empty(self):
    eg = ExtractedGraph(name="g")
    assert eg.nodes == []
    assert eg.edges == []

  def test_extracted_graph_with_data(self):
    eg = ExtractedGraph(
        name="g",
        nodes=[
            ExtractedNode(node_id="n1", entity_name="A"),
        ],
        edges=[
            ExtractedEdge(
                edge_id="e1",
                relationship_name="R",
                from_node_id="n1",
                to_node_id="n2",
            ),
        ],
    )
    assert len(eg.nodes) == 1
    assert len(eg.edges) == 1


# ------------------------------------------------------------------ #
# Inheritance Resolution                                               #
# ------------------------------------------------------------------ #


class TestResolveInheritance:

  def test_no_extends(self):
    spec = GraphSpec(name="g", entities=[_make_entity(name="A")])
    _resolve_inheritance(spec)
    assert spec.entities[0].labels == ["A"]

  def test_extends_adds_parent_label(self):
    spec = GraphSpec(
        name="g",
        entities=[_make_entity(name="Child", extends="Parent")],
    )
    _resolve_inheritance(spec)
    assert spec.entities[0].labels == ["Child", "Parent"]

  def test_extends_nonexistent_parent_allowed(self):
    """extends targets do NOT need to be defined entities."""
    spec = GraphSpec(
        name="g",
        entities=[_make_entity(name="X", extends="Ghost")],
    )
    _resolve_inheritance(spec)
    assert spec.entities[0].labels == ["X", "Ghost"]

  def test_multiple_entities_mixed(self):
    spec = GraphSpec(
        name="g",
        entities=[
            _make_entity(name="A"),
            _make_entity(name="B", extends="Base"),
            _make_entity(name="C"),
        ],
    )
    _resolve_inheritance(spec)
    assert spec.entities[0].labels == ["A"]
    assert spec.entities[1].labels == ["B", "Base"]
    assert spec.entities[2].labels == ["C"]


# ------------------------------------------------------------------ #
# Validation                                                           #
# ------------------------------------------------------------------ #


class TestValidateGraphSpec:

  def test_valid_spec_passes(self):
    spec = load_graph_spec_from_string(_TWO_ENTITY_YAML)
    # Should not raise.
    assert spec.name == "test_graph"

  def test_relationship_references_missing_from_entity(self):
    spec = GraphSpec(
        name="g",
        entities=[_make_entity(name="A")],
        relationships=[
            RelationshipSpec(
                name="R",
                from_entity="Missing",
                to_entity="A",
                binding=BindingSpec(source="p.d.t"),
            ),
        ],
    )
    _resolve_inheritance(spec)
    with pytest.raises(ValueError, match="from_entity.*Missing"):
      _validate_graph_spec(spec)

  def test_relationship_references_missing_to_entity(self):
    spec = GraphSpec(
        name="g",
        entities=[_make_entity(name="A")],
        relationships=[
            RelationshipSpec(
                name="R",
                from_entity="A",
                to_entity="Missing",
                binding=BindingSpec(source="p.d.t"),
            ),
        ],
    )
    _resolve_inheritance(spec)
    with pytest.raises(ValueError, match="to_entity.*Missing"):
      _validate_graph_spec(spec)

  def test_key_not_in_properties(self):
    entity = _make_entity(name="Bad", keys=["nonexistent"])
    spec = GraphSpec(name="g", entities=[entity])
    _resolve_inheritance(spec)
    with pytest.raises(ValueError, match="key column.*nonexistent"):
      _validate_graph_spec(spec)

  def test_relationship_from_columns_not_in_source_keys(self):
    spec = GraphSpec(
        name="g",
        entities=[
            _make_entity(name="A"),
            _make_entity(name="B"),
        ],
        relationships=[
            RelationshipSpec(
                name="R",
                from_entity="A",
                to_entity="B",
                binding=BindingSpec(
                    source="p.d.t",
                    from_columns=["bad_col"],
                    to_columns=["eid"],
                ),
            ),
        ],
    )
    _resolve_inheritance(spec)
    with pytest.raises(ValueError, match="from_column.*bad_col"):
      _validate_graph_spec(spec)

  def test_relationship_to_columns_not_in_target_keys(self):
    spec = GraphSpec(
        name="g",
        entities=[
            _make_entity(name="A"),
            _make_entity(name="B"),
        ],
        relationships=[
            RelationshipSpec(
                name="R",
                from_entity="A",
                to_entity="B",
                binding=BindingSpec(
                    source="p.d.t",
                    from_columns=["eid"],
                    to_columns=["bad_col"],
                ),
            ),
        ],
    )
    _resolve_inheritance(spec)
    with pytest.raises(ValueError, match="to_column.*bad_col"):
      _validate_graph_spec(spec)

  def test_duplicate_entity_names(self):
    spec = GraphSpec(
        name="g",
        entities=[_make_entity(name="Dup"), _make_entity(name="Dup")],
    )
    _resolve_inheritance(spec)
    with pytest.raises(ValueError, match="Duplicate entity name.*Dup"):
      _validate_graph_spec(spec)

  def test_duplicate_relationship_names(self):
    spec = GraphSpec(
        name="g",
        entities=[_make_entity(name="A"), _make_entity(name="B")],
        relationships=[
            RelationshipSpec(
                name="Dup",
                from_entity="A",
                to_entity="B",
                binding=BindingSpec(source="p.d.t"),
            ),
            RelationshipSpec(
                name="Dup",
                from_entity="A",
                to_entity="B",
                binding=BindingSpec(source="p.d.t2"),
            ),
        ],
    )
    _resolve_inheritance(spec)
    with pytest.raises(ValueError, match="Duplicate relationship name.*Dup"):
      _validate_graph_spec(spec)


# ------------------------------------------------------------------ #
# YAML Loading                                                         #
# ------------------------------------------------------------------ #


class TestLoadGraphSpec:

  def test_load_from_string_minimal(self):
    spec = load_graph_spec_from_string(_MINIMAL_YAML)
    assert spec.name == "test_graph"
    assert len(spec.entities) == 1
    assert spec.entities[0].name == "Foo"
    assert spec.entities[0].labels == ["Foo"]

  def test_load_from_string_with_env(self):
    yaml_str = textwrap.dedent(
        """\
      graph:
        name: g
        entities:
          - name: T
            binding:
              source: "{{ env }}.my_table"
            keys:
              primary: [tid]
            properties:
              - name: tid
                type: string
        relationships: []
    """
    )
    spec = load_graph_spec_from_string(yaml_str, env="proj.ds")
    assert spec.entities[0].binding.source == "proj.ds.my_table"

  def test_load_from_string_no_env_leaves_placeholder(self):
    yaml_str = textwrap.dedent(
        """\
      graph:
        name: g
        entities:
          - name: T
            binding:
              source: "{{ env }}.my_table"
            keys:
              primary: [tid]
            properties:
              - name: tid
                type: string
        relationships: []
    """
    )
    spec = load_graph_spec_from_string(yaml_str)
    assert "{{ env }}" in spec.entities[0].binding.source

  def test_env_substitution_no_spaces(self):
    """{{env}} (no spaces) also works."""
    yaml_str = textwrap.dedent(
        """\
      graph:
        name: g
        entities:
          - name: T
            binding:
              source: "{{env}}.my_table"
            keys:
              primary: [tid]
            properties:
              - name: tid
                type: string
        relationships: []
    """
    )
    spec = load_graph_spec_from_string(yaml_str, env="proj.ds")
    assert spec.entities[0].binding.source == "proj.ds.my_table"

  def test_load_from_file(self, tmp_path):
    yaml_file = tmp_path / "spec.yaml"
    yaml_file.write_text(_MINIMAL_YAML)
    spec = load_graph_spec(str(yaml_file))
    assert spec.name == "test_graph"

  def test_load_demo_yaml(self):
    """Load the real examples/ymgo_graph_spec.yaml."""
    demo_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "examples",
        "ymgo_graph_spec.yaml",
    )
    spec = load_graph_spec(demo_path, env="myproject.mydataset")
    assert spec.name == "YMGO_Context_Graph_V3"

  def test_demo_yaml_entities(self):
    demo_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "examples",
        "ymgo_graph_spec.yaml",
    )
    spec = load_graph_spec(demo_path, env="p.d")
    assert len(spec.entities) == 3
    names = [e.name for e in spec.entities]
    assert "mako_DecisionPoint" in names
    assert "sup_YahooAdUnit" in names
    assert "mako_RejectionReason" in names

  def test_demo_yaml_relationships(self):
    demo_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "examples",
        "ymgo_graph_spec.yaml",
    )
    spec = load_graph_spec(demo_path, env="p.d")
    assert len(spec.relationships) == 2
    rel_names = [r.name for r in spec.relationships]
    assert "CandidateEdge" in rel_names
    assert "ForCandidate" in rel_names

  def test_demo_yaml_inheritance(self):
    demo_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "examples",
        "ymgo_graph_spec.yaml",
    )
    spec = load_graph_spec(demo_path, env="p.d")
    ad_unit = next(e for e in spec.entities if e.name == "sup_YahooAdUnit")
    assert ad_unit.labels == ["sup_YahooAdUnit", "mako_Candidate"]

  def test_demo_yaml_env_bindings_resolved(self):
    demo_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "examples",
        "ymgo_graph_spec.yaml",
    )
    spec = load_graph_spec(demo_path, env="proj.ds")
    dp = next(e for e in spec.entities if e.name == "mako_DecisionPoint")
    assert dp.binding.source == "proj.ds.decision_points"

  def test_two_entity_yaml_with_relationship(self):
    spec = load_graph_spec_from_string(_TWO_ENTITY_YAML)
    assert len(spec.relationships) == 1
    rel = spec.relationships[0]
    assert rel.from_entity == "Alpha"
    assert rel.to_entity == "Beta"
    assert rel.binding.from_columns == ["alpha_id"]
    assert rel.binding.to_columns == ["beta_id"]


# ------------------------------------------------------------------ #
# Round-Trip Integration                                               #
# ------------------------------------------------------------------ #


class TestRoundTrip:

  def test_load_validate_and_construct_extracted(self):
    """Proves spec and extracted models are compatible."""
    spec = load_graph_spec_from_string(_TWO_ENTITY_YAML)

    graph = ExtractedGraph(
        name=spec.name,
        nodes=[
            ExtractedNode(
                node_id="a:1",
                entity_name="Alpha",
                labels=spec.entities[0].labels,
                properties=[ExtractedProperty(name="alpha_id", value="a1")],
            ),
        ],
        edges=[
            ExtractedEdge(
                edge_id="e:1",
                relationship_name="AlphaToBeta",
                from_node_id="a:1",
                to_node_id="b:1",
            ),
        ],
    )
    assert graph.name == "test_graph"
    assert len(graph.nodes) == 1
    assert graph.nodes[0].labels == ["Alpha"]
