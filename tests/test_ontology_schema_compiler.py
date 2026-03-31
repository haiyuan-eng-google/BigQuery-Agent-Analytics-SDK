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

"""Tests for ontology_schema_compiler — output_schema + prompt generation."""

from __future__ import annotations

import json
import os

import pytest

from bigquery_agent_analytics.ontology_models import BindingSpec
from bigquery_agent_analytics.ontology_models import EntitySpec
from bigquery_agent_analytics.ontology_models import GraphSpec
from bigquery_agent_analytics.ontology_models import KeySpec
from bigquery_agent_analytics.ontology_models import load_graph_spec
from bigquery_agent_analytics.ontology_models import PropertySpec
from bigquery_agent_analytics.ontology_models import RelationshipSpec
from bigquery_agent_analytics.ontology_schema_compiler import _bq_schema_type
from bigquery_agent_analytics.ontology_schema_compiler import compile_extraction_prompt
from bigquery_agent_analytics.ontology_schema_compiler import compile_output_schema

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


# ------------------------------------------------------------------ #
# Type Mapping                                                         #
# ------------------------------------------------------------------ #


class TestTypeMapping:

  def test_string(self):
    assert _bq_schema_type("string") == "STRING"

  def test_int64(self):
    assert _bq_schema_type("int64") == "INTEGER"

  def test_double(self):
    assert _bq_schema_type("double") == "NUMBER"

  def test_float64(self):
    assert _bq_schema_type("float64") == "NUMBER"

  def test_bool(self):
    assert _bq_schema_type("bool") == "BOOLEAN"

  def test_boolean(self):
    assert _bq_schema_type("boolean") == "BOOLEAN"

  def test_timestamp_maps_to_string(self):
    assert _bq_schema_type("timestamp") == "STRING"

  def test_case_insensitive(self):
    assert _bq_schema_type("String") == "STRING"
    assert _bq_schema_type("INT64") == "INTEGER"

  def test_unknown_type_raises(self):
    with pytest.raises(ValueError, match="Unsupported property type"):
      _bq_schema_type("array<string>")


# ------------------------------------------------------------------ #
# compile_output_schema                                                #
# ------------------------------------------------------------------ #


class TestCompileOutputSchema:

  def test_produces_valid_json(self):
    schema_str = compile_output_schema(_simple_spec())
    parsed = json.loads(schema_str)
    assert parsed["type"] == "OBJECT"

  def test_top_level_has_nodes_and_edges(self):
    parsed = json.loads(compile_output_schema(_simple_spec()))
    props = parsed["properties"]
    assert "nodes" in props
    assert "edges" in props
    assert props["nodes"]["type"] == "ARRAY"
    assert props["edges"]["type"] == "ARRAY"

  def test_node_schema_has_entity_name(self):
    parsed = json.loads(compile_output_schema(_simple_spec()))
    node_props = parsed["properties"]["nodes"]["items"]["properties"]
    assert "entity_name" in node_props
    assert node_props["entity_name"]["type"] == "STRING"

  def test_node_schema_merges_all_entity_properties(self):
    parsed = json.loads(compile_output_schema(_simple_spec()))
    node_props = parsed["properties"]["nodes"]["items"]["properties"]
    # From Alpha: alpha_id (STRING), score (NUMBER)
    # From Beta: beta_id (STRING), active (BOOLEAN)
    assert node_props["alpha_id"]["type"] == "STRING"
    assert node_props["score"]["type"] == "NUMBER"
    assert node_props["beta_id"]["type"] == "STRING"
    assert node_props["active"]["type"] == "BOOLEAN"

  def test_edge_schema_has_structural_fields(self):
    parsed = json.loads(compile_output_schema(_simple_spec()))
    edge_props = parsed["properties"]["edges"]["items"]["properties"]
    for field in [
        "relationship_name",
        "from_entity_name",
        "to_entity_name",
        "from_key",
        "to_key",
    ]:
      assert field in edge_props
      assert edge_props[field]["type"] == "STRING"

  def test_edge_schema_includes_relationship_properties(self):
    parsed = json.loads(compile_output_schema(_simple_spec()))
    edge_props = parsed["properties"]["edges"]["items"]["properties"]
    assert "weight" in edge_props
    assert edge_props["weight"]["type"] == "NUMBER"

  def test_compact_json_no_whitespace(self):
    schema_str = compile_output_schema(_simple_spec())
    assert " " not in schema_str

  def test_entity_subset_filters_entities(self):
    parsed = json.loads(
        compile_output_schema(_simple_spec(), entity_names=["Alpha"])
    )
    node_props = parsed["properties"]["nodes"]["items"]["properties"]
    assert "alpha_id" in node_props
    # Beta properties should not be present.
    assert "beta_id" not in node_props

  def test_entity_subset_filters_relationships(self):
    """Relationships with endpoints outside the subset are excluded."""
    parsed = json.loads(
        compile_output_schema(_simple_spec(), entity_names=["Alpha"])
    )
    edge_props = parsed["properties"]["edges"]["items"]["properties"]
    # AlphaToBeta excluded because Beta not in subset.
    # Only structural fields remain; weight should be absent.
    assert "weight" not in edge_props

  def test_unknown_entity_name_raises(self):
    with pytest.raises(ValueError, match="Entity.*NotHere.*not found"):
      compile_output_schema(_simple_spec(), entity_names=["NotHere"])

  def test_no_entities_produces_empty_arrays(self):
    spec = GraphSpec(name="empty")
    parsed = json.loads(compile_output_schema(spec))
    node_props = parsed["properties"]["nodes"]["items"]["properties"]
    # Only entity_name, no user properties.
    assert list(node_props.keys()) == ["entity_name"]

  def test_unsupported_property_type_raises(self):
    entity = _make_entity(
        "Bad",
        props=[PropertySpec(name="x", type="array<string>")],
        keys=["x"],
    )
    spec = GraphSpec(name="g", entities=[entity])
    with pytest.raises(ValueError, match="Unsupported property type"):
      compile_output_schema(spec)

  def test_demo_yaml_compiles(self):
    """The real YMGO spec compiles without error."""
    demo_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "examples",
        "ymgo_graph_spec.yaml",
    )
    spec = load_graph_spec(demo_path, env="p.d")
    schema_str = compile_output_schema(spec)
    parsed = json.loads(schema_str)
    node_props = parsed["properties"]["nodes"]["items"]["properties"]
    # Spot-check a few properties from different entities.
    assert "decision_id" in node_props
    assert "adUnitId" in node_props
    assert "rejectionType" in node_props

  def test_schema_matches_v3_dialect(self):
    """Schema uses the same BQ JSON Schema types as V3."""
    parsed = json.loads(compile_output_schema(_simple_spec()))
    # Verify we use the uppercase STRING/NUMBER/BOOLEAN/INTEGER
    # dialect, not lowercase json-schema types.
    node_props = parsed["properties"]["nodes"]["items"]["properties"]
    for v in node_props.values():
      assert v["type"] in ("STRING", "NUMBER", "BOOLEAN", "INTEGER")


# ------------------------------------------------------------------ #
# compile_extraction_prompt                                            #
# ------------------------------------------------------------------ #


class TestCompileExtractionPrompt:

  def test_contains_entity_names(self):
    prompt = compile_extraction_prompt(_simple_spec())
    assert "Alpha" in prompt
    assert "Beta" in prompt

  def test_contains_relationship_names(self):
    prompt = compile_extraction_prompt(_simple_spec())
    assert "AlphaToBeta" in prompt
    assert "Alpha -> Beta" in prompt

  def test_contains_property_names(self):
    prompt = compile_extraction_prompt(_simple_spec())
    assert "alpha_id" in prompt
    assert "score" in prompt

  def test_contains_rules(self):
    prompt = compile_extraction_prompt(_simple_spec())
    assert "Do not invent unknown entity types" in prompt

  def test_ends_with_payload_marker(self):
    prompt = compile_extraction_prompt(_simple_spec())
    assert prompt.rstrip().endswith("Payload:")

  def test_entity_subset(self):
    prompt = compile_extraction_prompt(_simple_spec(), entity_names=["Alpha"])
    assert "Alpha" in prompt
    assert "Beta" not in prompt.split("Relationship types:")[0]

  def test_no_relationships_section_when_empty(self):
    spec = GraphSpec(
        name="g",
        entities=[_make_entity("Solo")],
    )
    prompt = compile_extraction_prompt(spec)
    assert "Relationship types:" not in prompt

  def test_entity_description_included(self):
    entity = EntitySpec(
        name="Thing",
        description="A test thing.",
        binding=BindingSpec(source="p.d.t"),
        keys=KeySpec(primary=["tid"]),
        properties=[PropertySpec(name="tid", type="string")],
        labels=["Thing"],
    )
    spec = GraphSpec(name="g", entities=[entity])
    prompt = compile_extraction_prompt(spec)
    assert "A test thing." in prompt

  def test_demo_yaml_prompt(self):
    demo_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "examples",
        "ymgo_graph_spec.yaml",
    )
    spec = load_graph_spec(demo_path, env="p.d")
    prompt = compile_extraction_prompt(spec)
    assert "mako_DecisionPoint" in prompt
    assert "CandidateEdge" in prompt
    assert "ForCandidate" in prompt
