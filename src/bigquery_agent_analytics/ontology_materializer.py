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

"""Physical table materialization + routing for ontology-driven graphs.

Provides ``OntologyMaterializer`` — takes an ``ExtractedGraph`` produced
by ``OntologyGraphManager`` and persists nodes and edges into BigQuery
tables according to the ontology binding configuration.

Each entity maps to one physical table (``binding.source``), and each
relationship maps to one physical table.  Persistence follows the
same delete-then-insert idempotency pattern as V3's
``ContextGraphManager``.

Example usage::

    from bigquery_agent_analytics.ontology_models import load_graph_spec
    from bigquery_agent_analytics.ontology_graph import OntologyGraphManager
    from bigquery_agent_analytics.ontology_materializer import (
        OntologyMaterializer,
    )

    spec = load_graph_spec("examples/ymgo_graph_spec.yaml", env="p.d")
    mgr = OntologyGraphManager(
        project_id="my-project", dataset_id="analytics", spec=spec,
    )
    graph = mgr.extract_graph(session_ids=["sess-1"])

    mat = OntologyMaterializer(
        project_id="my-project", dataset_id="analytics", spec=spec,
    )
    mat.create_tables()
    result = mat.materialize(graph, session_ids=["sess-1"])
    print(result)  # {"mako_DecisionPoint": 3, "CandidateEdge": 5, ...}
"""

from __future__ import annotations

import logging
from typing import Optional

from google.cloud import bigquery

from .ontology_models import EntitySpec
from .ontology_models import ExtractedEdge
from .ontology_models import ExtractedGraph
from .ontology_models import ExtractedNode
from .ontology_models import GraphSpec
from .ontology_models import PropertySpec
from .ontology_models import RelationshipSpec

logger = logging.getLogger("bigquery_agent_analytics." + __name__)

# ------------------------------------------------------------------ #
# Type mapping: YAML property types -> BQ DDL types                    #
# ------------------------------------------------------------------ #

_DDL_TYPE_MAP: dict[str, str] = {
    "string": "STRING",
    "int64": "INT64",
    "double": "FLOAT64",
    "float64": "FLOAT64",
    "bool": "BOOL",
    "boolean": "BOOL",
    "timestamp": "TIMESTAMP",
    "date": "DATE",
    "bytes": "BYTES",
}


def _ddl_type(yaml_type: str) -> str:
  """Map a YAML property type to a BQ DDL column type."""
  normalized = yaml_type.strip().lower()
  if normalized not in _DDL_TYPE_MAP:
    raise ValueError(
        f"Unsupported property type {yaml_type!r}. "
        f"Supported: {sorted(_DDL_TYPE_MAP.keys())}."
    )
  return _DDL_TYPE_MAP[normalized]


# ------------------------------------------------------------------ #
# DDL Generation                                                       #
# ------------------------------------------------------------------ #


def compile_entity_ddl(
    entity: EntitySpec,
    project_id: str,
    dataset_id: str,
) -> str:
  """Generate ``CREATE TABLE IF NOT EXISTS`` DDL for an entity.

  Columns: all spec properties + metadata columns
  (``session_id``, ``extracted_at``).
  """
  cols = []
  for prop in entity.properties:
    cols.append(f"  {prop.name} {_ddl_type(prop.type)}")
  cols.append("  session_id STRING")
  cols.append("  extracted_at TIMESTAMP")

  table_ref = entity.binding.source
  # If binding.source is already fully qualified (3-part), use as-is.
  # Otherwise, prefix with project.dataset.
  if table_ref.count(".") < 2:
    table_ref = f"{project_id}.{dataset_id}.{table_ref}"

  return (
      f"CREATE TABLE IF NOT EXISTS `{table_ref}` (\n" + ",\n".join(cols) + "\n)"
  )


def compile_relationship_ddl(
    rel: RelationshipSpec,
    spec: GraphSpec,
    project_id: str,
    dataset_id: str,
) -> str:
  """Generate ``CREATE TABLE IF NOT EXISTS`` DDL for a relationship.

  Columns: from-entity key columns + to-entity key columns +
  relationship properties + metadata.
  """
  entity_map = {e.name: e for e in spec.entities}
  src = entity_map[rel.from_entity]
  tgt = entity_map[rel.to_entity]
  src_prop_map = {p.name: p for p in src.properties}
  tgt_prop_map = {p.name: p for p in tgt.properties}

  cols = []
  # From-entity key columns.
  for col in src.keys.primary:
    prop = src_prop_map[col]
    cols.append(f"  {col} {_ddl_type(prop.type)}")
  # To-entity key columns (skip if already added via from-keys).
  seen = set(src.keys.primary)
  for col in tgt.keys.primary:
    if col not in seen:
      prop = tgt_prop_map[col]
      cols.append(f"  {col} {_ddl_type(prop.type)}")
      seen.add(col)
  # Relationship properties.
  for prop in rel.properties:
    if prop.name not in seen:
      cols.append(f"  {prop.name} {_ddl_type(prop.type)}")
      seen.add(prop.name)
  cols.append("  session_id STRING")
  cols.append("  extracted_at TIMESTAMP")

  table_ref = rel.binding.source
  if table_ref.count(".") < 2:
    table_ref = f"{project_id}.{dataset_id}.{table_ref}"

  return (
      f"CREATE TABLE IF NOT EXISTS `{table_ref}` (\n" + ",\n".join(cols) + "\n)"
  )


# ------------------------------------------------------------------ #
# Routing: ExtractedGraph -> row dicts                                 #
# ------------------------------------------------------------------ #


def _route_node(
    node: ExtractedNode,
    entity_spec: EntitySpec,
    session_id: str,
) -> dict:
  """Convert an ``ExtractedNode`` to a row dict for ``insert_rows_json``."""
  row: dict = {}
  for prop in node.properties:
    row[prop.name] = prop.value
  row["session_id"] = session_id
  return row


def _parse_key_segment(node_id: str) -> dict[str, str]:
  """Parse the key segment from a node ID.

  Node IDs look like ``{session_id}:{entity_name}:{k1=v1,k2=v2}``.
  Returns a dict of key-value pairs from the last segment, or empty
  dict if the format is unexpected (e.g. index-based fallback IDs).
  """
  parts = node_id.split(":")
  if len(parts) < 3:
    return {}
  key_segment = parts[-1]
  if "=" not in key_segment:
    return {}
  result = {}
  for pair in key_segment.split(","):
    if "=" in pair:
      k, v = pair.split("=", 1)
      result[k] = v
  return result


def _route_edge(
    edge: ExtractedEdge,
    rel: RelationshipSpec,
    spec: GraphSpec,
    session_id: str,
) -> dict:
  """Convert an ``ExtractedEdge`` to a row dict for ``insert_rows_json``.

  Foreign key columns are populated from the edge's ``from_node_id``
  and ``to_node_id`` key segments, mapped through the relationship's
  ``from_columns``/``to_columns`` binding.
  """
  entity_map = {e.name: e for e in spec.entities}
  row: dict = {}

  # Map from-entity keys.
  from_keys = _parse_key_segment(edge.from_node_id)
  src = entity_map[rel.from_entity]
  if rel.binding.from_columns:
    for fk_col, pk_col in zip(rel.binding.from_columns, src.keys.primary):
      row[fk_col] = from_keys.get(pk_col, "")
  else:
    row.update(from_keys)

  # Map to-entity keys.
  to_keys = _parse_key_segment(edge.to_node_id)
  tgt = entity_map[rel.to_entity]
  if rel.binding.to_columns:
    for fk_col, pk_col in zip(rel.binding.to_columns, tgt.keys.primary):
      row[fk_col] = to_keys.get(pk_col, "")
  else:
    row.update(to_keys)

  # Edge properties.
  for prop in edge.properties:
    row[prop.name] = prop.value

  row["session_id"] = session_id
  return row


# ------------------------------------------------------------------ #
# Delete queries (session-scoped cleanup for idempotency)              #
# ------------------------------------------------------------------ #

_DELETE_FOR_SESSIONS = """\
DELETE FROM `{table_ref}`
WHERE session_id IN UNNEST(@session_ids)
"""


# ------------------------------------------------------------------ #
# OntologyMaterializer                                                 #
# ------------------------------------------------------------------ #


class OntologyMaterializer:
  """Persists extracted ontology graphs into BigQuery tables.

  Each entity and relationship in the spec maps to a physical table
  via ``binding.source``.  Persistence uses delete-then-insert
  (same pattern as V3) for session-scoped idempotency.

  Args:
      project_id: GCP project ID.
      dataset_id: BigQuery dataset ID.
      spec: A validated ``GraphSpec``.
      bq_client: Optional pre-configured BigQuery client.
      location: BigQuery location.
  """

  def __init__(
      self,
      project_id: str,
      dataset_id: str,
      spec: GraphSpec,
      bq_client: Optional[bigquery.Client] = None,
      location: Optional[str] = None,
  ) -> None:
    self.project_id = project_id
    self.dataset_id = dataset_id
    self.spec = spec
    self.location = location
    self._bq_client = bq_client

  @property
  def bq_client(self) -> bigquery.Client:
    """Lazily initializes the BigQuery client."""
    if self._bq_client is None:
      kwargs: dict = {"project": self.project_id}
      if self.location:
        kwargs["location"] = self.location
      self._bq_client = bigquery.Client(**kwargs)
    return self._bq_client

  def _table_ref(self, binding_source: str) -> str:
    """Resolve a binding source to a fully qualified table reference."""
    if binding_source.count(".") >= 2:
      return binding_source
    return f"{self.project_id}.{self.dataset_id}.{binding_source}"

  # ---- DDL --------------------------------------------------------

  def get_entity_ddl(self, entity_name: str) -> str:
    """Return the CREATE TABLE DDL for a single entity."""
    entity_map = {e.name: e for e in self.spec.entities}
    if entity_name not in entity_map:
      raise ValueError(
          f"Entity {entity_name!r} not found in spec. "
          f"Available: {sorted(entity_map.keys())}."
      )
    return compile_entity_ddl(
        entity_map[entity_name], self.project_id, self.dataset_id
    )

  def get_relationship_ddl(self, rel_name: str) -> str:
    """Return the CREATE TABLE DDL for a single relationship."""
    rel_map = {r.name: r for r in self.spec.relationships}
    if rel_name not in rel_map:
      raise ValueError(
          f"Relationship {rel_name!r} not found in spec. "
          f"Available: {sorted(rel_map.keys())}."
      )
    return compile_relationship_ddl(
        rel_map[rel_name], self.spec, self.project_id, self.dataset_id
    )

  def get_all_ddl(self) -> dict[str, str]:
    """Return DDL for all entities and relationships.

    Returns:
        Dict mapping ``{entity_or_rel_name}`` → DDL string.
    """
    result = {}
    for entity in self.spec.entities:
      result[entity.name] = compile_entity_ddl(
          entity, self.project_id, self.dataset_id
      )
    for rel in self.spec.relationships:
      result[rel.name] = compile_relationship_ddl(
          rel, self.spec, self.project_id, self.dataset_id
      )
    return result

  def create_tables(self) -> dict[str, str]:
    """Execute DDL to create all entity and relationship tables.

    Returns:
        Dict mapping ``{name}`` → table reference for created tables.
    """
    created = {}
    for entity in self.spec.entities:
      ddl = compile_entity_ddl(entity, self.project_id, self.dataset_id)
      table_ref = self._table_ref(entity.binding.source)
      try:
        job = self.bq_client.query(ddl)
        job.result()
        created[entity.name] = table_ref
      except Exception as e:
        logger.warning(
            "Failed to create table for entity %s: %s",
            entity.name,
            e,
        )

    for rel in self.spec.relationships:
      ddl = compile_relationship_ddl(
          rel, self.spec, self.project_id, self.dataset_id
      )
      table_ref = self._table_ref(rel.binding.source)
      try:
        job = self.bq_client.query(ddl)
        job.result()
        created[rel.name] = table_ref
      except Exception as e:
        logger.warning(
            "Failed to create table for relationship %s: %s",
            rel.name,
            e,
        )

    return created

  # ---- Materialization --------------------------------------------

  def _delete_for_sessions(
      self, table_ref: str, session_ids: list[str]
  ) -> None:
    """Delete rows for given sessions from a table."""
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("session_ids", "STRING", session_ids),
        ]
    )
    try:
      job = self.bq_client.query(
          _DELETE_FOR_SESSIONS.format(table_ref=table_ref),
          job_config=job_config,
      )
      job.result()
    except Exception as e:
      err_msg = str(e).lower()
      if "not found" in err_msg or "does not exist" in err_msg:
        logger.debug("Table %s does not exist yet: %s", table_ref, e)
      else:
        logger.warning("Delete for sessions failed on %s: %s", table_ref, e)

  def materialize(
      self,
      graph: ExtractedGraph,
      session_ids: list[str],
  ) -> dict[str, int]:
    """Materialize an ``ExtractedGraph`` into BigQuery tables.

    Uses delete-then-insert for session-scoped idempotency:
    existing rows for the given sessions are deleted before
    inserting the new graph data.

    Args:
        graph: The extracted graph to persist.
        session_ids: Sessions being materialized (scopes the
            delete for idempotency).

    Returns:
        Dict mapping entity/relationship name to row count inserted.
    """
    entity_map = {e.name: e for e in self.spec.entities}
    rel_map = {r.name: r for r in self.spec.relationships}
    result: dict[str, int] = {}

    # Group nodes by entity name.
    nodes_by_entity: dict[str, list[ExtractedNode]] = {}
    for node in graph.nodes:
      if node.entity_name in entity_map:
        nodes_by_entity.setdefault(node.entity_name, []).append(node)
      else:
        logger.debug("Skipping node with unknown entity %r", node.entity_name)

    # Group edges by relationship name.
    edges_by_rel: dict[str, list[ExtractedEdge]] = {}
    for edge in graph.edges:
      if edge.relationship_name in rel_map:
        edges_by_rel.setdefault(edge.relationship_name, []).append(edge)
      else:
        logger.debug(
            "Skipping edge with unknown relationship %r",
            edge.relationship_name,
        )

    # Derive session_id for rows from the session_ids parameter.
    # For single-session extractions this is straightforward; for
    # multi-session, nodes already carry session_id in their node_id.
    default_session_id = session_ids[0] if len(session_ids) == 1 else ""

    # Materialize entities.
    for entity_name, nodes in nodes_by_entity.items():
      entity = entity_map[entity_name]
      table_ref = self._table_ref(entity.binding.source)

      # Delete existing data for idempotency.
      self._delete_for_sessions(table_ref, session_ids)

      rows = []
      for node in nodes:
        # Extract session_id from node_id: "{session_id}:{entity}:..."
        parts = node.node_id.split(":")
        sid = parts[0] if parts else default_session_id
        rows.append(_route_node(node, entity, sid))

      if rows:
        try:
          errors = self.bq_client.insert_rows_json(table_ref, rows)
          if errors:
            logger.error("Insert errors for %s: %s", entity_name, errors)
          else:
            result[entity_name] = len(rows)
        except Exception as e:
          logger.warning("Failed to insert rows for %s: %s", entity_name, e)

    # Materialize relationships.
    for rel_name, edges in edges_by_rel.items():
      rel = rel_map[rel_name]
      table_ref = self._table_ref(rel.binding.source)

      self._delete_for_sessions(table_ref, session_ids)

      rows = []
      for edge in edges:
        parts = edge.from_node_id.split(":")
        sid = parts[0] if parts else default_session_id
        rows.append(_route_edge(edge, rel, self.spec, sid))

      if rows:
        try:
          errors = self.bq_client.insert_rows_json(table_ref, rows)
          if errors:
            logger.error("Insert errors for %s: %s", rel_name, errors)
          else:
            result[rel_name] = len(rows)
        except Exception as e:
          logger.warning("Failed to insert rows for %s: %s", rel_name, e)

    return result
