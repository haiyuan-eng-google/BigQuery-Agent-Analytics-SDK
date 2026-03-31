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

"""Ontology-driven graph extraction engine (V4).

Provides ``OntologyGraphManager`` — queries raw ADK telemetry from
``agent_events``, constructs ``AI.GENERATE`` queries using the compiled
ontology schema and prompt, and hydrates the results into typed
``ExtractedGraph`` objects.

This module sits beside the V3 ``ContextGraphManager`` and reuses the
same query patterns (parameterized queries, markdown fence stripping,
lazy client initialization) without modifying V3 code.

Example usage::

    from bigquery_agent_analytics.ontology_models import load_graph_spec
    from bigquery_agent_analytics.ontology_graph import OntologyGraphManager

    spec = load_graph_spec("examples/ymgo_graph_spec.yaml", env="p.d")
    mgr = OntologyGraphManager(
        project_id="my-project",
        dataset_id="analytics",
        spec=spec,
    )
    graph = mgr.extract_graph(session_ids=["sess-1", "sess-2"])
    print(len(graph.nodes), "nodes,", len(graph.edges), "edges")
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from google.cloud import bigquery

from .ontology_models import EntitySpec
from .ontology_models import ExtractedEdge
from .ontology_models import ExtractedGraph
from .ontology_models import ExtractedNode
from .ontology_models import ExtractedProperty
from .ontology_models import GraphSpec
from .ontology_schema_compiler import compile_extraction_prompt
from .ontology_schema_compiler import compile_output_schema

logger = logging.getLogger("bigquery_agent_analytics." + __name__)

# ------------------------------------------------------------------ #
# SQL Templates                                                        #
# ------------------------------------------------------------------ #


def _escape_sql_literal(value: str) -> str:
  """Escape a Python string for embedding in a BigQuery SQL single-quoted literal.

  Escapes single quotes (``'`` → ``\\'``) and newlines (``\\n`` → ``\\\\n``)
  so that the resulting SQL string literal is syntactically valid.
  """
  return value.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")


_EXTRACT_ONTOLOGY_AI_QUERY = """\
WITH session_transcripts AS (
  SELECT
    base.session_id,
    STRING_AGG(
      COALESCE(
        JSON_EXTRACT_SCALAR(base.content, '$.text_summary'),
        JSON_EXTRACT_SCALAR(base.content, '$.response'),
        JSON_EXTRACT_SCALAR(base.content, '$.text'),
        TO_JSON_STRING(base.content)
      ),
      '\\n---\\n'
      ORDER BY base.timestamp ASC
    ) AS transcript
  FROM `{project}.{dataset}.{table}` AS base
  WHERE base.session_id IN UNNEST(@session_ids)
    AND base.event_type IN (
      'LLM_RESPONSE',
      'TOOL_COMPLETED',
      'AGENT_COMPLETED',
      'HITL_CONFIRMATION_REQUEST_COMPLETED'
    )
    AND base.content IS NOT NULL
  GROUP BY base.session_id
)
SELECT
  st.session_id,
  REGEXP_REPLACE(
    REGEXP_REPLACE(
      AI.GENERATE(
        CONCAT(
          '{prompt}',
          '\\n',
          st.transcript
        ),
        endpoint => '{endpoint}',
        output_schema => 'graph_json STRING'
      ).graph_json,
      r'^```(?:json)?\\s*', ''),
    r'\\s*```$', '')
  AS graph_json
FROM session_transcripts AS st
"""

_EXTRACT_PAYLOADS_QUERY = """\
SELECT
  base.span_id,
  base.session_id,
  COALESCE(
    JSON_EXTRACT_SCALAR(base.content, '$.text_summary'),
    JSON_EXTRACT_SCALAR(base.content, '$.response'),
    JSON_EXTRACT_SCALAR(base.content, '$.text'),
    TO_JSON_STRING(base.content)
  ) AS payload_text
FROM `{project}.{dataset}.{table}` AS base
WHERE base.session_id IN UNNEST(@session_ids)
  AND base.event_type IN (
    'LLM_RESPONSE',
    'TOOL_COMPLETED',
    'AGENT_COMPLETED',
    'HITL_CONFIRMATION_REQUEST_COMPLETED'
  )
  AND base.content IS NOT NULL
ORDER BY base.timestamp ASC
"""

# ------------------------------------------------------------------ #
# Hydration                                                            #
# ------------------------------------------------------------------ #


def _hydrate_graph(
    spec: GraphSpec,
    raw_rows: list[dict],
) -> ExtractedGraph:
  """Hydrate raw AI.GENERATE JSON rows into an ``ExtractedGraph``.

  Each row contains ``session_id`` and ``graph_json`` (a JSON string
  with ``nodes`` and ``edges`` arrays).  One row per session is
  expected from the session-aggregated AI.GENERATE query.

  Node IDs are key-based: ``{session_id}:{entity_name}:{key1=v1,...}``,
  matching the edge reference scheme so that ``edge.from_node_id``
  resolves to a hydrated node's ``node_id``.  Key-based IDs require
  *all* primary key columns to be present; if any are missing (or the
  entity is unknown to the spec), the fallback index-based ID
  ``{session_id}:{entity_name}:{idx}`` is used instead.

  Duplicate nodes (same ``node_id``) are deduplicated: the last
  occurrence wins, keeping the most recent/complete data.

  Edge IDs: ``{session_id}:{relationship_name}:{idx}``.

  Args:
      spec: The ``GraphSpec`` used for extraction.
      raw_rows: List of dicts with ``session_id`` and
          ``graph_json`` keys.

  Returns:
      A merged ``ExtractedGraph`` with all nodes and edges.
  """
  entity_map = {e.name: e for e in spec.entities}
  seen_nodes: dict[str, ExtractedNode] = {}
  all_edges: list[ExtractedEdge] = []

  for row in raw_rows:
    session_id = row.get("session_id", "")
    raw_json = row.get("graph_json", "")

    if not raw_json:
      continue

    try:
      data = json.loads(raw_json)
    except (json.JSONDecodeError, TypeError):
      logger.debug("Could not parse graph JSON for session %s", session_id)
      continue

    if not isinstance(data, dict):
      logger.debug(
          "Expected dict from graph JSON, got %s for session %s",
          type(data).__name__,
          session_id,
      )
      continue

    # Hydrate nodes.
    for idx, raw_node in enumerate(data.get("nodes", [])):
      entity_name = raw_node.get("entity_name", "")
      entity_spec = entity_map.get(entity_name)
      labels = entity_spec.labels if entity_spec else [entity_name]

      node_id = _build_node_id(
          raw_node, entity_name, entity_spec, session_id, idx
      )
      props = []
      for key, value in raw_node.items():
        if key == "entity_name":
          continue
        props.append(ExtractedProperty(name=key, value=value))

      seen_nodes[node_id] = ExtractedNode(
          node_id=node_id,
          entity_name=entity_name,
          labels=labels,
          properties=props,
      )

    # Hydrate edges.
    for idx, raw_edge in enumerate(data.get("edges", [])):
      rel_name = raw_edge.get("relationship_name", "")
      edge_id = f"{session_id}:{rel_name}:{idx}"

      from_node_id = _build_edge_node_ref(raw_edge, "from", session_id)
      to_node_id = _build_edge_node_ref(raw_edge, "to", session_id)

      props = []
      skip_keys = {
          "relationship_name",
          "from_entity_name",
          "to_entity_name",
          "from_keys",
          "to_keys",
      }
      for key, value in raw_edge.items():
        if key in skip_keys:
          continue
        props.append(ExtractedProperty(name=key, value=value))

      all_edges.append(
          ExtractedEdge(
              edge_id=edge_id,
              relationship_name=rel_name,
              from_node_id=from_node_id,
              to_node_id=to_node_id,
              properties=props,
          )
      )

  return ExtractedGraph(
      name=spec.name,
      nodes=list(seen_nodes.values()),
      edges=all_edges,
  )


def _build_key_string(keys_obj: dict) -> str:
  """Build a sorted ``k1=v1,k2=v2`` string from a key-value dict."""
  if isinstance(keys_obj, dict) and keys_obj:
    return ",".join(f"{k}={v}" for k, v in sorted(keys_obj.items()))
  return ""


def _build_node_id(
    raw_node: dict,
    entity_name: str,
    entity_spec: Optional[EntitySpec],
    session_id: str,
    idx: int,
) -> str:
  """Build a deterministic node ID from primary key values.

  Uses the entity's primary key columns (from the spec) to extract
  key values from the raw node dict, producing IDs like
  ``{session_id}:{entity_name}:{key1=val1,key2=val2}``.

  Falls back to ``{session_id}:{entity_name}:{idx}`` when:
  - The entity is unknown to the spec
  - Primary key values are missing from the raw node
  """
  if entity_spec is not None:
    keys_obj = {}
    for col in entity_spec.keys.primary:
      if col not in raw_node:
        # Missing key column — fall back to index-based ID.
        return f"{session_id}:{entity_name}:{idx}"
      keys_obj[col] = raw_node[col]
    key_str = _build_key_string(keys_obj)
    if key_str:
      return f"{session_id}:{entity_name}:{key_str}"
  return f"{session_id}:{entity_name}:{idx}"


def _build_edge_node_ref(
    raw_edge: dict,
    direction: str,
    session_id: str,
) -> str:
  """Build a node reference string from an edge's key data.

  Uses ``from_keys``/``to_keys`` object to construct a deterministic
  reference like ``{session_id}:{entity_name}:{key1=val1,key2=val2}``.

  This matches the ID scheme used by ``_build_node_id``, so edge
  references resolve to hydrated node IDs in the same graph.
  """
  entity_name = raw_edge.get(f"{direction}_entity_name", "")
  keys_obj = raw_edge.get(f"{direction}_keys", {})
  key_str = _build_key_string(keys_obj)
  if key_str:
    return f"{session_id}:{entity_name}:{key_str}"
  return f"{session_id}:{entity_name}:unknown"


# ------------------------------------------------------------------ #
# OntologyGraphManager                                                 #
# ------------------------------------------------------------------ #


class OntologyGraphManager:
  """Configuration-driven graph extraction engine.

  Queries raw ADK telemetry from ``agent_events``, constructs
  ``AI.GENERATE`` queries using the compiled ontology schema and
  prompt, and hydrates the results into ``ExtractedGraph`` objects.

  Args:
      project_id: GCP project ID.
      dataset_id: BigQuery dataset ID.
      spec: A validated ``GraphSpec`` (from ``load_graph_spec``).
      table_id: Source telemetry table name.
      endpoint: AI.GENERATE model endpoint.
      location: BigQuery location.
      bq_client: Optional pre-configured BigQuery client.
  """

  def __init__(
      self,
      project_id: str,
      dataset_id: str,
      spec: GraphSpec,
      table_id: str = "agent_events",
      endpoint: str = "gemini-2.5-flash",
      location: Optional[str] = None,
      bq_client: Optional[bigquery.Client] = None,
  ) -> None:
    self.project_id = project_id
    self.dataset_id = dataset_id
    self.spec = spec
    self.table_id = table_id
    self.endpoint = endpoint
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

  def _resolve_endpoint(self) -> str:
    """Resolve the endpoint to a full Vertex AI URL.

    Reuses the same resolution logic as V3
    ``ContextGraphManager._resolve_endpoint``.

    Raises:
        ValueError: If the endpoint looks like a legacy BQ ML
            model reference.
    """
    ep = self.endpoint
    if ep.startswith("https://"):
      return ep
    if ep.count(".") >= 2:
      raise ValueError(
          f"Legacy BQ ML model reference '{ep}' is not supported "
          f"for AI.GENERATE. Use a Vertex AI model name "
          f"(e.g. 'gemini-2.5-flash') or a full endpoint URL."
      )
    return (
        f"https://aiplatform.googleapis.com/v1/projects/"
        f"{self.project_id}/locations/global/publishers/google/"
        f"models/{ep}"
    )

  def get_extraction_sql(
      self,
      session_ids: Optional[list[str]] = None,
  ) -> str:
    """Return the AI.GENERATE extraction SQL (for inspection).

    Args:
        session_ids: Ignored (the SQL uses a query parameter).
            Included for API symmetry.

    Returns:
        The formatted SQL string.
    """
    prompt = compile_extraction_prompt(self.spec)
    schema_hint = compile_output_schema(self.spec)
    full_prompt = prompt + "\n\nReturn a single JSON object:\n" + schema_hint
    return _EXTRACT_ONTOLOGY_AI_QUERY.format(
        prompt=_escape_sql_literal(full_prompt),
        endpoint=self._resolve_endpoint(),
        project=self.project_id,
        dataset=self.dataset_id,
        table=self.table_id,
    )

  def extract_graph(
      self,
      session_ids: list[str],
      use_ai_generate: bool = True,
  ) -> ExtractedGraph:
    """Extract a typed graph from agent telemetry.

    Args:
        session_ids: Sessions to extract from.
        use_ai_generate: If True, runs AI.GENERATE server-side.
            If False, fetches raw payloads (stub graph returned).

    Returns:
        An ``ExtractedGraph`` with nodes and edges.
    """
    if use_ai_generate:
      return self._extract_via_ai_generate(session_ids)
    return self._extract_payloads(session_ids)

  def _extract_via_ai_generate(self, session_ids: list[str]) -> ExtractedGraph:
    """Server-side extraction using AI.GENERATE with output_schema."""
    prompt = compile_extraction_prompt(self.spec)
    schema_hint = compile_output_schema(self.spec)
    full_prompt = prompt + "\n\nReturn a single JSON object:\n" + schema_hint

    query = _EXTRACT_ONTOLOGY_AI_QUERY.format(
        prompt=_escape_sql_literal(full_prompt),
        endpoint=self._resolve_endpoint(),
        project=self.project_id,
        dataset=self.dataset_id,
        table=self.table_id,
    )

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("session_ids", "STRING", session_ids),
        ]
    )

    try:
      job = self.bq_client.query(query, job_config=job_config)
      rows = [dict(row) for row in job.result()]
    except Exception as e:
      logger.warning("AI.GENERATE ontology extraction failed: %s", e)
      return ExtractedGraph(name=self.spec.name)

    return _hydrate_graph(self.spec, rows)

  def _extract_payloads(self, session_ids: list[str]) -> ExtractedGraph:
    """Fetch raw payloads without AI extraction (stub fallback)."""
    query = _EXTRACT_PAYLOADS_QUERY.format(
        project=self.project_id,
        dataset=self.dataset_id,
        table=self.table_id,
    )

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("session_ids", "STRING", session_ids),
        ]
    )

    try:
      job = self.bq_client.query(query, job_config=job_config)
      rows = list(job.result())
    except Exception as e:
      logger.warning("Payload extraction failed: %s", e)
      return ExtractedGraph(name=self.spec.name)

    # Return raw payloads as untyped nodes for client-side processing.
    nodes = []
    for idx, row in enumerate(rows):
      nodes.append(
          ExtractedNode(
              node_id=f"{row.get('session_id', '')}:{row.get('span_id', '')}:payload:{idx}",
              entity_name="raw_payload",
              labels=["raw_payload"],
              properties=[
                  ExtractedProperty(
                      name="payload_text",
                      value=row.get("payload_text", ""),
                  ),
                  ExtractedProperty(
                      name="session_id",
                      value=row.get("session_id", ""),
                  ),
                  ExtractedProperty(
                      name="span_id",
                      value=row.get("span_id", ""),
                  ),
              ],
          )
      )

    return ExtractedGraph(
        name=self.spec.name,
        nodes=nodes,
    )
