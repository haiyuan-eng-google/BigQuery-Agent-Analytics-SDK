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

"""Context Graph: Property Graph for Agent Trace + Business Entity linking.

This module provides the "System of Reasoning" for enterprise agents by
cross-linking the **Technical Graph** (execution lineage from the ADK
BigQuery Agent Analytics Plugin) with a **Business Graph** (domain
entities extracted via ``AI.GENERATE``).

Key capabilities:

- **Business entity extraction** — Use ``AI.GENERATE`` with
  ``output_schema`` to extract structured entities (e.g. Products,
  Targeting segments, Campaigns) from unstructured agent payloads.
- **Property Graph DDL** — Generate ``CREATE PROPERTY GRAPH`` DDL
  that formalizes Tech nodes, Biz nodes, ``CAUSED`` edges (parent→child
  span linkage), and ``EVALUATED`` cross-links.
- **GQL traversal** — Quantified-path GQL queries to answer "Why was
  X selected?" by tracing causal chains from a decision back to the
  business inputs.
- **World Change detection** — Compare business entities evaluated at
  agent-execution time against current availability to detect stale
  context in long-running A2A tasks.

Example usage::

    from bigquery_agent_analytics.context_graph import ContextGraphManager

    cgm = ContextGraphManager(
        project_id="my-project",
        dataset_id="agent_analytics",
    )

    # Extract business entities from agent traces
    biz_nodes = cgm.extract_biz_nodes(session_ids=["sess-1"])

    # Generate Property Graph DDL
    ddl = cgm.get_property_graph_ddl(graph_name="my_context_graph")

    # Traverse reasoning chains via GQL
    chain = cgm.explain_decision(
        decision_event_type="HITL_CONFIRMATION_REQUEST_COMPLETED",
        biz_entity="Yahoo Homepage",
    )
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timezone
import json
import logging
from typing import Any, Optional

from google.cloud import bigquery
from pydantic import BaseModel
from pydantic import Field

logger = logging.getLogger("bigquery_agent_analytics." + __name__)


# ------------------------------------------------------------------ #
# Data Models                                                          #
# ------------------------------------------------------------------ #


@dataclass
class BizNode:
  """A business-domain entity extracted from agent traces.

  Attributes:
      span_id: The span from which this entity was extracted.
      session_id: Session that produced this entity.
      node_type: Entity category (e.g. "Product", "Targeting",
          "Campaign", "Budget").
      node_value: Entity value (e.g. "Yahoo Homepage",
          "Millennials", "$8,000").
      confidence: Extraction confidence score (0.0-1.0).
      metadata: Additional extraction metadata.
  """

  span_id: str
  session_id: str
  node_type: str
  node_value: str
  confidence: float = 1.0
  metadata: dict[str, Any] = field(default_factory=dict)


class WorldChangeAlert(BaseModel):
  """An alert indicating a business entity has changed since evaluation.

  Attributes:
      biz_node: The business entity that changed.
      original_state: State at the time the agent evaluated it.
      current_state: Current state.
      drift_type: Type of drift (e.g. "unavailable",
          "price_changed", "inventory_depleted").
      severity: Drift severity (0.0-1.0).
      recommendation: Suggested action.
  """

  biz_node: str = Field(description="The business entity that changed.")
  original_state: str = Field(
      description="State when the agent evaluated it."
  )
  current_state: str = Field(description="Current state.")
  drift_type: str = Field(
      description="Type of drift detected."
  )
  severity: float = Field(
      description="Drift severity (0.0-1.0)."
  )
  recommendation: str = Field(
      default="Review before approving.",
      description="Suggested action.",
  )


class WorldChangeReport(BaseModel):
  """Report on world-state drift for a long-running agent task.

  Attributes:
      session_id: The session under review.
      alerts: List of detected world changes.
      total_entities_checked: Number of entities checked.
      stale_entities: Number of entities that drifted.
      is_safe_to_approve: Whether the context is still valid.
      checked_at: When the check was performed.
  """

  session_id: str = Field(description="Session under review.")
  alerts: list[WorldChangeAlert] = Field(default_factory=list)
  total_entities_checked: int = Field(default=0)
  stale_entities: int = Field(default=0)
  is_safe_to_approve: bool = Field(default=True)
  checked_at: datetime = Field(
      default_factory=lambda: datetime.now(timezone.utc)
  )

  model_config = {"arbitrary_types_allowed": True}

  def summary(self) -> str:
    """Returns a human-readable summary."""
    lines = [
        f"World Change Report — Session: {self.session_id}",
        f"  Entities checked : {self.total_entities_checked}",
        f"  Stale entities   : {self.stale_entities}",
        f"  Safe to approve  : {self.is_safe_to_approve}",
    ]
    for alert in self.alerts:
      lines.append(
          f"  [{alert.drift_type}] {alert.biz_node}: "
          f"{alert.original_state} -> {alert.current_state} "
          f"(severity={alert.severity:.2f})"
      )
    return "\n".join(lines)


class ContextGraphConfig(BaseModel):
  """Configuration for the Context Graph.

  Attributes:
      biz_nodes_table: Table name for extracted business nodes.
      cross_links_table: Table name for cross-link edges.
      graph_name: Name for the Property Graph.
      endpoint: AI.GENERATE endpoint for entity extraction.
      entity_types: Domain-specific entity types to extract.
      max_hops: Maximum causal hops for GQL traversal.
  """

  biz_nodes_table: str = Field(default="extracted_biz_nodes")
  cross_links_table: str = Field(default="context_cross_links")
  graph_name: str = Field(default="agent_context_graph")
  endpoint: str = Field(default="gemini-2.5-flash")
  entity_types: list[str] = Field(
      default_factory=lambda: [
          "Product",
          "Targeting",
          "Campaign",
          "Budget",
          "Audience",
          "Creative",
          "Placement",
      ]
  )
  max_hops: int = Field(default=20)


# ------------------------------------------------------------------ #
# SQL Templates                                                        #
# ------------------------------------------------------------------ #

_EXTRACT_BIZ_NODES_QUERY = """\
MERGE `{project}.{dataset}.{biz_table}` AS target
USING (
  SELECT
    CONCAT(base.span_id, ':', JSON_EXTRACT_SCALAR(entity, '$.entity_type'),
           ':', JSON_EXTRACT_SCALAR(entity, '$.entity_value')
    ) AS biz_node_id,
    base.span_id,
    base.session_id,
    JSON_EXTRACT_SCALAR(entity, '$.entity_type') AS node_type,
    JSON_EXTRACT_SCALAR(entity, '$.entity_value') AS node_value,
    CAST(
      COALESCE(JSON_EXTRACT_SCALAR(entity, '$.confidence'), '1.0')
      AS FLOAT64
    ) AS confidence
  FROM `{project}.{dataset}.{table}` AS base,
  UNNEST(JSON_EXTRACT_ARRAY(
    -- Strip markdown code fences (```json ... ```) from LLM output
    REGEXP_REPLACE(
      REGEXP_REPLACE(
        AI.GENERATE(
          CONCAT(
            'Extract business entities from this agent payload. ',
            'Entity types: {entity_types}. ',
            'Return ONLY a JSON array of objects with entity_type, ',
            'entity_value, and confidence (0-1). ',
            'No markdown, no explanation, just the JSON array.',
            '\\n\\nPayload:\\n',
            COALESCE(
              JSON_EXTRACT_SCALAR(base.content, '$.text_summary'),
              JSON_EXTRACT_SCALAR(base.content, '$.response'),
              JSON_EXTRACT_SCALAR(base.content, '$.text'),
              TO_JSON_STRING(base.content)
            )
          ),
          endpoint => '{endpoint}'
        ).result,
        r'^```(?:json)?\\s*', ''),
      r'\\s*```$', '')
  )) AS entity
  WHERE base.session_id IN UNNEST(@session_ids)
    AND base.event_type IN (
      'USER_MESSAGE_RECEIVED',
      'LLM_RESPONSE',
      'TOOL_COMPLETED',
      'AGENT_COMPLETED'
    )
    AND base.content IS NOT NULL
) AS source
ON target.biz_node_id = source.biz_node_id
WHEN MATCHED THEN
  UPDATE SET confidence = source.confidence
WHEN NOT MATCHED THEN
  INSERT (biz_node_id, span_id, session_id, node_type, node_value, confidence)
  VALUES (source.biz_node_id, source.span_id, source.session_id,
          source.node_type, source.node_value, source.confidence)
"""

_EXTRACT_BIZ_NODES_SIMPLE_QUERY = """\
SELECT
  base.span_id,
  base.session_id,
  base.event_type,
  COALESCE(
    JSON_EXTRACT_SCALAR(base.content, '$.text_summary'),
    JSON_EXTRACT_SCALAR(base.content, '$.response'),
    JSON_EXTRACT_SCALAR(base.content, '$.text'),
    TO_JSON_STRING(base.content)
  ) AS payload_text
FROM `{project}.{dataset}.{table}` AS base
WHERE base.session_id IN UNNEST(@session_ids)
  AND base.event_type IN (
    'USER_MESSAGE_RECEIVED',
    'LLM_RESPONSE',
    'TOOL_COMPLETED',
    'AGENT_COMPLETED'
  )
  AND base.content IS NOT NULL
ORDER BY base.timestamp ASC
"""

_CREATE_BIZ_NODES_TABLE_QUERY = """\
CREATE TABLE IF NOT EXISTS `{project}.{dataset}.{biz_table}` (
  biz_node_id STRING,
  span_id STRING,
  session_id STRING,
  node_type STRING,
  node_value STRING,
  confidence FLOAT64
)
"""

_INSERT_BIZ_NODES_QUERY = """\
INSERT INTO `{project}.{dataset}.{biz_table}`
  (biz_node_id, span_id, session_id, node_type, node_value, confidence)
VALUES
  {values}
"""

_CREATE_CROSS_LINKS_TABLE_QUERY = """\
CREATE TABLE IF NOT EXISTS `{project}.{dataset}.{cross_links_table}` (
  link_id STRING,
  span_id STRING,
  session_id STRING,
  biz_node_id STRING,
  node_value STRING,
  link_type STRING,
  created_at TIMESTAMP
)
"""

_DELETE_CROSS_LINKS_FOR_SESSIONS_QUERY = """\
DELETE FROM `{project}.{dataset}.{cross_links_table}`
WHERE session_id IN UNNEST(@session_ids)
"""

_INSERT_CROSS_LINKS_QUERY = """\
INSERT INTO `{project}.{dataset}.{cross_links_table}`
  (link_id, span_id, session_id, biz_node_id, node_value, link_type, created_at)
SELECT
  CONCAT(b.span_id, ':', b.node_value) AS link_id,
  b.span_id,
  b.session_id,
  b.biz_node_id,
  b.node_value,
  'EVALUATED' AS link_type,
  CURRENT_TIMESTAMP() AS created_at
FROM `{project}.{dataset}.{biz_table}` b
WHERE b.session_id IN UNNEST(@session_ids)
"""

_PROPERTY_GRAPH_DDL = """\
CREATE OR REPLACE PROPERTY GRAPH `{project}.{dataset}.{graph_name}`
  NODE TABLES (
    -- Technical execution nodes (spans from ADK plugin)
    `{project}.{dataset}.{table}` AS TechNode
      KEY (span_id)
      LABEL TechNode
      PROPERTIES (
        event_type,
        agent,
        timestamp,
        session_id,
        invocation_id,
        content,
        latency_ms,
        status,
        error_message
      ),
    -- Business domain nodes (extracted entities, keyed by composite ID)
    `{project}.{dataset}.{biz_table}` AS BizNode
      KEY (biz_node_id)
      LABEL BizNode
      PROPERTIES (
        node_type,
        node_value,
        confidence,
        session_id,
        span_id
      )
  )
  EDGE TABLES (
    -- Causal lineage: parent span -> child span
    `{project}.{dataset}.{table}` AS Caused
      KEY (span_id)
      SOURCE KEY (parent_span_id) REFERENCES TechNode (span_id)
      DESTINATION KEY (span_id) REFERENCES TechNode (span_id)
      LABEL Caused,

    -- Cross-link: technical event -> business entity it evaluated
    `{project}.{dataset}.{cross_links_table}` AS Evaluated
      KEY (link_id)
      SOURCE KEY (span_id) REFERENCES TechNode (span_id)
      DESTINATION KEY (biz_node_id) REFERENCES BizNode (biz_node_id)
      LABEL Evaluated
  )
"""

_GQL_REASONING_CHAIN_QUERY = """\
GRAPH `{project}.{dataset}.{graph_name}`
MATCH
  (decision:TechNode)-[c:Caused]->{{1,{max_hops}}}(step:TechNode)
    -[e:Evaluated]->(biz:BizNode)
WHERE decision.event_type = @decision_event_type
  {biz_filter_clause}
RETURN
  TO_JSON(decision) AS decision_node,
  decision.span_id AS decision_span_id,
  decision.event_type AS decision_type,
  step.span_id AS reasoning_span_id,
  step.event_type AS step_type,
  step.agent AS step_agent,
  COALESCE(
    JSON_EXTRACT_SCALAR(step.content, '$.text_summary'),
    JSON_EXTRACT_SCALAR(step.content, '$.response'),
    ''
  ) AS reasoning_text,
  step.latency_ms AS step_latency_ms,
  biz.node_type AS entity_type,
  biz.node_value AS entity_value,
  biz.confidence AS entity_confidence,
  TO_JSON(step) AS step_node,
  TO_JSON(biz) AS biz_node
ORDER BY step.timestamp ASC
LIMIT @result_limit
"""

_GQL_FULL_CAUSAL_CHAIN_QUERY = """\
GRAPH `{project}.{dataset}.{graph_name}`
MATCH
  (root:TechNode)-[c:Caused]->{{1,{max_hops}}}(leaf:TechNode)
WHERE root.session_id = @session_id
  AND root.event_type = 'USER_MESSAGE_RECEIVED'
RETURN
  TO_JSON(root) AS root_node,
  root.span_id AS root_span_id,
  leaf.span_id AS leaf_span_id,
  leaf.event_type AS leaf_event_type,
  leaf.agent AS leaf_agent,
  COALESCE(
    JSON_EXTRACT_SCALAR(leaf.content, '$.text_summary'),
    JSON_EXTRACT_SCALAR(leaf.content, '$.response'),
    ''
  ) AS leaf_content,
  leaf.latency_ms AS leaf_latency_ms,
  TO_JSON(leaf) AS leaf_node,
  TO_JSON(c) AS edge
ORDER BY leaf.timestamp ASC
LIMIT @result_limit
"""

_BIZ_NODES_FOR_SESSION_QUERY = """\
SELECT
  biz_node_id,
  node_type,
  node_value,
  confidence,
  span_id,
  session_id
FROM `{project}.{dataset}.{biz_table}`
WHERE session_id = @session_id
ORDER BY confidence DESC
"""

_WORLD_CHANGE_CHECK_QUERY = """\
SELECT
  b.node_type,
  b.node_value,
  b.confidence,
  b.span_id,
  e.timestamp AS evaluated_at
FROM `{project}.{dataset}.{biz_table}` b
JOIN `{project}.{dataset}.{table}` e
  ON b.span_id = e.span_id
WHERE b.session_id = @session_id
ORDER BY e.timestamp ASC
"""


# ------------------------------------------------------------------ #
# ContextGraphManager                                                  #
# ------------------------------------------------------------------ #


class ContextGraphManager:
  """Manages the Context Graph linking technical traces to business entities.

  This is the main entry point for building and querying the
  "System of Reasoning" Property Graph.

  Args:
      project_id: Google Cloud project ID.
      dataset_id: BigQuery dataset ID.
      table_id: Agent events table name.
      config: Optional context graph configuration.
      client: Optional BigQuery client instance.
  """

  def __init__(
      self,
      project_id: str,
      dataset_id: str,
      table_id: str = "agent_events",
      config: Optional[ContextGraphConfig] = None,
      client: Optional[bigquery.Client] = None,
      location: str = "US",
  ) -> None:
    self.project_id = project_id
    self.dataset_id = dataset_id
    self.table_id = table_id
    self.config = config or ContextGraphConfig()
    self._client = client
    self.location = location

  @property
  def client(self) -> bigquery.Client:
    """Lazily initializes the BigQuery client."""
    if self._client is None:
      self._client = bigquery.Client(
          project=self.project_id,
          location=self.location,
      )
    return self._client

  def _resolve_endpoint(self) -> str:
    """Resolves the AI.GENERATE endpoint to a full Vertex AI URL.

    Short model names like ``gemini-2.5-flash`` work for older models,
    but newer models (Gemini 3.x+) require the full Vertex AI endpoint
    URL.  This method converts short names to full URLs when necessary.
    """
    ep = self.config.endpoint
    if ep.startswith("https://"):
      return ep
    return (
        f"https://aiplatform.googleapis.com/v1/projects/"
        f"{self.project_id}/locations/global/publishers/google/"
        f"models/{ep}"
    )

  # -------------------------------------------------------------- #
  # Business Entity Extraction                                       #
  # -------------------------------------------------------------- #

  def extract_biz_nodes(
      self,
      session_ids: list[str],
      use_ai_generate: bool = True,
  ) -> list[BizNode]:
    """Extracts business entities from agent trace payloads.

    When *use_ai_generate* is True, runs the extraction as a
    BigQuery ``AI.GENERATE`` job that populates the biz nodes
    table directly.  When False, fetches payloads and returns
    them for client-side extraction.

    Args:
        session_ids: Sessions to extract entities from.
        use_ai_generate: Whether to use BigQuery AI.GENERATE
            (server-side) for extraction.

    Returns:
        List of extracted BizNode objects.
    """
    if use_ai_generate:
      return self._extract_via_ai_generate(session_ids)
    return self._extract_payloads_for_client(session_ids)

  def _ensure_biz_nodes_table(self) -> None:
    """Creates the biz_nodes table if it does not exist."""
    ddl = _CREATE_BIZ_NODES_TABLE_QUERY.format(
        project=self.project_id,
        dataset=self.dataset_id,
        biz_table=self.config.biz_nodes_table,
    )
    job = self.client.query(ddl)
    job.result()

  def _extract_via_ai_generate(
      self, session_ids: list[str]
  ) -> list[BizNode]:
    """Server-side extraction using AI.GENERATE with MERGE upsert."""
    self._ensure_biz_nodes_table()

    entity_types_str = ", ".join(self.config.entity_types)
    query = _EXTRACT_BIZ_NODES_QUERY.format(
        project=self.project_id,
        dataset=self.dataset_id,
        table=self.table_id,
        biz_table=self.config.biz_nodes_table,
        endpoint=self._resolve_endpoint(),
        entity_types=entity_types_str,
    )

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter(
                "session_ids", "STRING", session_ids
            ),
        ]
    )

    try:
      job = self.client.query(query, job_config=job_config)
      job.result()
      logger.info(
          "AI.GENERATE extraction complete — results in %s.%s",
          self.dataset_id,
          self.config.biz_nodes_table,
      )
    except Exception as e:
      logger.warning("AI.GENERATE extraction failed: %s", e)
      return []

    return self._read_biz_nodes(session_ids)

  def _extract_payloads_for_client(
      self, session_ids: list[str]
  ) -> list[BizNode]:
    """Fetches payloads for client-side entity extraction."""
    query = _EXTRACT_BIZ_NODES_SIMPLE_QUERY.format(
        project=self.project_id,
        dataset=self.dataset_id,
        table=self.table_id,
    )

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter(
                "session_ids", "STRING", session_ids
            ),
        ]
    )

    try:
      job = self.client.query(query, job_config=job_config)
      rows = list(job.result())
      nodes = []
      for row in rows:
        nodes.append(
            BizNode(
                span_id=row.get("span_id", ""),
                session_id=row.get("session_id", ""),
                node_type="raw_payload",
                node_value=row.get("payload_text", ""),
            )
        )
      return nodes
    except Exception as e:
      logger.warning("Payload extraction failed: %s", e)
      return []

  def _read_biz_nodes(
      self, session_ids: list[str]
  ) -> list[BizNode]:
    """Reads extracted biz nodes from the output table."""
    query = f"""\
    SELECT span_id, session_id, node_type, node_value, confidence
    FROM `{self.project_id}.{self.dataset_id}.{self.config.biz_nodes_table}`
    WHERE session_id IN UNNEST(@session_ids)
    ORDER BY confidence DESC
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter(
                "session_ids", "STRING", session_ids
            ),
        ]
    )

    try:
      job = self.client.query(query, job_config=job_config)
      rows = list(job.result())
      return [
          BizNode(
              span_id=row.get("span_id", ""),
              session_id=row.get("session_id", ""),
              node_type=row.get("node_type", ""),
              node_value=row.get("node_value", ""),
              confidence=float(row.get("confidence", 1.0)),
          )
          for row in rows
      ]
    except Exception as e:
      logger.warning("Failed to read biz nodes: %s", e)
      return []

  def store_biz_nodes(self, nodes: list[BizNode]) -> bool:
    """Stores pre-extracted business nodes into BigQuery.

    Use this when entities are extracted client-side (e.g. via
    the Gemini API directly) rather than through AI.GENERATE.

    Args:
        nodes: List of BizNode objects to store.

    Returns:
        True if successful.
    """
    if not nodes:
      return True

    create_query = _CREATE_BIZ_NODES_TABLE_QUERY.format(
        project=self.project_id,
        dataset=self.dataset_id,
        biz_table=self.config.biz_nodes_table,
    )

    try:
      job = self.client.query(create_query)
      job.result()
    except Exception as e:
      logger.warning("Failed to create biz nodes table: %s", e)
      return False

    rows = [
        {
            "biz_node_id": f"{n.span_id}:{n.node_type}:{n.node_value}",
            "span_id": n.span_id,
            "session_id": n.session_id,
            "node_type": n.node_type,
            "node_value": n.node_value,
            "confidence": n.confidence,
        }
        for n in nodes
    ]

    table_ref = (
        f"{self.project_id}.{self.dataset_id}"
        f".{self.config.biz_nodes_table}"
    )

    try:
      errors = self.client.insert_rows_json(table_ref, rows)
      if errors:
        logger.error("Failed to insert biz nodes: %s", errors)
        return False
      logger.info("Stored %d biz nodes", len(nodes))
      return True
    except Exception as e:
      logger.warning("Failed to store biz nodes: %s", e)
      return False

  # -------------------------------------------------------------- #
  # Cross-Link Generation                                            #
  # -------------------------------------------------------------- #

  def create_cross_links(
      self, session_ids: list[str]
  ) -> bool:
    """Creates EVALUATED edges linking TechNodes to BizNodes.

    Args:
        session_ids: Sessions to create cross-links for.

    Returns:
        True if successful.
    """
    create_query = _CREATE_CROSS_LINKS_TABLE_QUERY.format(
        project=self.project_id,
        dataset=self.dataset_id,
        cross_links_table=self.config.cross_links_table,
    )

    try:
      job = self.client.query(create_query)
      job.result()
    except Exception as e:
      logger.warning("Failed to create cross-links table: %s", e)
      return False

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter(
                "session_ids", "STRING", session_ids
            ),
        ]
    )

    # Delete existing cross-links for these sessions (idempotent)
    try:
      delete_query = _DELETE_CROSS_LINKS_FOR_SESSIONS_QUERY.format(
          project=self.project_id,
          dataset=self.dataset_id,
          cross_links_table=self.config.cross_links_table,
      )
      job = self.client.query(delete_query, job_config=job_config)
      job.result()
    except Exception:
      pass  # table may be empty or not exist yet

    insert_query = _INSERT_CROSS_LINKS_QUERY.format(
        project=self.project_id,
        dataset=self.dataset_id,
        biz_table=self.config.biz_nodes_table,
        cross_links_table=self.config.cross_links_table,
    )

    try:
      job = self.client.query(insert_query, job_config=job_config)
      job.result()
      logger.info("Cross-links created for %d sessions", len(session_ids))
      return True
    except Exception as e:
      logger.warning("Failed to create cross-links: %s", e)
      return False

  # -------------------------------------------------------------- #
  # Property Graph DDL                                               #
  # -------------------------------------------------------------- #

  def get_property_graph_ddl(
      self,
      graph_name: Optional[str] = None,
  ) -> str:
    """Returns the CREATE PROPERTY GRAPH DDL statement.

    Args:
        graph_name: Override the default graph name.

    Returns:
        The DDL SQL string.
    """
    name = graph_name or self.config.graph_name
    return _PROPERTY_GRAPH_DDL.format(
        project=self.project_id,
        dataset=self.dataset_id,
        table=self.table_id,
        biz_table=self.config.biz_nodes_table,
        cross_links_table=self.config.cross_links_table,
        graph_name=name,
    )

  def create_property_graph(
      self,
      graph_name: Optional[str] = None,
  ) -> bool:
    """Creates the Property Graph in BigQuery.

    Args:
        graph_name: Override the default graph name.

    Returns:
        True if successful.
    """
    ddl = self.get_property_graph_ddl(graph_name)
    try:
      job = self.client.query(ddl)
      job.result()
      logger.info(
          "Property Graph '%s' created",
          graph_name or self.config.graph_name,
      )
      return True
    except Exception as e:
      logger.warning("Failed to create Property Graph: %s", e)
      return False

  # -------------------------------------------------------------- #
  # GQL Traversal                                                    #
  # -------------------------------------------------------------- #

  def get_reasoning_chain_gql(
      self,
      decision_event_type: str = "HITL_CONFIRMATION_REQUEST_COMPLETED",
      biz_entity: Optional[str] = None,
      graph_name: Optional[str] = None,
      max_hops: Optional[int] = None,
      result_limit: int = 100,
  ) -> str:
    """Returns a GQL query for reasoning chain traversal.

    Traces causal hops from a decision event back to the business
    entities that informed it.

    Args:
        decision_event_type: The terminal event type to trace from.
        biz_entity: Optional specific business entity to filter.
        graph_name: Override graph name.
        max_hops: Override max causal hops.
        result_limit: Maximum results to return.

    Returns:
        The GQL query string.
    """
    name = graph_name or self.config.graph_name
    hops = max_hops or self.config.max_hops

    biz_filter_clause = ""
    if biz_entity:
      biz_filter_clause = "AND biz.node_value = @biz_entity"

    return _GQL_REASONING_CHAIN_QUERY.format(
        project=self.project_id,
        dataset=self.dataset_id,
        graph_name=name,
        max_hops=hops,
        biz_filter_clause=biz_filter_clause,
    )

  def explain_decision(
      self,
      decision_event_type: str = "HITL_CONFIRMATION_REQUEST_COMPLETED",
      biz_entity: Optional[str] = None,
      graph_name: Optional[str] = None,
      max_hops: Optional[int] = None,
      result_limit: int = 100,
  ) -> list[dict[str, Any]]:
    """Traverses the context graph to explain a decision.

    Answers "Why was X selected?" by following causal chains
    from a decision back to the business inputs.

    Args:
        decision_event_type: The terminal decision event type.
        biz_entity: Optional specific entity to explain.
        graph_name: Override graph name.
        max_hops: Override max causal hops.
        result_limit: Maximum results.

    Returns:
        List of reasoning chain steps as dicts.
    """
    query = self.get_reasoning_chain_gql(
        decision_event_type=decision_event_type,
        biz_entity=biz_entity,
        graph_name=graph_name,
        max_hops=max_hops,
        result_limit=result_limit,
    )

    params = [
        bigquery.ScalarQueryParameter(
            "decision_event_type", "STRING", decision_event_type
        ),
        bigquery.ScalarQueryParameter(
            "result_limit", "INT64", result_limit
        ),
    ]
    if biz_entity:
      params.append(
          bigquery.ScalarQueryParameter(
              "biz_entity", "STRING", biz_entity
          )
      )

    job_config = bigquery.QueryJobConfig(query_parameters=params)

    try:
      job = self.client.query(query, job_config=job_config)
      rows = list(job.result())
      return [dict(row) for row in rows]
    except Exception as e:
      logger.warning("GQL reasoning chain query failed: %s", e)
      return []

  def get_causal_chain_gql(
      self,
      session_id: str,
      graph_name: Optional[str] = None,
      max_hops: Optional[int] = None,
      result_limit: int = 200,
  ) -> str:
    """Returns a GQL query for the full causal chain of a session.

    Args:
        session_id: Session to traverse.
        graph_name: Override graph name.
        max_hops: Override max hops.
        result_limit: Maximum results.

    Returns:
        The GQL query string.
    """
    name = graph_name or self.config.graph_name
    hops = max_hops or self.config.max_hops
    return _GQL_FULL_CAUSAL_CHAIN_QUERY.format(
        project=self.project_id,
        dataset=self.dataset_id,
        graph_name=name,
        max_hops=hops,
    )

  def traverse_causal_chain(
      self,
      session_id: str,
      graph_name: Optional[str] = None,
      max_hops: Optional[int] = None,
      result_limit: int = 200,
  ) -> list[dict[str, Any]]:
    """Traverses the full causal chain for a session.

    Args:
        session_id: Session to traverse.
        graph_name: Override graph name.
        max_hops: Override max hops.
        result_limit: Maximum results.

    Returns:
        List of chain steps as dicts.
    """
    query = self.get_causal_chain_gql(
        session_id=session_id,
        graph_name=graph_name,
        max_hops=max_hops,
        result_limit=result_limit,
    )

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(
                "session_id", "STRING", session_id
            ),
            bigquery.ScalarQueryParameter(
                "result_limit", "INT64", result_limit
            ),
        ]
    )

    try:
      job = self.client.query(query, job_config=job_config)
      rows = list(job.result())
      return [dict(row) for row in rows]
    except Exception as e:
      logger.warning("GQL causal chain query failed: %s", e)
      return []

  # -------------------------------------------------------------- #
  # World Change Detection                                           #
  # -------------------------------------------------------------- #

  def get_biz_nodes_for_session(
      self, session_id: str
  ) -> list[BizNode]:
    """Returns all business entities evaluated in a session.

    Args:
        session_id: Session to query.

    Returns:
        List of BizNode objects.
    """
    query = _BIZ_NODES_FOR_SESSION_QUERY.format(
        project=self.project_id,
        dataset=self.dataset_id,
        biz_table=self.config.biz_nodes_table,
    )

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(
                "session_id", "STRING", session_id
            ),
        ]
    )

    try:
      job = self.client.query(query, job_config=job_config)
      rows = list(job.result())
      return [
          BizNode(
              span_id=row.get("span_id", ""),
              session_id=row.get("session_id", ""),
              node_type=row.get("node_type", ""),
              node_value=row.get("node_value", ""),
              confidence=float(row.get("confidence", 1.0)),
          )
          for row in rows
      ]
    except Exception as e:
      logger.warning(
          "Failed to get biz nodes for session %s: %s",
          session_id, e,
      )
      return []

  def detect_world_changes(
      self,
      session_id: str,
      current_state_fn: Any = None,
  ) -> WorldChangeReport:
    """Checks if business entities have drifted since evaluation.

    This implements the "World Change" detection pattern for
    long-running A2A tasks. Before a HITL approval is finalized,
    this method traverses the context graph to find the original
    BizNodes, queries current availability via *current_state_fn*,
    and reports any drift.

    Args:
        session_id: The session to check.
        current_state_fn: A callable that takes a BizNode and returns
            a dict with keys ``available`` (bool), ``current_value``
            (str), and optionally ``drift_type`` (str).  If None,
            no drift checks are performed and a safe report is
            returned with the entity count.

    Returns:
        WorldChangeReport with alerts for any detected drift.
    """
    nodes = self.get_biz_nodes_for_session(session_id)

    alerts: list[WorldChangeAlert] = []
    stale_count = 0

    for node in nodes:
      if current_state_fn is not None:
        try:
          state = current_state_fn(node)
        except Exception as e:
          logger.warning(
              "World state check failed for %s: %s",
              node.node_value, e,
          )
          continue

        if not state.get("available", True):
          stale_count += 1
          alerts.append(
              WorldChangeAlert(
                  biz_node=node.node_value,
                  original_state=f"{node.node_type}: {node.node_value}",
                  current_state=state.get(
                      "current_value", "unavailable"
                  ),
                  drift_type=state.get(
                      "drift_type", "unavailable"
                  ),
                  severity=state.get("severity", 0.8),
                  recommendation=state.get(
                      "recommendation",
                      "Review before approving.",
                  ),
              )
          )

    return WorldChangeReport(
        session_id=session_id,
        alerts=alerts,
        total_entities_checked=len(nodes),
        stale_entities=stale_count,
        is_safe_to_approve=(stale_count == 0),
    )

  # -------------------------------------------------------------- #
  # Pipeline: End-to-End                                             #
  # -------------------------------------------------------------- #

  def build_context_graph(
      self,
      session_ids: list[str],
      graph_name: Optional[str] = None,
      use_ai_generate: bool = True,
  ) -> dict[str, Any]:
    """End-to-end pipeline: extract, cross-link, and create graph.

    Runs all three steps in sequence:
    1. Extract business entities from traces
    2. Create cross-link edges
    3. Create the Property Graph

    Args:
        session_ids: Sessions to include.
        graph_name: Override graph name.
        use_ai_generate: Use AI.GENERATE for extraction.

    Returns:
        Dict with results of each step.
    """
    results: dict[str, Any] = {}

    # Step 1: Extract
    nodes = self.extract_biz_nodes(
        session_ids, use_ai_generate=use_ai_generate
    )
    results["biz_nodes_count"] = len(nodes)
    results["biz_nodes"] = nodes

    # Step 2: Cross-links
    cross_link_ok = self.create_cross_links(session_ids)
    results["cross_links_created"] = cross_link_ok

    # Step 3: Property Graph
    graph_ok = self.create_property_graph(graph_name)
    results["property_graph_created"] = graph_ok

    return results
