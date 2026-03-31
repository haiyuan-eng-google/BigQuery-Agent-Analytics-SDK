# Ontology Graph V4 — Configuration-Driven Context Graph

**Issue:** [#52 — Ontology graph demo v4](https://github.com/haiyuan-eng-google/BigQuery-Agent-Analytics-SDK/issues/52)

## Project Objective

Build an end-to-end pipeline that reads a logical YAML graph specification,
uses `AI.GENERATE` to extract unstructured ADK telemetry into typed
`ExtractedGraph` Pydantic models, and dynamically generates the BigQuery
Property Graph DDL.

---

## Step 1: Define the YMGO YAML Specification

**Task:** Create a YAML configuration file (`ymgo_graph_spec.yaml`) that
merges the Brainstorming Graph YAML syntax with Yahoo's YMGO v2 ontology.

*   **Instructions for Engineer:**
    *   Define `entities` (Nodes) representing YMGO classes like
        `DecisionPoint` and `Candidate`.
    *   Use the `binding.source` field to define the physical BigQuery table
        routing, mimicking Yahoo's `ymgo:bigqueryTable` annotations.
    *   Define `relationships` (Edges) like `CandidateEdge` with
        `from_entity` and `to_entity`.

**Example YAML:**
```yaml
graph:
  name: YMGO_Context_Graph
  entities:
    - name: DecisionPoint
      binding:
        source: project.dataset.decision_points
      properties:
        - name: decision_type
          type: string
    - name: Candidate
      binding:
        source: project.dataset.candidates
      properties:
        - name: score
          type: double
        - name: rejection_rationale
          type: string
  relationships:
    - name: CandidateEdge
      from_entity: DecisionPoint
      to_entity: Candidate
      binding:
        source: project.dataset.candidate_edges
```

---

## Step 2: Integrate the Pydantic `ExtractedGraph` Models

**Task:** Use the provided Pydantic models as the strict data container for
the extracted graph instances.

*   **Instructions for Engineer:**
    *   Import the `ExtractedGraph`, `ExtractedNode`, `ExtractedEdge`, and
        `ExtractedProperty` models exactly as defined.
    *   Write a Python parser that reads the `ymgo_graph_spec.yaml` file
        and generates a JSON schema representation of the `ExtractedGraph`
        Pydantic model. BigQuery's `AI.GENERATE` function requires a strict
        JSON `output_schema`. By translating the Pydantic model schema into
        this format, we force Gemini to output its extractions in a
        structure that perfectly deserializes into our Python models.

---

## Step 3: Dynamic Semantic Extraction Pipeline

**Task:** Implement the extraction logic that converts unstructured ADK
event telemetry into populated Pydantic objects.

*   **Instructions for Engineer:**
    *   Query the raw unstructured agent traces from the `agent_events`
        table.
    *   Construct an `AI.GENERATE` SQL query. The prompt should instruct
        the LLM to "Extract Decision Points and Candidates according to
        the provided ontology".
    *   Pass the JSON schema generated in Step 2 into the `output_schema`
        parameter.
    *   When the BigQuery results return, hydrate the LLM's output into
        the strongly-typed Python objects.

---

## Step 4: Automated Table Routing (Physical Binding)

**Task:** Route the extracted Pydantic `ExtractedNode` and `ExtractedEdge`
objects into their respective physical BigQuery tables using the YAML
bindings.

*   **Instructions for Engineer:**
    *   Iterate through `ExtractedGraph.nodes` and `ExtractedGraph.edges`.
    *   Look up the entity `name` in the loaded YAML configuration to find
        its `binding.source` table.
    *   Generate and execute BigQuery `INSERT` statements to write the
        properties of each Node and Edge into their dedicated tables (e.g.,
        inserting `DecisionPoint` nodes into
        `project.dataset.decision_points`).
    *   Use delete-then-insert for session-scoped idempotency.
    *   When multiple spec entries share the same `binding.source`,
        group all rows per physical table before persistence to avoid
        delete-then-insert races.

---

## Step 5: Dynamic Property Graph DDL Generation

**Task:** Transpile the YAML relationships into a native BigQuery
`CREATE PROPERTY GRAPH` statement.

*   **Instructions for Engineer:**
    *   Write a transpiler function that loops through the `entities` in
        the YAML to generate the `NODE TABLES` block of the DDL,
        referencing the bound tables and properties.
    *   Loop through the `relationships` in the YAML to generate the
        `EDGE TABLES` block. Map the `from_entity` to the `SOURCE KEY`
        and the `to_entity` to the `DESTINATION KEY`.
    *   Execute the generated DDL against BigQuery to instantiate the
        formal graph.

---

## Step 6: End-to-End Validation (The "Showcase" Query)

**Task:** Prove the pipeline works by running a native Graph Query
Language (GQL) query against the dynamically generated graph.

*   **Instructions for Engineer:**
    *   Provide a test script that executes a GQL forward traversal:
        `MATCH (dp:DecisionPoint)-[ce:CandidateEdge]->(cand:Candidate)`.
    *   Assert that the query returns the extracted candidate scores and
        rejection rationales, proving that the unstructured telemetry was
        successfully converted into a strongly-typed audit graph.

---

## Detailed YAML Ontology (from issue comment)

```yaml
graph:
  name: YMGO_Context_Graph_V3

  # ==========================================
  # 1. ENTITIES (Nodes)
  # ==========================================
  entities:
    - name: mako_DecisionPoint
      description: "The atomic unit of decisioning where an agent evaluates alternatives."
      binding:
        source: "{{ env }}.decision_points"
      keys:
        primary: [decision_id]
      properties:
        - name: decision_id
          type: string
        - name: decision_type
          type: string

    - name: sup_YahooAdUnit
      extends: mako_Candidate
      description: "A specific ad slot on a Yahoo property being evaluated as a candidate."
      binding:
        source: "{{ env }}.yahoo_ad_units"
      keys:
        primary: [adUnitId]
      properties:
        - name: adUnitId
          type: string
        - name: adUnitName
          type: string
        - name: adUnitSize
          type: string
          description: "e.g., '300x250', '728x90'"
        - name: adUnitPosition
          type: string
          description: "ATF (above the fold) | BTF (below the fold)"

    - name: mako_RejectionReason
      description: "Structured reason why a Candidate was not selected at a DecisionPoint."
      binding:
        source: "{{ env }}.rejection_reasons"
      keys:
        primary: [rejection_id]
      properties:
        - name: rejection_id
          type: string
        - name: rejectionType
          type: string
          description: "RULE_BASED | MODEL_BASED | TIMEOUT | ERROR"
        - name: rejectionRule
          type: string
          description: "The specific rule or model threshold that caused rejection."

  # ==========================================
  # 2. RELATIONSHIPS (Edges)
  # ==========================================
  relationships:
    - name: CandidateEdge
      description: "Connects a decision point to the evaluated Yahoo Ad Unit."
      from_entity: mako_DecisionPoint
      to_entity: sup_YahooAdUnit
      binding:
        source: "{{ env }}.candidate_edges"
        from_columns: [decision_id]
        to_columns: [adUnitId]
      properties:
        - name: edge_type
          type: string
          description: "SELECTED_CANDIDATE or DROPPED_CANDIDATE"
        - name: mako_scoreValue
          type: double
          description: "The confidence or predicted Q-value for this candidate."

    - name: ForCandidate
      description: "Maps the MAKO rejection rationale directly to the dropped candidate."
      from_entity: mako_RejectionReason
      to_entity: sup_YahooAdUnit
      binding:
        source: "{{ env }}.rejection_mappings"
        from_columns: [rejection_id]
        to_columns: [adUnitId]
```

### Why this is more realistic

1. **Inheritance & Ontology Alignment:** It demonstrates logical inheritance
   (`extends: mako_Candidate`), proving how business items like
   `sup_YahooAdUnit` naturally act as evaluable candidates in the graph.
2. **Explicit Table Bindings:** The `binding.source` accurately reflects
   the real-world physical table structure declared in the
   `ymgo:spannerTable` annotations in the TTL file.
3. **Primary & Foreign Key Routing:** The `relationships` block uses
   `from_columns` and `to_columns` to explicitly tell the transpiler
   exactly how to construct the `SOURCE KEY` and `DESTINATION KEY` for
   the BigQuery Property Graph DDL.

---

## Implementation Plan

### Three Layers

1. **`GraphSpec`** — YAML ontology definition (Pydantic config models)
2. **`ExtractedGraph`** — hydrated AI extraction result (runtime instances)
3. **Property Graph** — generated BigQuery graph DDL

### Scope Decisions

- `extends` is **label-only inheritance** in V4. No property or binding
  inheritance.
- `{{ env }}` substitution uses simple string replacement (no Jinja2).
- CLI exposure is added per-phase as needed.
- V4 sits beside V3 `ContextGraphManager`, not as a replacement.

### Phase Delivery

| Phase | Deliverable | Status |
|-------|-------------|--------|
| 1 | YAML Spec + Pydantic Models | Merged (PR #53) |
| 2 | Spec Loader + Schema Compiler | Merged (PR #54) |
| 3 | Ontology Extraction Engine | Merged (PR #55) |
| 4 | Physical Table Materialization + Routing | Merged (PR #56) |
| 5 | Dynamic Property Graph DDL Transpiler | Merged (PR #57) |
| 6 | Showcase Query Path + Orchestrator | Merged (PR #58) |

---

### Phase 5: Dynamic Property Graph DDL Transpiler

**Transpiler rules:**

- Each YAML entity becomes a `NODE TABLES` entry.
- Each relationship becomes an `EDGE TABLES` entry.
- Use YAML `binding.source` for physical table references.
- Use `keys.primary` for node keys.
- Use relationship `from_columns` / `to_columns` for source/destination
  keys.
- Reuse the style of existing V3 DDL generation in `context_graph.py`,
  but make it data-driven.

**Expected DDL shape:**

```sql
CREATE OR REPLACE PROPERTY GRAPH `project.dataset.YMGO_Context_Graph`
  NODE TABLES (
    `project.dataset.decision_points` AS mako_DecisionPoint
      KEY (decision_id, session_id)
      LABEL mako_DecisionPoint
      PROPERTIES (decision_type, extracted_at),
    `project.dataset.yahoo_ad_units` AS sup_YahooAdUnit
      KEY (adUnitId, session_id)
      LABEL sup_YahooAdUnit
      LABEL mako_Candidate
      PROPERTIES (adUnitName, adUnitSize, adUnitPosition, extracted_at),
    `project.dataset.rejection_reasons` AS mako_RejectionReason
      KEY (rejection_id, session_id)
      LABEL mako_RejectionReason
      PROPERTIES (rejectionType, rejectionRule, extracted_at)
  )
  EDGE TABLES (
    `project.dataset.candidate_edges` AS CandidateEdge
      KEY (decision_id, adUnitId, session_id)
      SOURCE KEY (decision_id, session_id) REFERENCES mako_DecisionPoint (decision_id, session_id)
      DESTINATION KEY (adUnitId, session_id) REFERENCES sup_YahooAdUnit (adUnitId, session_id)
      LABEL CandidateEdge
      PROPERTIES (edge_type, mako_scoreValue, extracted_at),
    `project.dataset.rejection_mappings` AS ForCandidate
      KEY (rejection_id, adUnitId, session_id)
      SOURCE KEY (rejection_id, session_id) REFERENCES mako_RejectionReason (rejection_id, session_id)
      DESTINATION KEY (adUnitId, session_id) REFERENCES sup_YahooAdUnit (adUnitId, session_id)
      LABEL ForCandidate
      PROPERTIES (extracted_at)
  )
```

> **Note:** Node KEY includes `session_id` so that the same business entity
> in different sessions produces distinct graph nodes, making multi-session
> builds safe.

**Deliverables:**

- `src/bigquery_agent_analytics/ontology_property_graph.py`
  - `compile_node_table_clause(entity, project_id, dataset_id)` → SQL
    fragment
  - `compile_edge_table_clause(rel, spec, project_id, dataset_id)` → SQL
    fragment
  - `compile_property_graph_ddl(spec, project_id, dataset_id,
    graph_name=None)` → full DDL
  - `OntologyPropertyGraphCompiler` class
- `tests/test_ontology_property_graph.py`
- Wire into `__init__.py` and CLI

---

### Phase 6: Showcase Query Path

- High-level orchestrator:
  `build_ontology_graph(session_ids, spec_path, graph_name=None)`
- GQL showcase query:
  `MATCH (dp:mako_DecisionPoint)-[ce:CandidateEdge]->(ad:sup_YahooAdUnit)`
- Demo: `examples/ontology_graph_v4_demo.html`

---

## Acceptance Criteria

- A YAML spec can define entities, relationships, bindings, keys, and
  inheritance.
- The SDK can load that spec and resolve it deterministically.
- ADK telemetry can be extracted into typed `ExtractedGraph` objects.
- Extracted nodes/edges are routed to the correct physical BigQuery tables.
- The SDK can generate and execute `CREATE PROPERTY GRAPH` from the YAML.
- A showcase GQL query returns expected candidate scores and rejection
  rationale.
