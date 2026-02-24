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

"""BigQuery Agent Analytics SDK Client.

The ``Client`` class is the primary entry point for the SDK. It
abstracts BigQuery SQL complexity and provides clean Python interfaces
for trace reconstruction, evaluation, and feedback loop curation.

Example usage::

    from bigquery_agent_analytics import Client

    client = Client(
        project_id="my-project",
        dataset_id="agent_analytics",
    )

    # Retrieve and visualize a trace
    trace = client.get_trace("trace-123")
    trace.render()

    # Run evaluation
    from bigquery_agent_analytics import (
        CodeEvaluator, LLMAsJudge, TraceFilter,
    )
    report = client.evaluate(
        filters=TraceFilter(agent_id="my_agent"),
        evaluator=CodeEvaluator.latency(threshold_ms=3000),
    )
    print(report.summary())
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from typing import Optional

from google.cloud import bigquery

from .evaluators import _parse_json_from_text
from .evaluators import AI_GENERATE_JUDGE_BATCH_QUERY
from .evaluators import CodeEvaluator
from .evaluators import DEFAULT_ENDPOINT
from .evaluators import EvaluationReport
from .evaluators import LLM_JUDGE_BATCH_QUERY
from .evaluators import LLMAsJudge
from .evaluators import SESSION_SUMMARY_QUERY
from .evaluators import SessionScore
from .feedback import AnalysisConfig
from .feedback import compute_drift
from .feedback import compute_question_distribution
from .feedback import DriftReport
from .feedback import QuestionDistribution
from .insights import _AI_GENERATE_FACET_EXTRACTION_QUERY
from .insights import _FACET_EXTRACTION_QUERY
from .insights import _SESSION_METADATA_QUERY
from .insights import _SESSION_TRANSCRIPT_QUERY
from .insights import aggregate_facets
from .insights import ANALYSIS_PROMPTS
from .insights import build_analysis_context
from .insights import build_facet_prompt
from .insights import extract_facets_via_api
from .insights import generate_executive_summary
from .insights import InsightsConfig
from .insights import InsightsReport
from .insights import parse_facet_from_ai_generate_row
from .insights import parse_facet_response
from .insights import run_analysis_prompt
from .insights import SessionFacet
from .insights import SessionMetadata
from .trace import Span
from .trace import Trace
from .trace import TraceFilter

logger = logging.getLogger("bigquery_agent_analytics." + __name__)


# ------------------------------------------------------------------ #
# SQL Templates                                                        #
# ------------------------------------------------------------------ #

_GET_TRACE_QUERY = """\
SELECT
  event_type,
  agent,
  timestamp,
  session_id,
  invocation_id,
  user_id,
  trace_id,
  span_id,
  parent_span_id,
  content,
  content_parts,
  attributes,
  latency_ms,
  status,
  error_message,
  is_truncated
FROM `{project}.{dataset}.{table}`
WHERE trace_id = @trace_id
ORDER BY timestamp ASC
"""

_LIST_TRACES_QUERY = """\
WITH trace_sessions AS (
  SELECT DISTINCT session_id
  FROM `{project}.{dataset}.{table}`
  WHERE {where}
  LIMIT @trace_limit
)
SELECT
  e.event_type,
  e.agent,
  e.timestamp,
  e.session_id,
  e.invocation_id,
  e.user_id,
  e.trace_id,
  e.span_id,
  e.parent_span_id,
  e.content,
  e.content_parts,
  e.attributes,
  e.latency_ms,
  e.status,
  e.error_message,
  e.is_truncated
FROM `{project}.{dataset}.{table}` e
JOIN trace_sessions ts ON e.session_id = ts.session_id
ORDER BY e.session_id, e.timestamp ASC
"""

_VERIFY_SCHEMA_QUERY = """\
SELECT column_name, data_type
FROM `{project}.{dataset}.INFORMATION_SCHEMA.COLUMNS`
WHERE table_name = @table_name
"""

_REQUIRED_COLUMNS = {
    "timestamp",
    "event_type",
    "session_id",
    "content",
}


# ------------------------------------------------------------------ #
# Client                                                               #
# ------------------------------------------------------------------ #


class Client:
  """BigQuery Agent Analytics SDK client.

  Provides a high-level Python interface for analyzing agent traces
  stored in BigQuery. Abstracts away SQL complexity, UNNEST
  operations, and BQML mechanics.

  Args:
      project_id: Google Cloud project ID.
      dataset_id: BigQuery dataset containing agent events.
      table_id: Table name for agent events.
      location: BigQuery dataset location.
      gcs_bucket_name: Optional GCS bucket for resolving offloaded
          multimodal payloads.
      verify_schema: Whether to verify the table schema on init.
      endpoint: AI.GENERATE endpoint (default gemini-2.5-flash).
          Pass a fully-qualified BQ ML model reference
          (``project.dataset.model``) to use legacy
          ``ML.GENERATE_TEXT`` instead.
      connection_id: Optional BigQuery connection resource ID
          for AI.GENERATE.
  """

  def __init__(
      self,
      project_id: str,
      dataset_id: str,
      table_id: str = "agent_events_v2",
      location: str = "us-central1",
      gcs_bucket_name: Optional[str] = None,
      verify_schema: bool = True,
      bq_client: Optional[bigquery.Client] = None,
      endpoint: Optional[str] = None,
      connection_id: Optional[str] = None,
  ) -> None:
    self.project_id = project_id
    self.dataset_id = dataset_id
    self.table_id = table_id
    self.location = location
    self.gcs_bucket_name = gcs_bucket_name
    self._bq_client = bq_client
    self._table_ref = f"{project_id}.{dataset_id}.{table_id}"
    self.endpoint = endpoint or DEFAULT_ENDPOINT
    self.connection_id = connection_id

    if verify_schema:
      self._verify_schema()

  @property
  def bq_client(self) -> bigquery.Client:
    """Lazily initializes the BigQuery client."""
    if self._bq_client is None:
      self._bq_client = bigquery.Client(
          project=self.project_id,
          location=self.location,
      )
    return self._bq_client

  # -------------------------------------------------------------- #
  # Schema Verification                                              #
  # -------------------------------------------------------------- #

  def _verify_schema(self) -> None:
    """Verifies the target table has expected columns."""
    try:
      query = _VERIFY_SCHEMA_QUERY.format(
          project=self.project_id,
          dataset=self.dataset_id,
      )
      job_config = bigquery.QueryJobConfig(
          query_parameters=[
              bigquery.ScalarQueryParameter(
                  "table_name",
                  "STRING",
                  self.table_id,
              ),
          ]
      )
      results = list(
          self.bq_client.query(query, job_config=job_config).result()
      )
      columns = {r.get("column_name") for r in results}

      missing = _REQUIRED_COLUMNS - columns
      if missing:
        logger.warning(
            "Table %s is missing columns: %s. Some SDK features may not work.",
            self._table_ref,
            missing,
        )
    except Exception as e:
      logger.warning(
          "Schema verification failed: %s. Continuing without verification.",
          e,
      )

  # -------------------------------------------------------------- #
  # Trace Retrieval                                                  #
  # -------------------------------------------------------------- #

  def get_trace(self, trace_id: str) -> Trace:
    """Fetches all spans for a specific trace.

    Automatically resolves GCS-offloaded payloads if
    ``gcs_bucket_name`` was provided during initialization.

    Args:
        trace_id: The trace ID to retrieve.

    Returns:
        A Trace object with all spans.

    Raises:
        ValueError: If no events found for the trace ID.
    """
    query = _GET_TRACE_QUERY.format(
        project=self.project_id,
        dataset=self.dataset_id,
        table=self.table_id,
    )
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(
                "trace_id",
                "STRING",
                trace_id,
            ),
        ]
    )

    results = list(self.bq_client.query(query, job_config=job_config).result())

    if not results:
      raise ValueError(f"No events found for trace_id={trace_id}")

    spans = [Span.from_bigquery_row(dict(row)) for row in results]

    # Determine trace metadata
    user_id = None
    session_id = None
    for row in results:
      if not user_id:
        user_id = row.get("user_id")
      if not session_id:
        session_id = row.get("session_id")

    timestamps = [s.timestamp for s in spans if s.timestamp]
    start = min(timestamps) if timestamps else None
    end = max(timestamps) if timestamps else None
    total_ms = None
    if start and end:
      total_ms = (end - start).total_seconds() * 1000

    return Trace(
        trace_id=trace_id,
        session_id=session_id or "",
        spans=spans,
        user_id=user_id,
        start_time=start,
        end_time=end,
        total_latency_ms=total_ms,
    )

  def list_traces(
      self,
      filter_criteria: Optional[TraceFilter] = None,
  ) -> list[Trace]:
    """Lists traces matching the given filter criteria.

    Args:
        filter_criteria: Optional filter. If None, returns
            recent traces (default limit 100).

    Returns:
        List of Trace objects, one per session.
    """
    filt = filter_criteria or TraceFilter()
    where, params = filt.to_sql_conditions()

    query = _LIST_TRACES_QUERY.format(
        project=self.project_id,
        dataset=self.dataset_id,
        table=self.table_id,
        where=where,
    )
    job_config = bigquery.QueryJobConfig(
        query_parameters=params,
    )

    results = list(self.bq_client.query(query, job_config=job_config).result())

    # Group by session
    sessions: dict[str, list[Span]] = {}
    meta: dict[str, dict[str, Any]] = {}
    for row in results:
      row_dict = dict(row)
      sid = row_dict.get("session_id", "unknown")
      span = Span.from_bigquery_row(row_dict)
      sessions.setdefault(sid, []).append(span)
      if sid not in meta:
        meta[sid] = {
            "user_id": row_dict.get("user_id"),
            "trace_id": row_dict.get("trace_id"),
        }

    traces = []
    for sid, spans in sessions.items():
      timestamps = [s.timestamp for s in spans if s.timestamp]
      start = min(timestamps) if timestamps else None
      end = max(timestamps) if timestamps else None
      total_ms = None
      if start and end:
        total_ms = (end - start).total_seconds() * 1000

      traces.append(
          Trace(
              trace_id=meta[sid].get("trace_id") or sid,
              session_id=sid,
              spans=spans,
              user_id=meta[sid].get("user_id"),
              start_time=start,
              end_time=end,
              total_latency_ms=total_ms,
          )
      )

    return traces

  # -------------------------------------------------------------- #
  # Evaluation                                                       #
  # -------------------------------------------------------------- #

  def evaluate(
      self,
      evaluator: CodeEvaluator | LLMAsJudge,
      filters: Optional[TraceFilter] = None,
      dataset: Optional[str] = None,
      golden_dataset: Optional[str] = None,
  ) -> EvaluationReport:
    """Runs batch evaluation over traces.

    Uses BigQuery native execution for scalable assessment.
    ``CodeEvaluator`` metrics are computed from session
    aggregates. ``LLMAsJudge`` metrics use BQML's
    ``ML.GENERATE_TEXT`` for zero-ETL evaluation.

    Args:
        evaluator: A CodeEvaluator or LLMAsJudge instance.
        filters: Optional trace filters.
        dataset: Optional table name override.
        golden_dataset: Optional golden dataset table for
            comparison evaluation.

    Returns:
        EvaluationReport with per-session and aggregate scores.
    """
    table = dataset or self.table_id
    filt = filters or TraceFilter()
    where, params = filt.to_sql_conditions()

    if isinstance(evaluator, CodeEvaluator):
      return self._evaluate_code(
          evaluator,
          table,
          where,
          params,
      )
    elif isinstance(evaluator, LLMAsJudge):
      return self._evaluate_llm_judge(
          evaluator,
          table,
          where,
          params,
          filt,
      )
    else:
      raise TypeError(f"Unsupported evaluator type: {type(evaluator)}")

  def _evaluate_code(
      self,
      evaluator: CodeEvaluator,
      table: str,
      where: str,
      params: list,
  ) -> EvaluationReport:
    """Runs code-based evaluation using session summaries."""
    query = SESSION_SUMMARY_QUERY.format(
        project=self.project_id,
        dataset=self.dataset_id,
        table=table,
        where=where,
    )
    job_config = bigquery.QueryJobConfig(
        query_parameters=params,
    )

    results = list(self.bq_client.query(query, job_config=job_config).result())

    session_scores = []
    for row in results:
      summary = dict(row)
      score = evaluator.evaluate_session(summary)
      session_scores.append(score)

    return _build_report(
        evaluator_name=evaluator.name,
        dataset=f"{self._table_ref} WHERE {where}",
        session_scores=session_scores,
    )

  @staticmethod
  def _is_legacy_model_ref(ref: str) -> bool:
    """Returns True when *ref* looks like a BQ ML model reference.

    Legacy model references have the form
    ``project.dataset.model_name`` (two or more dots).
    """
    return ref.count(".") >= 2

  def _evaluate_llm_judge(
      self,
      evaluator: LLMAsJudge,
      table: str,
      where: str,
      params: list,
      trace_filter: Optional[TraceFilter] = None,
  ) -> EvaluationReport:
    """Runs LLM-as-judge evaluation.

    Attempts AI.GENERATE first, then legacy ML.GENERATE_TEXT,
    then falls back to the Gemini API.
    """
    # Try AI.GENERATE (new path) when endpoint is not a legacy ref
    if not self._is_legacy_model_ref(self.endpoint):
      for criterion in evaluator._criteria:
        try:
          return self._ai_generate_judge(
              evaluator,
              criterion,
              table,
              where,
              params,
          )
        except Exception as e:
          logger.debug(
              "AI.GENERATE judge failed, trying legacy: %s",
              e,
          )

    # Try legacy BQML batch evaluation
    text_model = (
        self.endpoint
        if self._is_legacy_model_ref(self.endpoint)
        else (f"{self.project_id}.{self.dataset_id}.gemini_text_model")
    )

    for criterion in evaluator._criteria:
      try:
        return self._bqml_judge(
            evaluator,
            criterion,
            table,
            where,
            params,
            text_model,
        )
      except Exception as e:
        logger.debug(
            "BQML judge failed, falling back to API: %s",
            e,
        )

    # Fallback: fetch traces, evaluate via API
    return self._api_judge(evaluator, table, where, params, trace_filter)

  def _ai_generate_judge(
      self,
      evaluator,
      criterion,
      table,
      where,
      params,
  ) -> EvaluationReport:
    """Evaluates using BigQuery AI.GENERATE with typed output."""
    from google.cloud import bigquery as bq

    judge_params = list(params) + [
        bq.ScalarQueryParameter(
            "judge_prompt",
            "STRING",
            criterion.prompt_template.split("{trace_text}")[0],
        ),
    ]

    query = AI_GENERATE_JUDGE_BATCH_QUERY.format(
        project=self.project_id,
        dataset=self.dataset_id,
        table=table,
        where=where,
        endpoint=self.endpoint,
    )
    job_config = bq.QueryJobConfig(
        query_parameters=judge_params,
    )

    results = list(self.bq_client.query(query, job_config=job_config).result())

    session_scores = []
    for row in results:
      sid = row.get("session_id", "unknown")
      raw_score = row.get("score")
      justification = row.get("justification", "")

      scores: dict[str, float] = {}
      if raw_score is not None:
        scores[criterion.name] = max(
            0.0,
            min(1.0, float(raw_score) / 10.0),
        )

      passed = all(s >= criterion.threshold for s in scores.values())
      session_scores.append(
          SessionScore(
              session_id=sid,
              scores=scores,
              passed=passed,
              llm_feedback=justification,
          )
      )

    return _build_report(
        evaluator_name=evaluator.name,
        dataset=f"{self._table_ref} WHERE {where}",
        session_scores=session_scores,
    )

  def _bqml_judge(
      self,
      evaluator,
      criterion,
      table,
      where,
      params,
      text_model,
  ) -> EvaluationReport:
    """Evaluates using BigQuery ML.GENERATE_TEXT."""
    from google.cloud import bigquery as bq

    judge_params = list(params) + [
        bq.ScalarQueryParameter(
            "judge_prompt",
            "STRING",
            criterion.prompt_template.split("{trace_text}")[0],
        ),
    ]

    query = LLM_JUDGE_BATCH_QUERY.format(
        project=self.project_id,
        dataset=self.dataset_id,
        table=table,
        where=where,
        model=text_model,
    )
    job_config = bq.QueryJobConfig(
        query_parameters=judge_params,
    )

    results = list(self.bq_client.query(query, job_config=job_config).result())

    session_scores = []
    for row in results:
      sid = row.get("session_id", "unknown")
      eval_text = row.get("evaluation", "")
      parsed = _parse_json_from_text(eval_text or "")

      scores: dict[str, float] = {}
      if parsed and criterion.score_key in parsed:
        raw = float(parsed[criterion.score_key])
        scores[criterion.name] = raw / 10.0
      elif parsed:
        for k, v in parsed.items():
          if isinstance(v, (int, float)):
            scores[k] = float(v) / 10.0

      passed = all(s >= criterion.threshold for s in scores.values())
      session_scores.append(
          SessionScore(
              session_id=sid,
              scores=scores,
              passed=passed,
              llm_feedback=(
                  parsed.get("justification", "") if parsed else eval_text
              ),
          )
      )

    return _build_report(
        evaluator_name=evaluator.name,
        dataset=f"{self._table_ref} WHERE {where}",
        session_scores=session_scores,
    )

  def _api_judge(
      self,
      evaluator,
      table,
      where,
      params,
      trace_filter: Optional[TraceFilter] = None,
  ) -> EvaluationReport:
    """Evaluates using the Gemini API (fallback)."""
    traces = self.list_traces(trace_filter or TraceFilter(limit=100))

    loop = asyncio.new_event_loop()
    try:
      session_scores = loop.run_until_complete(
          self._run_api_judge(evaluator, traces)
      )
    finally:
      loop.close()

    return _build_report(
        evaluator_name=evaluator.name,
        dataset=f"{self._table_ref} WHERE {where}",
        session_scores=session_scores,
    )

  async def _run_api_judge(
      self,
      evaluator: LLMAsJudge,
      traces: list[Trace],
  ) -> list[SessionScore]:
    """Runs LLM judge via API for each trace."""
    scores = []
    for trace in traces:
      trace_lines = []
      for span in trace.spans:
        trace_lines.append(f"{span.event_type}: {span.summary}")
      trace_text = "\n".join(trace_lines)
      final = trace.final_response or ""

      score = await evaluator.evaluate_session(
          trace_text,
          final,
      )
      score.session_id = trace.session_id
      scores.append(score)

    return scores

  # -------------------------------------------------------------- #
  # Feedback & Curation                                              #
  # -------------------------------------------------------------- #

  def drift_detection(
      self,
      golden_dataset: str,
      filters: Optional[TraceFilter] = None,
      dataset: Optional[str] = None,
      embedding_model: Optional[str] = None,
  ) -> DriftReport:
    """Detects drift between golden dataset and production.

    Compares golden questions against production traces to
    determine coverage percentage and identify gaps.

    Args:
        golden_dataset: Table name containing golden questions
            (must have a ``question`` column).
        filters: Optional filters for production traces.
        dataset: Optional events table override.
        embedding_model: Optional model for semantic matching.

    Returns:
        DriftReport with coverage metrics.
    """
    table = dataset or self.table_id
    filt = filters or TraceFilter()
    where, params = filt.to_sql_conditions()

    loop = asyncio.new_event_loop()
    try:
      return loop.run_until_complete(
          compute_drift(
              bq_client=self.bq_client,
              project_id=self.project_id,
              dataset_id=self.dataset_id,
              table_id=table,
              golden_table=golden_dataset,
              where_clause=where,
              query_params=params,
              embedding_model=embedding_model,
          )
      )
    finally:
      loop.close()

  # -------------------------------------------------------------- #
  # Insights                                                         #
  # -------------------------------------------------------------- #

  def insights(
      self,
      filters: Optional[TraceFilter] = None,
      config: Optional[InsightsConfig] = None,
      dataset: Optional[str] = None,
      text_model: Optional[str] = None,
  ) -> InsightsReport:
    """Generates a comprehensive insights report.

    Runs a multi-stage pipeline:
    1. Session filtering and metadata extraction.
    2. Per-session facet extraction via LLM.
    3. Aggregation across sessions.
    4. Multi-prompt analysis.
    5. Executive summary generation.

    Args:
        filters: Optional trace filters.
        config: Insights configuration. Defaults to
            analyzing up to 50 recent sessions.
        dataset: Optional events table override.
        text_model: Optional BQML text model.

    Returns:
        InsightsReport with facets, analysis, and summary.
    """
    loop = asyncio.new_event_loop()
    try:
      return loop.run_until_complete(
          self._run_insights(
              filters=filters,
              config=config,
              dataset=dataset,
              text_model=text_model,
          )
      )
    finally:
      loop.close()

  async def _run_insights(
      self,
      filters: Optional[TraceFilter] = None,
      config: Optional[InsightsConfig] = None,
      dataset: Optional[str] = None,
      text_model: Optional[str] = None,
  ) -> InsightsReport:
    """Async implementation of the insights pipeline."""
    table = dataset or self.table_id
    filt = filters or TraceFilter()
    cfg = config or InsightsConfig()
    model = text_model or self.endpoint

    where, params = filt.to_sql_conditions()

    # Step 1: Extract session metadata
    metadata_list = await self._fetch_session_metadata(
        table,
        where,
        params,
        cfg,
    )

    if not metadata_list:
      return InsightsReport(config=cfg)

    session_ids = [m.session_id for m in metadata_list]

    # Step 2: Extract facets
    facets = await self._extract_facets(
        table,
        session_ids,
        model,
    )

    # Step 3: Aggregate
    agg = aggregate_facets(facets, metadata_list)

    # Step 4: Multi-prompt analysis
    context = build_analysis_context(
        agg,
        facets,
        metadata_list,
    )
    prompt_names = cfg.analysis_prompts or list(ANALYSIS_PROMPTS.keys())
    sections = []
    for name in prompt_names:
      section = await run_analysis_prompt(
          name,
          context,
          model="gemini-2.5-flash",
      )
      sections.append(section)

    # Step 5: Executive summary
    report = InsightsReport(
        config=cfg,
        session_facets=facets,
        session_metadata=metadata_list,
        aggregated=agg,
        analysis_sections=sections,
    )
    report.executive_summary = await generate_executive_summary(report)

    return report

  async def _fetch_session_metadata(
      self,
      table: str,
      where: str,
      params: list,
      config: InsightsConfig,
  ) -> list[SessionMetadata]:
    """Fetches session metadata from BigQuery."""
    from google.cloud import bigquery as bq

    loop = asyncio.get_event_loop()

    extra_params = list(params) + [
        bq.ScalarQueryParameter(
            "min_events",
            "INT64",
            config.min_events_per_session,
        ),
        bq.ScalarQueryParameter(
            "min_turns",
            "INT64",
            config.min_turns_per_session,
        ),
        bq.ScalarQueryParameter(
            "max_sessions",
            "INT64",
            config.max_sessions,
        ),
    ]

    query = _SESSION_METADATA_QUERY.format(
        project=self.project_id,
        dataset=self.dataset_id,
        table=table,
        where=where,
    )
    job_config = bq.QueryJobConfig(
        query_parameters=extra_params,
    )

    job = await loop.run_in_executor(
        None,
        lambda: self.bq_client.query(query, job_config=job_config),
    )
    rows = await loop.run_in_executor(
        None,
        lambda: list(job.result()),
    )

    result = []
    for row in rows:
      r = dict(row)
      result.append(
          SessionMetadata(
              session_id=r.get("session_id", ""),
              event_count=r.get("event_count", 0),
              tool_calls=r.get("tool_calls", 0),
              tool_errors=r.get("tool_errors", 0),
              llm_calls=r.get("llm_calls", 0),
              turn_count=r.get("turn_count", 0),
              total_latency_ms=float(r.get("total_latency_ms") or 0),
              avg_latency_ms=float(r.get("avg_latency_ms") or 0),
              agents_used=r.get("agents_used") or [],
              tools_used=r.get("tools_used") or [],
              has_error=bool(r.get("has_error")),
              start_time=r.get("start_time"),
              end_time=r.get("end_time"),
          )
      )
    return result

  async def _extract_facets(
      self,
      table: str,
      session_ids: list[str],
      text_model: str,
  ) -> list[SessionFacet]:
    """Extracts facets via AI.GENERATE, BQML, or API fallback."""
    # Try AI.GENERATE first (when not a legacy model ref)
    if not self._is_legacy_model_ref(self.endpoint):
      try:
        return await self._extract_facets_ai_generate(
            table,
            session_ids,
        )
      except Exception as e:
        logger.debug(
            "AI.GENERATE facet extraction failed: %s",
            e,
        )

    # Try legacy BQML batch extraction
    try:
      return await self._extract_facets_bqml(
          table,
          session_ids,
          text_model,
      )
    except Exception as e:
      logger.debug(
          "BQML facet extraction failed, falling back to API: %s",
          e,
      )

    # Fallback: fetch transcripts, extract via API
    transcripts = await self._fetch_transcripts(
        table,
        session_ids,
    )
    return await extract_facets_via_api(transcripts)

  async def _extract_facets_ai_generate(
      self,
      table: str,
      session_ids: list[str],
  ) -> list[SessionFacet]:
    """Extracts facets using AI.GENERATE with typed output."""
    from google.cloud import bigquery as bq

    loop = asyncio.get_event_loop()

    facet_prompt = build_facet_prompt()
    query = _AI_GENERATE_FACET_EXTRACTION_QUERY.format(
        project=self.project_id,
        dataset=self.dataset_id,
        table=table,
        endpoint=self.endpoint,
    )
    job_config = bq.QueryJobConfig(
        query_parameters=[
            bq.ArrayQueryParameter(
                "session_ids",
                "STRING",
                session_ids,
            ),
            bq.ScalarQueryParameter(
                "facet_prompt",
                "STRING",
                facet_prompt,
            ),
        ],
    )

    job = await loop.run_in_executor(
        None,
        lambda: self.bq_client.query(query, job_config=job_config),
    )
    rows = await loop.run_in_executor(
        None,
        lambda: list(job.result()),
    )

    facets = []
    for row in rows:
      r = dict(row)
      sid = r.get("session_id", "")
      facets.append(parse_facet_from_ai_generate_row(sid, r))
    return facets

  async def _extract_facets_bqml(
      self,
      table: str,
      session_ids: list[str],
      text_model: str,
  ) -> list[SessionFacet]:
    """Extracts facets using legacy ML.GENERATE_TEXT."""
    from google.cloud import bigquery as bq

    loop = asyncio.get_event_loop()

    facet_prompt = build_facet_prompt()
    query = _FACET_EXTRACTION_QUERY.format(
        project=self.project_id,
        dataset=self.dataset_id,
        table=table,
        model=text_model,
    )
    job_config = bq.QueryJobConfig(
        query_parameters=[
            bq.ArrayQueryParameter(
                "session_ids",
                "STRING",
                session_ids,
            ),
            bq.ScalarQueryParameter(
                "facet_prompt",
                "STRING",
                facet_prompt,
            ),
        ],
    )

    job = await loop.run_in_executor(
        None,
        lambda: self.bq_client.query(query, job_config=job_config),
    )
    rows = await loop.run_in_executor(
        None,
        lambda: list(job.result()),
    )

    facets = []
    for row in rows:
      r = dict(row)
      sid = r.get("session_id", "")
      raw = r.get("facets_json", "")
      facets.append(parse_facet_response(sid, raw or ""))
    return facets

  async def _fetch_transcripts(
      self,
      table: str,
      session_ids: list[str],
  ) -> dict[str, str]:
    """Fetches session transcripts from BigQuery."""
    from google.cloud import bigquery as bq

    loop = asyncio.get_event_loop()

    query = _SESSION_TRANSCRIPT_QUERY.format(
        project=self.project_id,
        dataset=self.dataset_id,
        table=table,
    )
    job_config = bq.QueryJobConfig(
        query_parameters=[
            bq.ArrayQueryParameter(
                "session_ids",
                "STRING",
                session_ids,
            ),
        ],
    )

    job = await loop.run_in_executor(
        None,
        lambda: self.bq_client.query(query, job_config=job_config),
    )
    rows = await loop.run_in_executor(
        None,
        lambda: list(job.result()),
    )

    return {
        dict(row).get("session_id", ""): dict(row).get("transcript", "")
        for row in rows
    }

  def deep_analysis(
      self,
      filters: Optional[TraceFilter] = None,
      configuration: Optional[AnalysisConfig] = None,
      dataset: Optional[str] = None,
      text_model: Optional[str] = None,
  ) -> QuestionDistribution:
    """Performs deep analysis of question distribution.

    Supports modes: ``frequently_asked``,
    ``frequently_unanswered``,
    ``auto_group_using_semantics``, or custom categories.

    Args:
        filters: Optional filters for production traces.
        configuration: Analysis configuration. Defaults to
            ``auto_group_using_semantics``.
        dataset: Optional events table override.
        text_model: Optional BQML text model for classification.

    Returns:
        QuestionDistribution with categorized results.
    """
    table = dataset or self.table_id
    filt = filters or TraceFilter()
    where, params = filt.to_sql_conditions()
    config = configuration or AnalysisConfig()

    model = text_model or self.endpoint

    loop = asyncio.new_event_loop()
    try:
      return loop.run_until_complete(
          compute_question_distribution(
              bq_client=self.bq_client,
              project_id=self.project_id,
              dataset_id=self.dataset_id,
              table_id=table,
              where_clause=where,
              query_params=params,
              config=config,
              text_model=model,
          )
      )
    finally:
      loop.close()


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #


def _build_report(
    evaluator_name: str,
    dataset: str,
    session_scores: list[SessionScore],
) -> EvaluationReport:
  """Builds an EvaluationReport from session scores."""
  total = len(session_scores)
  passed = sum(1 for s in session_scores if s.passed)
  failed = total - passed

  # Aggregate scores
  agg: dict[str, list[float]] = {}
  for ss in session_scores:
    for name, score in ss.scores.items():
      agg.setdefault(name, []).append(score)

  aggregate = {
      name: sum(vals) / len(vals) for name, vals in agg.items() if vals
  }

  return EvaluationReport(
      dataset=dataset,
      evaluator_name=evaluator_name,
      total_sessions=total,
      passed_sessions=passed,
      failed_sessions=failed,
      aggregate_scores=aggregate,
      session_scores=session_scores,
  )
