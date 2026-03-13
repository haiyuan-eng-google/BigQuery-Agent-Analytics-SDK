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

"""Core dispatch logic for the BigQuery Remote Function.

This module contains the business logic (dispatch routing, evaluator
factories, filter construction) that is independent of Flask /
functions-framework so it can be tested without those dependencies.
"""

from __future__ import annotations

import json
import os
from typing import Any

from bigquery_agent_analytics import Client
from bigquery_agent_analytics import CodeEvaluator
from bigquery_agent_analytics import LLMAsJudge
from bigquery_agent_analytics import serialize
from bigquery_agent_analytics import TraceFilter


def build_client_from_context(
    user_defined_context: dict[str, Any],
) -> Client:
  """Build a Client from userDefinedContext + env vars."""
  udc = user_defined_context
  project_id = udc.get("project_id", os.environ.get("BQ_AGENT_PROJECT"))
  dataset_id = udc.get("dataset_id", os.environ.get("BQ_AGENT_DATASET"))
  table_id = udc.get(
      "table_id",
      os.environ.get("BQ_AGENT_TABLE", "agent_events"),
  )
  location = udc.get(
      "location",
      os.environ.get("BQ_AGENT_LOCATION", "us-central1"),
  )
  endpoint = udc.get("endpoint") or os.environ.get("BQ_AGENT_ENDPOINT")
  connection_id = udc.get("connection_id") or os.environ.get(
      "BQ_AGENT_CONNECTION_ID"
  )

  if not project_id or not dataset_id:
    raise ValueError("project_id and dataset_id required")

  return Client(
      project_id=project_id,
      dataset_id=dataset_id,
      table_id=table_id,
      location=location,
      verify_schema=False,
      endpoint=endpoint,
      connection_id=connection_id,
  )


def process_calls(
    client: Client,
    calls: list[list[Any]],
) -> list[str]:
  """Process a batch of Remote Function calls.

  Args:
      client: An initialized SDK Client.
      calls: List of [operation, params_json] pairs.

  Returns:
      List of JSON strings, one per call (partial failure safe).
  """
  replies = []
  for call in calls:
    try:
      operation, params_json = call[0], call[1]
      params = (
          json.loads(params_json)
          if isinstance(params_json, str)
          else params_json
      )
      result = dispatch(client, operation, params)
      result["_version"] = "1.0"
      replies.append(json.dumps(result))
    except Exception as e:
      replies.append(
          json.dumps(
              {
                  "_error": {
                      "code": type(e).__name__,
                      "message": str(e),
                  },
                  "_version": "1.0",
              }
          )
      )
  return replies


def dispatch(client, operation, params):
  """Route operation to SDK method, return JSON-safe dict."""
  if operation == "analyze":
    trace = client.get_session_trace(params["session_id"])
    return serialize(trace)

  if operation == "evaluate":
    evaluator = build_evaluator(params)
    filters = build_filters(params)
    report = client.evaluate(evaluator=evaluator, filters=filters)
    return serialize(report)

  if operation == "judge":
    judge = build_judge(params)
    filters = build_filters(params)
    report = client.evaluate(evaluator=judge, filters=filters)
    return serialize(report)

  if operation == "insights":
    filters = build_filters(params)
    report = client.insights(filters=filters)
    return serialize(report)

  if operation == "drift":
    golden_dataset = params.get("golden_dataset")
    if not golden_dataset:
      raise ValueError("drift operation requires 'golden_dataset' param")
    filters = build_filters(params)
    report = client.drift_detection(
        golden_dataset=golden_dataset,
        filters=filters,
    )
    return serialize(report)

  raise ValueError(f"Unknown operation: {operation!r}")


def build_filters(params):
  """Build TraceFilter from params dict."""
  return TraceFilter.from_cli_args(
      session_id=params.get("session_id"),
      agent_id=params.get("agent_filter"),
      last=params.get("last"),
      limit=params.get("limit", 100),
  )


def build_evaluator(params):
  """Build CodeEvaluator from params dict."""
  metric = params.get("metric", "latency")
  threshold = params.get("threshold")

  factories_with_t = {
      "latency": lambda t: CodeEvaluator.latency(threshold_ms=t),
      "error_rate": lambda t: CodeEvaluator.error_rate(
          max_error_rate=t,
      ),
      "turn_count": lambda t: CodeEvaluator.turn_count(
          max_turns=int(t),
      ),
      "token_efficiency": lambda t: CodeEvaluator.token_efficiency(
          max_tokens=int(t),
      ),
      "ttft": lambda t: CodeEvaluator.ttft(threshold_ms=t),
      "cost": lambda t: CodeEvaluator.cost_per_session(
          max_cost_usd=t,
      ),
  }
  factories_default = {
      "latency": CodeEvaluator.latency,
      "error_rate": CodeEvaluator.error_rate,
      "turn_count": CodeEvaluator.turn_count,
      "token_efficiency": CodeEvaluator.token_efficiency,
      "ttft": CodeEvaluator.ttft,
      "cost": CodeEvaluator.cost_per_session,
  }

  if metric not in factories_with_t:
    raise ValueError(f"Unknown metric: {metric!r}")

  if threshold is not None:
    return factories_with_t[metric](threshold)
  return factories_default[metric]()


def build_judge(params):
  """Build LLMAsJudge from params dict."""
  criterion = params.get("criterion", "correctness")
  threshold = params.get("threshold")

  factories_with_t = {
      "correctness": lambda t: LLMAsJudge.correctness(threshold=t),
      "hallucination": lambda t: LLMAsJudge.hallucination(
          threshold=t,
      ),
      "sentiment": lambda t: LLMAsJudge.sentiment(threshold=t),
  }
  factories_default = {
      "correctness": LLMAsJudge.correctness,
      "hallucination": LLMAsJudge.hallucination,
      "sentiment": LLMAsJudge.sentiment,
  }

  if criterion not in factories_with_t:
    raise ValueError(f"Unknown criterion: {criterion!r}")

  if threshold is not None:
    return factories_with_t[criterion](threshold)
  return factories_default[criterion]()
