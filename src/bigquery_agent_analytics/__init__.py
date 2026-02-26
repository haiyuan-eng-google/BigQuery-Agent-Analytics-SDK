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

"""BigQuery Agent Analytics SDK.

This package provides the consumption-layer SDK for analyzing agent traces
stored in BigQuery, including:

1. **SDK Client** - High-level Python interface abstracting BigQuery SQL.
2. **Trace Reconstruction** - Retrieve and visualize agent conversation DAGs.
3. **Evaluation Engine** - Code-based and LLM-as-judge evaluation at scale.
4. **Feedback & Curation** - Drift detection and question distribution analysis.
5. **Agent Insights** - Multi-stage pipeline for comprehensive session analysis.
6. **Trace-Based Evaluation Harness** - Trajectory metrics and replay.
7. **Long-Horizon Agent Memory** - Cross-session context and semantic search.
8. **BigQuery AI/ML Integration** - AI.GENERATE, AI.EMBED, ML.DETECT_ANOMALIES.

Quick start::

    from bigquery_agent_analytics import Client

    client = Client(project_id="my-project", dataset_id="analytics")
    trace = client.get_trace("trace-123")
    trace.render()

    # Generate insights report
    report = client.insights(max_sessions=50)
    print(report.summary())
"""

import logging

logger = logging.getLogger("bigquery_agent_analytics." + __name__)

__all__ = []

# --- SDK Client & Core ---
try:
  from .client import Client
  from .evaluators import CodeEvaluator
  from .evaluators import EvaluationReport
  from .evaluators import LLMAsJudge
  from .evaluators import SessionScore
  from .feedback import AnalysisConfig
  from .feedback import DriftReport
  from .feedback import QuestionDistribution
  from .insights import InsightsConfig
  from .insights import InsightsReport
  from .insights import SessionFacet
  from .trace import ContentPart
  from .trace import EventType
  from .trace import ObjectRef
  from .trace import Span
  from .trace import Trace
  from .trace import TraceFilter
  from .views import ViewManager

  __all__.extend(
      [
          "Client",
          "Trace",
          "Span",
          "ContentPart",
          "EventType",
          "ObjectRef",
          "TraceFilter",
          "ViewManager",
          "CodeEvaluator",
          "LLMAsJudge",
          "EvaluationReport",
          "SessionScore",
          "DriftReport",
          "QuestionDistribution",
          "AnalysisConfig",
          "InsightsReport",
          "InsightsConfig",
          "SessionFacet",
      ]
  )
except ImportError as e:
  logger.debug(
      "Could not import SDK client components: %s. "
      "Ensure required dependencies are installed.",
      e,
  )

# Trace Evaluator
try:
  from .trace_evaluator import BigQueryTraceEvaluator
  from .trace_evaluator import EvaluationResult
  from .trace_evaluator import TraceReplayRunner
  from .trace_evaluator import TrajectoryMetrics

  __all__.extend(
      [
          "BigQueryTraceEvaluator",
          "EvaluationResult",
          "TraceReplayRunner",
          "TrajectoryMetrics",
      ]
  )
except ImportError as e:
  logger.debug(
      "Could not import trace evaluator components: %s. "
      "Ensure required dependencies are installed.",
      e,
  )

# Memory Service
try:
  from .memory_service import BigQueryMemoryService
  from .memory_service import BigQuerySessionMemory
  from .memory_service import ContextManager
  from .memory_service import Episode
  from .memory_service import UserProfileBuilder

  __all__.extend(
      [
          "BigQueryMemoryService",
          "BigQuerySessionMemory",
          "ContextManager",
          "Episode",
          "UserProfileBuilder",
      ]
  )
except ImportError as e:
  logger.debug(
      "Could not import memory service components: %s. "
      "Ensure required dependencies are installed.",
      e,
  )

# AI/ML Integration
try:
  from .ai_ml_integration import AnomalyDetector
  from .ai_ml_integration import BatchEvaluator
  from .ai_ml_integration import BigQueryAIClient
  from .ai_ml_integration import EmbeddingSearchClient

  __all__.extend(
      [
          "BigQueryAIClient",
          "EmbeddingSearchClient",
          "AnomalyDetector",
          "BatchEvaluator",
      ]
  )
except ImportError as e:
  logger.debug(
      "Could not import AI/ML integration components: %s. "
      "Ensure required dependencies are installed.",
      e,
  )

# Multi-Trial
try:
  from .multi_trial import MultiTrialReport
  from .multi_trial import TrialResult
  from .multi_trial import TrialRunner

  __all__.extend(
      [
          "TrialRunner",
          "TrialResult",
          "MultiTrialReport",
      ]
  )
except ImportError as e:
  logger.debug(
      "Could not import multi-trial components: %s. "
      "Ensure required dependencies are installed.",
      e,
  )

# Grader Pipeline
try:
  from .grader_pipeline import AggregateVerdict
  from .grader_pipeline import BinaryStrategy
  from .grader_pipeline import GraderPipeline
  from .grader_pipeline import GraderResult
  from .grader_pipeline import MajorityStrategy
  from .grader_pipeline import ScoringStrategy
  from .grader_pipeline import WeightedStrategy

  __all__.extend(
      [
          "AggregateVerdict",
          "BinaryStrategy",
          "GraderPipeline",
          "GraderResult",
          "MajorityStrategy",
          "ScoringStrategy",
          "WeightedStrategy",
      ]
  )
except ImportError as e:
  logger.debug(
      "Could not import grader pipeline components: %s. "
      "Ensure required dependencies are installed.",
      e,
  )

# Eval Suite
try:
  from .eval_suite import EvalCategory
  from .eval_suite import EvalSuite
  from .eval_suite import EvalTaskDef
  from .eval_suite import SuiteHealth

  __all__.extend(
      [
          "EvalCategory",
          "EvalSuite",
          "EvalTaskDef",
          "SuiteHealth",
      ]
  )
except ImportError as e:
  logger.debug(
      "Could not import eval suite components: %s. "
      "Ensure required dependencies are installed.",
      e,
  )

# Eval Validator
try:
  from .eval_validator import EvalValidator
  from .eval_validator import ValidationWarning

  __all__.extend(
      [
          "EvalValidator",
          "ValidationWarning",
      ]
  )
except ImportError as e:
  logger.debug(
      "Could not import eval validator components: %s. "
      "Ensure required dependencies are installed.",
      e,
  )

# BigFrames Evaluator (optional bigframes dependency)
try:
  from .bigframes_evaluator import BigFramesEvaluator

  __all__.append("BigFramesEvaluator")
except ImportError as e:
  logger.debug(
      "Could not import BigFramesEvaluator: %s. "
      "Install bigframes to use this feature.",
      e,
  )
