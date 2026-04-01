# BigQuery Agent Analytics SDK

An open-source Python SDK for analyzing, evaluating, and curating agent traces
stored in BigQuery. Built on top of the
[Agent Development Kit (ADK)](https://github.com/google/adk-python), it provides
a consumption-layer toolkit for agent observability at scale.

## Features

- **Trace Reconstruction** -- Retrieve and visualize agent conversation DAGs
- **Code-Based Evaluation** -- Deterministic metrics (latency, turn count,
  error rate, token efficiency, cost)
- **LLM-as-Judge** -- Semantic evaluation using LLM scoring (correctness,
  hallucination, sentiment)
- **Trajectory Matching** -- Exact, in-order, and any-order tool call matching
- **Multi-Trial Evaluation** -- Run N trials with pass@k / pass^k metrics for
  non-deterministic agents
- **Grader Composition** -- Combine code + LLM graders via weighted, binary, or
  majority strategies
- **Eval Suite Management** -- Lifecycle management with graduation, saturation
  detection, and balance checking
- **Eval Quality Validation** -- Static checks for ambiguous tasks, class
  imbalance, and suspicious thresholds
- **Drift Detection** -- Compare golden vs production question distributions
- **Agent Insights** -- Multi-stage pipeline for comprehensive session analysis
- **Long-Horizon Memory** -- Cross-session context and semantic search
- **BigQuery AI/ML Integration** -- AI.GENERATE, embeddings, anomaly detection,
  latency forecasting
- **Context Graph** -- Property Graph linking technical traces to business
  entities, with GQL traversal, world-change detection, and artifact lineage

## Installation

```bash
pip install bigquery-agent-analytics
```

With optional LLM judge support:

```bash
pip install bigquery-agent-analytics[llm]
```

With BigFrames support:

```bash
pip install bigquery-agent-analytics[bigframes]
```

## Quick Start

```python
from bigquery_agent_analytics import Client

client = Client(project_id="my-project", dataset_id="analytics")
trace = client.get_trace("trace-abc-123")
trace.render()
```

### Code-Based Evaluation

```python
from bigquery_agent_analytics import CodeEvaluator

evaluator = CodeEvaluator.latency(threshold_ms=5000)
score = evaluator.evaluate_session({
    "session_id": "s1",
    "avg_latency_ms": 2000,
})
print(score.passed)  # True
```

### Multi-Trial Evaluation

```python
from bigquery_agent_analytics import BigQueryTraceEvaluator, TrialRunner

evaluator = BigQueryTraceEvaluator(
    project_id="my-project",
    dataset_id="analytics",
)
runner = TrialRunner(evaluator, num_trials=5)

report = await runner.run_trials(
    session_id="sess-123",
    golden_trajectory=[{"tool_name": "search", "args": {}}],
)
print(f"pass@k: {report.pass_at_k}, pass^k: {report.pass_pow_k}")
```

### Grader Pipeline

```python
from bigquery_agent_analytics import (
    CodeEvaluator, GraderPipeline, LLMAsJudge, WeightedStrategy,
)

pipeline = (
    GraderPipeline(WeightedStrategy(
        weights={"latency_evaluator": 0.3, "correctness_judge": 0.7},
    ))
    .add_code_grader(CodeEvaluator.latency())
    .add_llm_grader(LLMAsJudge.correctness())
)

verdict = await pipeline.evaluate(
    session_summary={"session_id": "s1", "avg_latency_ms": 2000},
    trace_text="User: hello\nAgent: hi",
    final_response="hi",
)
```

### Eval Suite Management

```python
from bigquery_agent_analytics import EvalCategory, EvalSuite, EvalTaskDef

suite = EvalSuite(name="my_agent_evals")
suite.add_task(EvalTaskDef(
    task_id="t1",
    session_id="sess-123",
    description="Test basic search",
    expected_trajectory=[{"tool_name": "search", "args": {}}],
))

health = suite.check_health()
print(health.warnings)

# Auto-graduate stable tasks to regression
graduated = suite.auto_graduate(pass_history, threshold_runs=10)
```

### CLI Quick Start

The SDK includes a CLI for diagnostics, evaluation, and analytics:

```bash
# Health check
bq-agent-sdk doctor --project-id=P --dataset-id=D

# Retrieve a trace
bq-agent-sdk get-trace --project-id=P --dataset-id=D --session-id=S

# Run evaluation (CI gate with --exit-code)
bq-agent-sdk evaluate --project-id=P --dataset-id=D \
  --evaluator=latency --exit-code

# Generate insights
bq-agent-sdk insights --project-id=P --dataset-id=D --last=7d

# All 10 commands: doctor, get-trace, evaluate, insights, drift,
# distribution, hitl-metrics, list-traces, views create-all, views create
```

Set `BQ_AGENT_PROJECT` and `BQ_AGENT_DATASET` environment variables to
skip `--project-id` and `--dataset-id` on every call.

### Remote Function (BigQuery SQL)

Deploy the SDK as a BigQuery Remote Function for SQL-native access:

```bash
cd deploy/remote_function
./deploy.sh my-project us-central1 agent_analytics US
```

Then query from SQL (using the fully-qualified name from `register.sql`):

```sql
SELECT `my-project.agent_analytics.agent_analytics`(
  'analyze', JSON'{"session_id": "s1"}'
);
SELECT `my-project.agent_analytics.agent_analytics`(
  'evaluate', JSON'{"metric": "latency"}'
);
```

See [SDK.md](SDK.md) for the full CLI reference, Remote Function API,
and continuous query templates.

### Streaming Evaluation

Deploy a dedicated Cloud Run worker that scans recent `agent_events`
rows every `5` minutes through `Cloud Scheduler -> SDK -> BigQuery`:

```mermaid
flowchart LR
  A["agent_events"] --> B["Cloud Scheduler"]
  B --> C["Cloud Run worker"]
  C --> D["overlap scan + dedupe + checkpoint"]
  D --> E["streaming_evaluation_results"]
```

Quick start:

```bash
cd deploy/streaming_evaluation
./setup.sh up my-project my_dataset agent_events us-central1
```

Use [deploy/streaming_evaluation/README.md](deploy/streaming_evaluation/README.md)
for the full setup flow. This path intentionally avoids continuous-query
reservations because they can be too expensive as a default deployment
mode.

## Architecture

```
bigquery_agent_analytics/
├── __init__.py              # Package exports
├── client.py                # High-level SDK client
├── evaluators.py            # CodeEvaluator + LLMAsJudge
├── trace.py                 # Trace reconstruction & visualization
├── trace_evaluator.py       # Trajectory matching & replay
├── multi_trial.py           # Multi-trial runner + pass@k
├── grader_pipeline.py       # Grader composition pipeline
├── eval_suite.py            # Eval suite lifecycle management
├── eval_validator.py        # Static validation checks
├── feedback.py              # Drift detection & question distribution
├── insights.py              # Multi-stage insights pipeline
├── memory_service.py        # Long-horizon agent memory
├── ai_ml_integration.py     # BigQuery AI/ML capabilities
├── bigframes_evaluator.py   # BigFrames DataFrame evaluator
├── context_graph.py         # Property Graph: BizNode extraction, GQL, world-change
├── event_semantics.py       # Canonical event type helpers & predicates
├── views.py                 # Per-event-type BigQuery view management
├── cli.py                   # CLI entry point (bq-agent-sdk)
├── formatter.py             # Output formatting (json/text/table)
└── serialization.py         # Uniform serialization layer
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Format code
pyink --config pyproject.toml src/ tests/
isort src/ tests/
```

## License

Apache License 2.0 -- see [LICENSE](LICENSE) for details.
