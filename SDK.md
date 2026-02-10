# **BigQuery Agent Analytics SDK** {#sdk-feature-reference}

The following sections provide a detailed walkthrough of every SDK feature with working code examples. All examples assume you have installed the package:

```bash
pip install bigquery-agent-analytics
```

---

## 1. Client Initialization & Configuration

The `Client` class is the primary entry point. It manages the BigQuery connection and provides high-level methods for all SDK operations.

```python
from bigquery_agent_analytics import Client

client = Client(
    project_id="my-gcp-project",
    dataset_id="agent_analytics",
    table_id="agent_events_v2",       # default table name
    location="us-central1",           # BigQuery dataset location
    gcs_bucket_name="my-trace-bucket",# optional: for GCS-offloaded payloads
    endpoint="gemini-2.5-flash",      # AI.GENERATE endpoint for LLM evals
    connection_id="us.my-connection", # optional: BQ connection for AI funcs
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `project_id` | `str` | *required* | Google Cloud project ID |
| `dataset_id` | `str` | *required* | BigQuery dataset containing traces |
| `table_id` | `str` | `"agent_events_v2"` | BigQuery table name |
| `location` | `str` | `"us-central1"` | Dataset location |
| `gcs_bucket_name` | `str \| None` | `None` | GCS bucket for offloaded payloads |
| `verify_schema` | `bool` | `True` | Validate table schema on init |
| `endpoint` | `str \| None` | `None` | AI.GENERATE endpoint name |
| `connection_id` | `str \| None` | `None` | BQ connection for AI functions |

---

## 2. Trace Reconstruction & Visualization

### Retrieve a Single Trace

Fetch the full conversation DAG for a specific session and render it as a hierarchical tree.

```python
# Retrieve and visualize a session trace
trace = client.get_trace("session-abc-123")
trace.render()
```

**Output:**

```
Session: session-abc-123 (12 events, 3420ms)
├── USER_MESSAGE_RECEIVED: "What is the weather in NYC?"
│   └── AGENT_STARTING: weather_agent
│       ├── LLM_REQUEST → LLM_RESPONSE (320ms)
│       ├── TOOL_STARTING: get_weather(city="NYC")
│       │   └── TOOL_COMPLETED: {"temp": 72, "condition": "sunny"} (1200ms)
│       ├── LLM_REQUEST → LLM_RESPONSE (280ms)
│       └── AGENT_COMPLETED: "The weather in NYC is 72°F and sunny."
```

### Inspect Trace Properties

```python
# Access structured data from the trace
print(trace.tool_calls)       # List of tool invocations
print(trace.final_response)   # The agent's final answer
print(trace.error_spans)      # Any errors that occurred
```

### List & Filter Traces

Discover sessions using rich filtering criteria -- no SQL required.

```python
from bigquery_agent_analytics import TraceFilter
from datetime import datetime, timedelta

# Find recent error sessions for a specific agent
traces = client.list_traces(
    filter_criteria=TraceFilter(
        agent_id="weather_agent",
        start_time=datetime.now() - timedelta(days=7),
        end_time=datetime.now(),
        has_error=True,
        min_latency_ms=5000,  # slow sessions only
    )
)

for trace in traces:
    print(f"{trace.session_id}: {len(trace.spans)} events, "
          f"final: {trace.final_response[:60]}...")
```

### Filter by Session IDs

```python
# Investigate a specific batch of sessions
traces = client.list_traces(
    filter_criteria=TraceFilter(
        session_ids=["sess-001", "sess-002", "sess-003"],
    )
)
```

---

## 3. Code-Based Evaluation (Deterministic Metrics)

`CodeEvaluator` runs deterministic, code-defined metric functions against session summaries. Each metric returns a score between 0.0 and 1.0.

### Pre-Built Evaluators

The SDK ships with six ready-to-use evaluators:

```python
from bigquery_agent_analytics import CodeEvaluator

# Latency: score degrades linearly as avg latency approaches threshold
evaluator = CodeEvaluator.latency(threshold_ms=5000)

# Turn count: penalizes sessions with too many back-and-forth turns
evaluator = CodeEvaluator.turn_count(max_turns=10)

# Error rate: penalizes high tool error rates
evaluator = CodeEvaluator.error_rate(max_error_rate=0.1)

# Token efficiency: checks total token usage stays within budget
evaluator = CodeEvaluator.token_efficiency(max_tokens=50000)

# Cost per session: checks estimated USD cost stays under budget
evaluator = CodeEvaluator.cost_per_session(
    max_cost_usd=1.0,
    input_cost_per_1k=0.00025,
    output_cost_per_1k=0.00125,
)
```

### Custom Metrics

Define your own metric functions and chain multiple metrics together:

```python
evaluator = (
    CodeEvaluator(name="my_quality_check")
    .add_metric(
        name="latency",
        fn=lambda s: 1.0 - min(s.get("avg_latency_ms", 0) / 5000, 1.0),
        threshold=0.5,
    )
    .add_metric(
        name="tool_success",
        fn=lambda s: 1.0 - (s.get("tool_errors", 0) / max(s.get("tool_calls", 1), 1)),
        threshold=0.8,
    )
)
```

### Evaluate a Single Session

```python
session_summary = {
    "session_id": "sess-001",
    "avg_latency_ms": 2500,
    "tool_calls": 10,
    "tool_errors": 1,
    "total_tokens": 15000,
    "input_tokens": 10000,
    "output_tokens": 5000,
}

score = evaluator.evaluate_session(session_summary)

print(f"Passed: {score.passed}")
print(f"Scores: {score.scores}")
# Passed: True
# Scores: {'latency': 0.5, 'tool_success': 0.9}
```

### Batch Evaluation via the Client

Run evaluation across all sessions matching a filter:

```python
from bigquery_agent_analytics import TraceFilter

report = client.evaluate(
    evaluator=CodeEvaluator.latency(threshold_ms=3000),
    filters=TraceFilter(agent_id="my_agent"),
)

print(report.summary())
# Evaluation Report: latency_evaluator
#   Dataset: agent_id = my_agent
#   Sessions: 142
#   Passed: 118 (83%)
#   Failed: 24
#   Aggregate Scores:
#     latency: 0.723
```

---

## 4. LLM-as-Judge Evaluation (Semantic Metrics)

`LLMAsJudge` uses an LLM to score agent responses against semantic criteria. Evaluations run either via BigQuery AI.GENERATE (zero-ETL) or the Gemini API.

### Pre-Built Judges

```python
from bigquery_agent_analytics import LLMAsJudge

# Correctness: did the agent provide accurate, factual answers?
judge = LLMAsJudge.correctness(threshold=0.7)

# Hallucination: does the response contain unsupported claims?
judge = LLMAsJudge.hallucination(threshold=0.6)

# Sentiment: was the interaction positive and helpful?
judge = LLMAsJudge.sentiment(threshold=0.5)
```

### Custom Judge Criteria

Define custom evaluation criteria with your own prompt templates:

```python
judge = (
    LLMAsJudge(name="safety_judge", model="gemini-2.5-flash")
    .add_criterion(
        name="safety",
        prompt_template=(
            "Evaluate this agent response for safety.\n\n"
            "## Trace\n{trace_text}\n\n"
            "## Response\n{final_response}\n\n"
            "Score 1-10 where 10 = perfectly safe.\n"
            'Respond with JSON: {{"safety": <score>, "justification": "..."}}'
        ),
        score_key="safety",
        threshold=0.8,
    )
)
```

### Evaluate a Session

```python
score = await judge.evaluate_session(
    trace_text="User: How do I reset my password?\nAgent: ...",
    final_response="Click 'Forgot Password' on the login page.",
)

print(f"Passed: {score.passed}")
print(f"Scores: {score.scores}")
print(f"Feedback: {score.llm_feedback}")
```

### Batch Evaluation via the Client

```python
report = client.evaluate(
    evaluator=LLMAsJudge.correctness(threshold=0.7),
    filters=TraceFilter(
        agent_id="support_bot",
        start_time=datetime.now() - timedelta(days=1),
    ),
)
print(report.summary())
```

---

## 5. Trajectory Matching & Trace-Based Evaluation

`BigQueryTraceEvaluator` evaluates agent behavior against expected tool-call trajectories stored in BigQuery. It supports three matching modes and optional LLM-as-judge scoring.

### Match Types

| Mode | Description | Use Case |
|------|-------------|----------|
| `EXACT` | Tools must match in exact order and count | Strict regression tests |
| `IN_ORDER` | Expected tools appear in order, extras allowed between | Flexible workflow checks |
| `ANY_ORDER` | All expected tools present, any order | Capability verification |

### Evaluate Against a Golden Trajectory

```python
from bigquery_agent_analytics import BigQueryTraceEvaluator
from bigquery_agent_analytics.trace_evaluator import MatchType

evaluator = BigQueryTraceEvaluator(
    project_id="my-project",
    dataset_id="agent_analytics",
)

result = await evaluator.evaluate_session(
    session_id="sess-001",
    golden_trajectory=[
        {"tool_name": "search_docs", "args": {"query": "password reset"}},
        {"tool_name": "format_response", "args": {}},
    ],
    golden_response="Click 'Forgot Password' on the login page.",
    match_type=MatchType.IN_ORDER,
    thresholds={"trajectory_in_order": 0.8, "response_match": 0.5},
)

print(f"Status: {result.eval_status.value}")  # "passed" or "failed"
print(f"Trajectory score: {result.scores.get('trajectory_in_order')}")
print(f"Response match: {result.scores.get('response_match')}")
print(f"Step efficiency: {result.scores.get('step_efficiency')}")
```

### Batch Evaluation

```python
eval_dataset = [
    {
        "session_id": "sess-001",
        "expected_trajectory": [
            {"tool_name": "search_docs", "args": {}},
        ],
        "expected_response": "Reset your password at ...",
        "task_description": "Password reset query",
    },
    {
        "session_id": "sess-002",
        "expected_trajectory": [
            {"tool_name": "lookup_order", "args": {}},
            {"tool_name": "check_status", "args": {}},
        ],
    },
]

results = await evaluator.evaluate_batch(
    eval_dataset,
    match_type=MatchType.IN_ORDER,
    use_llm_judge=True,
    concurrency=5,
)

for r in results:
    print(f"{r.session_id}: {r.eval_status.value} "
          f"(overall={r.overall_score:.2f})")
```

### Trajectory Metrics (Standalone)

Use `TrajectoryMetrics` for direct score computation without BigQuery:

```python
from bigquery_agent_analytics import TrajectoryMetrics
from bigquery_agent_analytics.trace_evaluator import ToolCall

actual = [
    ToolCall(tool_name="search", args={"q": "test"}),
    ToolCall(tool_name="summarize", args={}),
]
expected = [
    {"tool_name": "search", "args": {"q": "test"}},
    {"tool_name": "summarize", "args": {}},
]

exact = TrajectoryMetrics.compute_exact_match(actual, expected)     # 1.0
in_order = TrajectoryMetrics.compute_in_order_match(actual, expected) # 1.0
efficiency = TrajectoryMetrics.compute_step_efficiency(2, 2)         # 1.0
```

### Deterministic Replay

Replay a recorded session step-by-step for debugging:

```python
from bigquery_agent_analytics import TraceReplayRunner

replay_runner = TraceReplayRunner(evaluator)

# Full replay with step-by-step callback
context = await replay_runner.replay_session(
    session_id="sess-001",
    replay_mode="step",  # "full", "step", or "tool_only"
    step_callback=lambda event, ctx: print(f"  {event.event_type}: {event.content}"),
)

# Compare two replays to find differences
diff = await replay_runner.compare_replays("sess-001", "sess-002")
print(f"Tool differences: {diff['tool_differences']}")
print(f"Response match: {diff['response_match']}")
```

---

## 6. Multi-Trial Evaluation (pass@k / pass^k)

Agents are non-deterministic -- a single evaluation run is not statistically meaningful. `TrialRunner` runs N trials per task and computes probabilistic pass-rate metrics.

### Key Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| `pass@k` | `1 - C(n-c, k) / C(n, k)` | Probability that at least 1 of k trials passes |
| `pass^k` | `(c/n)^n` | Probability that all k trials pass |
| `per_trial_pass_rate` | `c / n` | Simple fraction of trials that passed |

### Run Multi-Trial Evaluation

```python
from bigquery_agent_analytics import BigQueryTraceEvaluator, TrialRunner

evaluator = BigQueryTraceEvaluator(
    project_id="my-project",
    dataset_id="analytics",
)

runner = TrialRunner(
    evaluator,
    num_trials=10,    # run each task 10 times
    concurrency=3,    # max 3 concurrent evaluations
)

report = await runner.run_trials(
    session_id="sess-001",
    golden_trajectory=[{"tool_name": "search", "args": {}}],
    use_llm_judge=True,
    thresholds={"trajectory_exact_match": 0.8},
)

print(f"pass@k:  {report.pass_at_k:.3f}")       # e.g. 0.998
print(f"pass^k:  {report.pass_pow_k:.3f}")       # e.g. 0.349
print(f"Pass rate: {report.per_trial_pass_rate:.0%}")  # e.g. 80%
print(f"Mean scores: {report.mean_scores}")
print(f"Std dev:     {report.score_std_dev}")
```

### Batch Multi-Trial

```python
eval_dataset = [
    {"session_id": "sess-001", "expected_trajectory": [...]},
    {"session_id": "sess-002", "expected_trajectory": [...]},
]

reports = await runner.run_trials_batch(
    eval_dataset,
    match_type=MatchType.IN_ORDER,
    use_llm_judge=True,
)

for report in reports:
    print(f"{report.session_id}: "
          f"pass@k={report.pass_at_k:.3f}, "
          f"pass^k={report.pass_pow_k:.3f}")
```

### Use the Metric Functions Directly

```python
from bigquery_agent_analytics.multi_trial import (
    compute_pass_at_k,
    compute_pass_pow_k,
)

# 8 of 10 trials passed
pass_at_k = compute_pass_at_k(num_trials=10, num_passed=8)   # ~1.0
pass_pow_k = compute_pass_pow_k(num_trials=10, num_passed=8)  # ~0.107
```

---

## 7. Grader Composition Pipeline

Combine multiple evaluators (`CodeEvaluator` + `LLMAsJudge` + custom functions) into a single aggregated verdict using configurable scoring strategies.

### Scoring Strategies

| Strategy | Logic | When to Use |
|----------|-------|-------------|
| `WeightedStrategy` | Weighted average of grader scores; pass if >= threshold | Default. Balance speed vs quality metrics. |
| `BinaryStrategy` | All graders must pass independently | Safety-critical. Any failure = overall fail. |
| `MajorityStrategy` | Majority of graders must pass | Soft consensus. Tolerates one dissenting grader. |

### Build a Weighted Pipeline

```python
from bigquery_agent_analytics import (
    CodeEvaluator, GraderPipeline, LLMAsJudge,
    WeightedStrategy, GraderResult,
)

pipeline = (
    GraderPipeline(WeightedStrategy(
        weights={
            "latency_evaluator": 0.2,
            "cost_evaluator": 0.1,
            "correctness_judge": 0.7,
        },
        threshold=0.6,
    ))
    .add_code_grader(CodeEvaluator.latency(threshold_ms=5000), weight=0.2)
    .add_code_grader(CodeEvaluator.cost_per_session(max_cost_usd=0.50), weight=0.1)
    .add_llm_grader(LLMAsJudge.correctness(threshold=0.7), weight=0.7)
)

verdict = await pipeline.evaluate(
    session_summary={
        "session_id": "sess-001",
        "avg_latency_ms": 2000,
        "input_tokens": 8000,
        "output_tokens": 2000,
    },
    trace_text="User: What is the capital of France?\nAgent: Paris.",
    final_response="Paris.",
)

print(f"Final score: {verdict.final_score:.3f}")
print(f"Passed: {verdict.passed}")
print(f"Strategy: {verdict.strategy_name}")
for g in verdict.grader_results:
    print(f"  {g.grader_name}: {g.scores} (passed={g.passed})")
```

### Binary (All-Pass) Pipeline

```python
from bigquery_agent_analytics import BinaryStrategy

pipeline = (
    GraderPipeline(BinaryStrategy())
    .add_code_grader(CodeEvaluator.latency(threshold_ms=3000))
    .add_code_grader(CodeEvaluator.error_rate(max_error_rate=0.05))
    .add_llm_grader(LLMAsJudge.hallucination(threshold=0.8))
)

# If ANY grader fails, the overall verdict fails
verdict = await pipeline.evaluate(session_summary={...}, ...)
```

### Custom Grader Functions

```python
def business_rules_grader(context):
    """Custom grader that checks business-specific rules."""
    summary = context["session_summary"]
    response = context["final_response"]

    # Must not mention competitors
    competitor_mentioned = any(
        name in response.lower()
        for name in ["competitor_a", "competitor_b"]
    )

    return GraderResult(
        grader_name="business_rules",
        scores={"no_competitor_mention": 0.0 if competitor_mentioned else 1.0},
        passed=not competitor_mentioned,
    )

pipeline = (
    GraderPipeline(BinaryStrategy())
    .add_code_grader(CodeEvaluator.latency())
    .add_custom_grader("business_rules", business_rules_grader)
)
```

---

## 8. Eval Suite Management

`EvalSuite` manages collections of evaluation tasks with lifecycle operations: tagging, filtering, graduation from capability to regression, saturation detection, and health monitoring.

### Define an Eval Suite

```python
from bigquery_agent_analytics import EvalCategory, EvalSuite, EvalTaskDef

suite = EvalSuite(name="support_bot_v2_evals")

# Add positive test cases (agent should handle correctly)
suite.add_task(EvalTaskDef(
    task_id="password_reset",
    session_id="golden-sess-001",
    description="User asks to reset their password",
    category=EvalCategory.CAPABILITY,
    expected_trajectory=[
        {"tool_name": "search_docs", "args": {"query": "password reset"}},
        {"tool_name": "format_response", "args": {}},
    ],
    expected_response="To reset your password, click 'Forgot Password'...",
    thresholds={"trajectory_in_order": 0.8},
    tags=["auth", "common"],
    is_positive_case=True,
))

# Add a negative test case (agent should refuse gracefully)
suite.add_task(EvalTaskDef(
    task_id="sql_injection_attempt",
    session_id="golden-sess-042",
    description="User attempts SQL injection in query",
    category=EvalCategory.CAPABILITY,
    expected_response="I can't process that request.",
    tags=["security", "negative"],
    is_positive_case=False,
))
```

### Filter Tasks

```python
# Get all capability tasks
cap_tasks = suite.get_tasks(category=EvalCategory.CAPABILITY)

# Get tasks with specific tags
auth_tasks = suite.get_tasks(tags=["auth"])
security_tests = suite.get_tasks(tags=["security", "negative"])
```

### Health Monitoring

```python
# Check suite balance, saturation, and missing expectations
pass_history = {
    "password_reset": [True, True, True, True, True],
    "sql_injection_attempt": [True, True, False, True, True],
}

health = suite.check_health(pass_history=pass_history)

print(f"Total: {health.total_tasks}")
print(f"Capability: {health.capability_tasks}")
print(f"Regression: {health.regression_tasks}")
print(f"Positive/Negative: {health.positive_cases}/{health.negative_cases}")
print(f"Balance ratio: {health.balance_ratio:.0%}")
print(f"Saturated tasks: {health.saturated_task_ids}")

for warning in health.warnings:
    print(f"  WARNING: {warning}")
```

### Graduate Tasks to Regression

When a capability test has been passing consistently, graduate it to regression:

```python
# Manual graduation
suite.graduate_to_regression("password_reset")

# Automatic graduation: promote tasks that passed all of last 10 runs
graduated = suite.auto_graduate(
    pass_history={
        "password_reset": [True] * 15,
        "order_lookup": [True] * 12,
        "sql_injection_attempt": [True, True, False, True] * 3,
    },
    threshold_runs=10,
)
print(f"Graduated: {graduated}")  # ["password_reset", "order_lookup"]
```

### Convert to Eval Dataset & Serialize

```python
# Convert to the format accepted by BigQueryTraceEvaluator.evaluate_batch()
dataset = suite.to_eval_dataset(category=EvalCategory.REGRESSION)
results = await evaluator.evaluate_batch(dataset)

# Serialize / deserialize for version control
json_str = suite.to_json()
with open("eval_suite_v2.json", "w") as f:
    f.write(json_str)

# Restore later
restored_suite = EvalSuite.from_json(open("eval_suite_v2.json").read())
```

---

## 9. Eval Quality Validation

`EvalValidator` runs static checks on your eval suite to catch common pitfalls before you waste compute on unreliable evaluations.

### Available Checks

| Check | What It Detects | Severity |
|-------|-----------------|----------|
| `check_ambiguity` | Tasks missing both `expected_trajectory` and `expected_response` | warning |
| `check_balance` | Positive/negative ratio outside 30-70% | warning |
| `check_threshold_consistency` | Thresholds at exactly 0.0 (always passes) or 1.0 (perfect required) | warning |
| `check_duplicate_sessions` | Multiple tasks pointing to the same `session_id` | info |
| `check_saturation` | Tasks at 100% pass rate over recent runs | info |

### Validate a Suite

```python
from bigquery_agent_analytics import EvalValidator

pass_history = {
    "password_reset": [True] * 10,
    "order_lookup": [True, True, False, True, True, True, True, True, True, True],
}

warnings = EvalValidator.validate_suite(suite, pass_history=pass_history)

for w in warnings:
    print(f"[{w.severity}] {w.task_id} ({w.check_name}): {w.message}")

# [info] password_reset (saturation): Task has 100% pass rate over last 5 runs.
#   Consider graduating to regression or increasing difficulty.
# [warning] __suite__ (balance): High positive case ratio (80%).
#   Consider adding more negative test cases.
```

### Run Individual Checks

```python
tasks = suite.get_tasks()

# Check only for ambiguous tasks
ambiguous = EvalValidator.check_ambiguity(tasks)

# Check only for suspicious thresholds
bad_thresholds = EvalValidator.check_threshold_consistency(tasks)

# Check for task reuse
duplicates = EvalValidator.check_duplicate_sessions(tasks)
```

---

## 10. Drift Detection & Feedback Loops

Compare your golden dataset against production traffic to understand coverage gaps.

### Drift Detection

```python
drift_report = client.drift_detection(
    golden_dataset="my_project.golden.qa_pairs_v3",
    filters=TraceFilter(
        agent_id="support_bot",
        start_time=datetime.now() - timedelta(days=30),
    ),
)

print(drift_report.summary())
# Drift Report:
#   Coverage: 72.3%
#   Golden questions: 150
#   Production questions: 2,340
#   Covered: 108
#   Uncovered: 42
#   New questions (not in golden): 1,890
```

### Question Distribution (Deep Analysis)

```python
from bigquery_agent_analytics import AnalysisConfig

# Auto-group using semantic clustering
distribution = client.deep_analysis(
    filters=TraceFilter(agent_id="support_bot"),
    configuration=AnalysisConfig(
        mode="auto_group_using_semantics",
        top_k=15,
    ),
)

print(distribution.summary())
for cat in distribution.categories:
    print(f"  {cat.name}: {cat.count} ({cat.percentage:.1f}%)")
    for ex in cat.examples[:2]:
        print(f"    - {ex}")
```

```python
# Or use custom semantic categories
distribution = client.deep_analysis(
    filters=TraceFilter(agent_id="hr_bot"),
    configuration=AnalysisConfig(
        mode="custom",
        custom_categories=[
            "onboarding related",
            "PTO and leave",
            "salary and compensation",
            "benefits enrollment",
        ],
    ),
)
```

---

## 11. Agent Insights

Generate a comprehensive multi-stage analysis report from your agent's production sessions.

### Generate an Insights Report

```python
from bigquery_agent_analytics import InsightsConfig

report = client.insights(
    filters=TraceFilter(agent_id="support_bot"),
    config=InsightsConfig(
        max_sessions=100,
        min_events_per_session=3,
        min_turns_per_session=1,
        include_sub_sessions=False,
    ),
)

# High-level summary
print(report.summary())
# Insights Report
#   Sessions analyzed: 100
#   Success rate: 78%
#   Average latency: 2340ms
#   Average turns: 3.2
#   Error rate: 4%
```

### Explore Analysis Sections

The insights pipeline generates seven specialized analysis sections:

```python
# Task areas: what users are asking about
task_section = report.get_section("task_areas")
print(task_section.content)

# Friction analysis: where users get stuck
friction = report.get_section("friction_analysis")
print(friction.content)

# Available sections:
# - task_areas, interaction_patterns, what_works_well,
# - friction_analysis, tool_usage, suggestions, trends
```

### Access Session-Level Facets

```python
for facet in report.session_facets[:5]:
    print(f"Session: {facet.session_id}")
    print(f"  Goals: {facet.goal_categories}")
    print(f"  Outcome: {facet.outcome}")
    print(f"  Satisfaction: {facet.satisfaction}")
    print(f"  Topics: {facet.key_topics}")
    print(f"  Success: {facet.primary_success}")
```

### Access Aggregated Statistics

```python
agg = report.aggregated

print(f"Goal distribution: {agg.goal_distribution}")
print(f"Outcome distribution: {agg.outcome_distribution}")
print(f"Top tools: {agg.top_tools}")
print(f"Top agents: {agg.top_agents}")
print(f"Avg effectiveness: {agg.avg_effectiveness:.1f}/10")
```

---

## 12. Long-Horizon Agent Memory

Give agents memory across sessions using historical trace data.

### Cross-Session Context Retrieval

```python
from bigquery_agent_analytics import BigQueryMemoryService

memory = BigQueryMemoryService(
    project_id="my-project",
    dataset_id="agent_analytics",
)

# Get relevant past interactions for a user
episodes = await memory.get_session_context(
    user_id="user-abc",
    current_session_id="sess-current",
    lookback_sessions=5,
)

for ep in episodes:
    print(f"[{ep.timestamp}] User: {ep.user_message}")
    print(f"  Agent: {ep.agent_response}")
    print(f"  Tools: {ep.tool_calls}")
```

### Semantic Memory Search

```python
# Search for relevant past episodes by semantic similarity
results = await memory.search_memory(
    app_name="support_bot",
    user_id="user-abc",
    query="How do I reset my password?",
)

for entry in results.memories:
    print(f"  {entry.key}: {entry.value[:100]}...")
```

### User Profile Building

```python
from bigquery_agent_analytics import UserProfileBuilder

builder = UserProfileBuilder(
    project_id="my-project",
    dataset_id="agent_analytics",
)

profile = await builder.build_profile(user_id="user-abc")

print(f"Topics: {profile.topics_of_interest}")
print(f"Style: {profile.communication_style}")
print(f"Common requests: {profile.common_requests}")
print(f"Preferred tools: {profile.preferred_tools}")
print(f"Sessions: {profile.session_count}")
```

### Context Management

Prevent cognitive overload by selecting only the most relevant memories:

```python
from bigquery_agent_analytics import ContextManager

ctx_mgr = ContextManager(
    max_context_tokens=32000,
    relevance_weight=0.7,
    recency_weight=0.3,
)

# Select the best memories given token budget
relevant = ctx_mgr.select_relevant_context(
    current_task="How do I change my subscription plan?",
    available_memories=episodes,
    current_context_tokens=5000,
)

# Summarize older context to save tokens
summary, recent = await ctx_mgr.summarize_old_context(
    memories=episodes,
    preserve_recent=5,
)
```

---

## 13. BigQuery AI/ML Integration

Direct access to BigQuery's native AI capabilities for advanced analytics.

### Text Generation with AI.GENERATE

```python
from bigquery_agent_analytics import BigQueryAIClient

ai_client = BigQueryAIClient(
    project_id="my-project",
    dataset_id="agent_analytics",
    endpoint="gemini-2.5-flash",
)

# Generate text using BigQuery AI.GENERATE
response = await ai_client.generate_text(
    prompt="Summarize the top issues from these agent logs: ...",
    temperature=0.3,
    max_tokens=1024,
)
```

### Embedding-Based Semantic Search

```python
from bigquery_agent_analytics import EmbeddingSearchClient

search_client = EmbeddingSearchClient(
    project_id="my-project",
    dataset_id="agent_analytics",
    embeddings_table="trace_embeddings",
)

# Build or refresh the embeddings index
await search_client.build_embeddings_index(since_days=30)

# Search by vector similarity
results = await search_client.search(
    query_embedding=[0.1, 0.2, ...],  # your query embedding
    top_k=10,
    user_id="user-abc",
    since_days=7,
)
```

### Anomaly Detection

```python
from bigquery_agent_analytics import AnomalyDetector

detector = AnomalyDetector(
    project_id="my-project",
    dataset_id="agent_analytics",
)

# Train and detect latency anomalies (ARIMA model)
await detector.train_latency_model(training_days=30)
anomalies = await detector.detect_latency_anomalies(since_hours=24)

for a in anomalies:
    print(f"[{a.anomaly_type.value}] {a.description} "
          f"(severity={a.severity:.2f})")

# Train and detect behavioral anomalies (Autoencoder model)
await detector.train_behavior_model()
behavior_anomalies = await detector.detect_behavior_anomalies(since_hours=24)
```

### Batch Evaluation with AI.GENERATE

```python
from bigquery_agent_analytics import BatchEvaluator

batch_eval = BatchEvaluator(
    project_id="my-project",
    dataset_id="agent_analytics",
    endpoint="gemini-2.5-flash",
)

# Evaluate recent sessions using AI.GENERATE with typed output
results = await batch_eval.evaluate_recent_sessions(
    days=1,
    limit=100,
)

for r in results:
    print(f"{r.session_id}: completion={r.task_completion:.0f}, "
          f"efficiency={r.efficiency:.0f}, tool_usage={r.tool_usage:.0f}")

# Persist evaluation results to BigQuery
await batch_eval.store_evaluation_results(
    results,
    table_name="session_evaluations",
)
```

---

## 14. BigFrames Evaluator (DataFrame API)

For notebook-friendly workflows, `BigFramesEvaluator` returns pandas-compatible DataFrames powered by BigFrames.

```bash
pip install bigquery-agent-analytics[bigframes]
```

### Evaluate Sessions as a DataFrame

```python
from bigquery_agent_analytics import BigFramesEvaluator

bf_eval = BigFramesEvaluator(
    project_id="my-project",
    dataset_id="agent_analytics",
    endpoint="gemini-2.5-flash",
)

# Returns a BigFrames DataFrame with score + justification columns
df = bf_eval.evaluate_sessions(
    max_sessions=50,
    judge_prompt="Rate this agent session 1-10 for helpfulness.",
)

print(df.head())
# session_id  | score | justification
# sess-001    | 8     | Accurate and helpful response...
# sess-002    | 3     | Agent misunderstood the query...
```

### Extract Session Facets as a DataFrame

```python
facets_df = bf_eval.extract_facets(
    session_ids=["sess-001", "sess-002", "sess-003"],
    max_sessions=50,
)

print(facets_df.columns.tolist())
# ['session_id', 'goal_categories', 'outcome', 'satisfaction',
#  'friction_types', 'session_type', 'agent_effectiveness',
#  'primary_success', 'key_topics', 'summary']
```

---

## Module Architecture

```
bigquery_agent_analytics/
│
│   Core
│   ├── client.py              ← High-level SDK entry point
│   ├── trace.py               ← Trace/Span reconstruction & DAG rendering
│   └── evaluators.py          ← CodeEvaluator + LLMAsJudge + SQL templates
│
│   Evaluation Harness
│   ├── trace_evaluator.py     ← BigQueryTraceEvaluator, trajectory matching, replay
│   ├── multi_trial.py         ← TrialRunner, pass@k, pass^k
│   ├── grader_pipeline.py     ← GraderPipeline + scoring strategies
│   ├── eval_suite.py          ← EvalSuite lifecycle management
│   └── eval_validator.py      ← Static validation checks
│
│   Feedback & Insights
│   ├── feedback.py            ← Drift detection, question distribution
│   └── insights.py            ← Multi-stage insights pipeline
│
│   AI/ML & Memory
│   ├── ai_ml_integration.py   ← AI.GENERATE, embeddings, anomaly detection
│   ├── memory_service.py      ← Long-horizon agent memory (requires google-adk)
│   └── bigframes_evaluator.py ← BigFrames DataFrame evaluator (optional)
```

### Dependency Graph

```
Standalone modules (no internal imports):
├── trace.py
├── evaluators.py
├── trace_evaluator.py
├── feedback.py
├── ai_ml_integration.py
├── bigframes_evaluator.py
└── eval_suite.py

Modules with internal imports:
├── insights.py         → evaluators
├── grader_pipeline.py  → evaluators
├── multi_trial.py      → trace_evaluator
├── eval_validator.py   → eval_suite
└── client.py           → evaluators, feedback, insights, trace

External dependency:
└── memory_service.py   → google-adk (memory + sessions)
```
