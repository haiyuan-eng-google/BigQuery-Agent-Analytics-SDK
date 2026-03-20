# Python UDF Deployment

Register SDK analytical kernels as BigQuery Python UDFs for direct
in-engine execution with no Cloud Function required.

## Prerequisites

- BigQuery Python UDF support enabled (Preview)
- A BigQuery dataset to host the UDFs

## Quick Start

### Option 1: Run the static SQL

Replace `PROJECT` and `UDF_DATASET` in `register.sql`, then execute:

```bash
bq query --use_legacy_sql=false < register.sql
```

### Option 2: Generate SQL programmatically

```python
from bigquery_agent_analytics.udf_sql_templates import generate_all_udfs

sql = generate_all_udfs("my-project", "analytics")
print(sql)
```

Or generate a single UDF:

```python
from bigquery_agent_analytics.udf_sql_templates import generate_udf

sql = generate_udf("bqaa_score_latency", "my-project", "analytics")
```

## Available UDFs

### Tier 1: Event Semantics

| Function | Params | Returns | Description |
|----------|--------|---------|-------------|
| `bqaa_is_error_event` | `event_type, error_message, status` | `BOOL` | Error detection |
| `bqaa_tool_outcome` | `event_type, status` | `STRING` | Tool outcome classification |
| `bqaa_extract_response_text` | `content_json` | `STRING` | Response text extraction |
| `bqaa_normalize_event_label` | `event_type` | `STRING` | Event type normalization |

### Tier 2: Score Kernels

| Function | Params | Returns | Description |
|----------|--------|---------|-------------|
| `bqaa_score_latency` | `avg_latency_ms, threshold_ms` | `FLOAT64` | Latency scoring |
| `bqaa_score_error_rate` | `tool_calls, tool_errors, max_error_rate` | `FLOAT64` | Error rate scoring |
| `bqaa_score_turn_count` | `turn_count, max_turns` | `FLOAT64` | Turn count scoring |
| `bqaa_score_token_efficiency` | `total_tokens, max_tokens` | `FLOAT64` | Token efficiency scoring |
| `bqaa_score_ttft` | `avg_ttft_ms, threshold_ms` | `FLOAT64` | Time-to-first-token scoring |
| `bqaa_score_cost` | `input_tokens, output_tokens, max_cost_usd, input_cost_per_1k, output_cost_per_1k` | `FLOAT64` | Cost scoring |

All score kernels return a value in `[0.0, 1.0]` where 1.0 is best.

### Tier 3: Vectorized UDFs (Deferred)

Vectorized Python UDFs (`OPTIONS(vectorized = true)`) are deferred
until BigQuery adds `vectorized` option support for Python UDFs.  The
option is currently only supported for JavaScript UDFs.  When support
lands, batch-oriented scoring UDFs using numpy/pandas will be added.

### Tier 4: STRING Envelope UDFs

UDFs that return a JSON `STRING` for richer structured output.

| Function | Params | Returns | Description |
|----------|--------|---------|-------------|
| `bqaa_eval_summary_json` | `avg_latency_ms, tool_calls, tool_errors, turn_count, total_tokens, avg_ttft_ms, input_tokens, output_tokens, threshold_ms, max_error_rate, max_turns, max_tokens, ttft_threshold_ms, max_cost_usd, input_cost_per_1k, output_cost_per_1k` | `STRING` | All six scores + pass/fail in one JSON object |

Use `JSON_VALUE()` to extract individual scores from the result:

```sql
JSON_VALUE(summary, '$.latency')   -- individual score
JSON_VALUE(summary, '$.passed')    -- overall pass/fail
```

## Region Guidance

BigQuery UDFs are region-scoped. If your data lives in multiple
regions, register UDFs in each region or use dataset replication for
a shared utility dataset.

## Examples

- [python_udf_evaluation.sql](../../examples/python_udf_evaluation.sql) —
  Session scoring with SQL pre-aggregation + UDF score kernels
- [python_udf_event_semantics.sql](../../examples/python_udf_event_semantics.sql) —
  Event classification and response extraction
- [python_udf_eval_summary.sql](../../examples/python_udf_eval_summary.sql) —
  All-in-one session evaluation with JSON STRING envelope
