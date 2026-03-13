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

"""CLI entry point for BigQuery Agent Analytics SDK.

Usage::

    bq-agent-sdk doctor --project-id=P --dataset-id=D
    bq-agent-sdk get-trace --project-id=P --dataset-id=D --session-id=S
    bq-agent-sdk evaluate --project-id=P --dataset-id=D --evaluator=latency
"""

from __future__ import annotations

import sys
from typing import Optional

import typer

from .evaluators import CodeEvaluator
from .evaluators import LLMAsJudge
from .formatter import format_output
from .trace import TraceFilter

app = typer.Typer(
    name="bq-agent-sdk",
    help="BigQuery Agent Analytics SDK CLI.",
    add_completion=False,
)


# ------------------------------------------------------------------ #
# Shared options                                                       #
# ------------------------------------------------------------------ #

_PROJECT_HELP = "GCP project ID. [env: BQ_AGENT_PROJECT]"
_DATASET_HELP = "BigQuery dataset. [env: BQ_AGENT_DATASET]"


def _build_client(
    project_id: str,
    dataset_id: str,
    table_id: str,
    location: str,
    endpoint: Optional[str] = None,
    connection_id: Optional[str] = None,
):
  """Lazily import Client and construct an instance."""
  from .client import Client  # defer heavy import

  return Client(
      project_id=project_id,
      dataset_id=dataset_id,
      table_id=table_id,
      location=location,
      verify_schema=False,
      endpoint=endpoint,
      connection_id=connection_id,
  )


# ------------------------------------------------------------------ #
# Evaluator factories                                                  #
# ------------------------------------------------------------------ #

_CODE_EVALUATORS = {
    "latency": (
        lambda t: CodeEvaluator.latency(threshold_ms=t),
        lambda: CodeEvaluator.latency(),
    ),
    "error_rate": (
        lambda t: CodeEvaluator.error_rate(max_error_rate=t),
        lambda: CodeEvaluator.error_rate(),
    ),
    "turn_count": (
        lambda t: CodeEvaluator.turn_count(max_turns=int(t)),
        lambda: CodeEvaluator.turn_count(),
    ),
    "token_efficiency": (
        lambda t: CodeEvaluator.token_efficiency(max_tokens=int(t)),
        lambda: CodeEvaluator.token_efficiency(),
    ),
    "ttft": (
        lambda t: CodeEvaluator.ttft(threshold_ms=t),
        lambda: CodeEvaluator.ttft(),
    ),
    "cost": (
        lambda t: CodeEvaluator.cost_per_session(max_cost_usd=t),
        lambda: CodeEvaluator.cost_per_session(),
    ),
}

_LLM_JUDGES = {
    "correctness": (
        lambda t: LLMAsJudge.correctness(threshold=t),
        lambda: LLMAsJudge.correctness(),
    ),
    "hallucination": (
        lambda t: LLMAsJudge.hallucination(threshold=t),
        lambda: LLMAsJudge.hallucination(),
    ),
    "sentiment": (
        lambda t: LLMAsJudge.sentiment(threshold=t),
        lambda: LLMAsJudge.sentiment(),
    ),
}


# ------------------------------------------------------------------ #
# Commands                                                             #
# ------------------------------------------------------------------ #


@app.command()
def doctor(
    project_id: str = typer.Option(
        ..., envvar="BQ_AGENT_PROJECT", help=_PROJECT_HELP
    ),
    dataset_id: str = typer.Option(
        ..., envvar="BQ_AGENT_DATASET", help=_DATASET_HELP
    ),
    table_id: str = typer.Option("agent_events", help="Events table name."),
    location: str = typer.Option("us-central1", help="BQ location."),
    fmt: str = typer.Option(
        "json",
        "--format",
        help="Output format: json|text|table.",
    ),
) -> None:
  """Run diagnostic health check."""
  try:
    client = _build_client(project_id, dataset_id, table_id, location)
    result = client.doctor()
    typer.echo(format_output(result, fmt))
  except Exception as exc:
    typer.echo(f"Error: {exc}", err=True)
    raise typer.Exit(code=2)


@app.command("get-trace")
def get_trace(
    project_id: str = typer.Option(
        ..., envvar="BQ_AGENT_PROJECT", help=_PROJECT_HELP
    ),
    dataset_id: str = typer.Option(
        ..., envvar="BQ_AGENT_DATASET", help=_DATASET_HELP
    ),
    table_id: str = typer.Option("agent_events", help="Events table name."),
    location: str = typer.Option("us-central1", help="BQ location."),
    session_id: Optional[str] = typer.Option(
        None, help="Retrieve by session ID."
    ),
    trace_id: Optional[str] = typer.Option(None, help="Retrieve by trace ID."),
    fmt: str = typer.Option(
        "json",
        "--format",
        help="Output format: json|text|table.",
    ),
) -> None:
  """Retrieve and display a session trace."""
  if not session_id and not trace_id:
    typer.echo(
        "Error: provide --session-id or --trace-id.",
        err=True,
    )
    raise typer.Exit(code=2)
  try:
    client = _build_client(project_id, dataset_id, table_id, location)
    if session_id:
      trace = client.get_session_trace(session_id)
    else:
      trace = client.get_trace(trace_id)
    typer.echo(format_output(trace, fmt))
  except Exception as exc:
    typer.echo(f"Error: {exc}", err=True)
    raise typer.Exit(code=2)


@app.command()
def evaluate(
    project_id: str = typer.Option(
        ..., envvar="BQ_AGENT_PROJECT", help=_PROJECT_HELP
    ),
    dataset_id: str = typer.Option(
        ..., envvar="BQ_AGENT_DATASET", help=_DATASET_HELP
    ),
    table_id: str = typer.Option("agent_events", help="Events table name."),
    location: str = typer.Option("us-central1", help="BQ location."),
    evaluator: str = typer.Option(
        "latency",
        help=(
            "Evaluator: latency|error_rate|turn_count|"
            "token_efficiency|ttft|cost|llm-judge."
        ),
    ),
    threshold: Optional[float] = typer.Option(
        None, help="Pass/fail threshold (uses evaluator default if omitted)."
    ),
    criterion: str = typer.Option(
        "correctness",
        help=("LLM judge criterion: " "correctness|hallucination|sentiment."),
    ),
    agent_id: Optional[str] = typer.Option(None, help="Filter by agent name."),
    last: Optional[str] = typer.Option(
        None,
        help="Time window: 30m, 1h, 24h, 7d, 30d.",
    ),
    limit: int = typer.Option(100, help="Max sessions to evaluate."),
    exit_code: bool = typer.Option(
        False,
        "--exit-code",
        help="Return exit code 1 on evaluation failure.",
    ),
    strict: bool = typer.Option(
        False,
        help="Fail sessions with unparseable judge output.",
    ),
    endpoint: Optional[str] = typer.Option(
        None,
        help="AI.GENERATE endpoint for LLM judge.",
    ),
    connection_id: Optional[str] = typer.Option(
        None,
        help="BQ connection ID for AI.GENERATE.",
    ),
    fmt: str = typer.Option(
        "json",
        "--format",
        help="Output format: json|text|table.",
    ),
) -> None:
  """Run code-based or LLM evaluation over traces."""
  try:
    filters = TraceFilter.from_cli_args(
        last=last,
        agent_id=agent_id,
        limit=limit,
    )

    if evaluator == "llm-judge":
      entry = _LLM_JUDGES.get(criterion)
      if not entry:
        typer.echo(
            f"Error: unknown criterion: {criterion!r}.",
            err=True,
        )
        raise typer.Exit(code=2)
      with_t, without_t = entry
      ev = with_t(threshold) if threshold is not None else without_t()
    else:
      entry = _CODE_EVALUATORS.get(evaluator)
      if not entry:
        typer.echo(
            f"Error: unknown evaluator: {evaluator!r}.",
            err=True,
        )
        raise typer.Exit(code=2)
      with_t, without_t = entry
      ev = with_t(threshold) if threshold is not None else without_t()

    client = _build_client(
        project_id,
        dataset_id,
        table_id,
        location,
        endpoint=endpoint,
        connection_id=connection_id,
    )
    report = client.evaluate(evaluator=ev, filters=filters, strict=strict)
    typer.echo(format_output(report, fmt))

    if exit_code and report.pass_rate < 1.0:
      raise typer.Exit(code=1)
  except typer.Exit:
    raise
  except Exception as exc:
    typer.echo(f"Error: {exc}", err=True)
    raise typer.Exit(code=2)


def main() -> None:
  """Entry point for ``bq-agent-sdk``."""
  app()


if __name__ == "__main__":
  main()
