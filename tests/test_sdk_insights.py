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

"""Tests for the SDK insights module."""

from datetime import datetime
from datetime import timezone
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from bigquery_agent_analytics.insights import _AI_GENERATE_ANALYSIS_QUERY
from bigquery_agent_analytics.insights import _AI_GENERATE_FACET_EXTRACTION_QUERY
from bigquery_agent_analytics.insights import aggregate_facets
from bigquery_agent_analytics.insights import AggregatedInsights
from bigquery_agent_analytics.insights import ANALYSIS_PROMPTS
from bigquery_agent_analytics.insights import AnalysisSection
from bigquery_agent_analytics.insights import build_analysis_context
from bigquery_agent_analytics.insights import build_facet_prompt
from bigquery_agent_analytics.insights import FRICTION_TYPES
from bigquery_agent_analytics.insights import GOAL_CATEGORIES
from bigquery_agent_analytics.insights import InsightsConfig
from bigquery_agent_analytics.insights import InsightsReport
from bigquery_agent_analytics.insights import OUTCOMES
from bigquery_agent_analytics.insights import parse_facet_from_ai_generate_row
from bigquery_agent_analytics.insights import parse_facet_response
from bigquery_agent_analytics.insights import SATISFACTION_LEVELS
from bigquery_agent_analytics.insights import SESSION_TYPES
from bigquery_agent_analytics.insights import SessionFacet
from bigquery_agent_analytics.insights import SessionMetadata


class TestSessionFacet:
  """Tests for SessionFacet model."""

  def test_defaults(self):
    facet = SessionFacet(session_id="s1")
    assert facet.session_id == "s1"
    assert facet.outcome == "unclear"
    assert facet.satisfaction == "unknown"
    assert facet.session_type == "question_answer"
    assert facet.agent_effectiveness == 5.0
    assert facet.primary_success is False
    assert facet.goal_categories == []
    assert facet.friction_types == []

  def test_full_facet(self):
    facet = SessionFacet(
        session_id="s1",
        goal_categories=["question_answering", "analysis"],
        outcome="success",
        satisfaction="satisfied",
        friction_types=["slow_response"],
        session_type="task_execution",
        agent_effectiveness=8.5,
        primary_success=True,
        key_topics=["data analysis", "SQL"],
        summary="User asked about data analysis.",
    )
    assert facet.outcome == "success"
    assert len(facet.goal_categories) == 2
    assert facet.agent_effectiveness == 8.5
    assert facet.primary_success is True


class TestSessionMetadata:
  """Tests for SessionMetadata model."""

  def test_defaults(self):
    meta = SessionMetadata(session_id="s1")
    assert meta.event_count == 0
    assert meta.tool_calls == 0
    assert meta.has_error is False

  def test_full_metadata(self):
    meta = SessionMetadata(
        session_id="s1",
        event_count=20,
        tool_calls=5,
        tool_errors=1,
        llm_calls=3,
        turn_count=4,
        total_latency_ms=5000.0,
        avg_latency_ms=250.0,
        agents_used=["agent_a", "agent_b"],
        tools_used=["search", "calculator"],
        has_error=True,
    )
    assert meta.event_count == 20
    assert meta.tool_calls == 5
    assert len(meta.agents_used) == 2


class TestParseFacetResponse:
  """Tests for parse_facet_response function."""

  def test_valid_json(self):
    text = (
        '{"goal_categories": ["question_answering"],'
        ' "outcome": "success",'
        ' "satisfaction": "satisfied",'
        ' "friction_types": [],'
        ' "session_type": "question_answer",'
        ' "agent_effectiveness": 8,'
        ' "primary_success": true,'
        ' "key_topics": ["weather"],'
        ' "summary": "Asked about weather."}'
    )
    facet = parse_facet_response("s1", text)
    assert facet.session_id == "s1"
    assert facet.outcome == "success"
    assert facet.satisfaction == "satisfied"
    assert facet.agent_effectiveness == 8.0
    assert facet.primary_success is True
    assert "weather" in facet.key_topics

  def test_invalid_goal_filtered(self):
    text = (
        '{"goal_categories": ["invalid_cat", "analysis"], "outcome": "success"}'
    )
    facet = parse_facet_response("s1", text)
    assert "analysis" in facet.goal_categories
    assert "invalid_cat" not in facet.goal_categories

  def test_all_invalid_goals_default_to_other(self):
    text = '{"goal_categories": ["bogus"]}'
    facet = parse_facet_response("s1", text)
    assert facet.goal_categories == ["other"]

  def test_invalid_outcome_defaults(self):
    text = '{"outcome": "invalid_outcome"}'
    facet = parse_facet_response("s1", text)
    assert facet.outcome == "unclear"

  def test_invalid_satisfaction_defaults(self):
    text = '{"satisfaction": "ecstatic"}'
    facet = parse_facet_response("s1", text)
    assert facet.satisfaction == "unknown"

  def test_invalid_session_type_defaults(self):
    text = '{"session_type": "unknown_type"}'
    facet = parse_facet_response("s1", text)
    assert facet.session_type == "question_answer"

  def test_effectiveness_clamping(self):
    text = '{"agent_effectiveness": 15}'
    facet = parse_facet_response("s1", text)
    assert facet.agent_effectiveness == 10.0

    text2 = '{"agent_effectiveness": -5}'
    facet2 = parse_facet_response("s2", text2)
    assert facet2.agent_effectiveness == 1.0

  def test_empty_response(self):
    facet = parse_facet_response("s1", "")
    assert facet.session_id == "s1"
    assert facet.outcome == "unclear"

  def test_invalid_json(self):
    facet = parse_facet_response("s1", "not json at all")
    assert facet.session_id == "s1"
    assert facet.outcome == "unclear"

  def test_json_in_code_block(self):
    text = '```json\n{"goal_categories": ["search"], "outcome": "success"}\n```'
    facet = parse_facet_response("s1", text)
    assert facet.outcome == "success"

  def test_friction_types_filtered(self):
    text = '{"friction_types": ["tool_error", "fake_friction", "timeout"]}'
    facet = parse_facet_response("s1", text)
    assert "tool_error" in facet.friction_types
    assert "timeout" in facet.friction_types
    assert "fake_friction" not in facet.friction_types

  def test_key_topics_limited(self):
    text = '{"key_topics": ["a", "b", "c", "d", "e", "f", "g"]}'
    facet = parse_facet_response("s1", text)
    assert len(facet.key_topics) <= 5

  def test_summary_truncated(self):
    long_summary = "x" * 300
    text = '{"summary": "' + long_summary + '"}'
    facet = parse_facet_response("s1", text)
    assert len(facet.summary) <= 200


class TestAggregateFacets:
  """Tests for aggregate_facets function."""

  def _make_facets(self):
    return [
        SessionFacet(
            session_id="s1",
            goal_categories=["question_answering"],
            outcome="success",
            satisfaction="satisfied",
            friction_types=[],
            session_type="question_answer",
            agent_effectiveness=8.0,
            primary_success=True,
            key_topics=["weather"],
        ),
        SessionFacet(
            session_id="s2",
            goal_categories=[
                "task_automation",
                "code_generation",
            ],
            outcome="partial_success",
            satisfaction="neutral",
            friction_types=["slow_response", "tool_error"],
            session_type="task_execution",
            agent_effectiveness=6.0,
            primary_success=False,
            key_topics=["automation", "code"],
        ),
        SessionFacet(
            session_id="s3",
            goal_categories=["question_answering"],
            outcome="failure",
            satisfaction="dissatisfied",
            friction_types=["wrong_answer"],
            session_type="question_answer",
            agent_effectiveness=3.0,
            primary_success=False,
            key_topics=["weather"],
        ),
    ]

  def _make_metadata(self):
    return [
        SessionMetadata(
            session_id="s1",
            event_count=10,
            tool_calls=3,
            tool_errors=0,
            turn_count=2,
            avg_latency_ms=200.0,
            agents_used=["agent_a"],
            tools_used=["search"],
            has_error=False,
        ),
        SessionMetadata(
            session_id="s2",
            event_count=20,
            tool_calls=5,
            tool_errors=2,
            turn_count=4,
            avg_latency_ms=500.0,
            agents_used=["agent_a", "agent_b"],
            tools_used=["search", "code_exec"],
            has_error=True,
        ),
        SessionMetadata(
            session_id="s3",
            event_count=8,
            tool_calls=2,
            tool_errors=1,
            turn_count=1,
            avg_latency_ms=100.0,
            agents_used=["agent_a"],
            tools_used=["search"],
            has_error=True,
        ),
    ]

  def test_total_sessions(self):
    agg = aggregate_facets(self._make_facets(), self._make_metadata())
    assert agg.total_sessions == 3

  def test_goal_distribution(self):
    agg = aggregate_facets(self._make_facets(), self._make_metadata())
    assert agg.goal_distribution["question_answering"] == 2
    assert agg.goal_distribution["task_automation"] == 1
    assert agg.goal_distribution["code_generation"] == 1

  def test_outcome_distribution(self):
    agg = aggregate_facets(self._make_facets(), self._make_metadata())
    assert agg.outcome_distribution["success"] == 1
    assert agg.outcome_distribution["partial_success"] == 1
    assert agg.outcome_distribution["failure"] == 1

  def test_satisfaction_distribution(self):
    agg = aggregate_facets(self._make_facets(), self._make_metadata())
    assert agg.satisfaction_distribution["satisfied"] == 1
    assert agg.satisfaction_distribution["neutral"] == 1
    assert agg.satisfaction_distribution["dissatisfied"] == 1

  def test_friction_distribution(self):
    agg = aggregate_facets(self._make_facets(), self._make_metadata())
    assert agg.friction_distribution["slow_response"] == 1
    assert agg.friction_distribution["tool_error"] == 1
    assert agg.friction_distribution["wrong_answer"] == 1

  def test_success_rate(self):
    agg = aggregate_facets(self._make_facets(), self._make_metadata())
    assert abs(agg.success_rate - 1.0 / 3.0) < 0.01

  def test_avg_effectiveness(self):
    agg = aggregate_facets(self._make_facets(), self._make_metadata())
    expected = (8.0 + 6.0 + 3.0) / 3.0
    assert abs(agg.avg_effectiveness - expected) < 0.01

  def test_avg_latency(self):
    agg = aggregate_facets(self._make_facets(), self._make_metadata())
    expected = (200.0 + 500.0 + 100.0) / 3.0
    assert abs(agg.avg_latency_ms - expected) < 0.1

  def test_error_rate(self):
    agg = aggregate_facets(self._make_facets(), self._make_metadata())
    assert abs(agg.error_rate - 2.0 / 3.0) < 0.01

  def test_top_topics(self):
    agg = aggregate_facets(self._make_facets(), self._make_metadata())
    topic_names = [t[0] for t in agg.top_topics]
    assert "weather" in topic_names

  def test_top_tools(self):
    agg = aggregate_facets(self._make_facets(), self._make_metadata())
    tool_names = [t[0] for t in agg.top_tools]
    assert "search" in tool_names

  def test_empty_input(self):
    agg = aggregate_facets([], [])
    assert agg.total_sessions == 0
    assert agg.success_rate == 0.0


class TestInsightsReport:
  """Tests for InsightsReport model."""

  def test_defaults(self):
    report = InsightsReport()
    assert report.total_sessions == 0
    assert report.success_rate == 0.0
    assert report.executive_summary == ""

  def test_summary_output(self):
    agg = AggregatedInsights(
        total_sessions=10,
        success_rate=0.7,
        avg_effectiveness=7.5,
        avg_latency_ms=300.0,
        avg_turns=3.2,
        error_rate=0.1,
        goal_distribution={
            "question_answering": 5,
            "analysis": 3,
        },
        outcome_distribution={
            "success": 7,
            "failure": 3,
        },
        friction_distribution={
            "slow_response": 2,
            "tool_error": 1,
        },
    )
    report = InsightsReport(
        aggregated=agg,
        analysis_sections=[
            AnalysisSection(
                title="Task Areas",
                content="Users mostly ask questions.",
            ),
        ],
    )
    text = report.summary()
    assert "10" in text
    assert "70%" in text
    assert "7.5" in text
    assert "Task Areas" in text

  def test_get_section(self):
    report = InsightsReport(
        analysis_sections=[
            AnalysisSection(
                title="Task Areas",
                content="Content here.",
            ),
            AnalysisSection(
                title="Friction Analysis",
                content="Friction content.",
            ),
        ],
    )
    section = report.get_section("Task Areas")
    assert section is not None
    assert section.content == "Content here."

  def test_get_section_not_found(self):
    report = InsightsReport()
    assert report.get_section("Missing") is None

  def test_get_section_case_insensitive(self):
    report = InsightsReport(
        analysis_sections=[
            AnalysisSection(
                title="Task Areas",
                content="Content.",
            ),
        ],
    )
    assert report.get_section("task areas") is not None


class TestInsightsConfig:
  """Tests for InsightsConfig model."""

  def test_defaults(self):
    config = InsightsConfig()
    assert config.max_sessions == 50
    assert config.min_events_per_session == 3
    assert config.min_turns_per_session == 1
    assert config.include_sub_sessions is False
    assert config.analysis_prompts is None

  def test_custom_config(self):
    config = InsightsConfig(
        max_sessions=100,
        min_events_per_session=5,
        analysis_prompts=["task_areas", "friction_analysis"],
    )
    assert config.max_sessions == 100
    assert len(config.analysis_prompts) == 2


class TestBuildFacetPrompt:
  """Tests for build_facet_prompt function."""

  def test_contains_categories(self):
    prompt = build_facet_prompt()
    for cat in GOAL_CATEGORIES:
      assert cat in prompt

  def test_contains_outcomes(self):
    prompt = build_facet_prompt()
    for o in OUTCOMES:
      assert o in prompt

  def test_contains_satisfaction(self):
    prompt = build_facet_prompt()
    for s in SATISFACTION_LEVELS:
      assert s in prompt

  def test_contains_friction(self):
    prompt = build_facet_prompt()
    for f in FRICTION_TYPES:
      assert f in prompt

  def test_contains_session_types(self):
    prompt = build_facet_prompt()
    for s in SESSION_TYPES:
      assert s in prompt


class TestBuildAnalysisContext:
  """Tests for build_analysis_context function."""

  def _make_data(self):
    facets = [
        SessionFacet(
            session_id="s1",
            goal_categories=["analysis"],
            outcome="success",
            agent_effectiveness=8.0,
            primary_success=True,
            key_topics=["data"],
        ),
        SessionFacet(
            session_id="s2",
            goal_categories=["troubleshooting"],
            outcome="failure",
            agent_effectiveness=3.0,
            primary_success=False,
            key_topics=["error"],
        ),
    ]
    metadata = [
        SessionMetadata(
            session_id="s1",
            tool_calls=5,
            tool_errors=0,
            turn_count=3,
            avg_latency_ms=200.0,
            agents_used=["agent_a"],
            tools_used=["search"],
            has_error=False,
            start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        ),
        SessionMetadata(
            session_id="s2",
            tool_calls=3,
            tool_errors=2,
            turn_count=5,
            avg_latency_ms=800.0,
            agents_used=["agent_a"],
            tools_used=["search", "debug"],
            has_error=True,
            start_time=datetime(2024, 1, 15, tzinfo=timezone.utc),
        ),
    ]
    agg = aggregate_facets(facets, metadata)
    return agg, facets, metadata

  def test_context_keys(self):
    agg, facets, metadata = self._make_data()
    ctx = build_analysis_context(agg, facets, metadata)
    expected_keys = [
        "total",
        "goal_dist",
        "outcome_dist",
        "satisfaction_dist",
        "friction_dist",
        "session_type_dist",
        "top_topics",
        "top_tools",
        "top_agents",
        "success_rate",
        "avg_effectiveness",
        "avg_latency",
        "avg_turns",
        "error_rate",
        "successful_goals",
        "failed_topics",
        "avg_tool_calls",
        "tool_error_rate",
        "error_tools",
        "underused_tools",
        "low_success_goals",
        "time_range",
    ]
    for key in expected_keys:
      assert key in ctx, f"Missing key: {key}"

  def test_context_values(self):
    agg, facets, metadata = self._make_data()
    ctx = build_analysis_context(agg, facets, metadata)
    assert ctx["total"] == 2
    assert ctx["success_rate"] == 0.5
    assert "2024-01-01" in ctx["time_range"]


class TestAnalysisPrompts:
  """Tests for the analysis prompt definitions."""

  def test_all_prompts_defined(self):
    expected = [
        "task_areas",
        "interaction_patterns",
        "what_works_well",
        "friction_analysis",
        "tool_usage",
        "suggestions",
        "trends",
    ]
    for name in expected:
      assert name in ANALYSIS_PROMPTS

  def test_prompts_have_title_and_prompt(self):
    for name, config in ANALYSIS_PROMPTS.items():
      assert "title" in config, f"{name} missing title"
      assert "prompt" in config, f"{name} missing prompt"
      assert len(config["title"]) > 0
      assert len(config["prompt"]) > 0


class TestSchemaConstants:
  """Tests for schema constants."""

  def test_goal_categories_not_empty(self):
    assert len(GOAL_CATEGORIES) >= 10

  def test_outcomes_not_empty(self):
    assert len(OUTCOMES) >= 4

  def test_satisfaction_levels_not_empty(self):
    assert len(SATISFACTION_LEVELS) >= 5

  def test_friction_types_not_empty(self):
    assert len(FRICTION_TYPES) >= 10

  def test_session_types_not_empty(self):
    assert len(SESSION_TYPES) >= 4


class TestParseFacetFromAIGenerateRow:
  """Tests for parse_facet_from_ai_generate_row function."""

  def test_valid_typed_row(self):
    row = {
        "goal_categories": ["question_answering", "analysis"],
        "outcome": "success",
        "satisfaction": "satisfied",
        "friction_types": ["slow_response"],
        "session_type": "task_execution",
        "agent_effectiveness": 8,
        "primary_success": True,
        "key_topics": ["weather", "data"],
        "summary": "User asked about weather data.",
    }
    facet = parse_facet_from_ai_generate_row("s1", row)
    assert facet.session_id == "s1"
    assert facet.outcome == "success"
    assert facet.satisfaction == "satisfied"
    assert facet.agent_effectiveness == 8.0
    assert facet.primary_success is True
    assert "weather" in facet.key_topics

  def test_null_columns_use_defaults(self):
    row = {
        "goal_categories": None,
        "outcome": None,
        "satisfaction": None,
        "friction_types": None,
        "session_type": None,
        "agent_effectiveness": None,
        "primary_success": None,
        "key_topics": None,
        "summary": None,
    }
    facet = parse_facet_from_ai_generate_row("s1", row)
    assert facet.goal_categories == ["other"]
    assert facet.outcome == "unclear"
    assert facet.satisfaction == "unknown"
    assert facet.friction_types == []
    assert facet.session_type == "question_answer"
    assert facet.agent_effectiveness == 5.0
    assert facet.primary_success is False
    assert facet.key_topics == []
    assert facet.summary == ""

  def test_invalid_enum_values_default(self):
    row = {
        "goal_categories": ["bogus"],
        "outcome": "invalid_outcome",
        "satisfaction": "ecstatic",
        "friction_types": ["fake_friction"],
        "session_type": "unknown_type",
        "agent_effectiveness": 5,
        "primary_success": False,
        "key_topics": [],
        "summary": "",
    }
    facet = parse_facet_from_ai_generate_row("s1", row)
    assert facet.goal_categories == ["other"]
    assert facet.outcome == "unclear"
    assert facet.satisfaction == "unknown"
    assert facet.friction_types == []
    assert facet.session_type == "question_answer"

  def test_effectiveness_clamping(self):
    row_high = {"agent_effectiveness": 15}
    facet_high = parse_facet_from_ai_generate_row("s1", row_high)
    assert facet_high.agent_effectiveness == 10.0

    row_low = {"agent_effectiveness": -5}
    facet_low = parse_facet_from_ai_generate_row("s2", row_low)
    assert facet_low.agent_effectiveness == 1.0

  def test_array_columns_filtered(self):
    row = {
        "goal_categories": [
            "analysis",
            "invalid_cat",
            "search",
        ],
        "friction_types": [
            "tool_error",
            "fake",
            "timeout",
        ],
    }
    facet = parse_facet_from_ai_generate_row("s1", row)
    assert "analysis" in facet.goal_categories
    assert "search" in facet.goal_categories
    assert "invalid_cat" not in facet.goal_categories
    assert "tool_error" in facet.friction_types
    assert "timeout" in facet.friction_types
    assert "fake" not in facet.friction_types

  def test_key_topics_limited(self):
    row = {
        "key_topics": [
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            "g",
        ],
    }
    facet = parse_facet_from_ai_generate_row("s1", row)
    assert len(facet.key_topics) <= 5

  def test_summary_truncated(self):
    row = {"summary": "x" * 300}
    facet = parse_facet_from_ai_generate_row("s1", row)
    assert len(facet.summary) <= 200

  def test_empty_row(self):
    facet = parse_facet_from_ai_generate_row("s1", {})
    assert facet.session_id == "s1"
    assert facet.outcome == "unclear"
    assert facet.goal_categories == ["other"]


class TestAIGenerateTemplates:
  """Tests for AI.GENERATE SQL template strings."""

  def test_facet_query_contains_ai_generate(self):
    assert "AI.GENERATE" in _AI_GENERATE_FACET_EXTRACTION_QUERY

  def test_facet_query_contains_output_schema(self):
    assert "output_schema" in _AI_GENERATE_FACET_EXTRACTION_QUERY

  def test_facet_query_contains_endpoint(self):
    assert "{endpoint}" in _AI_GENERATE_FACET_EXTRACTION_QUERY

  def test_analysis_query_contains_ai_generate(self):
    assert "AI.GENERATE" in _AI_GENERATE_ANALYSIS_QUERY

  def test_analysis_query_contains_endpoint(self):
    assert "{endpoint}" in _AI_GENERATE_ANALYSIS_QUERY
