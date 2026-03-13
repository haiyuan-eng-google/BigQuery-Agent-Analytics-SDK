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

"""Tests for the SDK feedback module."""

import pytest

from bigquery_agent_analytics.feedback import _AI_GENERATE_SEMANTIC_GROUPING_QUERY
from bigquery_agent_analytics.feedback import AnalysisConfig
from bigquery_agent_analytics.feedback import DriftReport
from bigquery_agent_analytics.feedback import QuestionCategory
from bigquery_agent_analytics.feedback import QuestionDistribution


class TestDriftReport:
  """Tests for DriftReport class."""

  def test_basic_report(self):
    report = DriftReport(
        coverage_percentage=75.0,
        total_golden=20,
        total_production=100,
        covered_questions=["q1", "q2", "q3"],
        uncovered_questions=["q4"],
        new_questions=["new1", "new2"],
    )

    assert report.coverage_percentage == 75.0
    assert report.total_golden == 20
    assert len(report.covered_questions) == 3
    assert len(report.uncovered_questions) == 1
    assert len(report.new_questions) == 2

  def test_summary(self):
    report = DriftReport(
        coverage_percentage=80.0,
        total_golden=10,
        total_production=50,
        covered_questions=["a"] * 8,
        uncovered_questions=["b"] * 2,
        new_questions=["c"] * 5,
    )
    text = report.summary()
    assert "80.0%" in text
    assert "10" in text
    assert "50" in text

  def test_zero_coverage(self):
    report = DriftReport(
        coverage_percentage=0.0,
        total_golden=5,
        total_production=0,
    )
    assert report.coverage_percentage == 0.0


class TestQuestionDistribution:
  """Tests for QuestionDistribution class."""

  def test_basic_distribution(self):
    dist = QuestionDistribution(
        total_questions=100,
        categories=[
            QuestionCategory(
                name="Technical",
                count=60,
                percentage=60.0,
                examples=["How to deploy?"],
            ),
            QuestionCategory(
                name="Billing",
                count=40,
                percentage=40.0,
                examples=["What is the cost?"],
            ),
        ],
    )
    assert dist.total_questions == 100
    assert len(dist.categories) == 2

  def test_summary(self):
    dist = QuestionDistribution(
        total_questions=50,
        categories=[
            QuestionCategory(
                name="FAQ",
                count=30,
                percentage=60.0,
            ),
            QuestionCategory(
                name="Bug",
                count=20,
                percentage=40.0,
            ),
        ],
    )
    text = dist.summary()
    assert "50" in text
    assert "FAQ" in text
    assert "Bug" in text

  def test_empty_distribution(self):
    dist = QuestionDistribution()
    assert dist.total_questions == 0
    assert dist.categories == []


class TestAnalysisConfig:
  """Tests for AnalysisConfig class."""

  def test_defaults(self):
    config = AnalysisConfig()
    assert config.mode == "auto_group_using_semantics"
    assert config.custom_categories is None
    assert config.top_k == 20

  def test_custom_mode(self):
    config = AnalysisConfig(
        mode="custom",
        custom_categories=[
            "Onboarding",
            "PTO",
            "Salary",
        ],
        top_k=10,
    )
    assert config.mode == "custom"
    assert len(config.custom_categories) == 3
    assert config.top_k == 10

  def test_frequently_asked_mode(self):
    config = AnalysisConfig(mode="frequently_asked")
    assert config.mode == "frequently_asked"

  def test_frequently_unanswered_mode(self):
    config = AnalysisConfig(mode="frequently_unanswered")
    assert config.mode == "frequently_unanswered"


class TestAIGenerateSemanticGroupingQuery:
  """Tests for the AI.GENERATE semantic grouping template."""

  def test_contains_ai_generate(self):
    assert "AI.GENERATE" in _AI_GENERATE_SEMANTIC_GROUPING_QUERY

  def test_contains_output_schema(self):
    assert "output_schema" in _AI_GENERATE_SEMANTIC_GROUPING_QUERY

  def test_contains_endpoint_placeholder(self):
    assert "{endpoint}" in _AI_GENERATE_SEMANTIC_GROUPING_QUERY

  def test_contains_category_string(self):
    assert "category STRING" in _AI_GENERATE_SEMANTIC_GROUPING_QUERY

  def test_does_not_contain_ml_generate_text(self):
    assert "ML.GENERATE_TEXT" not in _AI_GENERATE_SEMANTIC_GROUPING_QUERY
