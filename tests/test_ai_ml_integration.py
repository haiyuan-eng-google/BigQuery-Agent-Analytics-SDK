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

"""Tests for BigQuery AI/ML integration."""

from datetime import datetime
from datetime import timezone
from unittest.mock import MagicMock

from bigquery_agent_analytics.ai_ml_integration import Anomaly
from bigquery_agent_analytics.ai_ml_integration import AnomalyDetector
from bigquery_agent_analytics.ai_ml_integration import AnomalyType
from bigquery_agent_analytics.ai_ml_integration import BatchEvaluationResult
from bigquery_agent_analytics.ai_ml_integration import BatchEvaluator
from bigquery_agent_analytics.ai_ml_integration import BigQueryAIClient
from bigquery_agent_analytics.ai_ml_integration import EmbeddingResult
from bigquery_agent_analytics.ai_ml_integration import EmbeddingSearchClient
import pytest


class TestEmbeddingResult:
  """Tests for EmbeddingResult class."""

  def test_embedding_result_creation(self):
    """Test creating an EmbeddingResult."""
    result = EmbeddingResult(
        text="Hello world",
        embedding=[0.1, 0.2, 0.3, 0.4],
        metadata={"source": "test"},
    )

    assert result.text == "Hello world"
    assert len(result.embedding) == 4
    assert result.metadata["source"] == "test"


class TestAnomaly:
  """Tests for Anomaly class."""

  def test_anomaly_creation(self):
    """Test creating an Anomaly."""
    now = datetime.now(timezone.utc)
    anomaly = Anomaly(
        anomaly_type=AnomalyType.LATENCY_SPIKE,
        timestamp=now,
        severity=0.8,
        description="High latency detected",
        affected_sessions=["sess-1", "sess-2"],
        details={"avg_latency_ms": 5000},
    )

    assert anomaly.anomaly_type == AnomalyType.LATENCY_SPIKE
    assert anomaly.severity == 0.8
    assert len(anomaly.affected_sessions) == 2
    assert anomaly.details["avg_latency_ms"] == 5000


class TestBatchEvaluationResult:
  """Tests for BatchEvaluationResult class."""

  def test_batch_evaluation_result_creation(self):
    """Test creating a BatchEvaluationResult."""
    result = BatchEvaluationResult(
        session_id="sess-123",
        task_completion=0.9,
        efficiency=0.85,
        tool_usage=0.95,
        evaluation_text='{"task_completion": 9}',
        error=None,
    )

    assert result.session_id == "sess-123"
    assert result.task_completion == 0.9
    assert result.efficiency == 0.85
    assert result.tool_usage == 0.95
    assert result.error is None

  def test_batch_evaluation_result_with_error(self):
    """Test BatchEvaluationResult with error."""
    result = BatchEvaluationResult(
        session_id="sess-123",
        task_completion=0.0,
        efficiency=0.0,
        tool_usage=0.0,
        evaluation_text=None,
        error="Evaluation failed",
    )

    assert result.error == "Evaluation failed"


class TestBigQueryAIClient:
  """Tests for BigQueryAIClient class."""

  @pytest.fixture
  def mock_client(self):
    """Create mock BigQuery client."""
    return MagicMock()

  @pytest.fixture
  def ai_client(self, mock_client):
    """Create AI client with mock."""
    return BigQueryAIClient(
        project_id="test-project",
        dataset_id="test-dataset",
        client=mock_client,
    )

  def test_default_endpoint(self, mock_client):
    """Test default endpoint value."""
    client = BigQueryAIClient(
        project_id="p",
        dataset_id="d",
        client=mock_client,
    )
    assert client.endpoint == "gemini-2.5-flash"

  def test_custom_endpoint(self, mock_client):
    """Test custom endpoint."""
    client = BigQueryAIClient(
        project_id="p",
        dataset_id="d",
        client=mock_client,
        endpoint="gemini-2.5-pro",
    )
    assert client.endpoint == "gemini-2.5-pro"

  def test_text_model_as_endpoint_alias(self, mock_client):
    """Test text_model backward compatibility."""
    client = BigQueryAIClient(
        project_id="p",
        dataset_id="d",
        client=mock_client,
        text_model="p.d.my_model",
    )
    assert client.endpoint == "p.d.my_model"
    assert client.text_model == "p.d.my_model"

  def test_endpoint_overrides_text_model(self, mock_client):
    """Test explicit endpoint takes priority."""
    client = BigQueryAIClient(
        project_id="p",
        dataset_id="d",
        client=mock_client,
        text_model="old_model",
        endpoint="gemini-2.5-pro",
    )
    assert client.endpoint == "gemini-2.5-pro"

  def test_connection_id(self, mock_client):
    """Test connection_id parameter."""
    client = BigQueryAIClient(
        project_id="p",
        dataset_id="d",
        client=mock_client,
        connection_id="us.conn",
    )
    assert client.connection_id == "us.conn"

  @pytest.mark.asyncio
  async def test_generate_text(self, ai_client, mock_client):
    """Test text generation uses AI.GENERATE."""
    mock_results = [{"result": "Generated text response"}]

    mock_query_job = MagicMock()
    mock_query_job.result.return_value = mock_results
    mock_client.query.return_value = mock_query_job

    result = await ai_client.generate_text("Test prompt")

    assert result == "Generated text response"
    # Verify AI.GENERATE SQL was used
    call_args = mock_client.query.call_args
    query_str = call_args[0][0]
    assert "AI.GENERATE" in query_str
    assert "endpoint" in query_str

  @pytest.mark.asyncio
  async def test_generate_text_empty_result(self, ai_client, mock_client):
    """Test text generation with empty result."""
    mock_query_job = MagicMock()
    mock_query_job.result.return_value = []
    mock_client.query.return_value = mock_query_job

    result = await ai_client.generate_text("Test prompt")

    assert result == ""

  @pytest.mark.asyncio
  async def test_generate_embeddings(self, ai_client, mock_client):
    """Test embedding generation."""
    mock_results = [
        {"content": "Text 1", "embedding": [0.1, 0.2, 0.3]},
        {"content": "Text 2", "embedding": [0.4, 0.5, 0.6]},
    ]

    mock_query_job = MagicMock()
    mock_query_job.result.return_value = mock_results
    mock_client.query.return_value = mock_query_job

    results = await ai_client.generate_embeddings(["Text 1", "Text 2"])

    assert len(results) == 2
    assert results[0].text == "Text 1"
    assert results[0].embedding == [0.1, 0.2, 0.3]

  @pytest.mark.asyncio
  async def test_generate_embeddings_empty(self, ai_client):
    """Test embedding generation with empty input."""
    results = await ai_client.generate_embeddings([])

    assert results == []

  @pytest.mark.asyncio
  async def test_analyze_trace(self, ai_client, mock_client):
    """Test trace analysis."""
    mock_results = [{"result": '{"score": 8, "feedback": "Good"}'}]

    mock_query_job = MagicMock()
    mock_query_job.result.return_value = mock_results
    mock_client.query.return_value = mock_query_job

    result = await ai_client.analyze_trace(
        trace_text="TOOL_STARTING: search\nTOOL_COMPLETED: search",
        analysis_prompt="Analyze this trace",
    )

    assert result.get("score") == 8
    assert result.get("feedback") == "Good"


class TestEmbeddingSearchClient:
  """Tests for EmbeddingSearchClient class."""

  @pytest.fixture
  def mock_client(self):
    """Create mock BigQuery client."""
    return MagicMock()

  @pytest.fixture
  def search_client(self, mock_client):
    """Create search client with mock."""
    return EmbeddingSearchClient(
        project_id="test-project",
        dataset_id="test-dataset",
        client=mock_client,
    )

  @pytest.mark.asyncio
  async def test_search(self, search_client, mock_client):
    """Test vector similarity search."""
    mock_results = [
        {
            "session_id": "sess-1",
            "content": "Weather forecast",
            "timestamp": datetime.now(timezone.utc),
            "distance": 0.1,
        },
        {
            "session_id": "sess-2",
            "content": "News headlines",
            "timestamp": datetime.now(timezone.utc),
            "distance": 0.3,
        },
    ]

    mock_query_job = MagicMock()
    mock_query_job.result.return_value = mock_results
    mock_client.query.return_value = mock_query_job

    results = await search_client.search(
        query_embedding=[0.1, 0.2, 0.3],
        top_k=10,
    )

    assert len(results) == 2
    assert results[0]["session_id"] == "sess-1"
    assert results[0]["similarity"] == 0.9  # 1.0 - 0.1

  @pytest.mark.asyncio
  async def test_search_with_filters(self, search_client, mock_client):
    """Test search with user and time filters."""
    mock_results = [
        {
            "session_id": "sess-1",
            "content": "Test content",
            "timestamp": datetime.now(timezone.utc),
            "distance": 0.2,
        },
    ]

    mock_query_job = MagicMock()
    mock_query_job.result.return_value = mock_results
    mock_client.query.return_value = mock_query_job

    results = await search_client.search(
        query_embedding=[0.1, 0.2],
        top_k=5,
        user_id="user-123",
        since_days=7,
    )

    assert len(results) == 1
    # Verify query was called with filters
    mock_client.query.assert_called_once()

  @pytest.mark.asyncio
  async def test_build_embeddings_index(self, search_client, mock_client):
    """Test building embeddings index."""
    mock_query_job = MagicMock()
    mock_query_job.result.return_value = None
    mock_client.query.return_value = mock_query_job

    success = await search_client.build_embeddings_index(since_days=30)

    assert success is True
    mock_client.query.assert_called_once()


class TestAnomalyDetector:
  """Tests for AnomalyDetector class."""

  @pytest.fixture
  def mock_client(self):
    """Create mock BigQuery client."""
    return MagicMock()

  @pytest.fixture
  def detector(self, mock_client):
    """Create anomaly detector with mock."""
    return AnomalyDetector(
        project_id="test-project",
        dataset_id="test-dataset",
        client=mock_client,
    )

  @pytest.mark.asyncio
  async def test_train_latency_model(self, detector, mock_client):
    """Test training latency anomaly model."""
    mock_query_job = MagicMock()
    mock_query_job.result.return_value = None
    mock_client.query.return_value = mock_query_job

    success = await detector.train_latency_model(training_days=30)

    assert success is True

  @pytest.mark.asyncio
  async def test_detect_latency_anomalies(self, detector, mock_client):
    """Test detecting latency anomalies."""
    now = datetime.now(timezone.utc)
    mock_results = [
        {
            "hour": now,
            "avg_latency": 5000,
            "anomaly_probability": 0.98,
            "lower_bound": 100,
            "upper_bound": 500,
        },
    ]

    mock_query_job = MagicMock()
    mock_query_job.result.return_value = mock_results
    mock_client.query.return_value = mock_query_job

    anomalies = await detector.detect_latency_anomalies(since_hours=24)

    assert len(anomalies) == 1
    assert anomalies[0].anomaly_type == AnomalyType.LATENCY_SPIKE
    assert anomalies[0].severity == 0.98
    assert "5000ms" in anomalies[0].description

  @pytest.mark.asyncio
  async def test_detect_latency_anomalies_empty(self, detector, mock_client):
    """Test detecting anomalies with no results."""
    mock_query_job = MagicMock()
    mock_query_job.result.return_value = []
    mock_client.query.return_value = mock_query_job

    anomalies = await detector.detect_latency_anomalies(since_hours=24)

    assert anomalies == []

  @pytest.mark.asyncio
  async def test_train_behavior_model(self, detector, mock_client):
    """Test training behavior anomaly model."""
    mock_query_job = MagicMock()
    mock_query_job.result.return_value = None
    mock_client.query.return_value = mock_query_job

    success = await detector.train_behavior_model()

    assert success is True
    # Should have called query twice (features table + model)
    assert mock_client.query.call_count == 2

  @pytest.mark.asyncio
  async def test_detect_behavior_anomalies(self, detector, mock_client):
    """Test detecting behavioral anomalies."""
    mock_results = [
        {
            "session_id": "sess-123",
            "total_events": 50,
            "tool_calls": 20,
            "tool_errors": 10,
            "llm_calls": 15,
            "avg_latency": 1000,
            "session_duration": 300,
        },
    ]

    mock_query_job = MagicMock()
    mock_query_job.result.return_value = mock_results
    mock_client.query.return_value = mock_query_job

    anomalies = await detector.detect_behavior_anomalies(since_hours=24)

    assert len(anomalies) == 1
    # High error rate should be detected as tool failure pattern
    assert anomalies[0].anomaly_type == AnomalyType.TOOL_FAILURE_PATTERN
    assert "sess-123" in anomalies[0].affected_sessions


class TestBatchEvaluator:
  """Tests for BatchEvaluator class."""

  @pytest.fixture
  def mock_client(self):
    """Create mock BigQuery client."""
    return MagicMock()

  @pytest.fixture
  def evaluator(self, mock_client):
    """Create batch evaluator with mock."""
    return BatchEvaluator(
        project_id="test-project",
        dataset_id="test-dataset",
        client=mock_client,
    )

  def test_default_endpoint(self, mock_client):
    """Test default endpoint value."""
    ev = BatchEvaluator(
        project_id="p",
        dataset_id="d",
        client=mock_client,
    )
    assert ev.endpoint == "gemini-2.5-flash"

  def test_custom_endpoint(self, mock_client):
    """Test custom endpoint."""
    ev = BatchEvaluator(
        project_id="p",
        dataset_id="d",
        client=mock_client,
        endpoint="gemini-2.5-pro",
    )
    assert ev.endpoint == "gemini-2.5-pro"

  def test_eval_model_as_endpoint_alias(self, mock_client):
    """Test eval_model backward compatibility."""
    ev = BatchEvaluator(
        project_id="p",
        dataset_id="d",
        client=mock_client,
        eval_model="p.d.eval_model",
    )
    assert ev.endpoint == "p.d.eval_model"
    assert ev.eval_model == "p.d.eval_model"

  @pytest.mark.asyncio
  async def test_evaluate_recent_sessions(self, evaluator, mock_client):
    """Test batch evaluation with typed AI.GENERATE output."""
    mock_results = [
        {
            "session_id": "sess-1",
            "trace_text": "USER: Hello\nAGENT: Hi there",
            "task_completion": 9,
            "efficiency": 8,
            "tool_usage": 7,
        },
        {
            "session_id": "sess-2",
            "trace_text": "USER: Help\nAGENT: Sure",
            "task_completion": 7,
            "efficiency": 6,
            "tool_usage": 8,
        },
    ]

    mock_query_job = MagicMock()
    mock_query_job.result.return_value = mock_results
    mock_client.query.return_value = mock_query_job

    results = await evaluator.evaluate_recent_sessions(days=1, limit=100)

    assert len(results) == 2
    assert results[0].session_id == "sess-1"
    assert results[0].task_completion == 0.9  # 9/10
    assert results[0].efficiency == 0.8  # 8/10
    assert results[0].tool_usage == 0.7  # 7/10
    assert results[0].error is None

  @pytest.mark.asyncio
  async def test_evaluate_recent_sessions_parse_error(
      self, evaluator, mock_client
  ):
    """Test handling of parse errors in typed output."""
    mock_results = [
        {
            "session_id": "sess-1",
            "trace_text": "USER: Hello",
            "task_completion": "invalid",
            "efficiency": None,
            "tool_usage": None,
        },
    ]

    mock_query_job = MagicMock()
    mock_query_job.result.return_value = mock_results
    mock_client.query.return_value = mock_query_job

    results = await evaluator.evaluate_recent_sessions(days=1, limit=100)

    assert len(results) == 1
    assert results[0].error is not None
    assert "Failed to parse" in results[0].error

  @pytest.mark.asyncio
  async def test_store_evaluation_results(self, evaluator, mock_client):
    """Test storing evaluation results."""
    mock_client.insert_rows_json.return_value = []

    results = [
        BatchEvaluationResult(
            session_id="sess-1",
            task_completion=0.9,
            efficiency=0.8,
            tool_usage=0.7,
        ),
    ]

    success = await evaluator.store_evaluation_results(results)

    assert success is True
    mock_client.insert_rows_json.assert_called_once()

  @pytest.mark.asyncio
  async def test_store_evaluation_results_empty(self, evaluator, mock_client):
    """Test storing empty results."""
    success = await evaluator.store_evaluation_results([])

    assert success is True
    mock_client.insert_rows_json.assert_not_called()

  @pytest.mark.asyncio
  async def test_store_evaluation_results_error(self, evaluator, mock_client):
    """Test handling storage errors."""
    mock_client.insert_rows_json.return_value = [{"error": "Insert failed"}]

    results = [
        BatchEvaluationResult(
            session_id="sess-1",
            task_completion=0.9,
            efficiency=0.8,
            tool_usage=0.7,
        ),
    ]

    success = await evaluator.store_evaluation_results(results)

    assert success is False
