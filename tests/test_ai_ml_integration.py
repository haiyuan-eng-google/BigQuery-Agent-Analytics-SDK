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

import pytest

from bigquery_agent_analytics.ai_ml_integration import Anomaly
from bigquery_agent_analytics.ai_ml_integration import AnomalyDetector
from bigquery_agent_analytics.ai_ml_integration import AnomalyType
from bigquery_agent_analytics.ai_ml_integration import BatchEvaluationResult
from bigquery_agent_analytics.ai_ml_integration import BatchEvaluator
from bigquery_agent_analytics.ai_ml_integration import BigQueryAIClient
from bigquery_agent_analytics.ai_ml_integration import EmbeddingResult
from bigquery_agent_analytics.ai_ml_integration import EmbeddingSearchClient
from bigquery_agent_analytics.ai_ml_integration import LatencyForecast


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


# ================================================================== #
# AI Operator Migration Tests                                          #
# ================================================================== #


class TestAIEmbedMigration:
  """Tests for AI.EMBED migration from ML.GENERATE_EMBEDDING."""

  def test_ai_embed_query_template_exists(self):
    """AI.EMBED query template is defined on BigQueryAIClient."""
    assert hasattr(BigQueryAIClient, "_AI_EMBED_QUERY")
    assert "AI.EMBED" in BigQueryAIClient._AI_EMBED_QUERY
    assert "endpoint" in BigQueryAIClient._AI_EMBED_QUERY

  def test_legacy_embedding_query_preserved(self):
    """Legacy ML.GENERATE_EMBEDDING template is preserved."""
    assert hasattr(BigQueryAIClient, "_LEGACY_GENERATE_EMBEDDING_QUERY")
    assert (
        "ML.GENERATE_EMBEDDING"
        in BigQueryAIClient._LEGACY_GENERATE_EMBEDDING_QUERY
    )

  def test_default_embedding_endpoint(self):
    """Default embedding endpoint is text-embedding-005."""
    client = BigQueryAIClient(
        project_id="p",
        dataset_id="d",
        client=MagicMock(),
    )
    assert client.embedding_endpoint == "text-embedding-005"
    assert client.embedding_model is None

  def test_custom_embedding_endpoint(self):
    """Custom embedding endpoint is respected."""
    client = BigQueryAIClient(
        project_id="p",
        dataset_id="d",
        client=MagicMock(),
        embedding_endpoint="text-multilingual-embedding-002",
    )
    assert client.embedding_endpoint == "text-multilingual-embedding-002"

  def test_legacy_model_routes_to_ml_generate_embedding(self):
    """When embedding_model is a BQ ML ref, uses legacy path."""
    client = BigQueryAIClient(
        project_id="p",
        dataset_id="d",
        client=MagicMock(),
        embedding_model="p.d.text_embedding_model",
    )
    assert client.embedding_model == "p.d.text_embedding_model"

  @pytest.mark.asyncio
  async def test_generate_embeddings_uses_ai_embed_by_default(self):
    """Default path uses AI.EMBED."""
    mock_bq = MagicMock()
    mock_job = MagicMock()
    mock_job.result.return_value = [
        {"content": "hello", "embedding": [0.1, 0.2]},
    ]
    mock_bq.query.return_value = mock_job

    client = BigQueryAIClient(
        project_id="p",
        dataset_id="d",
        client=mock_bq,
    )
    results = await client.generate_embeddings(["hello"])

    assert len(results) == 1
    query_str = mock_bq.query.call_args[0][0]
    assert "AI.EMBED" in query_str
    assert "ML.GENERATE_EMBEDDING" not in query_str

  @pytest.mark.asyncio
  async def test_generate_embeddings_uses_legacy_when_model_set(self):
    """Legacy path is used when embedding_model is a BQ ML ref."""
    mock_bq = MagicMock()
    mock_job = MagicMock()
    mock_job.result.return_value = [
        {"content": "hello", "embedding": [0.1, 0.2]},
    ]
    mock_bq.query.return_value = mock_job

    client = BigQueryAIClient(
        project_id="p",
        dataset_id="d",
        client=mock_bq,
        embedding_model="p.d.text_embedding_model",
    )
    results = await client.generate_embeddings(["hello"])

    assert len(results) == 1
    query_str = mock_bq.query.call_args[0][0]
    assert "ML.GENERATE_EMBEDDING" in query_str
    assert "AI.EMBED" not in query_str


class TestEmbeddingSearchAIEmbedMigration:
  """Tests for EmbeddingSearchClient AI.EMBED migration."""

  def test_ai_embed_index_query_exists(self):
    """AI.EMBED index query template is defined."""
    assert hasattr(EmbeddingSearchClient, "_AI_EMBED_INDEX_QUERY")
    assert "AI.EMBED" in EmbeddingSearchClient._AI_EMBED_INDEX_QUERY

  def test_legacy_index_query_preserved(self):
    """Legacy ML.GENERATE_EMBEDDING index template is preserved."""
    assert hasattr(EmbeddingSearchClient, "_LEGACY_INDEX_EMBEDDINGS_QUERY")
    assert "ML.GENERATE_EMBEDDING" in (
        EmbeddingSearchClient._LEGACY_INDEX_EMBEDDINGS_QUERY
    )

  def test_default_uses_ai_embed(self):
    """Default (no embedding_model) uses AI.EMBED endpoint."""
    client = EmbeddingSearchClient(
        project_id="p",
        dataset_id="d",
        client=MagicMock(),
    )
    assert client.embedding_model is None
    assert client.embedding_endpoint == "text-embedding-005"

  def test_legacy_model_set(self):
    """When embedding_model is set, legacy path is used."""
    client = EmbeddingSearchClient(
        project_id="p",
        dataset_id="d",
        client=MagicMock(),
        embedding_model="p.d.text_embedding_model",
    )
    assert client.embedding_model == "p.d.text_embedding_model"

  @pytest.mark.asyncio
  async def test_plain_endpoint_as_model_uses_ai_embed(self):
    """A plain endpoint string (no dots) must NOT trigger the legacy path.

    Regression test: build_embeddings_index() previously used a bare
    ``if self.embedding_model:`` check, which incorrectly routed
    plain endpoint names like ``text-embedding-005`` to the legacy
    ``ML.GENERATE_EMBEDDING`` template.  The fix applies the same
    ``count('.') >= 2`` rule used by ``generate_embeddings()``.
    """
    mock_bq = MagicMock()
    mock_job = MagicMock()
    mock_bq.query.return_value = mock_job
    mock_job.result.return_value = []

    client = EmbeddingSearchClient(
        project_id="p",
        dataset_id="d",
        client=mock_bq,
        embedding_model="text-embedding-005",
    )
    await client.build_embeddings_index(since_days=7)

    executed_sql = mock_bq.query.call_args[0][0]
    assert "AI.EMBED" in executed_sql
    assert "ML.GENERATE_EMBEDDING" not in executed_sql


class TestAIDetectAnomaliesMigration:
  """Tests for AI.DETECT_ANOMALIES migration from ML.DETECT_ANOMALIES."""

  def test_ai_detect_anomalies_query_exists(self):
    """AI.DETECT_ANOMALIES query template is defined."""
    assert hasattr(AnomalyDetector, "_AI_DETECT_LATENCY_ANOMALIES_QUERY")
    assert "AI.DETECT_ANOMALIES" in (
        AnomalyDetector._AI_DETECT_LATENCY_ANOMALIES_QUERY
    )
    assert "anomaly_prob_threshold" in (
        AnomalyDetector._AI_DETECT_LATENCY_ANOMALIES_QUERY
    )
    assert "timestamp_col" in (
        AnomalyDetector._AI_DETECT_LATENCY_ANOMALIES_QUERY
    )
    assert "data_col" in (AnomalyDetector._AI_DETECT_LATENCY_ANOMALIES_QUERY)

  def test_legacy_detect_anomalies_preserved(self):
    """Legacy ML.DETECT_ANOMALIES template is preserved."""
    assert hasattr(AnomalyDetector, "_LEGACY_DETECT_LATENCY_ANOMALIES_QUERY")
    assert "ML.DETECT_ANOMALIES" in (
        AnomalyDetector._LEGACY_DETECT_LATENCY_ANOMALIES_QUERY
    )

  def test_default_uses_ai_detect_anomalies(self):
    """Default (use_legacy_anomaly_model=False) uses AI path."""
    detector = AnomalyDetector(
        project_id="p",
        dataset_id="d",
        client=MagicMock(),
    )
    assert detector.use_legacy_anomaly_model is False

  def test_legacy_flag(self):
    """use_legacy_anomaly_model=True selects ML path."""
    detector = AnomalyDetector(
        project_id="p",
        dataset_id="d",
        client=MagicMock(),
        use_legacy_anomaly_model=True,
    )
    assert detector.use_legacy_anomaly_model is True

  @pytest.mark.asyncio
  async def test_train_skipped_for_ai_path(self):
    """train_latency_model returns True immediately for AI path."""
    detector = AnomalyDetector(
        project_id="p",
        dataset_id="d",
        client=MagicMock(),
    )
    result = await detector.train_latency_model()
    assert result is True
    # Should NOT have called BigQuery
    detector.client.query.assert_not_called()

  @pytest.mark.asyncio
  async def test_detect_uses_ai_by_default(self):
    """detect_latency_anomalies uses AI.DETECT_ANOMALIES by default."""
    mock_bq = MagicMock()
    mock_job = MagicMock()
    mock_job.result.return_value = [
        {
            "time_series_timestamp": datetime(
                2026, 3, 31, 12, tzinfo=timezone.utc
            ),
            "time_series_data": 5000.0,
            "is_anomaly": True,
            "anomaly_probability": 0.98,
            "lower_bound": 1000.0,
            "upper_bound": 3000.0,
        },
    ]
    mock_bq.query.return_value = mock_job

    detector = AnomalyDetector(
        project_id="p",
        dataset_id="d",
        client=mock_bq,
    )
    anomalies = await detector.detect_latency_anomalies(since_hours=24)

    assert len(anomalies) == 1
    assert anomalies[0].anomaly_type == AnomalyType.LATENCY_SPIKE
    assert anomalies[0].severity == 0.98

    query_str = mock_bq.query.call_args[0][0]
    assert "AI.DETECT_ANOMALIES" in query_str
    assert "ML.DETECT_ANOMALIES" not in query_str

  @pytest.mark.asyncio
  async def test_detect_uses_legacy_when_flag_set(self):
    """detect_latency_anomalies uses ML.DETECT_ANOMALIES with flag."""
    mock_bq = MagicMock()
    mock_job = MagicMock()
    mock_job.result.return_value = [
        {
            "hour": datetime(2026, 3, 31, 12, tzinfo=timezone.utc),
            "avg_latency": 5000.0,
            "is_anomaly": True,
            "anomaly_probability": 0.97,
            "lower_bound": 1000.0,
            "upper_bound": 3000.0,
        },
    ]
    mock_bq.query.return_value = mock_job

    detector = AnomalyDetector(
        project_id="p",
        dataset_id="d",
        client=mock_bq,
        use_legacy_anomaly_model=True,
    )
    anomalies = await detector.detect_latency_anomalies(since_hours=24)

    assert len(anomalies) == 1
    query_str = mock_bq.query.call_args[0][0]
    assert "ML.DETECT_ANOMALIES" in query_str

  def test_behavior_anomaly_still_uses_ml(self):
    """Behavioral anomaly detection still uses ML.DETECT_ANOMALIES."""
    assert "ML.DETECT_ANOMALIES" in (
        AnomalyDetector._DETECT_BEHAVIOR_ANOMALIES_QUERY
    )
    assert "AUTOENCODER" in AnomalyDetector._CREATE_BEHAVIOR_MODEL_QUERY


class TestAIForecastMigration:
  """Tests for AI.FORECAST integration."""

  def test_ai_forecast_query_exists(self):
    """AI.FORECAST query template is defined."""
    assert hasattr(AnomalyDetector, "_AI_FORECAST_LATENCY_QUERY")
    assert "AI.FORECAST" in AnomalyDetector._AI_FORECAST_LATENCY_QUERY

  def test_ai_forecast_query_uses_named_params(self):
    """AI.FORECAST template uses named parameter syntax."""
    q = AnomalyDetector._AI_FORECAST_LATENCY_QUERY
    assert "horizon =>" in q
    assert "timestamp_col =>" in q
    assert "data_col =>" in q

  def test_legacy_forecast_query_preserved(self):
    """Legacy ML.FORECAST template is preserved."""
    assert hasattr(AnomalyDetector, "_LEGACY_FORECAST_LATENCY_QUERY")
    q = AnomalyDetector._LEGACY_FORECAST_LATENCY_QUERY
    assert "ML.FORECAST" in q
    assert "MODEL" in q

  @pytest.mark.asyncio
  async def test_default_uses_ai_forecast(self):
    """forecast_latency() uses AI.FORECAST by default."""
    mock_bq = MagicMock()
    mock_job = MagicMock()
    mock_job.result.return_value = [
        {
            "time_series_timestamp": datetime(
                2026, 4, 1, 12, tzinfo=timezone.utc
            ),
            "time_series_data": 250.0,
            "prediction_interval_lower_bound": 200.0,
            "prediction_interval_upper_bound": 300.0,
            "ai_forecast_status": "",
        },
    ]
    mock_bq.query.return_value = mock_job

    detector = AnomalyDetector(
        project_id="p",
        dataset_id="d",
        client=mock_bq,
    )
    forecasts = await detector.forecast_latency(horizon_hours=24)

    assert len(forecasts) == 1
    query_str = mock_bq.query.call_args[0][0]
    assert "AI.FORECAST" in query_str
    assert "ML.FORECAST" not in query_str

  @pytest.mark.asyncio
  async def test_legacy_flag_uses_ml_forecast(self):
    """forecast_latency() uses ML.FORECAST with legacy flag."""
    mock_bq = MagicMock()
    mock_job = MagicMock()
    mock_job.result.return_value = [
        {
            "forecast_timestamp": datetime(2026, 4, 1, 12, tzinfo=timezone.utc),
            "forecast_value": 250.0,
            "prediction_interval_lower_bound": 200.0,
            "prediction_interval_upper_bound": 300.0,
        },
    ]
    mock_bq.query.return_value = mock_job

    detector = AnomalyDetector(
        project_id="p",
        dataset_id="d",
        client=mock_bq,
        use_legacy_anomaly_model=True,
    )
    forecasts = await detector.forecast_latency(horizon_hours=24)

    assert len(forecasts) == 1
    query_str = mock_bq.query.call_args[0][0]
    assert "ML.FORECAST" in query_str

  @pytest.mark.asyncio
  async def test_forecast_result_parsing(self):
    """Mock rows are parsed into LatencyForecast dataclasses."""
    mock_bq = MagicMock()
    mock_job = MagicMock()
    ts = datetime(2026, 4, 1, 13, tzinfo=timezone.utc)
    mock_job.result.return_value = [
        {
            "time_series_timestamp": ts,
            "time_series_data": 320.5,
            "prediction_interval_lower_bound": 280.0,
            "prediction_interval_upper_bound": 360.0,
            "ai_forecast_status": "",
        },
    ]
    mock_bq.query.return_value = mock_job

    detector = AnomalyDetector(
        project_id="p",
        dataset_id="d",
        client=mock_bq,
    )
    forecasts = await detector.forecast_latency()

    assert len(forecasts) == 1
    f = forecasts[0]
    assert isinstance(f, LatencyForecast)
    assert f.timestamp == ts
    assert f.forecast_value == 320.5
    assert f.lower_bound == 280.0
    assert f.upper_bound == 360.0

  @pytest.mark.asyncio
  async def test_forecast_skips_failed_status(self):
    """Rows with non-empty ai_forecast_status are excluded."""
    mock_bq = MagicMock()
    mock_job = MagicMock()
    mock_job.result.return_value = [
        {
            "time_series_timestamp": datetime(
                2026, 4, 1, 12, tzinfo=timezone.utc
            ),
            "time_series_data": 250.0,
            "prediction_interval_lower_bound": 200.0,
            "prediction_interval_upper_bound": 300.0,
            "ai_forecast_status": "",
        },
        {
            "time_series_timestamp": datetime(
                2026, 4, 1, 13, tzinfo=timezone.utc
            ),
            "time_series_data": 0.0,
            "prediction_interval_lower_bound": 0.0,
            "prediction_interval_upper_bound": 0.0,
            "ai_forecast_status": "Insufficient history for forecasting",
        },
    ]
    mock_bq.query.return_value = mock_job

    detector = AnomalyDetector(
        project_id="p",
        dataset_id="d",
        client=mock_bq,
    )
    forecasts = await detector.forecast_latency()

    assert len(forecasts) == 1
    assert forecasts[0].forecast_value == 250.0


class TestModuleDocstring:
  """Guard tests for module-level docstring accuracy."""

  def test_docstring_mentions_ai_forecast(self):
    """Docstring should mention AI.FORECAST."""
    import bigquery_agent_analytics.ai_ml_integration as mod

    assert "AI.FORECAST" in mod.__doc__
