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

"""Tests for BigQuery memory service."""

from datetime import datetime
from datetime import timedelta
from datetime import timezone
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from bigquery_agent_analytics.memory_service import BigQueryEpisodicMemory
from bigquery_agent_analytics.memory_service import BigQueryMemoryService
from bigquery_agent_analytics.memory_service import BigQuerySessionMemory
from bigquery_agent_analytics.memory_service import ContextManager
from bigquery_agent_analytics.memory_service import Episode
from bigquery_agent_analytics.memory_service import UserProfile
from bigquery_agent_analytics.memory_service import UserProfileBuilder


class TestEpisode:
  """Tests for Episode class."""

  def test_to_memory_entry(self):
    """Test converting Episode to MemoryEntry."""
    episode = Episode(
        session_id="sess-123",
        timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        user_message="What is the weather?",
        agent_response="It is sunny today.",
        tool_calls=["search", "format"],
        similarity_score=0.85,
        metadata={"source": "test"},
    )

    memory_entry = episode.to_memory_entry()

    assert memory_entry.content is not None
    assert len(memory_entry.content.parts) == 2
    assert "User: What is the weather?" in memory_entry.content.parts[0].text
    assert "Agent: It is sunny today." in memory_entry.content.parts[1].text
    assert memory_entry.custom_metadata["session_id"] == "sess-123"
    assert memory_entry.custom_metadata["similarity_score"] == 0.85

  def test_to_memory_entry_no_response(self):
    """Test converting Episode without agent response."""
    episode = Episode(
        session_id="sess-123",
        timestamp=datetime.now(timezone.utc),
        user_message="Hello",
        agent_response=None,
    )

    memory_entry = episode.to_memory_entry()

    assert len(memory_entry.content.parts) == 1
    assert "User: Hello" in memory_entry.content.parts[0].text


class TestUserProfile:
  """Tests for UserProfile class."""

  def test_user_profile_defaults(self):
    """Test UserProfile default values."""
    profile = UserProfile(user_id="user-123")

    assert profile.user_id == "user-123"
    assert profile.topics_of_interest == []
    assert profile.communication_style is None
    assert profile.common_requests == []
    assert profile.preferred_tools == []
    assert profile.session_count == 0
    assert profile.last_interaction is None

  def test_user_profile_full(self):
    """Test UserProfile with all fields."""
    now = datetime.now(timezone.utc)
    profile = UserProfile(
        user_id="user-123",
        topics_of_interest=["weather", "news"],
        communication_style="casual",
        common_requests=["forecasts", "summaries"],
        preferred_tools=["search", "summarize"],
        session_count=50,
        last_interaction=now,
        custom_metadata={"vip": True},
    )

    assert profile.topics_of_interest == ["weather", "news"]
    assert profile.communication_style == "casual"
    assert profile.session_count == 50
    assert profile.custom_metadata == {"vip": True}


class TestBigQuerySessionMemory:
  """Tests for BigQuerySessionMemory class."""

  @pytest.fixture
  def mock_client(self):
    """Create mock BigQuery client."""
    return MagicMock()

  @pytest.fixture
  def session_memory(self, mock_client):
    """Create session memory with mock client."""
    return BigQuerySessionMemory(
        project_id="test-project",
        dataset_id="test-dataset",
        table_id="test-table",
        client=mock_client,
    )

  @pytest.mark.asyncio
  async def test_get_recent_context(self, session_memory, mock_client):
    """Test getting recent context from past sessions."""
    mock_results = [
        {
            "session_id": "sess-old-1",
            "event_type": "USER_MESSAGE_RECEIVED",
            "timestamp": datetime.now(timezone.utc) - timedelta(hours=2),
            "content_summary": "Previous question",
            "response": None,
            "agent": "agent",
        },
        {
            "session_id": "sess-old-1",
            "event_type": "AGENT_COMPLETED",
            "timestamp": datetime.now(timezone.utc) - timedelta(hours=2),
            "content_summary": None,
            "response": "Previous answer",
            "agent": "agent",
        },
    ]

    mock_query_job = MagicMock()
    mock_query_job.result.return_value = mock_results
    mock_client.query.return_value = mock_query_job

    episodes = await session_memory.get_recent_context(
        user_id="user-123",
        current_session_id="sess-current",
        lookback_sessions=5,
        max_events=50,
    )

    assert len(episodes) == 1
    assert episodes[0].session_id == "sess-old-1"
    assert episodes[0].user_message == "Previous question"
    assert episodes[0].agent_response == "Previous answer"


class TestBigQueryEpisodicMemory:
  """Tests for BigQueryEpisodicMemory class."""

  @pytest.fixture
  def mock_client(self):
    """Create mock BigQuery client."""
    return MagicMock()

  @pytest.fixture
  def episodic_memory(self, mock_client):
    """Create episodic memory with mock client."""
    return BigQueryEpisodicMemory(
        project_id="test-project",
        dataset_id="test-dataset",
        table_id="test-table",
        client=mock_client,
    )

  @pytest.mark.asyncio
  async def test_keyword_search(self, episodic_memory, mock_client):
    """Test keyword-based search fallback."""
    mock_results = [
        {
            "session_id": "sess-1",
            "content": '{"text_summary": "weather forecast sunny"}',
            "timestamp": datetime.now(timezone.utc),
            "user_id": "user-123",
            "event_type": "USER_MESSAGE_RECEIVED",
        },
        {
            "session_id": "sess-2",
            "content": '{"text_summary": "news headlines today"}',
            "timestamp": datetime.now(timezone.utc),
            "user_id": "user-123",
            "event_type": "USER_MESSAGE_RECEIVED",
        },
    ]

    mock_query_job = MagicMock()
    mock_query_job.result.return_value = mock_results
    mock_client.query.return_value = mock_query_job

    episodes = await episodic_memory.retrieve_similar_episodes(
        query="weather forecast",
        user_id="user-123",
        top_k=5,
    )

    # Should find the weather-related episode
    assert len(episodes) >= 1
    weather_episode = next(
        (e for e in episodes if "weather" in e.user_message.lower()), None
    )
    assert weather_episode is not None
    assert weather_episode.similarity_score > 0


class TestContextManager:
  """Tests for ContextManager class."""

  @pytest.fixture
  def context_manager(self):
    """Create context manager."""
    return ContextManager(
        max_context_tokens=1000,
        relevance_weight=0.7,
        recency_weight=0.3,
    )

  def test_select_relevant_context_empty(self, context_manager):
    """Test selecting context with empty memories."""
    result = context_manager.select_relevant_context(
        current_task="test task",
        available_memories=[],
    )
    assert result == []

  def test_select_relevant_context_by_relevance(self, context_manager):
    """Test selecting context based on relevance."""
    now = datetime.now(timezone.utc)
    memories = [
        Episode(
            session_id="1",
            timestamp=now - timedelta(hours=1),
            user_message="weather forecast tomorrow",
            agent_response="It will be sunny",
        ),
        Episode(
            session_id="2",
            timestamp=now - timedelta(hours=2),
            user_message="stock market news",
            agent_response="Markets are up",
        ),
        Episode(
            session_id="3",
            timestamp=now - timedelta(hours=3),
            user_message="weather today sunny",
            agent_response="Yes it is sunny",
        ),
    ]

    result = context_manager.select_relevant_context(
        current_task="weather forecast",
        available_memories=memories,
    )

    # Weather-related memories should be prioritized
    assert len(result) > 0
    # First result should be weather-related
    assert "weather" in result[0].user_message.lower()

  def test_compute_relevance(self, context_manager):
    """Test relevance computation."""
    memory = Episode(
        session_id="1",
        timestamp=datetime.now(timezone.utc),
        user_message="weather forecast sunny day",
        agent_response=None,
    )

    score = context_manager._compute_relevance(memory, "weather forecast")

    # "weather" and "forecast" are in memory, so score should be 1.0
    assert score == 1.0

  def test_compute_recency_weight(self, context_manager):
    """Test recency weight computation."""
    now = datetime.now(timezone.utc)

    # Recent timestamp should have high weight
    recent = now - timedelta(hours=1)
    recent_weight = context_manager._compute_recency_weight(recent, now)
    assert recent_weight > 0.9

    # Old timestamp should have low weight
    old = now - timedelta(days=7)
    old_weight = context_manager._compute_recency_weight(old, now)
    assert old_weight < 0.01

  def test_estimate_tokens(self, context_manager):
    """Test token estimation."""
    short_episode = Episode(
        session_id="1",
        timestamp=datetime.now(timezone.utc),
        user_message="Hi",
        agent_response="Hello",
    )

    long_episode = Episode(
        session_id="2",
        timestamp=datetime.now(timezone.utc),
        user_message="What is the weather forecast for tomorrow?",
        agent_response="Tomorrow will be sunny with a high of 75 degrees.",
    )

    short_tokens = context_manager._estimate_tokens(short_episode)
    long_tokens = context_manager._estimate_tokens(long_episode)

    assert short_tokens < long_tokens


class TestUserProfileBuilder:
  """Tests for UserProfileBuilder class."""

  @pytest.fixture
  def mock_client(self):
    """Create mock BigQuery client."""
    return MagicMock()

  @pytest.fixture
  def profile_builder(self, mock_client):
    """Create profile builder with mock client."""
    return UserProfileBuilder(
        project_id="test-project",
        dataset_id="test-dataset",
        table_id="test-table",
        client=mock_client,
    )

  @pytest.mark.asyncio
  async def test_build_profile_basic(self, profile_builder, mock_client):
    """Test building basic user profile."""
    # Mock stats query
    stats_results = [
        {
            "session_count": 25,
            "last_interaction": datetime.now(timezone.utc),
            "tools_used": ["search", "format", "send"],
        }
    ]

    # Mock messages query
    messages_results = [
        {"message": "What is the weather?", "timestamp": datetime.now()},
        {"message": "Show me the news", "timestamp": datetime.now()},
    ]

    mock_query_job = MagicMock()
    mock_query_job.result.side_effect = [stats_results, messages_results]
    mock_client.query.return_value = mock_query_job

    # Patch LLM call for pattern extraction
    with patch.object(
        profile_builder,
        "_extract_patterns_with_llm",
        new_callable=AsyncMock,
    ):
      profile = await profile_builder.build_profile("user-123")

    assert profile.user_id == "user-123"
    assert profile.session_count == 25
    assert profile.preferred_tools == ["search", "format", "send"]


class TestBigQueryMemoryService:
  """Tests for BigQueryMemoryService class."""

  @pytest.fixture
  def mock_client(self):
    """Create mock BigQuery client."""
    return MagicMock()

  @pytest.fixture
  def memory_service(self, mock_client):
    """Create memory service with mock client."""
    return BigQueryMemoryService(
        project_id="test-project",
        dataset_id="test-dataset",
        table_id="test-table",
        client=mock_client,
    )

  @pytest.mark.asyncio
  async def test_search_memory(self, memory_service, mock_client):
    """Test searching memory."""
    # Mock episodic memory results
    mock_results = [
        {
            "session_id": "sess-1",
            "content": '{"text_summary": "weather forecast"}',
            "timestamp": datetime.now(timezone.utc),
            "user_id": "user-123",
            "event_type": "USER_MESSAGE_RECEIVED",
        }
    ]

    mock_query_job = MagicMock()
    mock_query_job.result.return_value = mock_results
    mock_client.query.return_value = mock_query_job

    response = await memory_service.search_memory(
        app_name="test_app",
        user_id="user-123",
        query="weather",
    )

    assert response.memories is not None

  @pytest.mark.asyncio
  async def test_add_session_to_memory(self, memory_service):
    """Test adding session to memory (no-op with analytics plugin)."""
    from google.adk.sessions.session import Session

    session = Session(
        id="sess-123",
        app_name="test_app",
        user_id="user-123",
        state={},
        events=[],
        last_update_time=0.0,
    )

    # Should not raise
    await memory_service.add_session_to_memory(session)

  @pytest.mark.asyncio
  async def test_get_session_context(self, memory_service, mock_client):
    """Test getting session context."""
    mock_results = [
        {
            "session_id": "sess-old",
            "event_type": "USER_MESSAGE_RECEIVED",
            "timestamp": datetime.now(timezone.utc),
            "content_summary": "Previous message",
            "response": None,
            "agent": "agent",
        }
    ]

    mock_query_job = MagicMock()
    mock_query_job.result.return_value = mock_results
    mock_client.query.return_value = mock_query_job

    episodes = await memory_service.get_session_context(
        user_id="user-123",
        current_session_id="sess-current",
        lookback_sessions=5,
    )

    assert len(episodes) == 1
    assert episodes[0].session_id == "sess-old"
