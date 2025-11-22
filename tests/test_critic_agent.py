"""
Unit tests for CriticAgent reward components.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.agents.critic import CriticAgent
from app.agents.base import AgentResult


@pytest.fixture
def mock_settings():
    """Create mock settings object."""
    settings = MagicMock()
    settings.database = MagicMock()
    settings.database.user = "testuser"
    settings.database.password = "testpass"
    settings.database.host = "localhost"
    settings.database.port = 5432
    settings.database.database = "testdb"
    settings.data = MagicMock()
    settings.data.citations = MagicMock()
    settings.data.citations.max_hops = 2
    return settings


@pytest.fixture
def critic_agent(mock_settings):
    """Create CriticAgent instance."""
    return CriticAgent(settings=mock_settings)


@pytest.mark.asyncio
async def test_citation_overlap_no_chunks(critic_agent):
    """Test citation overlap with no retrieved chunks."""
    score = await critic_agent._compute_citation_overlap([])
    assert score == 0.0


@pytest.mark.asyncio
async def test_citation_overlap_with_mock_db(critic_agent):
    """Test citation overlap calculation with mocked database."""
    retrieved_chunks = [
        {"patent_id": "US-123-A1"},
        {"patent_id": "US-456-B2"},
    ]
    
    # Mock database connection and cursor
    mock_cursor = MagicMock()
    mock_cursor.fetchall.side_effect = [
        [("US-789-C1",)],  # One-hop cited
        [("US-101-D1",)],  # One-hop citing
        [("US-202-E1",)],  # Two-hop cited
        [("US-303-F1",)],  # Two-hop citing
    ]
    mock_cursor.fetchone.return_value = (2,)  # Internal citations
    
    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    
    with patch("app.agents.critic.psycopg.connect", return_value=mock_conn):
        score = await critic_agent._compute_citation_overlap(retrieved_chunks)
        assert 0.0 <= score <= 1.0


def test_cpc_relevance_exact_match(critic_agent):
    """Test CPC relevance with exact matches."""
    claims_analysis = {
        "cpc_codes": [
            {"code": "G06N3/063"},
            {"code": "H01L23/00"},
        ]
    }
    retrieved_chunks = [
        {"cpc_codes": [{"code": "G06N3/063"}]},
        {"cpc_codes": [{"code": "H01L23/00"}]},
    ]
    
    score = critic_agent._compute_cpc_relevance(claims_analysis, retrieved_chunks)
    assert score == 1.0


def test_cpc_relevance_hierarchical_match(critic_agent):
    """Test CPC relevance with hierarchical matching."""
    claims_analysis = {
        "cpc_codes": [
            {"code": "G06N3/063"},
        ]
    }
    retrieved_chunks = [
        {"cpc_codes": [{"code": "G06N3/04"}]},  # Should match at subclass level
    ]
    
    score = critic_agent._compute_cpc_relevance(claims_analysis, retrieved_chunks)
    assert 0.0 <= score <= 1.0


def test_cpc_relevance_no_match(critic_agent):
    """Test CPC relevance with no matches."""
    claims_analysis = {
        "cpc_codes": [
            {"code": "G06N3/063"},
        ]
    }
    retrieved_chunks = [
        {"cpc_codes": [{"code": "H01L23/00"}]},  # Different class
    ]
    
    score = critic_agent._compute_cpc_relevance(claims_analysis, retrieved_chunks)
    assert score == 0.0


def test_temporal_diversity_single_year(critic_agent):
    """Test temporal diversity with single year."""
    retrieved_chunks = [
        {"publication_date": "2020-01-15"},
        {"publication_date": "2020-06-20"},
    ]
    
    score = critic_agent._compute_temporal_diversity(retrieved_chunks)
    assert score == 0.0  # No diversity if same year


def test_temporal_diversity_multiple_years(critic_agent):
    """Test temporal diversity with multiple years."""
    retrieved_chunks = [
        {"publication_date": "2018-01-15"},
        {"publication_date": "2020-06-20"},
        {"publication_date": "2022-12-10"},
    ]
    
    score = critic_agent._compute_temporal_diversity(retrieved_chunks)
    assert 0.0 < score <= 1.0  # Should have some diversity


def test_temporal_diversity_integer_dates(critic_agent):
    """Test temporal diversity with integer date format."""
    retrieved_chunks = [
        {"publication_date": 20180115},
        {"publication_date": 20200620},
    ]
    
    score = critic_agent._compute_temporal_diversity(retrieved_chunks)
    assert 0.0 < score <= 1.0


@pytest.mark.asyncio
async def test_llm_fluency_with_mock_llm(critic_agent):
    """Test LLM fluency with mocked LLM service."""
    synthesis_output = {
        "executive_summary": "This is a well-written patent analysis with clear recommendations."
    }
    
    # Mock LLM service
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = "0.85"
    critic_agent.llm = mock_llm
    
    score = await critic_agent._compute_llm_fluency(synthesis_output)
    assert 0.0 <= score <= 1.0
    assert score == 0.85


@pytest.mark.asyncio
async def test_llm_fluency_no_llm(critic_agent):
    """Test LLM fluency when LLM is unavailable."""
    critic_agent.llm = None
    synthesis_output = {"executive_summary": "Test"}
    
    score = await critic_agent._compute_llm_fluency(synthesis_output)
    assert score == 0.5  # Default neutral score


@pytest.mark.asyncio
async def test_critic_run_complete(mock_settings):
    """Test complete CriticAgent.run() method."""
    critic_agent = CriticAgent(settings=mock_settings)
    
    # Mock all component methods
    with patch.object(critic_agent, "_compute_citation_overlap", return_value=0.8):
        with patch.object(critic_agent, "_compute_cpc_relevance", return_value=0.7):
            with patch.object(critic_agent, "_compute_temporal_diversity", return_value=0.6):
                with patch.object(critic_agent, "_compute_llm_fluency", return_value=0.9):
                    result = await critic_agent.run(
                        query="test query",
                        retrieved_chunks=[{"patent_id": "US-123"}],
                        claims_analysis={"cpc_codes": [{"code": "G06N"}]},
                        synthesis_output={"executive_summary": "Test"},
                    )
    
    assert result.success
    assert "score" in result.data
    assert "components" in result.data
    assert result.data["components"]["citation_overlap"] == 0.8
    assert result.data["components"]["cpc_relevance"] == 0.7
    assert result.data["components"]["temporal_diversity"] == 0.6
    assert result.data["components"]["llm_fluency"] == 0.9
    
    # Check weighted sum: 0.4*0.8 + 0.3*0.7 + 0.2*0.6 + 0.1*0.9 = 0.32 + 0.21 + 0.12 + 0.09 = 0.74
    expected_score = 0.4 * 0.8 + 0.3 * 0.7 + 0.2 * 0.6 + 0.1 * 0.9
    assert abs(result.data["score"] - expected_score) < 0.01


def test_cpc_hierarchical_match(critic_agent):
    """Test CPC hierarchical matching logic."""
    # Same main class, different subclass - should not match
    assert not critic_agent._cpc_hierarchical_match("G06N3/063", "G06N5/100")
    
    # Same main class and subclass - should match
    assert critic_agent._cpc_hierarchical_match("G06N3/063", "G06N3/04")
    
    # Different main class - should not match
    assert not critic_agent._cpc_hierarchical_match("G06N3/063", "H01L23/00")

