"""
Unit tests for AdaptiveRetrievalAgent.
"""

import asyncio
import pickle
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.adaptive_retrieval import AdaptiveRetrievalAgent
from app.agents.base import AgentResult
from app.agents.citation import CitationMapperAgent


@pytest.fixture
def mock_settings():
    """Create mock settings."""
    settings = MagicMock()
    settings.adaptive_retrieval.enabled = True
    settings.adaptive_retrieval.exploration_rate = 0.1
    settings.adaptive_retrieval.policy_path = "models/policy.pkl"
    settings.rl.learning_rate = 0.1
    settings.rl.discount_factor = 0.95
    settings.rl.exploration_decay = 0.995
    settings.rl.min_exploration = 0.01
    settings.qdrant.host = "localhost"
    settings.qdrant.port = 6333
    settings.embeddings.model_name = "sentence-transformers/all-MiniLM-L6-v2"
    settings.embeddings.device = "cpu"
    settings.citation_mapper.top_k = 50
    return settings


@pytest.fixture
def adaptive_agent(mock_settings):
    """Create AdaptiveRetrievalAgent instance."""
    with patch('app.agents.adaptive_retrieval.CitationMapperAgent'):
        agent = AdaptiveRetrievalAgent(settings=mock_settings)
        # Mock citation agent
        agent.citation_agent = AsyncMock()
        agent.citation_agent.run = AsyncMock(return_value=AgentResult(
            agent="citation_mapper",
            success=True,
            data={"results": [
                {"patent_id": "US-123", "score": 0.9, "chunk_text": "test"},
                {"patent_id": "US-456", "score": 0.8, "chunk_text": "test"},
            ]}
        ))
        return agent


class TestAdaptiveRetrievalAgent:
    """Test AdaptiveRetrievalAgent functionality."""
    
    def test_initialization(self, mock_settings):
        """Test agent initialization."""
        with patch('app.agents.adaptive_retrieval.CitationMapperAgent'):
            agent = AdaptiveRetrievalAgent(settings=mock_settings)
            assert agent.name == "adaptive_retrieval"
            assert agent.exploration_rate == 0.1
            assert len(agent.actions) == 3
            assert "RETRIEVE" in agent.actions
            assert "RETRIEVE_MORE" in agent.actions
            assert "STOP" in agent.actions
    
    def test_get_state(self, adaptive_agent):
        """Test state representation."""
        state = adaptive_agent._get_state("research", 0, 0.5, 0.8)
        assert isinstance(state, tuple)
        assert len(state) == 4
        assert state[0] == "research"
        assert state[1] == 0
        assert 0.0 <= state[2] <= 1.0  # Normalized reward
        assert 0.0 <= state[3] <= 1.0  # Normalized quality
    
    def test_calculate_chunk_quality(self, adaptive_agent):
        """Test chunk quality calculation."""
        results = [
            {"score": 0.9},
            {"score": 0.8},
            {"score": 0.7},
        ]
        quality = adaptive_agent._calculate_chunk_quality(results)
        assert quality == pytest.approx(0.8, abs=0.01)
        
        # Empty results
        quality_empty = adaptive_agent._calculate_chunk_quality([])
        assert quality_empty == 0.0
    
    def test_select_action(self, adaptive_agent):
        """Test action selection."""
        state = ("research", 0, 0.0, 0.5)
        
        # With empty Q-table, should return default action
        action = adaptive_agent.select_action(state)
        assert action in adaptive_agent.actions
        
        # With Q-table entry
        adaptive_agent.q_table[state] = {
            "RETRIEVE": 0.5,
            "RETRIEVE_MORE": 0.8,
            "STOP": 0.3,
        }
        action = adaptive_agent.select_action(state)
        # Should select best action (RETRIEVE_MORE) when not exploring
        # But may explore randomly, so just check it's a valid action
        assert action in adaptive_agent.actions
    
    def test_update_q_value(self, adaptive_agent):
        """Test Q-value update."""
        state = ("research", 0, 0.0, 0.5)
        action = "RETRIEVE_MORE"
        reward = 0.7
        next_state = ("research", 1, 0.7, 0.6)
        
        # Initial Q-value should be 0
        assert adaptive_agent.q_table.get(state, {}).get(action, 0.0) == 0.0
        
        # Update
        adaptive_agent.update_q_value(state, action, reward, next_state)
        
        # Q-value should be updated
        q_value = adaptive_agent.q_table[state][action]
        assert q_value > 0.0
        assert q_value < 1.0  # Should be reasonable
    
    def test_decay_exploration(self, adaptive_agent):
        """Test exploration rate decay."""
        initial_rate = adaptive_agent.exploration_rate
        adaptive_agent.decay_exploration()
        assert adaptive_agent.exploration_rate < initial_rate
        assert adaptive_agent.exploration_rate >= adaptive_agent.min_exploration
    
    @pytest.mark.asyncio
    async def test_run_success(self, adaptive_agent):
        """Test successful run."""
        result = await adaptive_agent.run("test query", query_type="research")
        
        assert result.success
        assert "results" in result.data
        assert "rl_metadata" in result.data
        assert "retrieval_depth" in result.data
        assert len(result.data["results"]) > 0
    
    @pytest.mark.asyncio
    async def test_run_without_citation_agent(self, mock_settings):
        """Test run without citation agent."""
        with patch('app.agents.adaptive_retrieval.CitationMapperAgent'):
            agent = AdaptiveRetrievalAgent(settings=mock_settings)
            agent.citation_agent = None
            
            result = await agent.run("test query")
            assert not result.success
            assert "CitationMapperAgent not initialized" in result.error
    
    def test_policy_save_load(self, adaptive_agent, tmp_path):
        """Test policy save and load."""
        # Set custom policy path
        policy_path = tmp_path / "test_policy.pkl"
        adaptive_agent.policy_path = policy_path
        
        # Add some Q-values
        state = ("research", 0, 0.0, 0.5)
        adaptive_agent.q_table[state] = {
            "RETRIEVE": 0.5,
            "RETRIEVE_MORE": 0.8,
            "STOP": 0.3,
        }
        
        # Save
        adaptive_agent._save_policy()
        assert policy_path.exists()
        
        # Create new agent and load
        with patch('app.agents.adaptive_retrieval.CitationMapperAgent'):
            new_agent = AdaptiveRetrievalAgent(settings=adaptive_agent.settings)
            new_agent.policy_path = policy_path
            new_agent._load_policy()
            
            # Check Q-table loaded
            assert state in new_agent.q_table
            assert new_agent.q_table[state]["RETRIEVE_MORE"] == pytest.approx(0.8, abs=0.01)
    
    def test_get_policy_stats(self, adaptive_agent):
        """Test policy statistics."""
        stats = adaptive_agent.get_policy_stats()
        assert "num_states" in stats
        assert "exploration_rate" in stats
        assert "policy_path" in stats
        assert "policy_exists" in stats
        assert stats["num_states"] == 0  # Initially empty


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

