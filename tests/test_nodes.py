"""Unit tests for individual agent nodes."""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from langchain_core.messages import AIMessage
import json

from graph.nodes import router_node, extractor_node, critic_node


@pytest.mark.unit
class TestRouterNode:
    """Test router node classification logic."""
    
    @pytest.mark.asyncio
    async def test_router_legal_intent(self):
        """Test router correctly identifies LEGAL intent."""
        mock_state = {"query": "lawsuit against apple"}
        
        with patch('graph.nodes.router_llm') as mock_llm:
            mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content='{"intent": "LEGAL"}'))
            
            result = await router_node(mock_state)
            
            assert result["intent"] == "LEGAL"
            assert "lawsuit" in mock_state["query"].lower()
    
    @pytest.mark.asyncio
    async def test_router_technical_intent(self):
        """Test router correctly identifies TECHNICAL intent."""
        mock_state = {"query": "prior art for transformer neural networks"}
        
        with patch('graph.nodes.router_llm') as mock_llm:
            mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content='{"intent": "TECHNICAL"}'))
            
            result = await router_node(mock_state)
            
            assert result["intent"] == "TECHNICAL"
    
    @pytest.mark.asyncio
    async def test_router_both_intent(self):
        """Test router correctly identifies BOTH intent."""
        mock_state = {"query": "patents on touchscreens and litigation cases"}
        
        with patch('graph.nodes.router_llm') as mock_llm:
            mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content='{"intent": "BOTH"}'))
            
            result = await router_node(mock_state)
            
            assert result["intent"] == "BOTH"
    
    @pytest.mark.asyncio
    async def test_router_json_parsing(self):
        """Test router handles JSON parsing correctly."""
        mock_state = {"query": "test query"}
        
        with patch('graph.nodes.router_llm') as mock_llm:
            # Test with markdown code block
            mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content='```json\n{"intent": "LEGAL"}\n```'))
            
            result = await router_node(mock_state)
            
            # Should extract JSON from markdown
            assert "intent" in result


@pytest.mark.unit
class TestExtractorNode:
    """Test extractor node query expansion."""
    
    @pytest.mark.asyncio
    async def test_extractor_output_schema(self):
        """Test extractor produces correct JSON structure."""
        mock_state = {"query": "car suspension"}
        
        mock_response = '{"keywords": ["car suspension", "vehicle damping system", "active chassis control", "CPC: B60G"], "date_range": null}'
        
        with patch('graph.nodes.extractor_llm') as mock_llm:
            mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content=mock_response))
            
            result = await extractor_node(mock_state)
            
            assert "keywords" in result
            assert isinstance(result["keywords"], list)
            assert len(result["keywords"]) >= 3
            assert len(result["keywords"]) <= 5
            assert "CPC: B60G" in result["keywords"]
    
    @pytest.mark.asyncio
    async def test_extractor_query_expansion(self):
        """Test extractor generates 3-5 variations."""
        mock_state = {"query": "transformer neural network"}
        
        mock_response = '{"keywords": ["transformer neural network", "neural network transformer", "sequence modeling", "deep learning model"], "date_range": null}'
        
        with patch('graph.nodes.extractor_llm') as mock_llm:
            mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content=mock_response))
            
            result = await extractor_node(mock_state)
            
            keywords = result["keywords"]
            assert len(keywords) >= 3
            assert len(keywords) <= 5
            assert all(isinstance(kw, str) for kw in keywords)
    
    @pytest.mark.asyncio
    async def test_extractor_fallback(self):
        """Test extractor fallback when LLM returns invalid JSON."""
        mock_state = {"query": "test query"}
        
        with patch('graph.nodes.extractor_llm') as mock_llm:
            mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Invalid response"))
            
            result = await extractor_node(mock_state)
            
            # Should fallback to using original query
            assert "keywords" in result
            assert len(result["keywords"]) >= 1


@pytest.mark.unit
class TestCriticNode:
    """Test critic node citation verification."""
    
    @pytest.mark.asyncio
    async def test_critic_citation_extraction(self):
        """Test critic extracts citations correctly."""
        mock_state = {
            "draft": "The system is described in [[US-2016168309-A1]](https://patents.google.com/patent/US-2016168309-A1).",
            "documents": [
                {
                    "patent_id": "US-2016168309-A1",
                    "title": "Test Patent",
                    "abstract": "Test",
                }
            ],
        }
        
        with patch('graph.nodes.critic_llm') as mock_llm:
            mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content='{"status": "PASS", "feedback": "All citations valid"}'))
            
            result = await critic_node(mock_state)
            
            assert "critique" in result
            critique = result["critique"]
            assert "status" in critique
    
    @pytest.mark.asyncio
    async def test_critic_validation_logic(self):
        """Test critic rejects invalid citations."""
        mock_state = {
            "draft": "The system is described in [[US9999999]](url).",
            "documents": [
                {
                    "patent_id": "US-2016168309-A1",
                    "title": "Test Patent",
                    "abstract": "Test",
                }
            ],
        }
        
        with patch('graph.nodes.critic_llm') as mock_llm:
            mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content='{"status": "FAIL", "feedback": "Invalid citation"}'))
            
            result = await critic_node(mock_state)
            
            critique = result["critique"]
            # Should fail due to missing citation
            assert critique["status"] == "FAIL"
    
    @pytest.mark.asyncio
    async def test_critic_regex_parsing(self):
        """Test critic correctly parses citation patterns."""
        import re
        
        draft = "Patent [[US123]](url1) and [[US456]](url2)"
        citations = re.findall(r'\[\[([^\]]+)\]\]', draft)
        
        assert len(citations) == 2
        assert "US123" in citations
        assert "US456" in citations


