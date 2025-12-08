"""Integration tests for graph flow and state transitions."""
import pytest
from graph.graph import should_retry, AgentState


@pytest.mark.integration
class TestGraphFlow:
    """Test graph flow and conditional edges."""
    
    def test_critic_retry_logic(self):
        """Test that graph retries on critic FAIL."""
        initial_state: AgentState = {
            "query": "test",
            "draft": "Bad draft with no citations",
            "critique": {"status": "FAIL", "feedback": "Missing citations"},
            "retry_count": 0,
            "intent": None,
            "keywords": None,
            "date_range": None,
            "documents": [],
            "litigation_context": [],
            "final_response": None,
        }
        
        next_node = should_retry(initial_state)
        assert next_node == "retry"  # Should retry
    
    def test_max_retries(self):
        """Test that graph stops after max retries."""
        initial_state: AgentState = {
            "query": "test",
            "draft": "Bad draft",
            "critique": {"status": "FAIL", "feedback": "Still bad"},
            "retry_count": 3,  # At max
            "intent": None,
            "keywords": None,
            "date_range": None,
            "documents": [],
            "litigation_context": [],
            "final_response": None,
        }
        
        next_node = should_retry(initial_state)
        assert next_node == "end"  # Should give up
    
    def test_critic_pass_ends_workflow(self):
        """Test that PASS from critic ends workflow."""
        initial_state: AgentState = {
            "query": "test",
            "draft": "Good draft with citations",
            "critique": {"status": "PASS", "feedback": "All good"},
            "retry_count": 0,
            "intent": None,
            "keywords": None,
            "date_range": None,
            "documents": [],
            "litigation_context": [],
            "final_response": None,
        }
        
        next_node = should_retry(initial_state)
        assert next_node == "end"  # Should end
    
    def test_conditional_edge_logic(self):
        """Test conditional edge function logic."""
        # Test FAIL with retry_count < 3
        state1: AgentState = {
            "query": "test",
            "critique": {"status": "FAIL"},
            "retry_count": 1,
            "intent": None,
            "keywords": None,
            "date_range": None,
            "documents": [],
            "litigation_context": [],
            "draft": None,
            "final_response": None,
        }
        assert should_retry(state1) == "retry"
        
        # Test FAIL with retry_count >= 3
        state2: AgentState = {
            "query": "test",
            "critique": {"status": "FAIL"},
            "retry_count": 3,
            "intent": None,
            "keywords": None,
            "date_range": None,
            "documents": [],
            "litigation_context": [],
            "draft": None,
            "final_response": None,
        }
        assert should_retry(state2) == "end"
        
        # Test PASS
        state3: AgentState = {
            "query": "test",
            "critique": {"status": "PASS"},
            "retry_count": 0,
            "intent": None,
            "keywords": None,
            "date_range": None,
            "documents": [],
            "litigation_context": [],
            "draft": None,
            "final_response": None,
        }
        assert should_retry(state3) == "end"
    
    def test_state_transitions(self):
        """Test that state updates correctly through workflow."""
        from graph.graph import increment_retry_count
        
        initial_state: AgentState = {
            "query": "test",
            "retry_count": 0,
            "intent": None,
            "keywords": None,
            "date_range": None,
            "documents": [],
            "litigation_context": [],
            "draft": None,
            "critique": None,
            "final_response": None,
        }
        
        # Test retry count increment
        updated = increment_retry_count(initial_state)
        assert updated["retry_count"] == 1
        
        updated = increment_retry_count(updated)
        assert updated["retry_count"] == 2


