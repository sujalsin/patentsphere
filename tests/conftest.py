"""Pytest configuration and shared fixtures."""
import pytest
from unittest.mock import MagicMock, AsyncMock
from typing import Dict, Any

# Note: We don't define a custom event_loop fixture because pytest-asyncio
# in auto mode handles event loop creation automatically.


@pytest.fixture
def mock_router_llm():
    """Mock router LLM."""
    mock = AsyncMock()
    return mock


@pytest.fixture
def mock_extractor_llm():
    """Mock extractor LLM."""
    mock = AsyncMock()
    return mock


@pytest.fixture
def mock_synthesizer_llm():
    """Mock synthesizer LLM."""
    mock = AsyncMock()
    return mock


@pytest.fixture
def mock_critic_llm():
    """Mock critic LLM."""
    mock = AsyncMock()
    return mock


@pytest.fixture
def sample_state():
    """Sample agent state for testing."""
    return {
        "query": "test query",
        "intent": None,
        "keywords": None,
        "date_range": None,
        "documents": [],
        "litigation_context": [],
        "draft": None,
        "critique": None,
        "retry_count": 0,
        "final_response": None,
    }


@pytest.fixture
def sample_documents():
    """Sample patent documents for testing."""
    return [
        {
            "patent_id": "US-2016168309-A1",
            "title": "Test Patent 1",
            "abstract": "Test abstract for patent 1",
            "url": "https://patents.google.com/patent/US-2016168309-A1",
        },
        {
            "patent_id": "US-2016168354-A1",
            "title": "Test Patent 2",
            "abstract": "Test abstract for patent 2",
            "url": "https://patents.google.com/patent/US-2016168354-A1",
        },
    ]


@pytest.fixture
def sample_litigation():
    """Sample litigation data for testing."""
    return [
        {
            "case_number": "1:00-cv-00037",
            "case_name": "Test Case",
            "court_name": "E.D.Tex.",
            "filing_date": "2000-01-13",
            "case_status": "active",
            "plaintiff_name": "Test Plaintiff",
            "defendant_name": "Test Defendant",
            "patent_id": "US5655545",
            "outcome": None,
        }
    ]

