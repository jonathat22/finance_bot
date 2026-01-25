"""
Pytest configuration and shared fixtures.
"""

from unittest.mock import MagicMock, Mock

import pytest


@pytest.fixture
def sample_citation_data():
    """Sample citation data for testing."""
    return {
        "title": "Apple Q4 2024 Earnings Report",
        "url": "https://investor.apple.com/earnings/2024/Q4",
        "snippet": "Apple reports strong Q4 results",
        "source_type": "Company IR",
    }


@pytest.fixture
def sample_research_answer_data():
    """Sample research answer data for testing."""
    return {
        "answer": "Apple's market cap is approximately $3 trillion as of Q4 2024.",
        "citations": [
            {
                "title": "Apple Investor Relations",
                "url": "https://investor.apple.com",
                "snippet": "Market capitalization data",
                "source_type": "Company IR",
            }
        ],
        "confidence": "high",
    }


@pytest.fixture
def mock_gemini_response():
    """Mock Gemini API response."""
    mock_response = Mock()
    mock_response.text = "Apple's current market cap is approximately $3 trillion."

    # Mock grounding metadata
    mock_chunk = Mock()
    mock_web = Mock()
    mock_web.uri = "https://investor.apple.com"
    mock_web.title = "Apple Investor Relations"
    mock_chunk.web = mock_web

    mock_grounding = Mock()
    mock_grounding.grounding_chunks = [mock_chunk]
    mock_grounding.web_search_queries = ["apple market cap"]

    mock_candidate = Mock()
    mock_candidate.grounding_metadata = mock_grounding

    mock_response.candidates = [mock_candidate]

    return mock_response
