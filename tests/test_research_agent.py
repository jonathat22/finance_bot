"""
Tests for ResearchAgent class.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from utils.models import Citation, ResearchAnswer
from utils.research_agent import (
    RESEARCH_SYSTEM_PROMPT,
    ConversationMessage,
    ResearchAgent,
    exponential_backoff_retry,
)


class TestExponentialBackoffRetry:
    """Tests for exponential backoff retry logic."""

    def test_successful_first_try(self):
        """Test that function succeeds on first try."""
        mock_func = Mock(return_value="success")
        result = exponential_backoff_retry(mock_func, max_retries=3)

        assert result == "success"
        assert mock_func.call_count == 1

    def test_retry_on_rate_limit(self):
        """Test retry logic on rate limit error."""
        mock_func = Mock(side_effect=[Exception("429 rate limit"), "success"])

        result = exponential_backoff_retry(mock_func, max_retries=3, base_delay=0.01)

        assert result == "success"
        assert mock_func.call_count == 2

    def test_max_retries_exceeded(self):
        """Test that max retries raises exception."""
        mock_func = Mock(side_effect=Exception("429 rate limit"))

        with pytest.raises(Exception, match="API quota exhausted"):
            exponential_backoff_retry(mock_func, max_retries=2, base_delay=0.01)

        assert mock_func.call_count == 2

    def test_non_rate_limit_error_raises_immediately(self):
        """Test that non-rate-limit errors are raised immediately."""
        mock_func = Mock(side_effect=ValueError("Invalid input"))

        with pytest.raises(ValueError, match="Invalid input"):
            exponential_backoff_retry(mock_func, max_retries=3)

        assert mock_func.call_count == 1


class TestConversationMessage:
    """Tests for ConversationMessage dataclass."""

    def test_conversation_message_creation(self):
        """Test creating a conversation message."""
        msg = ConversationMessage(role="user", content="What is Apple's market cap?")

        assert msg.role == "user"
        assert msg.content == "What is Apple's market cap?"


class TestResearchAgent:
    """Tests for ResearchAgent class."""

    @patch("utils.research_agent.genai.Client")
    def test_initialization(self, mock_client):
        """Test ResearchAgent initialization."""
        agent = ResearchAgent(model="gemini-2.5-flash", temperature=0.1, verbose=True)

        assert agent.model == "gemini-2.5-flash"
        assert agent.temperature == 0.1
        assert agent.verbose is True
        assert agent.search_tool is not None

    @patch("utils.research_agent.genai.Client")
    def test_log_verbose_enabled(self, mock_client):
        """Test that logging works when verbose is enabled."""
        agent = ResearchAgent(verbose=True)

        # Just ensure _log doesn't raise an error
        agent._log("Test message")

    @patch("utils.research_agent.genai.Client")
    def test_log_verbose_disabled(self, mock_client):
        """Test that logging is silent when verbose is disabled."""
        agent = ResearchAgent(verbose=False)

        # Just ensure _log doesn't raise an error
        agent._log("Test message")

    @patch("utils.research_agent.genai.Client")
    def test_build_context_empty_history(self, mock_client):
        """Test building context with empty history."""
        agent = ResearchAgent()
        context = agent._build_context(None)

        assert context == ""

    @patch("utils.research_agent.genai.Client")
    def test_build_context_with_history(self, mock_client):
        """Test building context with conversation history."""
        agent = ResearchAgent()
        history = [
            {"role": "user", "content": "What is Apple's market cap?"},
            {"role": "assistant", "content": "Apple's market cap is $3 trillion."},
        ]

        context = agent._build_context(history)

        assert "Previous conversation:" in context
        assert "User: What is Apple's market cap?" in context
        assert "Assistant: Apple's market cap is $3 trillion." in context

    @patch("utils.research_agent.genai.Client")
    def test_build_context_limits_messages(self, mock_client):
        """Test that context only includes last 6 messages."""
        agent = ResearchAgent()
        history = [{"role": "user", "content": f"Question {i}"} for i in range(10)]

        context = agent._build_context(history)

        # Should only include last 6 messages (indices 4-9)
        assert "Question 9" in context
        assert "Question 4" in context
        assert "Question 3" not in context

    @patch("utils.research_agent.genai.Client")
    def test_categorize_source_sec_filing(self, mock_client):
        """Test categorizing SEC filing sources."""
        agent = ResearchAgent()

        assert (
            agent._categorize_source("https://www.sec.gov/edgar/10-k") == "SEC Filing"
        )
        assert agent._categorize_source("https://edgar.sec.gov/file") == "SEC Filing"

    @patch("utils.research_agent.genai.Client")
    def test_categorize_source_company_ir(self, mock_client):
        """Test categorizing company IR sources."""
        agent = ResearchAgent()

        assert agent._categorize_source("https://investor.apple.com") == "Company IR"
        assert agent._categorize_source("https://apple.com/investors") == "Company IR"
        assert agent._categorize_source("https://ir.microsoft.com") == "Company IR"

    @patch("utils.research_agent.genai.Client")
    def test_categorize_source_government(self, mock_client):
        """Test categorizing government sources."""
        agent = ResearchAgent()

        assert (
            agent._categorize_source("https://www.federalreserve.gov") == "Government"
        )
        assert agent._categorize_source("https://treasury.gov") == "Government"

    @patch("utils.research_agent.genai.Client")
    def test_categorize_source_financial_news(self, mock_client):
        """Test categorizing financial news sources."""
        agent = ResearchAgent()

        assert (
            agent._categorize_source("https://www.reuters.com/article")
            == "Financial News"
        )
        assert (
            agent._categorize_source("https://www.bloomberg.com/news")
            == "Financial News"
        )
        assert (
            agent._categorize_source("https://www.cnbc.com/markets") == "Financial News"
        )
        assert (
            agent._categorize_source("https://www.wsj.com/articles") == "Financial News"
        )

    @patch("utils.research_agent.genai.Client")
    def test_categorize_source_other(self, mock_client):
        """Test categorizing unknown sources."""
        agent = ResearchAgent()

        assert agent._categorize_source("https://www.example.com") == "Other"
        assert agent._categorize_source("https://www.randomsite.com") == "Other"

    @patch("utils.research_agent.genai.Client")
    def test_get_title_from_url(self, mock_client):
        """Test extracting title from URL."""
        agent = ResearchAgent()

        title = agent._get_title_from_url("https://www.example.com/article/test-page")
        assert "example.com" in title
        assert "article" in title

        title = agent._get_title_from_url("https://example.com")
        assert title == "example.com"

    @patch("utils.research_agent.genai.Client")
    def test_extract_citations_pattern1(self, mock_client):
        """Test extracting citations with [Source: Title - URL] pattern."""
        agent = ResearchAgent()
        text = (
            "Apple's market cap is $3T [Source: Apple IR - https://investor.apple.com]"
        )

        cleaned_text, citations = agent._extract_citations(text)

        assert len(citations) == 1
        assert citations[0].title == "Apple IR"
        assert citations[0].url == "https://investor.apple.com"
        assert "[Source:" not in cleaned_text

    @patch("utils.research_agent.genai.Client")
    def test_extract_citations_pattern2(self, mock_client):
        """Test extracting citations with [Title](URL) markdown pattern."""
        agent = ResearchAgent()
        text = "Check out [Apple Investor Relations](https://investor.apple.com)"

        cleaned_text, citations = agent._extract_citations(text)

        assert len(citations) == 1
        assert citations[0].title == "Apple Investor Relations"
        assert citations[0].url == "https://investor.apple.com"

    @patch("utils.research_agent.genai.Client")
    def test_extract_citations_removes_sources_section(self, mock_client):
        """Test that Sources: section is removed from cleaned text."""
        agent = ResearchAgent()
        text = """Apple's market cap is $3 trillion.

Sources:
- [Apple IR](https://investor.apple.com)
- [Bloomberg](https://bloomberg.com)"""

        cleaned_text, citations = agent._extract_citations(text)

        assert "Sources:" not in cleaned_text
        assert "Apple's market cap is $3 trillion." in cleaned_text
        assert len(citations) == 2

    @patch("utils.research_agent.genai.Client")
    def test_extract_citations_from_grounding(self, mock_client, mock_gemini_response):
        """Test extracting citations from grounding metadata."""
        agent = ResearchAgent()
        citations = agent._extract_citations_from_grounding(mock_gemini_response)

        assert len(citations) == 1
        assert citations[0].url == "https://investor.apple.com"
        assert citations[0].title == "Apple Investor Relations"
        assert citations[0].source_type == "Company IR"

    @patch("utils.research_agent.genai.Client")
    def test_answer_question_with_mock(self, mock_client, mock_gemini_response):
        """Test answer_question method with mocked API response."""
        # Setup mock
        mock_instance = mock_client.return_value
        mock_instance.models.generate_content.return_value = mock_gemini_response

        agent = ResearchAgent()
        result = agent.answer_question("What is Apple's market cap?")

        assert isinstance(result, ResearchAnswer)
        assert "Apple's current market cap" in result.answer
        assert len(result.citations) > 0
        assert result.confidence in ["high", "medium", "low"]

    @patch("utils.research_agent.genai.Client")
    def test_answer_question_simple(self, mock_client, mock_gemini_response):
        """Test answer_question_simple returns just the answer text."""
        # Setup mock
        mock_instance = mock_client.return_value
        mock_instance.models.generate_content.return_value = mock_gemini_response

        agent = ResearchAgent()
        result = agent.answer_question_simple("What is Apple's market cap?")

        assert isinstance(result, str)
        assert "Apple's current market cap" in result
