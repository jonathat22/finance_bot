"""
Tests for pydantic models in utils.models.
"""

import pytest
from pydantic import ValidationError
from utils.models import (
    Citation,
    ResearchAnswer,
    SourceType,
    FinancialDocument,
    DocumentSearchResult
)


class TestCitation:
    """Tests for Citation model."""

    def test_citation_creation(self, sample_citation_data):
        """Test creating a citation with valid data."""
        citation = Citation(**sample_citation_data)

        assert citation.title == sample_citation_data["title"]
        assert citation.url == sample_citation_data["url"]
        assert citation.snippet == sample_citation_data["snippet"]
        assert citation.source_type == sample_citation_data["source_type"]

    def test_citation_minimal(self):
        """Test creating a citation with minimal required fields."""
        citation = Citation(
            title="Test Title",
            url="https://example.com"
        )

        assert citation.title == "Test Title"
        assert citation.url == "https://example.com"
        assert citation.snippet is None
        assert citation.source_type is None

    def test_citation_missing_required_fields(self):
        """Test that citation requires title and url."""
        with pytest.raises(ValidationError):
            Citation(title="Test Title")  # Missing url

        with pytest.raises(ValidationError):
            Citation(url="https://example.com")  # Missing title


class TestResearchAnswer:
    """Tests for ResearchAnswer model."""

    def test_research_answer_creation(self, sample_research_answer_data):
        """Test creating a research answer with valid data."""
        answer = ResearchAnswer(**sample_research_answer_data)

        assert answer.answer == sample_research_answer_data["answer"]
        assert len(answer.citations) == 1
        assert answer.confidence == "high"

    def test_research_answer_empty_citations(self):
        """Test research answer with no citations."""
        answer = ResearchAnswer(
            answer="This is an answer without citations."
        )

        assert answer.answer == "This is an answer without citations."
        assert answer.citations == []
        assert answer.confidence is None

    def test_research_answer_confidence_levels(self):
        """Test different confidence levels."""
        for confidence in ["high", "medium", "low"]:
            answer = ResearchAnswer(
                answer="Test answer",
                confidence=confidence
            )
            assert answer.confidence == confidence


class TestSourceType:
    """Tests for SourceType enum."""

    def test_source_type_values(self):
        """Test that all expected source types exist."""
        assert SourceType.SEC_FILING.value == "SEC Filing"
        assert SourceType.COMPANY_IR.value == "Company IR"
        assert SourceType.NEWS.value == "Financial News"
        assert SourceType.RESEARCH.value == "Research Report"
        assert SourceType.GOVERNMENT.value == "Government"
        assert SourceType.OTHER.value == "Other"


class TestFinancialDocument:
    """Tests for FinancialDocument model."""

    def test_financial_document_creation(self):
        """Test creating a financial document."""
        doc = FinancialDocument(
            url="https://sec.gov/10-k",
            title="Apple 10-K 2024",
            source_type="SEC_EDGAR",
            document_type="10-K",
            company_name="Apple Inc.",
            year="2024"
        )

        assert doc.url == "https://sec.gov/10-k"
        assert doc.title == "Apple 10-K 2024"
        assert doc.source_type == "SEC_EDGAR"
        assert doc.document_type == "10-K"
        assert doc.company_name == "Apple Inc."
        assert doc.year == "2024"

    def test_financial_document_minimal(self):
        """Test creating a financial document with minimal fields."""
        doc = FinancialDocument(
            url="https://example.com/doc",
            title="Test Document",
            source_type="OTHER"
        )

        assert doc.url == "https://example.com/doc"
        assert doc.title == "Test Document"
        assert doc.source_type == "OTHER"
        assert doc.document_type is None
        assert doc.company_name is None
        assert doc.year is None


class TestDocumentSearchResult:
    """Tests for DocumentSearchResult model."""

    def test_document_search_result_creation(self):
        """Test creating a document search result."""
        result = DocumentSearchResult(
            company_identified="Apple Inc.",
            documents_found=[
                FinancialDocument(
                    url="https://sec.gov/10-k",
                    title="Apple 10-K",
                    source_type="SEC_EDGAR"
                )
            ],
            search_summary="Found 1 document for Apple Inc."
        )

        assert result.company_identified == "Apple Inc."
        assert len(result.documents_found) == 1
        assert result.search_summary == "Found 1 document for Apple Inc."
