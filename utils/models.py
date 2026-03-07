from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, HttpUrl


class SourceType(str, Enum):
    """Types of sources for financial information."""

    SEC_FILING = "SEC Filing"
    COMPANY_IR = "Company IR"
    NEWS = "Financial News"
    RESEARCH = "Research Report"
    GOVERNMENT = "Government"
    OTHER = "Other"


class PageRequest(BaseModel):
    url: HttpUrl


class Summary(BaseModel):
    summary: str


class Citation(BaseModel):
    """Citation for a source used in research response."""

    title: str = Field(description="The title of the cited source.")
    url: str = Field(description="The URL of the cited source.")
    snippet: Optional[str] = Field(
        default=None, description="Relevant snippet or excerpt from the source."
    )
    source_type: Optional[str] = Field(
        default=None, description="Type of source (SEC Filing, News, etc.)"
    )


class ResearchAnswer(BaseModel):
    """Structured response from the research agent."""

    answer: str = Field(description="The main answer to the user's question.")
    citations: List[Citation] = Field(
        default_factory=list, description="List of citations supporting the answer."
    )
    confidence: Optional[str] = Field(
        default=None, description="Confidence level: 'high', 'medium', or 'low'"
    )


class AgentAnswer(BaseModel):
    """Legacy agent answer model."""

    answer: str = Field(description="The final answer provided by the agent.")
    citations: list[Citation] = Field(
        description="A list of citations supporting the answer."
    )


class FinancialDocument(BaseModel):
    """Model representing a discovered financial document."""

    url: str = Field(description="URL of the financial document")
    title: str = Field(description="Title or description of the document")
    source_type: str = Field(
        description="Type of source: 'SEC_EDGAR', 'COMPANY_IR', or 'OTHER'"
    )
    document_type: Optional[str] = Field(
        default=None,
        description="Type of document: '10-K', '10-Q', 'Annual Report', etc.",
    )
    company_name: Optional[str] = Field(
        default=None, description="Name of the company the document is about"
    )
    year: Optional[str] = Field(
        default=None, description="Year or fiscal period of the document"
    )


class DocumentSearchResult(BaseModel):
    """Model for document search results from Gemini."""

    company_identified: str = Field(
        description="The company name or ticker identified from the user query"
    )
    documents_found: list[FinancialDocument] = Field(
        description="List of financial documents found, prioritized by relevance and source type"
    )
    search_summary: str = Field(
        description="Brief summary of what was searched and found"
    )


class CompanyMetrics(BaseModel):
    """Structured financial metrics for a company from yfinance."""

    ticker: str = Field(description="Stock ticker symbol")
    company_name: str = Field(description="Full company name")
    market_cap: Optional[float] = Field(default=None, description="Market cap in USD")
    pe_ratio: Optional[float] = Field(default=None, description="Trailing P/E ratio")
    forward_pe: Optional[float] = Field(default=None, description="Forward P/E ratio")
    eps: Optional[float] = Field(default=None, description="Earnings per share (TTM)")
    revenue_ttm: Optional[float] = Field(
        default=None, description="Total revenue (TTM) in USD"
    )
    revenue_growth: Optional[float] = Field(
        default=None, description="Revenue growth rate (decimal)"
    )
    profit_margin: Optional[float] = Field(
        default=None, description="Profit margin (decimal)"
    )
    debt_to_equity: Optional[float] = Field(default=None, description="Debt to equity")
    current_ratio: Optional[float] = Field(default=None, description="Current ratio")
    dividend_yield: Optional[float] = Field(
        default=None, description="Dividend yield (decimal)"
    )
    fifty_two_week_high: Optional[float] = Field(
        default=None, description="52-week high price"
    )
    fifty_two_week_low: Optional[float] = Field(
        default=None, description="52-week low price"
    )
    current_price: Optional[float] = Field(
        default=None, description="Current stock price"
    )
    data_timestamp: datetime = Field(description="When data was fetched")


class HistoricalData(BaseModel):
    """Time series price/volume data for charting."""

    ticker: str = Field(description="Stock ticker symbol")
    dates: List[datetime] = Field(description="List of dates")
    close_prices: List[float] = Field(description="Closing prices")
    volumes: List[int] = Field(description="Trading volumes")


