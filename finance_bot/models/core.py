"""Core data models migrated from utils/models.py."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    """Types of sources for financial information."""

    SEC_FILING = "SEC Filing"
    COMPANY_IR = "Company IR"
    NEWS = "Financial News"
    RESEARCH = "Research Report"
    GOVERNMENT = "Government"
    OTHER = "Other"


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


class HistoricalMarginData(BaseModel):
    """Historical margin data for competitive advantage analysis."""

    ticker: str = Field(description="Stock ticker symbol")
    company_name: str = Field(description="Full company name")
    years: List[int] = Field(description="Fiscal years (e.g., [2024, 2023, 2022...])")
    gross_profit_margins: List[Optional[float]] = Field(
        description="Gross profit margin by year (decimal, e.g., 0.45 = 45%)"
    )
    operating_expense_ratios: List[Optional[float]] = Field(
        description="Operating expense ratio by year (decimal)"
    )
    net_margins: List[Optional[float]] = Field(
        description="Net profit margin by year (decimal)"
    )
    competitive_advantage_gaps: List[Optional[float]] = Field(
        description="Gross margin minus op expense ratio by year"
    )
    data_timestamp: datetime = Field(description="When data was fetched")

    @property
    def has_consistent_advantage(self) -> bool:
        """Check if gross margin consistently exceeds operating expense ratio."""
        gaps = [g for g in self.competitive_advantage_gaps if g is not None]
        return len(gaps) >= 3 and all(g > 0 for g in gaps)

    @property
    def average_advantage_gap(self) -> Optional[float]:
        """Calculate average competitive advantage gap across available years."""
        gaps = [g for g in self.competitive_advantage_gaps if g is not None]
        return sum(gaps) / len(gaps) if gaps else None


class HistoricalPEData(BaseModel):
    """Historical trailing P/E ratio data for a company."""

    ticker: str = Field(description="Stock ticker symbol")
    company_name: str = Field(description="Full company name")
    years: List[int] = Field(
        description="Fiscal years (e.g., [2021, 2022, 2023, 2024])"
    )
    pe_ratios: List[Optional[float]] = Field(
        description="Trailing P/E ratio by year (None if EPS <= 0 or unavailable)"
    )
    data_timestamp: datetime = Field(description="When data was fetched")
