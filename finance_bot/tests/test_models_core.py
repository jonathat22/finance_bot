"""Tests for core data models."""

from datetime import datetime

import pytest

from finance_bot.models.core import (
    Citation,
    CompanyMetrics,
    HistoricalMarginData,
    ResearchAnswer,
    SourceType,
)


class TestSourceType:
    def test_enum_values(self):
        assert SourceType.SEC_FILING == "SEC Filing"
        assert SourceType.NEWS == "Financial News"
        assert SourceType.OTHER == "Other"


class TestCitation:
    def test_basic_citation(self):
        c = Citation(title="Test", url="https://example.com")
        assert c.title == "Test"
        assert c.snippet is None
        assert c.source_type is None

    def test_full_citation(self):
        c = Citation(
            title="SEC Filing",
            url="https://sec.gov/filing",
            snippet="Revenue was $100B",
            source_type="SEC Filing",
        )
        assert c.source_type == "SEC Filing"


class TestResearchAnswer:
    def test_empty_citations(self):
        ra = ResearchAnswer(answer="Test answer")
        assert ra.citations == []
        assert ra.confidence is None

    def test_with_citations(self):
        ra = ResearchAnswer(
            answer="Apple is worth $3T",
            citations=[Citation(title="Source", url="https://example.com")],
            confidence="high",
        )
        assert len(ra.citations) == 1


class TestCompanyMetrics:
    def test_required_fields(self):
        m = CompanyMetrics(
            ticker="AAPL",
            company_name="Apple Inc.",
            data_timestamp=datetime.now(),
        )
        assert m.ticker == "AAPL"
        assert m.market_cap is None

    def test_all_fields(self, sample_company_metrics_data):
        m = CompanyMetrics(**sample_company_metrics_data)
        assert m.current_price == 187.44
        assert m.pe_ratio == 29.3


class TestHistoricalMarginData:
    def test_consistent_advantage(self):
        data = HistoricalMarginData(
            ticker="AAPL",
            company_name="Apple Inc.",
            years=[2022, 2023, 2024],
            gross_profit_margins=[0.43, 0.44, 0.46],
            operating_expense_ratios=[0.30, 0.29, 0.28],
            net_margins=[0.25, 0.26, 0.27],
            competitive_advantage_gaps=[0.13, 0.15, 0.18],
            data_timestamp=datetime.now(),
        )
        assert data.has_consistent_advantage is True
        assert data.average_advantage_gap == pytest.approx(0.1533, rel=0.01)

    def test_inconsistent_advantage(self):
        data = HistoricalMarginData(
            ticker="XYZ",
            company_name="XYZ Corp",
            years=[2022, 2023, 2024],
            gross_profit_margins=[0.20, 0.15, 0.18],
            operating_expense_ratios=[0.22, 0.20, 0.19],
            net_margins=[0.05, -0.02, 0.03],
            competitive_advantage_gaps=[-0.02, -0.05, -0.01],
            data_timestamp=datetime.now(),
        )
        assert data.has_consistent_advantage is False


@pytest.fixture
def sample_company_metrics_data():
    return {
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "market_cap": 2890000000000,
        "pe_ratio": 29.3,
        "forward_pe": 28.1,
        "eps": 6.42,
        "revenue_ttm": 383300000000,
        "revenue_growth": 0.05,
        "profit_margin": 0.263,
        "debt_to_equity": 173.0,
        "current_ratio": 1.07,
        "dividend_yield": 0.005,
        "fifty_two_week_high": 199.62,
        "fifty_two_week_low": 164.08,
        "current_price": 187.44,
        "data_timestamp": datetime.now(),
    }
