"""Tests for VisualizationAgent."""

import pytest

from finance_bot.visualize import (
    comparison_table,
    eps_chart,
    margin_chart,
    metrics_table,
    valuation_table,
    _fmt_large,
    _fmt_price,
    _fmt_pct,
    _fmt_multiple,
    _fmt_ratio,
)


@pytest.fixture
def sample_data():
    return {
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "current_price": 187.44,
        "metrics": {
            "current_price": 187.44,
            "market_cap": 2890000000000,
            "pe_ratio": 29.3,
            "forward_pe": 28.1,
            "eps": 6.42,
            "revenue_ttm": 383300000000,
            "profit_margin": 0.263,
            "revenue_growth": 0.05,
            "debt_to_equity": 173.0,
            "current_ratio": 1.07,
            "dividend_yield": 0.005,
            "fifty_two_week_low": 164.08,
            "fifty_two_week_high": 199.62,
        },
        "margins": {
            "years": [2021, 2022, 2023, 2024],
            "gross_profit_margins": [0.418, 0.433, 0.441, 0.462],
        },
        "eps_history": {
            "years": [2020, 2021, 2022, 2023, 2024],
            "eps": [3.28, 5.61, 6.11, 6.13, 6.42],
        },
    }


@pytest.fixture
def sample_scorecard():
    return {
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "current_price": 187.44,
        "intrinsic_values": [
            {
                "method": "dcf",
                "intrinsic_value_per_share": 158.20,
                "current_price": 187.44,
                "margin_of_safety_pct": -15.6,
            },
            {
                "method": "graham_number",
                "intrinsic_value_per_share": 25.16,
                "current_price": 187.44,
                "margin_of_safety_pct": -86.5,
            },
        ],
        "moat": {"moat_rating": "wide"},
        "earnings": {"is_predictable": True},
        "debt": {"rating": "aggressive"},
        "overall_rating": "hold",
    }


class TestFormatHelpers:
    def test_fmt_price(self):
        assert _fmt_price(187.44) == "$187.44"
        assert _fmt_price(None) == "N/A"

    def test_fmt_large(self):
        assert _fmt_large(2890000000000) == "$2.89T"
        assert _fmt_large(383300000000) == "$383.3B"
        assert _fmt_large(5000000) == "$5.0M"
        assert _fmt_large(None) == "N/A"

    def test_fmt_pct(self):
        assert _fmt_pct(0.263) == "26.3%"
        assert _fmt_pct(None) == "N/A"

    def test_fmt_multiple(self):
        assert _fmt_multiple(29.3) == "29.3x"
        assert _fmt_multiple(None) == "N/A"

    def test_fmt_ratio_converts_percentage(self):
        assert _fmt_ratio(173.0) == "1.73x"
        assert _fmt_ratio(0.5) == "0.50x"


class TestMetricsTable:
    def test_renders_output(self, sample_data):
        result = metrics_table(sample_data)
        assert isinstance(result, str)
        assert len(result) > 0
        assert "Price" in result or "$187.44" in result

    def test_handles_missing_metrics(self):
        result = metrics_table({"metrics": {}})
        assert isinstance(result, str)
        assert "N/A" in result


class TestValuationTable:
    def test_renders_output(self, sample_scorecard):
        result = valuation_table(sample_scorecard)
        assert isinstance(result, str)
        assert "Dcf" in result or "dcf" in result.lower()

    def test_empty_intrinsic_values(self):
        result = valuation_table({"intrinsic_values": []})
        assert isinstance(result, str)


class TestComparisonTable:
    def test_renders_output(self, sample_scorecard):
        sc2 = dict(sample_scorecard)
        sc2["ticker"] = "MSFT"
        result = comparison_table([sample_scorecard, sc2])
        assert isinstance(result, str)
        assert "AAPL" in result
        assert "MSFT" in result


class TestMarginChart:
    def test_renders_chart(self, sample_data):
        result = margin_chart(sample_data)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_no_data(self):
        result = margin_chart({"margins": None})
        assert "No margin data" in result


class TestEpsChart:
    def test_renders_chart(self, sample_data):
        result = eps_chart(sample_data)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_no_data(self):
        result = eps_chart({"eps_history": {"years": [], "eps": []}})
        assert "No EPS data" in result
