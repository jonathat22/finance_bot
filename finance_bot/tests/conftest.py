"""Pytest configuration and shared fixtures for finance_bot."""

from datetime import datetime
from unittest.mock import Mock

import pytest


@pytest.fixture
def sample_company_metrics_data():
    """Sample company metrics dict."""
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


@pytest.fixture
def sample_eps_history():
    """10-year EPS history for testing."""
    return {
        "years": [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
        "eps": [2.31, 2.08, 2.30, 2.98, 2.97, 3.28, 5.61, 6.11, 6.13, 6.42],
    }


@pytest.fixture
def mock_yfinance_ticker():
    """Mock yfinance Ticker object with realistic data."""
    mock = Mock()
    mock.info = {
        "symbol": "AAPL",
        "longName": "Apple Inc.",
        "marketCap": 2890000000000,
        "trailingPE": 29.3,
        "forwardPE": 28.1,
        "trailingEps": 6.42,
        "totalRevenue": 383300000000,
        "revenueGrowth": 0.05,
        "profitMargins": 0.263,
        "debtToEquity": 173.0,
        "currentRatio": 1.07,
        "dividendYield": 0.005,
        "fiftyTwoWeekHigh": 199.62,
        "fiftyTwoWeekLow": 164.08,
        "currentPrice": 187.44,
        "regularMarketPrice": 187.44,
        "bookValue": 4.38,
        "sharesOutstanding": 15400000000,
    }
    return mock
