"""Tests for the FinancialDataProvider."""

import time
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from utils.financial_data import COMPANY_TICKER_MAP, FinancialDataProvider
from utils.models import CompanyMetrics, HistoricalData


@pytest.fixture
def mock_yfinance_info():
    """Mock yfinance Ticker.info response."""
    return {
        "symbol": "AAPL",
        "longName": "Apple Inc.",
        "shortName": "Apple",
        "marketCap": 3000000000000,
        "trailingPE": 28.5,
        "forwardPE": 25.0,
        "trailingEps": 6.15,
        "totalRevenue": 383000000000,
        "revenueGrowth": 0.08,
        "profitMargins": 0.25,
        "debtToEquity": 150.0,
        "currentRatio": 1.1,
        "dividendYield": 0.005,
        "fiftyTwoWeekHigh": 200.0,
        "fiftyTwoWeekLow": 150.0,
        "currentPrice": 185.0,
    }


@pytest.fixture
def mock_yfinance_history():
    """Mock yfinance Ticker.history response."""
    dates = pd.date_range(start="2024-01-01", periods=5, freq="D")
    return pd.DataFrame(
        {
            "Open": [180.0, 181.0, 182.0, 183.0, 184.0],
            "High": [182.0, 183.0, 184.0, 185.0, 186.0],
            "Low": [179.0, 180.0, 181.0, 182.0, 183.0],
            "Close": [181.0, 182.0, 183.0, 184.0, 185.0],
            "Volume": [1000000, 1100000, 1200000, 1300000, 1400000],
        },
        index=dates,
    )


@pytest.fixture
def provider():
    """Create a FinancialDataProvider instance."""
    return FinancialDataProvider(cache_ttl_seconds=300, verbose=False)


class TestResolveCompanyToTicker:
    """Tests for company name to ticker resolution."""

    def test_resolve_known_company(self, provider):
        """Test resolving a known company name."""
        ticker, error = provider.resolve_company_to_ticker("Apple")
        assert ticker == "AAPL"
        assert error is None

        ticker, error = provider.resolve_company_to_ticker("google")
        assert ticker == "GOOGL"
        assert error is None

    @patch("utils.financial_data.yf.Ticker")
    def test_resolve_already_ticker(self, mock_ticker_class, provider):
        """Test that valid tickers pass through."""
        # Mock valid ticker response
        mock_ticker = MagicMock()
        mock_ticker.info = {"regularMarketPrice": 185.0}
        mock_ticker_class.return_value = mock_ticker

        ticker, error = provider.resolve_company_to_ticker("AAPL")
        assert ticker == "AAPL"
        assert error is None

    def test_resolve_case_insensitive(self, provider):
        """Test case insensitivity."""
        ticker, _ = provider.resolve_company_to_ticker("apple")
        assert ticker == "AAPL"

        ticker, _ = provider.resolve_company_to_ticker("Apple")
        assert ticker == "AAPL"

    def test_ticker_map_coverage(self):
        """Test that common companies are in the map."""
        expected = ["apple", "microsoft", "google", "amazon", "tesla", "nvidia"]
        for company in expected:
            assert company in COMPANY_TICKER_MAP


class TestGetCompanyMetrics:
    """Tests for fetching company metrics."""

    @patch("utils.financial_data.yf.Ticker")
    def test_get_metrics_success(self, mock_ticker_class, provider, mock_yfinance_info):
        """Test successful metrics fetch."""
        mock_ticker = MagicMock()
        mock_ticker.info = mock_yfinance_info
        mock_ticker_class.return_value = mock_ticker

        metrics, error = provider.get_company_metrics("AAPL")

        assert metrics is not None
        assert error is None
        assert isinstance(metrics, CompanyMetrics)
        assert metrics.ticker == "AAPL"
        assert metrics.company_name == "Apple Inc."
        assert metrics.market_cap == 3000000000000
        assert metrics.pe_ratio == 28.5
        assert metrics.eps == 6.15
        assert metrics.current_price == 185.0

    @patch("utils.financial_data.yf.Ticker")
    def test_get_metrics_invalid_ticker(self, mock_ticker_class, provider):
        """Test handling of invalid ticker."""
        mock_ticker = MagicMock()
        mock_ticker.info = {}
        mock_ticker_class.return_value = mock_ticker

        metrics, error = provider.get_company_metrics("INVALID123")

        assert metrics is None
        assert error is not None
        assert "INVALID123" in error

    @patch("utils.financial_data.yf.Ticker")
    def test_get_metrics_api_error(self, mock_ticker_class, provider):
        """Test handling of API errors."""
        mock_ticker_class.side_effect = Exception("API Error")

        metrics, error = provider.get_company_metrics("AAPL")

        assert metrics is None
        assert error is not None
        assert "API Error" in error

    @patch("utils.financial_data.yf.Ticker")
    def test_get_metrics_caching(self, mock_ticker_class, provider, mock_yfinance_info):
        """Test that results are cached."""
        mock_ticker = MagicMock()
        mock_ticker.info = mock_yfinance_info
        mock_ticker_class.return_value = mock_ticker

        # First call
        metrics1, _ = provider.get_company_metrics("AAPL")
        # Second call should use cache
        metrics2, _ = provider.get_company_metrics("AAPL")

        assert metrics1 == metrics2
        # Ticker should only be created once
        assert mock_ticker_class.call_count == 1

    @patch("utils.financial_data.yf.Ticker")
    def test_get_metrics_cache_expiry(self, mock_ticker_class, mock_yfinance_info):
        """Test cache expiration."""
        provider = FinancialDataProvider(cache_ttl_seconds=1)
        mock_ticker = MagicMock()
        mock_ticker.info = mock_yfinance_info
        mock_ticker_class.return_value = mock_ticker

        # First call
        provider.get_company_metrics("AAPL")
        assert mock_ticker_class.call_count == 1

        # Wait for cache to expire
        time.sleep(1.1)

        # Second call should fetch again
        provider.get_company_metrics("AAPL")
        assert mock_ticker_class.call_count == 2


class TestGetHistoricalData:
    """Tests for fetching historical data."""

    @patch("utils.financial_data.yf.Ticker")
    def test_get_history_success(
        self, mock_ticker_class, provider, mock_yfinance_history
    ):
        """Test successful historical data fetch."""
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_yfinance_history
        mock_ticker_class.return_value = mock_ticker

        data = provider.get_historical_data("AAPL", "1y")

        assert data is not None
        assert isinstance(data, HistoricalData)
        assert data.ticker == "AAPL"
        assert len(data.dates) == 5
        assert len(data.close_prices) == 5
        assert len(data.volumes) == 5
        assert data.close_prices[0] == 181.0

    @patch("utils.financial_data.yf.Ticker")
    def test_get_history_empty(self, mock_ticker_class, provider):
        """Test handling of empty history."""
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_ticker_class.return_value = mock_ticker

        data = provider.get_historical_data("INVALID")

        assert data is None

    @patch("utils.financial_data.yf.Ticker")
    def test_get_history_caching(
        self, mock_ticker_class, provider, mock_yfinance_history
    ):
        """Test that historical data is cached."""
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_yfinance_history
        mock_ticker_class.return_value = mock_ticker

        # First call
        data1 = provider.get_historical_data("AAPL", "1y")
        # Second call should use cache
        data2 = provider.get_historical_data("AAPL", "1y")

        assert data1 == data2
        # History should only be called once
        assert mock_ticker.history.call_count == 1


class TestGetMultipleCompanies:
    """Tests for batch fetching."""

    @patch("utils.financial_data.yf.Ticker")
    def test_get_multiple_success(
        self, mock_ticker_class, provider, mock_yfinance_info
    ):
        """Test fetching multiple companies."""
        mock_ticker = MagicMock()
        mock_ticker.info = mock_yfinance_info
        mock_ticker_class.return_value = mock_ticker

        successes, errors = provider.get_multiple_companies(["AAPL", "MSFT"])

        assert len(successes) == 2
        assert "AAPL" in successes
        assert "MSFT" in successes
        assert len(errors) == 0

    @patch("utils.financial_data.yf.Ticker")
    def test_get_multiple_partial_failure(self, mock_ticker_class, provider):
        """Test partial failure in batch fetch."""
        call_count = [0]

        def side_effect(ticker):
            call_count[0] += 1
            mock = MagicMock()
            if ticker == "INVALID":
                mock.info = {}
            else:
                mock.info = {"symbol": ticker, "longName": f"{ticker} Inc."}
            return mock

        mock_ticker_class.side_effect = side_effect

        successes, errors = provider.get_multiple_companies(["AAPL", "INVALID"])

        assert "AAPL" in successes
        assert successes["AAPL"] is not None
        assert "INVALID" in errors
        assert errors["INVALID"] is not None


class TestCacheManagement:
    """Tests for cache management."""

    @patch("utils.financial_data.yf.Ticker")
    def test_clear_cache(self, mock_ticker_class, provider, mock_yfinance_info):
        """Test cache clearing."""
        mock_ticker = MagicMock()
        mock_ticker.info = mock_yfinance_info
        mock_ticker_class.return_value = mock_ticker

        # Populate cache
        provider.get_company_metrics("AAPL")
        assert mock_ticker_class.call_count == 1

        # Clear and fetch again
        provider.clear_cache()
        provider.get_company_metrics("AAPL")
        assert mock_ticker_class.call_count == 2
