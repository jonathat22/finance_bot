"""Tests for DataAgent."""

from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from finance_bot.data import DataAgent


@pytest.fixture
def mock_income_stmt():
    """Mock income statement DataFrame."""
    dates = [pd.Timestamp("2024-09-30"), pd.Timestamp("2023-09-30")]
    data = {
        dates[0]: {
            "Total Revenue": 383300000000,
            "Gross Profit": 176400000000,
            "Operating Income": 118700000000,
            "EBIT": 118700000000,
            "Net Income": 100800000000,
            "Interest Expense": -3900000000,
            "Diluted EPS": 6.42,
            "Operating Expense": 56000000000,
        },
        dates[1]: {
            "Total Revenue": 383300000000,
            "Gross Profit": 169100000000,
            "Operating Income": 114300000000,
            "EBIT": 114300000000,
            "Net Income": 97000000000,
            "Interest Expense": -3700000000,
            "Diluted EPS": 6.13,
            "Operating Expense": 54000000000,
        },
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_cash_flow():
    """Mock cash flow statement DataFrame."""
    dates = [
        pd.Timestamp("2024-09-30"),
        pd.Timestamp("2023-09-30"),
        pd.Timestamp("2022-09-30"),
    ]
    data = {
        dates[0]: {
            "Operating Cash Flow": 118000000000,
            "Capital Expenditure": -9900000000,
            "Depreciation And Amortization": 11400000000,
        },
        dates[1]: {
            "Operating Cash Flow": 110500000000,
            "Capital Expenditure": -11000000000,
            "Depreciation And Amortization": 11500000000,
        },
        dates[2]: {
            "Operating Cash Flow": 122000000000,
            "Capital Expenditure": -10700000000,
            "Depreciation And Amortization": 11100000000,
        },
    }
    return pd.DataFrame(data)


@pytest.fixture
def data_agent():
    return DataAgent(verbose=False)


class TestDataAgent:
    @patch("finance_bot.data.yf.Ticker")
    def test_get_eps_history(self, mock_ticker_cls, data_agent, mock_income_stmt):
        mock_stock = Mock()
        mock_stock.income_stmt = mock_income_stmt
        mock_stock.info = {"sharesOutstanding": 15400000000}

        result = data_agent._get_eps_history(mock_stock)
        assert len(result["years"]) == 2
        assert result["years"] == [2023, 2024]  # sorted chronologically
        assert result["eps"][1] == 6.42

    @patch("finance_bot.data.yf.Ticker")
    def test_get_free_cash_flow(self, mock_ticker_cls, data_agent, mock_cash_flow):
        mock_stock = Mock()
        mock_stock.cash_flow = mock_cash_flow

        result = data_agent._get_free_cash_flow(mock_stock)
        assert len(result["years"]) == 3
        # FCF = operating CF + capex (capex is negative)
        # 2024: 118B + (-9.9B) = 108.1B
        assert result["fcf"][-1] == 108100000000.0

    def test_get_free_cash_flow_empty(self, data_agent):
        mock_stock = Mock()
        mock_stock.cash_flow = pd.DataFrame()

        result = data_agent._get_free_cash_flow(mock_stock)
        assert result == {"years": [], "fcf": []}

    @patch("finance_bot.data.yf.Ticker")
    def test_get_interest_coverage(
        self, mock_ticker_cls, data_agent, mock_income_stmt
    ):
        mock_stock = Mock()
        mock_stock.income_stmt = mock_income_stmt

        result = data_agent._get_interest_coverage(mock_stock)
        # EBIT: 118.7B, Interest: 3.9B => ~30.44
        assert result is not None
        assert result == pytest.approx(30.44, rel=0.01)

    @patch("finance_bot.data.yf.Ticker")
    def test_get_owner_earnings_components(
        self, mock_ticker_cls, data_agent, mock_income_stmt, mock_cash_flow
    ):
        mock_stock = Mock()
        mock_stock.income_stmt = mock_income_stmt
        mock_stock.cash_flow = mock_cash_flow

        result = data_agent._get_owner_earnings_components(mock_stock)
        assert result["net_income"] == 100800000000
        assert result["depreciation_amortization"] == 11400000000
        # Avg capex: (9.9 + 11 + 10.7) / 3 = 10.533B
        assert result["avg_capex_3yr"] == pytest.approx(10533333333.33, rel=0.01)
        assert result["owner_earnings"] is not None

    def test_safe_float_none(self, data_agent):
        assert data_agent._safe_float(None) is None

    def test_safe_float_valid(self, data_agent):
        assert data_agent._safe_float(42.5) == 42.5

    def test_safe_float_string(self, data_agent):
        assert data_agent._safe_float("not_a_number") is None

    @patch("finance_bot.data.yf.Ticker")
    @patch.object(DataAgent, "_get_eps_history")
    @patch.object(DataAgent, "_get_free_cash_flow")
    @patch.object(DataAgent, "_get_interest_coverage")
    @patch.object(DataAgent, "_get_owner_earnings_components")
    def test_get_full_data_success(
        self,
        mock_owner,
        mock_interest,
        mock_fcf,
        mock_eps,
        mock_ticker_cls,
        data_agent,
    ):
        mock_owner.return_value = {"owner_earnings": 100000}
        mock_interest.return_value = 10.0
        mock_fcf.return_value = {"years": [2024], "fcf": [50000]}
        mock_eps.return_value = {"years": [2024], "eps": [5.0]}

        mock_stock = Mock()
        mock_stock.info = {
            "symbol": "AAPL",
            "longName": "Apple Inc.",
            "marketCap": 2890000000000,
            "currentPrice": 187.44,
            "regularMarketPrice": 187.44,
            "bookValue": 4.38,
            "sharesOutstanding": 15400000000,
        }
        mock_ticker_cls.return_value = mock_stock

        # Also need to mock the provider's get_company_metrics
        with patch.object(
            data_agent._provider,
            "get_company_metrics",
            return_value=(
                MagicMock(
                    company_name="Apple Inc.",
                    current_price=187.44,
                    model_dump_json=lambda: '{"ticker":"AAPL","company_name":"Apple Inc.","current_price":187.44}',
                ),
                None,
            ),
        ), patch.object(
            data_agent._provider,
            "get_historical_margins",
            return_value=(None, None),
        ):
            result, err = data_agent.get_full_data("AAPL")

        assert err is None
        assert result is not None
        assert result["ticker"] == "AAPL"
        assert result["book_value_per_share"] == 4.38

    def test_get_full_data_invalid_ticker(self, data_agent):
        with patch.object(
            data_agent._provider,
            "get_company_metrics",
            return_value=(None, "Not found"),
        ):
            result, err = data_agent.get_full_data("INVALID")
        assert result is None
        assert err == "Not found"
