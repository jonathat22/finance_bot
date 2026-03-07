"""DataAgent: Fetch and compute financial data for value investing analysis.

Wraps FinancialDataProvider and adds new computations:
- 10-year EPS history
- Free cash flow (operating CF - capex)
- Book value per share
- Interest coverage (EBIT / interest expense)
- Owner Earnings components
- Shares outstanding
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

from finance_bot.utils.financial_data import FinancialDataProvider

logger = logging.getLogger(__name__)


class DataAgent:
    """Fetches comprehensive financial data for value investing analysis."""

    def __init__(self, verbose: bool = False):
        self._provider = FinancialDataProvider(verbose=verbose)
        self._verbose = verbose

    def _log(self, msg: str) -> None:
        if self._verbose:
            logger.info(msg)

    def _safe_float(self, val) -> Optional[float]:
        """Convert a value to float, returning None on failure."""
        if val is None:
            return None
        try:
            if hasattr(val, "isna") and val.isna():
                return None
            f = float(val)
            if pd.isna(f):
                return None
            return f
        except (ValueError, TypeError):
            return None

    def get_full_data(self, ticker: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Fetch all data needed for value investing analysis.

        Returns:
            Tuple of (data_dict, error_message). Error is None on success.
        """
        ticker = ticker.upper()

        # Get basic metrics
        metrics, err = self._provider.get_company_metrics(ticker)
        if err or not metrics:
            return None, err or f"Could not fetch data for {ticker}"

        stock = yf.Ticker(ticker)
        info = stock.info

        # Build result dict
        result: Dict[str, Any] = {
            "ticker": ticker,
            "company_name": metrics.company_name,
            "current_price": metrics.current_price,
            "metrics": json.loads(metrics.model_dump_json()),
        }

        # EPS history
        eps_data = self._get_eps_history(stock)
        result["eps_history"] = eps_data

        # Free cash flow
        fcf_data = self._get_free_cash_flow(stock)
        result["free_cash_flow"] = fcf_data

        # Book value per share
        result["book_value_per_share"] = self._safe_float(info.get("bookValue"))

        # Shares outstanding
        result["shares_outstanding"] = self._safe_float(info.get("sharesOutstanding"))

        # Interest coverage
        result["interest_coverage"] = self._get_interest_coverage(stock)

        # Owner earnings components
        result["owner_earnings"] = self._get_owner_earnings_components(stock)

        # Margin data
        margin_data, _ = self._provider.get_historical_margins(ticker)
        if margin_data:
            result["margins"] = json.loads(margin_data.model_dump_json())
        else:
            result["margins"] = None

        result["data_timestamp"] = datetime.now().isoformat()
        return result, None

    def _get_eps_history(self, stock) -> Dict[str, Any]:
        """Extract up to 10 years of EPS from income statement."""
        try:
            income_stmt = stock.income_stmt
            if income_stmt is None or income_stmt.empty:
                return {"years": [], "eps": []}

            info = stock.info
            shares = self._safe_float(info.get("sharesOutstanding"))

            years: List[int] = []
            eps_values: List[Optional[float]] = []

            for col in income_stmt.columns:
                year = col.year if hasattr(col, "year") else int(str(col)[:4])

                # Try Diluted EPS directly first
                eps = self._get_stmt_value(income_stmt, col, ["Diluted EPS", "Basic EPS"])

                # Fallback: compute from net income / shares
                if eps is None and shares and shares > 0:
                    net_income = self._get_stmt_value(
                        income_stmt, col, ["Net Income", "Net Income Common Stockholders"]
                    )
                    if net_income is not None:
                        eps = net_income / shares

                years.append(year)
                eps_values.append(round(eps, 4) if eps is not None else None)

            # Sort chronologically
            paired = sorted(zip(years, eps_values), key=lambda x: x[0])
            return {
                "years": [p[0] for p in paired],
                "eps": [p[1] for p in paired],
            }
        except Exception as e:
            self._log(f"EPS history error: {e}")
            return {"years": [], "eps": []}

    def _get_free_cash_flow(self, stock) -> Dict[str, Any]:
        """Compute FCF = Operating CF - CapEx from cash flow statement."""
        try:
            cf = stock.cash_flow
            if cf is None or cf.empty:
                return {"years": [], "fcf": []}

            years: List[int] = []
            fcf_values: List[Optional[float]] = []

            for col in cf.columns:
                year = col.year if hasattr(col, "year") else int(str(col)[:4])

                operating_cf = self._get_stmt_value(
                    cf,
                    col,
                    ["Operating Cash Flow", "Total Cash From Operating Activities"],
                )
                capex = self._get_stmt_value(
                    cf, col, ["Capital Expenditure", "Capital Expenditures"]
                )

                if operating_cf is not None and capex is not None:
                    # capex is typically negative in yfinance
                    fcf = operating_cf + capex if capex < 0 else operating_cf - capex
                    fcf_values.append(round(fcf, 2))
                elif operating_cf is not None:
                    fcf_values.append(round(operating_cf, 2))
                else:
                    fcf_values.append(None)

                years.append(year)

            paired = sorted(zip(years, fcf_values), key=lambda x: x[0])
            return {
                "years": [p[0] for p in paired],
                "fcf": [p[1] for p in paired],
            }
        except Exception as e:
            self._log(f"FCF error: {e}")
            return {"years": [], "fcf": []}

    def _get_interest_coverage(self, stock) -> Optional[float]:
        """Compute interest coverage = EBIT / Interest Expense (most recent year)."""
        try:
            income_stmt = stock.income_stmt
            if income_stmt is None or income_stmt.empty:
                return None

            col = income_stmt.columns[0]  # Most recent year

            ebit = self._get_stmt_value(income_stmt, col, ["EBIT", "Operating Income"])
            interest = self._get_stmt_value(
                income_stmt, col, ["Interest Expense", "Interest Expense Non Operating"]
            )

            if ebit is not None and interest is not None and interest != 0:
                # Interest expense is often negative (an expense)
                interest_abs = abs(interest)
                return round(ebit / interest_abs, 2) if interest_abs > 0 else None

            return None
        except Exception as e:
            self._log(f"Interest coverage error: {e}")
            return None

    def _get_owner_earnings_components(self, stock) -> Dict[str, Any]:
        """Compute Owner Earnings = Net Income + D&A - Avg 3yr CapEx."""
        try:
            income_stmt = stock.income_stmt
            cf = stock.cash_flow

            if income_stmt is None or income_stmt.empty or cf is None or cf.empty:
                return {}

            col = income_stmt.columns[0]  # Most recent year

            net_income = self._get_stmt_value(
                income_stmt, col, ["Net Income", "Net Income Common Stockholders"]
            )
            depreciation = self._get_stmt_value(
                cf,
                col,
                [
                    "Depreciation And Amortization",
                    "Depreciation & Amortization",
                    "Depreciation",
                ],
            )

            # Get up to 3 years of capex for averaging
            capex_values = []
            for c in cf.columns[:3]:
                capex = self._get_stmt_value(
                    cf, c, ["Capital Expenditure", "Capital Expenditures"]
                )
                if capex is not None:
                    capex_values.append(abs(capex))

            avg_capex = sum(capex_values) / len(capex_values) if capex_values else None

            result = {
                "net_income": self._safe_float(net_income),
                "depreciation_amortization": self._safe_float(depreciation),
                "avg_capex_3yr": round(avg_capex, 2) if avg_capex else None,
            }

            if net_income is not None and depreciation is not None and avg_capex is not None:
                owner_earnings = net_income + depreciation - avg_capex
                result["owner_earnings"] = round(owner_earnings, 2)
            else:
                result["owner_earnings"] = None

            return result
        except Exception as e:
            self._log(f"Owner earnings error: {e}")
            return {}

    def _get_stmt_value(self, stmt, col, row_names: List[str]) -> Optional[float]:
        """Extract a value from a financial statement."""
        for name in row_names:
            if name in stmt.index:
                val = stmt.loc[name, col]
                if val is not None and not (hasattr(val, "isna") and val.isna()):
                    f = float(val)
                    if not pd.isna(f):
                        return f
        return None


def _sanitize_for_json(obj):
    """Replace float NaN/Inf with None for valid JSON output."""
    import math

    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def main():
    parser = argparse.ArgumentParser(description="Fetch financial data for analysis")
    parser.add_argument("--ticker", required=True, help="Stock ticker symbol")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    agent = DataAgent(verbose=args.verbose)
    data, error = agent.get_full_data(args.ticker)

    if error:
        print(json.dumps({"error": error}), file=sys.stderr)
        sys.exit(1)

    print(json.dumps(_sanitize_for_json(data), default=str))


if __name__ == "__main__":
    main()
