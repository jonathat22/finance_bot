"""Financial data provider using yfinance API."""

import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

import yfinance as yf

from utils.models import CompanyMetrics, HistoricalData

logger = logging.getLogger(__name__)


class TickerError(Enum):
    """Error types for ticker validation."""

    INVALID_FORMAT = "invalid_format"
    NOT_FOUND = "not_found"
    NO_DATA = "no_data"
    API_ERROR = "api_error"


@dataclass
class TickerValidationResult:
    """Result of ticker validation."""

    valid: bool
    ticker: Optional[str] = None
    error: Optional[TickerError] = None
    message: Optional[str] = None


# Common company name to ticker mappings
COMPANY_TICKER_MAP = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "amazon": "AMZN",
    "meta": "META",
    "facebook": "META",
    "tesla": "TSLA",
    "nvidia": "NVDA",
    "netflix": "NFLX",
    "berkshire hathaway": "BRK-B",
    "jpmorgan": "JPM",
    "jp morgan": "JPM",
    "johnson & johnson": "JNJ",
    "visa": "V",
    "walmart": "WMT",
    "mastercard": "MA",
    "procter & gamble": "PG",
    "unitedhealth": "UNH",
    "home depot": "HD",
    "bank of america": "BAC",
    "chevron": "CVX",
    "exxon": "XOM",
    "exxon mobil": "XOM",
    "coca-cola": "KO",
    "pepsi": "PEP",
    "pepsico": "PEP",
    "disney": "DIS",
    "walt disney": "DIS",
    "intel": "INTC",
    "amd": "AMD",
    "salesforce": "CRM",
    "adobe": "ADBE",
    "oracle": "ORCL",
    "ibm": "IBM",
    "cisco": "CSCO",
    "paypal": "PYPL",
    "boeing": "BA",
    "mcdonald's": "MCD",
    "mcdonalds": "MCD",
    "nike": "NKE",
    "starbucks": "SBUX",
}


class FinancialDataProvider:
    """Fetches structured financial data from Yahoo Finance."""

    def __init__(self, cache_ttl_seconds: int = 300, verbose: bool = False):
        """Initialize provider with optional caching.

        Args:
            cache_ttl_seconds: Cache time-to-live in seconds (default 5 min)
            verbose: Enable verbose logging
        """
        self._cache: Dict[str, Tuple[CompanyMetrics, float]] = {}
        self._history_cache: Dict[str, Tuple[HistoricalData, float]] = {}
        self._cache_ttl = cache_ttl_seconds
        self._verbose = verbose

    def _log(self, message: str) -> None:
        """Log message if verbose mode is enabled."""
        if self._verbose:
            logger.info(message)

    def _is_cache_valid(self, cache_time: float) -> bool:
        """Check if cached data is still valid."""
        return (time.time() - cache_time) < self._cache_ttl

    def _is_valid_ticker_format(self, ticker: str) -> bool:
        """Check if ticker has valid format (1-5 letters, may include -, .)."""
        pattern = r"^[A-Za-z]{1,5}(-[A-Za-z])?(\.[A-Za-z]{1,2})?$"
        return bool(re.match(pattern, ticker.upper()))

    def validate_ticker(self, ticker: str) -> TickerValidationResult:
        """Validate a ticker symbol and check if data exists.

        Args:
            ticker: Stock ticker symbol

        Returns:
            TickerValidationResult with validation status and any errors
        """
        ticker = ticker.upper().strip()

        # Check format
        if not self._is_valid_ticker_format(ticker):
            return TickerValidationResult(
                valid=False,
                ticker=ticker,
                error=TickerError.INVALID_FORMAT,
                message=f"'{ticker}' is not a valid ticker format. "
                "Tickers are 1-5 letters (e.g., AAPL, MSFT, BRK-B).",
            )

        # Try to fetch basic info
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            if not info or info.get("regularMarketPrice") is None:
                # No price data = ticker doesn't exist or is delisted
                return TickerValidationResult(
                    valid=False,
                    ticker=ticker,
                    error=TickerError.NOT_FOUND,
                    message=f"'{ticker}' not found. Check the symbol.",
                )

            return TickerValidationResult(valid=True, ticker=ticker)

        except Exception as e:
            self._log(f"Validation error for {ticker}: {e}")
            return TickerValidationResult(
                valid=False,
                ticker=ticker,
                error=TickerError.API_ERROR,
                message=f"Could not verify '{ticker}'. Please try again.",
            )

    def resolve_company_to_ticker(
        self, company_name: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Resolve company name to ticker symbol.

        Args:
            company_name: Company name or ticker

        Returns:
            Tuple of (ticker_symbol, error_message). Error is None on success.
        """
        name_lower = company_name.lower().strip()

        # Check common mappings first
        if name_lower in COMPANY_TICKER_MAP:
            return COMPANY_TICKER_MAP[name_lower], None

        # Check if it's already a ticker (all caps, short)
        if company_name.isupper() and len(company_name) <= 5:
            # Validate it exists
            validation = self.validate_ticker(company_name)
            if validation.valid:
                return company_name, None
            return None, validation.message

        # Try yfinance search as fallback
        try:
            ticker_obj = yf.Ticker(company_name.upper())
            info = ticker_obj.info
            if info and info.get("symbol"):
                return info["symbol"], None
        except Exception as e:
            self._log(f"Could not resolve '{company_name}': {e}")

        return (
            None,
            f"Could not find ticker for '{company_name}'. "
            "Try using the symbol directly (e.g., AAPL).",
        )

    def get_company_metrics(
        self, ticker: str
    ) -> Tuple[Optional[CompanyMetrics], Optional[str]]:
        """Get current metrics for a company.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Tuple of (CompanyMetrics, error_message). Error is None on success.
        """
        ticker = ticker.upper()

        # Check cache
        if ticker in self._cache:
            cached_data, cache_time = self._cache[ticker]
            if self._is_cache_valid(cache_time):
                self._log(f"Cache hit for {ticker}")
                return cached_data, None

        self._log(f"Fetching metrics for {ticker}")

        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            if not info or "symbol" not in info:
                self._log(f"No data found for {ticker}")
                return (
                    None,
                    f"No data for '{ticker}'. Symbol may be invalid or delisted.",
                )

            metrics = CompanyMetrics(
                ticker=ticker,
                company_name=info.get("longName", info.get("shortName", ticker)),
                market_cap=info.get("marketCap"),
                pe_ratio=info.get("trailingPE"),
                forward_pe=info.get("forwardPE"),
                eps=info.get("trailingEps"),
                revenue_ttm=info.get("totalRevenue"),
                revenue_growth=info.get("revenueGrowth"),
                profit_margin=info.get("profitMargins"),
                debt_to_equity=info.get("debtToEquity"),
                current_ratio=info.get("currentRatio"),
                dividend_yield=info.get("dividendYield"),
                fifty_two_week_high=info.get("fiftyTwoWeekHigh"),
                fifty_two_week_low=info.get("fiftyTwoWeekLow"),
                current_price=info.get("currentPrice")
                or info.get("regularMarketPrice"),
                data_timestamp=datetime.now(),
            )

            # Cache the result
            self._cache[ticker] = (metrics, time.time())
            return metrics, None

        except Exception as e:
            logger.error(f"Error fetching metrics for {ticker}: {e}")
            return None, f"Error fetching data for '{ticker}': {str(e)}"

    def get_historical_data(
        self, ticker: str, period: str = "1y"
    ) -> Optional[HistoricalData]:
        """Get historical price data.

        Args:
            ticker: Stock ticker symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)

        Returns:
            HistoricalData or None if fetch failed
        """
        ticker = ticker.upper()
        cache_key = f"{ticker}_{period}"

        # Check cache
        if cache_key in self._history_cache:
            cached_data, cache_time = self._history_cache[cache_key]
            if self._is_cache_valid(cache_time):
                self._log(f"Cache hit for {cache_key}")
                return cached_data

        self._log(f"Fetching historical data for {ticker} ({period})")

        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)

            if hist.empty:
                self._log(f"No historical data for {ticker}")
                return None

            data = HistoricalData(
                ticker=ticker,
                dates=hist.index.tolist(),
                close_prices=hist["Close"].tolist(),
                volumes=hist["Volume"].astype(int).tolist(),
            )

            # Cache the result
            self._history_cache[cache_key] = (data, time.time())
            return data

        except Exception as e:
            logger.error(f"Error fetching history for {ticker}: {e}")
            return None

    def get_multiple_companies(
        self, tickers: List[str]
    ) -> Tuple[Dict[str, CompanyMetrics], Dict[str, str]]:
        """Batch fetch metrics for multiple companies.

        Args:
            tickers: List of ticker symbols

        Returns:
            Tuple of (success_dict, errors_dict) where:
            - success_dict maps ticker to CompanyMetrics
            - errors_dict maps ticker to error message
        """
        successes: Dict[str, CompanyMetrics] = {}
        errors: Dict[str, str] = {}

        for ticker in tickers:
            metrics, error = self.get_company_metrics(ticker)
            if metrics:
                successes[ticker] = metrics
            else:
                errors[ticker] = error or f"Unknown error fetching {ticker}"

        return successes, errors

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._cache.clear()
        self._history_cache.clear()
        self._log("Cache cleared")
