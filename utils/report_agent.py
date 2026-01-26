"""Report agent for generating comparative financial analysis reports."""

import json
import os
import re
from datetime import datetime
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig

from utils.chart_generator import ChartGenerator
from utils.financial_data import FinancialDataProvider
from utils.models import ChartSpec, Citation, CompanyMetrics, ComparisonReport
from utils.research_agent import ResearchAgent, exponential_backoff_retry

load_dotenv()

# System prompt for extracting companies from user queries
COMPANY_EXTRACTION_PROMPT = """
You are a financial query parser. Extract company names or ticker symbols.

Return ONLY valid JSON in this exact format:
{"companies": ["TICKER1", "TICKER2"]}

Rules:
- Return ticker symbols when possible (AAPL not Apple)
- Maximum 5 companies
- If a company name is given, convert to ticker (e.g., "Apple" -> "AAPL")
- If no companies found, return {"companies": []}

Examples:
- "Compare Apple and Google" -> {"companies": ["AAPL", "GOOGL"]}
- "Is Tesla overvalued?" -> {"companies": ["TSLA"]}
- "Microsoft vs Amazon financials" -> {"companies": ["MSFT", "AMZN"]}
- "Tech sector analysis" -> {"companies": []}
"""

# System prompt for determining relevant charts
CHART_SELECTION_PROMPT = """
You are a financial visualization expert. Select the most relevant charts.

Available chart types:
1. "valuation" - P/E vs growth scatter (valuation comparisons)
2. "profitability" - Profit margin and revenue bars (financial health)
3. "price_history" - Stock price trends over time (performance)
4. "growth" - Revenue growth comparison (growth analysis)

Return ONLY valid JSON:
{"charts": ["chart_type1", "chart_type2"]}

Select 1-3 most relevant charts. Default to ["valuation", "profitability"].
"""


class ReportAgent:
    """Orchestrates report generation with metrics, charts, and analysis."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        verbose: bool = False,
    ):
        """Initialize report agent.

        Args:
            model: Gemini model for company extraction
            verbose: Enable verbose logging
        """
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = model
        self.verbose = verbose

        self.data_provider = FinancialDataProvider(verbose=verbose)
        self.chart_generator = ChartGenerator()
        self.research_agent = ResearchAgent(verbose=verbose)

    def _log(self, message: str) -> None:
        """Log message if verbose mode enabled."""
        if self.verbose:
            print(f"[ReportAgent] {message}")

    def _extract_companies_from_query(self, query: str) -> List[str]:
        """Extract company tickers from user query using Gemini.

        Args:
            query: User's natural language query

        Returns:
            List of ticker symbols
        """
        self._log(f"Extracting companies from: {query}")

        def make_request():
            response = self.client.models.generate_content(
                model=self.model,
                config=GenerateContentConfig(
                    system_instruction=COMPANY_EXTRACTION_PROMPT,
                    temperature=0.0,
                ),
                contents=query,
            )
            return response.text

        response_text = exponential_backoff_retry(make_request)

        # Parse JSON response
        try:
            # Handle markdown code blocks
            json_match = re.search(
                r"```(?:json)?\s*(.*?)\s*```", response_text, re.DOTALL
            )
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response_text.strip()

            result = json.loads(json_str)
            companies = result.get("companies", [])
            self._log(f"Extracted companies: {companies}")
            return companies[:5]  # Max 5 companies
        except json.JSONDecodeError as e:
            self._log(f"JSON parse error: {e}, response: {response_text}")
            return []

    def _select_charts(self, query: str, companies: List[CompanyMetrics]) -> List[str]:
        """Select relevant chart types based on query.

        Args:
            query: User's query
            companies: List of company metrics

        Returns:
            List of chart type strings
        """
        self._log(f"Selecting charts for query: {query}")

        # Build context about available data
        metrics_summary = []
        for c in companies:
            available = []
            if c.pe_ratio:
                available.append("P/E")
            if c.revenue_growth:
                available.append("growth")
            if c.profit_margin:
                available.append("margin")
            metrics_summary.append(f"{c.ticker}: {', '.join(available)}")

        context = f"Query: {query}\nAvailable metrics: {'; '.join(metrics_summary)}"

        def make_request():
            response = self.client.models.generate_content(
                model=self.model,
                config=GenerateContentConfig(
                    system_instruction=CHART_SELECTION_PROMPT,
                    temperature=0.0,
                ),
                contents=context,
            )
            return response.text

        try:
            response_text = exponential_backoff_retry(make_request)

            # Parse JSON
            json_match = re.search(
                r"```(?:json)?\s*(.*?)\s*```", response_text, re.DOTALL
            )
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response_text.strip()

            result = json.loads(json_str)
            charts = result.get("charts", ["valuation", "profitability"])
            self._log(f"Selected charts: {charts}")
            return charts[:3]
        except Exception as e:
            self._log(f"Chart selection error: {e}, using defaults")
            return ["valuation", "profitability"]

    def _build_chart_specs(
        self,
        chart_types: List[str],
        companies: List[CompanyMetrics],
    ) -> List[ChartSpec]:
        """Build ChartSpec objects for selected chart types.

        Args:
            chart_types: List of chart type strings
            companies: Company metrics data

        Returns:
            List of ChartSpec objects
        """
        specs = []
        tickers = [c.ticker for c in companies]

        for chart_type in chart_types:
            if chart_type == "valuation":
                specs.append(
                    ChartSpec(
                        chart_type="scatter",
                        title="Valuation: P/E vs Revenue Growth",
                        x_label="Revenue Growth (%)",
                        y_label="P/E Ratio",
                        data={
                            "x_metric": "revenue_growth",
                            "y_metric": "pe_ratio",
                            "companies": tickers,
                        },
                    )
                )
            elif chart_type == "profitability":
                specs.append(
                    ChartSpec(
                        chart_type="grouped_bar",
                        title="Profitability Comparison",
                        x_label="Metric",
                        y_label="Value",
                        data={
                            "metrics": ["profit_margin", "revenue_growth"],
                            "companies": tickers,
                        },
                    )
                )
            elif chart_type == "price_history":
                specs.append(
                    ChartSpec(
                        chart_type="line",
                        title="1-Year Price Performance (Normalized)",
                        x_label="Date",
                        y_label="Normalized Price",
                        data={"companies": tickers, "period": "1y", "normalize": True},
                    )
                )
            elif chart_type == "growth":
                specs.append(
                    ChartSpec(
                        chart_type="bar",
                        title="Revenue Growth Comparison",
                        x_label="Company",
                        y_label="Revenue Growth (%)",
                        data={"metric": "revenue_growth", "companies": tickers},
                    )
                )

        return specs

    def _generate_analysis(
        self,
        query: str,
        companies: List[CompanyMetrics],
    ) -> Tuple[str, List[Citation]]:
        """Generate analysis text using ResearchAgent.

        Args:
            query: Original user query
            companies: Company metrics for context

        Returns:
            Tuple of (analysis_text, citations)
        """

        # Build context with metrics
        def fmt_pct(val):
            return f"{val*100:.1f}%" if val else "N/A"

        metrics_context = "\n".join(
            [
                f"- {c.company_name} ({c.ticker}): "
                f"P/E={c.pe_ratio or 'N/A'}, "
                f"Growth={fmt_pct(c.revenue_growth)}, "
                f"Margin={fmt_pct(c.profit_margin)}, "
                f"MCap={self._format_market_cap(c.market_cap)}"
                for c in companies
            ]
        )

        enhanced_query = (
            f"{query}\n\n"
            f"Current metrics for context:\n{metrics_context}\n\n"
            f"Provide analysis comparing these companies based on the query. "
            f"Include recent news or developments if relevant."
        )

        self._log(f"Generating analysis for: {query}")
        result = self.research_agent.answer_question(enhanced_query)

        return result.answer, result.citations

    def _format_market_cap(self, value: Optional[float]) -> str:
        """Format market cap for display."""
        if value is None:
            return "N/A"
        if value >= 1e12:
            return f"${value / 1e12:.2f}T"
        if value >= 1e9:
            return f"${value / 1e9:.2f}B"
        if value >= 1e6:
            return f"${value / 1e6:.2f}M"
        return f"${value:,.0f}"

    def generate_report(self, query: str) -> ComparisonReport:
        """Generate a full comparison report for the query.

        Args:
            query: User's natural language query

        Returns:
            ComparisonReport with metrics, analysis, charts, and citations
        """
        self._log(f"Generating report for: {query}")

        # Step 1: Extract companies
        tickers = self._extract_companies_from_query(query)

        if not tickers:
            # No companies found - return error report
            return ComparisonReport(
                query=query,
                companies=[],
                analysis_text=(
                    "Could not identify specific companies in your query. "
                    "Please mention company names or ticker symbols "
                    "(e.g., 'Compare Apple and Microsoft' or 'Analyze AAPL vs MSFT')."
                ),
                charts=[],
                citations=[],
                generated_at=datetime.now(),
            )

        # Step 2: Fetch metrics for each company
        companies: List[CompanyMetrics] = []
        failed_tickers: List[Tuple[str, str]] = []  # (ticker, error_message)

        for ticker in tickers:
            # Try to resolve if not already a ticker
            resolved, resolve_error = self.data_provider.resolve_company_to_ticker(
                ticker
            )
            if resolve_error:
                failed_tickers.append((ticker, resolve_error))
                continue

            if resolved:
                ticker = resolved

            metrics, fetch_error = self.data_provider.get_company_metrics(ticker)
            if metrics:
                companies.append(metrics)
            else:
                failed_tickers.append((ticker, fetch_error or "Unknown error"))

        if not companies:
            # Build detailed error message
            error_details = "\n".join([f"• {t}: {msg}" for t, msg in failed_tickers])
            return ComparisonReport(
                query=query,
                companies=[],
                analysis_text=(
                    f"Could not fetch data for any companies:\n\n{error_details}\n\n"
                    "**Tips:**\n"
                    "• Use ticker symbols (AAPL, MSFT) instead of company names\n"
                    "• Check spelling of ticker symbols\n"
                    "• Some international stocks may not be available"
                ),
                charts=[],
                citations=[],
                generated_at=datetime.now(),
            )

        # Step 3: Select and build charts
        # For single company, only show price history (comparisons don't make sense)
        if len(companies) == 1:
            chart_types = ["price_history"]
        else:
            chart_types = self._select_charts(query, companies)
        chart_specs = self._build_chart_specs(chart_types, companies)

        # Step 4: Generate analysis with ResearchAgent
        analysis_text, citations = self._generate_analysis(query, companies)

        # Add note about failed tickers if any
        if failed_tickers:
            failed_names = [t for t, _ in failed_tickers]
            analysis_text = (
                f"⚠️ **Note:** Could not fetch data for: {', '.join(failed_names)}\n\n"
                + analysis_text
            )

        return ComparisonReport(
            query=query,
            companies=companies,
            analysis_text=analysis_text,
            charts=chart_specs,
            citations=citations,
            generated_at=datetime.now(),
        )

    def render_charts(self, report: ComparisonReport) -> List:
        """Render Plotly figures from report chart specs.

        Args:
            report: ComparisonReport with chart specs

        Returns:
            List of Plotly Figure objects
        """
        figures = []

        for spec in report.charts:
            try:
                if spec.chart_type == "scatter":
                    fig = self.chart_generator.create_valuation_scatter(
                        report.companies,
                        x_metric=spec.data.get("x_metric", "revenue_growth"),
                        y_metric=spec.data.get("y_metric", "pe_ratio"),
                        title=spec.title,
                    )
                    figures.append(fig)

                elif spec.chart_type == "grouped_bar":
                    fig = self.chart_generator.create_metric_comparison_bar(
                        report.companies,
                        metrics=spec.data.get(
                            "metrics", ["profit_margin", "revenue_growth"]
                        ),
                        title=spec.title,
                    )
                    figures.append(fig)

                elif spec.chart_type == "bar":
                    fig = self.chart_generator.create_single_metric_bar(
                        report.companies,
                        metric=spec.data.get("metric", "revenue_growth"),
                        title=spec.title,
                    )
                    figures.append(fig)

                elif spec.chart_type == "line":
                    # Fetch historical data for price chart
                    historical_data = []
                    for company in report.companies:
                        hist = self.data_provider.get_historical_data(
                            company.ticker,
                            period=spec.data.get("period", "1y"),
                        )
                        if hist:
                            historical_data.append(hist)

                    if historical_data:
                        fig = self.chart_generator.create_price_history_line(
                            historical_data,
                            normalize=spec.data.get("normalize", True),
                            title=spec.title,
                        )
                        figures.append(fig)

            except Exception as e:
                self._log(f"Error rendering chart {spec.chart_type}: {e}")

        return figures
