"""Tests for chart_generator module."""

from datetime import datetime, timedelta

import plotly.graph_objects as go
import pytest

from utils.chart_generator import ChartGenerator
from utils.models import CompanyMetrics, HistoricalData


@pytest.fixture
def chart_generator():
    """Create ChartGenerator instance."""
    return ChartGenerator()


@pytest.fixture
def sample_companies():
    """Create sample company metrics for testing."""
    return [
        CompanyMetrics(
            ticker="AAPL",
            company_name="Apple Inc.",
            market_cap=3_000_000_000_000,
            pe_ratio=28.5,
            forward_pe=25.2,
            eps=6.42,
            revenue_ttm=385_000_000_000,
            revenue_growth=0.08,
            profit_margin=0.26,
            debt_to_equity=1.87,
            current_ratio=0.99,
            dividend_yield=0.005,
            fifty_two_week_high=199.62,
            fifty_two_week_low=164.08,
            current_price=185.50,
            data_timestamp=datetime.now(),
        ),
        CompanyMetrics(
            ticker="MSFT",
            company_name="Microsoft Corporation",
            market_cap=2_800_000_000_000,
            pe_ratio=35.2,
            forward_pe=30.1,
            eps=11.53,
            revenue_ttm=227_000_000_000,
            revenue_growth=0.12,
            profit_margin=0.35,
            debt_to_equity=0.42,
            current_ratio=1.77,
            dividend_yield=0.007,
            fifty_two_week_high=420.82,
            fifty_two_week_low=309.98,
            current_price=405.20,
            data_timestamp=datetime.now(),
        ),
        CompanyMetrics(
            ticker="GOOGL",
            company_name="Alphabet Inc.",
            market_cap=1_900_000_000_000,
            pe_ratio=24.3,
            forward_pe=21.8,
            eps=5.80,
            revenue_ttm=307_000_000_000,
            revenue_growth=0.15,
            profit_margin=0.25,
            debt_to_equity=0.10,
            current_ratio=2.10,
            dividend_yield=None,
            fifty_two_week_high=155.74,
            fifty_two_week_low=120.21,
            current_price=148.50,
            data_timestamp=datetime.now(),
        ),
    ]


@pytest.fixture
def sample_historical_data():
    """Create sample historical data for testing."""
    base_date = datetime.now()
    dates = [base_date - timedelta(days=x) for x in range(30, 0, -1)]

    return [
        HistoricalData(
            ticker="AAPL",
            dates=dates,
            close_prices=[175 + i * 0.3 for i in range(30)],
            volumes=[50_000_000 + i * 100_000 for i in range(30)],
        ),
        HistoricalData(
            ticker="MSFT",
            dates=dates,
            close_prices=[390 + i * 0.5 for i in range(30)],
            volumes=[30_000_000 + i * 80_000 for i in range(30)],
        ),
    ]


class TestChartGenerator:
    """Tests for ChartGenerator class."""

    def test_init(self, chart_generator):
        """Test ChartGenerator initialization."""
        assert chart_generator._layout_defaults is not None
        assert "template" in chart_generator._layout_defaults
        assert len(chart_generator.COLORS) > 0

    def test_get_color(self, chart_generator):
        """Test color palette cycling."""
        assert chart_generator._get_color(0) == "#636EFA"
        assert chart_generator._get_color(1) == "#EF553B"
        # Test wrapping
        assert chart_generator._get_color(8) == chart_generator._get_color(0)

    def test_format_large_number_trillions(self, chart_generator):
        """Test formatting trillion values."""
        assert chart_generator._format_large_number(3_000_000_000_000) == "$3.00T"
        assert chart_generator._format_large_number(1_500_000_000_000) == "$1.50T"

    def test_format_large_number_billions(self, chart_generator):
        """Test formatting billion values."""
        assert chart_generator._format_large_number(385_000_000_000) == "$385.00B"
        assert chart_generator._format_large_number(2_500_000_000) == "$2.50B"

    def test_format_large_number_millions(self, chart_generator):
        """Test formatting million values."""
        assert chart_generator._format_large_number(50_000_000) == "$50.00M"
        assert chart_generator._format_large_number(1_200_000) == "$1.20M"

    def test_format_large_number_small(self, chart_generator):
        """Test formatting smaller values."""
        assert chart_generator._format_large_number(50_000) == "$50,000"

    def test_format_large_number_none(self, chart_generator):
        """Test formatting None value."""
        assert chart_generator._format_large_number(None) == "N/A"


class TestMetricComparisonBar:
    """Tests for metric comparison bar charts."""

    def test_creates_figure(self, chart_generator, sample_companies):
        """Test that metric comparison creates a valid figure."""
        fig = chart_generator.create_metric_comparison_bar(
            sample_companies, ["pe_ratio", "profit_margin"]
        )
        assert isinstance(fig, go.Figure)

    def test_has_correct_traces(self, chart_generator, sample_companies):
        """Test that chart has one trace per company."""
        fig = chart_generator.create_metric_comparison_bar(
            sample_companies, ["pe_ratio"]
        )
        assert len(fig.data) == len(sample_companies)

    def test_trace_names_match_companies(self, chart_generator, sample_companies):
        """Test that trace names match company names."""
        fig = chart_generator.create_metric_comparison_bar(
            sample_companies, ["pe_ratio"]
        )
        trace_names = [trace.name for trace in fig.data]
        company_names = [c.company_name for c in sample_companies]
        assert trace_names == company_names

    def test_custom_title(self, chart_generator, sample_companies):
        """Test custom chart title."""
        custom_title = "Custom Comparison Title"
        fig = chart_generator.create_metric_comparison_bar(
            sample_companies, ["pe_ratio"], title=custom_title
        )
        assert fig.layout.title.text == custom_title

    def test_multiple_metrics(self, chart_generator, sample_companies):
        """Test chart with multiple metrics."""
        metrics = ["pe_ratio", "profit_margin", "revenue_growth"]
        fig = chart_generator.create_metric_comparison_bar(sample_companies, metrics)
        # Each trace should have values for all metrics
        assert len(fig.data[0].x) == len(metrics)

    def test_handles_none_values(self, chart_generator):
        """Test handling of None metric values."""
        companies = [
            CompanyMetrics(
                ticker="TEST",
                company_name="Test Corp",
                pe_ratio=None,
                profit_margin=0.15,
                data_timestamp=datetime.now(),
            )
        ]
        fig = chart_generator.create_metric_comparison_bar(
            companies, ["pe_ratio", "profit_margin"]
        )
        assert isinstance(fig, go.Figure)


class TestPriceHistoryLine:
    """Tests for price history line charts."""

    def test_creates_figure(self, chart_generator, sample_historical_data):
        """Test that price history creates a valid figure."""
        fig = chart_generator.create_price_history_line(sample_historical_data)
        assert isinstance(fig, go.Figure)

    def test_has_correct_traces(self, chart_generator, sample_historical_data):
        """Test that chart has one trace per ticker."""
        fig = chart_generator.create_price_history_line(sample_historical_data)
        assert len(fig.data) == len(sample_historical_data)

    def test_normalized_prices(self, chart_generator, sample_historical_data):
        """Test that prices are normalized to 100 at start."""
        fig = chart_generator.create_price_history_line(
            sample_historical_data, normalize=True
        )
        # First price should be 100 for each trace
        for trace in fig.data:
            assert trace.y[0] == 100

    def test_non_normalized_prices(self, chart_generator, sample_historical_data):
        """Test non-normalized prices."""
        fig = chart_generator.create_price_history_line(
            sample_historical_data, normalize=False
        )
        # First price should be original price
        assert fig.data[0].y[0] == sample_historical_data[0].close_prices[0]

    def test_custom_title(self, chart_generator, sample_historical_data):
        """Test custom chart title."""
        custom_title = "1 Year Price Comparison"
        fig = chart_generator.create_price_history_line(
            sample_historical_data, title=custom_title
        )
        assert fig.layout.title.text == custom_title

    def test_trace_names_match_tickers(self, chart_generator, sample_historical_data):
        """Test that trace names match tickers."""
        fig = chart_generator.create_price_history_line(sample_historical_data)
        trace_names = [trace.name for trace in fig.data]
        tickers = [d.ticker for d in sample_historical_data]
        assert trace_names == tickers


class TestValuationScatter:
    """Tests for valuation scatter plots."""

    def test_creates_figure(self, chart_generator, sample_companies):
        """Test that valuation scatter creates a valid figure."""
        fig = chart_generator.create_valuation_scatter(sample_companies)
        assert isinstance(fig, go.Figure)

    def test_default_axes(self, chart_generator, sample_companies):
        """Test default x and y axes."""
        fig = chart_generator.create_valuation_scatter(sample_companies)
        assert "Growth" in fig.layout.xaxis.title.text
        assert "P/E" in fig.layout.yaxis.title.text

    def test_custom_axes(self, chart_generator, sample_companies):
        """Test custom x and y metrics."""
        fig = chart_generator.create_valuation_scatter(
            sample_companies, x_metric="profit_margin", y_metric="debt_to_equity"
        )
        assert "Profit Margin" in fig.layout.xaxis.title.text
        assert "Debt/Equity" in fig.layout.yaxis.title.text

    def test_custom_title(self, chart_generator, sample_companies):
        """Test custom chart title."""
        custom_title = "Growth vs Value Analysis"
        fig = chart_generator.create_valuation_scatter(
            sample_companies, title=custom_title
        )
        assert fig.layout.title.text == custom_title

    def test_skips_none_values(self, chart_generator):
        """Test that companies with None values are skipped."""
        companies = [
            CompanyMetrics(
                ticker="TEST1",
                company_name="Test Corp 1",
                pe_ratio=25.0,
                revenue_growth=0.10,
                market_cap=1_000_000_000,
                data_timestamp=datetime.now(),
            ),
            CompanyMetrics(
                ticker="TEST2",
                company_name="Test Corp 2",
                pe_ratio=None,  # Missing PE
                revenue_growth=0.15,
                market_cap=500_000_000,
                data_timestamp=datetime.now(),
            ),
        ]
        fig = chart_generator.create_valuation_scatter(companies)
        # Only one company should be plotted
        assert len(fig.data) == 1


class TestSingleMetricBar:
    """Tests for single metric bar charts."""

    def test_creates_figure(self, chart_generator, sample_companies):
        """Test that single metric bar creates a valid figure."""
        fig = chart_generator.create_single_metric_bar(sample_companies, "market_cap")
        assert isinstance(fig, go.Figure)

    def test_auto_generated_title(self, chart_generator, sample_companies):
        """Test auto-generated title from metric name."""
        fig = chart_generator.create_single_metric_bar(sample_companies, "pe_ratio")
        assert "P/E Ratio" in fig.layout.title.text

    def test_custom_title(self, chart_generator, sample_companies):
        """Test custom title override."""
        custom_title = "Custom Bar Title"
        fig = chart_generator.create_single_metric_bar(
            sample_companies, "pe_ratio", title=custom_title
        )
        assert fig.layout.title.text == custom_title

    def test_percentage_metrics_scaled(self, chart_generator, sample_companies):
        """Test that percentage metrics are scaled to 0-100."""
        fig = chart_generator.create_single_metric_bar(
            sample_companies, "profit_margin"
        )
        # Values should be in percentage form (e.g., 26 not 0.26)
        assert fig.data[0].y[0] > 1  # AAPL profit margin is 0.26 -> 26

    def test_skips_none_values(self, chart_generator):
        """Test that companies with None values are skipped."""
        companies = [
            CompanyMetrics(
                ticker="TEST1",
                company_name="Test Corp 1",
                pe_ratio=25.0,
                data_timestamp=datetime.now(),
            ),
            CompanyMetrics(
                ticker="TEST2",
                company_name="Test Corp 2",
                pe_ratio=None,
                data_timestamp=datetime.now(),
            ),
        ]
        fig = chart_generator.create_single_metric_bar(companies, "pe_ratio")
        # Only one bar should be present
        assert len(fig.data[0].x) == 1
