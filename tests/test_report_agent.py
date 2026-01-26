"""Tests for ReportAgent."""

from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import pytest

from utils.models import ChartSpec, Citation, CompanyMetrics, ComparisonReport
from utils.report_agent import ReportAgent


@pytest.fixture
def sample_company_metrics():
    """Sample company metrics for testing."""
    return [
        CompanyMetrics(
            ticker="AAPL",
            company_name="Apple Inc.",
            market_cap=3e12,
            pe_ratio=28.5,
            forward_pe=26.0,
            eps=6.05,
            revenue_ttm=383e9,
            revenue_growth=0.08,
            profit_margin=0.25,
            debt_to_equity=1.5,
            current_ratio=1.1,
            dividend_yield=0.005,
            fifty_two_week_high=200.0,
            fifty_two_week_low=150.0,
            current_price=185.0,
            data_timestamp=datetime.now(),
        ),
        CompanyMetrics(
            ticker="GOOGL",
            company_name="Alphabet Inc.",
            market_cap=2e12,
            pe_ratio=25.0,
            forward_pe=22.0,
            eps=5.80,
            revenue_ttm=320e9,
            revenue_growth=0.12,
            profit_margin=0.22,
            debt_to_equity=0.5,
            current_ratio=2.0,
            dividend_yield=None,
            fifty_two_week_high=180.0,
            fifty_two_week_low=120.0,
            current_price=165.0,
            data_timestamp=datetime.now(),
        ),
    ]


@pytest.fixture
def mock_gemini_extraction_response():
    """Mock Gemini response for company extraction."""
    mock = Mock()
    mock.text = '{"companies": ["AAPL", "GOOGL"]}'
    return mock


@pytest.fixture
def mock_gemini_chart_response():
    """Mock Gemini response for chart selection."""
    mock = Mock()
    mock.text = '{"charts": ["valuation", "profitability"]}'
    return mock


class TestReportAgentExtraction:
    """Tests for company extraction."""

    @patch("utils.report_agent.genai.Client")
    def test_extract_companies_simple(self, mock_client):
        """Test basic company extraction."""
        mock_model = MagicMock()
        mock_response = Mock()
        mock_response.text = '{"companies": ["AAPL", "MSFT"]}'
        mock_model.generate_content.return_value = mock_response
        mock_client.return_value.models = mock_model

        agent = ReportAgent()
        companies = agent._extract_companies_from_query("Compare Apple and Microsoft")

        assert companies == ["AAPL", "MSFT"]

    @patch("utils.report_agent.genai.Client")
    def test_extract_companies_with_code_block(self, mock_client):
        """Test extraction with markdown code block."""
        mock_model = MagicMock()
        mock_response = Mock()
        mock_response.text = '```json\n{"companies": ["TSLA"]}\n```'
        mock_model.generate_content.return_value = mock_response
        mock_client.return_value.models = mock_model

        agent = ReportAgent()
        companies = agent._extract_companies_from_query("Is Tesla overvalued?")

        assert companies == ["TSLA"]

    @patch("utils.report_agent.genai.Client")
    def test_extract_companies_empty(self, mock_client):
        """Test extraction when no companies found."""
        mock_model = MagicMock()
        mock_response = Mock()
        mock_response.text = '{"companies": []}'
        mock_model.generate_content.return_value = mock_response
        mock_client.return_value.models = mock_model

        agent = ReportAgent()
        companies = agent._extract_companies_from_query("What is the stock market?")

        assert companies == []

    @patch("utils.report_agent.genai.Client")
    def test_extract_companies_max_five(self, mock_client):
        """Test that extraction limits to 5 companies."""
        mock_model = MagicMock()
        mock_response = Mock()
        mock_response.text = (
            '{"companies": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]}'
        )
        mock_model.generate_content.return_value = mock_response
        mock_client.return_value.models = mock_model

        agent = ReportAgent()
        companies = agent._extract_companies_from_query("Compare all FAANG stocks")

        assert len(companies) == 5

    @patch("utils.report_agent.genai.Client")
    def test_extract_companies_invalid_json(self, mock_client):
        """Test handling of invalid JSON response."""
        mock_model = MagicMock()
        mock_response = Mock()
        mock_response.text = "invalid json"
        mock_model.generate_content.return_value = mock_response
        mock_client.return_value.models = mock_model

        agent = ReportAgent()
        companies = agent._extract_companies_from_query("Compare Apple and Google")

        assert companies == []


class TestReportAgentChartSelection:
    """Tests for chart selection."""

    @patch("utils.report_agent.genai.Client")
    def test_select_charts_default(self, mock_client, sample_company_metrics):
        """Test chart selection with valid response."""
        mock_model = MagicMock()
        mock_response = Mock()
        mock_response.text = '{"charts": ["valuation", "price_history"]}'
        mock_model.generate_content.return_value = mock_response
        mock_client.return_value.models = mock_model

        agent = ReportAgent()
        charts = agent._select_charts("Which stock is better?", sample_company_metrics)

        assert "valuation" in charts
        assert "price_history" in charts

    @patch("utils.report_agent.genai.Client")
    def test_select_charts_max_three(self, mock_client, sample_company_metrics):
        """Test that chart selection limits to 3."""
        mock_model = MagicMock()
        mock_response = Mock()
        mock_response.text = (
            '{"charts": ["valuation", "profitability", "price_history", "growth"]}'
        )
        mock_model.generate_content.return_value = mock_response
        mock_client.return_value.models = mock_model

        agent = ReportAgent()
        charts = agent._select_charts("Full analysis", sample_company_metrics)

        assert len(charts) == 3


class TestReportAgentChartSpecs:
    """Tests for chart spec building."""

    def test_build_valuation_chart_spec(self, sample_company_metrics):
        """Test building valuation scatter chart spec."""
        agent = ReportAgent.__new__(ReportAgent)
        specs = agent._build_chart_specs(["valuation"], sample_company_metrics)

        assert len(specs) == 1
        assert specs[0].chart_type == "scatter"
        assert "valuation" in specs[0].title.lower()
        assert specs[0].data["x_metric"] == "revenue_growth"
        assert specs[0].data["y_metric"] == "pe_ratio"

    def test_build_profitability_chart_spec(self, sample_company_metrics):
        """Test building profitability bar chart spec."""
        agent = ReportAgent.__new__(ReportAgent)
        specs = agent._build_chart_specs(["profitability"], sample_company_metrics)

        assert len(specs) == 1
        assert specs[0].chart_type == "grouped_bar"
        assert "profit_margin" in specs[0].data["metrics"]

    def test_build_price_history_chart_spec(self, sample_company_metrics):
        """Test building price history line chart spec."""
        agent = ReportAgent.__new__(ReportAgent)
        specs = agent._build_chart_specs(["price_history"], sample_company_metrics)

        assert len(specs) == 1
        assert specs[0].chart_type == "line"
        assert specs[0].data["normalize"] is True

    def test_build_growth_chart_spec(self, sample_company_metrics):
        """Test building growth bar chart spec."""
        agent = ReportAgent.__new__(ReportAgent)
        specs = agent._build_chart_specs(["growth"], sample_company_metrics)

        assert len(specs) == 1
        assert specs[0].chart_type == "bar"
        assert specs[0].data["metric"] == "revenue_growth"

    def test_build_multiple_chart_specs(self, sample_company_metrics):
        """Test building multiple chart specs."""
        agent = ReportAgent.__new__(ReportAgent)
        specs = agent._build_chart_specs(
            ["valuation", "profitability", "price_history"], sample_company_metrics
        )

        assert len(specs) == 3
        chart_types = [s.chart_type for s in specs]
        assert "scatter" in chart_types
        assert "grouped_bar" in chart_types
        assert "line" in chart_types


class TestReportAgentGeneration:
    """Tests for full report generation."""

    @patch("utils.report_agent.ResearchAgent")
    @patch("utils.report_agent.FinancialDataProvider")
    @patch("utils.report_agent.genai.Client")
    def test_generate_report_success(
        self,
        mock_client,
        mock_data_provider,
        mock_research_agent,
        sample_company_metrics,
    ):
        """Test successful report generation."""
        # Mock Gemini for extraction and chart selection
        mock_model = MagicMock()

        def side_effect(*args, **kwargs):
            contents = kwargs.get("contents", args[0] if args else "")
            if isinstance(contents, str) and "Compare" in contents:
                resp = Mock()
                resp.text = '{"companies": ["AAPL", "GOOGL"]}'
                return resp
            resp = Mock()
            resp.text = '{"charts": ["valuation"]}'
            return resp

        mock_model.generate_content.side_effect = side_effect
        mock_client.return_value.models = mock_model

        # Mock data provider
        mock_provider_instance = MagicMock()
        mock_provider_instance.resolve_company_to_ticker.side_effect = lambda x: (
            x,
            None,
        )
        mock_provider_instance.get_company_metrics.side_effect = lambda t: (
            (sample_company_metrics[0], None)
            if t == "AAPL"
            else (sample_company_metrics[1], None)
        )
        mock_data_provider.return_value = mock_provider_instance

        # Mock research agent
        mock_research_instance = MagicMock()
        mock_answer = Mock()
        mock_answer.answer = "Apple and Google are both strong companies."
        mock_answer.citations = [
            Citation(title="Source", url="https://example.com", source_type="News")
        ]
        mock_research_instance.answer_question.return_value = mock_answer
        mock_research_agent.return_value = mock_research_instance

        agent = ReportAgent()
        report = agent.generate_report("Compare Apple and Google")

        assert isinstance(report, ComparisonReport)
        assert report.query == "Compare Apple and Google"
        assert len(report.companies) == 2
        assert len(report.charts) > 0
        assert report.analysis_text != ""
        assert report.generated_at is not None

    @patch("utils.report_agent.ResearchAgent")
    @patch("utils.report_agent.FinancialDataProvider")
    @patch("utils.report_agent.genai.Client")
    def test_generate_report_no_companies(
        self, mock_client, mock_data_provider, mock_research_agent
    ):
        """Test report generation when no companies found."""
        mock_model = MagicMock()
        mock_response = Mock()
        mock_response.text = '{"companies": []}'
        mock_model.generate_content.return_value = mock_response
        mock_client.return_value.models = mock_model

        agent = ReportAgent()
        report = agent.generate_report("What is the stock market?")

        assert report.companies == []
        assert "Could not identify" in report.analysis_text
        assert report.charts == []

    @patch("utils.report_agent.ResearchAgent")
    @patch("utils.report_agent.FinancialDataProvider")
    @patch("utils.report_agent.genai.Client")
    def test_generate_report_failed_ticker(
        self, mock_client, mock_data_provider, mock_research_agent
    ):
        """Test report generation with invalid ticker."""
        mock_model = MagicMock()
        mock_response = Mock()
        mock_response.text = '{"companies": ["INVALID"]}'
        mock_model.generate_content.return_value = mock_response
        mock_client.return_value.models = mock_model

        mock_provider_instance = MagicMock()
        mock_provider_instance.resolve_company_to_ticker.return_value = (
            "INVALID",
            None,
        )
        mock_provider_instance.get_company_metrics.return_value = (
            None,
            "No data for INVALID",
        )
        mock_data_provider.return_value = mock_provider_instance

        agent = ReportAgent()
        report = agent.generate_report("Analyze INVALID")

        assert report.companies == []
        assert "Could not fetch data" in report.analysis_text


class TestReportAgentRenderCharts:
    """Tests for chart rendering."""

    @patch("utils.report_agent.genai.Client")
    def test_render_scatter_chart(self, mock_client, sample_company_metrics):
        """Test rendering scatter chart."""
        agent = ReportAgent()

        report = ComparisonReport(
            query="Test",
            companies=sample_company_metrics,
            analysis_text="Test analysis",
            charts=[
                ChartSpec(
                    chart_type="scatter",
                    title="Test Scatter",
                    data={
                        "x_metric": "revenue_growth",
                        "y_metric": "pe_ratio",
                        "companies": ["AAPL", "GOOGL"],
                    },
                )
            ],
            citations=[],
            generated_at=datetime.now(),
        )

        figures = agent.render_charts(report)

        assert len(figures) == 1

    @patch("utils.report_agent.genai.Client")
    def test_render_bar_chart(self, mock_client, sample_company_metrics):
        """Test rendering bar chart."""
        agent = ReportAgent()

        report = ComparisonReport(
            query="Test",
            companies=sample_company_metrics,
            analysis_text="Test analysis",
            charts=[
                ChartSpec(
                    chart_type="bar",
                    title="Test Bar",
                    data={"metric": "revenue_growth", "companies": ["AAPL", "GOOGL"]},
                )
            ],
            citations=[],
            generated_at=datetime.now(),
        )

        figures = agent.render_charts(report)

        assert len(figures) == 1

    @patch("utils.report_agent.genai.Client")
    def test_render_grouped_bar_chart(self, mock_client, sample_company_metrics):
        """Test rendering grouped bar chart."""
        agent = ReportAgent()

        report = ComparisonReport(
            query="Test",
            companies=sample_company_metrics,
            analysis_text="Test analysis",
            charts=[
                ChartSpec(
                    chart_type="grouped_bar",
                    title="Test Grouped Bar",
                    data={
                        "metrics": ["profit_margin", "revenue_growth"],
                        "companies": ["AAPL", "GOOGL"],
                    },
                )
            ],
            citations=[],
            generated_at=datetime.now(),
        )

        figures = agent.render_charts(report)

        assert len(figures) == 1


class TestFormatMarketCap:
    """Tests for market cap formatting."""

    def test_format_trillion(self):
        """Test trillion formatting."""
        agent = ReportAgent.__new__(ReportAgent)
        assert agent._format_market_cap(3.5e12) == "$3.50T"

    def test_format_billion(self):
        """Test billion formatting."""
        agent = ReportAgent.__new__(ReportAgent)
        assert agent._format_market_cap(250e9) == "$250.00B"

    def test_format_million(self):
        """Test million formatting."""
        agent = ReportAgent.__new__(ReportAgent)
        assert agent._format_market_cap(50e6) == "$50.00M"

    def test_format_none(self):
        """Test None value."""
        agent = ReportAgent.__new__(ReportAgent)
        assert agent._format_market_cap(None) == "N/A"
