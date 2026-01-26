"""Chart generation utilities using Plotly."""

from typing import List, Optional

import plotly.graph_objects as go

from utils.models import CompanyMetrics, HistoricalData


class ChartGenerator:
    """Generates Plotly charts for financial data visualization."""

    # Consistent color palette for companies
    COLORS = [
        "#636EFA",  # Blue
        "#EF553B",  # Red
        "#00CC96",  # Green
        "#AB63FA",  # Purple
        "#FFA15A",  # Orange
        "#19D3F3",  # Cyan
        "#FF6692",  # Pink
        "#B6E880",  # Light green
    ]

    def __init__(self):
        """Initialize chart generator."""
        self._layout_defaults = {
            "template": "plotly_white",
            "font": {"family": "Arial, sans-serif", "size": 12},
            "margin": {"l": 60, "r": 40, "t": 60, "b": 60},
            "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02},
        }

    def _get_color(self, index: int) -> str:
        """Get color for company at index."""
        return self.COLORS[index % len(self.COLORS)]

    def _format_large_number(self, value: Optional[float]) -> str:
        """Format large numbers for display."""
        if value is None:
            return "N/A"
        if abs(value) >= 1e12:
            return f"${value / 1e12:.2f}T"
        if abs(value) >= 1e9:
            return f"${value / 1e9:.2f}B"
        if abs(value) >= 1e6:
            return f"${value / 1e6:.2f}M"
        return f"${value:,.0f}"

    def create_metric_comparison_bar(
        self,
        companies: List[CompanyMetrics],
        metrics: List[str],
        title: str = "Company Metrics Comparison",
    ) -> go.Figure:
        """Create grouped bar chart comparing metrics across companies.

        Args:
            companies: List of CompanyMetrics to compare
            metrics: List of metric names (e.g., ['pe_ratio', 'profit_margin'])
            title: Chart title

        Returns:
            Plotly Figure object
        """
        fig = go.Figure()

        # Metric display names and formatting
        metric_config = {
            "market_cap": {"label": "Market Cap", "format": "currency_large"},
            "pe_ratio": {"label": "P/E Ratio", "format": "number"},
            "forward_pe": {"label": "Forward P/E", "format": "number"},
            "eps": {"label": "EPS", "format": "currency"},
            "revenue_ttm": {"label": "Revenue (TTM)", "format": "currency_large"},
            "revenue_growth": {"label": "Revenue Growth", "format": "percent"},
            "profit_margin": {"label": "Profit Margin", "format": "percent"},
            "debt_to_equity": {"label": "Debt/Equity", "format": "number"},
            "current_ratio": {"label": "Current Ratio", "format": "number"},
            "dividend_yield": {"label": "Dividend Yield", "format": "percent"},
        }

        # Get metric labels for x-axis
        x_labels = [
            metric_config.get(m, {"label": m})["label"]
            for m in metrics
            if m in metric_config
        ]

        # Add bar for each company
        for i, company in enumerate(companies):
            values = []
            hover_texts = []

            for metric in metrics:
                value = getattr(company, metric, None)
                values.append(value if value is not None else 0)

                # Format hover text
                config = metric_config.get(metric, {"format": "number"})
                if value is None:
                    hover_texts.append("N/A")
                elif config["format"] == "currency_large":
                    hover_texts.append(self._format_large_number(value))
                elif config["format"] == "percent":
                    hover_texts.append(f"{value * 100:.1f}%")
                elif config["format"] == "currency":
                    hover_texts.append(f"${value:.2f}")
                else:
                    hover_texts.append(f"{value:.2f}")

            fig.add_trace(
                go.Bar(
                    name=company.company_name,
                    x=x_labels,
                    y=values,
                    marker_color=self._get_color(i),
                    text=hover_texts,
                    textposition="outside",
                    hovertemplate="%{x}: %{text}<extra>%{fullData.name}</extra>",
                )
            )

        fig.update_layout(
            **self._layout_defaults,
            title={"text": title, "x": 0.5, "xanchor": "center"},
            barmode="group",
            xaxis_title="Metric",
            yaxis_title="Value",
            showlegend=True,
        )

        return fig

    def create_price_history_line(
        self,
        historical_data: List[HistoricalData],
        normalize: bool = True,
        title: str = "Price History",
    ) -> go.Figure:
        """Create line chart showing price history for multiple companies.

        Args:
            historical_data: List of HistoricalData for each company
            normalize: If True, normalize prices to 100 at start for comparison
            title: Chart title

        Returns:
            Plotly Figure object
        """
        fig = go.Figure()

        for i, data in enumerate(historical_data):
            prices = data.close_prices

            if normalize and prices:
                # Normalize to 100 at start
                base_price = prices[0]
                if base_price > 0:
                    prices = [(p / base_price) * 100 for p in prices]

            fig.add_trace(
                go.Scatter(
                    name=data.ticker,
                    x=data.dates,
                    y=prices,
                    mode="lines",
                    line={"color": self._get_color(i), "width": 2},
                    hovertemplate=(
                        "%{x|%b %d, %Y}<br>"
                        + ("Normalized: %{y:.1f}" if normalize else "Price: $%{y:.2f}")
                        + "<extra>%{fullData.name}</extra>"
                    ),
                )
            )

        y_title = "Normalized Price (Base=100)" if normalize else "Price ($)"

        fig.update_layout(
            **self._layout_defaults,
            title={"text": title, "x": 0.5, "xanchor": "center"},
            xaxis_title="Date",
            yaxis_title=y_title,
            hovermode="x unified",
            showlegend=True,
        )

        return fig

    def create_valuation_scatter(
        self,
        companies: List[CompanyMetrics],
        x_metric: str = "revenue_growth",
        y_metric: str = "pe_ratio",
        title: str = "Valuation Analysis",
    ) -> go.Figure:
        """Create scatter plot for valuation analysis.

        Args:
            companies: List of CompanyMetrics to plot
            x_metric: Metric for x-axis (default: revenue_growth)
            y_metric: Metric for y-axis (default: pe_ratio)
            title: Chart title

        Returns:
            Plotly Figure object
        """
        fig = go.Figure()

        # Metric labels
        labels = {
            "pe_ratio": "P/E Ratio",
            "forward_pe": "Forward P/E",
            "revenue_growth": "Revenue Growth (%)",
            "profit_margin": "Profit Margin (%)",
            "market_cap": "Market Cap",
            "debt_to_equity": "Debt/Equity",
            "eps": "EPS ($)",
        }

        for i, company in enumerate(companies):
            x_val = getattr(company, x_metric, None)
            y_val = getattr(company, y_metric, None)

            # Convert percentages for display
            x_display = x_val
            y_display = y_val
            if x_metric in ["revenue_growth", "profit_margin"] and x_val is not None:
                x_display = x_val * 100
            if y_metric in ["revenue_growth", "profit_margin"] and y_val is not None:
                y_display = y_val * 100

            if x_display is not None and y_display is not None:
                # Size based on market cap (normalized)
                market_cap = company.market_cap or 1e9
                size = max(20, min(60, (market_cap / 1e11) * 30 + 20))

                fig.add_trace(
                    go.Scatter(
                        name=company.company_name,
                        x=[x_display],
                        y=[y_display],
                        mode="markers+text",
                        marker={
                            "color": self._get_color(i),
                            "size": size,
                            "opacity": 0.7,
                            "line": {"width": 2, "color": "white"},
                        },
                        text=[company.ticker],
                        textposition="top center",
                        hovertemplate=(
                            f"<b>{company.company_name}</b><br>"
                            f"{labels.get(x_metric, x_metric)}: %{{x:.1f}}<br>"
                            f"{labels.get(y_metric, y_metric)}: %{{y:.1f}}<br>"
                            f"Market Cap: "
                            f"{self._format_large_number(company.market_cap)}"
                            "<extra></extra>"
                        ),
                    )
                )

        fig.update_layout(
            **self._layout_defaults,
            title={"text": title, "x": 0.5, "xanchor": "center"},
            xaxis_title=labels.get(x_metric, x_metric),
            yaxis_title=labels.get(y_metric, y_metric),
            showlegend=False,
        )

        return fig

    def create_single_metric_bar(
        self,
        companies: List[CompanyMetrics],
        metric: str,
        title: Optional[str] = None,
    ) -> go.Figure:
        """Create simple bar chart for single metric comparison.

        Args:
            companies: List of CompanyMetrics
            metric: Single metric to compare
            title: Chart title (auto-generated if None)

        Returns:
            Plotly Figure object
        """
        metric_labels = {
            "market_cap": "Market Cap",
            "pe_ratio": "P/E Ratio",
            "forward_pe": "Forward P/E",
            "eps": "EPS",
            "revenue_ttm": "Revenue (TTM)",
            "revenue_growth": "Revenue Growth",
            "profit_margin": "Profit Margin",
            "debt_to_equity": "Debt/Equity",
            "current_ratio": "Current Ratio",
            "dividend_yield": "Dividend Yield",
        }

        fig = go.Figure()

        names = []
        values = []
        colors = []
        hover_texts = []

        for i, company in enumerate(companies):
            value = getattr(company, metric, None)
            if value is not None:
                names.append(company.ticker)
                colors.append(self._get_color(i))

                # Format value for display
                if metric in ["revenue_growth", "profit_margin", "dividend_yield"]:
                    values.append(value * 100)
                    hover_texts.append(f"{value * 100:.1f}%")
                elif metric in ["market_cap", "revenue_ttm"]:
                    values.append(value)
                    hover_texts.append(self._format_large_number(value))
                else:
                    values.append(value)
                    hover_texts.append(f"{value:.2f}")

        fig.add_trace(
            go.Bar(
                x=names,
                y=values,
                marker_color=colors,
                text=hover_texts,
                textposition="outside",
                hovertemplate="%{x}: %{text}<extra></extra>",
            )
        )

        chart_title = title or f"{metric_labels.get(metric, metric)} Comparison"
        y_label = metric_labels.get(metric, metric)
        if metric in ["revenue_growth", "profit_margin", "dividend_yield"]:
            y_label += " (%)"

        fig.update_layout(
            **self._layout_defaults,
            title={"text": chart_title, "x": 0.5, "xanchor": "center"},
            xaxis_title="Company",
            yaxis_title=y_label,
            showlegend=False,
        )

        return fig
