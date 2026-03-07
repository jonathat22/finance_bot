"""VisualizationAgent: ASCII charts and rich tables for CLI output.

All output is plain text strings suitable for display in a terminal.
"""

import argparse
import json
import sys
from io import StringIO
from typing import Any, Dict, List, Optional

import plotext as plt
from rich.console import Console
from rich.table import Table


def metrics_table(data: Dict[str, Any]) -> str:
    """Render a company metrics summary table."""
    metrics = data.get("metrics", {})
    console = Console(file=StringIO(), width=72, force_terminal=True)
    table = Table(title="Current Metrics", show_header=False, padding=(0, 1))
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_column("Metric", style="bold")
    table.add_column("Value")

    rows = [
        ("Price", _fmt_price(metrics.get("current_price")),
         "Market Cap", _fmt_large(metrics.get("market_cap"))),
        ("P/E (TTM)", _fmt_multiple(metrics.get("pe_ratio")),
         "Fwd P/E", _fmt_multiple(metrics.get("forward_pe"))),
        ("EPS (TTM)", _fmt_price(metrics.get("eps")),
         "Revenue", _fmt_large(metrics.get("revenue_ttm"))),
        ("Net Margin", _fmt_pct(metrics.get("profit_margin")),
         "Rev Growth", _fmt_pct(metrics.get("revenue_growth"))),
        ("D/E", _fmt_ratio(metrics.get("debt_to_equity")),
         "Current Ratio", _fmt_ratio(metrics.get("current_ratio"))),
        ("Div Yield", _fmt_pct(metrics.get("dividend_yield")),
         "52W Range", _fmt_range(
             metrics.get("fifty_two_week_low"),
             metrics.get("fifty_two_week_high"))),
    ]

    for m1, v1, m2, v2 in rows:
        table.add_row(m1, v1, m2, v2)

    console.print(table)
    return console.file.getvalue()


def valuation_table(scorecard: Dict[str, Any]) -> str:
    """Render intrinsic value estimates as a table."""
    console = Console(file=StringIO(), width=72, force_terminal=True)
    table = Table(title="Intrinsic Value Estimates", padding=(0, 1))
    table.add_column("Method", style="bold")
    table.add_column("Value/Share", justify="right")
    table.add_column("Current", justify="right")
    table.add_column("Margin of Safety", justify="right")

    for iv in scorecard.get("intrinsic_values", []):
        method = iv["method"].replace("_", " ").title()
        value = _fmt_price(iv["intrinsic_value_per_share"])
        current = _fmt_price(iv["current_price"])
        mos = iv["margin_of_safety_pct"]
        label = "undervalued" if mos > 0 else "overvalued"
        mos_str = f"{mos:+.1f}% ({label})"
        table.add_row(method, value, current, mos_str)

    console.print(table)
    return console.file.getvalue()


def comparison_table(scorecards: List[Dict[str, Any]]) -> str:
    """Render side-by-side comparison of multiple companies."""
    console = Console(file=StringIO(), width=90, force_terminal=True)

    # Fundamentals table
    fund_table = Table(title="Fundamentals Comparison", padding=(0, 1))
    fund_table.add_column("Metric", style="bold")
    for sc in scorecards:
        fund_table.add_column(sc["ticker"], justify="right")

    metric_rows = [
        ("Price", lambda sc: _fmt_price(sc.get("current_price"))),
        ("Rating", lambda sc: sc.get("overall_rating", "N/A").upper()),
        ("Avg MOS", lambda sc: _avg_mos(sc)),
        ("Moat", lambda sc: (sc.get("moat") or {}).get("moat_rating", "N/A").capitalize()),
        ("Earnings Pred.", lambda sc: "Yes" if (sc.get("earnings") or {}).get("is_predictable") else "No"),
        ("Debt Rating", lambda sc: (sc.get("debt") or {}).get("rating", "N/A").capitalize()),
    ]

    for label, extractor in metric_rows:
        row = [label] + [extractor(sc) for sc in scorecards]
        fund_table.add_row(*row)

    console.print(fund_table)
    return console.file.getvalue()


def margin_chart(data: Dict[str, Any]) -> str:
    """ASCII line chart of gross profit margin trend."""
    margins = data.get("margins")
    if not margins:
        return "No margin data available."

    years = margins.get("years", [])
    gpms = margins.get("gross_profit_margins", [])

    valid = [(y, g) for y, g in zip(years, gpms) if g is not None]
    if not valid:
        return "No valid margin data."

    ys, gs = zip(*valid)
    pct_gs = [g * 100 for g in gs]

    plt.clear_figure()
    plt.plot(list(ys), list(pct_gs), marker="braille")
    plt.title("Gross Profit Margin Trend")
    plt.xlabel("Year")
    plt.ylabel("GPM %")
    plt.plotsize(60, 15)

    return plt.build()


def eps_chart(data: Dict[str, Any]) -> str:
    """ASCII bar chart of EPS history."""
    eps_data = data.get("eps_history", {})
    years = eps_data.get("years", [])
    eps_list = eps_data.get("eps", [])

    valid = [(y, e) for y, e in zip(years, eps_list) if e is not None]
    if not valid:
        return "No EPS data available."

    ys, es = zip(*valid)

    plt.clear_figure()
    plt.bar(list(ys), list(es))
    plt.title("EPS History")
    plt.xlabel("Year")
    plt.ylabel("EPS ($)")
    plt.plotsize(60, 15)

    return plt.build()


# --- Formatting helpers ---

def _fmt_price(val) -> str:
    if val is None:
        return "N/A"
    return f"${val:,.2f}"


def _fmt_large(val) -> str:
    if val is None:
        return "N/A"
    if abs(val) >= 1e12:
        return f"${val / 1e12:.2f}T"
    if abs(val) >= 1e9:
        return f"${val / 1e9:.1f}B"
    if abs(val) >= 1e6:
        return f"${val / 1e6:.1f}M"
    return f"${val:,.0f}"


def _fmt_multiple(val) -> str:
    if val is None:
        return "N/A"
    return f"{val:.1f}x"


def _fmt_pct(val) -> str:
    if val is None:
        return "N/A"
    return f"{val * 100:.1f}%"


def _fmt_ratio(val) -> str:
    if val is None:
        return "N/A"
    # Handle yfinance D/E as percentage
    if val > 10:
        val = val / 100
    return f"{val:.2f}x"


def _fmt_range(low, high) -> str:
    if low is None or high is None:
        return "N/A"
    return f"${low:.0f}-${high:.0f}"


def _avg_mos(sc: Dict[str, Any]) -> str:
    ivs = sc.get("intrinsic_values", [])
    if not ivs:
        return "N/A"
    import statistics
    avg = statistics.mean([iv["margin_of_safety_pct"] for iv in ivs])
    return f"{avg:+.1f}%"


CHART_TYPES = {
    "metrics_table": metrics_table,
    "valuation_table": valuation_table,
    "comparison_table": comparison_table,
    "margin_chart": margin_chart,
    "eps_chart": eps_chart,
}


def main():
    parser = argparse.ArgumentParser(description="Generate visualizations")
    parser.add_argument(
        "--type",
        required=True,
        choices=list(CHART_TYPES.keys()),
        help="Chart/table type",
    )
    parser.add_argument("--data", required=True, help="JSON data")
    args = parser.parse_args()

    data = json.loads(args.data)
    func = CHART_TYPES[args.type]
    print(func(data))


if __name__ == "__main__":
    main()
