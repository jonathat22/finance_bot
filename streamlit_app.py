"""
Financial Research Chatbot.

A conversational interface for financial research using Gemini
with Google Search grounding.
"""

import streamlit as st
from dotenv import load_dotenv

from utils.models import Citation, CompanyMetrics, ComparisonReport, ResearchAnswer
from utils.report_agent import ReportAgent
from utils.research_agent import ResearchAgent

# Load environment variables
load_dotenv()


def get_source_badge_color(source_type: str) -> str:
    """Get color for source type badge."""
    colors = {
        "SEC Filing": "🟦",
        "Company IR": "🟩",
        "Financial News": "🟨",
        "Government": "🟪",
        "Research Report": "🟧",
        "Other": "⬜",
    }
    return colors.get(source_type, "⬜")


def display_citations(citations: list[Citation]):
    """Display citations with enhanced formatting and expandable sections."""
    if not citations:
        return

    st.markdown("---")
    st.markdown("### 📚 Sources")

    # Group citations by source type
    by_type: dict[str, list[Citation]] = {}
    for citation in citations:
        source_type = citation.source_type or "Other"
        if source_type not in by_type:
            by_type[source_type] = []
        by_type[source_type].append(citation)

    # Display citations grouped by type
    for source_type, cites in by_type.items():
        badge = get_source_badge_color(source_type)
        suffix = "s" if len(cites) > 1 else ""
        label = f"{badge} **{source_type}** ({len(cites)} source{suffix})"
        with st.expander(label, expanded=True):
            for i, citation in enumerate(cites, 1):
                col1, col2 = st.columns([0.95, 0.05])
                with col1:
                    # Display citation with title and URL
                    st.markdown(f"**{i}. [{citation.title}]({citation.url})**")

                    # Display snippet if available
                    if citation.snippet:
                        st.caption(f"_{citation.snippet}_")

                    # Display URL in small text
                    st.caption(f"🔗 {citation.url}")

                if i < len(cites):
                    st.divider()


def display_message_with_citations(message: dict):
    """Display a message with properly formatted citations."""
    if message["role"] == "user":
        st.markdown(message["content"])
    else:
        # For assistant messages, check if we have structured data
        if isinstance(message.get("data"), dict) and "answer" in message["data"]:
            # Display answer
            st.markdown(message["data"]["answer"])

            # Display citations if available
            if message["data"].get("citations"):
                display_citations(message["data"]["citations"])
        else:
            # Fallback for legacy messages
            st.markdown(message["content"])


def format_large_number(value: float | None) -> str:
    """Format large numbers for display (e.g., $1.5T, $500B)."""
    if value is None:
        return "N/A"
    if value >= 1e12:
        return f"${value / 1e12:.2f}T"
    if value >= 1e9:
        return f"${value / 1e9:.2f}B"
    if value >= 1e6:
        return f"${value / 1e6:.2f}M"
    return f"${value:,.0f}"


def format_percent(value: float | None) -> str:
    """Format decimal as percentage."""
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def display_company_card(company: CompanyMetrics):
    """Display a company metrics card."""
    with st.container():
        st.subheader(f"{company.company_name} ({company.ticker})")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Price",
                f"${company.current_price:.2f}" if company.current_price else "N/A",
            )
            st.metric("Market Cap", format_large_number(company.market_cap))
            st.metric(
                "P/E Ratio", f"{company.pe_ratio:.2f}" if company.pe_ratio else "N/A"
            )
            st.metric(
                "Forward P/E",
                f"{company.forward_pe:.2f}" if company.forward_pe else "N/A",
            )

        with col2:
            st.metric("Revenue (TTM)", format_large_number(company.revenue_ttm))
            st.metric("Revenue Growth", format_percent(company.revenue_growth))
            st.metric("Profit Margin", format_percent(company.profit_margin))
            st.metric("EPS", f"${company.eps:.2f}" if company.eps else "N/A")

        with col3:
            st.metric(
                "52W High",
                (
                    f"${company.fifty_two_week_high:.2f}"
                    if company.fifty_two_week_high
                    else "N/A"
                ),
            )
            st.metric(
                "52W Low",
                (
                    f"${company.fifty_two_week_low:.2f}"
                    if company.fifty_two_week_low
                    else "N/A"
                ),
            )
            st.metric(
                "Debt/Equity",
                f"{company.debt_to_equity:.2f}" if company.debt_to_equity else "N/A",
            )
            st.metric("Dividend Yield", format_percent(company.dividend_yield))

        st.caption(f"Data as of {company.data_timestamp.strftime('%Y-%m-%d %H:%M')}")


def display_report(report: ComparisonReport, report_agent: ReportAgent):
    """Display a full comparison report with metrics, charts, and analysis."""
    # Adjust header based on number of companies
    if len(report.companies) == 1:
        st.header(f"Financial Report: {report.companies[0].company_name}")
    else:
        st.header("Financial Comparison Report")

    st.caption(f"Generated {report.generated_at.strftime('%Y-%m-%d %H:%M')}")

    # Check for error reports (no companies)
    if not report.companies:
        st.error(report.analysis_text)
        return

    # Company metrics cards
    st.subheader("Company Metrics")
    # Limit columns to max 3 for readability
    num_cols = min(len(report.companies), 3)
    cols = st.columns(num_cols)
    for i, company in enumerate(report.companies):
        with cols[i % num_cols]:
            display_company_card(company)

    st.divider()

    # Charts
    if report.charts:
        st.subheader("Visualizations")
        figures = report_agent.render_charts(report)
        for fig in figures:
            st.plotly_chart(fig, use_container_width=True)

        st.divider()

    # Analysis text
    st.subheader("Analysis")
    st.markdown(report.analysis_text)

    # Citations
    if report.citations:
        display_citations(report.citations)


def main():
    st.set_page_config(
        page_title="Financial Research Chatbot", page_icon="💹", layout="wide"
    )

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "mode" not in st.session_state:
        st.session_state.mode = "Chat"
    if "current_report" not in st.session_state:
        st.session_state.current_report = None

    # Sidebar
    with st.sidebar:
        st.header("Mode")
        mode = st.radio(
            "Select mode:",
            ["Chat", "Report"],
            index=0 if st.session_state.mode == "Chat" else 1,
            horizontal=True,
            help="Chat: Ask questions. Report: Generate comparative analysis.",
        )
        st.session_state.mode = mode

        st.divider()

        if mode == "Chat":
            st.subheader("Example questions:")
            st.markdown(
                """
- What is Apple's current market cap?
- How did Tesla perform in Q3 2024?
- Compare Microsoft vs Google revenue
- What are the latest Fed interest rate decisions?
        """
            )
        else:
            st.subheader("Example reports:")
            st.markdown(
                """
- Compare Apple and Microsoft financials
- Analyze Tesla vs Rivian - which is better valued?
- NVDA vs AMD vs INTC comparison
- Is Meta overvalued compared to Google?
        """
            )

        st.divider()

        # Citation legend
        st.subheader("Source Types:")
        st.markdown(
            """
🟦 SEC Filing
🟩 Company IR
🟨 Financial News
🟪 Government
🟧 Research Report
⬜ Other
        """
        )

        st.divider()

        if mode == "Chat":
            if st.button("🗑️ Clear Conversation", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        else:
            if st.button("🗑️ Clear Report", use_container_width=True):
                st.session_state.current_report = None
                st.rerun()

        st.divider()
        st.caption("Powered by Gemini with Google Search")

    # Main content area - changes based on mode
    if st.session_state.mode == "Chat":
        # Header for Chat mode
        st.title("Financial Research Chatbot 💹")
        st.markdown("Ask any question about companies, markets, or financial topics.")

        # Main chat area
        chat_container = st.container()

        # Display chat messages
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    display_message_with_citations(message)

        # Chat input
        if prompt := st.chat_input("Ask a financial question..."):
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Display user message
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)

            # Generate and display assistant response
            with chat_container:
                with st.chat_message("assistant"):
                    with st.spinner("🔍 Researching..."):
                        try:
                            # Get structured response from agent
                            agent = ResearchAgent()
                            result: ResearchAnswer = agent.answer_question(
                                prompt,
                                conversation_history=st.session_state.messages[:-1],
                            )

                            # Display answer
                            st.markdown(result.answer)

                            # Display citations with enhanced formatting
                            display_citations(result.citations)

                            # Store structured data in message history
                            st.session_state.messages.append(
                                {
                                    "role": "assistant",
                                    "content": result.answer,
                                    "data": {
                                        "answer": result.answer,
                                        "citations": result.citations,
                                        "confidence": result.confidence,
                                    },
                                }
                            )
                        except Exception as e:
                            error_msg = f"Sorry, I encountered an error: {str(e)}"
                            st.error(error_msg)
                            st.session_state.messages.append(
                                {"role": "assistant", "content": error_msg}
                            )

    else:
        # Report mode
        st.title("Financial Comparison Report 📊")
        st.markdown(
            "Enter companies to compare and get detailed metrics, charts, and analysis."
        )

        # Report input form
        with st.form("report_form"):
            query = st.text_input(
                "What would you like to analyze?",
                placeholder="e.g., Compare Apple and Microsoft financials",
            )
            submitted = st.form_submit_button(
                "📊 Generate Report", use_container_width=True
            )

        if submitted and query:
            # Validate input
            if len(query.strip()) < 3:
                st.warning("Please enter a longer query with company names or tickers.")
            else:
                with st.spinner("🔄 Fetching data and generating report..."):
                    try:
                        report_agent = ReportAgent(verbose=False)
                        report = report_agent.generate_report(query)
                        st.session_state.current_report = report
                        st.session_state.report_agent = report_agent
                    except ConnectionError:
                        st.error(
                            "🌐 **Connection Error**\n\n"
                            "Could not connect to data provider. "
                            "Please check your internet connection and try again."
                        )
                        st.session_state.current_report = None
                    except TimeoutError:
                        st.error(
                            "⏱️ **Timeout Error**\n\n"
                            "The request took too long. Please try again."
                        )
                        st.session_state.current_report = None
                    except Exception as e:
                        error_msg = str(e)
                        if "API" in error_msg or "key" in error_msg.lower():
                            st.error(
                                "🔑 **API Error**\n\n"
                                "There was an issue with the API. "
                                "Please check your GEMINI_API_KEY in the .env file."
                            )
                        else:
                            st.error(
                                f"❌ **Error generating report**\n\n{error_msg}\n\n"
                                "**Try:**\n"
                                "• Using ticker symbols (AAPL, MSFT) instead of names\n"
                                "• Checking your internet connection\n"
                                "• Trying a different query"
                            )
                        st.session_state.current_report = None

        # Display current report
        if st.session_state.current_report:
            st.divider()
            report = st.session_state.current_report
            report_agent = st.session_state.get("report_agent") or ReportAgent()
            display_report(report, report_agent)


if __name__ == "__main__":
    main()
