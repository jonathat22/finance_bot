"""
Financial Research Chatbot
A conversational interface for financial research using Gemini with Google Search grounding.
"""

import streamlit as st
from dotenv import load_dotenv

from utils.models import Citation, ResearchAnswer
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
    by_type = {}
    for citation in citations:
        source_type = citation.source_type or "Other"
        if source_type not in by_type:
            by_type[source_type] = []
        by_type[source_type].append(citation)

    # Display citations grouped by type
    for source_type, cites in by_type.items():
        badge = get_source_badge_color(source_type)
        with st.expander(
            f"{badge} **{source_type}** ({len(cites)} source{'s' if len(cites) > 1 else ''})",
            expanded=True,
        ):
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


def main():
    st.set_page_config(
        page_title="Financial Research Chatbot", page_icon="💹", layout="wide"
    )

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Header
    st.title("Financial Research Chatbot 💹")
    st.markdown(
        "Ask any question about companies, markets, or financial topics. I'll search the web for accurate, up-to-date information."
    )

    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown(
            "This chatbot uses AI-powered web search to answer your financial questions "
            "with information from reliable sources."
        )

        st.subheader("Example questions:")
        st.markdown(
            """
- What is Apple's current market cap?
- How did Tesla perform in Q3 2024?
- Compare Microsoft vs Google revenue
- What are the latest Fed interest rate decisions?
- Explain P/E ratio
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

        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.caption("Powered by Gemini with Google Search")

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
                            prompt, conversation_history=st.session_state.messages[:-1]
                        )

                        # Display answer
                        st.markdown(result.answer)

                        # Display citations with enhanced formatting
                        display_citations(result.citations)

                        # Store structured data in message history
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": result.answer,  # For legacy compatibility
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


if __name__ == "__main__":
    main()
