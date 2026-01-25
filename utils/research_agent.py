"""
Research Agent for financial questions using Gemini with Google Search grounding.
Provides fact-based answers with citations from reliable sources.
"""

import os
import random
import re
import time
from dataclasses import dataclass
from typing import List, Tuple

from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig, GoogleSearch, Tool

from .models import Citation, ResearchAnswer

load_dotenv()

# System prompt for research-focused responses
RESEARCH_SYSTEM_PROMPT = """
You are a financial research assistant that provides accurate, fact-based answers 
to questions about companies, markets, investments, and financial topics.

Your approach:
1. Search for current, reliable information from authoritative sources
2. Prioritize SEC filings, company investor relations, and reputable financial news
3. Provide specific facts with dates/periods when discussing financial data
4. Cite your sources clearly so users can verify information
5. Acknowledge uncertainty when information is unclear or conflicting

Response format:
- Start with a direct answer to the question
- Provide supporting details and context
- Include relevant numbers, dates, and facts
- End with source citations in the format: [Source: Title - URL]

For follow-up questions, use the conversation context to provide coherent, connected answers.

Always prioritize accuracy over completeness. If you're not certain about something, say so.
"""


def exponential_backoff_retry(func, max_retries: int = 5, base_delay: float = 2.0):
    """
    Retry a function with exponential backoff for rate limiting.
    """
    last_exception = None

    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_exception = e
            error_str = str(e).lower()

            if (
                "429" in str(e)
                or "resource_exhausted" in error_str
                or "quota" in error_str
            ):
                if attempt == max_retries - 1:
                    raise Exception(
                        "API quota exhausted. Please wait a moment and try again."
                    )

                delay = base_delay * (2**attempt) + random.uniform(0, 1)
                time.sleep(delay)
            else:
                raise e

    raise last_exception if last_exception else Exception("Max retries exceeded")


@dataclass
class ConversationMessage:
    """A message in the conversation history."""

    role: str  # 'user' or 'assistant'
    content: str


class ResearchAgent:
    """
    Research agent that uses Gemini with Google Search grounding
    to answer financial questions with citations.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.1,
        verbose: bool = False,
    ):
        """
        Initialize the research agent.

        Args:
            model: Gemini model to use
            temperature: Temperature for response generation (low = more factual)
            verbose: Whether to print debug information
        """
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = model
        self.temperature = temperature
        self.verbose = verbose
        self.search_tool = Tool(google_search=GoogleSearch())

    def _log(self, message: str):
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            print(f"[ResearchAgent] {message}")

    def _build_context(self, conversation_history: List[dict] = None) -> str:
        """
        Build context string from conversation history.

        Args:
            conversation_history: List of message dicts with 'role' and 'content'

        Returns:
            Formatted context string
        """
        if not conversation_history:
            return ""

        # Include last 6 messages for context
        recent = (
            conversation_history[-6:]
            if len(conversation_history) > 6
            else conversation_history
        )

        context_parts = ["Previous conversation:"]
        for msg in recent:
            role = "User" if msg.get("role") == "user" else "Assistant"
            context_parts.append(f"{role}: {msg.get('content', '')}")

        return "\n".join(context_parts) + "\n\n"

    def _extract_citations_from_grounding(self, response) -> List[Citation]:
        """
        Extract citations from Gemini's grounding metadata.

        Args:
            response: The full response object from Gemini

        Returns:
            List of Citation objects
        """
        citations = []

        try:
            # Check if response has grounding_metadata
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]

                if (
                    hasattr(candidate, "grounding_metadata")
                    and candidate.grounding_metadata
                ):
                    grounding_metadata = candidate.grounding_metadata

                    # Extract web search results from grounding chunks
                    if hasattr(grounding_metadata, "grounding_chunks"):
                        for chunk in grounding_metadata.grounding_chunks:
                            if hasattr(chunk, "web") and chunk.web:
                                web = chunk.web
                                url = getattr(web, "uri", None)
                                title = getattr(web, "title", None)

                                if url:
                                    citations.append(
                                        Citation(
                                            title=title
                                            or self._get_title_from_url(url),
                                            url=url,
                                            snippet=None,
                                            source_type=self._categorize_source(url),
                                        )
                                    )

                    # Also check for web_search_queries for additional context
                    if hasattr(grounding_metadata, "web_search_queries"):
                        self._log(
                            f"Search queries used: {grounding_metadata.web_search_queries}"
                        )

        except Exception as e:
            self._log(f"Error extracting grounding metadata: {e}")

        return citations

    def _extract_citations(self, response_text: str) -> Tuple[str, List[Citation]]:
        """
        Extract citations from the response text and remove citation markers.

        Args:
            response_text: The raw response from Gemini

        Returns:
            Tuple of (cleaned_text, list_of_citations)
        """
        citations = []
        cleaned_text = response_text

        # Pattern 1: [Source: Title - URL] format
        pattern1 = r"\[Source:\s*([^\]]+?)\s*-\s*(https?://[^\]]+)\]"
        matches1 = re.findall(pattern1, response_text)
        for title, url in matches1:
            citations.append(
                Citation(
                    title=title.strip(),
                    url=url.strip(),
                    snippet=None,
                    source_type=self._categorize_source(url),
                )
            )
        # Remove [Source: ...] markers from cleaned text
        cleaned_text = re.sub(pattern1, "", cleaned_text)

        # Pattern 2: [Title](URL) markdown links
        pattern2 = r"\[([^\]]+)\]\((https?://[^\)]+)\)"
        matches2 = re.findall(pattern2, response_text)
        for title, url in matches2:
            # Avoid duplicates
            if not any(c.url == url for c in citations):
                citations.append(
                    Citation(
                        title=title.strip(),
                        url=url.strip(),
                        snippet=None,
                        source_type=self._categorize_source(url),
                    )
                )

        # Pattern 3: Plain URLs
        pattern3 = r"(?<!\()(https?://[^\s\)\]]+)"
        matches3 = re.findall(pattern3, response_text)
        for url in matches3:
            # Avoid duplicates
            if not any(c.url == url for c in citations):
                citations.append(
                    Citation(
                        title=self._get_title_from_url(url),
                        url=url.strip(),
                        snippet=None,
                        source_type=self._categorize_source(url),
                    )
                )

        # Remove trailing "Sources:" or "References:" sections with citation lists
        # This captures common patterns like "Sources:\n- [link](url)\n- [link](url)"
        sources_section_pattern = r"\n*(?:---\s*)?\n*\*{0,2}(?:Sources?|References?|Citations?):?\*{0,2}\s*\n+(?:[-*•]?\s*\[?[^\n]+\]?\(?https?://[^\s\)]+\)?[^\n]*\n*)+$"
        cleaned_text = re.sub(
            sources_section_pattern, "", cleaned_text, flags=re.IGNORECASE
        )

        # Remove standalone plain URLs on their own lines
        cleaned_text = re.sub(
            r"^\s*https?://[^\s]+\s*$", "", cleaned_text, flags=re.MULTILINE
        )

        # Clean up extra whitespace and trailing newlines
        cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
        cleaned_text = cleaned_text.strip()

        return cleaned_text, citations

    def _categorize_source(self, url: str) -> str:
        """Categorize a source based on its URL."""
        url_lower = url.lower()

        # Extract domain for better matching
        domain_match = re.search(r"https?://(?:www\.)?([^/]+)", url_lower)
        domain = domain_match.group(1) if domain_match else url_lower

        # SEC Filing - most specific first
        if "sec.gov" in url_lower or "edgar" in url_lower:
            return "SEC Filing"

        # Company IR pages
        elif any(
            pattern in url_lower
            for pattern in [
                "/investor",
                "/ir/",
                "investor.",
                "ir.",
                "/shareholders",
                "/investors",
                "investors.",
                "corporate.",
                "/corporate",
            ]
        ):
            return "Company IR"

        # Government sources
        elif ".gov" in domain or "federalreserve" in domain:
            return "Government"

        # Financial News - check domain names
        elif any(
            news_domain in domain
            for news_domain in [
                "reuters.com",
                "bloomberg.com",
                "wsj.com",
                "cnbc.com",
                "yahoo.com",
                "finance.yahoo",
                "marketwatch.com",
                "financialtimes.com",
                "ft.com",
                "barrons.com",
                "fool.com",
                "morningstar.com",
                "seekingalpha.com",
                "marketbeat.com",
                "nasdaq.com",
                "investing.com",
                "cnn.com",
                "forbes.com",
                "businessinsider.com",
                "bloomberg",
                "reuters",
                "wsj",
                "cnbc",
            ]
        ):
            return "Financial News"

        # Research Reports - check for research domains or PDF patterns
        elif any(
            pattern in url_lower
            for pattern in ["/research", "/reports", ".pdf", "research.", "report."]
        ) and any(
            research_domain in domain
            for research_domain in [
                "goldman",
                "morgan",
                "jpmorgan",
                "wellsfargo",
                "citi.com",
                "bofa",
                "mckinsey",
                "deloitte",
            ]
        ):
            return "Research Report"

        else:
            return "Other"

    def _get_title_from_url(self, url: str) -> str:
        """Extract a readable title from a URL."""
        # Remove protocol and www
        title = re.sub(r"https?://(www\.)?", "", url)
        # Get domain and first path segment
        parts = title.split("/")
        if len(parts) > 1 and parts[1]:
            return f"{parts[0]} - {parts[1][:50]}"
        return parts[0]

    def answer_question(
        self, question: str, conversation_history: List[dict] = None
    ) -> ResearchAnswer:
        """
        Answer a financial question using web search.

        Args:
            question: The user's question
            conversation_history: List of previous messages for context

        Returns:
            ResearchAnswer with answer text and citations
        """
        self._log(f"Answering question: {question}")

        # Build context from conversation history
        context = self._build_context(conversation_history)

        # Construct the prompt
        prompt = f"{context}Current question: {question}"

        def make_request():
            response = self.client.models.generate_content(
                model=self.model,
                config=GenerateContentConfig(
                    system_instruction=RESEARCH_SYSTEM_PROMPT,
                    tools=[self.search_tool],
                    temperature=self.temperature,
                ),
                contents=prompt,
            )
            return response  # Return full response object

        # Make the request with retry logic
        response = exponential_backoff_retry(make_request)
        response_text = response.text

        self._log(f"Received response: {len(response_text)} characters")

        # Extract citations from grounding metadata (primary) and text (fallback)
        citations = self._extract_citations_from_grounding(response)

        if not citations:
            # Fallback to text parsing if no grounding metadata
            self._log("No grounding metadata citations found, trying text parsing")
            _, citations = self._extract_citations(response_text)

        # Clean the response text (remove citation markers if present)
        cleaned_text, _ = self._extract_citations(response_text)

        self._log(f"Extracted {len(citations)} citations")

        return ResearchAnswer(
            answer=cleaned_text,
            citations=citations,
            confidence="high" if citations else "medium",
        )

    def answer_question_simple(
        self, question: str, conversation_history: List[dict] = None
    ) -> str:
        """
        Simple method that returns just the answer text.

        Args:
            question: The user's question
            conversation_history: Previous conversation messages

        Returns:
            The answer text as a string
        """
        result = self.answer_question(question, conversation_history)
        return result.answer


def get_research_response(
    question: str,
    conversation_history: List[dict] = None,
    include_citations: bool = True,
) -> str:
    """
    Convenience function to get a research response.

    Args:
        question: The user's question
        conversation_history: Previous conversation messages
        include_citations: Whether to format citations in the response

    Returns:
        Formatted response string
    """
    agent = ResearchAgent()
    result = agent.answer_question(question, conversation_history)

    if include_citations and result.citations:
        # Format citations at the end
        citation_text = "\n\n---\n**Sources:**\n"
        for i, citation in enumerate(result.citations, 1):
            citation_text += f"{i}. [{citation.title}]({citation.url})"
            if citation.source_type:
                citation_text += f" ({citation.source_type})"
            citation_text += "\n"

        return result.answer + citation_text

    return result.answer
