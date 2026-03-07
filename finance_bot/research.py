"""ResearchAgent wrapper: Value-investing-enhanced web search.

Wraps utils/research_agent.py with value investing query enhancement.
"""

import argparse
import json
import sys
from typing import Optional

from finance_bot.utils.research_agent import ResearchAgent as BaseResearchAgent


VALUE_INVESTING_CONTEXT = (
    "Focus on: durable competitive advantages (economic moat), "
    "capital allocation quality, management track record, "
    "earnings power sustainability, return on invested capital, "
    "and margin of safety relative to intrinsic value."
)


class ValueResearchAgent:
    """Research agent with value investing lens."""

    def __init__(self, verbose: bool = False):
        self._agent = BaseResearchAgent(verbose=verbose)
        self._verbose = verbose

    def research(self, query: str, conversation_history=None) -> dict:
        """Run a value-investing-enhanced research query.

        Returns dict with 'answer', 'citations', 'confidence'.
        """
        enhanced_query = f"{query}\n\n({VALUE_INVESTING_CONTEXT})"

        result = self._agent.answer_question(enhanced_query, conversation_history)

        citations = [
            {
                "title": c.title,
                "url": c.url,
                "snippet": c.snippet,
                "source_type": c.source_type,
            }
            for c in result.citations
        ]

        return {
            "answer": result.answer,
            "citations": citations,
            "confidence": result.confidence,
        }


def main():
    parser = argparse.ArgumentParser(description="Value investing research")
    parser.add_argument("--query", required=True, help="Research query")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    agent = ValueResearchAgent(verbose=args.verbose)
    result = agent.research(args.query)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
