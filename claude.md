# Finance Bot - Project Context

This document provides persistent context about the Finance Bot project for AI assistants to maintain consistency across conversations.

## Project Overview

**Name**: Finance Bot
**Purpose**: A conversational financial research chatbot that uses AI-powered web search to answer questions about companies, markets, and financial topics with citations from reliable sources.

**Current Status**: Active development. Core research functionality is complete with recent bug fixes for source categorization.

## Technology Stack

### Core Technologies
- **Language**: Python 3.10+ (supports 3.10, 3.11, 3.12)
- **UI Framework**: Streamlit
- **AI/LLM**: Google Gemini 2.5-flash with Google Search grounding
- **Data Models**: Pydantic v2
- **Environment Management**: python-dotenv

### Development Tools
- **Testing**: pytest with coverage reporting
- **Linting**: flake8
- **Formatting**: Black (line length: 88)
- **Import Sorting**: isort (Black-compatible profile)
- **Type Checking**: mypy
- **Security**: bandit, safety

## Project Structure

```
finance_bot/
├── streamlit_app.py          # Main Streamlit UI application
├── utils/                     # Core business logic
│   ├── __init__.py
│   ├── models.py             # Pydantic data models (Citation, ResearchAnswer, SourceType)
│   └── research_agent.py     # ResearchAgent class with Google Search integration
├── tests/                     # Test suite
│   ├── __init__.py
│   ├── conftest.py           # pytest fixtures
│   ├── test_models.py        # Model tests
│   └── test_research_agent.py # Agent tests
├── .github/
│   └── workflows/
│       └── ci.yml            # CI/CD pipeline (test + security)
├── requirements.txt          # Production dependencies
├── requirements-dev.txt      # Development dependencies
├── pyproject.toml           # Python project config (Black, isort)
├── pytest.ini               # pytest configuration
├── IMPLEMENTATION_SUMMARY.md # Implementation notes
├── README.md                # Project README
└── .env                     # Environment variables (not in git)
```

## Core Components

### 1. ResearchAgent (`utils/research_agent.py`)

**Purpose**: Answers financial questions using Gemini with Google Search grounding.

**Key Features**:
- Google Search integration via Gemini API
- Citation extraction from grounding metadata
- Source categorization (SEC Filing, Company IR, Financial News, etc.)
- URL normalization (handles redirects, AMP URLs, tracking parameters)
- Query enhancement for financial keywords
- Conversation history support

**Important Methods**:
- `answer_question(question, conversation_history)` - Main entry point
- `_extract_clean_domain(url)` - Normalizes URLs (Google redirects, AMP, tracking params)
- `_categorize_source(url)` - Categorizes sources by URL patterns
- `_enhance_query_for_financials(question)` - Adds search hints for financial queries
- `_extract_citations_from_grounding(response)` - Extracts citations from Gemini metadata
- `_extract_citations(response_text)` - Fallback text-based citation extraction

**Source Categories** (with color badges):
- 🟦 SEC Filing (sec.gov, edgar)
- 🟩 Company IR (investor relations pages)
- 🟨 Financial News (38+ recognized domains)
- 🟪 Government (.gov domains, Federal Reserve)
- 🟧 Research Report (investment bank research)
- ⬜ Other (unrecognized sources)

**Financial News Domains**: Comprehensive set of 38 domains including reuters.com, bloomberg.com, wsj.com, cnbc.com, yahoo.com, seekingalpha.com, morningstar.com, fool.com, and many others.

### 2. Data Models (`utils/models.py`)

**SourceType** (Enum):
- SEC_FILING, COMPANY_IR, NEWS, RESEARCH, GOVERNMENT, OTHER

**Citation** (Pydantic model):
- title, url, snippet (optional), source_type (optional)

**ResearchAnswer** (Pydantic model):
- answer, citations (list), confidence (optional)

**FinancialDocument** (Pydantic model):
- url, title, source_type, document_type, company_name, year

**DocumentSearchResult** (Pydantic model):
- company_identified, documents_found, search_summary

### 3. Streamlit App (`streamlit_app.py`)

**Purpose**: Interactive chat interface for financial research.

**Key Features**:
- Chat history with session state
- Source citations grouped by type with color badges
- Expandable citation sections
- Sidebar with examples and source type legend
- Error handling and retry logic

**Important Functions**:
- `get_source_badge_color(source_type)` - Maps source types to emoji badges
- `display_citations(citations)` - Groups and displays citations by type
- `display_message_with_citations(message)` - Renders chat messages with citations
- `main()` - Main Streamlit app entry point

## Development Workflow

### Git Workflow

**Main Branches**:
- `main` - Production-ready code
- `develop` - Integration branch for features

**Feature Development**:
1. Create feature branch from `develop`
2. Name format: `feature/description` or `fix/description`
3. Make changes with atomic commits
4. Run tests locally before pushing
5. Create PR to `develop` (not `main`)
6. CI/CD must pass before merge
7. Merge to `develop`, then eventually to `main`

**Commit Messages**:
- Use conventional commits format
- Include co-authorship: `Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>`
- Focus on "why" rather than "what"

### Code Quality Standards

**Formatting** (enforced by CI):
- Black formatter (line length: 88)
- isort for import sorting (Black profile)
- flake8 for linting (max complexity: 10)

**Type Hints**:
- Required for all function signatures
- Use `from typing import` for generic types
- mypy type checking in CI

**Testing**:
- pytest for all tests
- Aim for high coverage (tracked with codecov)
- Tests in `tests/` directory mirror source structure
- Use fixtures in `conftest.py` for reusable test setup

**Security**:
- bandit for security scanning
- safety for dependency vulnerability checking
- Never commit API keys (use .env)

### Running the Application

**Local Development**:
```bash
# Install dependencies
pip install -r requirements-dev.txt

# Run app
streamlit run streamlit_app.py

# Run tests
pytest tests/ --cov=utils --cov=streamlit_app

# Format code
black utils streamlit_app.py tests
isort utils streamlit_app.py tests

# Lint
flake8 utils streamlit_app.py --max-line-length=88

# Type check
mypy utils streamlit_app.py --ignore-missing-imports
```

**Environment Variables** (in `.env`):
```
GEMINI_API_KEY=your_api_key_here
```

## Recent Important Changes

### Source Color Coding Fix (2026-01)

**Problem**: All internet sources were being categorized as "Other" (white badge) instead of being properly categorized.

**Root Causes**:
1. Google Search returns redirect URLs (`google.com/url?q=...`) and AMP URLs that didn't match domain patterns
2. Limited financial news domain list (only ~24 domains)
3. No active search prioritization for financial statements

**Solution** (in `research_agent.py`):
1. Added `_extract_clean_domain()` to handle:
   - Google redirect URLs
   - AMP URLs (both `/amp/s/` and `/amp/` formats)
   - Tracking parameters
   - Subdomain extraction (e.g., `finance.yahoo.com` → `yahoo.com`)

2. Expanded `FINANCIAL_NEWS_DOMAINS` to 38 domains

3. Updated `RESEARCH_SYSTEM_PROMPT` to explicitly prioritize:
   - SEC filings (10-K, 10-Q, 8-K)
   - Company investor relations
   - Official financial statements

4. Added `_enhance_query_for_financials()` to append search hints for financial queries

**Testing**: All URL categorization tests pass (8/8) including Google redirects, AMP URLs, and domain extraction.

## System Prompts & AI Behavior

### Research System Prompt

The ResearchAgent uses a carefully crafted system prompt that:
1. Prioritizes official financial documents (SEC filings, IR pages)
2. Guides search queries to include terms like "10-K", "sec.gov edgar", "investor relations"
3. Emphasizes accuracy over completeness
4. Requires clear source citations
5. Acknowledges uncertainty when appropriate

### Query Enhancement

For queries containing financial keywords (revenue, earnings, profit, etc.), the system automatically appends:
```
(Prioritize SEC filings, 10-K/10-Q reports, and official investor relations sources)
```

This guides Google Search to prefer authoritative sources.

## Testing Strategy

### Unit Tests
- Model validation (`test_models.py`)
- Research agent functionality (`test_research_agent.py`)
- URL categorization and domain extraction

### Integration Tests
- End-to-end research queries
- Citation extraction from real responses
- Source categorization accuracy

### Manual Testing Checklist
1. Run app with verbose mode: `ResearchAgent(verbose=True)`
2. Test queries:
   - "What is Apple's revenue?" → Should find SEC filings or Yahoo Finance
   - "Tesla quarterly earnings Q3 2024" → Should find 10-Q or earnings reports
   - "Microsoft market cap" → Should find Financial News sources
3. Verify color coding:
   - SEC filings: 🟦 Blue
   - Company IR: 🟩 Green
   - Financial News: 🟨 Yellow (should be common)
   - Government: 🟪 Purple
   - Research Reports: 🟧 Orange
   - Other: ⬜ White (should be rare)

## Common Patterns

### Adding a New Source Domain
1. Add domain to `FINANCIAL_NEWS_DOMAINS` set in `research_agent.py`
2. Use base domain format (e.g., `"example.com"` not `"www.example.com"`)
3. Test with sample URL to verify categorization

### Modifying Source Categories
1. Update `SourceType` enum in `models.py`
2. Update color mapping in `streamlit_app.py` `get_source_badge_color()`
3. Update categorization logic in `research_agent.py` `_categorize_source()`
4. Update sidebar legend in `streamlit_app.py`

### Adding New Features
1. Create feature branch from `develop`
2. Add tests first (TDD approach preferred)
3. Implement feature
4. Update documentation (this file, README, IMPLEMENTATION_SUMMARY)
5. Ensure CI passes
6. Create PR to `develop`

## Known Limitations

1. **Google Search Dependency**: Relies on Gemini's Google Search grounding, which may have rate limits
2. **URL-based Categorization**: Source categorization relies on URL patterns, may miss new domains
3. **No PDF Processing**: Currently only extracts citations, doesn't process PDF content
4. **English-focused**: Financial news domains primarily English-language sources

## Future Enhancements

Based on IMPLEMENTATION_SUMMARY.md, planned features include:
- Document download & caching
- PDF processing pipeline
- Enhanced document analysis
- Report generation
- Multi-document comparison

## Important Notes for AI Assistants

1. **Always use Black formatting** (88 character line length)
2. **Never commit API keys** - use .env file
3. **Feature branches off `develop`**, not `main`
4. **Run tests before committing** - CI will catch failures anyway
5. **Verbose mode helpful for debugging** - `ResearchAgent(verbose=True)`
6. **Source categorization is URL-based** - domain extraction is critical
7. **Keep FINANCIAL_NEWS_DOMAINS updated** - add new legitimate sources as discovered
8. **Financial queries get auto-enhanced** - check `_enhance_query_for_financials()`
9. **Citation extraction has fallbacks** - grounding metadata → text parsing
10. **Session state in Streamlit** - messages, company, documents stored there

## Contact & Resources

- **Repository**: /Users/Jonathan/Desktop/finance_bot
- **Main Branch**: main
- **Development Branch**: develop
- **CI/CD**: GitHub Actions (.github/workflows/ci.yml)
- **Python Version**: 3.10+ (supports 3.10, 3.11, 3.12)

---

**Last Updated**: 2026-01-25
**Version**: 1.0
