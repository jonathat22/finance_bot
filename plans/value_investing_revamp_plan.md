# Revamp Finance Bot: Claude Code-Native Multi-Agent Value Investing System

## Context

The current finance bot is a Streamlit web app for financial Q&A and comparative reports. The goal is to rebuild it as a **Claude Code CLI-native** system where Python scripts run under the hood, results are displayed directly in the Claude Code interface, and the user interacts entirely through Claude Code's conversational CLI. The system follows a **value investing philosophy** (Buffett/Graham) and uses Claude Code features — skills, hooks, rules, and reference files — to create a polished, professional experience.

**Key decisions:**
- Interaction entirely through Claude Code CLI (no separate UI)
- Three valuation methods: DCF, Graham Number, Owner Earnings
- Interactive session with follow-ups
- Quick analysis by default, full depth on request
- Professional equity research analyst communication style
- Leverage Claude Code skills, hooks, rules, and reference files

## Architecture Overview

Claude Code is the orchestrator. The user asks financial questions in Claude Code, which calls Python modules via Bash, receives structured JSON, and renders professional-grade markdown output. Custom **skills** provide slash commands (`/analyze`, `/compare`, `/research`). **Hooks** enforce code quality. **Rules** govern communication style and analysis standards. **Reference files** supply persistent value investing frameworks.

```
User (Claude Code CLI)
  |
  |-- /analyze AAPL        (skill invocation)
  |-- "Is Apple undervalued?"  (natural language)
  |
  v
Claude Code (orchestrator, guided by CLAUDE.md + rules + skills)
  |
  |-- python -m finance_bot.data --ticker AAPL
  |-- python -m finance_bot.valuation --ticker AAPL --data <json>
  |-- python -m finance_bot.research --query "..."
  |-- python -m finance_bot.visualize --type margin_chart --data <json>
  |
  v
Structured JSON output -> Claude Code renders as professional markdown
```

## Project Structure

```
finance_bot/
├── __init__.py
├── __main__.py                     # python -m finance_bot dispatcher
├── data.py                         # DataAgent: fetch & compute financial data
├── valuation.py                    # ValuationAgent: DCF, Graham, Owner Earnings
├── research.py                     # ResearchAgent: value-investing web search
├── visualize.py                    # ASCII charts (plotext) + rich tables
├── models/
│   ├── __init__.py
│   ├── core.py                     # Migrated from utils/models.py
│   └── valuation.py                # New value investing models
├── utils/
│   ├── __init__.py
│   ├── financial_data.py           # KEEP AS-IS (yfinance wrapper)
│   └── research_agent.py           # KEEP AS-IS (Gemini + Google Search)
├── tests/
│   ├── conftest.py
│   ├── test_data.py
│   ├── test_valuation.py
│   ├── test_visualize.py
│   └── test_models_valuation.py
├── .claude/
│   ├── skills/
│   │   ├── analyze/
│   │   │   └── SKILL.md            # /analyze <ticker> [--full]
│   │   ├── compare/
│   │   │   └── SKILL.md            # /compare <ticker1> <ticker2>
│   │   └── research/
│   │       └── SKILL.md            # /research <query>
│   ├── rules/
│   │   ├── communication.md        # Equity research analyst tone
│   │   ├── analysis-standards.md   # Value investing methodology
│   │   └── testing.md              # Test standards for finance_bot/
│   └── settings.json               # Hooks configuration
├── docs/
│   └── value-investing-framework.md  # Reference: valuation methodology
├── requirements.txt
├── pyproject.toml
└── CLAUDE.md                       # Master instructions for Claude Code
```

**Removed:** `streamlit_app.py`, `utils/chart_generator.py` (Plotly), `utils/report_agent.py`
**Kept as-is:** `utils/financial_data.py`, `utils/research_agent.py`
**Migrated:** `utils/models.py` -> `models/core.py`

## Claude Code Features

### Skills (`.claude/skills/`)

#### `/analyze` — Single Company Value Analysis
```yaml
---
name: analyze
description: Run value investing analysis on a company. Use when the user asks about a specific stock, whether it's undervalued, or wants financial metrics.
argument-hint: <ticker> [--full]
user-invocable: true
allowed-tools: Bash(python *), Read
---
```
Behavior: Calls data.py -> valuation.py -> (if --full) research.py + visualize.py. Renders a professional equity research note with valuation, moat assessment, and verdict.

#### `/compare` — Multi-Company Comparison
```yaml
---
name: compare
description: Compare two or more companies on valuation, margins, moat, and fundamentals. Use when the user asks to compare stocks side by side.
argument-hint: <ticker1> <ticker2> [<ticker3>]
user-invocable: true
allowed-tools: Bash(python *), Read
---
```
Behavior: Runs data + valuation for each ticker, renders side-by-side comparison tables and relative valuation assessment.

#### `/research` — Web Research with Value Lens
```yaml
---
name: research
description: Research a company, sector, or investment topic using web search with a value investing lens. Use when the user wants current news, competitive analysis, or qualitative assessment.
argument-hint: <query>
user-invocable: true
allowed-tools: Bash(python *), Read
---
```
Behavior: Calls research.py with value-investing-enhanced query, returns findings with citations.

Each skill's SKILL.md will contain detailed instructions on output formatting, the professional tone to use, and which Python modules to call in what order.

### Rules (`.claude/rules/`)

#### `communication.md` — Professional Equity Research Tone
```yaml
---
paths:
  - "**/*"
---
```
Content instructs Claude Code to communicate as a senior equity research analyst:
- Lead with the investment thesis, not methodology
- Use precise financial terminology (not casual language)
- Present data in structured formats (tables, bullet points)
- State conviction level clearly (High/Medium/Low Confidence)
- Flag assumptions and risks explicitly
- Use phrases like "Our analysis indicates...", "The current valuation implies...", "Key risk to our thesis..."
- Never use emojis; maintain professional register throughout
- When comparing, always anchor to intrinsic value and margin of safety

#### `analysis-standards.md` — Value Investing Methodology
```yaml
---
paths:
  - "finance_bot/**/*.py"
---
```
Governs how analysis is performed:
- Conservative assumptions by default (10% discount rate floor, growth capped at 8%)
- Always compute margin of safety
- Moat assessment required for any buy/strong_buy rating
- Earnings consistency check: minimum 5 years of data
- Debt analysis required: flag any D/E > 2.0

#### `testing.md` — Test Standards
```yaml
---
paths:
  - "finance_bot/tests/**/*.py"
---
```
Standard testing rules: pytest, mock external APIs, >85% coverage target.

### Hooks (`.claude/settings.json`)

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [
          {
            "type": "command",
            "command": "file_path=$(echo $TOOL_INPUT | python3 -c \"import sys,json; print(json.load(sys.stdin).get('file_path',''))\"); if [[ \"$file_path\" == *.py ]]; then python3 -m black --quiet \"$file_path\" 2>/dev/null; fi",
            "timeout": 10
          }
        ]
      }
    ]
  }
}
```
Auto-formats Python files with Black after every write/edit.

### Reference File: Value Investing Framework (`docs/value-investing-framework.md`)

Referenced from CLAUDE.md via `@docs/value-investing-framework.md`. Contains:
- DCF methodology with conservative assumptions
- Graham Number formula and applicability criteria
- Owner Earnings calculation (Buffett's 1986 letter method)
- Moat classification framework (wide/narrow/none)
- Margin of safety thresholds (>30% for buy, >50% for strong_buy)
- Earnings consistency criteria
- Debt safety thresholds
- Rating system definitions (strong_buy / buy / hold / avoid)

This file is loaded on-demand when Claude Code needs valuation context, keeping CLAUDE.md lean.

### CLAUDE.md (Updated)

Concise master instructions (~150 lines):
- Project overview: CLI-native value investing multi-agent system
- How to use skills: `/analyze`, `/compare`, `/research`
- How to call Python modules (command patterns, JSON I/O)
- Communication style: professional equity research analyst (defer to `@.claude/rules/communication.md`)
- Valuation methodology reference: `@docs/value-investing-framework.md`
- Code quality: Black, isort, pytest, type hints
- Git workflow: feature branches off `develop`

## New Value Investing Models (`models/valuation.py`)

```python
class IntrinsicValueEstimate(BaseModel):
    ticker: str
    method: Literal["dcf", "graham_number", "owner_earnings"]
    intrinsic_value_per_share: float
    current_price: float
    margin_of_safety_pct: float        # (intrinsic - current) / intrinsic
    assumptions: Dict[str, Any]        # method-specific params

class MoatAssessment(BaseModel):
    ticker: str
    moat_rating: Literal["wide", "narrow", "none"]
    moat_sources: List[str]            # e.g., ["brand", "switching_costs"]
    gross_margin_stability: float      # std dev of GPM
    avg_competitive_advantage_gap: Optional[float]
    summary: str

class EarningsConsistency(BaseModel):
    ticker: str
    eps_history: List[Optional[float]]
    years: List[int]
    earnings_growth_cagr: Optional[float]
    positive_earnings_streak: int
    coefficient_of_variation: Optional[float]
    is_predictable: bool

class DebtAnalysis(BaseModel):
    ticker: str
    debt_to_equity: Optional[float]
    current_ratio: Optional[float]
    interest_coverage: Optional[float]
    debt_to_earnings: Optional[float]
    rating: Literal["conservative", "moderate", "aggressive", "dangerous"]

class ValueScorecard(BaseModel):
    ticker: str
    company_name: str
    current_price: float
    intrinsic_values: List[IntrinsicValueEstimate]
    moat: Optional[MoatAssessment]
    earnings: Optional[EarningsConsistency]
    debt: Optional[DebtAnalysis]
    overall_rating: Literal["strong_buy", "buy", "hold", "avoid"]
    verdict: str                       # Concise investment thesis
    key_strengths: List[str]
    key_risks: List[str]
```

## Agent Details

### 1. DataAgent (`data.py`)
**Wraps:** `utils/financial_data.py` (`FinancialDataProvider`)
**Existing data:** CompanyMetrics, HistoricalMarginData, HistoricalPEData
**New computations:**
- 10-year EPS history (from `income_stmt`)
- Free cash flow (operating CF - capex, from `cash_flow` stmt)
- Book value per share (from `balance_sheet`)
- Interest coverage (EBIT / interest expense)
- ROIC (NOPAT / invested capital)
- Owner Earnings components (net income + D&A - avg capex)
- Shares outstanding (for per-share calculations)

**CLI:** `python -m finance_bot.data --ticker AAPL [--full]`
**Output:** JSON dict with all data keyed by category

### 2. ValuationAgent (`valuation.py`)
**DCF:** FCF base, min(historical growth, 8%) growth, 10% discount, 3% terminal, 10yr horizon
**Graham Number:** sqrt(22.5 * EPS * BVPS), N/A if either negative
**Owner Earnings:** (Net Income + D&A - 3yr avg CapEx) / 0.10 discount rate / shares outstanding
**Moat:** GPM std dev, competitive advantage gap, qualitative via Gemini
**Earnings:** 10yr EPS trend, CAGR, CV, streak
**Debt:** Categorize from D/E, current ratio, interest coverage
**Scorecard:** Composite rating with thresholds: strong_buy (>50% MOS + wide moat), buy (>30% MOS), hold (<30% MOS or narrow moat), avoid (negative MOS + weak fundamentals)

**CLI:** `python -m finance_bot.valuation --ticker AAPL --data <json>`
**Output:** JSON ValueScorecard

### 3. ResearchAgent (`research.py`)
**Wraps:** `utils/research_agent.py` (Gemini + Google Search grounding)
**Enhancement:** Value-investing query context — searches for durable competitive advantages, capital allocation quality, management track record, not just generic financial news.
**CLI:** `python -m finance_bot.research --query "..."`
**Output:** JSON ResearchAnswer with citations

### 4. VisualizationAgent (`visualize.py`)
**Libraries:** `plotext` (ASCII charts), `rich` (tables rendered to string)
**Chart types:** margin trends, P/E history, EPS growth bars, valuation comparison table, metrics summary table
**CLI:** `python -m finance_bot.visualize --type <chart_type> --data <json>`
**Output:** Plain text (ASCII charts + tables)

## Example Output (Professional Equity Research Style)

When the user runs `/analyze AAPL`, Claude Code renders:

```
APPLE INC. (AAPL) — Value Assessment
=====================================

Investment Thesis: HOLD
Apple possesses a wide economic moat supported by ecosystem lock-in and
brand strength, with highly predictable earnings. However, shares currently
trade above our conservative intrinsic value estimates across all three
methodologies, offering insufficient margin of safety for new positions.

CURRENT METRICS
---------------------------------------------------------------
 Price      | $187.44    Market Cap  | $2.89T     P/E (TTM)  | 29.3x
 EPS (TTM)  | $6.42      Revenue     | $383.3B    Net Margin | 26.3%
 D/E        | 1.73       Div Yield   | 0.5%       Fwd P/E    | 28.1x
---------------------------------------------------------------

INTRINSIC VALUE ESTIMATES
---------------------------------------------------------------
 Method           | Value/Share | Current  | Margin of Safety
 DCF              | $158.20     | $187.44  | -15.6% (overvalued)
 Graham Number    | $121.40     | $187.44  | -35.2% (overvalued)
 Owner Earnings   | $172.80     | $187.44  |  -7.8% (overvalued)
---------------------------------------------------------------
 Assumptions: 10% discount rate, 6.5% FCF growth, 3% terminal

MOAT ASSESSMENT: WIDE
 Sources: Ecosystem lock-in, brand pricing power, services recurring revenue
 GPM Stability: 0.8% std dev (5Y) — excellent consistency
 Avg Competitive Advantage Gap: +12.3%

EARNINGS CONSISTENCY
 10Y EPS CAGR: 14.2%  |  Positive Streak: 10 years
 Predictability: HIGH (CV: 0.18)
 Verdict: Passes Buffett earnings consistency screen

KEY STRENGTHS                      KEY RISKS
 - Wide moat, durable advantages    - Greater China revenue exposure
 - 10-year earnings growth track    - Regulatory/antitrust headwinds
 - Growing services mix (>22%)      - Hardware refresh cycle maturation
 - Strong FCF generation ($110B)    - Elevated valuation vs. history

Sources:
 [1] Apple Inc. 10-K FY2025 — sec.gov/cgi-bin/browse-edgar
 [2] Apple Services Revenue Analysis — reuters.com
```

## Dependencies

**Remove:** `streamlit`, `plotly`
**Add:** `rich>=13.0`, `plotext>=5.2`
**Keep:** `google-genai`, `yfinance>=0.2.36`, `pandas>=2.0.0`, `pydantic`, `python-dotenv`

## Phased Implementation

### Phase 1: Foundation + Claude Code Config
- Create `finance_bot/` package structure (models/, utils/, tests/)
- Migrate `utils/models.py` -> `models/core.py`
- Copy `utils/financial_data.py` and `utils/research_agent.py` into `finance_bot/utils/`
- Create `models/valuation.py` with all new Pydantic models
- Create `.claude/skills/` with analyze, compare, research SKILL.md files
- Create `.claude/rules/` with communication.md, analysis-standards.md, testing.md
- Create `.claude/settings.json` with hooks (Black auto-format)
- Create `docs/value-investing-framework.md` reference file
- Update `CLAUDE.md` with new instructions
- Update `requirements.txt`
- Tests for new models

### Phase 2: DataAgent
- Create `data.py` wrapping FinancialDataProvider
- Add new computations: 10yr EPS, FCF, BVPS, interest coverage, owner earnings components, shares outstanding
- JSON CLI interface (argparse, stdout)
- Tests with mocked yfinance

### Phase 3: ValuationAgent
- Create `valuation.py`
- Implement DCF, Graham Number, Owner Earnings
- Implement MoatAssessment (quantitative + Gemini qualitative)
- Implement EarningsConsistency, DebtAnalysis
- Composite ValueScorecard with rating logic
- Tests with sample data

### Phase 4: VisualizationAgent
- Create `visualize.py`
- ASCII charts: margin trends, P/E history, EPS growth
- Rich tables: metrics summary, valuation comparison, side-by-side company comparison
- All output as plain text strings
- Tests verifying output format

### Phase 5: ResearchAgent + Integration
- Create `research.py` with value-investing query enhancement
- Create `__main__.py` dispatcher
- End-to-end testing of full pipeline
- Verify skills work: `/analyze AAPL`, `/compare AAPL MSFT`, `/research "Apple moat"`

### Phase 6: Polish + Cleanup
- Multi-company comparison rendering
- Error handling (missing data, API failures, invalid tickers)
- Remove old files: `streamlit_app.py`, `utils/chart_generator.py`, `utils/report_agent.py`
- Final documentation pass

## Verification

1. **Unit tests:** `pytest finance_bot/tests/ --cov=finance_bot` for each module
2. **Module CLI tests:** Run each module individually, verify valid JSON output
   - `python -m finance_bot.data --ticker AAPL`
   - `python -m finance_bot.valuation --ticker AAPL --data "$(python -m finance_bot.data --ticker AAPL)"`
   - `python -m finance_bot.visualize --type margin_chart --data <json>`
   - `python -m finance_bot.research --query "Apple competitive advantages"`
3. **Skill tests:** In Claude Code, run `/analyze AAPL` and `/compare AAPL MSFT`, verify professional output format
4. **Hook test:** Edit a .py file, confirm Black auto-formats
5. **Rule test:** Verify Claude Code uses professional analyst tone (no emojis, structured output, investment thesis lead)

## Reusable Existing Code
- `utils/financial_data.py:91` — `FinancialDataProvider` class (caching, metrics, margins, P/E, EPS)
- `utils/financial_data.py:45` — `COMPANY_TICKER_MAP` (40+ name->ticker mappings)
- `utils/financial_data.py:419` — `_get_stmt_value()` (safe income statement extraction)
- `utils/research_agent.py:138` — `ResearchAgent` class (Gemini + Google Search)
- `utils/research_agent.py:99` — `exponential_backoff_retry()` (rate limit handling)
- `utils/models.py:95` — `CompanyMetrics` model
- `utils/models.py:139` — `HistoricalMarginData` model (with `has_consistent_advantage`, `average_advantage_gap`)
- `utils/models.py:175` — `HistoricalPEData` model

## Status

- [x] Plan written and saved
- [x] Phase 1: Foundation + Claude Code Config
- [x] Phase 2: DataAgent
- [x] Phase 3: ValuationAgent
- [x] Phase 4: VisualizationAgent
- [x] Phase 5: ResearchAgent + Integration
- [x] Phase 6: Polish + Cleanup

## Branch

Implementation branch: `feat/value-investing-revamp` (off `develop`)
