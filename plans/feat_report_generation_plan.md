# Financial Report Mode - Implementation Plan

**Data Source**: yfinance (free, no API key)
**Charts**: All types (comparison bars, price trends, valuation scatter)

## Overview
Add a "Report Mode" to the Finance Bot that generates comparative financial analysis with visualizations. User enters query like "Analyze Apple and Google - which is more undervalued?" and gets structured metrics + charts + analysis.

## Architecture

```
New Files:
├── utils/financial_data.py    # yfinance wrapper for structured data
├── utils/report_agent.py      # Orchestrates report generation
├── utils/chart_generator.py   # Plotly chart creation
└── tests/test_*.py            # Tests for each new module

Modified Files:
├── utils/models.py            # Add CompanyMetrics, ComparisonReport models
├── streamlit_app.py           # Add mode toggle + report display UI
└── requirements.txt           # Add yfinance, plotly, pandas
```

## New Dependencies
```
yfinance>=0.2.36    # Free Yahoo Finance API (no key needed)
plotly>=5.18.0      # Interactive charts (native Streamlit support)
pandas>=2.0.0       # Data manipulation
```

## New Models (`utils/models.py`)

```python
class CompanyMetrics(BaseModel):
    ticker: str
    company_name: str
    market_cap: Optional[float]
    pe_ratio: Optional[float]
    forward_pe: Optional[float]
    eps: Optional[float]
    revenue_ttm: Optional[float]
    revenue_growth: Optional[float]
    profit_margin: Optional[float]
    debt_to_equity: Optional[float]
    current_price: Optional[float]
    fifty_two_week_high: Optional[float]
    fifty_two_week_low: Optional[float]
    data_timestamp: datetime

class ChartSpec(BaseModel):
    chart_type: Literal["bar", "line", "scatter", "grouped_bar"]
    title: str
    data: Dict[str, Any]

class ComparisonReport(BaseModel):
    query: str
    companies: List[CompanyMetrics]
    analysis_text: str
    charts: List[ChartSpec]
    citations: List[Citation]
    generated_at: datetime
```

## Core Components

### 1. FinancialDataProvider (`utils/financial_data.py`)
- Wraps yfinance API with caching (5 min TTL)
- `get_company_metrics(ticker)` → CompanyMetrics
- `get_historical_data(ticker, period)` → price history
- `resolve_company_to_ticker(name)` → ticker symbol
- Handles rate limits with retry logic

### 2. ChartGenerator (`utils/chart_generator.py`)
- `create_metric_comparison_bar()` - Compare metrics across companies
- `create_price_history_line()` - Historical prices (normalized to 100)
- `create_valuation_scatter()` - P/E vs Growth scatter plot

### 3. ReportAgent (`utils/report_agent.py`)
- Main entry: `generate_report(query) → ComparisonReport`
- Uses Gemini to extract company names from query
- Fetches metrics via FinancialDataProvider
- Generates analysis text via existing ResearchAgent
- Determines relevant charts based on query

### 4. Streamlit UI Changes (`streamlit_app.py`)
- Sidebar radio: "Chat" / "Report" mode toggle
- Report mode: text input + "Generate Report" button
- Display: company metric cards, plotly charts, analysis, citations

## Implementation Phases

### Phase 1: Data Infrastructure ✅
- [x] Add dependencies to requirements.txt
- [x] Create `utils/financial_data.py` with FinancialDataProvider
- [x] Add new models to `utils/models.py`
- [x] Unit tests for data fetching (15 tests passing)

### Phase 2: Chart Generation ✅
- [x] Create `utils/chart_generator.py`
- [x] Implement bar, line, scatter chart generators
- [x] Unit tests with sample data (29 tests passing)

### Phase 3: Report Agent ✅
- [x] Create `utils/report_agent.py`
- [x] Company extraction via Gemini (JSON output)
- [x] Report generation flow
- [x] Integration with ResearchAgent for analysis
- [x] Integration tests (22 tests passing)

### Phase 4: Streamlit UI ✅
- [x] Add mode toggle to sidebar
- [x] Report input UI
- [x] `display_report()` and `display_company_card()` functions
- [x] Loading states and error handling

### Phase 5: Polish ✅
- [x] Handle unknown tickers gracefully (TickerValidationResult, validate_ticker())
- [x] Improve error messages (detailed errors with tips)
- [x] Edge case handling (single company, partial failures, connection errors)

## Data Flow

```
User Query: "Compare Apple and Google's financials"
    ↓
ReportAgent._extract_companies_from_query()  # Gemini → ["AAPL", "GOOGL"]
    ↓
FinancialDataProvider.get_company_metrics()  # yfinance → CompanyMetrics
    ↓
ChartGenerator.create_*()                    # plotly figures
    ↓
ResearchAgent.answer_question()              # Web-grounded analysis + citations
    ↓
ComparisonReport                             # Final structured output
    ↓
Streamlit display_report()                   # Render UI
```

## Key Files to Modify

| File | Changes |
|------|---------|
| `utils/models.py` | Add CompanyMetrics, ChartSpec, ComparisonReport |
| `streamlit_app.py` | Add mode toggle, report UI, display functions |
| `requirements.txt` | Add yfinance, plotly, pandas |

## Verification

1. **Unit tests**: Run `pytest tests/ -v` for all new modules
2. **Manual test**:
   - Start app: `streamlit run streamlit_app.py`
   - Switch to Report mode
   - Query: "Compare Apple and Microsoft financials"
   - Verify: metrics cards, charts render, analysis has citations
3. **Edge cases**: Test invalid ticker, single company, missing metrics
