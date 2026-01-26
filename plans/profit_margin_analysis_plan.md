# Gross Profit Margin Analysis Feature

Add gross profit margin analysis to report generation with competitive advantage visualization.

## Overview

**Goal**: Plot gross profit margin vs operating expense ratio over time, emphasizing competitive advantage (margin > expenses consistently).

**Limitation**: yfinance provides ~5 years of annual income statement data (not 10). Plan proceeds with available data.

**Calculations**:
- Gross Profit Margin = Gross Profit / Total Revenue
- Operating Expense Ratio = Operating Expense / Total Revenue
- Competitive Advantage Gap = Gross Margin - Op Expense Ratio (positive = advantage)

---

## Phase 1: Data Model

**File**: [models.py](../utils/models.py)

Add `HistoricalMarginData` model after `HistoricalData` (line ~137):

```python
class HistoricalMarginData(BaseModel):
    """Historical margin data for competitive advantage analysis."""

    ticker: str
    company_name: str
    years: List[int]  # [2024, 2023, 2022...]
    gross_profit_margins: List[Optional[float]]  # decimals
    operating_expense_ratios: List[Optional[float]]
    net_margins: List[Optional[float]]
    competitive_advantage_gaps: List[Optional[float]]  # GPM - OER
    data_timestamp: datetime

    @property
    def has_consistent_advantage(self) -> bool:
        """True if gap > 0 for all available years (min 3)."""
        gaps = [g for g in self.competitive_advantage_gaps if g is not None]
        return len(gaps) >= 3 and all(g > 0 for g in gaps)

    @property
    def average_advantage_gap(self) -> Optional[float]:
        """Mean gap across years."""
        gaps = [g for g in self.competitive_advantage_gaps if g is not None]
        return sum(gaps) / len(gaps) if gaps else None
```

Update `ChartSpec.chart_type` Literal to include `"margin_trend"`, `"advantage_bar"`.

---

## Phase 2: Data Fetching

**File**: [financial_data.py](../utils/financial_data.py)

Add `_margin_cache` dict in `__init__` (line ~94).

Add `get_historical_margins(ticker)` method:

```python
def get_historical_margins(self, ticker: str) -> Tuple[Optional[HistoricalMarginData], Optional[str]]:
    """Fetch historical margin data from income statement."""
    # Check cache
    # Fetch stock.income_stmt via yfinance
    # Extract: Total Revenue, Gross Profit, Operating Expense, Net Income
    # Calculate: GPM, OER, Net Margin, Advantage Gap for each year
    # Return HistoricalMarginData or error
```

Key yfinance fields: `stock.income_stmt` DataFrame with index rows `'Total Revenue'`, `'Gross Profit'`, `'Operating Expense'`, `'Net Income'`.

---

## Phase 3: Chart Generation

**File**: [chart_generator.py](../utils/chart_generator.py)

Add import for `HistoricalMarginData` model.

### 3.1 `create_margin_trend_chart()`

Line chart showing GPM (solid) vs OER (dashed) over years:
- For single company: shade area between lines (competitive advantage gap)
- For multi-company: overlay lines with color coding
- Y-axis: Percentage (%)
- X-axis: Fiscal Year

### 3.2 `create_competitive_advantage_bar()`

Bar chart comparing average advantage gap across companies:
- Green bars for positive gaps (has advantage)
- Red bars for negative gaps (no advantage)
- Horizontal line at y=0 for reference
- Star annotation for companies with consistent advantage

---

## Phase 4: Report Integration

**File**: [report_agent.py](../utils/report_agent.py)

### 4.1 Update `CHART_SELECTION_PROMPT`

Add new chart types:
- `"margin_trend"` - GPM vs OER over time (competitive advantage)
- `"advantage_comparison"` - Compare advantage gaps across companies

Guide: Include `margin_trend` for queries about margins, profitability, moats, competitive advantage.

### 4.2 Update `_build_chart_specs()`

Add cases for new chart types:
```python
elif chart_type == "margin_trend":
    specs.append(ChartSpec(
        chart_type="margin_trend",
        title="Gross Profit Margin vs Operating Expenses",
        data={"companies": tickers, "show_gap": len(tickers) == 1}
    ))
elif chart_type == "advantage_comparison":
    specs.append(ChartSpec(
        chart_type="advantage_bar",
        title="Competitive Advantage Comparison",
        data={"companies": tickers}
    ))
```

### 4.3 Update `render_charts()`

Add rendering logic:
- Fetch margin data via `get_historical_margins()` for each company
- Call appropriate chart generator method
- Handle missing data gracefully

### 4.4 Update `_generate_analysis()`

Enhance AI prompt with margin context:
```
Historical margin analysis (5-year):
- AAPL: Avg Gross Margin=45.2%, Consistent Advantage=Yes, Avg Gap=30.1%
- MSFT: Avg Gross Margin=69.8%, Consistent Advantage=Yes, Avg Gap=28.5%

When analyzing competitive advantage:
- Company has advantage when gross margin consistently exceeds operating expenses
- Larger positive gaps = stronger pricing power or cost efficiency
- Consistent gaps over time = durable competitive moat
```

---

## Phase 5: Testing

**New file**: `tests/test_margin_analysis.py`

### Test cases:

1. **HistoricalMarginData model**
   - `has_consistent_advantage` True/False scenarios
   - `average_advantage_gap` calculation
   - Handling of None values

2. **get_historical_margins()**
   - Success with mocked income_stmt DataFrame
   - Empty data handling
   - Cache behavior

3. **create_margin_trend_chart()**
   - Single company with gap shading
   - Multi-company overlay
   - Correct axis labels

4. **create_competitive_advantage_bar()**
   - Color coding (green/red)
   - Star annotation for consistent advantage

**Update existing**: `tests/test_chart_generator.py` - add fixtures for HistoricalMarginData

---

## Files to Modify

| File | Changes |
|------|---------|
| [utils/models.py](../utils/models.py) | Add `HistoricalMarginData`, update `ChartSpec` |
| [utils/financial_data.py](../utils/financial_data.py) | Add `_margin_cache`, `get_historical_margins()` |
| [utils/chart_generator.py](../utils/chart_generator.py) | Add `create_margin_trend_chart()`, `create_competitive_advantage_bar()` |
| [utils/report_agent.py](../utils/report_agent.py) | Update prompts, `_build_chart_specs()`, `render_charts()`, `_generate_analysis()` |
| tests/test_margin_analysis.py | New test file |

---

## Verification

1. **Unit tests**: `pytest tests/test_margin_analysis.py -v`

2. **Manual test queries**:
   - "Analyze Apple's competitive advantage" → margin_trend chart with gap shading
   - "Compare Apple and Microsoft profitability" → margin_trend + advantage_comparison charts
   - "Which company has a stronger moat: Coca-Cola or Pepsi?" → advantage analysis in text + charts

3. **Check chart output**:
   - GPM line (solid) above OER line (dashed) = advantage
   - Green shaded area between lines for single company
   - Bar chart shows positive (green) vs negative (red) gaps

4. **Run full test suite**: `pytest tests/ --cov=utils`
