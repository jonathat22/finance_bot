# Value Investing Framework

Reference document for valuation methodology used by the finance_bot system.

## DCF (Discounted Cash Flow)

- **Base**: Most recent trailing twelve months free cash flow (operating CF - capex)
- **Growth rate**: min(historical FCF CAGR, 8%) — never project above 8%
- **Discount rate**: 10% (WACC floor for conservative estimate)
- **Terminal growth**: 3% (roughly long-term GDP growth)
- **Projection horizon**: 10 years
- **Formula**: Sum of discounted future FCFs + terminal value, divided by shares outstanding

## Graham Number

- **Formula**: sqrt(22.5 * EPS * BVPS)
- **EPS**: Trailing twelve months earnings per share
- **BVPS**: Book value per share (total equity / shares outstanding)
- **Applicability**: Only valid when both EPS > 0 and BVPS > 0
- **Origin**: Benjamin Graham's criteria from The Intelligent Investor
- **Note**: Best suited for asset-heavy, stable-earnings companies

## Owner Earnings (Buffett Method)

- **Formula**: (Net Income + Depreciation & Amortization - Average 3yr CapEx) / discount rate / shares outstanding
- **Source**: Warren Buffett's 1986 Berkshire Hathaway shareholder letter
- **Discount rate**: 10% (same conservative floor)
- **Rationale**: Represents cash that could be extracted without impairing the business

## Moat Classification

### Wide Moat
- Gross margin stability: std dev < 3% over 5+ years
- Consistent competitive advantage gap > 10%
- At least 2 identifiable moat sources

### Narrow Moat
- Gross margin stability: std dev < 5%
- Competitive advantage gap > 5%
- At least 1 identifiable moat source

### No Moat
- Volatile margins or negative competitive advantage gap
- No identifiable structural advantages

### Moat Sources
- **Brand**: Pricing power above commodity level
- **Switching costs**: High customer retention, integration depth
- **Network effects**: Value increases with user base
- **Cost advantage**: Structural cost leadership (scale, process, location)
- **Intangible assets**: Patents, licenses, regulatory barriers

## Margin of Safety Thresholds

| MOS Range | Interpretation |
|-----------|---------------|
| > 50% | Deep value — strong_buy candidate (if moat is wide) |
| 30-50% | Attractive value — buy candidate |
| 10-30% | Fair value — hold |
| 0-10% | Fully valued — hold with caution |
| < 0% | Overvalued — avoid for new positions |

## Earnings Consistency Criteria

- **Predictable**: Coefficient of variation < 0.30, positive streak >= 5 years
- **Moderate**: CV < 0.50 or positive streak >= 3 years
- **Unpredictable**: CV >= 0.50 or positive streak < 3 years
- **CAGR**: Compound annual growth rate of EPS over available history

## Debt Safety Thresholds

| D/E Ratio | Current Ratio | Interest Coverage | Rating |
|-----------|--------------|-------------------|--------|
| < 0.5 | > 2.0 | > 8x | Conservative |
| 0.5 - 1.5 | 1.5 - 2.0 | 4x - 8x | Moderate |
| 1.5 - 2.5 | 1.0 - 1.5 | 2x - 4x | Aggressive |
| > 2.5 | < 1.0 | < 2x | Dangerous |

## Rating System

- **strong_buy**: MOS > 50% AND wide moat AND predictable earnings AND conservative/moderate debt
- **buy**: MOS > 30% AND at least narrow moat
- **hold**: MOS < 30% OR no moat assessment available
- **avoid**: Negative MOS AND (dangerous debt OR no moat OR unpredictable earnings)
