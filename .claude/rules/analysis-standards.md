---
paths:
  - "finance_bot/**/*.py"
---

# Value Investing Analysis Standards

When implementing or modifying analysis code in finance_bot/:

- Conservative assumptions by default: 10% discount rate floor, growth capped at 8%
- Always compute margin of safety: (intrinsic - current) / intrinsic
- Moat assessment required for any buy or strong_buy rating
- Earnings consistency check: minimum 5 years of positive EPS data for "predictable"
- Debt analysis required: flag any D/E > 2.0 as "aggressive" or "dangerous"
- DCF uses 10-year projection horizon with 3% terminal growth rate
- Graham Number: N/A if EPS or book value per share is negative
- Owner Earnings: net income + depreciation/amortization - average 3yr capex
- Rating thresholds:
  - strong_buy: >50% margin of safety AND wide moat
  - buy: >30% margin of safety
  - hold: <30% margin of safety or narrow moat
  - avoid: negative margin of safety AND weak fundamentals (high debt or no moat)
