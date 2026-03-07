---
paths:
  - "finance_bot/tests/**/*.py"
---

# Testing Standards

- Use pytest for all tests
- Mock all external API calls (yfinance, Gemini) — tests must not hit the network
- Use fixtures in conftest.py for reusable test setup
- Target >85% code coverage for finance_bot/ modules
- Test edge cases: missing data, negative values, zero division, None fields
- Test model validation: ensure Pydantic models reject invalid data
- Name test files test_<module>.py to mirror source structure
