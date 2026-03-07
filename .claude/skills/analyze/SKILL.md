---
name: analyze
description: Run value investing analysis on a company. Use when the user asks about a specific stock, whether it's undervalued, or wants financial metrics.
argument-hint: <ticker> [--full]
user-invocable: true
allowed-tools: Bash(python *), Read
---

# Value Analysis Skill

Run a comprehensive value investing analysis on a single company.

## Steps

1. **Fetch data**: `python -m finance_bot.data --ticker <TICKER>`
2. **Run valuation**: `python -m finance_bot.valuation --ticker <TICKER> --data '<json_from_step_1>'`
3. **If `--full` flag**: Also run:
   - `python -m finance_bot.research --query "<company> competitive advantages moat durability"`
   - `python -m finance_bot.visualize --type metrics_table --data '<json_from_step_1>'`
   - `python -m finance_bot.visualize --type valuation_table --data '<json_from_step_2>'`

## Output Format

Render the JSON results as a professional equity research note:

```
<COMPANY NAME> (<TICKER>) — Value Assessment
=============================================

Investment Thesis: <RATING>
<2-3 sentence verdict>

CURRENT METRICS
---------------------------------------------------------------
 Price | $X    Market Cap | $X    P/E (TTM) | Xx
 ...
---------------------------------------------------------------

INTRINSIC VALUE ESTIMATES
---------------------------------------------------------------
 Method           | Value/Share | Current  | Margin of Safety
 DCF              | $X          | $X       | X% (over/undervalued)
 Graham Number    | $X          | $X       | X%
 Owner Earnings   | $X          | $X       | X%
---------------------------------------------------------------

MOAT ASSESSMENT: <WIDE/NARROW/NONE>
 ...

EARNINGS CONSISTENCY
 ...

KEY STRENGTHS                      KEY RISKS
 - ...                              - ...
```

## Tone

- Professional equity research analyst
- Lead with the investment thesis
- Use precise financial terminology
- No emojis
- State conviction level (High/Medium/Low Confidence)
- Flag assumptions explicitly
