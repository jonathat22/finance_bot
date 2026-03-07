---
name: compare
description: Compare two or more companies on valuation, margins, moat, and fundamentals. Use when the user asks to compare stocks side by side.
argument-hint: <ticker1> <ticker2> [<ticker3>]
user-invocable: true
allowed-tools: Bash(python *), Read
---

# Company Comparison Skill

Compare multiple companies on value investing metrics.

## Steps

For each ticker:
1. `python -m finance_bot.data --ticker <TICKER>`
2. `python -m finance_bot.valuation --ticker <TICKER> --data '<json>'`

Then optionally:
3. `python -m finance_bot.visualize --type comparison_table --data '<combined_json>'`

## Output Format

Render as a side-by-side comparison:

```
COMPARATIVE VALUE ASSESSMENT
=============================
<TICKER1> vs <TICKER2> [vs <TICKER3>]

FUNDAMENTALS
--------------------------------------------------
 Metric          | <T1>      | <T2>      | <T3>
 Price           | $X        | $X        | $X
 Market Cap      | $X        | $X        | $X
 P/E (TTM)       | Xx        | Xx        | Xx
 EPS             | $X        | $X        | $X
 Revenue (TTM)   | $X        | $X        | $X
 Net Margin      | X%        | X%        | X%
 D/E             | X         | X         | X
--------------------------------------------------

VALUATION
--------------------------------------------------
 DCF Value       | $X        | $X        | $X
 Graham Number   | $X        | $X        | $X
 Owner Earnings  | $X        | $X        | $X
 Avg MOS         | X%        | X%        | X%
--------------------------------------------------

MOAT & QUALITY
--------------------------------------------------
 Moat Rating     | Wide      | Narrow    | None
 Earnings Pred.  | HIGH      | MEDIUM    | LOW
 Debt Rating     | Conserv.  | Moderate  | Aggress.
 Overall Rating  | BUY       | HOLD      | AVOID
--------------------------------------------------

RELATIVE ASSESSMENT
<2-3 sentences on which company offers better value and why>
```

## Tone

- Professional, comparative analysis style
- Anchor to intrinsic value and margin of safety
- Clearly state which company is more attractively valued
- No emojis
