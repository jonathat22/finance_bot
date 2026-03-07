---
name: research
description: Research a company, sector, or investment topic using web search with a value investing lens. Use when the user wants current news, competitive analysis, or qualitative assessment.
argument-hint: <query>
user-invocable: true
allowed-tools: Bash(python *), Read
---

# Value Research Skill

Research a financial topic with a value investing lens.

## Steps

1. `python -m finance_bot.research --query "<user_query>"`
2. Parse the JSON ResearchAnswer (answer + citations)

## Output Format

Present findings as a research note:

```
RESEARCH NOTE: <Topic>
======================

<Answer text, formatted with headers and bullet points as appropriate>

---
Sources:
 [1] <Title> — <domain> (<source_type>)
 [2] ...
```

## Query Enhancement

The research module automatically enhances queries for value investing context, but you should also frame the query to focus on:
- Durable competitive advantages
- Capital allocation quality
- Management track record
- Earnings power sustainability
- Industry structure and competitive dynamics

## Tone

- Analytical and evidence-based
- Cite sources with numbered references
- Distinguish between facts and opinions
- Flag uncertainty where appropriate
- No emojis
