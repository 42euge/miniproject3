# Oguz's Implementation — MP3

## Baseline Prompt
```
You are a financial analyst assistant. Answer the user's question
to the best of your knowledge. You have no access to external tools,
live data, or databases. If you are unsure about specific current values,
say so clearly.
```

## Single Agent Prompt
Detailed system prompt with explicit tool-use rules:
- Lists all 7 tools with when-to-use guidance
- Critical rules: sector questions → lookup tickers first, then fetch data
- Comparison questions → fetch ALL tickers
- Multi-condition → fetch BOTH time periods, filter and compare
- Trust tool data, report errors explicitly, never guess values

## Multi-Agent Architecture: Orchestrator → Parallel Specialists → Critic → Synthesizer

```text
User Question
     │
 Orchestrator ── analyzes question, selects specialists, detects ranking dependency
     │
     ├─ Single-phase: ThreadPoolExecutor([selected agents]) ───────────┐
     └─ Two-phase:  Phase 1 agent → extract tickers → Phase 2 parallel ┤
                                                                       ▼
                                                                    Critic
                                                                       │
                                                                  Synthesizer
                                                                       │
                                                                    Answer
```

### 5 Stages:

1. **Orchestrator** (1 LLM call, no tools) — analyzes question, selects specialists, detects ranking dependencies. Returns JSON with agent list, phased flag, and per-agent sub-tasks.

2. **Specialists** (3 agents, fixed tool subsets):
   - Price Agent: get_tickers_by_sector, get_price_performance, get_market_status, get_top_gainers_losers
   - Fundamentals Agent: get_company_overview, query_local_db, get_tickers_by_sector
   - Sentiment Agent: get_news_sentiment, query_local_db

3. **Execution Mode**:
   - Single-phase (default): all specialists run concurrently via ThreadPoolExecutor
   - Two-phase (ranking dependency): Phase 1 runs alone → extract tickers → Phase 2 parallel

4. **Critic** (1 LLM call per specialist, no tools) — checks internal consistency between answer and raw tool data, produces confidence score (0.0–1.0) and issues_found list.

5. **Synthesizer** (1 LLM call, no tools) — merges specialist answers. Table format for multi-ticker comparisons, prose for others. Low-confidence results noted explicitly.

### LLM Call Budget
- Single-phase, 1 specialist: 4 calls
- Single-phase, 2–3 specialists: 6–8 calls
- Two-phase, 3 specialists: 8 calls
