# agents.py — extracted and refactored from mp3_assignment.ipynb
#
# Key changes vs the notebook:
#   1. Absolute paths for stocks.db and .env (resolved via __file__)
#   2. ACTIVE_MODEL global removed — every public function accepts model: str explicitly
#   3. verbose defaults changed to False (no console in Streamlit)
#   4. re and ThreadPoolExecutor imports moved to top-level

import os, json, time, sqlite3, requests, textwrap, re
import pathlib
import pandas as pd
import yfinance as yf
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI

# ── Paths ─────────────────────────────────────────────────────
_MODULE_DIR   = pathlib.Path(__file__).parent          # streamlitapp/
_PROJECT_ROOT = _MODULE_DIR.parent                     # miniproject3/

load_dotenv(_PROJECT_ROOT / ".env")

OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY",       "")
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "")
DB_PATH              = str(_PROJECT_ROOT / "stocks.db")

MODEL_SMALL = "gpt-4o-mini"
MODEL_LARGE = "gpt-4o"

client = OpenAI(api_key=OPENAI_API_KEY)


# ── Tool functions ────────────────────────────────────────────

def get_price_performance(tickers: list, period: str = "1y") -> dict:
    """% price change for a list of tickers over a period.
    Valid periods: '1mo', '3mo', '6mo', 'ytd', '1y'"""
    results = {}
    for ticker in tickers:
        try:
            data = yf.download(ticker, period=period, progress=False, auto_adjust=True)
            if data.empty:
                results[ticker] = {"error": "No data — possibly delisted"}
                continue
            start = float(data["Close"].iloc[0].item())
            end   = float(data["Close"].iloc[-1].item())
            results[ticker] = {
                "start_price": round(start, 2),
                "end_price"  : round(end,   2),
                "pct_change" : round((end - start) / start * 100, 2),
                "period"     : period,
            }
        except Exception as e:
            results[ticker] = {"error": str(e)}
    return results


def get_market_status() -> dict:
    """Open / closed status for global stock exchanges."""
    return requests.get(
        f"https://www.alphavantage.co/query?function=MARKET_STATUS"
        f"&apikey={ALPHAVANTAGE_API_KEY}", timeout=10
    ).json()


def get_top_gainers_losers() -> dict:
    """Today's top gaining, top losing, and most active tickers."""
    return requests.get(
        f"https://www.alphavantage.co/query?function=TOP_GAINERS_LOSERS"
        f"&apikey={ALPHAVANTAGE_API_KEY}", timeout=10
    ).json()


def get_news_sentiment(ticker: str, limit: int = 5) -> dict:
    """Latest headlines + Bullish/Bearish/Neutral sentiment for a ticker."""
    data = requests.get(
        f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT"
        f"&tickers={ticker}&limit={limit}&apikey={ALPHAVANTAGE_API_KEY}", timeout=10
    ).json()
    return {
        "ticker": ticker,
        "articles": [
            {
                "title"    : a.get("title"),
                "source"   : a.get("source"),
                "sentiment": a.get("overall_sentiment_label"),
                "score"    : a.get("overall_sentiment_score"),
            }
            for a in data.get("feed", [])[:limit]
        ],
    }


def query_local_db(sql: str) -> dict:
    """Run any SQL SELECT on stocks.db.
    Table 'stocks' columns: ticker, company, sector, industry, market_cap, exchange
    market_cap values: 'Large' | 'Mid' | 'Small'"""
    try:
        conn = sqlite3.connect(DB_PATH)
        df   = pd.read_sql_query(sql, conn)
        conn.close()
        return {"columns": list(df.columns), "rows": df.to_dict(orient="records")}
    except Exception as e:
        return {"error": str(e)}


def get_company_overview(ticker: str) -> dict:
    """Fundamentals for one stock: P/E ratio, EPS, market cap, 52-week high/low."""
    url = (
        f"https://www.alphavantage.co/query?function=OVERVIEW"
        f"&symbol={ticker}&apikey={ALPHAVANTAGE_API_KEY}"
    )
    data = requests.get(url, timeout=10).json()
    if "Name" in data:
        return {
            "ticker"    : ticker,
            "name"      : data.get("Name", ""),
            "sector"    : data.get("Sector", ""),
            "pe_ratio"  : data.get("PERatio", ""),
            "eps"       : data.get("EPS", ""),
            "market_cap": data.get("MarketCapitalization", ""),
            "52w_high"  : data.get("52WeekHigh", ""),
            "52w_low"   : data.get("52WeekLow", ""),
        }
    # Alpha Vantage rate-limited or invalid ticker — fall back to yfinance
    try:
        info = yf.Ticker(ticker).info
        if not info.get("shortName") and not info.get("longName"):
            return {"error": f"No overview data for {ticker}"}
        return {
            "ticker"    : ticker,
            "name"      : info.get("longName") or info.get("shortName", ""),
            "sector"    : info.get("sector", ""),
            "pe_ratio"  : str(info.get("trailingPE", "")),
            "eps"       : str(info.get("trailingEps", "")),
            "market_cap": str(info.get("marketCap", "")),
            "52w_high"  : str(info.get("fiftyTwoWeekHigh", "")),
            "52w_low"   : str(info.get("fiftyTwoWeekLow", "")),
        }
    except Exception:
        return {"error": f"No overview data for {ticker}"}


def get_tickers_by_sector(sector: str) -> dict:
    """Return all stocks matching a sector or industry name from the local database."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT ticker, company, industry FROM stocks WHERE LOWER(sector) = LOWER(?)",
        conn, params=(sector,)
    )
    if df.empty:
        df = pd.read_sql_query(
            "SELECT ticker, company, industry FROM stocks WHERE LOWER(industry) LIKE LOWER(?)",
            conn, params=(f"%{sector}%",)
        )
    conn.close()
    return {"sector": sector, "stocks": df.to_dict(orient="records")}


# ── Schemas ───────────────────────────────────────────────────

def _s(name, desc, props, req):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": req}}}

SCHEMA_TICKERS  = _s("get_tickers_by_sector",
    "Return all stocks in a sector or industry from the local database. "
    "Use broad sector names ('Information Technology', 'Energy') or sub-sectors ('semiconductor', 'insurance').",
    {"sector": {"type": "string", "description": "Sector or industry name"}}, ["sector"])

SCHEMA_PRICE    = _s("get_price_performance",
    "Get % price change for a list of tickers over a time period. "
    "Periods: '1mo','3mo','6mo','ytd','1y'.",
    {"tickers": {"type": "array", "items": {"type": "string"}},
     "period":  {"type": "string", "default": "1y"}}, ["tickers"])

SCHEMA_OVERVIEW = _s("get_company_overview",
    "Get fundamentals for one stock: P/E ratio, EPS, market cap, 52-week high and low.",
    {"ticker": {"type": "string", "description": "Ticker symbol e.g. 'AAPL'"}}, ["ticker"])

SCHEMA_STATUS   = _s("get_market_status",
    "Check whether global stock exchanges are currently open or closed.", {}, [])

SCHEMA_MOVERS   = _s("get_top_gainers_losers",
    "Get today's top gaining, top losing, and most actively traded stocks.", {}, [])

SCHEMA_NEWS     = _s("get_news_sentiment",
    "Get latest news headlines and Bullish/Bearish/Neutral sentiment scores for a stock.",
    {"ticker": {"type": "string"}, "limit": {"type": "integer", "default": 5}}, ["ticker"])

SCHEMA_SQL      = _s("query_local_db",
    "Run a SQL SELECT on stocks.db. "
    "Table 'stocks': ticker, company, sector, industry, market_cap (Large/Mid/Small), exchange.",
    {"sql": {"type": "string", "description": "A valid SQL SELECT statement"}}, ["sql"])

ALL_SCHEMAS = [SCHEMA_TICKERS, SCHEMA_PRICE, SCHEMA_OVERVIEW,
               SCHEMA_STATUS,  SCHEMA_MOVERS, SCHEMA_NEWS, SCHEMA_SQL]

ALL_TOOL_FUNCTIONS = {
    "get_tickers_by_sector" : get_tickers_by_sector,
    "get_price_performance"  : get_price_performance,
    "get_company_overview"   : get_company_overview,
    "get_market_status"      : get_market_status,
    "get_top_gainers_losers" : get_top_gainers_losers,
    "get_news_sentiment"     : get_news_sentiment,
    "query_local_db"         : query_local_db,
}


# ── AgentResult ───────────────────────────────────────────────

@dataclass
class AgentResult:
    agent_name   : str
    answer       : str
    tools_called : list  = field(default_factory=list)
    raw_data     : dict  = field(default_factory=dict)
    confidence   : float = 0.0
    issues_found : list  = field(default_factory=list)
    reasoning    : str   = ""


# ── Core agent loop ───────────────────────────────────────────

def run_specialist_agent(
    agent_name   : str,
    system_prompt: str,
    task         : str,
    tool_schemas : list,
    model        : str,
    client       = client,
    max_iters    : int  = 8,
    verbose      : bool = False,
) -> AgentResult:
    """
    Core agentic loop used by every agent.
    Sends system_prompt + task to the LLM, executes tool calls until
    the model produces a final response with no tool calls.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": task},
    ]
    tools_called = []
    raw_data     = {}

    kwargs = {"model": model, "messages": messages}
    if tool_schemas:
        kwargs["tools"]       = tool_schemas
        kwargs["tool_choice"] = "auto"

    for _ in range(max_iters):
        response = client.chat.completions.create(**kwargs)
        msg = response.choices[0].message

        if not msg.tool_calls:
            return AgentResult(
                agent_name   = agent_name,
                answer       = msg.content or "",
                tools_called = tools_called,
                raw_data     = raw_data,
            )

        messages.append(msg)
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments)
            if verbose:
                print(f"  [{agent_name}] → {fn_name}({fn_args})")
            fn     = ALL_TOOL_FUNCTIONS.get(fn_name)
            result = fn(**fn_args) if fn else {"error": f"unknown tool: {fn_name}"}
            tools_called.append(fn_name)
            raw_data[fn_name] = result
            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      json.dumps(result),
            })

    return AgentResult(
        agent_name   = agent_name,
        answer       = "Max iterations reached without a final answer.",
        tools_called = tools_called,
        raw_data     = raw_data,
    )


# ── Baseline ──────────────────────────────────────────────────

def run_baseline(question: str, model: str, client=client, verbose: bool = False) -> AgentResult:
    """Single LLM call with no tools — control condition."""
    system_prompt = (
        "You are a knowledgeable financial assistant. "
        "Answer questions as accurately as you can based on your training knowledge. "
        "If you are uncertain or the data may be outdated, say so honestly — do not invent numbers."
    )
    return run_specialist_agent(
        agent_name    = "Baseline",
        system_prompt = system_prompt,
        task          = question,
        tool_schemas  = [],
        model         = model,
        client        = client,
        verbose       = verbose,
    )


# ── Single Agent ──────────────────────────────────────────────

SINGLE_AGENT_SYSTEM_PROMPT = """You are a precise financial research assistant with access to real-time tools.

Rules:
- Always use a tool to retrieve live data before answering financial questions.
- Never fabricate or estimate numbers. If a tool returns an error or empty data, say so explicitly.
- For questions requiring multiple steps (e.g. filter by sector, then get performance), call tools in sequence.
- When listing stocks, include ticker symbols.
- Be concise: lead with the answer, follow with supporting data.
"""

def run_single_agent(question: str, model: str, client=client, verbose: bool = False) -> AgentResult:
    return run_specialist_agent(
        agent_name    = "SingleAgent",
        system_prompt = SINGLE_AGENT_SYSTEM_PROMPT,
        task          = question,
        tool_schemas  = ALL_SCHEMAS,
        model         = model,
        client        = client,
        verbose       = verbose,
    )


# ── Multi-Agent: Adaptive Router → Parallel Specialists → Evidence Verifier → Aggregator ──

ROUTER_PROMPT = """You are a task router for a financial research system.
Classify the question and output a JSON object with this exact structure:
{
  "needs_market":       true/false,
  "needs_fundamentals": true/false,
  "needs_news":         true/false,
  "subtask_market":       "<sub-question for market agent, or null>",
  "subtask_fundamentals": "<sub-question for fundamentals agent, or null>",
  "subtask_news":         "<sub-question for news agent, or null>",
  "execution_mode":     "parallel" or "staged",
  "staged_first":       "market" or "fundamentals" or null,
  "staged_reason":      "<reason why staged execution is needed, or null>"
}

Domain rules:
- needs_market:       price performance, market status, top gainers/losers, stock price trends
- needs_fundamentals: P/E ratio, EPS, market cap, 52-week high/low, sector/company lookup
- needs_news:         news sentiment, recent headlines, narrative trends

Execution mode rules:
- Use "parallel" when the sub-tasks are independent (most cases)
- Use "staged" only when one agent's output is required as input for another:
  * "which semiconductor stocks had best 1-year return, then get their P/E" → staged, market first
  * "find tickers for energy sector, then get price performance" → staged, fundamentals first
- staged_first: which agent runs first in staged mode ("market" or "fundamentals"), or null

Be conservative: only activate an agent if the question clearly requires its domain.
Output valid JSON only, no extra text."""

MARKET_AGENT_PROMPT = """You are a market data specialist. Use your tools to retrieve
price performance, top movers, and market status. Never fabricate data.
If a tool returns an error, report it clearly. Include ticker symbols and exact values."""

FUNDAMENTALS_AGENT_PROMPT = """You are a fundamentals and sector specialist. Use your tools
to look up P/E ratios, EPS, market cap, sector lists, and run SQL on the local stock database.
Never invent financial figures. Report missing data honestly. Include ticker symbols."""

NEWS_AGENT_PROMPT = """You are a news sentiment analyst. Use your tool to retrieve
recent news sentiment for tickers. Summarise the sentiment with direction (Bullish/Bearish/Neutral)
and the most relevant headlines."""

AGGREGATOR_PROMPT = """You are a financial report aggregator. You receive verified findings
from specialist agents (sorted by confidence) and must merge them into one concise final answer.
- Lead with the direct answer to the question
- Integrate data from all specialists coherently
- If an agent flagged issues or low confidence, note it briefly
- Never invent data not present in the specialist outputs
- If there are conflicts between agents, prefer the higher-confidence source"""


def _verify_agent(result: AgentResult, allowed_tool_names: list) -> AgentResult:
    """
    Evidence Verifier: mechanically score each specialist's result.

    Scoring (base = 0.5, additive):
      +0.3  called at least one tool  (grounded in real data)
      +0.1  used only allowed tools   (schema compliance)
      +0.1  answer is substantive (>20 chars)
      -0.3  called no tools           (hallucination risk)
      -0.1  answer signals data gap   ("cannot", "no data", etc.)
      -0.1  used an unauthorized tool (schema violation)
    Score capped to [0.0, 1.0].
    """
    issues = []
    delta  = 0.0

    if result.tools_called:
        delta += 0.3
    else:
        delta -= 0.3
        issues.append("no_tools_called")

    unauthorized = [t for t in result.tools_called if t not in allowed_tool_names]
    if unauthorized:
        delta -= 0.1
        issues.append(f"unauthorized_tools:{','.join(unauthorized)}")
    elif result.tools_called:
        delta += 0.1

    if len(result.answer) > 20:
        delta += 0.1

    failure_phrases = ["i don't know", "cannot", "no data", "unable to",
                       "not available", "error retrieving", "failed to"]
    if any(p in result.answer.lower() for p in failure_phrases):
        delta -= 0.1
        issues.append("reported_data_gap")

    result.confidence   = max(0.0, min(1.0, 0.5 + delta))
    result.issues_found = issues
    return result


def run_multi_agent(question: str, model: str, client=client, verbose: bool = False) -> dict:
    """
    Adaptive Router → Parallel Specialists → Evidence Verifier → Aggregator.

    Returns:
        {
            "final_answer" : str,
            "agent_results": list[AgentResult],
            "elapsed_sec"  : float,
            "architecture" : "adaptive-router-verifier",
        }
    """
    start = time.time()

    # ── Step 1: Task Router ──────────────────────────────────
    router_response = client.chat.completions.create(
        model    = model,
        messages = [
            {"role": "system", "content": ROUTER_PROMPT},
            {"role": "user",   "content": question},
        ],
    )
    try:
        plan = json.loads(router_response.choices[0].message.content)
    except Exception:
        plan = {
            "needs_market": True, "needs_fundamentals": True, "needs_news": False,
            "subtask_market": question, "subtask_fundamentals": question, "subtask_news": None,
            "execution_mode": "parallel", "staged_first": None, "staged_reason": None,
        }

    active_domains = [
        d for d in ("market", "fundamentals", "news")
        if plan.get(f"needs_{d}") and plan.get(f"subtask_{d}")
    ]
    if verbose:
        print(f"[Router] domains={active_domains}  "
              f"mode={plan.get('execution_mode','parallel')}  "
              f"staged_first={plan.get('staged_first')}")

    # ── Specialist configs ────────────────────────────────────
    SPECIALIST_CFG = {
        "market":       ("MarketAgent",       MARKET_AGENT_PROMPT,
                         [SCHEMA_PRICE, SCHEMA_MOVERS, SCHEMA_STATUS]),
        "fundamentals": ("FundamentalsAgent", FUNDAMENTALS_AGENT_PROMPT,
                         [SCHEMA_OVERVIEW, SCHEMA_TICKERS, SCHEMA_SQL]),
        "news":         ("NewsAgent",         NEWS_AGENT_PROMPT,
                         [SCHEMA_NEWS]),
    }
    ALLOWED_TOOLS = {
        "market":       ["get_price_performance", "get_top_gainers_losers", "get_market_status"],
        "fundamentals": ["get_company_overview", "get_tickers_by_sector", "query_local_db"],
        "news":         ["get_news_sentiment"],
    }

    def _run_domain(domain: str, extra_context: str = "") -> AgentResult:
        name, prompt, schemas = SPECIALIST_CFG[domain]
        task = plan[f"subtask_{domain}"]
        if extra_context:
            task = task + extra_context
        return run_specialist_agent(
            agent_name    = name,
            system_prompt = prompt,
            task          = task,
            tool_schemas  = schemas,
            model         = model,
            client        = client,
            verbose       = verbose,
        )

    agent_results: list = []

    # ── Step 2: Parallel or Staged Specialist Execution ──────
    staged_first = plan.get("staged_first")
    if (plan.get("execution_mode") == "staged"
            and staged_first in active_domains
            and len(active_domains) > 1):
        remaining = [d for d in active_domains if d != staged_first]
        if verbose:
            print(f"[Router] Staged: {staged_first} → then parallel {remaining}")
        first_result = _run_domain(staged_first)
        agent_results.append(first_result)
        context_suffix = (
            f"\n\nContext from {SPECIALIST_CFG[staged_first][0]} (run first):\n"
            f"{first_result.answer}"
        )
        with ThreadPoolExecutor(max_workers=len(remaining)) as ex:
            futures = {ex.submit(_run_domain, d, context_suffix): d for d in remaining}
            for fut in as_completed(futures):
                agent_results.append(fut.result())
    else:
        if verbose:
            print(f"[Router] Parallel execution: {active_domains}")
        with ThreadPoolExecutor(max_workers=max(1, len(active_domains))) as ex:
            futures = {ex.submit(_run_domain, d): d for d in active_domains}
            for fut in as_completed(futures):
                agent_results.append(fut.result())

    # ── Step 3: Evidence Verifier ────────────────────────────
    name_to_domain = {SPECIALIST_CFG[d][0]: d for d in SPECIALIST_CFG}
    for i, result in enumerate(agent_results):
        domain = name_to_domain.get(result.agent_name)
        if domain:
            agent_results[i] = _verify_agent(result, ALLOWED_TOOLS[domain])
        if verbose:
            print(f"[Verifier] {agent_results[i].agent_name}: "
                  f"confidence={agent_results[i].confidence:.2f}  "
                  f"issues={agent_results[i].issues_found or 'none'}")

    # ── Step 4: Aggregator ────────────────────────────────────
    if agent_results:
        sorted_results = sorted(agent_results, key=lambda r: r.confidence, reverse=True)
        parts = "\n\n".join(
            f"[{r.agent_name}] (confidence={r.confidence:.2f}, "
            f"issues={r.issues_found or 'none'}):\n{r.answer}"
            for r in sorted_results
        )
        agg_resp = client.chat.completions.create(
            model    = model,
            messages = [
                {"role": "system", "content": AGGREGATOR_PROMPT},
                {"role": "user",   "content":
                    f"Question: {question}\n\nVerified specialist findings:\n{parts}"},
            ],
        )
        final_answer = agg_resp.choices[0].message.content or ""
    else:
        final_answer = "No agents were activated for this question."

    return {
        "final_answer" : final_answer,
        "agent_results": agent_results,
        "elapsed_sec"  : round(time.time() - start, 2),
        "architecture" : "adaptive-router-verifier",
    }
