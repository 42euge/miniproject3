"""
compare_architectures.py
========================
Head-to-head comparison of Eugenio's and Oguz's multi-agent architectures
and evaluators on the shared 15-question benchmark.

Outputs:
  - compare_results.json   (full raw data)
  - compare_summary.txt    (formatted tables)

Usage:
  python compare_architectures.py [--model gpt-4o-mini]
"""

import sys, os, json, time, argparse, textwrap
from pathlib import Path

# ── Setup paths and env ──────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "streamlitapp"))
sys.path.insert(0, str(ROOT / "miniproject3_OguzSinanoglu"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

# ── Import Eugenio's multi-agent ─────────────────────────────
import streamlitapp.agents as eugenio_agents

# ── Import Oguz's multi-agent ────────────────────────────────
import miniproject3_OguzSinanoglu.finagents as oguz_agents

# ── Benchmark questions (shared) ─────────────────────────────
BENCHMARK_QUESTIONS = [
    {"id":"Q01","complexity":"easy","category":"sector_lookup",
     "question":"List all semiconductor companies in the database.",
     "expected":"Should return company names and tickers for semiconductor stocks from the local DB. "
                "Tickers include NVDA, AMD, INTC, QCOM, AVGO, TXN, ADI, MU and others."},
    {"id":"Q02","complexity":"easy","category":"market_status",
     "question":"Are the US stock markets open right now?",
     "expected":"Should return the current open/closed status for NYSE and NASDAQ "
                "with their trading hours."},
    {"id":"Q03","complexity":"easy","category":"fundamentals",
     "question":"What is the P/E ratio of Apple (AAPL)?",
     "expected":"Should return AAPL P/E ratio as a single numeric value fetched from Alpha Vantage."},
    {"id":"Q04","complexity":"easy","category":"sentiment",
     "question":"What is the latest news sentiment for Microsoft (MSFT)?",
     "expected":"Should return 3-5 recent MSFT headlines with Bullish/Bearish/Neutral labels and scores."},
    {"id":"Q05","complexity":"easy","category":"price",
     "question":"What is NVIDIA's stock price performance over the last month?",
     "expected":"Should return NVDA start price, end price, and % change for the 1-month period."},
    {"id":"Q06","complexity":"medium","category":"price_comparison",
     "question":"Compare the 1-year price performance of AAPL, MSFT, and GOOGL. Which grew the most?",
     "expected":"Should fetch 1y performance for all 3 tickers, return % change for each, "
                "and identify the highest performer."},
    {"id":"Q07","complexity":"medium","category":"fundamentals",
     "question":"Compare the P/E ratios of AAPL, MSFT, and NVDA. Which looks most expensive?",
     "expected":"Should return P/E ratios for all 3 tickers and identify which has the highest P/E."},
    {"id":"Q08","complexity":"medium","category":"sector_price",
     "question":"Which energy stocks in the database had the best 6-month performance?",
     "expected":"Should query the DB for energy sector tickers, fetch 6-month price performance "
                "for each, and return them ranked by % change."},
    {"id":"Q09","complexity":"medium","category":"sentiment",
     "question":"What is the news sentiment for Tesla (TSLA) and how has its stock moved this month?",
     "expected":"Should return TSLA news sentiment (label + score) AND 1-month price % change "
                "from two separate tool calls."},
    {"id":"Q10","complexity":"medium","category":"fundamentals",
     "question":"What are the 52-week high and low for JPMorgan (JPM) and Goldman Sachs (GS)?",
     "expected":"Should return 52-week high and low for both JPM and GS fetched from Alpha Vantage."},
    {"id":"Q11","complexity":"hard","category":"multi_condition",
     "question":"Which tech stocks dropped this month but grew this year? Return the top 3.",
     "expected":"Should get tech tickers from DB, fetch both 1-month and year-to-date performance, "
                "filter for negative 1-month AND positive YTD, return top 3 by yearly growth with "
                "exact percentages. Results must satisfy both conditions simultaneously."},
    {"id":"Q12","complexity":"hard","category":"multi_condition",
     "question":"Which large-cap technology stocks on NASDAQ have grown more than 20% this year?",
     "expected":"Should query DB for large-cap NASDAQ tech stocks, fetch YTD performance, "
                "filter for >20% growth, and return matching tickers with exact % change."},
    {"id":"Q13","complexity":"hard","category":"cross_domain",
     "question":"For the top 3 semiconductor stocks by 1-year return, what are their P/E ratios "
                "and current news sentiment?",
     "expected":"Should find semiconductor tickers in DB, rank by 1-year return to find top 3, "
                "then fetch P/E ratio AND news sentiment for each -- requiring three separate "
                "data domains (price, fundamentals, sentiment)."},
    {"id":"Q14","complexity":"hard","category":"cross_domain",
     "question":"Compare the market cap, P/E ratio, and 1-year stock performance of JPM, GS, and BAC.",
     "expected":"Should return market cap, P/E, and 1-year % change for all 3 tickers, "
                "combining Alpha Vantage fundamentals and yfinance price data."},
    {"id":"Q15","complexity":"hard","category":"multi_condition",
     "question":"Which finance sector stocks are trading closer to their 52-week low than their "
                "52-week high? Return the news sentiment for each.",
     "expected":"Should get finance sector tickers from DB, fetch 52-week high and low for each, "
                "compute proximity to the low, then fetch news sentiment for qualifying stocks."},
]


# ── Evaluators ───────────────────────────────────────────────

HEDGING_PHRASES = [
    "approximately", "roughly", "around", "about",
    "based on current market conditions", "based on my knowledge",
    "based on my training", "as of my last update", "estimated",
    "i believe", "i think", "likely around", "probably",
]
TOOL_SOURCE_PHRASES = [
    "alpha vantage", "from the api", "from the tool", "from the database",
    "from the data", "real-time", "live data",
]

def evaluator_eugenio(question: str, expected: str, answer: str, model: str) -> dict:
    """Eugenio's evaluator: rule-based hedging pre-check + LLM judge."""
    answer_lower = answer.lower()
    expected_lower = expected.lower()
    requires_tool = any(p in expected_lower for p in TOOL_SOURCE_PHRASES)
    uses_hedging = any(p in answer_lower for p in HEDGING_PHRASES)
    is_refusal = any(p in answer_lower for p in [
        "i cannot", "i can't", "i don't", "i am unable", "unable to",
        "please check", "not able to retrieve",
    ])
    if requires_tool and uses_hedging and not is_refusal:
        return {
            "score": 1, "max_score": 3,
            "reasoning": "Hedged estimate instead of tool retrieval.",
            "hallucination_detected": True,
            "key_issues": ["hedged estimate"],
        }

    rubric = textwrap.dedent("""\
    You are an expert financial AI evaluator. Score the agent_answer against the expected_answer description.

    The expected_answer is a DESCRIPTION of what a correct answer should look like.
    The agent may have used real-time tools. Do NOT penalise for providing a specific number
    just because it does not appear in the expected_answer text.

    Scoring rubric (0-3):
      3 -- Fully correct: satisfies expected, on-topic, no invented entities
      2 -- Mostly correct: minor omission or slight inaccuracy
      1 -- Partially correct: relevant attempt but missing key parts
      0 -- Incorrect or refusal

    Hallucination: true ONLY if agent invents entities, states wildly implausible numbers,
    or uses hedging language suggesting training-data guessing.
    Refusal is NOT hallucination. Precise numbers are NOT hallucination.

    Return JSON only: {"score":<0-3>,"max_score":3,"reasoning":"<one sentence>",
    "hallucination_detected":<bool>,"key_issues":[<strings>]}""")

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": rubric},
                {"role": "user", "content": f"Question: {question}\nExpected: {expected}\nAgent answer: {answer}"},
            ],
        )
        content = resp.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        result = json.loads(content)
        result.setdefault("max_score", 3)
        return result
    except Exception:
        return {"score": 0, "max_score": 3, "reasoning": "parse error",
                "hallucination_detected": False, "key_issues": ["parse error"]}


def evaluator_oguz(question: str, expected: str, answer: str, model: str) -> dict:
    """Oguz's evaluator: pure LLM-based with financial-domain hallucination rules."""
    system_prompt = textwrap.dedent("""\
    You are a financial-domain answer evaluator.
    Score the agent answer against the expected answer description using the rubric below.
    Respond ONLY with a valid JSON object.

    CONTEXT: The agents call live data tools (Alpha Vantage, yfinance, SQL DB).
    A clean answer with a specific numeric value is the CORRECT output from an agent that
    successfully called a tool. Do NOT penalise for lacking a visible source citation.

    SCORING RUBRIC:
      3 -- Fully correct: specific data returned, matches expected format, no fabrication
      2 -- Partially correct: data present but incomplete, or minor inaccuracies
      1 -- Mostly wrong: clear fabrication signals, wrong data, missed required conditions
      0 -- Complete failure: refused entirely, directed user elsewhere, or completely irrelevant

    HALLUCINATION -- set hallucination_detected=true ONLY for clear fabrication signals:
      - Agent admits it cannot access live data AND still provides a number
      - Hedging language paired with a vague or round number and NO precise decimal
      - 'as of my last update', 'I estimate', 'historically around'
      - Tickers or companies invented or irrelevant
    DO NOT flag hallucination for:
      - A precise decimal like 32.59 -- precision signals a real tool call
      - 'approximately 32.59' -- the exact decimal proves a tool call
      - Any refusal with no fabricated number (score=0, hallucination=false)
      - Returning MORE tickers than expected, as long as required ones are present

    OUTPUT FORMAT (JSON only):
    {"score": <int 0-3>, "max_score": 3, "reasoning": "<one sentence>",
     "hallucination_detected": <true|false>, "key_issues": [<str>, ...]}""")

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"QUESTION: {question}\n\nEXPECTED ANSWER DESCRIPTION: {expected}\n\nAGENT ANSWER:\n{answer}"},
            ],
            temperature=0,
            max_tokens=300,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        result = json.loads(raw)
        result["score"] = int(result.get("score", 0))
        result["max_score"] = 3
        result["hallucination_detected"] = bool(result.get("hallucination_detected", False))
        result["reasoning"] = str(result.get("reasoning", ""))
        result["key_issues"] = list(result.get("key_issues", []))
        return result
    except Exception:
        return {"score": 0, "max_score": 3, "reasoning": "parse error",
                "hallucination_detected": False, "key_issues": ["parse error"]}


# ── Runner ───────────────────────────────────────────────────

def run_comparison(model: str):
    print(f"\n{'='*70}")
    print(f"  ARCHITECTURE & EVALUATOR COMPARISON — {model}")
    print(f"{'='*70}\n")

    results = []

    for i, q in enumerate(BENCHMARK_QUESTIONS):
        qid = q["id"]
        print(f"[{i+1}/15] {qid}: {q['question'][:60]}...")

        # ── Run Eugenio's multi-agent ────────────────────────
        print(f"  Running Eugenio MA...", end=" ", flush=True)
        t0 = time.time()
        try:
            e_out = eugenio_agents.run_multi_agent(
                question=q["question"], model=model,
                client=client, verbose=False,
            )
            e_answer = e_out["final_answer"]
            e_time = round(time.time() - t0, 2)
            e_tools = []
            for r in e_out["agent_results"]:
                e_tools.extend(r.tools_called)
            e_agents = [r.agent_name for r in e_out["agent_results"]]
        except Exception as ex:
            e_answer = f"ERROR: {ex}"
            e_time = round(time.time() - t0, 2)
            e_tools, e_agents = [], []
        print(f"{e_time}s")

        # ── Run Oguz's multi-agent ───────────────────────────
        print(f"  Running Oguz MA...", end=" ", flush=True)
        t0 = time.time()
        try:
            oguz_agents.set_active_model(model)
            o_out = oguz_agents.run_multi_agent(
                question=q["question"], verbose=False,
            )
            o_answer = o_out["final_answer"]
            o_time = round(time.time() - t0, 2)
            o_tools = []
            for r in o_out["agent_results"]:
                o_tools.extend(r.tools_called)
            o_agents = [r.agent_name for r in o_out["agent_results"]]
        except Exception as ex:
            o_answer = f"ERROR: {ex}"
            o_time = round(time.time() - t0, 2)
            o_tools, o_agents = [], []
        print(f"{o_time}s")

        # ── Cross-evaluate with BOTH evaluators ──────────────
        print(f"  Evaluating...", end=" ", flush=True)
        # Eugenio's evaluator scores both answers
        ee_on_e = evaluator_eugenio(q["question"], q["expected"], e_answer, model)
        ee_on_o = evaluator_eugenio(q["question"], q["expected"], o_answer, model)
        # Oguz's evaluator scores both answers
        oe_on_e = evaluator_oguz(q["question"], q["expected"], e_answer, model)
        oe_on_o = evaluator_oguz(q["question"], q["expected"], o_answer, model)
        print("done")

        record = {
            "id": qid,
            "complexity": q["complexity"],
            "category": q["category"],
            "question": q["question"],
            "eugenio": {
                "answer": e_answer,
                "time": e_time,
                "tools": e_tools,
                "agents": e_agents,
                "score_by_eugenio_eval": ee_on_e["score"],
                "score_by_oguz_eval": oe_on_e["score"],
                "halluc_eugenio_eval": ee_on_e["hallucination_detected"],
                "halluc_oguz_eval": oe_on_e["hallucination_detected"],
                "reasoning_eugenio_eval": ee_on_e.get("reasoning", ""),
                "reasoning_oguz_eval": oe_on_e.get("reasoning", ""),
            },
            "oguz": {
                "answer": o_answer,
                "time": o_time,
                "tools": o_tools,
                "agents": o_agents,
                "score_by_eugenio_eval": ee_on_o["score"],
                "score_by_oguz_eval": oe_on_o["score"],
                "halluc_eugenio_eval": ee_on_o["hallucination_detected"],
                "halluc_oguz_eval": oe_on_o["hallucination_detected"],
                "reasoning_eugenio_eval": ee_on_o.get("reasoning", ""),
                "reasoning_oguz_eval": oe_on_o.get("reasoning", ""),
            },
        }
        results.append(record)

        # Quick inline summary
        print(f"    Eugenio: E-eval={ee_on_e['score']}/3  O-eval={oe_on_e['score']}/3")
        print(f"    Oguz:    E-eval={ee_on_o['score']}/3  O-eval={oe_on_o['score']}/3")
        print()

    return results


def print_summary(results: list, model: str):
    """Print formatted summary tables."""
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f"  SUMMARY — {model}")
    lines.append(f"{'='*80}\n")

    # ── Per-question table ───────────────────────────────────
    header = f"{'QID':<5} {'Diff':<7} {'Eug(E-eval)':<12} {'Eug(O-eval)':<12} {'Oguz(E-eval)':<13} {'Oguz(O-eval)':<13} {'E-time':<7} {'O-time':<7}"
    lines.append(header)
    lines.append("-" * len(header))

    for r in results:
        lines.append(
            f"{r['id']:<5} {r['complexity']:<7} "
            f"{r['eugenio']['score_by_eugenio_eval']:<12} "
            f"{r['eugenio']['score_by_oguz_eval']:<12} "
            f"{r['oguz']['score_by_eugenio_eval']:<13} "
            f"{r['oguz']['score_by_oguz_eval']:<13} "
            f"{r['eugenio']['time']:<7} "
            f"{r['oguz']['time']:<7}"
        )

    # ── Tier summaries ───────────────────────────────────────
    lines.append(f"\n{'─'*60}")
    lines.append("TIER AVERAGES (score /3)")
    lines.append(f"{'─'*60}")

    for tier in ["easy", "medium", "hard"]:
        tier_qs = [r for r in results if r["complexity"] == tier]
        n = len(tier_qs)
        if n == 0:
            continue

        e_ee = sum(r["eugenio"]["score_by_eugenio_eval"] for r in tier_qs) / n
        e_oe = sum(r["eugenio"]["score_by_oguz_eval"] for r in tier_qs) / n
        o_ee = sum(r["oguz"]["score_by_eugenio_eval"] for r in tier_qs) / n
        o_oe = sum(r["oguz"]["score_by_oguz_eval"] for r in tier_qs) / n
        e_t = sum(r["eugenio"]["time"] for r in tier_qs) / n
        o_t = sum(r["oguz"]["time"] for r in tier_qs) / n

        lines.append(f"\n  {tier.upper()} ({n} questions):")
        lines.append(f"    Eugenio MA:  Eugenio-eval={e_ee:.2f}  Oguz-eval={e_oe:.2f}  avg_time={e_t:.1f}s")
        lines.append(f"    Oguz MA:     Eugenio-eval={o_ee:.2f}  Oguz-eval={o_oe:.2f}  avg_time={o_t:.1f}s")

    # ── Overall ──────────────────────────────────────────────
    n = len(results)
    e_ee_all = sum(r["eugenio"]["score_by_eugenio_eval"] for r in results) / n
    e_oe_all = sum(r["eugenio"]["score_by_oguz_eval"] for r in results) / n
    o_ee_all = sum(r["oguz"]["score_by_eugenio_eval"] for r in results) / n
    o_oe_all = sum(r["oguz"]["score_by_oguz_eval"] for r in results) / n
    e_t_all = sum(r["eugenio"]["time"] for r in results) / n
    o_t_all = sum(r["oguz"]["time"] for r in results) / n
    e_halluc_ee = sum(1 for r in results if r["eugenio"]["halluc_eugenio_eval"])
    e_halluc_oe = sum(1 for r in results if r["eugenio"]["halluc_oguz_eval"])
    o_halluc_ee = sum(1 for r in results if r["oguz"]["halluc_eugenio_eval"])
    o_halluc_oe = sum(1 for r in results if r["oguz"]["halluc_oguz_eval"])

    lines.append(f"\n{'─'*60}")
    lines.append("OVERALL")
    lines.append(f"{'─'*60}")
    lines.append(f"  Eugenio MA:  Eugenio-eval={e_ee_all:.2f}  Oguz-eval={e_oe_all:.2f}  avg_time={e_t_all:.1f}s  halluc(E/O)={e_halluc_ee}/{e_halluc_oe}")
    lines.append(f"  Oguz MA:     Eugenio-eval={o_ee_all:.2f}  Oguz-eval={o_oe_all:.2f}  avg_time={o_t_all:.1f}s  halluc(E/O)={o_halluc_ee}/{o_halluc_oe}")

    # ── Evaluator agreement ──────────────────────────────────
    agree = sum(1 for r in results
                if r["eugenio"]["score_by_eugenio_eval"] == r["eugenio"]["score_by_oguz_eval"])
    agree += sum(1 for r in results
                 if r["oguz"]["score_by_eugenio_eval"] == r["oguz"]["score_by_oguz_eval"])
    total = 2 * n
    lines.append(f"\n  Evaluator agreement (exact score match): {agree}/{total} ({100*agree/total:.0f}%)")

    # ── Which architecture won per question ──────────────────
    e_wins = 0
    o_wins = 0
    ties = 0
    for r in results:
        # Average across both evaluators for fairness
        e_avg = (r["eugenio"]["score_by_eugenio_eval"] + r["eugenio"]["score_by_oguz_eval"]) / 2
        o_avg = (r["oguz"]["score_by_eugenio_eval"] + r["oguz"]["score_by_oguz_eval"]) / 2
        if e_avg > o_avg:
            e_wins += 1
        elif o_avg > e_avg:
            o_wins += 1
        else:
            ties += 1

    lines.append(f"\n  Head-to-head (avg of both evaluators): Eugenio wins={e_wins}  Oguz wins={o_wins}  Ties={ties}")

    text = "\n".join(lines)
    print(text)
    return text


# ── Main ─────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4o-mini", choices=["gpt-4o-mini", "gpt-4o"])
    args = parser.parse_args()

    results = run_comparison(args.model)

    # Save raw results
    out_json = ROOT / "compare_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nRaw results saved to {out_json}")

    # Print and save summary
    summary_text = print_summary(results, args.model)
    out_txt = ROOT / "compare_summary.txt"
    with open(out_txt, "w") as f:
        f.write(summary_text)
    print(f"Summary saved to {out_txt}")
