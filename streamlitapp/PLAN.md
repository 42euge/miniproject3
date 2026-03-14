# Streamlit App — Implementation Plan

## File Structure

```
streamlitapp/
  agents.py      ← all notebook logic extracted + refactored
  app.py         ← Streamlit UI
  PLAN.md        ← this file

# Run from anywhere:
#   streamlit run streamlitapp/app.py
```

`stocks.db` and `.env` remain in the project root (parent of `streamlitapp/`).
`agents.py` uses `__file__`-relative paths so both files are always found regardless of cwd.

---

## Part 1 — `agents.py`

### What to extract from the notebook

Copy verbatim:
- All imports (os, json, time, sqlite3, requests, textwrap, pandas, yfinance, dataclasses, dotenv, openai)
- All 7 tool functions
- All 7 schemas + `ALL_SCHEMAS` + `ALL_TOOL_FUNCTIONS`
- `AgentResult` dataclass
- All prompts (SINGLE_AGENT, ROUTER, MARKET_AGENT, FUNDAMENTALS_AGENT, NEWS_AGENT, AGGREGATOR)
- `_verify_agent`

### Path handling

```python
import pathlib
_MODULE_DIR   = pathlib.Path(__file__).parent   # streamlitapp/
_PROJECT_ROOT = _MODULE_DIR.parent              # miniproject3/

load_dotenv(_PROJECT_ROOT / ".env")
DB_PATH = str(_PROJECT_ROOT / "stocks.db")
```

Replace the notebook's `DB_PATH = "stocks.db"` with the above so SQL tool always works.

### Fix the global `ACTIVE_MODEL` problem

The notebook uses a module-level mutable `ACTIVE_MODEL`. This is broken for Streamlit (not thread-safe, resets on rerun). **Solution: pass `model` and `client` as explicit parameters.**

Refactor signatures:

```python
def run_specialist_agent(
    agent_name, system_prompt, task, tool_schemas,
    model: str,    # replaces ACTIVE_MODEL global
    client,        # injected OpenAI client
    max_iters=8,
    verbose=False,
) -> AgentResult:

def run_baseline(question: str, model: str, client, verbose=False) -> AgentResult:
def run_single_agent(question: str, model: str, client, verbose=False) -> AgentResult:
def run_multi_agent(question: str, model: str, client, verbose=False) -> dict:
```

Every `client.chat.completions.create(model=ACTIVE_MODEL, ...)` inside these functions
becomes `client.chat.completions.create(model=model, ...)`.

Keep `MODEL_SMALL = "gpt-4o-mini"` and `MODEL_LARGE = "gpt-4o"` as module constants.
Build `client = OpenAI(api_key=OPENAI_API_KEY)` once at module level — `app.py` imports it.

### `run_multi_agent` internal closure after refactor

```python
def _run_domain(domain: str, extra_context: str = "") -> AgentResult:
    name, prompt, schemas = SPECIALIST_CFG[domain]
    task = plan[f"subtask_{domain}"]
    if extra_context:
        task = task + extra_context
    return run_specialist_agent(
        agent_name=name, system_prompt=prompt, task=task,
        tool_schemas=schemas, model=model, client=client, verbose=verbose,
    )
```

The Router LLM call and Aggregator LLM call inside `run_multi_agent` also need `model=model`.

### `verbose` default

Change all defaults from `verbose=True` to `verbose=False`. No console in Streamlit.

---

## Part 2 — `app.py`

### Session state schema

```python
st.session_state.setdefault("messages", [])

# Each message dict:
# {
#   "role":     "user" | "assistant",
#   "content":  str,
#   "metadata": None | {
#       "architecture": str,       # "single-agent" | "multi-agent"
#       "model":        str,
#       "tools_called": list[str],
#       "confidence":   float | None,
#       "elapsed_sec":  float | None,
#       "agent_names":  list[str],
#   }
# }
```

### Sidebar

```python
with st.sidebar:
    st.title("FinTech AI Agents")

    architecture = st.radio(
        "Agent Architecture",
        options=["Single Agent", "Multi-Agent"],
    )

    model = st.selectbox(
        "Model",
        options=["gpt-4o-mini", "gpt-4o"],
    )

    st.divider()

    if st.button("Clear conversation", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

    # Brief description
    if architecture == "Single Agent":
        st.caption("One LLM with access to all 7 tools.")
    else:
        st.caption("Router → Parallel Specialists → Verifier → Aggregator.")
```

### Chat history rendering loop

```python
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and msg.get("metadata"):
            m = msg["metadata"]
            parts = [
                f"Architecture: **{m['architecture']}**",
                f"Model: **{m['model']}**",
            ]
            if m.get("tools_called"):
                parts.append(f"Tools: `{', '.join(m['tools_called'])}`")
            if m.get("elapsed_sec") is not None:
                parts.append(f"Time: {m['elapsed_sec']:.1f}s")
            if m.get("confidence") is not None:
                parts.append(f"Avg confidence: {m['confidence']:.0%}")
            st.caption(" | ".join(parts))

            if m.get("agent_names") and len(m["agent_names"]) > 1:
                with st.expander("Specialist breakdown"):
                    for name in m["agent_names"]:
                        st.markdown(f"- {name}")
```

### Chat input + agent dispatch

```python
if prompt := st.chat_input("Ask about stocks, sectors, P/E ratios..."):
    st.session_state["messages"].append({"role": "user", "content": prompt, "metadata": None})

    with st.chat_message("user"):
        st.markdown(prompt)

    question_with_history = build_question_with_history(
        st.session_state["messages"][:-1],  # exclude just-appended user turn
        prompt,
    )

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response_text, metadata = call_agent(question_with_history, architecture, model)
            except Exception as e:
                response_text = f"Error: {e}"
                metadata = {"architecture": architecture.lower().replace(" ", "-"),
                            "model": model, "tools_called": [], "confidence": None,
                            "elapsed_sec": None, "agent_names": []}
        st.markdown(response_text)
        # render metadata inline (same as history loop above)

    st.session_state["messages"].append({
        "role": "assistant", "content": response_text, "metadata": metadata,
    })
```

### `call_agent` helper

```python
from agents import run_single_agent, run_multi_agent, client as _default_client

def call_agent(question: str, architecture: str, model: str) -> tuple[str, dict]:
    if architecture == "Single Agent":
        result = run_single_agent(question=question, model=model,
                                  client=_default_client, verbose=False)
        return result.answer, {
            "architecture": "single-agent",
            "model": model,
            "tools_called": result.tools_called,
            "confidence": result.confidence or None,
            "elapsed_sec": None,
            "agent_names": [result.agent_name],
        }
    else:
        out = run_multi_agent(question=question, model=model,
                              client=_default_client, verbose=False)
        all_tools = []
        for r in out["agent_results"]:
            all_tools.extend(r.tools_called)
        confs = [r.confidence for r in out["agent_results"] if r.confidence > 0]
        return out["final_answer"], {
            "architecture": "multi-agent",
            "model": model,
            "tools_called": list(dict.fromkeys(all_tools)),  # deduped, ordered
            "confidence": sum(confs) / len(confs) if confs else None,
            "elapsed_sec": out["elapsed_sec"],
            "agent_names": [r.agent_name for r in out["agent_results"]],
        }
```

---

## Part 3 — Conversational Memory

### Strategy: inject history into the question string

The agent functions take a single `question: str` and start a fresh `[system, user]`
message list internally. Conversational context is injected by prepending the conversation
history to the question string before any agent call.

This works for both Single Agent and Multi-Agent — the Router also sees the history
and produces context-aware subtask strings for each specialist.

### `build_question_with_history`

```python
def build_question_with_history(history: list[dict], current_question: str) -> str:
    MAX_TURNS = 6  # 3 exchanges (3 user + 3 assistant)

    relevant = [m for m in history if m["role"] in ("user", "assistant")]
    recent   = relevant[-MAX_TURNS:]

    if not recent:
        return current_question

    lines = ["The following is the conversation so far:\n"]
    for m in recent:
        label   = "User" if m["role"] == "user" else "Assistant"
        content = m["content"]
        if m["role"] == "assistant" and len(content) > 800:
            content = content[:800] + "... [truncated]"
        lines.append(f"{label}: {content}")

    lines.append(
        f"\nNow answer this follow-up question, "
        f"using the conversation above as context:\n{current_question}"
    )
    return "\n".join(lines)
```

### Why this works for the 3-turn demo

- **Q1** "What is NVIDIA's P/E ratio?" → no history, question passed unchanged
- **Q2** "How does that compare to AMD?" → history contains Q1+A1; LLM sees "that = NVIDIA's P/E" and fetches AMD's P/E
- **Q3** "Which of the two has better news sentiment?" → history contains Q1+A1+Q2+A2; LLM resolves "the two" = NVIDIA and AMD, calls `get_news_sentiment` for both

---

## Part 4 — Edge Cases

| Case | Handling |
|---|---|
| Missing `OPENAI_API_KEY` | `st.error` + `st.stop()` at startup |
| Missing `stocks.db` | `st.warning` at startup (SQL calls will fail gracefully) |
| Router JSON parse failure | fallback in `run_multi_agent` activates all agents |
| No agents activated | `run_multi_agent` returns "No agents were activated" |
| Max iterations hit | `run_specialist_agent` returns informative string |
| API error mid-call | `call_agent` wrapped in `try/except`; displays `st.error` |
| Thread safety | `ThreadPoolExecutor` in `run_multi_agent` is safe; `client` is read-only |

---

## Part 5 — Implementation Sequence

1. **Create `agents.py`**
   - Copy imports, add `pathlib`, fix paths
   - Remove `ACTIVE_MODEL` global; keep `MODEL_SMALL`/`MODEL_LARGE` constants
   - Copy all 7 tools, schemas, `AgentResult`, prompts, `_verify_agent`
   - Refactor `run_specialist_agent` + all public agent functions to accept `model`, `client`
   - Change `verbose` defaults to `False`

2. **Create `app.py`**
   - Startup checks
   - Session state initialization
   - Sidebar
   - `build_question_with_history`
   - `call_agent`
   - History rendering loop
   - `st.chat_input` block

3. **Test locally**
   ```bash
   cd miniproject3/streamlitapp
   streamlit run app.py
   ```
   - Verify single-agent response with tools shown
   - Verify multi-agent shows specialist names and confidence
   - Run the 3-turn NVIDIA → AMD → sentiment demo
   - Switch models mid-conversation and verify correct model is used
   - Clear conversation and verify history resets
