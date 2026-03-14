import pathlib
import streamlit as st
from agents import (
    run_single_agent, run_multi_agent,
    client as _default_client,
    MODEL_SMALL, MODEL_LARGE,
    OPENAI_API_KEY, _PROJECT_ROOT,
)

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="FinTech AI Agents",
    page_icon="📈",
    layout="wide",
)

# ── Startup checks ────────────────────────────────────────────
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found. Add it to the .env file in the project root.")
    st.stop()

if not (pathlib.Path(_PROJECT_ROOT) / "stocks.db").exists():
    st.warning("stocks.db not found in project root — SQL tool calls will fail. "
               "Run the notebook cell to create it first.")

# ── Session state ─────────────────────────────────────────────
st.session_state.setdefault("messages", [])
# Each message:
# { "role": "user"|"assistant", "content": str, "metadata": None | {...} }


# ── Helpers ───────────────────────────────────────────────────

def build_question_with_history(history: list[dict], current_question: str) -> str:
    """
    Prepend the last 6 messages (3 exchanges) to the current question so
    agents can resolve follow-up references like 'that', 'the two', 'it'.
    """
    MAX_TURNS = 6
    relevant = [m for m in history if m["role"] in ("user", "assistant")]
    recent   = relevant[-MAX_TURNS:]

    if not recent:
        return current_question

    lines = ["The following is the conversation so far:\n"]
    for m in recent:
        label   = "User" if m["role"] == "user" else "Assistant"
        content = m["content"]
        # Truncate long assistant responses to avoid token bloat
        if m["role"] == "assistant" and len(content) > 800:
            content = content[:800] + "... [truncated]"
        lines.append(f"{label}: {content}")

    lines.append(
        f"\nNow answer this follow-up question, "
        f"using the conversation above as context:\n{current_question}"
    )
    return "\n".join(lines)


def call_agent(question: str, architecture: str, model: str) -> tuple[str, dict]:
    """Dispatch to the correct agent and return (response_text, metadata)."""
    if architecture == "Single Agent":
        result = run_single_agent(
            question=question, model=model, client=_default_client, verbose=False
        )
        return result.answer, {
            "architecture": "single-agent",
            "model"        : model,
            "tools_called" : result.tools_called,
            "confidence"   : result.confidence or None,
            "elapsed_sec"  : None,
            "agent_names"  : [result.agent_name],
        }
    else:  # Multi-Agent
        out = run_multi_agent(
            question=question, model=model, client=_default_client, verbose=False
        )
        all_tools = []
        for r in out["agent_results"]:
            all_tools.extend(r.tools_called)
        confs = [r.confidence for r in out["agent_results"] if r.confidence > 0]
        return out["final_answer"], {
            "architecture": "multi-agent",
            "model"        : model,
            "tools_called" : list(dict.fromkeys(all_tools)),  # deduped, insertion order
            "confidence"   : sum(confs) / len(confs) if confs else None,
            "elapsed_sec"  : out["elapsed_sec"],
            "agent_names"  : [r.agent_name for r in out["agent_results"]],
        }


def render_metadata(m: dict):
    """Render the compact metadata caption below an assistant message."""
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


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 FinTech AI Agents")
    st.markdown("---")

    architecture = st.radio(
        "Agent Architecture",
        options=["Single Agent", "Multi-Agent"],
        index=0,
    )

    model = st.selectbox(
        "Model",
        options=[MODEL_SMALL, MODEL_LARGE],
        index=0,
    )

    st.markdown("---")

    if st.button("Clear conversation", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

    st.markdown("---")
    if architecture == "Single Agent":
        st.caption("One LLM with access to all 7 tools. "
                   "Plans, executes, and synthesises in a single context window.")
    else:
        st.caption("Router → Parallel Specialists → Evidence Verifier → Aggregator. "
                   "Only the relevant agents are activated per question.")


# ── Main area ─────────────────────────────────────────────────
st.title("FinTech AI Agents")
st.caption("Ask about stock prices, P/E ratios, sector performance, news sentiment, and more.")

# Render conversation history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("metadata"):
            render_metadata(msg["metadata"])

# Chat input
if prompt := st.chat_input("Ask about stocks, sectors, P/E ratios..."):
    # Show user message immediately
    st.session_state["messages"].append({"role": "user", "content": prompt, "metadata": None})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build history-aware question
    question_with_history = build_question_with_history(
        st.session_state["messages"][:-1],  # exclude the just-appended user turn
        prompt,
    )

    # Call agent and stream result into assistant bubble
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response_text, metadata = call_agent(
                    question_with_history, architecture, model
                )
            except Exception as e:
                response_text = f"Error: {e}"
                metadata = {
                    "architecture": architecture.lower().replace(" ", "-"),
                    "model"        : model,
                    "tools_called" : [],
                    "confidence"   : None,
                    "elapsed_sec"  : None,
                    "agent_names"  : [],
                }
        st.markdown(response_text)
        render_metadata(metadata)

    # Persist assistant message
    st.session_state["messages"].append({
        "role"    : "assistant",
        "content" : response_text,
        "metadata": metadata,
    })
