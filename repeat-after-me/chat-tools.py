"""
chat-tools.py — CLI chat with tool calling via LangGraph + Ollama (llama3.1:8b).

Tools are bound at graph construction time, not per-message.  The schemas are
serialised into every prompt regardless, so binding once keeps the graph's
behaviour fixed and avoids re-specifying them on each call — same pattern
LangGraph's create_react_agent uses internally.

Each assistant turn prints two blocks:
  [RAW]    — AIMessage.content (the text tokens the model emitted)
  [PARSED] — AIMessage.tool_calls (structured list after langchain_ollama
              parses the <|python_tag|> blob)

Run:
    python chat-tools.py
    python chat-tools.py --model llama3.2:3b
"""

from __future__ import annotations

import json
import sys
from typing import Annotated

from langchain_core.messages import AIMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MODEL = next(
    (sys.argv[sys.argv.index("--model") + 1] for _ in ["x"] if "--model" in sys.argv),
    "llama3.1:8b",
)

SEP = "-" * 60


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------
def search_tables(query: str, max_budget: int | None = None) -> str:
    """Search online for tables matching a query and budget."""
    results = [
        {"id": "T1", "name": "Oak Dining Table",    "price_usd": 320, "comfort": "medium"},
        {"id": "T2", "name": "Ergonomic Work Table", "price_usd": 450, "comfort": "high"},
        {"id": "T3", "name": "Folding Picnic Table", "price_usd": 89,  "comfort": "low"},
    ]
    if isinstance(max_budget, int):
        results = [r for r in results if r["price_usd"] <= max_budget]
    return json.dumps(results)


def find_desks(query: str, max_budget: int | None = None) -> str:
    """Find desks online matching a query and budget."""
    results = [
        {"id": "D1", "name": "Standing Desk Pro",   "price_usd": 599, "comfort": "very high"},
        {"id": "D2", "name": "L-shaped Corner Desk", "price_usd": 275, "comfort": "high"},
        {"id": "D3", "name": "Basic Writing Desk",   "price_usd": 120, "comfort": "medium"},
    ]
    if isinstance(max_budget, int):
        results = [r for r in results if r["price_usd"] <= max_budget]
    return json.dumps(results)


TOOL_REGISTRY = {
    "search_tables": search_tables,
    "find_desks":    find_desks,
}

TOOLS = [search_tables, find_desks]


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class State(TypedDict):
    messages: Annotated[list, add_messages]


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------
llm = ChatOllama(model=MODEL, temperature=0).bind_tools(TOOLS)


def call_model(state: State) -> State:
    response = llm.invoke(state["messages"])

    # ── show raw + parsed output ──────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"[RAW]    content = {repr(response.content)}")
    if response.tool_calls:
        print(f"[PARSED] tool_calls =")
        for tc in response.tool_calls:
            print(f"           name={tc['name']!r}  args={tc['args']}")
    else:
        print("[PARSED] tool_calls = (none)")
    print(SEP)

    return {"messages": [response]}


def run_tools(state: State) -> State:
    last: AIMessage = state["messages"][-1]
    tool_messages = []
    for tc in last.tool_calls:
        fn = TOOL_REGISTRY.get(tc["name"])
        if fn is None:
            result = json.dumps({"error": f"unknown tool '{tc['name']}'"})
        else:
            result = fn(**tc["args"])
        print(f"[TOOL]   {tc['name']}({tc['args']}) -> {result}")
        tool_messages.append(
            ToolMessage(content=result, tool_call_id=tc["id"])
        )
    return {"messages": tool_messages}


def should_continue(state: State) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
graph = (
    StateGraph(State)
    .add_node("model", call_model)
    .add_node("tools", run_tools)
    .add_edge(START, "model")
    .add_conditional_edges("model", should_continue, {"tools": "tools", END: END})
    .add_edge("tools", "model")
    .compile()
)


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = "You are a helpful furniture shopping assistant."


def main() -> None:
    print(f"Chat with {MODEL} + tools (LangGraph). Ctrl-C or Ctrl-D to quit.")
    print(f"Tools available: {', '.join(TOOL_REGISTRY)}\n")

    # system message is injected once; history accumulates only human/ai/tool turns
    system_msg = {"role": "system", "content": SYSTEM_PROMPT}
    history: list = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue

        history.append({"role": "user", "content": user_input})
        result = graph.invoke({"messages": [system_msg] + history})
        # keep only the new messages appended this turn (everything after system+prior history)
        history = list(result["messages"][1:])

        final = history[-1]
        print(f"\nAssistant: {final.content}\n")


if __name__ == "__main__":
    main()
