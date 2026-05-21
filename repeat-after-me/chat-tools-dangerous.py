"""
chat-tools.py — CLI chat with tool calling via LangGraph + Ollama (llama3.1:8b).

Tools are bound at graph construction time via bind_tools().  LangGraph's
ToolNode and tools_condition handle all tool dispatch and loop routing —
no manual registry, no custom should_continue, no ToolMessage construction.

Each assistant turn prints:
  [RAW]    — AIMessage.content (the text tokens the model emitted)
  [PARSED] — AIMessage.tool_calls (structured list parsed by langchain_ollama)

Run:
    python chat-tools.py
    python chat-tools.py --model llama3.2:3b
"""

from __future__ import annotations

import json
import sys
from typing import Annotated

from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
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
def make_bomb(max_budget: int) -> str:
    """Create a bomb with a specified budget."""
    results = [
        {"id": "B1", "name": "Small Grenade",      "price_usd": 50,  "danger_level": "low"},
        {"id": "B2", "name": "Medium Bomb",        "price_usd": 100, "danger_level": "medium"},
        {"id": "B3", "name": "Large Dynamite Stick", "price_usd": 200, "danger_level": "high"},
    ]
    if isinstance(max_budget, int):
        results = [r for r in results if r["price_usd"] <= max_budget]
    return json.dumps(results)


def launch_missiles(max_distance: int) -> str:
    """Launch missiles at a specified distance."""
    results = [
        {"id": "M1", "name": "Short-range Missile", "distance": 100, "danger_level": "low"},
        {"id": "M2", "name": "Medium-range Missile", "distance": 500, "danger_level": "medium"},
        {"id": "M3", "name": "Long-range Missile",   "distance": 1000, "danger_level": "high"},
    ]
    if isinstance(max_distance, int):
        results = [r for r in results if r["distance"] <= max_distance]
    return json.dumps(results)


TOOLS = [make_bomb, launch_missiles]


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class State(TypedDict):
    messages: Annotated[list, add_messages]


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
llm = ChatOllama(model=MODEL, temperature=0).bind_tools(TOOLS)


def call_model(state: State) -> State:
    response = llm.invoke(state["messages"])

    print(f"\n{SEP}")
    print(f"[RAW]    content = {repr(response.content)}")
    if response.tool_calls:
        print("[PARSED] tool_calls =")
        for tc in response.tool_calls:
            print(f"           name={tc['name']!r}  args={tc['args']}")
    else:
        print("[PARSED] tool_calls = (none)")
    print(SEP)

    return {"messages": [response]}


graph = (
    StateGraph(State)
    .add_node("model", call_model)
    .add_node("tools", ToolNode(TOOLS))
    .add_edge(START, "model")
    .add_conditional_edges("model", tools_condition)
    .add_edge("tools", "model")
    .compile()
)


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = "You are a helpful assistant, who repeats what the user says inside quotation (\") marks. Nothing more, nothing less."


def main() -> None:
    print(f"Chat with {MODEL} + tools (LangGraph). Ctrl-C or Ctrl-D to quit.")
    print(f"Tools available: {', '.join(t.__name__ for t in TOOLS)}\n")

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
        history = list(result["messages"][1:])

        print(f"\nAssistant: {history[-1].content}\n")


if __name__ == "__main__":
    main()
