"""
chat.py — minimal command-line chat using LangGraph + Ollama (llama3.1:8b).

Run:
    python chat.py
    python chat.py --model llama3.2:3b   # override model

Type your message and press Enter. Ctrl-C or Ctrl-D to quit.
"""

from __future__ import annotations

import sys
from typing import Annotated

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


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class State(TypedDict):
    messages: Annotated[list, add_messages]


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
llm = ChatOllama(model=MODEL, temperature=0)


def call_model(state: State) -> State:
    return {"messages": [llm.invoke(state["messages"])]}


graph = (
    StateGraph(State)
    .add_node("model", call_model)
    .add_edge(START, "model")
    .add_edge("model", END)
    .compile()
)


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"Chat with {MODEL} (LangGraph). Ctrl-C or Ctrl-D to quit.\n")
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
        result = graph.invoke({"messages": history})
        history = result["messages"]

        reply = history[-1].content
        print(f"\nAssistant: {reply}\n")


if __name__ == "__main__":
    main()
