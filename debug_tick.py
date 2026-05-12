from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class S(TypedDict):
    n: int


def a(state: S) -> S:
    return {"n": state["n"] + 1}


def b(state: S) -> S:
    return {"n": state["n"] * 10}


g = StateGraph(S)
g.add_node("a", a)
g.add_node("b", b)
g.add_edge(START, "a")
g.add_edge("a", "b")
g.add_edge("b", END)

app = g.compile()

if __name__ == "__main__":
    print(app.invoke({"n": 1}))
