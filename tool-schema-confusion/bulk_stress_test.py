"""
bulk_stress_test.py — reliability test for complex tool schemas.

For each of the 4 tools (D1–D4), runs 5 different prompts × 5 times each
= 25 attempts per tool, 100 total.  Uses temperature=0.7 so repetitions of
the same prompt produce genuine variation.

Success criteria per attempt:
  1. Model called at least one tool.
  2. The tool called matches the expected tool name.
  3. ToolNode executed without raising an error.

Prints a live progress ticker while running, then a full summary table.

Run:
    python bulk_stress_test.py
    python bulk_stress_test.py --model llama3.2:3b
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Annotated

from langchain_core.messages import AIMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

# import tool definitions from the companion file
sys.path.insert(0, os.path.dirname(__file__))
from schema_stress_test import (  # noqa: E402
    book_restaurant,
    deploy_application,
    place_order,
    plan_vacation,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MODEL = next(
    (sys.argv[sys.argv.index("--model") + 1] for _ in ["x"] if "--model" in sys.argv),
    "llama3.1:8b",
)

RUNS_PER_PROMPT = 5
BAR_WIDTH = 25


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class State(TypedDict):
    messages: Annotated[list, add_messages]


# ---------------------------------------------------------------------------
# Test definitions — 5 prompts per tool
# ---------------------------------------------------------------------------
TESTS = [
    {
        "tool": book_restaurant,
        "degree": 1,
        "prompts": [
            "Book a table for 4 at 'The Golden Fork' on 2026-07-15 at 19:30. Outdoor seating please.",
            "Reserve a spot for 2 at 'Sakura Garden' on 2026-06-11 at 20:00. Indoor is fine.",
            "Can you book La Bella Italia for 6 guests on 2026-08-01 at 13:00? We want outdoor seating.",
            "Please reserve The Steakhouse for 8 people on 2026-09-20 at 18:45. No outdoor seating.",
            "Get me a table at Café Paris for 3 on 2026-07-04 at 12:30. Outdoor seating preferred.",
        ],
    },
    {
        "tool": plan_vacation,
        "degree": 2,
        "prompts": [
            "Plan 10 days in Tokyo for Alice Smith, age 29, passport AB123456. Budget 3000 USD. Hotel, culture-focused.",
            "Plan a 7-day trip to Paris for Bob Jones, 45 years old, passport CD789012. 5000 EUR, airbnb, adventure.",
            "Organise 14 days in Bali for Carol Lee, age 33, passport EF345678. 2000 USD, hostel, relaxation.",
            "I need a 5-day New York trip for David Park, 27, passport GH901234. 1500 USD, hotel, culture.",
            "Set up 21 days in Australia for Emma Wilson, 52, passport IJ567890. 8000 AUD, hotel, adventure.",
        ],
    },
    {
        "tool": place_order,
        "degree": 3,
        "prompts": [
            "Order 2 units of PROD-789. Ship to 42 Maple St, Springfield, USA, 62701, express, leave at door. Card 4111111111111111 exp 09/27 CVV 321.",
            "Buy 1 of ITEM-456. Deliver to 15 Oak Ave, London, UK, EC1A 1BB, standard shipping, ring the bell. Visa 5555555555554444 exp 12/25 CVV 123.",
            "Purchase 3x SKU-101. Address: 100 Baker St, Manchester, UK, M1 1AD, express, leave at door. Card 4242424242424242 exp 03/28 CVV 456.",
            "Order 5 of GADGET-222. Ship to 7 Pine Rd, Toronto, Canada, M5V 3A8, standard, don't leave unattended. Mastercard 378282246310005 exp 11/26 CVV 7890.",
            "Get me 10x BULK-333. 88 Elm St, Sydney, Australia, 2000, express, leave at door. Card 2223003122003222 exp 06/29 CVV 111.",
        ],
    },
    {
        "tool": deploy_application,
        "degree": 4,
        "prompts": [
            (
                "Deploy 'payments-service' image tag v2.5.1. Cluster: prod-cluster in us-central1. "
                "Node pool: 3 nodes, n2-standard-4, autoscaling enabled min 2 max 10, "
                "cpu_utilization target 70%, cooldown 120s. "
                "Resources: CPU request 250m limit 1, memory request 512Mi limit 1Gi."
            ),
            (
                "Deploy 'auth-service' tag v1.0.0. Cluster: staging-cluster in eu-west1. "
                "Node pool: 2 nodes, e2-standard-2, autoscaling enabled min 1 max 5, "
                "cpu_utilization target 80%, cooldown 60s. "
                "Resources: CPU request 100m limit 500m, memory request 256Mi limit 512Mi."
            ),
            (
                "Deploy 'frontend' image v3.1.0 to dev-cluster in us-east1. "
                "Node pool: 1 node, n1-standard-1, autoscaling enabled min 1 max 3, "
                "memory_utilization target 75%, cooldown 90s. "
                "Resources: CPU request 50m limit 200m, memory request 128Mi limit 256Mi."
            ),
            (
                "Push 'worker-service' v0.9.5 to prod-cluster in asia-east1. "
                "Node pool: 5 nodes, n2-standard-8, autoscaling enabled min 3 max 15, "
                "cpu_utilization 60%, cooldown 180s. "
                "Resources: CPU request 500m limit 2, memory request 1Gi limit 4Gi."
            ),
            (
                "Deploy 'data-pipeline' v4.2.3 to prod-cluster in us-west2. "
                "Node pool: 4 nodes, c2-standard-4, autoscaling enabled min 2 max 8, "
                "cpu_utilization 85%, cooldown 240s. "
                "Resources: CPU request 1 limit 4, memory request 2Gi limit 8Gi."
            ),
        ],
    },
]


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------
@dataclass
class AttemptResult:
    success: bool
    reason: str          # "ok" or short failure description
    elapsed_s: float


@dataclass
class PromptResult:
    prompt: str
    attempts: list[AttemptResult] = field(default_factory=list)

    @property
    def successes(self) -> int:
        return sum(1 for a in self.attempts if a.success)


@dataclass
class ToolResult:
    tool_name: str
    degree: int
    prompts: list[PromptResult] = field(default_factory=list)

    @property
    def total_successes(self) -> int:
        return sum(p.successes for p in self.prompts)

    @property
    def total_attempts(self) -> int:
        return sum(len(p.attempts) for p in self.prompts)


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------
def build_graph(tool) -> StateGraph:
    llm = ChatOllama(model=MODEL, temperature=0.7).bind_tools([tool])

    def call_model(state: State) -> State:
        return {"messages": [llm.invoke(state["messages"])]}

    tool_node = ToolNode([tool], handle_tool_errors=True)

    return (
        StateGraph(State)
        .add_node("model", call_model)
        .add_node("tools", tool_node)
        .add_edge(START, "model")
        .add_conditional_edges("model", tools_condition)
        .add_edge("tools", "model")
        .compile()
    )


# ---------------------------------------------------------------------------
# Single attempt
# ---------------------------------------------------------------------------
def run_attempt(graph, tool, system_msg: dict, user_prompt: str) -> AttemptResult:
    t0 = time.time()
    try:
        result = graph.invoke({"messages": [system_msg, {"role": "user", "content": user_prompt}]})
    except Exception as exc:
        return AttemptResult(False, f"graph error: {exc}", time.time() - t0)

    messages = result["messages"]
    elapsed = time.time() - t0

    # find the first AIMessage that has tool_calls
    ai_with_calls = next(
        (m for m in messages if isinstance(m, AIMessage) and m.tool_calls),
        None,
    )
    if ai_with_calls is None:
        return AttemptResult(False, "no tool called", elapsed)

    called = ai_with_calls.tool_calls[0]["name"]
    if called != tool.__name__:
        return AttemptResult(False, f"wrong tool: {called!r}", elapsed)

    # check ToolMessages for error content from handle_tool_errors
    for m in messages:
        if isinstance(m, ToolMessage):
            content = m.content if isinstance(m.content, str) else json.dumps(m.content)
            if content.strip().startswith("Error"):
                return AttemptResult(False, f"tool error: {content[:80]}", elapsed)

    return AttemptResult(True, "ok", elapsed)


# ---------------------------------------------------------------------------
# Progress bar
# ---------------------------------------------------------------------------
def bar(successes: int, total: int, width: int = BAR_WIDTH) -> str:
    filled = round(width * successes / total) if total else 0
    return "[" + "#" * filled + "." * (width - filled) + "]"


def pct(successes: int, total: int) -> str:
    return f"{100 * successes // total:>3}%" if total else "  -%"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"Model : {MODEL}")
    print(f"Plan  : {len(TESTS)} tools × 5 prompts × {RUNS_PER_PROMPT} runs = "
          f"{len(TESTS) * 5 * RUNS_PER_PROMPT} total attempts\n")

    all_tool_results: list[ToolResult] = []

    for test in TESTS:
        tool      = test["tool"]
        degree    = test["degree"]
        prompts   = test["prompts"]
        tool_res  = ToolResult(tool_name=tool.__name__, degree=degree)

        print(f"{'=' * 70}")
        print(f"D{degree}  {tool.__name__}")
        print(f"{'=' * 70}")

        graph      = build_graph(tool)
        system_msg = {
            "role": "system",
            "content": "You are a helpful assistant. Call the appropriate tool with all required arguments.",
        }

        for p_idx, prompt in enumerate(prompts, 1):
            p_result = PromptResult(prompt=prompt)
            short    = prompt[:60] + ("..." if len(prompt) > 60 else "")
            print(f"\n  P{p_idx}: {short!r}")

            for r_idx in range(1, RUNS_PER_PROMPT + 1):
                attempt = run_attempt(graph, tool, system_msg, prompt)
                p_result.attempts.append(attempt)

                tick  = "+" if attempt.success else "x"
                label = "ok" if attempt.success else attempt.reason[:40]
                print(f"    run {r_idx}/{RUNS_PER_PROMPT}  {tick}  {label}  ({attempt.elapsed_s:.1f}s)")

            sub = p_result.successes
            print(f"  -> {sub}/{RUNS_PER_PROMPT}  {bar(sub, RUNS_PER_PROMPT)}")
            tool_res.prompts.append(p_result)

        all_tool_results.append(tool_res)
        tot = tool_res.total_successes
        mx  = tool_res.total_attempts
        print(f"\n  SUBTOTAL  {tot}/{mx}  {bar(tot, mx)}  {pct(tot, mx)}\n")

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Degree':<8}{'Tool':<26}{'Score':<10}{'Bar':<{BAR_WIDTH + 3}}{'%':>4}")
    print("-" * 70)
    for tr in all_tool_results:
        s = tr.total_successes
        t = tr.total_attempts
        print(f"D{tr.degree:<7}{tr.tool_name:<26}{s:>2}/{t:<7}{bar(s, t)}  {pct(s, t)}")

    grand_s = sum(tr.total_successes for tr in all_tool_results)
    grand_t = sum(tr.total_attempts  for tr in all_tool_results)
    print("-" * 70)
    print(f"{'TOTAL':<8}{'':<26}{grand_s:>2}/{grand_t:<7}{bar(grand_s, grand_t)}  {pct(grand_s, grand_t)}")

    print("\nPer-prompt breakdown:")
    for tr in all_tool_results:
        print(f"\n  D{tr.degree}  {tr.tool_name}:")
        for i, pr in enumerate(tr.prompts, 1):
            s     = pr.successes
            short = pr.prompt[:55] + ("..." if len(pr.prompt) > 55 else "")
            print(f"    P{i}  {short!r:<60}  {s}/{RUNS_PER_PROMPT}  {bar(s, RUNS_PER_PROMPT, 10)}")

    print()


if __name__ == "__main__":
    main()
