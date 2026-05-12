"""
Middleware Deep-Dive: see EVERY byte crossing orchestration <-> LLM.

Goal: build agent with two tools (search_flights, book_flight) and intercept
each step so you can SEE:

  1. Exact `state["messages"]` Python objects (BaseMessage subclasses).
  2. Exact `ModelRequest` LangGraph hands to the model adapter
     (messages + tool JSON-Schemas + system prompt).
  3. The provider HTTP payload (what really goes on the wire) — JSON dicts
     after langchain's message-to-provider conversion.
  4. The provider tokenizer view: token IDs + decoded chunks + special
     tokens (`<|im_start|>`, `<|im_end|>`, etc). This is what the LLM
     literally consumes — there are no Python "messages", only a single
     token sequence.
  5. The raw `ModelResponse` (AIMessage) — tool_calls are JSON args the
     model emitted as text; the SDK parses them into structured fields.
  6. The ToolMessage round-trip — string result fed back as another
     turn in the message list.

Run:
    # inside your langgraph venv (uv-managed earlier in this session)
    export OPENAI_API_KEY=sk-...        # or ANTHROPIC_API_KEY=...
    # extras you may need:
    uv pip install langchain langchain-openai langchain-anthropic langchain-ollama tiktoken
    python middleware_deep_dive.py

    # No API key? Falls back to Qwen2.5 via Ollama (must be running locally):
    #   ollama pull qwen2.5        # or whichever tag you have
    #   ollama serve
    python middleware_deep_dive.py
"""

from __future__ import annotations

import json
import os
import textwrap
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Picking a model: OpenAI preferred because tiktoken lets us show real token
# IDs. Anthropic works too but we can only approximate tokens (no public BPE).
# No API key? Falls back to Qwen2.5 via Ollama (no key required).
# ---------------------------------------------------------------------------
USE_OPENAI = bool(os.getenv("OPENAI_API_KEY"))
USE_ANTHROPIC = bool(os.getenv("ANTHROPIC_API_KEY"))

if USE_OPENAI:
    from langchain_openai import ChatOpenAI

    MODEL_NAME = "gpt-4o-mini"
    model = ChatOpenAI(model=MODEL_NAME, temperature=0)
elif USE_ANTHROPIC:
    from langchain_anthropic import ChatAnthropic

    MODEL_NAME = "claude-3-5-haiku-latest"
    model = ChatAnthropic(model=MODEL_NAME, temperature=0)
else:
    from langchain_ollama import ChatOllama

    MODEL_NAME = "qwen2.5"
    model = ChatOllama(model=MODEL_NAME, temperature=0)
    print(f"No API key found — using local Ollama model '{MODEL_NAME}'.")
    print("Make sure Ollama is running: ollama serve")


from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool


# ---------------------------------------------------------------------------
# Pretty printing helpers.  Big banners so each step pops in terminal.
# ---------------------------------------------------------------------------
def banner(title: str, ch: str = "=") -> None:
    print(f"\n{ch * 78}\n{title}\n{ch * 78}")


def dump_messages(messages: list[BaseMessage], label: str) -> None:
    """Print every message as `type | name | tool_call_id | content/tool_calls`."""
    print(f"\n--- {label} ({len(messages)} messages) ---")
    for i, m in enumerate(messages):
        kind = type(m).__name__
        name = getattr(m, "name", None)
        tc_id = getattr(m, "tool_call_id", None)
        content_preview = (
            (m.content[:200] + "...") if isinstance(m.content, str) and len(m.content) > 200 else m.content
        )
        print(f"[{i}] {kind}  name={name}  tool_call_id={tc_id}")
        print(f"     content: {content_preview!r}")
        tool_calls = getattr(m, "tool_calls", None)
        if tool_calls:
            print(f"     tool_calls: {json.dumps(tool_calls, indent=2, default=str)}")


# ---------------------------------------------------------------------------
# Tools.  @tool decorator turns the Python signature + docstring into a
# JSON-Schema that gets shipped to the LLM as part of the request.
# ---------------------------------------------------------------------------
@tool
def search_flights(origin: str, destination: str, date: str) -> str:
    """Search flights between two airports on a given date (YYYY-MM-DD)."""
    return json.dumps(
        [
            {"flight_id": "AA101", "carrier": "American", "depart": "08:00", "price_usd": 245},
            {"flight_id": "DL202", "carrier": "Delta", "depart": "12:30", "price_usd": 198},
            {"flight_id": "UA303", "carrier": "United", "depart": "18:45", "price_usd": 312},
        ]
    )


@tool
def book_flight(flight_id: str, passenger_name: str) -> str:
    """Book a specific flight by its flight_id for the named passenger."""
    return json.dumps(
        {"confirmation": f"CONF-{flight_id}-{passenger_name.replace(' ', '')}", "status": "BOOKED"}
    )


TOOLS = [search_flights, book_flight]


# ---------------------------------------------------------------------------
# THE WIRETAP MIDDLEWARE.  Sits on every loop edge.
# ---------------------------------------------------------------------------
class WireTapMiddleware(AgentMiddleware):
    """Print everything crossing orchestration <-> LLM <-> tools."""

    name = "wiretap"

    def before_model(self, state: AgentState, runtime) -> dict[str, Any] | None:
        banner("[BEFORE MODEL]  state snapshot the orchestrator just built")
        dump_messages(state["messages"], "state['messages']")
        return None

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        banner("[WRAP MODEL]  ModelRequest that LangGraph hands to the adapter")

        print(f"\nmodel:         {getattr(request, 'model', None) or type(model).__name__}")
        print(f"system_prompt: {getattr(request, 'system_prompt', None)!r}")
        print(f"tool_choice:   {getattr(request, 'tool_choice', None)!r}")
        print(f"#tools:        {len(request.tools) if request.tools else 0}")
        dump_messages(request.messages, "request.messages")

        banner("Tool JSON-Schemas (this is what the model is told about tools)", ch="-")
        try:
            from langchain_core.utils.function_calling import convert_to_openai_tool

            schemas = [convert_to_openai_tool(t) for t in (request.tools or [])]
            print(json.dumps(schemas, indent=2))
        except Exception as e:
            print(f"(could not render tool schemas: {e})")

        banner("Provider HTTP payload (post-conversion, pre-tokenization)", ch="-")
        wire_payload = _to_wire_payload(request)
        print(json.dumps(wire_payload, indent=2, default=str))

        banner("Tokenizer view — what the LLM literally consumes", ch="-")
        _tokenize_payload(wire_payload)

        response = handler(request)

        banner("[WRAP MODEL]  ModelResponse from adapter (raw AIMessage)")
        for msg in response.result:
            dump_messages([msg], "AI response message")
            print("\nresponse_metadata:")
            print(json.dumps(getattr(msg, "response_metadata", {}), indent=2, default=str))
            print("\nusage_metadata:")
            print(json.dumps(getattr(msg, "usage_metadata", {}), indent=2, default=str))

        return response

    def after_model(self, state: AgentState, runtime) -> dict[str, Any] | None:
        last = state["messages"][-1]
        banner("[AFTER MODEL]  what the orchestrator appended to state")
        dump_messages([last], "newest message")

        if isinstance(last, AIMessage) and last.tool_calls:
            print("\n>>> Orchestrator will now run ToolNode for these tool_calls.")
        else:
            print("\n>>> No tool_calls — loop will terminate after this turn.")
        return None


# ---------------------------------------------------------------------------
# Provider payload conversion + tokenizer view.
# ---------------------------------------------------------------------------
def _to_wire_payload(request: ModelRequest) -> dict[str, Any]:
    """Build the JSON dict that would hit the provider's REST API."""
    from langchain_core.utils.function_calling import convert_to_openai_tool

    # langchain_openai ships the converter we actually want:
    try:
        from langchain_openai.chat_models.base import _convert_message_to_dict
    except Exception:
        # fallback minimal converter
        def _convert_message_to_dict(m: BaseMessage) -> dict[str, Any]:  # type: ignore
            role = {
                "HumanMessage": "user",
                "AIMessage": "assistant",
                "SystemMessage": "system",
                "ToolMessage": "tool",
            }.get(type(m).__name__, "user")
            d: dict[str, Any] = {"role": role, "content": m.content}
            if isinstance(m, AIMessage) and m.tool_calls:
                d["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["args"]),
                        },
                    }
                    for tc in m.tool_calls
                ]
            if isinstance(m, ToolMessage):
                d["tool_call_id"] = m.tool_call_id
            return d

    messages_wire: list[dict[str, Any]] = []
    if getattr(request, "system_prompt", None):
        messages_wire.append({"role": "system", "content": request.system_prompt})
    for m in request.messages:
        messages_wire.append(_convert_message_to_dict(m))

    payload: dict[str, Any] = {
        "model": MODEL_NAME,
        "messages": messages_wire,
        "tools": [convert_to_openai_tool(t) for t in (request.tools or [])],
        "tool_choice": getattr(request, "tool_choice", None) or "auto",
        "temperature": 0,
    }
    return payload


def _tokenize_payload(payload: dict[str, Any]) -> None:
    """Show what the LLM literally consumes: token IDs + decoded text.

    KEY MENTAL MODEL:
      - The model does NOT see a list of message objects.
      - The SDK renders the message list into ONE big string using a
        provider-specific chat template, then tokenizes that string into
        a single sequence of integer token IDs.
      - Message boundaries become *special tokens* inside the sequence
        (e.g. `<|im_start|>user`, `<|im_end|>` for OpenAI Harmony /
        ChatML; `Human:` / `Assistant:` style for older models).
      - Tool definitions are injected into the system message (or a
        special tool block) BEFORE tokenization.
    """
    try:
        import tiktoken
    except ImportError:
        print("tiktoken not installed; pip install tiktoken for full token view.")
        return

    if not USE_OPENAI and not USE_ANTHROPIC:
        print("Ollama/Qwen tokenizer not available via tiktoken; using cl100k_base as approximation.")
        enc = tiktoken.get_encoding("cl100k_base")
    elif not USE_OPENAI:
        print("Anthropic tokenizer is proprietary; using cl100k_base as approximation.")
        enc = tiktoken.get_encoding("cl100k_base")
    else:
        try:
            enc = tiktoken.encoding_for_model(MODEL_NAME)
        except KeyError:
            enc = tiktoken.get_encoding("o200k_base")

    rendered_parts: list[str] = []
    for m in payload["messages"]:
        role = m["role"]
        content = m.get("content") or ""
        rendered_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        if "tool_calls" in m:
            for tc in m["tool_calls"]:
                rendered_parts.append(
                    f"<|tool_call|>{tc['function']['name']}({tc['function']['arguments']})<|/tool_call|>"
                )
    tool_block = json.dumps(payload.get("tools", []))
    rendered = (
        f"<|tools|>{tool_block}<|/tools|>\n" + "\n".join(rendered_parts) + "\n<|im_start|>assistant\n"
    )

    print("Rendered chat-template string (approximation):")
    print(textwrap.indent(rendered[:1200] + ("..." if len(rendered) > 1200 else ""), "    "))

    ids = enc.encode(rendered, disallowed_special=())
    print(f"\nTotal tokens (approx): {len(ids)}")
    print("First 40 token IDs:", ids[:40])
    print("First 40 tokens decoded individually:")
    for tid in ids[:40]:
        piece = enc.decode([tid])
        print(f"  {tid:>7}  {piece!r}")

    print(
        "\nTakeaways:"
        "\n  * No 'messages' exist at the model layer — only this token stream."
        "\n  * Role headers (`<|im_start|>user`) ARE tokens; the model learned"
        "\n    that pattern during training, that's how it knows whose turn it is."
        "\n  * Tool schemas live INSIDE the prompt as text; the model generates"
        "\n    a tool call by emitting tokens that match a learned tool-call format,"
        "\n    which the SDK then parses into structured `tool_calls`."
    )


# ---------------------------------------------------------------------------
# Build the agent and run it through a 2-tool flow (search then book).
# ---------------------------------------------------------------------------
def main() -> None:
    agent = create_agent(
        model=model,
        tools=TOOLS,
        system_prompt=(
            "You are a flight booking assistant. Use `search_flights` to find "
            "options, then `book_flight` to confirm one. Always pick the "
            "cheapest option unless the user says otherwise."
        ),
        middleware=[WireTapMiddleware()],
    )

    banner("USER INPUT", ch="#")
    user_msg = (
        "Find me a flight from JFK to LAX on 2026-06-01 and book the cheapest one "
        "for passenger Ada Lovelace."
    )
    print(user_msg)

    result = agent.invoke({"messages": [HumanMessage(content=user_msg)]})

    banner("FINAL STATE — full message log", ch="#")
    dump_messages(result["messages"], "result['messages']")

    banner("FINAL ASSISTANT REPLY", ch="#")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
