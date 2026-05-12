"""
Middleware Deep-Dive: see EVERY byte crossing orchestration <-> LLM.

Goal: build agent with two tools (search_flights, book_flight) and intercept
each step so you can SEE:

  1. Exact `state["messages"]` Python objects (BaseMessage subclasses).
  2. Exact `ModelRequest` LangGraph hands to the model adapter
     (messages + tool JSON-Schemas + system prompt).
  3. The provider HTTP payload (what really goes on the wire) — JSON dicts
     after langchain's message-to-provider conversion.
  4. The REAL tokenizer view via Ollama's /api/tokenize — actual LLaMA 3.1
     token IDs, not tiktoken approximations.
  5. Token → embedding vector lookup: each token ID maps to a row in the
     model's embedding matrix (shape: [vocab_size=128256, hidden_dim=4096]).
     LLaMA 3's weights are public (Meta license), so you can inspect any
     token's vector. We query Ollama's /api/embed as a proxy; for raw matrix
     access see the "Raw embedding lookup" note at the bottom.
  6. The raw `ModelResponse` (AIMessage) — tool_calls, response_metadata,
     usage_metadata.
  7. The ToolMessage round-trip.

Run (no API key needed — uses local Ollama):
    ollama pull llama3.1:8b
    ollama serve
    uv pip install langchain langchain-openai langchain-anthropic langchain-ollama tiktoken requests
    python middleware_deep_dive.py

With an API key (OpenAI or Anthropic takes priority):
    export OPENAI_API_KEY=sk-...
    python middleware_deep_dive.py
"""

from __future__ import annotations

import json
import os
import textwrap
from typing import Any, Callable

import requests

# ---------------------------------------------------------------------------
# Model selection. OpenAI → Anthropic → LLaMA 3.1 via Ollama (no key needed).
# ---------------------------------------------------------------------------
USE_OPENAI = bool(os.getenv("OPENAI_API_KEY"))
USE_ANTHROPIC = bool(os.getenv("ANTHROPIC_API_KEY"))
OLLAMA_BASE = os.getenv("OLLAMA_HOST", "http://localhost:11434")

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

    MODEL_NAME = "llama3.1:8b"
    model = ChatOllama(model=MODEL_NAME, temperature=0)
    print(f"No API key found — using local Ollama model '{MODEL_NAME}'.")
    print(f"Ollama expected at: {OLLAMA_BASE}")
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
# Pretty printing helpers.
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
# Tools.
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
# Ollama tokenizer helpers — real LLaMA 3.1 token IDs, not approximations.
# ---------------------------------------------------------------------------
def _ollama_tokenize(text: str) -> list[int]:
    """Return real LLaMA 3.1 token IDs for `text` via Ollama's tokenize API."""
    resp = requests.post(
        f"{OLLAMA_BASE}/api/tokenize",
        json={"model": MODEL_NAME, "prompt": text},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["tokens"]


def _ollama_detokenize(token_ids: list[int]) -> str:
    """Convert a list of token IDs back to a string via Ollama's detokenize API."""
    resp = requests.post(
        f"{OLLAMA_BASE}/api/detokenize",
        json={"model": MODEL_NAME, "tokens": token_ids},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["content"]


def _ollama_embed(text: str) -> list[float]:
    """Get the embedding vector for `text` from Ollama."""
    resp = requests.post(
        f"{OLLAMA_BASE}/api/embed",
        json={"model": MODEL_NAME, "input": text},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


# ---------------------------------------------------------------------------
# THE WIRETAP MIDDLEWARE.
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

        banner("Tokenizer view — real LLaMA 3.1 token IDs via Ollama", ch="-")
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
# Provider payload conversion.
# ---------------------------------------------------------------------------
def _to_wire_payload(request: ModelRequest) -> dict[str, Any]:
    """Build the JSON dict that would hit the provider's REST API."""
    from langchain_core.utils.function_calling import convert_to_openai_tool

    try:
        from langchain_openai.chat_models.base import _convert_message_to_dict
    except Exception:
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
                        "function": {"name": tc["name"], "arguments": json.dumps(tc["args"])},
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

    return {
        "model": MODEL_NAME,
        "messages": messages_wire,
        "tools": [convert_to_openai_tool(t) for t in (request.tools or [])],
        "tool_choice": getattr(request, "tool_choice", None) or "auto",
        "temperature": 0,
    }


# ---------------------------------------------------------------------------
# Tokenizer + embedding view.
# ---------------------------------------------------------------------------
def _tokenize_payload(payload: dict[str, Any]) -> None:
    """Show real LLaMA 3.1 token IDs and token → vector mappings.

    KEY MENTAL MODEL (same for every transformer LLM):
      Step 1 — Render: the message list is flattened into one string using
               the model's chat template. For LLaMA 3.1 this is the
               <|begin_of_text|> / <|start_header_id|> / <|end_header_id|>
               / <|eot_id|> format. Tool schemas are injected into the system
               message as JSON text.
      Step 2 — Tokenize: the string is split into subword tokens by BPE.
               LLaMA 3 uses a 128,256-token vocabulary (tiktoken-based, but
               with Meta's custom merges — that's why tiktoken's built-in
               encodings don't match exactly). Each token → one integer ID.
      Step 3 — Embed: the integer IDs index into the embedding matrix
               E ∈ ℝ^{128256 × 4096}. Row E[token_id] is a 4096-dim vector —
               that is the only input the first transformer layer ever sees.
               Because Meta released the weights, you can inspect any row:
                   from transformers import AutoModel
                   m = AutoModel.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
                   E = m.model.embed_tokens.weight          # [128256, 4096]
                   vec = E[token_id].detach().numpy()       # [4096]
    """
    # --- render the chat template the way LLaMA 3.1 expects it -------------
    rendered_parts: list[str] = ["<|begin_of_text|>"]
    for m in payload["messages"]:
        role = m["role"]
        content = m.get("content") or ""
        rendered_parts.append(
            f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
        )
        if "tool_calls" in m:
            for tc in m["tool_calls"]:
                rendered_parts.append(
                    f"<|start_header_id|>tool_call<|end_header_id|>\n\n"
                    f"{tc['function']['name']}({tc['function']['arguments']})<|eot_id|>"
                )
    # Tool schemas go into a special block before the assistant turn
    tool_block = json.dumps(payload.get("tools", []))
    rendered_parts.insert(
        1,
        f"<|start_header_id|>system<|end_header_id|>\n\n[TOOLS]{tool_block}[/TOOLS]<|eot_id|>",
    )
    rendered_parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    rendered = "".join(rendered_parts)

    print("Rendered chat-template string (LLaMA 3.1 format, first 1200 chars):")
    print(textwrap.indent(rendered[:1200] + ("..." if len(rendered) > 1200 else ""), "    "))

    # --- real tokenization via Ollama ---------------------------------------
    if not USE_OPENAI and not USE_ANTHROPIC:
        _show_real_tokens(rendered)
    else:
        # For OpenAI/Anthropic we fall back to tiktoken as an approximation
        _show_tiktoken_approx(rendered)


def _show_real_tokens(rendered: str) -> None:
    """Use Ollama's /api/tokenize + /api/detokenize for exact LLaMA 3.1 IDs."""
    try:
        ids = _ollama_tokenize(rendered)
    except Exception as e:
        print(f"\n(Ollama tokenize failed: {e} — is Ollama running?)")
        return

    print(f"\nTotal tokens (exact, from Ollama): {len(ids)}")
    print("First 40 token IDs:", ids[:40])

    print("\nFirst 40 tokens decoded individually (Ollama detokenize):")
    for i, tid in enumerate(ids[:40]):
        try:
            piece = _ollama_detokenize([tid])
        except Exception:
            piece = "?"
        print(f"  [{i:>2}]  id={tid:>7}  decoded={piece!r}")

    print(
        "\nTakeaways:"
        "\n  * No 'messages' at the model layer — one flat token sequence."
        "\n  * LLaMA 3.1 role boundaries: <|start_header_id|>user<|end_header_id|>"
        "\n    These ARE tokens (special IDs the model learned during training)."
        "\n  * Tool schemas are plain JSON text tokens; the model generates a"
        "\n    tool call by emitting tokens matching a learned format."
    )

    banner("Token → embedding vector (first 5 tokens via Ollama /api/embed)", ch="-")
    _show_token_embeddings(ids[:5])


def _show_token_embeddings(token_ids: list[int]) -> None:
    """Show the embedding vector for each token.

    Ollama's /api/embed runs the full model encoder, so the vectors are
    contextualised. For the raw embedding-matrix row (E[token_id] with no
    context), load the weights directly:

        from transformers import AutoModel
        m = AutoModel.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
        E = m.model.embed_tokens.weight          # shape [128256, 4096]
        vec = E[token_id].detach().float().numpy()
    """
    print(
        "Note: vectors below come from Ollama /api/embed (contextualised).\n"
        "For the raw embedding-matrix row, see the docstring above.\n"
    )
    for tid in token_ids:
        try:
            decoded = _ollama_detokenize([tid])
        except Exception:
            decoded = "?"
        try:
            vec = _ollama_embed(decoded if decoded.strip() else " ")
            dim = len(vec)
            preview = [round(v, 4) for v in vec[:6]]
            print(f"  id={tid:>7}  token={decoded!r:<15}  dim={dim}  vec[:6]={preview}")
        except Exception as e:
            print(f"  id={tid:>7}  token={decoded!r:<15}  embed failed: {e}")


def _show_tiktoken_approx(rendered: str) -> None:
    """Fallback for OpenAI/Anthropic: tiktoken approximation."""
    try:
        import tiktoken
    except ImportError:
        print("tiktoken not installed; pip install tiktoken for token view.")
        return

    if USE_OPENAI:
        try:
            enc = tiktoken.encoding_for_model(MODEL_NAME)
        except KeyError:
            enc = tiktoken.get_encoding("o200k_base")
    else:
        print("Anthropic tokenizer is proprietary; using cl100k_base as approximation.")
        enc = tiktoken.get_encoding("cl100k_base")

    ids = enc.encode(rendered, disallowed_special=())
    print(f"\nTotal tokens (approx via tiktoken): {len(ids)}")
    print("First 40 token IDs:", ids[:40])
    print("First 40 tokens decoded individually:")
    for tid in ids[:40]:
        piece = enc.decode([tid])
        print(f"  {tid:>7}  {piece!r}")


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
