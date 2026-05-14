"""
ollama_direct.py — raw Ollama /api/chat calls, zero LangGraph/LangChain.

Replicates exactly what LangGraph does under the hood for the flight-booking
workflow: build the JSON payload, POST to Ollama, parse tool_calls from the
raw HTTP response, execute the tools, feed results back, get the final reply.

The request JSON structure mirrors what middleware_deep_dive.py logs as
"Provider HTTP payload (post-conversion, pre-tokenization)".

Run:
    ollama serve          # make sure llama3.1:8b is pulled
    python ollama_direct.py          # full agentic loop via /api/chat
    python ollama_direct.py --raw    # single shot via /api/generate, no parsing
"""

from __future__ import annotations

import json
import sys
import uuid

import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

OLLAMA_BASE = "http://localhost:11434"
MODEL = "llama3.1:8b"

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------
def search_flights(origin: str, destination: str, date: str) -> str:
    """Search flights between two airports on a given date (YYYY-MM-DD)."""
    return json.dumps([
        {"flight_id": "AA101", "carrier": "American", "depart": "08:00", "price_usd": 245},
        {"flight_id": "DL202", "carrier": "Delta",    "depart": "12:30", "price_usd": 198},
        {"flight_id": "UA303", "carrier": "United",   "depart": "18:45", "price_usd": 312},
    ])


def book_flight(flight_id: str, passenger_name: str) -> str:
    """Book a specific flight by its flight_id for the named passenger."""
    return json.dumps({
        "confirmation": f"CONF-{flight_id}-{passenger_name.replace(' ', '')}",
        "status": "BOOKED",
    })


TOOL_REGISTRY = {
    "search_flights": search_flights,
    "book_flight":    book_flight,
}

# ---------------------------------------------------------------------------
# Tool schemas — OpenAI format, identical to what LangGraph sends
# (copied from output_logs/middleware_deep_dive-llama3.1-hf-tokenizer.out)
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": "Search flights between two airports on a given date (YYYY-MM-DD).",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin":      {"type": "string"},
                    "destination": {"type": "string"},
                    "date":        {"type": "string"},
                },
                "required": ["origin", "destination", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_flight",
            "description": "Book a specific flight by its flight_id for the named passenger.",
            "parameters": {
                "type": "object",
                "properties": {
                    "flight_id":      {"type": "string"},
                    "passenger_name": {"type": "string"},
                },
                "required": ["flight_id", "passenger_name"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Ollama /api/chat — single POST, returns full response JSON
# ---------------------------------------------------------------------------
def ollama_chat(messages: list[dict], tools: list[dict] | None = None) -> dict:
    """POST to Ollama /api/chat and return the parsed response JSON.

    Ollama response shape:
    {
      "model": "llama3.1:8b",
      "message": {
        "role": "assistant",
        "content": "",                  # empty when tool_calls are present
        "tool_calls": [                 # only present when model wants to call a tool
          {
            "id": "call_oz7y4epv",     # Ollama >= 0.23 includes this
            "function": {
              "name": "search_flights",
              "arguments": {"origin": "JFK", "destination": "LAX", "date": "..."}
            }
          }
        ]
      },
      "done": true,
      "done_reason": "stop",
      "prompt_eval_count": 286,
      "eval_count": 74,
      ...
    }
    """
    payload = {
        "model": MODEL,
        "messages": messages,
        "tools": tools or [],
        "tool_choice": "auto",
        "temperature": 0,
        "stream": False,        # get one JSON blob, not a stream of chunks
    }

    print("\n" + "=" * 70)
    print("REQUEST  ->  POST /api/chat")
    print("=" * 70)
    print(json.dumps(payload, indent=2))

    resp = requests.post(f"{OLLAMA_BASE}/api/chat", json=payload, timeout=120)
    resp.raise_for_status()
    raw = resp.json()

    print("\n" + "=" * 70)
    print("RAW RESPONSE  <-  Ollama")
    print("=" * 70)
    print(json.dumps(raw, indent=2))

    return raw


# ---------------------------------------------------------------------------
# Raw generation — bypasses /api/chat tool parsing entirely
# ---------------------------------------------------------------------------
def render_llama_prompt(messages: list[dict], tools: list[dict]) -> str:
    """Render the LLaMA 3.1 chat template manually, the same way Ollama does
    internally before feeding it to the model.

    This is the exact string the model tokenizes and runs on. Tool schemas are
    injected as plain JSON text inside a [TOOLS]...[/TOOLS] system block —
    which is why the model can 'see' tools at all; it's just text tokens.
    """
    parts = ["<|begin_of_text|>"]

    # inject tools as a system block before the first message
    if tools:
        tool_json = json.dumps(tools)
        parts.append(
            f"<|start_header_id|>system<|end_header_id|>\n\n"
            f"[TOOLS]{tool_json}[/TOOLS]<|eot_id|>"
        )

    for m in messages:
        role = m["role"]
        content = m.get("content") or ""
        parts.append(
            f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
        )

    # leave the assistant header open so the model completes from here
    parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(parts)


def ollama_generate_raw(messages: list[dict], tools: list[dict]) -> str:
    """POST to /api/generate with a manually rendered prompt.

    Unlike /api/chat, this endpoint does NO tool-call parsing — you get back
    the raw text tokens the model emitted. For LLaMA 3.1 a tool call looks
    something like:

        <|python_tag|>{"name": "search_flights", "parameters": {...}}

    or in newer fine-tunes, a plain JSON blob. Ollama's /api/chat detects
    this pattern and converts it into the structured tool_calls field you
    normally see — this function lets you observe the text before that step.
    """
    prompt = render_llama_prompt(messages, tools)

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "raw": True,        # skip Ollama's own template rendering — we did it
        "stream": False,
        "temperature": 0,
    }

    print("\n" + "=" * 70)
    print("RAW PROMPT  ->  POST /api/generate  (manually rendered chat template)")
    print("=" * 70)
    print(prompt)

    resp = requests.post(f"{OLLAMA_BASE}/api/generate", json=payload, timeout=120)
    resp.raise_for_status()
    result = resp.json()

    raw_text = result.get("response", "")
    print("\n" + "=" * 70)
    print("RAW MODEL OUTPUT  <-  /api/generate  (unparsed token text)")
    print("=" * 70)
    print(repr(raw_text))
    print("\n(plain view):")
    print(raw_text)

    return raw_text


# ---------------------------------------------------------------------------
# Parse tool_calls out of the raw Ollama response
# ---------------------------------------------------------------------------
def parse_tool_calls(raw: dict) -> list[dict]:
    """Extract tool calls from the raw Ollama /api/chat response.

    Newer Ollama versions include an id field on each tool call; older ones
    don't. We fall back to generating a UUID when the id is absent so the
    tool result messages always have a matching tool_call_id.

    Returns list of:
        {"id": str, "name": str, "args": dict}
    """
    raw_calls = raw.get("message", {}).get("tool_calls", [])
    parsed = []
    for tc in raw_calls:
        fn = tc.get("function", {})
        args = fn.get("arguments", {})
        # Ollama may return arguments as a JSON string or already a dict
        if isinstance(args, str):
            args = json.loads(args)
        parsed.append({
            "id":   tc.get("id") or str(uuid.uuid4()),  # Ollama >= 0.23 includes id
            "name": fn["name"],
            "args": args,
        })
    return parsed


# ---------------------------------------------------------------------------
# Execute tool calls and build tool result messages
# ---------------------------------------------------------------------------
def execute_tools(tool_calls: list[dict]) -> list[dict]:
    """Run each tool call and return a list of tool result messages.

    Tool result message shape (what goes back into messages[]):
        {
          "role": "tool",
          "tool_call_id": "<matches the id we assigned above>",
          "content": "<JSON string returned by the tool>"
        }
    """
    results = []
    for tc in tool_calls:
        fn = TOOL_REGISTRY.get(tc["name"])
        if fn is None:
            content = json.dumps({"error": f"unknown tool '{tc['name']}'"})
        else:
            print(f"\n>>> TOOL CALL: {tc['name']}({json.dumps(tc['args'])})")
            content = fn(**tc["args"])
            print(f"    RESULT:    {content}")

        results.append({
            "role":         "tool",
            "tool_call_id": tc["id"],
            "content":      content,
        })
    return results


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------
def main() -> None:
    # Pass --raw to see the unparsed model output via /api/generate instead
    # of the structured tool_calls returned by /api/chat.
    # Usage:  python ollama_direct.py --raw
    raw_mode = "--raw" in sys.argv

    messages: list[dict] = [
        {
            "role": "system",
            "content": (
                "You are a flight booking assistant. "
                "IMPORTANT: call search_flights FIRST and wait for the results. "
                "Only after you have the search results should you call book_flight "
                "with the actual flight_id from those results. "
                "Always pick the cheapest option unless the user says otherwise."
            ),
        },
        {
            "role": "user",
            "content": (
                "Find me a flight from JFK to LAX on 2026-06-01 and book the "
                "cheapest one for passenger Ada Lovelace."
            ),
        },
    ]

    if raw_mode:
        # --raw: render the chat template manually and hit /api/generate.
        # Shows the exact text the model emits before Ollama parses tool calls.
        # Does NOT run the agentic loop — it's a single shot to see raw output.
        print(f"\n{'#' * 70}")
        print("RAW MODE — single /api/generate call, no tool parsing")
        print(f"{'#' * 70}")
        ollama_generate_raw(messages, TOOLS)
        return

    turn = 1
    while True:
        print(f"\n{'#' * 70}")
        print(f"TURN {turn}")
        print(f"{'#' * 70}")

        raw = ollama_chat(messages, tools=TOOLS)
        assistant_msg = raw["message"]
        tool_calls = parse_tool_calls(raw)

        # Build the assistant message to append to history.
        # When tool_calls are present, content is empty and we add a tool_calls
        # field in OpenAI format (with the IDs we generated) so the model can
        # match tool results to its original calls in the next turn.
        history_msg: dict = {
            "role":    "assistant",
            "content": assistant_msg.get("content") or "",
        }
        if tool_calls:
            history_msg["tool_calls"] = [
                {
                    "type": "function",
                    "id":   tc["id"],
                    "function": {
                        "name":      tc["name"],
                        "arguments": json.dumps(tc["args"]),
                    },
                }
                for tc in tool_calls
            ]
        messages.append(history_msg)

        if not tool_calls:
            break

        # Execute tools and add results to the conversation
        tool_results = execute_tools(tool_calls)
        messages.extend(tool_results)
        turn += 1

    print(f"\n{'#' * 70}")
    print("FINAL REPLY")
    print(f"{'#' * 70}")
    print(assistant_msg.get("content", ""))


if __name__ == "__main__":
    main()
