# Learning Notes — LangGraph Internals Walkthrough

Personal scratchpad. Tracks what got added to this fork, why, and how to
resume from another machine.

Repo: `niananto/langgraph` (fork of `langchain-ai/langgraph`), branch `main`.

---

## Goal

Understand LangGraph internals by watching one super-step happen end-to-end
under a debugger, and see every byte that crosses the orchestrator ↔ LLM
boundary in an agent with two tools.

Static reading hits a ceiling fast. Stepping through `tick()` with live
state in the watch panel is worth more than an hour of static reading.

---

## Resume-on-new-machine checklist

```bash
git clone https://github.com/niananto/langgraph.git
cd langgraph

# editable installs — package internals do `from langgraph.x import ...`,
# so the package MUST be importable by its canonical top-level name.
# Sitting next to source is not enough.
uv venv && source .venv/bin/activate          # or: python -m venv .venv
uv pip install -e libs/checkpoint -e libs/prebuilt -e libs/langgraph

# extras for the middleware deep-dive script
uv pip install langchain langchain-openai langchain-anthropic tiktoken

# sanity check — should print a path under libs/langgraph/langgraph/__init__.py
python -c "import langgraph; print(langgraph.__file__)"

# run the two-node graph
python debug_tick.py
# expected: {'n': 20}

# run the wiretap agent (need API key)
export OPENAI_API_KEY=sk-...                  # or ANTHROPIC_API_KEY=...
python middleware_deep_dive.py
```

---

## Files added in this work

### 1. `debug_tick.py` — minimal two-node `StateGraph`

Purpose: drive `pregel/_loop.py:tick()` under a debugger so one super-step
is observable end-to-end with live state.

Graph: `START → a (n+=1) → b (n*=10) → END`. Input `{"n": 1}` →
`{"n": 20}`. Three super-steps: START tick + node `a` tick + node `b` tick.

### 2. `middleware_deep_dive.py` — wiretap on the agent loop

Agent with two tools:

- `search_flights(origin, destination, date)`
- `book_flight(flight_id, passenger_name)`

`WireTapMiddleware` subclasses `AgentMiddleware` and prints at every edge:

- `before_model` — `state["messages"]` Python view
- `wrap_model_call` (request side):
  - `ModelRequest` fields (system_prompt, tools, messages)
  - Tool **JSON-Schemas** via `convert_to_openai_tool`
  - **Provider HTTP payload** (post-langchain conversion, pre-tokenization)
  - **Tokenizer view** via `tiktoken` — token IDs + per-token decoded
    strings — shows that the LLM consumes one token stream, not a list
    of messages; message boundaries are special tokens
    (`<|im_start|>user`, `<|im_end|>`); tool schemas are inlined JSON
    text inside the prompt
- `wrap_model_call` (response side) — raw `AIMessage`, `.tool_calls`,
  `.response_metadata`, `.usage_metadata`
- `after_model` — newest message orchestrator appended
- Final state — full message log including `ToolMessage`s

User prompt forces a multi-step chain: search → pick cheapest → book →
final natural-language reply. Yields 2 model turns + 2 tool turns in
the trace.

---

## VS Code setup (not committed — `.vscode/` is gitignored)

Recreate `.vscode/launch.json` on each clone:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "debug tick",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/debug_tick.py",
      "console": "integratedTerminal",
      "justMyCode": false
    }
  ]
}
```

`justMyCode: false` mandatory — otherwise breakpoints inside the
installed `langgraph` package (even editable) get skipped.

### Breakpoints to set

- `libs/langgraph/langgraph/pregel/_loop.py` line **538** (first line
  inside `def tick(self) -> bool:`) — entry per super-step
- `libs/langgraph/langgraph/pregel/_algo.py` — `prepare_next_tasks`
  (task selection)
- `libs/langgraph/langgraph/pregel/_loop.py` — `_apply_writes`
  (channel updates)

### Watch panel suggestions

- `self.step` — super-step counter
- `self.tasks` — dict task_id → `PregelExecutableTask`
- `self.channels` — channel state pre-update
- `self.checkpoint["channel_versions"]` — version bump per write
- `self.input` / `self.output`

Conditional breakpoint to skip first tick: right-click red dot → Edit
Breakpoint → `self.step == 1`.

### Faster signal without a debugger

```python
for ev in app.stream({"n": 1}, stream_mode="debug"):
    print(ev)
```

Pair with breakpoint to map debug events ↔ `tick()` internals.

---

## Lessons learned (gotchas to remember)

1. **Package internals dictate import path.**
   `langgraph/graph/__init__.py` does `from langgraph.constants import ...`.
   The Python import system must therefore resolve `langgraph` as a
   top-level package. An import like `from libs.langgraph.langgraph.graph
   import ...` blows up because the re-entry inside the package uses the
   canonical name, which isn't installed.

2. **Monorepo ⇒ editable installs required.**
   `pip install -e libs/<name>` registers each library so imports resolve.
   `langgraph` depends on `checkpoint` and `prebuilt`, so install all
   three or the import chain breaks downstream.

3. **Relative paths bite.**
   `../checkpoint` resolves vs cwd, not vs project root. From repo root
   use `libs/checkpoint`.

4. **`uv pip` ≠ `pip`** in a uv-managed venv. `command not found: pip`
   when uv didn't shim a `pip` binary.

5. **Debugger `justMyCode: false`** is mandatory for breakpoints in
   installed (even editable) library code.

6. **Strings vs tokens at the LLM layer.**
   - Model never sees a list of `BaseMessage` objects.
   - SDK renders the message list into one big string via a
     provider-specific chat template, then tokenizes into a single
     sequence of integer IDs.
   - Message boundaries are *special tokens* inside that sequence
     (`<|im_start|>user`, `<|im_end|>` for OpenAI Harmony/ChatML;
     `<|start_header_id|>` for Llama; etc.). Model learned the pattern
     during training — that's how it knows whose turn it is.
   - Tool definitions are inlined as JSON text in the prompt. The model
     generates a tool call by emitting tokens that match a learned
     tool-call format. The SDK parses those tokens back into the
     structured `tool_calls` field on `AIMessage`.

---

## Next steps

- Step through `tick()` for the flight-booking agent (run
  `middleware_deep_dive.py` under debugger) — see the same loop with
  real tool execution, multiple super-steps, and `ToolNode` channel
  writes.
- Add a `wrap_tool_call` hook to `WireTapMiddleware` so tool inputs /
  outputs also get dumped per call (currently only model edges are
  tapped).
- Try the notebook patterns inline: `interrupt()` on `book_flight` for
  human approval, `dynamic_prompt` middleware to swap persona based on
  user role.
- Read `pregel/_algo.py:prepare_next_tasks` carefully — that is where
  the super-step → task-set mapping happens.

---

## Commit history (this work)

- `b5fc294c` — chore: add learning scripts for LangGraph internals
  (`debug_tick.py`, `middleware_deep_dive.py`)

(Run `git log --oneline -- debug_tick.py middleware_deep_dive.py
LEARNING_NOTES.md` to find newer entries.)
