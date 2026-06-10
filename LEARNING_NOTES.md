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
uv pip install langchain langchain-openai langchain-anthropic langchain-ollama tiktoken transformers requests python-dotenv

# copy env template and fill in HF_TOKEN (and optionally API keys)
cp .env.example .env

# sanity check — should print a path under libs/langgraph/langgraph/__init__.py
python -c "import langgraph; print(langgraph.__file__)"

# run the two-node graph
python debug_tick.py
# expected: {'n': 20}

# run the wiretap agent (needs Ollama running, or set OPENAI/ANTHROPIC key)
ollama pull llama3.1:8b
ollama serve
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
  - **Real LLaMA 3.1 token IDs** via `transformers.AutoTokenizer`
  - **Token → embedding vector** via Ollama `/api/embed`
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
     (`<|start_header_id|>user<|end_header_id|>` for LLaMA 3.1;
     `<|im_start|>user` / `<|im_end|>` for OpenAI ChatML).
   - Tool definitions are inlined as JSON text in the prompt. The model
     generates a tool call by emitting tokens that match a learned
     tool-call format. The SDK parses those tokens back into the
     structured `tool_calls` field on `AIMessage`.

---

## Token → embedding deep-dive (parked, resume here)

`middleware_deep_dive.py` already has `_show_token_embeddings()` wired up.

**HuggingFace access — current status:**
- Meta gated access (`meta-llama/Meta-Llama-3.1-8B`) — **approved**.
- Add `HF_TOKEN=hf_...` to `.env` and the official model will be used.
- The script automatically falls back to `NousResearch/Meta-Llama-3.1-8B`
  (public mirror, identical tokenizer) if no token is present.

**Note:** Ollama does NOT have a `/api/tokenize` endpoint at all (that was a
mistake in earlier notes). Tokenization is handled by `transformers.AutoTokenizer`.

Once the tokenizer loads the script will automatically:
1. Return exact LLaMA 3.1 BPE token IDs (128 256-token vocab).
2. Decode each token ID back to its subword string.
3. Call Ollama `/api/embed` per token → 4096-dim contextualised vector.

**Going further — raw embedding matrix lookup (no Ollama needed):**

The contextualised `/api/embed` vectors pass through the full model.
To get the *raw* embedding-matrix row `E[token_id]` (just the lookup table,
no transformer layers applied):

```python
from transformers import AutoModel
import torch

# Requires: pip install transformers torch
# Requires HF account + accepted Meta license at
#   https://huggingface.co/meta-llama/Meta-Llama-3.1-8B
# Downloads ~16 GB of weights on first run.
m = AutoModel.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B",
    torch_dtype=torch.float16,   # halves RAM to ~8 GB
    device_map="cpu",
)
E = m.model.embed_tokens.weight   # shape [128256, 4096]

# Example: look up token 128000 (<|begin_of_text|>)
vec = E[128000].detach().float().numpy()
print(vec.shape)   # (4096,)
print(vec[:8])     # first 8 dims
```

Key facts about LLaMA 3.1 8B embeddings:
- Vocab size: 128 256 tokens (tiktoken BPE with Meta's custom merges).
- Hidden dim: 4096 (that's the length of every token vector).
- Special tokens start at ID 128 000: `<|begin_of_text|>` = 128000,
  `<|start_header_id|>` = 128006, `<|eot_id|>` = 128009.
- The embedding matrix is tied with the LM head (output projection) —
  the same 128 256 × 4096 matrix is used both to look up input vectors
  and to score output logits. This is why the model can predict tokens
  in the same space it reads them.

---

## Next steps

- Step through `tick()` for the flight-booking agent (run
  `middleware_deep_dive.py` under debugger) — see the same loop with
  real tool execution, multiple super-steps, and `ToolNode` channel
  writes.
- Confirmed official `meta-llama/Meta-Llama-3.1-8B` tokenizer loads correctly
  with approved HF token.
- Add a `wrap_tool_call` hook to `WireTapMiddleware` so tool inputs /
  outputs also get dumped per call (currently only model edges are
  tapped).
- Try the notebook patterns inline: `interrupt()` on `book_flight` for
  human approval, `dynamic_prompt` middleware to swap persona based on
  user role.
- Read `pregel/_algo.py:prepare_next_tasks` carefully — that is where
  the super-step → task-set mapping happens.

---

## Parked ideas (unexplored, revisit later)

### Debug Ollama internals with Delve (Go debugger)

Ollama is a Go process (`ollama serve`) — a completely separate layer from
Python. To put breakpoints inside it:

1. Install Go + VS Code Go extension
2. Clone `github.com/ollama/ollama`
3. Build debug binary: `go build -gcflags="all=-N -l" .`
4. Install Delve: `go install github.com/go-delve/delve/cmd/dlv@latest`
5. Launch Ollama through Delve (VS Code Go launch config, type `"go"`)
6. Key files to break in:
   - `server/routes.go` — `/api/chat` handler, where raw token text gets
     converted into the structured `tool_calls` JSON field
   - `llm/server.go` — manages the llama.cpp subprocess underneath

Below Ollama sits **llama.cpp (C++)** — the actual inference engine. Debugging
that layer requires gdb/lldb and compiling llama.cpp from source with debug
symbols. Much deeper rabbit hole.

Why this is interesting: `ollama_direct.py --raw` shows what comes *out* of
the raw token stream (`<|python_tag|>{...}`). Setting a breakpoint in
`server/routes.go` would show exactly how Ollama detects that pattern and
converts it into the `tool_calls` dict the Python layer receives.

---

## Commit history (this work)

- `b5fc294c` — chore: add learning scripts for LangGraph internals
  (`debug_tick.py`, `middleware_deep_dive.py`)

(Run `git log --oneline -- debug_tick.py middleware_deep_dive.py
LEARNING_NOTES.md` to find newer entries.)
