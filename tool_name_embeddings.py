"""
tool_name_embeddings.py — do near-synonym tool names cluster in embedding space?

Experiment:
  Tools:  search_tables(query, max_budget)   vs   find_desks(query, max_budget)
  Prompt: "I want to buy a table or desk with the best comfort — look for both
           online and recommend the most comfortable one."

Hypothesis: "search" ~ "find" and "tables" ~ "desks" as tokens, so
search_tables and find_desks should sit near each other in embedding space.
If true, the model can't reliably distinguish which tool to call — a real
reliability risk when designing multi-tool agents.

What this script shows:
  1. Full tokenized INPUT  — rendered chat template + every token ID
  2. Raw model OUTPUT text — /api/generate (no tool parsing), as the model wrote it
  3. Tokenized OUTPUT      — token IDs of what the model generated
  4a. Raw embeddings       — E[token_id] from the LLaMA 3.1 weight matrix
                             (static lookup, pre-transformer, the input to layer 0)
                             Requires downloading one model shard ~4.8 GB on first run.
                             Skip with --no-weights.
  4b. Contextual embeddings — Ollama /api/embed (all 32 transformer layers applied)
  5. Cosine similarities   — how close are the pairs in each embedding space?

Run:
    python tool_name_embeddings.py             # all sections (downloads weights on first run)
    python tool_name_embeddings.py --no-weights  # skip raw embedding matrix
"""

from __future__ import annotations

import json
import os
import sys
import textwrap

import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

OLLAMA_BASE  = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL        = "llama3.1:8b"
SKIP_WEIGHTS = "--no-weights" in sys.argv


# ---------------------------------------------------------------------------
# Tools — similar names on purpose
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_tables",
            "description": "Search online for tables matching a query and budget.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query":      {"type": "string"},
                    "max_budget": {"type": "integer"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_desks",
            "description": "Find desks online matching a query and budget.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query":      {"type": "string"},
                    "max_budget": {"type": "integer"},
                },
                "required": ["query"],
            },
        },
    },
]

MESSAGES = [
    {
        "role": "system",
        "content": (
            "You are a furniture shopping assistant. "
            "Use search_tables to find tables and find_desks to find desks. "
            "When the user wants both, call both tools."
        ),
    },
    {
        "role": "user",
        "content": (
            "I want to buy a table or desk that has the best comfort. "
            "Look for both online and recommend me the most comfortable one."
        ),
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def banner(title: str, ch: str = "=") -> None:
    print(f"\n{ch * 72}\n{title}\n{ch * 72}")


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def vec_preview(v: np.ndarray, n: int = 6) -> list[float]:
    return [round(float(x), 4) for x in v[:n]]


# ---------------------------------------------------------------------------
# Tokenizer (tokenizer files only, ~few MB)
# ---------------------------------------------------------------------------
def load_tokenizer():
    from transformers import AutoTokenizer
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        try:
            return AutoTokenizer.from_pretrained(
                "meta-llama/Meta-Llama-3.1-8B",
                token=hf_token,
                clean_up_tokenization_spaces=False,
            )
        except Exception:
            pass
    return AutoTokenizer.from_pretrained(
        "NousResearch/Meta-Llama-3.1-8B",
        clean_up_tokenization_spaces=False,
    )


# ---------------------------------------------------------------------------
# Raw embedding matrix — one shard download (~4.8 GB first run, then cached)
# ---------------------------------------------------------------------------
def load_embed_matrix() -> np.ndarray | None:
    """Return E in R^{128256 x 4096}: the static token embedding lookup table.

    Uses safetensors lazy-loading so only the embed_tokens tensor is read into
    memory (~2 GB float32) even though the shard file is ~4.8 GB on disk.
    The shard is cached in ~/.cache/huggingface after the first download.
    """
    if SKIP_WEIGHTS:
        print("  (--no-weights passed: skipping raw embedding matrix download)")
        return None
    try:
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open

        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        model_id  = "meta-llama/Meta-Llama-3.1-8B" if hf_token else "NousResearch/Meta-Llama-3.1-8B"

        print(f"  Model: {model_id}")
        print("  Fetching shard index (tiny file)...")
        index_path = hf_hub_download(model_id, "model.safetensors.index.json", token=hf_token)
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]

        shard_name = weight_map["model.embed_tokens.weight"]
        print(f"  embed_tokens is in: {shard_name}")
        print("  Downloading shard (~4.8 GB on first run, cached afterwards)...")
        shard_path = hf_hub_download(model_id, shard_name, token=hf_token)

        print("  Extracting embed_tokens.weight via safetensors lazy-load...")
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            E = f.get_tensor("model.embed_tokens.weight")   # [128256, 4096]

        arr = E.float().numpy()
        print(f"  Loaded: shape={arr.shape}  dtype=float32  mem~{arr.nbytes // 1_000_000} MB")
        return arr

    except Exception as exc:
        print(f"  Could not load embedding matrix: {exc}")
        print("  Tip: pip install safetensors huggingface_hub")
        return None


# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------
def ollama_embed(text: str) -> np.ndarray:
    resp = requests.post(
        f"{OLLAMA_BASE}/api/embed",
        json={"model": MODEL, "input": text},
        timeout=60,
    )
    resp.raise_for_status()
    return np.array(resp.json()["embeddings"][0], dtype=np.float32)


def render_llama_prompt(messages: list[dict], tools: list[dict]) -> str:
    """Render LLaMA 3.1 chat template: tools injected as JSON in a system block."""
    parts = ["<|begin_of_text|>"]
    if tools:
        parts.append(
            f"<|start_header_id|>system<|end_header_id|>\n\n"
            f"[TOOLS]{json.dumps(tools)}[/TOOLS]<|eot_id|>"
        )
    for m in messages:
        parts.append(
            f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n"
            f"{m.get('content', '')}<|eot_id|>"
        )
    parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Loading tokenizer (tokenizer files only, ~few MB)...")
    tok = load_tokenizer()

    banner("4a. RAW EMBEDDING MATRIX SETUP")
    E = load_embed_matrix()   # None if --no-weights or download failed

    # ── Section 1: Tokenized INPUT ─────────────────────────────────────────
    banner("1. TOKENIZED INPUT (rendered LLaMA 3.1 chat template + token IDs)")

    prompt    = render_llama_prompt(MESSAGES, TOOLS)
    input_ids = tok.encode(prompt, add_special_tokens=False)

    print(f"\nTotal input tokens: {len(input_ids)}")
    print("\nRendered prompt (first 1200 chars):")
    print(textwrap.indent(prompt[:1200] + ("..." if len(prompt) > 1200 else ""), "  "))

    print(f"\nFirst 60 token IDs: {input_ids[:60]}")
    print("\nFirst 60 tokens decoded individually:")
    for i, tid in enumerate(input_ids[:60]):
        print(f"  [{i:>2}]  id={tid:>7}  {tok.decode([tid])!r}")

    # ── Section 2: Raw model output ────────────────────────────────────────
    banner("2. RAW MODEL OUTPUT  (/api/generate, stream=false, raw=true)")
    print("\nPosting to Ollama... (may take ~30s)")

    gen_resp = requests.post(
        f"{OLLAMA_BASE}/api/generate",
        json={"model": MODEL, "prompt": prompt, "raw": True,
              "stream": False, "temperature": 0},
        timeout=180,
    )
    gen_resp.raise_for_status()
    gen_result = gen_resp.json()
    raw_output = gen_result.get("response", "")

    print(f"\nrepr:  {repr(raw_output)}")
    print(f"\nplain:\n{raw_output}")
    print(f"\neval_count (output tokens reported by Ollama): {gen_result.get('eval_count')}")

    # ── Section 3: Tokenized OUTPUT ────────────────────────────────────────
    banner("3. TOKENIZED OUTPUT (what token IDs did the model generate?)")

    output_ids = tok.encode(raw_output, add_special_tokens=False)
    print(f"\nTotal output tokens (tokenizer count): {len(output_ids)}")
    print(f"All output token IDs: {output_ids}")
    print("\nAll output tokens decoded individually:")
    for i, tid in enumerate(output_ids):
        print(f"  [{i:>2}]  id={tid:>7}  {tok.decode([tid])!r}")

    # ── Section 4: Embedding comparison ────────────────────────────────────
    tool_names = ["search_tables", "find_desks"]
    keywords   = ["search", "find", "tables", "desks"]
    all_terms  = tool_names + keywords

    banner("TOKENIZATION OF EACH TERM")
    term_ids: dict[str, list[int]] = {}
    for term in all_terms:
        ids    = tok.encode(term, add_special_tokens=False)
        pieces = [tok.decode([i]) for i in ids]
        term_ids[term] = ids
        print(f"  {term!r:<20}  ids={ids}  pieces={pieces}")

    # ── 4a: Raw embeddings ─────────────────────────────────────────────────
    raw_vecs: dict[str, np.ndarray] = {}

    if E is not None:
        banner("4a. RAW EMBEDDINGS  (E[token_id]  —  static matrix row, pre-transformer)", ch="-")
        print(
            "These are the vectors handed to transformer layer 0.\n"
            "The same token always gets the same vector regardless of context.\n"
            "Multi-token terms: vectors are averaged across their constituent tokens.\n"
        )
        for term in all_terms:
            ids      = term_ids[term]
            tok_vecs = [E[tid] for tid in ids]
            avg_vec  = np.mean(tok_vecs, axis=0)
            raw_vecs[term] = avg_vec

            print(f"  {term!r}")
            for tid, vec in zip(ids, tok_vecs):
                piece = tok.decode([tid])
                print(f"    id={tid:>7}  {piece!r:<14}"
                      f"  norm={np.linalg.norm(vec):.4f}"
                      f"  vec[:6]={vec_preview(vec)}")
            if len(ids) > 1:
                print(f"    averaged ({len(ids)} tokens):"
                      f"  norm={np.linalg.norm(avg_vec):.4f}"
                      f"  vec[:6]={vec_preview(avg_vec)}")
            print()

        print("--- Cosine similarities (raw / static embeddings) ---")
        pairs = [
            ("search_tables", "find_desks",  "tool names  [hypothesis]"),
            ("search",        "find",         "synonym verbs"),
            ("tables",        "desks",        "synonym nouns"),
            ("search",        "tables",       "unrelated baseline"),
        ]
        for a, b, label in pairs:
            sim = cosine_sim(raw_vecs[a], raw_vecs[b])
            print(f"  cos({a!r:<15}, {b!r:<10}) = {sim:+.4f}   # {label}")
    else:
        banner("4a. RAW EMBEDDINGS — skipped (run without --no-weights to enable)", ch="-")

    # ── 4b: Contextual embeddings ──────────────────────────────────────────
    banner("4b. CONTEXTUAL EMBEDDINGS  (Ollama /api/embed  —  all 32 layers applied)", ch="-")
    print(
        "Each string is passed through the full model. The output is pooled\n"
        "across all token positions — context-sensitive, not static.\n"
        "The same word embedded alone vs inside a sentence gets a different vector.\n"
    )
    ctx_vecs: dict[str, np.ndarray] = {}
    for term in all_terms:
        vec = ollama_embed(term)
        ctx_vecs[term] = vec
        print(f"  {term!r:<20}  dim={len(vec)}"
              f"  norm={np.linalg.norm(vec):.4f}"
              f"  vec[:6]={vec_preview(vec)}")

    print("\n--- Cosine similarities (contextual embeddings) ---")
    pairs = [
        ("search_tables", "find_desks",  "tool names  [hypothesis]"),
        ("search",        "find",         "synonym verbs"),
        ("tables",        "desks",        "synonym nouns"),
        ("search",        "tables",       "unrelated baseline"),
    ]
    for a, b, label in pairs:
        sim = cosine_sim(ctx_vecs[a], ctx_vecs[b])
        print(f"  cos({a!r:<15}, {b!r:<10}) = {sim:+.4f}   # {label}")

    # ── Section 5: Summary ─────────────────────────────────────────────────
    banner("5. SUMMARY")

    st_fd_ctx = cosine_sim(ctx_vecs["search_tables"], ctx_vecs["find_desks"])
    s_f_ctx   = cosine_sim(ctx_vecs["search"],        ctx_vecs["find"])
    t_d_ctx   = cosine_sim(ctx_vecs["tables"],        ctx_vecs["desks"])

    print(f"""
Contextual cosine similarities:
  search_tables vs find_desks : {st_fd_ctx:+.4f}
  search        vs find       : {s_f_ctx:+.4f}
  tables        vs desks      : {t_d_ctx:+.4f}""")

    if raw_vecs:
        st_fd_raw = cosine_sim(raw_vecs["search_tables"], raw_vecs["find_desks"])
        s_f_raw   = cosine_sim(raw_vecs["search"],        raw_vecs["find"])
        t_d_raw   = cosine_sim(raw_vecs["tables"],        raw_vecs["desks"])
        print(f"""
Raw (static) cosine similarities:
  search_tables vs find_desks : {st_fd_raw:+.4f}
  search        vs find       : {s_f_raw:+.4f}
  tables        vs desks      : {t_d_raw:+.4f}""")

    print("""
WHAT THIS MEANS FOR AGENT DESIGN:

If cos(search_tables, find_desks) is high (~0.9+):
  The model's internal representation can't cleanly separate these tools.
  Routing depends on tiny prompt details, not a robust semantic distinction.
  In production, prefer unambiguous tool names: browse_furniture / reserve_item.

The gap between raw and contextual similarity is informative:
  Raw  = pure vocabulary overlap at token level (before any reasoning).
  Ctx  = what the model 'thinks' each term means after reading it in isolation.
  If contextual >> raw: the model is actively resolving the semantic link.
  If they're similar: the conflation comes from the token embeddings themselves.
""")


if __name__ == "__main__":
    main()
