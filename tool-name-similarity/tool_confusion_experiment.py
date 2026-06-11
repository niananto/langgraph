"""
tool_confusion_experiment.py — at what embedding distance do tools stop confusing an LLM?

4 tools whose names are all dictionary synonyms for "locate something"
(search / find / look up / track) but whose intents, contexts, and argument
types are completely distinct:

  T1  search_products(keywords)       — catalog discovery
  T2  find_stores(city)               — physical store locator
  T3  lookup_warranty(serial_number)  — warranty coverage status
  T4  track_order(order_id)           — shipping status

The verbs overlap at the dictionary level; nothing else does.  This isolates
the question: does name-level synonymy alone confuse the router, or does it
take intent overlap (as in the earlier search_products/find_items design,
which shared a `query` argument and 38% cross-confusion)?

12 prompts (3 per tool, ground-truth labelled but tool name never mentioned;
each prompt carries the identifier its tool needs, so correct routing can
never fail on missing arguments).  4 runs per prompt = 48 total runs.

5 similarity metrics computed once upfront for all 6 tool pairs:
  1. TF-IDF cosine           (sklearn)
  2. BERTScore F1            (bert-score, ~500 MB download first run)
  3. Word Mover's Distance   (gensim + GloVe-100, ~128 MB download first run)
  4. Sentence-BERT cosine    (sentence-transformers, ~80 MB download first run)
  5. Ollama embedding cosine (/api/embed — same model that routes the calls)

Headline output: for each metric, the similarity threshold below which
confusion dropped to zero — the "safe separation" for tool naming.

Run:
    python tool_confusion_experiment.py
    python tool_confusion_experiment.py --no-bert --no-wmd        # skip large downloads
    python tool_confusion_experiment.py --model llama3.2:3b       # different chat model
    python tool_confusion_experiment.py --embed-model nomic-embed-text  # dedicated embed model
    python tool_confusion_experiment.py --runs 2                  # faster smoke-test

If Ollama returns 501 for /api/embed, pull a dedicated embedding model:
    ollama pull nomic-embed-text
    python tool_confusion_experiment.py --embed-model nomic-embed-text
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from itertools import combinations
from typing import Annotated

import numpy as np
import requests
from langchain_core.messages import AIMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# CLI flags
# ---------------------------------------------------------------------------
_argv = sys.argv[1:]
SKIP_BERT = "--no-bert" in _argv
SKIP_WMD  = "--no-wmd"  in _argv
MODEL       = next((_argv[i + 1] for i, a in enumerate(_argv) if a == "--model"),       "llama3.1:8b")
EMBED_MODEL = next((_argv[i + 1] for i, a in enumerate(_argv) if a == "--embed-model"), MODEL)
RUNS        = int(next((_argv[i + 1] for i, a in enumerate(_argv) if a == "--runs"),    4))
OLLAMA_BASE = os.getenv("OLLAMA_HOST", "http://localhost:11434")

SEP  = "=" * 72
SEP2 = "-" * 72


# ---------------------------------------------------------------------------
# T1 — search_products  (catalog discovery by keyword)
# ---------------------------------------------------------------------------
def search_products(keywords: str) -> str:
    """Search the product catalog by keyword to discover products to buy."""
    return json.dumps([
        {"id": "P1", "name": f"Match for '{keywords}' #1", "price_usd": 49},
        {"id": "P2", "name": f"Match for '{keywords}' #2", "price_usd": 89},
    ])


# ---------------------------------------------------------------------------
# T2 — find_stores  ("find" ≈ "search" in the dictionary, but the intent is
#                     a physical store locator — completely different job)
# ---------------------------------------------------------------------------
def find_stores(city: str) -> str:
    """Find physical retail store locations in a given city."""
    return json.dumps([
        {"store_id": "S1", "address": f"12 Main St, {city}", "open_until": "21:00"},
        {"store_id": "S2", "address": f"450 Market Ave, {city}", "open_until": "20:00"},
    ])


# ---------------------------------------------------------------------------
# T3 — lookup_warranty  ("look up" ≈ "search" in the dictionary, but the
#                         intent is warranty status by serial number)
# ---------------------------------------------------------------------------
def lookup_warranty(serial_number: str) -> str:
    """Look up the warranty coverage status of a purchased device by its serial number."""
    return json.dumps({
        "serial_number": serial_number,
        "covered": True,
        "expires": "2027-03-15",
        "plan": "standard 2-year",
    })


# ---------------------------------------------------------------------------
# T4 — track_order  ("track" ≈ "follow/locate" in the dictionary, but the
#                     intent is shipping status by order ID)
# ---------------------------------------------------------------------------
def track_order(order_id: str) -> str:
    """Track the shipping and delivery status of a customer order by its order ID."""
    return json.dumps({
        "order_id": order_id,
        "status": "in_transit",
        "estimated_delivery": "2026-06-14",
        "carrier": "UPS",
    })


TOOLS = [search_products, find_stores, lookup_warranty, track_order]
TOOL_NAMES = [t.__name__ for t in TOOLS]


# ---------------------------------------------------------------------------
# 12 prompts — ground truth labelled, tool name never mentioned.
# Every prompt includes the identifier its tool requires (serial, order id,
# city) so a correct routing decision can never fail on missing arguments.
# ---------------------------------------------------------------------------
PROMPTS: list[dict] = [
    # ── T1: search_products  (shopping/discovery intent) ─────────────────
    {"tool": "search_products",  "text": "I want to buy a wireless mouse — what do you have?"},
    {"tool": "search_products",  "text": "Show me some noise-cancelling headphones I could order."},
    {"tool": "search_products",  "text": "I'm shopping for a budget mechanical keyboard."},
    # ── T2: find_stores  (physical location intent) ──────────────────────
    {"tool": "find_stores",      "text": "Is there a branch of yours in Chicago?"},
    {"tool": "find_stores",      "text": "Where can I visit you in person around Seattle?"},
    {"tool": "find_stores",      "text": "I'd like to walk into one of your shops in Boston — where exactly?"},
    # ── T3: lookup_warranty  (coverage status intent) ─────────────────────
    {"tool": "lookup_warranty",  "text": "My laptop's serial number is SN-99887 — is it still covered?"},
    {"tool": "lookup_warranty",  "text": "I bought a blender last year, serial BL-1234. Am I covered if it breaks?"},
    {"tool": "lookup_warranty",  "text": "Can you check coverage for my device with serial X5-0042?"},
    # ── T4: track_order  (shipping status intent) ─────────────────────────
    {"tool": "track_order",      "text": "Order ORD-5522 — where is it right now?"},
    {"tool": "track_order",      "text": "My package from order ORD-1001 hasn't arrived. What's the status?"},
    {"tool": "track_order",      "text": "When will order ORD-7733 be delivered?"},
]

SYSTEM_PROMPT = (
    "You are a helpful store assistant. "
    "Use the available tools to answer customer questions."
)


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------
@dataclass
class RunResult:
    prompt_idx: int
    run_idx: int
    ground_truth: str
    tool_called: str | None   # None = model gave no tool call
    success: bool
    reason: str
    elapsed_s: float


# ---------------------------------------------------------------------------
# LangGraph setup
# ---------------------------------------------------------------------------
class State(TypedDict):
    messages: Annotated[list, add_messages]


def build_graph() -> object:
    llm = ChatOllama(model=MODEL, temperature=0.7).bind_tools(TOOLS)

    def call_model(state: State) -> State:
        return {"messages": [llm.invoke(state["messages"])]}

    return (
        StateGraph(State)
        .add_node("model", call_model)
        .add_node("tools", ToolNode(TOOLS, handle_tool_errors=True))
        .add_edge(START, "model")
        .add_conditional_edges("model", tools_condition)
        .add_edge("tools", "model")
        .compile()
    )


def run_once(graph, system_msg: dict, prompt: str, ground_truth: str,
             p_idx: int, r_idx: int) -> RunResult:
    t0 = time.time()
    try:
        result = graph.invoke({
            "messages": [system_msg, {"role": "user", "content": prompt}]
        })
    except Exception as exc:
        return RunResult(p_idx, r_idx, ground_truth, None, False,
                         f"graph error: {exc}", time.time() - t0)

    elapsed = time.time() - t0
    messages = result["messages"]

    ai_with_calls = next(
        (m for m in messages if isinstance(m, AIMessage) and m.tool_calls), None
    )
    if ai_with_calls is None:
        return RunResult(p_idx, r_idx, ground_truth, None, False,
                         "no tool called", elapsed)

    called = ai_with_calls.tool_calls[0]["name"]
    ok = called == ground_truth

    for m in messages:
        if isinstance(m, ToolMessage):
            content = m.content if isinstance(m.content, str) else str(m.content)
            if content.strip().startswith("Error"):
                return RunResult(p_idx, r_idx, ground_truth, called, False,
                                 f"tool error: {content[:60]}", elapsed)

    return RunResult(p_idx, r_idx, ground_truth, called, ok,
                     "ok" if ok else f"called {called!r}", elapsed)


# ---------------------------------------------------------------------------
# Similarity metrics
# ---------------------------------------------------------------------------
TOOL_TEXTS = {t.__name__: f"{t.__name__}: {t.__doc__}" for t in TOOLS}
TOOL_PAIRS  = list(combinations(TOOL_NAMES, 2))   # 6 pairs


def tfidf_similarities() -> dict[tuple, float]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sk_cos

    docs = [TOOL_TEXTS[n] for n in TOOL_NAMES]
    # use_idf=False: removes the inverse-document-frequency weighting.
    # With only 4 documents, IDF aggressively penalizes shared terms and
    # collapses all pair scores to near-zero. Plain TF cosine gives a
    # meaningful spread where shared vocabulary actually raises similarity.
    mat  = TfidfVectorizer(use_idf=False).fit_transform(docs).toarray()
    sims = {}
    for a, b in TOOL_PAIRS:
        va = mat[TOOL_NAMES.index(a)]
        vb = mat[TOOL_NAMES.index(b)]
        sims[(a, b)] = float(sk_cos([va], [vb])[0][0])
    return sims


def bert_similarities() -> dict[tuple, float] | None:
    if SKIP_BERT:
        return None
    try:
        from bert_score import score as bscore
        cands = [TOOL_TEXTS[a] for a, _ in TOOL_PAIRS]
        refs  = [TOOL_TEXTS[b] for _, b in TOOL_PAIRS]
        _, _, F1 = bscore(cands, refs, lang="en", verbose=False)
        return {pair: float(F1[i]) for i, pair in enumerate(TOOL_PAIRS)}
    except Exception as exc:
        print(f"  [BERTScore] skipped — {exc}")
        return None


def wmd_similarities() -> dict[tuple, float] | None:
    if SKIP_WMD:
        return None
    try:
        import gensim.downloader as gensim_api

        def tokenize(text: str) -> list[str]:
            return [w.lower() for w in re.findall(r"[a-zA-Z]+", text)]

        print("  Loading GloVe-100 vectors (first run ~128 MB)...")
        wv = gensim_api.load("glove-wiki-gigaword-100")

        raw_dists = {}
        for a, b in TOOL_PAIRS:
            ta, tb = tokenize(TOOL_TEXTS[a]), tokenize(TOOL_TEXTS[b])
            # filter to tokens in vocab
            ta = [w for w in ta if w in wv]
            tb = [w for w in tb if w in wv]
            if ta and tb:
                raw_dists[(a, b)] = wv.wmdistance(ta, tb)
            else:
                raw_dists[(a, b)] = float("inf")

        # normalise to 0–1 similarity: 1 / (1 + dist)
        return {pair: 1.0 / (1.0 + d) for pair, d in raw_dists.items()}
    except Exception as exc:
        print(f"  [WMD] skipped — {exc}")
        print("  Tip: WMD needs the POT package ->  pip install POT")
        return None


def sbert_similarities() -> dict[tuple, float] | None:
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        vecs  = {n: model.encode(TOOL_TEXTS[n]) for n in TOOL_NAMES}
        sims  = {}
        for a, b in TOOL_PAIRS:
            va, vb = vecs[a], vecs[b]
            sims[(a, b)] = float(
                np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9)
            )
        return sims
    except Exception as exc:
        print(f"  [SBERT] skipped — {exc}")
        return None


def _ollama_embed_one(text: str) -> np.ndarray:
    """Try known Ollama embed formats in order of most-likely-to-work.

    String input  ("/api/embed", input=str)  — works on 0.30.x and confirmed
    by tool_name_embeddings.py.  Array input and the legacy /api/embeddings
    endpoint are kept as fallbacks for other versions.
    """
    candidates = [
        ("/api/embed",       {"model": EMBED_MODEL, "input": text},    "embeddings"),
        ("/api/embed",       {"model": EMBED_MODEL, "input": [text]},  "embeddings"),
        ("/api/embeddings",  {"model": EMBED_MODEL, "prompt": text},   "embedding"),
    ]
    for endpoint, payload, result_key in candidates:
        try:
            resp = requests.post(
                f"{OLLAMA_BASE}{endpoint}", json=payload, timeout=300
            )
            if resp.status_code in (404, 501):
                continue
            resp.raise_for_status()
            vec = resp.json()[result_key]
            if isinstance(vec[0], list):   # /api/embed wraps in outer list
                vec = vec[0]
            return np.array(vec, dtype=np.float32)
        except (requests.HTTPError, KeyError, IndexError):
            continue
    raise RuntimeError(
        f"No working Ollama embed endpoint for model '{EMBED_MODEL}'.\n"
        "  Try:  ollama pull nomic-embed-text\n"
        "  Then: python tool_confusion_experiment.py --embed-model nomic-embed-text"
    )


def ollama_similarities() -> dict[tuple, float] | None:
    try:
        vecs = {}
        for name in TOOL_NAMES:
            vecs[name] = _ollama_embed_one(TOOL_TEXTS[name])

        sims = {}
        for a, b in TOOL_PAIRS:
            va, vb = vecs[a], vecs[b]
            sims[(a, b)] = float(
                np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9)
            )
        return sims
    except Exception as exc:
        print(f"  [Ollama] skipped — {exc}")
        return None


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------
def confusion_rate(runs: list[RunResult], tool_a: str, tool_b: str) -> float:
    """Fraction of runs where the model crossed the pair: truth was tool_a but
    it called tool_b, or vice versa.  Tool execution errors on the *correct*
    tool are routing successes, not confusion, so they're excluded.
    """
    relevant = [r for r in runs if r.ground_truth in (tool_a, tool_b)]
    if not relevant:
        return 0.0
    confused = [
        r for r in relevant
        if r.tool_called in (tool_a, tool_b) and r.tool_called != r.ground_truth
    ]
    return len(confused) / len(relevant)


def bar(v: float, width: int = 20) -> str:
    filled = round(v * width)
    return "[" + "#" * filled + "." * (width - filled) + "]"


def fmt(v: float | None) -> str:
    return f"{v:.4f}" if v is not None else "  N/A "


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(SEP)
    print(f"tool_confusion_experiment  |  model={MODEL}  runs/prompt={RUNS}")
    print(SEP)

    # ── Similarity metrics (computed once) ────────────────────────────────
    print("\n[1/3]  Computing similarity metrics...")
    metrics: dict[str, dict[tuple, float] | None] = {}

    print("  TF-IDF...", end=" ", flush=True)
    metrics["TF-IDF"] = tfidf_similarities()
    print("done")

    print("  BERTScore...", end=" ", flush=True)
    metrics["BERTScore"] = bert_similarities()
    print("done" if metrics["BERTScore"] else "skipped")

    print("  Word Mover's Distance...")
    metrics["WMD"] = wmd_similarities()

    print("  Sentence-BERT...", end=" ", flush=True)
    metrics["SBERT"] = sbert_similarities()
    print("done" if metrics["SBERT"] else "skipped")

    print("  Ollama embeddings...", end=" ", flush=True)
    metrics["Ollama"] = ollama_similarities()
    print("done" if metrics["Ollama"] else "skipped")

    active_metrics = {k: v for k, v in metrics.items() if v is not None}

    # ── Print similarity matrix ───────────────────────────────────────────
    print(f"\n{SEP}")
    print("SIMILARITY MATRIX  (all 6 tool pairs × active metrics)")
    print(SEP)
    header = f"{'Pair':<42}" + "".join(f"  {k:<10}" for k in active_metrics)
    print(header)
    print(SEP2)
    for a, b in TOOL_PAIRS:
        label = f"{a}  ↔  {b}"
        row = f"{label:<42}" + "".join(
            f"  {fmt(m.get((a, b))):<10}" for m in active_metrics.values()
        )
        print(row)

    # ── Run the 48 experiments ─────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"[2/3]  Running experiments  ({len(PROMPTS)} prompts × {RUNS} runs = "
          f"{len(PROMPTS) * RUNS} total)")
    print(SEP)

    system_msg = {"role": "system", "content": SYSTEM_PROMPT}
    graph = build_graph()
    all_runs: list[RunResult] = []

    for p_idx, prompt_entry in enumerate(PROMPTS):
        gt   = prompt_entry["tool"]
        text = prompt_entry["text"]
        short = text[:55] + ("..." if len(text) > 55 else "")
        print(f"\nP{p_idx+1:02d} [{gt}]  {short!r}")

        for r_idx in range(1, RUNS + 1):
            res = run_once(graph, system_msg, text, gt, p_idx, r_idx)
            all_runs.append(res)
            tick  = "+" if res.success else "x"
            label = res.reason if not res.success else f"called {res.tool_called!r}"
            print(f"    run {r_idx}/{RUNS}  {tick}  {label:<35}  ({res.elapsed_s:.1f}s)")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("[3/3]  RESULTS SUMMARY")
    print(SEP)

    # Per-tool accuracy
    print("\nPer-tool accuracy:")
    for tool_name in TOOL_NAMES:
        tool_runs  = [r for r in all_runs if r.ground_truth == tool_name]
        correct    = sum(1 for r in tool_runs if r.success)
        total      = len(tool_runs)
        wrong_calls = {}
        for r in tool_runs:
            # only count routing mistakes — a tool error on the right tool
            # is an execution failure, not confusion with itself
            if not r.success and r.tool_called and r.tool_called != r.ground_truth:
                wrong_calls[r.tool_called] = wrong_calls.get(r.tool_called, 0) + 1
        confusion_str = "  confused with: " + ", ".join(
            f"{k}×{v}" for k, v in sorted(wrong_calls.items(), key=lambda x: -x[1])
        ) if wrong_calls else ""
        pct = 100 * correct // total if total else 0
        print(f"  {tool_name:<22}  {correct:>2}/{total}  {bar(correct/total, 12)}  {pct:>3}%{confusion_str}")

    # Grand total
    grand_ok = sum(1 for r in all_runs if r.success)
    grand_n  = len(all_runs)
    print(f"\n  {'TOTAL':<22}  {grand_ok:>2}/{grand_n}  "
          f"{bar(grand_ok/grand_n, 12)}  {100*grand_ok//grand_n:>3}%")

    # Per-pair confusion rate
    print(f"\n{SEP2}")
    print("Confusion rate per tool pair vs similarity scores:")
    print(SEP2)
    metric_keys = list(active_metrics)
    header = f"{'Pair':<42}  {'Confusion':>9}" + "".join(
        f"  {k:<10}" for k in metric_keys
    )
    print(header)
    print(SEP2)
    for a, b in TOOL_PAIRS:
        cr    = confusion_rate(all_runs, a, b)
        label = f"{a}  ↔  {b}"
        row = (f"{label:<42}  {cr:>8.1%}" +
               "".join(f"  {fmt(active_metrics[k].get((a, b))):<10}"
                       for k in metric_keys))
        print(row)

    # Threshold analysis
    print(f"\n{SEP}")
    print("THRESHOLD ANALYSIS — at what similarity does confusion drop to zero?")
    print(SEP)
    print("(sorted by similarity score per metric; confusion% beside each pair)\n")

    for metric_name, metric_sims in active_metrics.items():
        print(f"  Metric: {metric_name}")
        sorted_pairs = sorted(TOOL_PAIRS, key=lambda p: metric_sims.get(p, 0), reverse=True)
        last_confused_sim  = None
        first_safe_sim     = None

        for a, b in sorted_pairs:
            sim = metric_sims.get((a, b), 0.0)
            cr  = confusion_rate(all_runs, a, b)
            marker = " <-- confused" if cr > 0 else ""
            print(f"    {sim:.4f}  {a} ↔ {b:<26}  confusion={cr:.0%}{marker}")
            if cr > 0:
                last_confused_sim = sim
            elif first_safe_sim is None:
                first_safe_sim = sim

        if last_confused_sim is not None:
            print(f"    => confusion present up to sim={last_confused_sim:.4f}")
            if first_safe_sim is not None and first_safe_sim < last_confused_sim:
                print(f"    => safe below sim={first_safe_sim:.4f}  "
                      f"(ambiguity zone: {first_safe_sim:.4f}–{last_confused_sim:.4f})")
        else:
            print("    => no confusion observed at any similarity level")
        print()

    # Raw run log
    print(SEP2)
    print("RAW RUN LOG")
    print(SEP2)
    print(f"{'#':<4}  {'prompt':<4}  {'run':<3}  {'ground_truth':<22}  "
          f"{'called':<22}  {'ok':<4}  reason")
    print(SEP2)
    for i, r in enumerate(all_runs, 1):
        called_str = r.tool_called or "(none)"
        tick = "+" if r.success else "x"
        print(f"{i:<4}  P{r.prompt_idx+1:<3}  R{r.run_idx:<2}  "
              f"{r.ground_truth:<22}  {called_str:<22}  {tick:<4}  {r.reason}")

    print(f"\n{SEP}")
    print("Experiment complete.")
    print(SEP)


if __name__ == "__main__":
    main()
