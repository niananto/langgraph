"""
tool_confusion_experiment.py — at what embedding distance do tools stop confusing an LLM?

4 tools in an e-commerce store assistant domain, designed on a deliberate
similarity ladder:

  T1  search_products  ↔  T2  find_items       ~very similar  (synonyms)
  T1  search_products  ↔  T3  browse_category   ~moderate      (same domain)
  T1  search_products  ↔  T4  track_order       ~low           (different purpose)

12 prompts (3 per tool, ground-truth labelled but tool name never mentioned).
4 runs per prompt = 48 total runs.

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
    python tool_confusion_experiment.py --no-bert --no-wmd   # skip large downloads
    python tool_confusion_experiment.py --model llama3.2:3b  # different model
    python tool_confusion_experiment.py --runs 2             # faster smoke-test
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
MODEL = next((_argv[i + 1] for i, a in enumerate(_argv) if a == "--model"), "llama3.1:8b")
RUNS  = int(next((_argv[i + 1] for i, a in enumerate(_argv) if a == "--runs"), 4))
OLLAMA_BASE = os.getenv("OLLAMA_HOST", "http://localhost:11434")

SEP  = "=" * 72
SEP2 = "-" * 72


# ---------------------------------------------------------------------------
# T1 — search_products  (broad keyword search)
# ---------------------------------------------------------------------------
def search_products(query: str) -> str:
    """Search the product catalog for items matching a keyword or name."""
    return json.dumps([
        {"id": "P1", "name": f"Result for '{query}' #1", "price_usd": 49},
        {"id": "P2", "name": f"Result for '{query}' #2", "price_usd": 89},
    ])


# ---------------------------------------------------------------------------
# T2 — find_items  (stock / availability — synonym of T1)
# ---------------------------------------------------------------------------
def find_items(query: str) -> str:
    """Find available items in the store inventory that match a search term."""
    return json.dumps([
        {"id": "I1", "name": f"In-stock match for '{query}'", "available": True},
        {"id": "I2", "name": f"Limited stock: '{query}'",     "available": True},
    ])


# ---------------------------------------------------------------------------
# T3 — browse_category  (moderately similar to T1/T2)
# ---------------------------------------------------------------------------
def browse_category(category: str) -> str:
    """Browse and list all products belonging to a specific store category."""
    return json.dumps([
        {"id": "C1", "category": category, "name": "Product A"},
        {"id": "C2", "category": category, "name": "Product B"},
        {"id": "C3", "category": category, "name": "Product C"},
    ])


# ---------------------------------------------------------------------------
# T4 — track_order  (clearly different domain)
# ---------------------------------------------------------------------------
def track_order(order_id: str) -> str:
    """Track the shipping and delivery status of a customer order by ID."""
    return json.dumps({
        "order_id": order_id,
        "status": "in_transit",
        "estimated_delivery": "2026-06-14",
        "carrier": "UPS",
    })


TOOLS = [search_products, find_items, browse_category, track_order]
TOOL_NAMES = [t.__name__ for t in TOOLS]


# ---------------------------------------------------------------------------
# 12 prompts — ground truth labelled, tool name never mentioned
# ---------------------------------------------------------------------------
PROMPTS: list[dict] = [
    # ── T1: search_products ──────────────────────────────────────────────
    {"tool": "search_products",  "text": "I'm looking for a wireless mouse."},
    {"tool": "search_products",  "text": "Do you carry noise-cancelling headphones?"},
    {"tool": "search_products",  "text": "Can you find me something suitable for 4K gaming?"},
    # ── T2: find_items ───────────────────────────────────────────────────
    {"tool": "find_items",       "text": "Is the Samsung Galaxy S24 available right now?"},
    {"tool": "find_items",       "text": "What gaming keyboards do you currently have in stock?"},
    {"tool": "find_items",       "text": "I need a USB-C hub I can buy and receive today."},
    # ── T3: browse_category ──────────────────────────────────────────────
    {"tool": "browse_category",  "text": "What's in your laptop section?"},
    {"tool": "browse_category",  "text": "Show me everything under home office furniture."},
    {"tool": "browse_category",  "text": "What products do you carry in the audio accessories area?"},
    # ── T4: track_order ──────────────────────────────────────────────────
    {"tool": "track_order",      "text": "I placed an order last week — where is it?"},
    {"tool": "track_order",      "text": "Can you look up order ORD-5522 for me?"},
    {"tool": "track_order",      "text": "My delivery is late, can you check on it?"},
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
    mat  = TfidfVectorizer().fit_transform(docs).toarray()
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


def ollama_similarities() -> dict[tuple, float] | None:
    try:
        vecs = {}
        for name in TOOL_NAMES:
            resp = requests.post(
                f"{OLLAMA_BASE}/api/embed",
                json={"model": MODEL, "input": TOOL_TEXTS[name]},
                timeout=60,
            )
            resp.raise_for_status()
            vecs[name] = np.array(resp.json()["embeddings"][0], dtype=np.float32)

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
    """Fraction of runs where truth=tool_a but model called tool_b, or vice versa."""
    relevant = [r for r in runs if r.ground_truth in (tool_a, tool_b)]
    if not relevant:
        return 0.0
    confused = [r for r in relevant if not r.success
                and r.tool_called in (tool_a, tool_b)]
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
            if not r.success and r.tool_called:
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
