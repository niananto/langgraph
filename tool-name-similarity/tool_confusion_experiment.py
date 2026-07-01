"""
tool_confusion_experiment.py — at what embedding distance do tools stop confusing an LLM?

v3 — similarity LADDER design.  Earlier versions established the endpoints:
  v1: same intent + same `query` arg            → SBERT 0.73, 38% confusion
  v2: dictionary-synonym names, distinct intent → SBERT max 0.47, ~no confusion

v4 spans the zones with 5 tools, all defensibly distinct in real life but
with description vocabulary engineered to hit SBERT similarity targets:

  T1  search_products(keywords)        — list items matching a keyword
  T2  check_stock(product_name)        — availability of a KNOWN product
                                          target vs T1: SBERT > 0.7
  T3  list_category_products(category) — category browsing
                                          target vs T1/T2: SBERT 0.5–0.7
  T4  track_order(order_id)            — shipping status (low anchor < 0.35)
  T5  recommend_products(need)         — near-PARAPHRASE of T1's description
                                          target vs T1: SBERT > 0.8

KEY FINDING (v6 run): confusion tracked Llama's OWN contextual geometry, not
surface words. The only confused pair (search ↔ list_category, 12.5%) was the
#1-ranked pair under Llama-Ctx (0.8992) — beating search ↔ recommend (0.8937)
— even though every lexical metric (BoW 0.93, BERTScore 0.98, WMD 0.88) rated
search ↔ recommend as most similar and it never confused. So the router
conflates tools its own embedding places closest, not the lexically closest.

v7 leans into that: a LISTING CLUSTER of tools that all perform the SAME
operation — "return a filtered list of products" — differing only in the
filter dimension. These should sit very high in Llama-Ctx and confuse most:
  T1  search_products(keywords)          — filter by keyword
  T3  list_category_products(category)   — filter by category/section
  T5  recommend_products(need)           — filter + pick best
  T6  list_brand_products(brand)         — filter by brand
  T7  filter_products_by_price(max_price)— filter by budget
plus two anchors that should stay distinct:
  T2  check_stock(product_name)  — single named product's availability
  T4  track_order(order_id)      — shipping status (low-similarity anchor)

Prompts naturally bleed across the cluster: a brand name ("show me Logitech")
reads as a keyword search; a category word reads as a keyword; a price prompt
overlaps nothing lexically but shares the "list products" intent. Ground truth
is the dimension the prompt names (brand→brand, category→category, budget→price).

21 prompts (3 per tool, ground-truth labelled but tool name never mentioned).
4 runs per prompt = 84 total runs.

5 similarity metrics computed once upfront for all tool pairs. They split into
LEXICAL (surface word overlap), STATIC embeddings, and CONTEXTUAL embeddings:
  1. BoW-Cos    (sklearn CountVectorizer — raw word-count cosine)      LEXICAL
  2. BERTScore  (bert-score, RoBERTa token matching, ~500 MB 1st run)  contextual (RoBERTa)
  3. WMD        (gensim + GloVe-100, ~128 MB 1st run)                   STATIC word vectors
  4. SBERT      (sentence-transformers MiniLM, ~80 MB 1st run)         contextual (MiniLM)
  5. Llama-Ctx  (transformers, full Llama 3.1 8B, ~16 GB 1st run)      contextual (Llama itself)

Llama-Ctx is the faithful match to the routing model's internal similarity:
the same network embeds the tool's JSON schema text via all 32 layers, mean-
pooled. It is the contextual counterpart to the raw embed_tokens lookup in
tool_name_embeddings.py (which reads layer-0 vectors only, no transformer applied).

All 5 metrics embed the FULL OpenAI-format tool schema (name + description +
parameters, via convert_to_openai_tool — the same helper middleware_deep_dive.py
uses to show the real provider payload), not just "name: docstring". Argument
names/types are part of what the model reads, so they're part of what can
cause — or prevent — confusion.

Headline output: for each metric, the similarity threshold below which
confusion dropped to zero — the "safe separation" for tool naming.

Run:
    python tool_confusion_experiment.py
    python tool_confusion_experiment.py --no-bert --no-wmd --no-llama-ctx  # skip big downloads
    python tool_confusion_experiment.py --model llama3.2:3b                # different chat model
    python tool_confusion_experiment.py --runs 2                           # faster smoke-test

Llama-Ctx needs an HF token (HF_TOKEN in .env) for the gated meta-llama repo;
without one it falls back to the public NousResearch/Meta-Llama-3.1-8B mirror.
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
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from dotenv import load_dotenv
load_dotenv()  # load .env for HF_TOKEN if present

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# CLI flags
# ---------------------------------------------------------------------------
_argv = sys.argv[1:]
SKIP_BERT      = "--no-bert"      in _argv
SKIP_WMD       = "--no-wmd"       in _argv
SKIP_LLAMA_CTX = "--no-llama-ctx" in _argv
MODEL = next((_argv[i + 1] for i, a in enumerate(_argv) if a == "--model"), "llama3.1:8b")
RUNS  = int(next((_argv[i + 1] for i, a in enumerate(_argv) if a == "--runs"), 4))

SEP  = "=" * 72
SEP2 = "-" * 72


# ---------------------------------------------------------------------------
# T1 — search_products  (catalog discovery by keyword)
# ---------------------------------------------------------------------------
def search_products(keywords: str) -> str:
    """In the store's product catalog, search for the products that match the customer's request so the customer can buy them."""
    return json.dumps([
        {"id": "P1", "name": f"Match for '{keywords}' #1", "price_usd": 49},
        {"id": "P2", "name": f"Match for '{keywords}' #2", "price_usd": 89},
    ])


# ---------------------------------------------------------------------------
# T2 — check_stock  (availability of a KNOWN product — distinct intent from
#                    T1's discovery, but maximal description overlap:
#                    product / store / catalog / customers / buy)
#                    target: SBERT > 0.7 vs T1
# ---------------------------------------------------------------------------
def check_stock(product_name: str) -> str:
    """In the store's product catalog, check the product that matches the customer's request so the customer can buy it if in stock."""
    return json.dumps({
        "product_name": product_name,
        "in_stock": True,
        "units_left": 7,
        "restock_date": None,
    })


# ---------------------------------------------------------------------------
# T3 — list_category_products  (category browsing — same store/product
#                               domain, different operation)
#                               target: SBERT 0.5–0.7 vs T1 and T2
# ---------------------------------------------------------------------------
def list_category_products(category: str) -> str:
    """In the store's product catalog, list the products that match the customer's category so the customer can buy them."""
    return json.dumps([
        {"id": "C1", "category": category, "name": "Product A", "price_usd": 120},
        {"id": "C2", "category": category, "name": "Product B", "price_usd": 250},
        {"id": "C3", "category": category, "name": "Product C", "price_usd": 75},
    ])


# ---------------------------------------------------------------------------
# T4 — track_order  (low-similarity anchor: different domain entirely)
#                    target: SBERT < 0.35 vs everything
# ---------------------------------------------------------------------------
def track_order(order_id: str) -> str:
    """Track the shipping and delivery status of a customer order by its order ID."""
    return json.dumps({
        "order_id": order_id,
        "status": "in_transit",
        "estimated_delivery": "2026-06-14",
        "carrier": "UPS",
    })


# ---------------------------------------------------------------------------
# T5 — recommend_products  (near-PARAPHRASE of T1's description — engineered
#                           for the highest SBERT pair in the set, target >0.8.
#                           Defensible distinction: T1 lists items that MATCH a
#                           keyword; T5 recommends the single BEST one for a
#                           need. Same domain, same objects, near-identical
#                           wording — the question is whether the model can
#                           route on "what do you have" vs "what's best".)
# ---------------------------------------------------------------------------
def recommend_products(need: str) -> str:
    """In the store's product catalog, recommend the products that match the customer's request so the customer can buy them."""
    return json.dumps({
        "recommended": {"id": "R1", "name": f"Best pick for '{need}'", "price_usd": 129},
        "why": "highest rated within budget",
    })


# ---------------------------------------------------------------------------
# T6 — list_brand_products  (LISTING CLUSTER: same operation as search /
#                            list_category / recommend — "return a filtered
#                            list of products" — differing only in the filter
#                            dimension (brand). Llama's contextual embedding
#                            should rate the whole listing cluster very high;
#                            brand names in prompts also bleed toward
#                            check_stock, which names products.)
# ---------------------------------------------------------------------------
def list_brand_products(brand: str) -> str:
    """In the store's product catalog, list the products that match the customer's preferred brand so the customer can buy them."""
    return json.dumps([
        {"id": "B1", "brand": brand, "name": f"{brand} item #1", "price_usd": 60},
        {"id": "B2", "brand": brand, "name": f"{brand} item #2", "price_usd": 140},
    ])


# ---------------------------------------------------------------------------
# T7 — filter_products_by_price  (LISTING CLUSTER member: filter by budget.
#                                 Same "return a filtered product list"
#                                 operation as the other listing tools.)
# ---------------------------------------------------------------------------
def filter_products_by_price(max_price: int) -> str:
    """In the store's product catalog, list the products that match the customer's price budget so the customer can buy them."""
    return json.dumps([
        {"id": "F1", "name": "Budget item A", "price_usd": min(max_price, 25)},
        {"id": "F2", "name": "Budget item B", "price_usd": min(max_price, 40)},
    ])


TOOLS = [
    search_products,
    check_stock,
    list_category_products,
    track_order,
    recommend_products,
    list_brand_products,
    filter_products_by_price,
]
TOOL_NAMES = [t.__name__ for t in TOOLS]


# ---------------------------------------------------------------------------
# 15 prompts — ground truth labelled, tool name never mentioned.
# T1 (search) and T5 (recommend) prompts use the SAME products and differ only
# on a subtle signal: "what do you have / show me what exists" (list matches)
# vs "what's best / what would you recommend" (recommend one). This isolates
# the search↔recommend axis — the highest-SBERT pair in the set.
# T2 prompts name an EXACT product + availability keyword (strong signal).
# ---------------------------------------------------------------------------
PROMPTS: list[dict] = [
    # ── T1: search_products  (generic keyword — no brand/category/price/budget) ─
    {"tool": "search_products",         "text": "I want to buy a wireless mouse — what do you have?"},
    {"tool": "search_products",         "text": "Do you sell noise-cancelling headphones? Show me what there is."},
    {"tool": "search_products",         "text": "I need a mechanical keyboard. What options exist in the catalog?"},
    # ── T2: check_stock  (exact product named + availability) ─────────────
    {"tool": "check_stock",             "text": "Is the Logitech MX Master 3S in stock right now?"},
    {"tool": "check_stock",             "text": "Do you currently have the Sony WH-1000XM5 available?"},
    {"tool": "check_stock",             "text": "Can I get a Keychron K2 today, or is it sold out?"},
    # ── T3: list_category_products  (explicit section/department) ─────────
    {"tool": "list_category_products",  "text": "What's in your laptop section?"},
    {"tool": "list_category_products",  "text": "Show me everything in the home office department."},
    {"tool": "list_category_products",  "text": "Browse the audio accessories category for me."},
    # ── T4: track_order  (shipping status) ────────────────────────────────
    {"tool": "track_order",             "text": "Order ORD-5522 — where is it right now?"},
    {"tool": "track_order",             "text": "My package from order ORD-1001 hasn't arrived. What's the status?"},
    {"tool": "track_order",             "text": "When will order ORD-7733 be delivered?"},
    # ── T5: recommend_products  (recommend the BEST — "what's best") ──────
    {"tool": "recommend_products",      "text": "What's the best wireless mouse you'd recommend?"},
    {"tool": "recommend_products",      "text": "Which noise-cancelling headphones would you suggest for travel?"},
    {"tool": "recommend_products",      "text": "Can you recommend a good budget mechanical keyboard for me?"},
    # ── T6: list_brand_products  (explicit brand name) ────────────────────
    {"tool": "list_brand_products",     "text": "Show me everything Logitech makes."},
    {"tool": "list_brand_products",     "text": "What Sony products do you carry?"},
    {"tool": "list_brand_products",     "text": "List all the Anker items you stock."},
    # ── T7: filter_products_by_price  (explicit budget/price ceiling) ─────
    {"tool": "filter_products_by_price","text": "What do you have for under 50 dollars?"},
    {"tool": "filter_products_by_price","text": "Show me anything cheaper than 100 dollars."},
    {"tool": "filter_products_by_price","text": "List products that fit a 30 dollar budget."},
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
    llm = ChatOllama(model=MODEL, temperature=1).bind_tools(TOOLS)

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
# TOOL_TEXTS is the JSON schema LangGraph actually sends the model — name,
# description, AND the argument schema (property names, types, required list)
# via convert_to_openai_tool, the same helper middleware_deep_dive.py and
# schema_stress_test.py use to inspect the real provider payload. Argument
# names/types matter for confusion too: two tools with identical descriptions
# but different parameter shapes are less likely to be swapped by the model,
# and this makes that visible to every similarity metric below.
TOOL_TEXTS = {t.__name__: json.dumps(convert_to_openai_tool(t)) for t in TOOLS}
TOOL_PAIRS = list(combinations(TOOL_NAMES, 2))


def cosine_similarities() -> dict[tuple, float]:
    """Basic bag-of-words cosine similarity on raw word-count vectors.

    The textbook "cosine similarity": each description becomes a vector of
    raw term counts (no TF normalization, no IDF weighting), then cosine of
    the angle between the two count vectors. Pure lexical overlap.
    """
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sk_cos

    docs = [TOOL_TEXTS[n] for n in TOOL_NAMES]
    mat  = CountVectorizer().fit_transform(docs).toarray()
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


def llama_contextual_similarities() -> dict[tuple, float] | None:
    """Contextual cosine from the routing model's OWN hidden states.

    Loads the full Llama 3.1 8B model (all 32 transformer layers, unlike the
    raw embed_tokens-only lookup in tool_name_embeddings.py) and mean-pools the
    final-layer hidden states into one vector per tool description. This is the
    faithful counterpart to the model's internal similarity — the same network
    that does the tool routing, embedding the same tool-description text.

    Heavy: ~16 GB download on first run (cached afterwards) and a CPU forward
    pass per description. Skip with --no-llama-ctx.
    """
    if SKIP_LLAMA_CTX:
        print("  (--no-llama-ctx passed: skipping Llama contextual embeddings)")
        return None
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer

        # same HF-token + public-mirror fallback as tool_name_embeddings.py
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        model_id = "meta-llama/Meta-Llama-3.1-8B" if hf_token else "NousResearch/Meta-Llama-3.1-8B"

        print(f"  Model: {model_id} (full model — ~16 GB first run)")
        tok = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        # No device_map / low_cpu_mem_usage: those require the `accelerate`
        # package. Without them the model loads on CPU by default.
        # bfloat16 (not float16): CPU has no float16 matmul kernels, so a
        # float16 forward pass raises "not implemented for 'Half'"; bf16 works.
        model = AutoModel.from_pretrained(
            model_id,
            token=hf_token,
            dtype=torch.bfloat16,
        ).eval()

        vecs = {}
        for name in TOOL_NAMES:
            inputs = tok(TOOL_TEXTS[name], return_tensors="pt")
            with torch.no_grad():
                out = model(**inputs)
            # mean-pool final hidden states over the token sequence
            hidden = out.last_hidden_state[0]            # [seq_len, hidden]
            vecs[name] = hidden.mean(dim=0).float().numpy()

        sims = {}
        for a, b in TOOL_PAIRS:
            va, vb = vecs[a], vecs[b]
            sims[(a, b)] = float(
                np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9)
            )
        return sims
    except Exception as exc:
        print(f"  [Llama-Ctx] skipped — {exc}")
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

    print("  BoW-Cos (bag-of-words cosine)...", end=" ", flush=True)
    metrics["BoW-Cos"] = cosine_similarities()
    print("done")

    print("  BERTScore...", end=" ", flush=True)
    metrics["BERTScore"] = bert_similarities()
    print("done" if metrics["BERTScore"] else "skipped")

    print("  Word Mover's Distance...")
    metrics["WMD"] = wmd_similarities()

    print("  Sentence-BERT...", end=" ", flush=True)
    metrics["SBERT"] = sbert_similarities()
    print("done" if metrics["SBERT"] else "skipped")

    print("  Llama 3.1 contextual (transformers)...")
    metrics["Llama-Ctx"] = llama_contextual_similarities()
    print("  Llama-Ctx done" if metrics["Llama-Ctx"] else "  Llama-Ctx skipped")

    active_metrics = {k: v for k, v in metrics.items() if v is not None}

    # ── Print similarity matrix ───────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"SIMILARITY MATRIX  (all {len(TOOL_PAIRS)} tool pairs × active metrics)")
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
