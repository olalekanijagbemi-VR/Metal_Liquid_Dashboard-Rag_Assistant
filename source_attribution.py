"""
source_attribution.py
======================
Fixes misleading source attribution in a hybrid (FAISS + BM25 + RRF) RAG
pipeline.

ROOT CAUSE
----------
RRF scores are rank-based, not magnitude-based, so two chunks with very
different true relevance can land a few ranks apart and appear almost
equally "relevant". Fixed thresholds can't separate them.

FIX
---
1. Carry raw similarity magnitude through fusion.
2. Re-score each merged chunk using IDF-weighted keyword overlap.
3. Optional single batched Groq LLM reranker for ambiguous cases.
4. Aggregate per source, then cut using GAP DETECTION on sorted scores.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional


# --------------------------------------------------------------------------
# 1. Tokenization + IDF-weighted keyword overlap
# --------------------------------------------------------------------------

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on",
    "for", "and", "or", "what", "which", "how", "does", "do", "did", "can",
    "with", "this", "that", "it", "as", "be", "by", "at", "from", "about",
}

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text) if t.lower() not in _STOPWORDS]


def build_corpus_idf(all_chunk_texts: list[str]) -> dict[str, float]:
    N = len(all_chunk_texts)
    df: dict[str, int] = defaultdict(int)
    for text in all_chunk_texts:
        for tok in set(tokenize(text)):
            df[tok] += 1
    return {tok: math.log((N + 1) / (freq + 0.5)) + 1.0 for tok, freq in df.items()}


def keyword_overlap_score(
    query: str, chunk_text: str, idf: Optional[dict[str, float]] = None
) -> float:
    q_tokens = set(tokenize(query))
    if not q_tokens:
        return 0.0
    c_tokens = set(tokenize(chunk_text))
    if idf is None:
        idf = {}
    weight = lambda t: idf.get(t, 1.0)
    matched_weight = sum(weight(t) for t in q_tokens if t in c_tokens)
    total_weight = sum(weight(t) for t in q_tokens)
    return matched_weight / total_weight if total_weight > 0 else 0.0


# --------------------------------------------------------------------------
# 2. Optional Groq LLM reranker
# --------------------------------------------------------------------------

def groq_llm_rerank(
    query: str,
    chunks: list[dict[str, Any]],
    groq_client,
    model: str = "llama-3.3-70b-versatile",
    max_chunks: int = 10,
) -> dict[int, float]:
    subset = chunks[:max_chunks]
    numbered = "\n\n".join(
        f"[{i}] (source: {c['source']})\n{c['text'][:600]}"
        for i, c in enumerate(subset)
    )
    prompt = f"""You are grading passage relevance for a search system.

Query: "{query}"

Below are numbered passages. For EACH passage, give a relevance score from
0 (completely unrelated to the query) to 10 (directly and specifically
answers the query). Passages that only share generic vocabulary with the
query but don't address it should score low (0-3).

Passages:
{numbered}

Respond with ONLY a JSON array of integers, one per passage, in order.
Example: [8, 2, 9, 0, 1]"""
    try:
        resp = groq_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200,
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        scores = json.loads(raw)
        return {i: max(0.0, min(10.0, float(s))) / 10.0 for i, s in enumerate(scores)}
    except Exception:
        return {}


# --------------------------------------------------------------------------
# 3. Composite chunk scoring
# --------------------------------------------------------------------------

def _normalize(values: list[float]) -> list[float]:
    if not values:
        return values
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


@dataclass
class ScoredChunk:
    source: str
    text: str
    page: Optional[int]
    rrf_score: float
    keyword_score: float = 0.0
    llm_score: Optional[float] = None
    composite: float = 0.0


def score_chunks(
    query: str,
    merged_chunks: list[dict[str, Any]],
    idf: Optional[dict[str, float]] = None,
    groq_client=None,
    llm_model: str = "llama-3.3-70b-versatile",
    use_llm: bool = True,
) -> list[ScoredChunk]:
    rrf_norm = _normalize([c.get("rrf_score", 0.0) for c in merged_chunks])
    kw_scores = [keyword_overlap_score(query, c["text"], idf) for c in merged_chunks]

    llm_scores: dict[int, float] = {}
    if use_llm and groq_client is not None:
        llm_scores = groq_llm_rerank(query, merged_chunks, groq_client, model=llm_model)

    scored: list[ScoredChunk] = []
    have_llm = bool(llm_scores)
    if have_llm:
        w_rrf, w_kw, w_llm = 0.25, 0.30, 0.45
    else:
        w_rrf, w_kw, w_llm = 0.40, 0.60, 0.0

    for i, c in enumerate(merged_chunks):
        llm_s = llm_scores.get(i)
        composite = w_rrf * rrf_norm[i] + w_kw * kw_scores[i]
        if llm_s is not None:
            composite += w_llm * llm_s
        scored.append(
            ScoredChunk(
                source=c["source"],
                text=c["text"],
                page=c.get("page"),
                rrf_score=c.get("rrf_score", 0.0),
                keyword_score=kw_scores[i],
                llm_score=llm_s,
                composite=composite,
            )
        )
    return scored


# --------------------------------------------------------------------------
# 4. Per-source aggregation + gap-based cutoff
# --------------------------------------------------------------------------

def aggregate_by_source(scored_chunks: list[ScoredChunk]) -> list[dict[str, Any]]:
    by_source: dict[str, list[ScoredChunk]] = defaultdict(list)
    for c in scored_chunks:
        by_source[c.source].append(c)

    aggregated = []
    for source, chunks in by_source.items():
        best = max(chunks, key=lambda c: c.composite)
        aggregated.append(
            {
                "source": source,
                "score": best.composite,
                "page": best.page,
                "preview": best.text[:220].strip() + ("..." if len(best.text) > 220 else ""),
                "n_chunks": len(chunks),
            }
        )
    aggregated.sort(key=lambda s: s["score"], reverse=True)
    return aggregated


def gap_cutoff(
    sorted_sources: list[dict[str, Any]],
    gap_ratio: float = 0.55,
    min_keep_score: float = 0.15,
) -> list[dict[str, Any]]:
    if not sorted_sources:
        return []
    kept = [sorted_sources[0]] if sorted_sources[0]["score"] >= min_keep_score else []
    if not kept:
        return []
    for prev, curr in zip(sorted_sources, sorted_sources[1:]):
        if curr["score"] < min_keep_score:
            break
        ratio = curr["score"] / prev["score"] if prev["score"] > 0 else 0
        if ratio < gap_ratio:
            break
        kept.append(curr)
    return kept


# --------------------------------------------------------------------------
# 5. Public entry point
# --------------------------------------------------------------------------

def filter_relevant_sources(
    query: str,
    merged_chunks: list[dict[str, Any]],
    idf: Optional[dict[str, float]] = None,
    groq_client=None,
    llm_model: str = "llama-3.3-70b-versatile",
    use_llm: bool = True,
    gap_ratio: float = 0.55,
    min_keep_score: float = 0.15,
) -> list[dict[str, Any]]:
    scored = score_chunks(
        query, merged_chunks, idf=idf, groq_client=groq_client,
        llm_model=llm_model, use_llm=use_llm,
    )
    by_source = aggregate_by_source(scored)
    kept = gap_cutoff(by_source, gap_ratio=gap_ratio, min_keep_score=min_keep_score)
    return [
        {"source": s["source"], "score": round(s["score"], 4),
         "page": s["page"], "preview": s["preview"]}
        for s in kept
    ]
