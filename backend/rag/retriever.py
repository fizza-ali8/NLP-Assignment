"""
retriever.py
------------
Runtime retrieval (Assignment 2, Phase II/III/IV). Loads the index built by
ingest.py, then for each user message:
  1. Builds a retrieval query (build_retrieval_query): the message itself,
     plus the previous patron message when the new one is a short follow-up
     such as "how many copies of it?".
  2. Embeds that query with the same local model used at indexing time.
  3. Computes cosine similarity against every chunk (flat/brute-force
     search - see the "why no FAISS/Chroma" note below).
  4. Returns the top-k chunks above a minimum relevance threshold.

Why a plain numpy flat index instead of FAISS/Chroma: the assignment
names both as acceptable, but a flat (brute-force) index IS what FAISS's
simplest index type (IndexFlatIP) does internally - exact cosine/dot-product
search over every vector, no approximation. At our corpus size (~76
chunks) that is well under a millisecond, so an approximate-nearest-
neighbour library would add install risk (another native wheel, on a
machine that already hit wheel problems on a very new Python) for zero
latency benefit. The interface below (`retrieve(query, k)`) is the only
thing the rest of the system uses, so swapping in FAISS/Chroma later is a
localized change.

Failure handling (Phase IV): `retrieve_safe()` wraps embedding + search in
a timeout and a broad except, returning an empty list (never raising) if
the embedding step fails, the index is missing, or the call is too slow -
so a retrieval problem degrades the assistant to its Assignment 1
behaviour (static knowledge base only) instead of crashing or hanging the
WebSocket. Every degradation is LOGGED (logger "rag") so it is not silent,
and /rag/status in main.py reports the current state.

Staleness safety: the index is re-read automatically when embeddings.npy
changes on disk (so `python -m rag.ingest` can be re-run while the server
is up), and a missing index is retried every few seconds rather than being
remembered as "missing" forever. Failed/empty-because-unavailable lookups
are never put in the query cache.

Caching (Phase III): repeated queries are common ("what are your hours"
asked by many users). Results are cached in a small in-memory LRU keyed by
the normalized retrieval query, so a repeat skips embedding and search.

Concurrency (Phase III): the embedding model is CPU-bound, so retrieval
runs on a small dedicated thread pool (RETRIEVAL_WORKERS). That keeps the
asyncio event loop free for other users' WebSocket traffic and also bounds
how many embedding calls fight over the CPU at once (and over the CPU that
the LLM itself needs). Time spent waiting for a free worker counts toward
RETRIEVAL_TIMEOUT_S, so a burst of users degrades to "no context" rather
than stalling replies.
"""

import asyncio
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

logger = logging.getLogger("rag")

INDEX_DIR = os.path.join(os.path.dirname(__file__), "index")
TOP_K = 4                 # assignment requires k >= 3
MIN_SCORE = 0.30          # tuned with eval_retrieval.py --sweep on the real model (0.30: hit@4 45/48, no-match 9/12; 0.35 was worse)
MAX_CONTEXT_CHARS = 1000  # char budget for injected excerpts (see conversation_manager.py); 1500 gave ~11.6 s TTFT on the dev laptop
INJECT_MARGIN = 0.10      # inject a chunk only if it scores within this of the best chunk (retrieval itself still returns k=4)
RETRIEVAL_TIMEOUT_S = 2.0
RETRIEVAL_WORKERS = 2
CACHE_SIZE = 256
REL_MARGIN = 0.0          # optional trim of weak tail; evaluated at 0.10 (same hit@4) but left OFF: the brief requires k >= 3
FOLLOWUP_STICKY_CHUNKS = 2  # chunks of the previous turn's top document kept for a follow-up
RELOAD_RETRY_S = 5.0      # how often a missing/broken index is retried

_pool = ThreadPoolExecutor(max_workers=RETRIEVAL_WORKERS, thread_name_prefix="rag")


class RetrievalUnavailable(Exception):
    """The index is missing or unreadable (distinct from 'no relevant match')."""


@dataclass
class RetrievedChunk:
    text: str
    source: str
    title: str
    score: float


class _Index:
    """Loads embeddings.npy + chunks.json once and keeps them in memory."""

    def __init__(self):
        self.loaded = False
        self.vectors: np.ndarray | None = None
        self.metadata: list[dict] = []
        self.load_error: str | None = None
        self.mtime: float | None = None
        self.attempted_at = time.monotonic()
        self._load()

    def _load(self):
        emb_path = os.path.join(INDEX_DIR, "embeddings.npy")
        meta_path = os.path.join(INDEX_DIR, "chunks.json")
        if not (os.path.exists(emb_path) and os.path.exists(meta_path)):
            self.load_error = (
                "RAG index not found. Run `python -m rag.ingest` from the "
                "backend/ directory to build it."
            )
            return
        try:
            self.mtime = os.path.getmtime(emb_path)
            self.vectors = np.load(emb_path)
            with open(meta_path, "r", encoding="utf-8") as fh:
                self.metadata = json.load(fh)
            if self.vectors.ndim != 2 or len(self.metadata) != self.vectors.shape[0]:
                raise ValueError("embeddings.npy and chunks.json are out of sync")
            self.loaded = True
            self.load_error = None
        except Exception as e:  # noqa: BLE001
            self.vectors, self.metadata, self.loaded = None, [], False
            self.load_error = f"Failed to load RAG index: {e}"


_index: _Index | None = None


def _get_index() -> _Index:
    """Returns the index, re-reading it if the file changed or a load failed earlier."""
    global _index
    if _index is None:
        _index = _Index()
        if not _index.loaded:
            logger.warning("RAG index unavailable: %s", _index.load_error)
        return _index

    emb_path = os.path.join(INDEX_DIR, "embeddings.npy")
    stale = False
    if _index.loaded:
        try:
            stale = os.path.getmtime(emb_path) != _index.mtime
        except OSError:
            stale = True
    elif time.monotonic() - _index.attempted_at > RELOAD_RETRY_S:
        stale = True  # retry a missing/broken index every few seconds

    if stale:
        was_loaded = _index.loaded
        _index = _Index()
        _query_cache_search.cache_clear()
        if _index.loaded:
            logger.info("RAG index (re)loaded: %d chunks", len(_index.metadata))
        elif was_loaded:
            logger.warning("RAG index became unavailable: %s", _index.load_error)
    return _index


def reload_index():
    """Forces a fresh load (used by POST /rag/reload)."""
    global _index
    _index = _Index()
    _query_cache_search.cache_clear()
    if _index.loaded:
        logger.info("RAG index reloaded: %d chunks", len(_index.metadata))
    else:
        logger.warning("RAG index unavailable after reload: %s", _index.load_error)
    return _index


def status() -> dict:
    """Small health summary for GET /rag/status."""
    idx = _get_index()
    info = _query_cache_search.cache_info()
    return {
        "index_loaded": idx.loaded,
        "chunks": len(idx.metadata) if idx.loaded else 0,
        "embedding_dim": int(idx.vectors.shape[1]) if idx.loaded and idx.vectors is not None else None,
        "error": idx.load_error,
        "top_k": TOP_K,
        "min_score": MIN_SCORE,
        "cache": {"size": info.currsize, "max": info.maxsize, "hits": info.hits, "misses": info.misses},
    }


def _cosine_search(query_vec, k: int) -> list[RetrievedChunk]:
    idx = _get_index()
    if not idx.loaded or idx.vectors is None:
        return []

    q = np.array(query_vec, dtype=np.float32)
    norm = np.linalg.norm(q)
    if norm > 0:
        q = q / norm

    scores = idx.vectors @ q  # vectors are pre-normalized in ingest.py -> cosine similarity
    top_indices = np.argsort(-scores)[:k]

    results = []
    for i in top_indices:
        score = float(scores[i])
        if score < MIN_SCORE:
            continue
        meta = idx.metadata[i]
        results.append(RetrievedChunk(
            text=meta["text"], source=meta["source"], title=meta["title"], score=score,
        ))
    if REL_MARGIN and results:
        best = results[0].score
        results = [r for r in results if r.score >= best - REL_MARGIN]
    return results


@lru_cache(maxsize=CACHE_SIZE)
def _query_cache_search(normalized_query: str, k: int) -> tuple:
    """
    Cached core of retrieval, keyed by normalized query text + k. Returns a
    tuple (hashable, so cacheable) of plain tuples rather than dataclasses.
    Only called when the index is loaded, so an "unavailable" state is never
    cached as an empty result.
    """
    from rag.embedder import embed_query  # lazy import, see embedder.py

    query_vec = embed_query(normalized_query)
    results = _cosine_search(query_vec, k)
    return tuple((r.text, r.source, r.title, r.score) for r in results)


# Only real back-references. Words like "there", "also", "then", "one" appear in
# ordinary stand-alone questions ("Is there a book club?") and must not turn them
# into follow-ups (that wrongly pulled in the previous topic's document).
_FOLLOW_UP_HINT = re.compile(r"\b(it|its|that|this|those|these|them|they|he|she|him|her)\b", re.I)


def build_retrieval_query(user_message: str, previous_user_message: str | None = None) -> str:
    """
    Follow-up questions are meaningless on their own ("how many copies of
    it?", "and for Premium?", "yes"). When the new message is short or
    refers back with a pronoun, the previous patron message is prepended so
    the embedding still carries the topic. Long, self-contained questions
    are used as-is (so an unrelated new topic is not polluted by the old
    one).
    """
    msg = user_message.strip()
    if not previous_user_message:
        return msg
    words = msg.split()
    if len(words) <= 4 or (len(words) <= 9 and _FOLLOW_UP_HINT.search(msg)):
        return f"{previous_user_message.strip()} {msg}"
    return msg


def retrieve(query: str, k: int = TOP_K) -> list[RetrievedChunk]:
    """Synchronous retrieval with caching. Raises RetrievalUnavailable / embedder errors - see retrieve_safe()."""
    normalized = " ".join(query.strip().lower().split())
    if not normalized:
        return []
    idx = _get_index()
    if not idx.loaded:
        raise RetrievalUnavailable(idx.load_error or "index not loaded")
    raw = _query_cache_search(normalized, k)
    return [RetrievedChunk(text=t, source=s, title=ti, score=sc) for t, s, ti, sc in raw]


async def retrieve_safe(query: str, k: int = TOP_K) -> list[RetrievedChunk]:
    """
    Failure-handling wrapper (Phase IV). Runs retrieve() on the retrieval
    thread pool with a timeout and swallows every exception - embedding
    failure, missing/corrupt index, or a slow search all degrade to "no
    context found" rather than propagating. Each degradation is logged.
    """
    loop = asyncio.get_running_loop()
    t0 = time.perf_counter()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(_pool, retrieve, query, k),
            timeout=RETRIEVAL_TIMEOUT_S,
        )
        logger.info("retrieval ok: %d chunks in %.0f ms", len(result), (time.perf_counter() - t0) * 1000)
        return result
    except asyncio.TimeoutError:
        logger.warning("retrieval timed out after %.1fs - continuing without context", RETRIEVAL_TIMEOUT_S)
        return []
    except RetrievalUnavailable as e:
        logger.warning("retrieval unavailable (%s) - continuing without context", e)
        return []
    except Exception as e:  # noqa: BLE001 - any retrieval failure degrades gracefully
        logger.warning("retrieval failed (%r) - continuing without context", e)
        return []


def chunks_for_source(source: str, limit: int = FOLLOWUP_STICKY_CHUNKS) -> list[RetrievedChunk]:
    """First chunks of one document, straight from the in-memory index (no embedding, microseconds)."""
    idx = _get_index()
    if not idx.loaded:
        return []
    out = []
    for meta in idx.metadata:
        if meta["source"] == source:
            out.append(RetrievedChunk(text=meta["text"], source=source, title=meta["title"], score=1.0))
            if len(out) >= limit:
                break
    return out


def merge_followup(sticky: list[RetrievedChunk], found: list[RetrievedChunk], k: int = TOP_K) -> list[RetrievedChunk]:
    """Previous turn's document first, then the new matches, de-duplicated, at most k chunks."""
    top = found[0].score if found else MIN_SCORE
    merged, seen = [], set()
    for c in [RetrievedChunk(text=x.text, source=x.source, title=x.title, score=top) for x in sticky] + list(found):
        key = (c.source, c.text)
        if key not in seen:
            seen.add(key)
            merged.append(c)
    return merged[:k]


def warm_up() -> float:
    """
    Loads the embedding model and runs one embedding so the FIRST patron
    query does not pay the model-load cost (which could exceed the retrieval
    timeout). Called in a background thread at server start. Returns seconds taken.
    """
    from rag.embedder import embed_query

    t0 = time.perf_counter()
    embed_query("warm up")
    return time.perf_counter() - t0


def select_for_context(chunks: list[RetrievedChunk], max_chars: int = MAX_CONTEXT_CHARS) -> list[RetrievedChunk]:
    """
    Which retrieved chunks are actually injected into the prompt (Phase II:
    inject without exceeding the model's context window, explicit strategy).
    Retrieval returns the top k = 4; injection is then limited by
      1. relevance order,
      2. relevance margin: a chunk must score within INJECT_MARGIN of the best
         chunk (weaker ones add prompt tokens, i.e. CPU time and noise, not signal),
      3. a character budget (max_chars),
    always keeping at least the best chunk. Citations are built from THIS list,
    so a source is only ever cited if the model saw it.
    """
    selected, used = [], 0
    best = chunks[0].score if chunks else 0.0
    for c in chunks:
        if selected and c.score < best - INJECT_MARGIN - 1e-9:
            break
        block_len = len(f"[Source: {c.title} ({c.source})]\n{c.text}")
        if used + block_len > max_chars and selected:
            break
        selected.append(c)
        used += block_len
    return selected


def format_context(chunks: list[RetrievedChunk], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Builds the excerpts block attached to the patron's message from select_for_context()."""
    return "\n\n---\n\n".join(
        f"[Source: {c.title} ({c.source})]\n{c.text}" for c in select_for_context(chunks, max_chars)
    )
