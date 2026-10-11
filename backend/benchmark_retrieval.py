"""
benchmark_retrieval.py
-----------------------
Standalone latency benchmark for RETRIEVAL ONLY (Assignment 2, Phase III):
query embedding time + vector search time, measured separately from LLM
generation (that's benchmark.py). Run this after building the index:

    python -m rag.ingest        # build the index once
    python benchmark_retrieval.py --runs 10

This does NOT require `ollama serve` to be running - it only exercises
the embedding model and the numpy flat-index search.
"""

import argparse
import statistics
import time
import sys

sys.path.insert(0, ".")

from rag.embedder import embed_query
from rag.retriever import _cosine_search, _get_index, TOP_K  # noqa: F401 (internal, benchmark-only use)

TEST_QUERIES = [
    "What are your library hours?",
    "Do you have Atomic Habits available?",
    "What's the overdue fine policy?",
    "Can I book a study room?",
    "How do I get a library card?",
    "Tell me about the summer reading program",
    "What happens if I lose a book?",
    "Is there a membership fee?",
    "Can I renew my loan online?",
    "Do you have any books about Mars?",
]


def run_once(query: str):
    t0 = time.perf_counter()
    vec = embed_query(query)
    t1 = time.perf_counter()
    results = _cosine_search(vec, TOP_K)
    t2 = time.perf_counter()
    return (t1 - t0), (t2 - t1), len(results)


def main(runs: int):
    idx = _get_index()
    if not idx.loaded:
        print(f"ERROR: {idx.load_error}")
        print("Run `python -m rag.ingest` first to build the index.")
        return

    print(f"Index loaded: {idx.vectors.shape[0]} chunks, {idx.vectors.shape[1]}-dim embeddings.\n")

    embed_times, search_times, totals = [], [], []
    for i in range(runs):
        q = TEST_QUERIES[i % len(TEST_QUERIES)]
        embed_t, search_t, n_results = run_once(q)
        embed_times.append(embed_t)
        search_times.append(search_t)
        totals.append(embed_t + search_t)
        print(f"[run {i+1}] {q!r}")
        print(f"          embed={embed_t*1000:.1f}ms  search={search_t*1000:.1f}ms  "
              f"total={(embed_t+search_t)*1000:.1f}ms  matches={n_results}")

    print("\n--- Summary (paste into README) ---")
    print(f"Runs: {runs}")
    print(f"Avg query embedding time: {statistics.mean(embed_times)*1000:.1f}ms")
    print(f"Avg vector search time:   {statistics.mean(search_times)*1000:.1f}ms")
    print(f"Avg total retrieval time: {statistics.mean(totals)*1000:.1f}ms")
    print(f"Max total retrieval time: {max(totals)*1000:.1f}ms")
    under_1s = sum(1 for t in totals if t < 1.0)
    print(f"Runs under 1s target: {under_1s}/{runs}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()
    main(args.runs)
