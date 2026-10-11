"""
benchmark_e2e.py
----------------
END-TO-END latency through the real backend, measured from the client's
side of the WebSocket: send message -> first token -> last token. This is
what a user actually feels, so it includes retrieval + prompt building +
the model. (benchmark.py measures the model alone; benchmark_retrieval.py
measures retrieval alone.)

To measure what RAG costs, run it twice against the same model:

    # terminal A: normal server (RAG on)
    uvicorn main:app --port 8000
    python benchmark_e2e.py --label rag_on

    # stop the server, then restart with retrieval switched off
    #   Windows cmd:   set RAG_ENABLED=0
    #   then:          uvicorn main:app --port 8000
    python benchmark_e2e.py --label rag_off

Each run uses a fresh session per query (so conversation history does not
skew the numbers), does one unmeasured warm-up request first, and repeats
every query --repeat times.
"""

import argparse
import json
import statistics
import time

from websockets.sync.client import connect

QUERIES = [
    "What are your library hours on Saturday?",
    "How much is the overdue fine if I return a book 4 days late?",
    "Is Atomic Habits available right now?",
    "Can I renew a book and how many times?",
    "How do I get a library card?",
    "What is the difference between Standard and Premium membership?",
]


def one(url, text):
    with connect(url, max_size=None) as ws:
        t0 = time.perf_counter()
        ws.send(json.dumps({"session_id": "", "message": text}))
        first = None
        done = None
        while True:
            ev = json.loads(ws.recv(timeout=300))
            if ev["type"] == "token" and first is None:
                first = time.perf_counter()
            if ev["type"] in ("done", "error"):
                done = ev
                break
        total = time.perf_counter() - t0
        if done["type"] == "error":
            raise RuntimeError(done["content"])
        return (first - t0) if first else total, total, done


def pct(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, int(round(p / 100 * (len(values) - 1))))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://localhost:8000/ws/chat")
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--label", default="run")
    args = ap.parse_args()

    print(f"[{args.label}] warm-up request (not measured) ...")
    one(args.url, "hello")

    ttfts, totals, grounded = [], [], 0
    first_ttfts, repeat_ttfts = [], []   # first ask of a query vs identical repeats (cache hits)
    for q in QUERIES:
        for r in range(args.repeat):
            ttft, total, done = one(args.url, q)
            ttfts.append(ttft)
            totals.append(total)
            (first_ttfts if r == 0 else repeat_ttfts).append(ttft)
            grounded += bool(done.get("sources"))
            print(f"[{args.label}] {q[:52]:<52} ttft={ttft:6.2f}s total={total:6.2f}s "
                  f"sources={len(done.get('sources', []))}")

    n = len(ttfts)
    print(f"\n--- Summary [{args.label}] (paste into README) ---")
    print(f"requests: {n} ({len(QUERIES)} queries x {args.repeat}), answers with citations: {grounded}/{n}")
    print(f"TTFT, first ask of each query (realistic, new prompt): avg {statistics.mean(first_ttfts):.2f}s | median {statistics.median(first_ttfts):.2f}s | max {max(first_ttfts):.2f}s")
    if repeat_ttfts:
        print(f"TTFT, identical repeat (retrieval cache + Ollama prompt cache): avg {statistics.mean(repeat_ttfts):.2f}s | max {max(repeat_ttfts):.2f}s")
    print(f"TTFT   avg {statistics.mean(ttfts):.2f}s | median {statistics.median(ttfts):.2f}s | p95 {pct(ttfts, 95):.2f}s | max {max(ttfts):.2f}s")
    print(f"Total  avg {statistics.mean(totals):.2f}s | median {statistics.median(totals):.2f}s | p95 {pct(totals, 95):.2f}s | max {max(totals):.2f}s")


if __name__ == "__main__":
    main()
