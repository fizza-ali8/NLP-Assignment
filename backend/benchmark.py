"""
benchmark.py
------------
Standalone latency benchmark for the LOCAL MODEL (generation only, not
retrieval - see benchmark_retrieval.py for that, measured separately as
required by Assignment 2 Phase III). Run this AFTER `ollama serve` is
running and you have pulled a model (see README for exact commands).

Usage:
    python benchmark.py
    python benchmark.py --model qwen2.5:3b --runs 5

It measures, per prompt:
  - time to first token (TTFT)
  - total generation time
  - tokens/second (approx, using whitespace-split token count)

and prints per-run numbers plus an average, in a format you can paste
straight into the README.
"""

import argparse
import asyncio
import time
import statistics
import httpx

OLLAMA_URL = "http://localhost:11434/api/chat"

TEST_PROMPTS = [
    "Hi, what are your library hours?",
    "Do you have 'Atomic Habits' available right now?",
    "Can I renew 'Clean Code' if no one else has it on hold?",
    "What's the overdue fine if I return a book 4 days late?",
    "I'd like to place a hold on 'The Silent Patient'.",
]


async def run_once(client: httpx.AsyncClient, model: str, prompt: str):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a concise library assistant."},
            {"role": "user", "content": prompt},
        ],
        "stream": True,
        "options": {"num_ctx": 8192},
    }
    start = time.perf_counter()
    first_token = None
    token_count = 0

    async with client.stream("POST", OLLAMA_URL, json=payload, timeout=120.0) as resp:
        async for line in resp.aiter_lines():
            if not line.strip():
                continue
            import json
            chunk = json.loads(line)
            piece = chunk.get("message", {}).get("content", "")
            if piece:
                if first_token is None:
                    first_token = time.perf_counter()
                token_count += len(piece.split()) or 1
            if chunk.get("done"):
                break

    end = time.perf_counter()
    ttft = (first_token - start) if first_token else 0.0
    total = end - start
    tps = token_count / total if total > 0 else 0.0
    return ttft, total, tps


async def main(model: str, runs: int):
    ttfts, totals, tps_list = [], [], []
    async with httpx.AsyncClient() as client:
        for i in range(runs):
            prompt = TEST_PROMPTS[i % len(TEST_PROMPTS)]
            ttft, total, tps = await run_once(client, model, prompt)
            ttfts.append(ttft)
            totals.append(total)
            tps_list.append(tps)
            print(f"[run {i+1}] prompt={prompt!r}")
            print(f"          TTFT={ttft:.3f}s  total={total:.3f}s  ~tok/s={tps:.2f}")

    print("\n--- Summary (paste into README) ---")
    print(f"Model: {model}")
    print(f"Runs: {runs}")
    print(f"Avg TTFT: {statistics.mean(ttfts):.3f}s")
    print(f"Avg total response time: {statistics.mean(totals):.3f}s")
    print(f"Avg tokens/sec: {statistics.mean(tps_list):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    asyncio.run(main(args.model, args.runs))
