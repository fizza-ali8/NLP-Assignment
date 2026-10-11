"""
eval_retrieval.py
-----------------
Measures RETRIEVAL QUALITY (not speed) with the real embedding model, on
60 labelled queries in eval/retrieval_queries.json:
  - 48 in-domain questions, many deliberately paraphrased (so a semantic
    model is needed), each labelled with the document(s) that answer it
  - 12 questions that must retrieve NOTHING (off-topic, or library-related
    but not covered by the corpus)

Run after building the index:

    python -m rag.ingest
    python eval_retrieval.py                # uses MIN_SCORE / TOP_K from rag/retriever.py
    python eval_retrieval.py --sweep        # also tries other thresholds
    python eval_retrieval.py --verbose      # prints every query with its top hits

Reported (paste the summary into the README):
  hit@1, hit@k, MRR      - did the right document come back, and how high?
  no-match precision     - share of the "must retrieve nothing" queries that
                           correctly returned nothing at the current threshold
  score separation       - lowest score of a correct top hit vs highest top
                           score among the "must retrieve nothing" queries.
                           If the first number is above the second there is a
                           threshold that separates them perfectly; the
                           script suggests the midpoint.

`--fake` swaps in the lexical stand-in embedder used by the unit tests so
the script itself can be checked offline. Its numbers say nothing about the
real model - do not report them.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
QUERIES = os.path.join(HERE, "eval", "retrieval_queries.json")


def top_hits(query, k, min_score, margin=0.0):
    """Returns (ranked top-k, ranked top-k above threshold, best score of every source)."""
    from rag import retriever as ret
    from rag.embedder import embed_query

    idx = ret._get_index()
    if not idx.loaded:
        raise SystemExit(f"Index not available: {idx.load_error}")
    import numpy as np

    q = np.array(embed_query(query), dtype=np.float32)
    q = q / (np.linalg.norm(q) or 1.0)
    scores = idx.vectors @ q
    order = np.argsort(-scores)
    seen, ranked = set(), []
    for i in order:  # collapse several chunks of one document into one rank
        src = idx.metadata[i]["source"]
        if src not in seen:
            seen.add(src)
            ranked.append((src, float(scores[i])))
        if len(ranked) >= k:
            break
    kept = [(s, sc) for s, sc in ranked if sc >= min_score]
    if margin and kept:
        kept = [(s, sc) for s, sc in kept if sc >= kept[0][1] - margin]
    best_by_source = {}
    for i, sc in enumerate(scores):
        src = idx.metadata[i]["source"]
        if sc > best_by_source.get(src, -1.0):
            best_by_source[src] = float(sc)
    return ranked, kept, best_by_source


def evaluate(queries, k, min_score, verbose=False, margin=0.0):
    n_dom = hit1 = hitk = 0
    rr_sum = 0.0
    none_total = none_ok = 0
    best_correct, worst_none = [], []
    misses = []
    for item in queries:
        ranked, kept, best = top_hits(item["q"], k, min_score, margin)
        expect = set(item["expect"])
        if expect:
            n_dom += 1
            rank = next((r for r, (s, _) in enumerate(kept, 1) if s in expect), None)
            if rank == 1:
                hit1 += 1
            if rank:
                hitk += 1
                rr_sum += 1.0 / rank
            # score of the best correct doc, even if below threshold, for separation stats
            best_correct.append(max(best.get(e, 0.0) for e in expect))
            if not rank:
                misses.append((item["q"], sorted(expect), ranked[:2]))
        else:
            none_total += 1
            top = ranked[0][1] if ranked else 0.0
            worst_none.append((top, item["q"]))
            if not kept:
                none_ok += 1
        if verbose:
            tag = "OK  " if (expect and any(s in expect for s, _ in kept)) or (not expect and not kept) else "MISS"
            print(f"[{tag}] {item['q']}")
            for s, sc in ranked[:3]:
                mark = "*" if s in expect else " "
                keep = "" if sc >= min_score else " (below threshold)"
                print(f"        {mark} {sc:.3f}  {s}{keep}")
    return {
        "n_dom": n_dom, "hit1": hit1, "hitk": hitk, "mrr": rr_sum / max(n_dom, 1),
        "none_total": none_total, "none_ok": none_ok,
        "min_correct": min(best_correct) if best_correct else 0.0,
        "max_none": max((t for t, _ in worst_none), default=0.0),
        "worst_none": sorted(worst_none, reverse=True)[:3],
        "misses": misses,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=None, help="top-k (default: retriever.TOP_K)")
    ap.add_argument("--min-score", type=float, default=None, help="threshold (default: retriever.MIN_SCORE)")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--fake", action="store_true", help="lexical stand-in embedder; NOT the real model")
    args = ap.parse_args()

    from rag import retriever as ret

    ctx = None
    if args.fake:
        from tests import support
        ctx = support.FakeRag()
        ctx.__enter__()
        print("!! --fake: lexical stand-in embedder. These numbers are NOT the real model's. !!\n")

    k = args.k or ret.TOP_K
    min_score = args.min_score if args.min_score is not None else ret.MIN_SCORE
    with open(QUERIES, encoding="utf-8") as fh:
        queries = json.load(fh)["queries"]

    r = evaluate(queries, k, min_score, args.verbose)
    print(f"Index: {len(ret._get_index().metadata)} chunks | k={k} | MIN_SCORE={min_score}")
    print(f"In-domain queries: {r['n_dom']}")
    print(f"  hit@1 : {r['hit1']}/{r['n_dom']} = {r['hit1'] / r['n_dom']:.1%}")
    print(f"  hit@{k} : {r['hitk']}/{r['n_dom']} = {r['hitk'] / r['n_dom']:.1%}")
    print(f"  MRR   : {r['mrr']:.3f}")
    print(f"Must-retrieve-nothing queries: {r['none_total']}")
    print(f"  correctly empty: {r['none_ok']}/{r['none_total']} = {r['none_ok'] / r['none_total']:.1%}")
    print("Score separation:")
    print(f"  lowest score of a correct top hit      : {r['min_correct']:.3f}")
    print(f"  highest top score on no-match queries  : {r['max_none']:.3f}")
    if r["min_correct"] > r["max_none"]:
        print(f"  -> a perfect threshold exists; suggested MIN_SCORE ~ {(r['min_correct'] + r['max_none']) / 2:.2f}")
    else:
        print("  -> scores overlap: no single threshold separates every case (see misses / sweep)")
    if r["worst_none"]:
        print("  closest false positives:", "; ".join(f"{s:.3f} '{q}'" for s, q in r["worst_none"]))
    if r["misses"]:
        print("\nIn-domain misses:")
        for q, exp, got in r["misses"]:
            print(f"  - {q!r}\n      expected {exp}\n      got      {[(s, round(sc, 3)) for s, sc in got]}")

    if args.sweep:
        print("\nThreshold sweep (k fixed):")
        print("  MIN_SCORE | hit@k  | correctly-empty")
        for t in (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50):
            rr = evaluate(queries, k, t)
            print(f"   {t:.2f}     | {rr['hitk']:>2}/{rr['n_dom']} | {rr['none_ok']:>2}/{rr['none_total']}")

    if args.sweep:
        print("\nRelative-margin sweep (drop chunks scoring more than M below the best; fewer chunks = shorter prompt = lower TTFT):")
        print("  margin | hit@1 | hit@k | MRR  | avg docs kept")
        for m in (0.0, 0.20, 0.15, 0.10, 0.07, 0.05):
            rr = evaluate(queries, k, min_score, margin=m)
            kept_counts = [len(top_hits(i["q"], k, min_score, m)[1]) for i in queries if i["expect"]]
            print(f"  {m:5.2f}  | {rr['hit1']:>2}/{rr['n_dom']} | {rr['hitk']:>2}/{rr['n_dom']} | {rr['mrr']:.3f} | {sum(kept_counts) / len(kept_counts):.2f}")

    print("\n--- Summary (paste into README) ---")
    print(f"k={k}, MIN_SCORE={min_score}: hit@1 {r['hit1'] / r['n_dom']:.0%}, hit@{k} {r['hitk'] / r['n_dom']:.0%}, "
          f"MRR {r['mrr']:.2f}, no-match correct {r['none_ok']}/{r['none_total']} "
          f"({r['n_dom']} in-domain + {r['none_total']} no-match queries)")
    if ctx:
        ctx.__exit__(None, None, None)


if __name__ == "__main__":
    main()
