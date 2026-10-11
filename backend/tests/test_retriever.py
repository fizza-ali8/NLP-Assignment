"""Retriever: top-k, threshold, caching, follow-ups, formatting, failure modes, concurrency."""

import asyncio
import os
import tempfile
import time
import unittest

from tests import support
from rag import retriever as ret
from rag.retriever import (
    RetrievedChunk, RetrievalUnavailable, build_retrieval_query, format_context,
    retrieve, retrieve_safe,
)


class WithIndex(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rag = support.FakeRag()
        cls.rag.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.rag.__exit__(None, None, None)

    def setUp(self):
        ret._query_cache_search.cache_clear()


class Search(WithIndex):
    def test_top_k_is_at_least_three_and_sorted(self):
        res = retrieve("how much is the overdue fine per day and what happens with lost books")
        self.assertGreaterEqual(len(res), 3, "assignment requires k >= 3")
        self.assertLessEqual(len(res), ret.TOP_K)
        scores = [r.score for r in res]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_most_relevant_document_ranks_first(self):
        cases = {
            "what are the overdue fines per day": "overdue_fines_and_lost_items",
            "when is the library open on saturday": "library_hours",
            "how do i get a library card": "library_card",
            "can i book a study room": "study_rooms",
            "is atomic habits available": "atomic_habits",
            "how much does printing cost per page": "printing_and_computers",
            "interlibrary loan request turnaround": "interlibrary_loan",
        }
        for query, expected in cases.items():
            res = retrieve(query)
            self.assertTrue(res, query)
            self.assertIn(expected, res[0].source, f"{query!r} -> {res[0].source}")

    def test_unrelated_query_returns_nothing_instead_of_a_weak_match(self):
        for q in ("how is the weather in paris", "quantum chromodynamics lattice gauge", "tell me a joke about penguins"):
            self.assertEqual(retrieve(q), [], q)

    def test_empty_query(self):
        self.assertEqual(retrieve("   "), [])

    def test_threshold_is_respected(self):
        old = ret.MIN_SCORE
        try:
            ret.MIN_SCORE = 0.99
            ret._query_cache_search.cache_clear()
            self.assertEqual(retrieve("overdue fines per day"), [])
        finally:
            ret.MIN_SCORE = old


class Caching(WithIndex):
    def test_repeat_and_normalized_repeat_hit_the_cache(self):
        retrieve("What are your hours")
        before = ret._query_cache_search.cache_info()
        retrieve("What are your hours")
        retrieve("  what ARE   your hours ")
        after = ret._query_cache_search.cache_info()
        self.assertEqual(after.hits - before.hits, 2)
        self.assertEqual(after.misses, before.misses)

    def test_cached_hit_does_not_call_the_embedder(self):
        retrieve("printing cost per page")
        import rag.embedder as emb
        original = emb.embed_query
        emb.embed_query = lambda t: (_ for _ in ()).throw(AssertionError("embedder called on a cache hit"))
        try:
            self.assertTrue(retrieve("printing cost per page"))
        finally:
            emb.embed_query = original


class FollowUps(unittest.TestCase):
    def test_short_or_pronoun_followups_inherit_previous_topic(self):
        prev = "Do you have Atomic Habits?"
        self.assertEqual(build_retrieval_query("how many copies of it?", prev), f"{prev} how many copies of it?")
        self.assertEqual(build_retrieval_query("yes", prev), f"{prev} yes")
        self.assertEqual(build_retrieval_query("and for Premium?", prev), f"{prev} and for Premium?")

    def test_ordinary_questions_with_filler_words_are_not_follow_ups(self):
        prev = "What is the late fee?"
        for q in ("Is there a book club?", "Is there a study room?", "Do you also offer printing?"):
            self.assertEqual(build_retrieval_query(q, prev), q, q)

    def test_self_contained_questions_are_left_alone(self):
        prev = "Do you have Atomic Habits?"
        q = "What is the overdue fine for a book returned four days late?"
        self.assertEqual(build_retrieval_query(q, prev), q)
        self.assertEqual(build_retrieval_query("hello there", None), "hello there")

    def test_followup_retrieves_the_right_book_end_to_end(self):
        with support.FakeRag():
            q = build_retrieval_query("how many copies of it?", "Do you have Atomic Habits?")
            self.assertIn("atomic_habits", retrieve(q)[0].source)
            self.assertNotIn("atomic_habits", " ".join(r.source for r in retrieve("how many copies of it?")))


class StickyFollowUp(WithIndex):
    def test_chunks_for_source_and_merge(self):
        sticky = ret.chunks_for_source("books/atomic_habits.md")
        self.assertTrue(sticky and all(c.source == "books/atomic_habits.md" for c in sticky))
        self.assertEqual(ret.chunks_for_source("books/does_not_exist.md"), [])
        found = retrieve("overdue fines per day")
        merged = ret.merge_followup(sticky, found + sticky)
        self.assertEqual(merged[0].source, "books/atomic_habits.md")
        self.assertEqual(len({(c.source, c.text) for c in merged}), len(merged), "no duplicates")
        self.assertLessEqual(len(merged), ret.TOP_K)

    def test_relative_margin_drops_weak_tail(self):
        old = ret.REL_MARGIN
        try:
            ret.REL_MARGIN = 0.0
            ret._query_cache_search.cache_clear()
            wide = retrieve("overdue fines per day and lost books")
            ret.REL_MARGIN = 0.01
            ret._query_cache_search.cache_clear()
            narrow = retrieve("overdue fines per day and lost books")
        finally:
            ret.REL_MARGIN = old
            ret._query_cache_search.cache_clear()
        self.assertLessEqual(len(narrow), len(wide))
        self.assertTrue(narrow and narrow[0].source == wide[0].source)


class InjectionSelection(unittest.TestCase):
    def mk(self, n, score, words=60):
        return RetrievedChunk(text=" ".join(["w"] * words), source=f"policies/p{n}.md", title=f"P{n}", score=score)

    def test_weak_tail_is_not_injected_but_best_is_always_kept(self):
        chunks = [self.mk(0, 0.80), self.mk(1, 0.74), self.mk(2, 0.60), self.mk(3, 0.55)]
        sel = ret.select_for_context(chunks, max_chars=5000)
        self.assertEqual([c.title for c in sel], ["P0", "P1"])
        self.assertEqual(len(ret.select_for_context([self.mk(0, 0.31)], max_chars=10)), 1)

    def test_followup_sticky_chunk_does_not_hide_new_matches(self):
        sticky = [self.mk(9, 1.0)]
        found = [self.mk(1, 0.52), self.mk(2, 0.50)]
        merged = ret.merge_followup(sticky, found)
        sel = ret.select_for_context(merged, max_chars=5000)
        self.assertEqual([c.title for c in sel], ["P9", "P1", "P2"])


class Formatting(unittest.TestCase):
    def chunk(self, n, words=100):
        return RetrievedChunk(text=" ".join(["word"] * words), source=f"policies/p{n}.md", title=f"P{n}", score=0.9 - n / 10)

    def test_empty(self):
        self.assertEqual(format_context([]), "")

    def test_budget_is_enforced_but_best_chunk_always_kept(self):
        chunks = [self.chunk(i, 120) for i in range(4)]
        out = format_context(chunks, max_chars=1500)
        self.assertLessEqual(len(out), 1500 + 200)
        self.assertIn("[Source: P0 (policies/p0.md)]", out)
        self.assertNotIn("P3", out)
        giant = [self.chunk(0, 2000)]
        self.assertIn("P0", format_context(giant, max_chars=100))

    def test_sources_are_labelled_in_relevance_order(self):
        out = format_context([self.chunk(0, 10), self.chunk(1, 10)])
        self.assertLess(out.index("P0"), out.index("P1"))


class Failures(unittest.TestCase):
    def run_async(self, coro):
        return asyncio.run(coro)

    def test_missing_index_degrades_and_is_not_cached_forever(self):
        with support.FakeRag(build=False) as rag:
            with self.assertRaises(RetrievalUnavailable):
                retrieve("overdue fines")
            self.assertEqual(self.run_async(retrieve_safe("overdue fines")), [])
            self.assertEqual(ret._query_cache_search.cache_info().currsize, 0, "an outage must not be cached as 'no results'")
            st = ret.status()
            self.assertFalse(st["index_loaded"])
            self.assertIn("rag.ingest", st["error"])

            # the index appears later (user runs ingest) -> picked up with no restart
            old = ret.RELOAD_RETRY_S
            ret.RELOAD_RETRY_S = 0
            try:
                rag.ing.build_index.__globals__  # ensure module alive
                import io
                from contextlib import redirect_stdout
                with redirect_stdout(io.StringIO()):
                    rag.ing.build_index()
                self.assertTrue(retrieve("overdue fines per day"))
            finally:
                ret.RELOAD_RETRY_S = old

    def test_index_is_hot_reloaded_when_file_changes(self):
        with tempfile.TemporaryDirectory() as corpus:
            os.makedirs(os.path.join(corpus, "policies"))
            support.write_text(os.path.join(corpus, "policies", "a.md"), "# Alpha\n\n" + "alpha apple " * 30)
            with support.FakeRag(corpus_dir=corpus) as rag:
                self.assertTrue(retrieve("alpha apple"))
                self.assertEqual(retrieve("zebra stripes"), [])
                support.write_text(os.path.join(corpus, "policies", "b.md"), "# Zebra\n\n" + "zebra stripes " * 30)
                import io
                from contextlib import redirect_stdout
                with redirect_stdout(io.StringIO()):
                    rag.ing.build_index()
                os.utime(os.path.join(rag.index_dir, "embeddings.npy"), (time.time() + 5, time.time() + 5))
                res = retrieve("zebra stripes")
                self.assertTrue(res, "new document should be searchable without restarting")
                self.assertIn("b.md", res[0].source)

    def test_corrupt_and_out_of_sync_index(self):
        with support.FakeRag() as rag:
            support.write_text(os.path.join(rag.index_dir, "embeddings.npy"), b"not a numpy file", "wb")
            os.utime(os.path.join(rag.index_dir, "embeddings.npy"), (time.time() + 5, time.time() + 5))
            self.assertEqual(self.run_async(retrieve_safe("overdue fines")), [])
            self.assertIn("Failed to load", ret.status()["error"])
        with support.FakeRag() as rag:
            import json
            meta_path = os.path.join(rag.index_dir, "chunks.json")
            meta = support.load_json(meta_path)
            support.write_text(meta_path, json.dumps(meta[:-3]))
            ret.reload_index()
            self.assertEqual(self.run_async(retrieve_safe("overdue fines")), [])
            self.assertIn("out of sync", ret.status()["error"])

    def test_embedder_exception_degrades(self):
        with support.FakeRag() as rag:
            def boom(_):
                raise RuntimeError("onnx exploded")
            rag.emb.embed_query = boom
            self.assertEqual(self.run_async(retrieve_safe("overdue fines")), [])

    def test_slow_embedder_times_out_quickly(self):
        with support.FakeRag() as rag:
            def slow(t):
                time.sleep(0.6)
                return support.fake_vec(t)
            rag.emb.embed_query = slow
            old = ret.RETRIEVAL_TIMEOUT_S
            ret.RETRIEVAL_TIMEOUT_S = 0.1
            try:
                t0 = time.perf_counter()
                self.assertEqual(self.run_async(retrieve_safe("brand new slow query")), [])
                self.assertLess(time.perf_counter() - t0, 0.4, "must give up at the timeout, not wait for the embedder")
            finally:
                ret.RETRIEVAL_TIMEOUT_S = old


class Concurrency(unittest.TestCase):
    def test_burst_of_users_keeps_event_loop_responsive(self):
        with support.FakeRag() as rag:
            def slowish(t):
                time.sleep(0.03)
                return support.fake_vec(t)
            rag.emb.embed_query = slowish

            async def scenario():
                gaps, stop = [], False

                async def ticker():
                    last = time.perf_counter()
                    while not stop:
                        await asyncio.sleep(0.005)
                        now = time.perf_counter()
                        gaps.append(now - last)
                        last = now

                task = asyncio.create_task(ticker())
                results = await asyncio.gather(*[
                    retrieve_safe(f"overdue fines per day question number {i}") for i in range(20)
                ])
                stop = True
                await task
                return results, max(gaps)

            results, worst_gap = asyncio.run(scenario())
            self.assertTrue(all(len(r) >= 1 for r in results), "every user should still get context")
            self.assertLess(worst_gap, 0.1, f"event loop stalled for {worst_gap:.3f}s during a retrieval burst")


if __name__ == "__main__":
    unittest.main()
