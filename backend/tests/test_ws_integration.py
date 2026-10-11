"""
End-to-end tests through the real FastAPI app over real WebSockets.

The local model is replaced by a tiny fake Ollama server that streams a
canned reply and RECORDS every request it receives, so these tests can prove
what the backend actually sends to the model (static system prompt, excerpts
attached to the last user turn, num_ctx, ...). Retrieval uses the real corpus
and the lexical test embedder (see tests/support.py).
"""

import asyncio
import json
import socket
import threading
import time
import unittest

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from websockets.sync.client import connect

from tests import support
import llm_engine
from prompts import SYSTEM_PROMPT


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ServerThread:
    def __init__(self, app):
        self.port = free_port()
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="error"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def start(self):
        self.thread.start()
        for _ in range(200):
            if self.server.started:
                return self
            time.sleep(0.05)
        raise RuntimeError("server did not start")

    def stop(self):
        self.server.should_exit = True
        self.thread.join(5)


class FakeOllama:
    """Streams a canned reply token by token and records every request body."""

    def __init__(self, token_delay=0.0):
        self.requests = []
        self.token_delay = token_delay
        app = FastAPI()

        @app.post("/api/chat")
        async def chat(request: Request):
            body = await request.json()
            self.requests.append(body)
            if body.get("stream") is False:  # warm-up call
                return {"message": {"content": "ok"}, "done": True}

            async def gen():
                for tok in ["Sure", ", ", "happy ", "to ", "help", "."]:
                    if self.token_delay:
                        await asyncio.sleep(self.token_delay)
                    yield json.dumps({"message": {"content": tok}, "done": False}) + "\n"
                yield json.dumps({"done": True}) + "\n"

            return StreamingResponse(gen(), media_type="application/x-ndjson")

        self.server = ServerThread(app).start()
        self.url = f"http://127.0.0.1:{self.server.port}/api/chat"

    def chat_requests(self):
        return [r for r in self.requests if r.get("stream") is True]


def chat(ws, message, session_id=""):
    """Send one message; return every event up to and including 'done'/'error'."""
    ws.send(json.dumps({"session_id": session_id, "message": message}))
    events = []
    while True:
        ev = json.loads(ws.recv(timeout=30))
        events.append(ev)
        if ev["type"] in ("done", "error"):
            return events


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rag = support.FakeRag()
        cls.rag.__enter__()
        cls.fake = FakeOllama()
        cls._old_url = llm_engine.OLLAMA_URL
        llm_engine.OLLAMA_URL = cls.fake.url
        import main
        cls.main = main
        main.RAG_ENABLED = True
        cls.app_server = ServerThread(main.app).start()
        cls.ws_url = f"ws://127.0.0.1:{cls.app_server.port}/ws/chat"
        cls.http = f"http://127.0.0.1:{cls.app_server.port}"

    @classmethod
    def tearDownClass(cls):
        cls.app_server.stop()
        cls.fake.server.stop()
        llm_engine.OLLAMA_URL = cls._old_url
        cls.rag.__exit__(None, None, None)

    def setUp(self):
        self.fake.requests.clear()
        self.main.RAG_ENABLED = True
        self.rag.ret._query_cache_search.cache_clear()

    def new_ws(self):
        return connect(self.ws_url)


class Contract(Base):
    def test_events_follow_the_assignment_1_contract(self):
        with self.new_ws() as ws:
            events = chat(ws, "What is the overdue fine per day?")
        self.assertEqual(events[0]["type"], "session")
        self.assertTrue(events[0]["session_id"])
        tokens = [e for e in events if e["type"] == "token"]
        self.assertGreater(len(tokens), 1, "response must stream token by token")
        done = events[-1]
        self.assertEqual(done["type"], "done")
        self.assertEqual(done["content"], "".join(t["content"] for t in tokens))
        self.assertIsInstance(done["ttft"], float)
        self.assertIsInstance(done["tokens_per_sec"], float)

    def test_malformed_requests_get_clear_errors_and_the_socket_survives(self):
        with self.new_ws() as ws:
            for raw in ("{not json", json.dumps({"session_id": ""}), json.dumps({"session_id": "", "message": "   "}),
                        json.dumps(["a", "list"])):
                ws.send(raw)
                ev = json.loads(ws.recv(timeout=10))
                self.assertEqual(ev["type"], "error", raw)
                self.assertTrue(ev["content"])
            events = chat(ws, "What are your hours?")  # still usable afterwards
            self.assertEqual(events[-1]["type"], "done")

    def test_session_memory_is_kept_per_session_id(self):
        with self.new_ws() as ws:
            first = chat(ws, "Do you have Atomic Habits?")
            sid = first[0]["session_id"]
            chat(ws, "ok thanks", session_id=sid)
        last = self.fake.chat_requests()[-1]["messages"]
        joined = " ".join(m["content"] for m in last)
        self.assertIn("Do you have Atomic Habits?", joined)
        self.assertIn("Sure, happy to help.", joined, "previous assistant reply is part of the history")


class PromptSentToTheModel(Base):
    def test_system_prompt_is_static_and_excerpts_ride_on_the_last_user_turn(self):
        with self.new_ws() as ws:
            chat(ws, "What is the overdue fine per day?")
            chat(ws, "When is the library open on Saturday?")
        reqs = self.fake.chat_requests()
        self.assertEqual(len(reqs), 2)
        sys1, sys2 = reqs[0]["messages"][0], reqs[1]["messages"][0]
        self.assertEqual(sys1["role"], "system")
        self.assertEqual(sys1["content"], SYSTEM_PROMPT)
        self.assertEqual(sys1["content"], sys2["content"], "system prompt must not change between turns (prefix cache)")
        last1 = reqs[0]["messages"][-1]
        self.assertEqual(last1["role"], "user")
        self.assertTrue(last1["content"].startswith("What is the overdue fine per day?"))
        self.assertIn("LIBRARY DOCUMENTS", last1["content"])
        self.assertIn("$0.25", last1["content"], "the grounded fact must reach the model")

    def test_request_options(self):
        with self.new_ws() as ws:
            chat(ws, "hello")
        req = self.fake.chat_requests()[-1]
        self.assertTrue(req["stream"])
        self.assertEqual(req["options"]["num_ctx"], llm_engine.NUM_CTX)
        self.assertEqual(req["model"], llm_engine.MODEL_NAME)
        self.assertEqual(req["keep_alive"], llm_engine.KEEP_ALIVE)

    def test_stored_history_is_free_of_retrieved_text(self):
        with self.new_ws() as ws:
            ev = chat(ws, "What is the overdue fine per day?")
        hist = self.main.conv_manager.get_full_history(ev[0]["session_id"])
        self.assertEqual(hist[0]["content"], "What is the overdue fine per day?")

    def test_follow_up_question_retrieves_the_book_from_the_previous_turn(self):
        with self.new_ws() as ws:
            e1 = chat(ws, "Do you have Atomic Habits?")
            e2 = chat(ws, "how many copies of it?", session_id=e1[0]["session_id"])
        self.assertEqual(e2[-1]["sources"][0]["title"], "Atomic Habits")
        self.assertIn("5 total", self.fake.chat_requests()[-1]["messages"][-1]["content"])

    def test_follow_up_keeps_previous_document_even_if_retrieval_drifts(self):
        # Real embeddings can drift on a rewritten follow-up (all book entries look alike).
        # Simulate that: the rewritten query retrieves nothing useful at all.
        import main as m
        original = m.retrieve_safe
        calls = {"n": 0}

        async def drifting(query, k=4):
            calls["n"] += 1
            return await original(query, k) if calls["n"] == 1 else []
        m.retrieve_safe = drifting
        try:
            with self.new_ws() as ws:
                e1 = chat(ws, "Do you have Atomic Habits?")
                e2 = chat(ws, "how many copies of it?", session_id=e1[0]["session_id"])
        finally:
            m.retrieve_safe = original
        self.assertEqual(e2[-1]["sources"][0]["title"], "Atomic Habits")
        self.assertIn("5 total", self.fake.chat_requests()[-1]["messages"][-1]["content"])

    def test_new_topic_does_not_inherit_the_previous_document(self):
        with self.new_ws() as ws:
            e1 = chat(ws, "Do you have Atomic Habits?")
            e2 = chat(ws, "What is the overdue fine per day for late items?", session_id=e1[0]["session_id"])
        self.assertNotIn("Atomic Habits", [x["title"] for x in e2[-1].get("sources", [])])

    def test_long_conversation_never_exceeds_the_context_budget(self):
        import conversation_manager as cm
        with self.new_ws() as ws:
            sid = ""
            for i in range(10):
                ev = chat(ws, f"Question {i}: " + "tell me about overdue fines " * 70, session_id=sid)
                sid = ev[0]["session_id"] if ev[0]["type"] == "session" else sid
        for req in self.fake.chat_requests():
            self.assertLessEqual(sum(len(m["content"]) for m in req["messages"]), cm.CONTEXT_CHAR_BUDGET)
            self.assertEqual(req["messages"][-1]["role"], "user")


class Citations(Base):
    def test_grounded_answer_carries_deduplicated_sources_in_relevance_order(self):
        with self.new_ws() as ws:
            done = chat(ws, "What happens if I lose a book and how big is the overdue fine?")[-1]
        src = done["sources"]
        self.assertTrue(src)
        self.assertEqual(src[0]["source"], "policies/overdue_fines_and_lost_items.md")
        self.assertEqual(src[0]["title"], "Overdue Fines and Lost Items")
        self.assertEqual(len({s["source"] for s in src}), len(src), "no duplicates")

    def test_only_documents_the_model_actually_saw_are_cited(self):
        with self.new_ws() as ws:
            ev = chat(ws, "What is the difference between Standard and Premium membership, fines and holds?")
        sent = self.fake.chat_requests()[-1]["messages"][-1]["content"]
        cited = [x["source"] for x in ev[-1]["sources"]]
        self.assertTrue(cited)
        for src in cited:
            self.assertIn(f"({src})", sent, f"{src} cited but never shown to the model")

    def test_ungrounded_turn_has_no_sources_field_and_no_excerpts(self):
        with self.new_ws() as ws:
            done = chat(ws, "How is the weather today?")[-1]
        self.assertEqual(done["type"], "done")
        self.assertNotIn("sources", done)
        self.assertNotIn("LIBRARY DOCUMENTS", self.fake.chat_requests()[-1]["messages"][-1]["content"])


class FailureHandling(Base):
    def test_rag_disabled_behaves_like_assignment_1(self):
        self.main.RAG_ENABLED = False
        with self.new_ws() as ws:
            done = chat(ws, "What is the overdue fine per day?")[-1]
        self.assertNotIn("sources", done)
        self.assertEqual(self.fake.chat_requests()[-1]["messages"][-1]["content"], "What is the overdue fine per day?")

    def test_missing_index_still_answers(self):
        import os
        import shutil
        idx = self.rag.index_dir
        backup = idx + "_bak"
        shutil.move(idx, backup)
        try:
            self.rag.ret._index = None
            with self.new_ws() as ws:
                done = chat(ws, "What is the overdue fine per day?")[-1]
            self.assertEqual(done["type"], "done")
            self.assertNotIn("sources", done)
            import urllib.request
            st = json.loads(urllib.request.urlopen(f"{self.http}/rag/status", timeout=5).read())
            self.assertFalse(st["index_loaded"])
            self.assertIn("rag.ingest", st["error"])
        finally:
            shutil.move(backup, idx)
            self.rag.ret._index = None

    def test_embedder_crash_still_answers(self):
        original = self.rag.emb.embed_query
        self.rag.emb.embed_query = lambda t: (_ for _ in ()).throw(RuntimeError("model file corrupt"))
        try:
            with self.new_ws() as ws:
                done = chat(ws, "a never seen before query about hours")[-1]
            self.assertEqual(done["type"], "done")
            self.assertNotIn("sources", done)
        finally:
            self.rag.emb.embed_query = original

    def test_slow_embedder_does_not_stall_the_reply(self):
        original, old_timeout = self.rag.emb.embed_query, self.rag.ret.RETRIEVAL_TIMEOUT_S

        def slow(t):
            time.sleep(1.0)
            return original(t)

        self.rag.emb.embed_query = slow
        self.rag.ret.RETRIEVAL_TIMEOUT_S = 0.2
        try:
            with self.new_ws() as ws:
                t0 = time.perf_counter()
                done = chat(ws, "yet another unique slow question about fines")[-1]
                elapsed = time.perf_counter() - t0
            self.assertEqual(done["type"], "done")
            self.assertNotIn("sources", done)
            self.assertLess(elapsed, 0.9, "reply must not wait for the slow embedder")
        finally:
            self.rag.emb.embed_query, self.rag.ret.RETRIEVAL_TIMEOUT_S = original, old_timeout

    def test_model_server_down_returns_error_then_recovers(self):
        good = llm_engine.OLLAMA_URL
        llm_engine.OLLAMA_URL = f"http://127.0.0.1:{free_port()}/api/chat"
        try:
            with self.new_ws() as ws:
                ev = chat(ws, "hello")[-1]
                self.assertEqual(ev["type"], "error")
                self.assertIn("ollama", ev["content"].lower())
                llm_engine.OLLAMA_URL = good
                self.assertEqual(chat(ws, "hello again")[-1]["type"], "done")
        finally:
            llm_engine.OLLAMA_URL = good

    def test_client_disconnect_mid_stream_does_not_hurt_the_server(self):
        self.fake.token_delay = 0.05
        try:
            with self.new_ws() as ws:
                ws.send(json.dumps({"session_id": "", "message": "Explain all your membership tiers."}))
                while json.loads(ws.recv(timeout=10))["type"] != "token":
                    pass
            # leaving the 'with' closes the socket while tokens are still being streamed
            time.sleep(0.5)
        finally:
            self.fake.token_delay = 0.0
        import urllib.request
        self.assertEqual(json.loads(urllib.request.urlopen(f"{self.http}/health", timeout=5).read())["status"], "ok")
        with self.new_ws() as ws2:
            self.assertEqual(chat(ws2, "hello")[-1]["type"], "done")


class Concurrency(Base):
    def test_many_simultaneous_users_all_get_isolated_replies(self):
        self.fake.token_delay = 0.03
        results, errors = {}, []

        def user(k):
            try:
                with self.new_ws() as ws:
                    ev = chat(ws, f"unique-marker-{k} what is the overdue fine per day?")
                    results[k] = (ev[0]["session_id"], ev[-1])
            except Exception as e:  # noqa: BLE001
                errors.append(repr(e))

        try:
            t0 = time.perf_counter()
            threads = [threading.Thread(target=user, args=(k,)) for k in range(8)]
            [t.start() for t in threads]
            [t.join(30) for t in threads]
            elapsed = time.perf_counter() - t0
        finally:
            self.fake.token_delay = 0.0
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 8)
        self.assertEqual(len({sid for sid, _ in results.values()}), 8, "every user gets their own session")
        self.assertTrue(all(done["type"] == "done" and done["sources"] for _, done in results.values()))
        # no cross-talk: each request the model saw mentions exactly one user's marker
        for req in self.fake.chat_requests():
            text = " ".join(m["content"] for m in req["messages"][1:])
            self.assertEqual(sum(f"unique-marker-{k}" in text for k in range(8)), 1)
        self.assertLess(elapsed, 10, "8 concurrent users should finish quickly with a fast model stub")

    def test_status_and_reload_endpoints(self):
        import urllib.request
        st = json.loads(urllib.request.urlopen(f"{self.http}/rag/status", timeout=5).read())
        self.assertTrue(st["index_loaded"])
        self.assertGreaterEqual(st["chunks"], 69)
        self.assertGreaterEqual(st["top_k"], 3)
        req = urllib.request.Request(f"{self.http}/rag/reload", method="POST")
        rl = json.loads(urllib.request.urlopen(req, timeout=5).read())
        self.assertTrue(rl["index_loaded"])


if __name__ == "__main__":
    unittest.main()
