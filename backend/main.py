"""
main.py
-------
FastAPI backend for the Riverbend Library Assistant.

Endpoints:
  GET  /health              - simple liveness check
  POST /session/new         - create a new session id
  POST /session/{id}/reset  - wipe a session's history
  WS   /ws/chat             - real-time streaming chat

Assignment 2 additions
  GET  /rag/status          - is the index loaded, how many chunks, cache stats
  POST /rag/reload          - re-read the index after re-running rag.ingest

The WebSocket message contract from Assignment 1 is unchanged: same message
types, same required fields. The only addition is an OPTIONAL "sources"
array on the "done" event, present only when retrieval found relevant
chunks for that turn (bonus: visible citations). A client that ignores
unknown fields behaves exactly as in Assignment 1.

Per-turn flow:  user message -> retrieve (timeout-guarded, never raises)
-> conversation manager builds [system][older turns][user + excerpts]
-> stream tokens from the local model -> "done" (+ sources).

Concurrency notes:
  - Each WebSocket connection is its own asyncio task (Starlette does this),
    so one user's slow request does not block another's.
  - The ConversationManager is a single in-memory object keyed by
    session_id, so concurrent sessions never see each other's history.
  - Retrieval is CPU-bound, so it runs on a small dedicated thread pool
    (rag/retriever.py), keeping the event loop free for other users.
  - A single local Ollama instance generates one reply at a time unless
    OLLAMA_NUM_PARALLEL is raised, so concurrent *generation* queues; that
    limit is documented in the README.
  - Set RAG_ENABLED=0 to switch retrieval off (identical to Assignment 1),
    which is how the README's with/without-RAG latency comparison is made.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import json
import logging
import os
import time

from conversation_manager import ConversationManager, MAX_TURNS
from llm_engine import stream_chat, warm_up as llm_warm_up
from prompts import SYSTEM_PROMPT
from rag import retriever
from rag.retriever import (
    retrieve_safe, format_context, select_for_context, build_retrieval_query, chunks_for_source, merge_followup,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app")

RAG_ENABLED = os.environ.get("RAG_ENABLED", "1") != "0"

async def _warm_up_background():
    """
    Pre-loads the embedding model and the LLM (with the static system prompt)
    so the first patron message is not slow. Best effort: failures are
    logged and never stop the server from starting or serving.
    """
    if RAG_ENABLED:
        try:
            secs = await asyncio.to_thread(retriever.warm_up)
            logger.info("embedding model warmed up in %.1fs", secs)
        except Exception as e:  # noqa: BLE001
            logger.warning("embedding warm-up failed (%r); retrieval will degrade to no-context", e)
    secs = await llm_warm_up(SYSTEM_PROMPT)
    if secs:
        logger.info("LLM warmed up in %.1fs", secs)
    else:
        logger.warning("LLM warm-up skipped (is `ollama serve` running?)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_warm_up_background())  # background: never delays startup
    try:
        yield
    finally:
        task.cancel()


app = FastAPI(title="Riverbend Library Assistant API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # fine for a local demo; tighten before any real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

conv_manager = ConversationManager()


class NewSessionResponse(BaseModel):
    session_id: str


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/rag/status")
async def rag_status():
    """Reports whether retrieval is working, so a broken index is not silent."""
    return {"rag_enabled": RAG_ENABLED, **retriever.status()}


@app.post("/rag/reload")
async def rag_reload():
    """Re-reads the index from disk (also happens automatically when the file changes)."""
    idx = await asyncio.to_thread(retriever.reload_index)
    return {"index_loaded": idx.loaded, "chunks": len(idx.metadata), "error": idx.load_error}


@app.post("/session/new", response_model=NewSessionResponse)
async def new_session():
    sid = conv_manager.create_session()
    return {"session_id": sid}


@app.post("/session/{session_id}/reset", response_model=NewSessionResponse)
async def reset_session(session_id: str):
    sid = conv_manager.reset_session(session_id)
    return {"session_id": sid}


@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket):
    """
    Expected incoming JSON message shape (unchanged from Assignment 1):
      {"session_id": "<uuid or empty>", "message": "<user text>"}

    Outgoing JSON message shapes (unchanged types; "done" gains an
    optional "sources" field):
      {"type": "session", "session_id": "<uuid>"}          -- sent once, on connect/first turn
      {"type": "token", "content": "<partial text>"}        -- streamed repeatedly
      {"type": "done", "content": "<full text>",
       "ttft": <float>, "tokens_per_sec": <float>,
       "sources": [{"title": "...", "source": "..."}, ...]} -- "sources" present only when
                                                                 retrieval was used this turn
      {"type": "error", "content": "<message>"}             -- malformed input or model failure
    """
    await websocket.accept()
    session_id = None

    try:
        while True:
            raw = await websocket.receive_text()

            # --- robust parsing: never let a bad payload crash the connection ---
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json(
                    {"type": "error", "content": "Malformed request: expected valid JSON."}
                )
                continue

            if not isinstance(data, dict) or "message" not in data:
                await websocket.send_json(
                    {"type": "error", "content": "Malformed request: missing 'message' field."}
                )
                continue

            user_message = str(data.get("message", "")).strip()
            incoming_sid = data.get("session_id") or session_id

            if not user_message:
                await websocket.send_json(
                    {"type": "error", "content": "Empty message ignored - please type something."}
                )
                continue

            session_id = conv_manager.ensure_session(incoming_sid)
            if session_id != incoming_sid:
                await websocket.send_json({"type": "session", "session_id": session_id})

            conv_manager.add_user_message(session_id, user_message)

            # --- retrieval (Assignment 2, Phases II-IV) --------------------
            # retrieve_safe() never raises: a failure, timeout or missing
            # index yields an empty list, so the turn proceeds with the
            # static knowledge base only (Assignment 1 behaviour).
            retrieved_chunks = []
            if RAG_ENABLED:
                query = build_retrieval_query(
                    user_message, conv_manager.previous_user_message(session_id)
                )
                retrieved_chunks = await retrieve_safe(query)
                # A rewritten (follow-up) query keeps the document discussed in the
                # previous turn, so "how many copies of it?" still sees that book.
                if query != user_message:
                    sticky_src = conv_manager.get_last_source(session_id)
                    if sticky_src:
                        retrieved_chunks = merge_followup(chunks_for_source(sticky_src), retrieved_chunks)
                if retrieved_chunks:
                    conv_manager.set_last_source(session_id, retrieved_chunks[0].source)
            injected_chunks = select_for_context(retrieved_chunks)  # exactly what the model will see
            context_str = format_context(retrieved_chunks)

            prompt_messages, turns_used = conv_manager.build_prompt_messages(
                session_id, retrieved_context=context_str
            )
            if turns_used < MAX_TURNS and conv_manager.turn_count(session_id) > turns_used + 1:
                logger.info("context budget: history trimmed to %d turn pair(s)", turns_used)

            assistant_text = ""
            try:
                async for event in stream_chat(prompt_messages):
                    if event["type"] == "token":
                        assistant_text += event["content"]
                        await websocket.send_json(event)
                    elif event["type"] == "done":
                        if injected_chunks:
                            # cite only what was injected; de-duplicate by source document, preserve relevance order
                            seen = set()
                            sources = []
                            for c in injected_chunks:
                                if c.source not in seen:
                                    seen.add(c.source)
                                    sources.append({"title": c.title, "source": c.source})
                            event = {**event, "sources": sources}
                        await websocket.send_json(event)
                        conv_manager.add_assistant_message(session_id, event["content"])
                    elif event["type"] == "error":
                        await websocket.send_json(event)
            except Exception as e:  # noqa: BLE001 - keep the socket alive on model errors
                await websocket.send_json(
                    {"type": "error", "content": f"Model error while streaming: {e}"}
                )

    except WebSocketDisconnect:
        # Client closed the tab / lost connection mid-stream - nothing to crash.
        pass
    except Exception as e:  # noqa: BLE001 - last-resort guard so the server never dies
        try:
            await websocket.send_json({"type": "error", "content": f"Server error: {e}"})
        except Exception:
            pass
