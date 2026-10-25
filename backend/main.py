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

Assignment 3 additions (alongside the text contract, not instead of it)
  GET  /voice/status        - is local ASR / TTS installed
  GET  /tools               - tool names and JSON schemas
  WS frames type=audio_start|audio_chunk|audio_end, and server events
  transcript / speech_end / audio / audio_done. A text-only client never
  sends these and never has to understand them.

The WebSocket message contract from Assignment 1 is unchanged: same message
types, same required fields. Optional extras: "sources" on done (Assignment 2),
and "timings" on done for a voice turn (Assignment 3). A client that ignores
unknown fields behaves exactly as in Assignment 1.

Per-turn flow:  user message (typed, or the final transcript) -> retrieve
-> tools if the utterance needs one -> conversation manager builds
[system][older turns][user + excerpts + tool result] -> stream tokens
-> for voice, speak each finished sentence while tokens are still arriving.

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

from conversation_manager import ConversationManager
from llm_engine import warm_up as llm_warm_up
from prompts import SYSTEM_PROMPT
from rag import retriever
from rag.retriever import retrieve_safe
from tools.specs import SCHEMAS
from turn import complete_turn
from voice import asr, tts
from voice.pipeline import AUDIO_TYPES, handle_voice_message

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


@app.get("/voice/status")
async def voice_status():
    return {"asr": asr.status(), "tts": tts.status()}


@app.get("/tools")
async def tool_schemas():
    return {"tools": SCHEMAS}


@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket):
    """
    Expected incoming JSON message shape (unchanged from Assignment 1):
      {"session_id": "<uuid or empty>", "message": "<user text>"}

    Outgoing JSON message shapes (unchanged types; "done" gains an
    optional "sources" field, and on a voice turn an optional "timings"):
      {"type": "session", "session_id": "<uuid>"}          -- sent once, on connect/first turn
      {"type": "token", "content": "<partial text>"}        -- streamed repeatedly
      {"type": "done", "content": "<full text>",
       "ttft": <float>, "tokens_per_sec": <float>,
       "sources": [{"title": "...", "source": "..."}, ...]} -- "sources" present only when
                                                                 retrieval was used this turn
      {"type": "error", "content": "<message>"}             -- malformed input or model failure

    Voice frames (ignored by a text-only client, because that client never
    sends them) use the same socket:
      -> {"type": "audio_start"|"audio_chunk"|"audio_end", "session_id", "user_id"?}
      <- {"type": "transcript", "content", "final"}
      <- {"type": "speech_end"}
      <- {"type": "audio", "seq", "mime", "audio"}   base64 WAV, in order
      <- {"type": "audio_done"}
    The final transcript then goes through the same turn as typed text.
    """
    await websocket.accept()
    session_id = None
    voice_state: dict = {}

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

            if isinstance(data, dict) and data.get("type") in AUDIO_TYPES:
                session_id = await handle_voice_message(
                    websocket, data, voice_state, conv_manager, session_id, rag_enabled=RAG_ENABLED,
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

            user_id = data.get("user_id") or None
            await complete_turn(
                websocket, conv_manager, session_id, user_message,
                voice=False, user_id=str(user_id) if user_id else None, rag_enabled=RAG_ENABLED,
                retrieve=retrieve_safe,
            )

    except WebSocketDisconnect:
        # Client closed the tab / lost connection mid-stream - nothing to crash.
        pass
    except Exception as e:  # noqa: BLE001 - last-resort guard so the server never dies
        try:
            await websocket.send_json({"type": "error", "content": f"Server error: {e}"})
        except Exception:
            pass
