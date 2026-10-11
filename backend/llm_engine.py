"""
llm_engine.py
-------------
Thin async wrapper around a locally running Ollama server. Ollama exposes
an OpenAI-ish HTTP API on http://localhost:11434 by default. We use the
/api/chat endpoint with "stream": true and yield tokens as they arrive,
so the FastAPI layer can forward them to the browser over WebSocket in
real time.

No cloud model APIs are used anywhere in this file - inference happens
entirely on the machine running Ollama.

NUM_CTX (Assignment 2 addition): our system prompt grew substantially
once the off-topic few-shot examples and RAG policy text were added
(~9,100-10,850 chars, roughly 2,300-2,700 tokens, measured directly - see
conversation_manager.py). Ollama's default context window for many models
is only 2048 tokens, which the system prompt alone could already exceed,
silently truncating the model's own instructions. We explicitly request
a larger context window via the "options.num_ctx" field on every request
so there's real headroom for the prompt *and* a full response - this is
the other half of the context-overflow strategy described in
conversation_manager.py (that file manages what we *send*; this setting
controls how much the model is actually able to *receive*).
"""

import json
import time
import httpx
from typing import AsyncGenerator, List, Dict

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5:3b"   # change to whatever CPU-friendly model you pulled
NUM_CTX = 8192               # see module docstring - sized for our system prompt + RAG context
KEEP_ALIVE = "30m"           # keep the model in RAM between messages (avoids ~1 min cold reloads)
REQUEST_TIMEOUT = 120.0


class LLMEngineError(Exception):
    """Raised when the local model backend fails or is unreachable."""
    pass


async def stream_chat(messages: List[Dict[str, str]]) -> AsyncGenerator[dict, None]:
    """
    Streams a chat completion from Ollama.

    Yields dicts of the shape:
      {"type": "token", "content": "<partial text>"}
      {"type": "done", "content": "<full text>", "ttft": float, "tokens_per_sec": float}
      {"type": "error", "content": "<message>"}
    """
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": True,
        "options": {"num_ctx": NUM_CTX},
        "keep_alive": KEEP_ALIVE,
    }

    full_text = ""
    token_count = 0
    start_time = time.perf_counter()
    first_token_time = None

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            async with client.stream("POST", OLLAMA_URL, json=payload) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    raise LLMEngineError(
                        f"Ollama returned status {response.status_code}: {body[:300]!r}"
                    )

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    piece = chunk.get("message", {}).get("content", "")
                    if piece:
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        full_text += piece
                        token_count += 1
                        yield {"type": "token", "content": piece}

                    if chunk.get("done"):
                        end_time = time.perf_counter()
                        ttft = (first_token_time - start_time) if first_token_time else 0.0
                        total_time = end_time - start_time
                        tps = token_count / total_time if total_time > 0 else 0.0
                        yield {
                            "type": "done",
                            "content": full_text,
                            "ttft": round(ttft, 3),
                            "tokens_per_sec": round(tps, 2),
                        }
                        return

    except httpx.ConnectError:
        yield {
            "type": "error",
            "content": (
                "Could not reach the local model server (Ollama). "
                "Is `ollama serve` running on localhost:11434?"
            ),
        }
    except LLMEngineError as e:
        yield {"type": "error", "content": str(e)}
    except Exception as e:  # noqa: BLE001 - surface any unexpected failure safely
        yield {"type": "error", "content": f"Unexpected model error: {e}"}


async def warm_up(system_prompt: str) -> float:
    """
    Loads the model into RAM and pre-processes the (static) system prompt so
    the first real patron message does not pay the cold-start cost. Called
    once, in the background, at server start. Never raises: if Ollama is not
    running yet, the first real request simply pays the cost instead.
    Returns seconds taken (0.0 on failure).
    """
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "hi"},
        ],
        "stream": False,
        "options": {"num_ctx": NUM_CTX, "num_predict": 1},
        "keep_alive": KEEP_ALIVE,
    }
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            r = await client.post(OLLAMA_URL, json=payload)
            r.raise_for_status()
        return time.perf_counter() - t0
    except Exception:  # noqa: BLE001 - warm-up is best effort
        return 0.0
