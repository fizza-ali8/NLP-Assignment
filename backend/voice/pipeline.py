"""WebSocket audio frames: buffer PCM, detect the end of speech, transcribe, then the normal turn."""

import asyncio
import base64
import logging
import time

from voice import asr
from voice.endpoint import SpeechBuffer
from turn import ASR_TIMEOUT_S, complete_turn

logger = logging.getLogger("voice.pipeline")

AUDIO_TYPES = {"audio_start", "audio_chunk", "audio_end"}
_MAX_CHUNK = 1_000_000


async def handle_voice_message(websocket, data: dict, state: dict, conv_manager, session_id, *, rag_enabled: bool):
    """
    Handle one audio frame. Returns the session id (it may have been created here).
    Text frames never come through this function.
    """
    kind = data.get("type")
    incoming = data.get("session_id") or session_id
    user_id = data.get("user_id") or state.get("user_id")
    if user_id:
        state["user_id"] = str(user_id)

    if kind == "audio_start":
        session_id = conv_manager.ensure_session(incoming)
        if session_id != incoming:
            await websocket.send_json({"type": "session", "session_id": session_id})
        state["session_id"] = session_id
        state["buffer"] = SpeechBuffer()
        state["final_sent"] = False
        state["partial_launched"] = False
        await websocket.send_json({"type": "transcript", "content": "", "final": False, "status": "listening"})
        return session_id

    buf: SpeechBuffer | None = state.get("buffer")
    if buf is None:
        await websocket.send_json({
            "type": "error",
            "content": "Start speaking with audio_start before sending audio.",
        })
        return session_id
    if buf.closed:
        return session_id

    session_id = state.get("session_id") or conv_manager.ensure_session(incoming)

    if kind == "audio_chunk":
        raw_b64 = data.get("audio") or ""
        if not isinstance(raw_b64, str) or len(raw_b64) > _MAX_CHUNK:
            await websocket.send_json({"type": "error", "content": "Audio chunk was empty or too large."})
            return session_id
        try:
            pcm = base64.b64decode(raw_b64, validate=False)
        except Exception:  # noqa: BLE001
            await websocket.send_json({"type": "error", "content": "Audio chunk was not valid base64."})
            return session_id
        try:
            rate = int(data.get("sample_rate") or 16000)
        except (TypeError, ValueError):
            rate = 16000
        signal = buf.add(pcm, rate)
        if (
            not state.get("partial_launched")
            and buf.heard
            and buf.duration_s() >= 4.0
            and signal != "endpoint"
        ):
            state["partial_launched"] = True
            snapshot = bytes(buf.pcm)
            asyncio.create_task(_send_partial(websocket, state, snapshot, rate))
        if signal == "endpoint":
            await _finalize(websocket, state, conv_manager, session_id, rag_enabled=rag_enabled, reason="silence")
        return session_id

    if kind == "audio_end":
        if buf.close():
            await _finalize(websocket, state, conv_manager, session_id, rag_enabled=rag_enabled, reason="client")
        elif not state.get("final_sent"):
            await websocket.send_json({
                "type": "error",
                "content": "I didn't catch that. Try again, or type your question.",
            })
            state["final_sent"] = True
        return session_id

    return session_id


async def _send_partial(websocket, state: dict, pcm: bytes, sample_rate: int):
    try:
        text = await asyncio.wait_for(asyncio.to_thread(asr.transcribe_pcm, pcm, sample_rate), timeout=ASR_TIMEOUT_S)
    except Exception as e:  # noqa: BLE001 - a late partial must not break the socket
        logger.info("partial transcript skipped: %r", e)
        return
    if state.get("final_sent") or not text:
        return
    try:
        await websocket.send_json({"type": "transcript", "content": text, "final": False})
    except Exception:  # noqa: BLE001
        return


async def _finalize(websocket, state, conv_manager, session_id, *, rag_enabled: bool, reason: str):
    if state.get("final_sent"):
        return
    state["final_sent"] = True
    buf: SpeechBuffer = state["buffer"]
    speech_end_at = time.perf_counter()
    try:
        await websocket.send_json({"type": "speech_end"})
    except Exception:  # noqa: BLE001
        return
    try:
        t0 = time.perf_counter()
        text = await asyncio.wait_for(
            asyncio.to_thread(asr.transcribe_pcm, bytes(buf.pcm), buf.sample_rate),
            timeout=ASR_TIMEOUT_S,
        )
        asr_s = round(time.perf_counter() - t0, 3)
    except asyncio.TimeoutError:
        await websocket.send_json({
            "type": "error",
            "content": "Speech recognition took too long. Please try a shorter question, or type it.",
        })
        return
    except Exception as e:  # noqa: BLE001
        logger.warning("ASR failed (%s): %r", reason, e)
        await websocket.send_json({"type": "error", "content": str(e)})
        return

    text = " ".join(text.split())
    if not text:
        await websocket.send_json({
            "type": "error",
            "content": "I didn't catch that. Try again, or type your question.",
        })
        return

    await websocket.send_json({"type": "transcript", "content": text, "final": True})
    import main as app_main

    timings = {"asr_s": asr_s, "speech_end_at": speech_end_at}
    await complete_turn(
        websocket, conv_manager, session_id, text,
        voice=True, user_id=state.get("user_id"), rag_enabled=rag_enabled, timings=timings,
        retrieve=app_main.retrieve_safe,
    )
