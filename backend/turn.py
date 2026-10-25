"""
One chat turn, shared by typed text and the final speech transcript.

Retrieval, memory, tools, and token streaming all happen here. Voice adds
sentence-by-sentence TTS on the same tokens; it does not build a second prompt.
"""

import asyncio
import base64
import logging
import time

from fastapi import WebSocketDisconnect

import llm_engine
from conversation_manager import MAX_TURNS
from rag.retriever import (
    build_retrieval_query,
    chunks_for_source,
    format_context,
    merge_followup,
    retrieve_safe,
    select_for_context,
)
from tools.orchestrator import (
    StreamGate, allows_cross_conversation, execute, memory_block, parse_tool_call, detect, run_all,
)
from voice.sentences import SentenceSplitter

logger = logging.getLogger("turn")

ASR_TIMEOUT_S = 8.0


def _with_tool_result(messages: list, result_text: str) -> list:
    copied = [dict(m) for m in messages]
    copied[-1] = dict(copied[-1])
    copied[-1]["content"] = (
        copied[-1]["content"]
        + "\n\nTOOL RESULT (from a library tool; trust these figures):\n"
        + result_text
    )
    return copied


async def iter_visible_reply(messages: list, *, allow_tool: bool, user_id: str, fallback: str = "",
                             allow_saved_memory: bool = False):
    """
    Yields the same event dicts as llm_engine.stream_chat, except a TOOL_CALL
    line is executed and replaced by the follow-up answer.
    """
    gate = StreamGate()
    pending_call = None
    bad_call = False
    blocked_call = False

    async for event in llm_engine.stream_chat(messages):
        if event["type"] == "error":
            if pending_call is None and not bad_call and not blocked_call:
                yield event
            return
        if event["type"] == "token":
            if pending_call is not None or bad_call or blocked_call:
                continue
            kind, payload = gate.feed(event["content"])
            if kind == "hold":
                continue
            if kind == "token":
                yield {"type": "token", "content": payload}
                continue
            call = parse_tool_call(payload, user_id=user_id, allow_saved_memory=allow_saved_memory)
            if call is None:
                bad_call = True
                continue
            if not allow_tool:
                blocked_call = True
                continue
            pending_call = call
            continue
        if event["type"] == "done":
            if pending_call is None and not bad_call and not blocked_call:
                kind, payload = gate.finish()
                if kind == "token" and payload:
                    yield {"type": "token", "content": payload}
                elif kind == "tool":
                    call = parse_tool_call(payload, user_id=user_id, allow_saved_memory=allow_saved_memory)
                    if call is not None and allow_tool:
                        pending_call = call
                    elif call is not None:
                        blocked_call = True
                    else:
                        bad_call = True
                if pending_call is None and not bad_call and not blocked_call:
                    yield event
                    return
            break

    if pending_call is not None:
        result = await execute(pending_call)
        logger.info("model tool call %s", pending_call.name)
        follow = _with_tool_result(messages, result)
        async for ev in iter_visible_reply(
            follow, allow_tool=False, user_id=user_id, fallback=result, allow_saved_memory=False,
        ):
            yield ev
        return

    if bad_call or blocked_call:
        text = fallback or "I couldn't complete that tool request. Please rephrase it or ask at the front desk."
        if bad_call:
            text = "I couldn't read that tool request. Please rephrase it."
        yield {"type": "token", "content": text}
        yield {"type": "done", "content": text, "ttft": 0.0, "tokens_per_sec": 0.0}


async def _speak(websocket, queue: asyncio.Queue, timings: dict):
    from voice import tts

    seq = 0
    while True:
        sentence = await queue.get()
        if sentence is None:
            try:
                await websocket.send_json({"type": "audio_done"})
            except Exception:  # noqa: BLE001
                return
            return
        t0 = time.perf_counter()
        wav = await asyncio.to_thread(tts.synthesize, sentence)
        if wav:
            if "tts_first_s" not in timings:
                timings["tts_first_s"] = round(time.perf_counter() - t0, 3)
                anchor = timings.get("speech_end_at")
                if anchor is not None:
                    timings["end_of_speech_to_audio_s"] = round(time.perf_counter() - anchor, 3)
            try:
                await websocket.send_json({
                    "type": "audio",
                    "seq": seq,
                    "mime": "audio/wav",
                    "audio": base64.b64encode(wav).decode("ascii"),
                })
            except Exception:  # noqa: BLE001
                return
            seq += 1


async def complete_turn(websocket, conv_manager, session_id: str, user_message: str, *,
                        voice: bool, user_id: str | None, rag_enabled: bool,
                        timings: dict | None = None, retrieve=None):
    """Retrieval + tools + streamed reply. `timings` is filled in for voice turns."""
    retrieve_fn = retrieve or retrieve_safe
    timings = timings if timings is not None else {}
    conv_manager.add_user_message(session_id, user_message)

    memory = ""
    tool_text = ""
    tool_names: list[str] = []
    tool_s = 0.0
    share = allows_cross_conversation(user_message)
    if share:
        try:
            memory = await asyncio.to_thread(memory_block, user_id)
        except Exception as e:  # noqa: BLE001
            logger.warning("memory lookup failed: %r", e)
    note = conv_manager.conversation_note(session_id, user_message)
    if note:
        memory = (memory + "\n\n" + note).strip() if memory else note

    async def _tools():
        calls = detect(user_message, user_id or "")
        if not calls:
            return "", [], 0.0
        return await run_all(calls)

    tool_task = asyncio.create_task(_tools())

    retrieved_chunks = []
    retrieval_s = 0.0
    if rag_enabled:
        t0 = time.perf_counter()
        query = build_retrieval_query(user_message, conv_manager.previous_user_message(session_id))
        retrieved_chunks = await retrieve_fn(query)
        if query != user_message:
            sticky_src = conv_manager.get_last_source(session_id)
            if sticky_src:
                retrieved_chunks = merge_followup(chunks_for_source(sticky_src), retrieved_chunks)
        if retrieved_chunks:
            conv_manager.set_last_source(session_id, retrieved_chunks[0].source)
        retrieval_s = time.perf_counter() - t0

    try:
        tool_text, tool_names, tool_s = await tool_task
    except Exception as e:  # noqa: BLE001
        logger.warning("tool detection failed: %r", e)
        tool_text, tool_names, tool_s = "", [], 0.0

    blocks = [b for b in (memory, tool_text) if b]
    tool_context = ""
    if blocks:
        labelled = []
        if memory:
            labelled.append(memory)
        if tool_text:
            labelled.append("TOOL RESULT (from a library tool; trust these figures):\n" + tool_text)
        tool_context = "\n\n".join(labelled)

    injected_chunks = select_for_context(retrieved_chunks)
    context_str = format_context(retrieved_chunks)
    prompt_messages, turns_used = conv_manager.build_prompt_messages(
        session_id, retrieved_context=context_str, tool_context=tool_context
    )
    if turns_used < MAX_TURNS and conv_manager.turn_count(session_id) > turns_used + 1:
        logger.info("context budget: history trimmed to %d turn pair(s)", turns_used)

    domain_used = any(name != "patron_memory" for name in tool_names)
    splitter = SentenceSplitter() if voice else None
    audio_q: asyncio.Queue | None = asyncio.Queue() if voice else None
    speaker = asyncio.create_task(_speak(websocket, audio_q, timings)) if voice else None

    def enqueue(text: str):
        if splitter is None:
            return
        for sentence in splitter.feed(text):
            audio_q.put_nowait(sentence)

    try:
        async for event in iter_visible_reply(
            prompt_messages,
            allow_tool=not domain_used,
            user_id=user_id or "",
            fallback=tool_text,
            allow_saved_memory=share,
        ):
            if event["type"] == "token":
                enqueue(event["content"])
                await websocket.send_json(event)
            elif event["type"] == "done":
                if splitter is not None:
                    for sentence in splitter.flush():
                        audio_q.put_nowait(sentence)
                    await audio_q.put(None)
                    await speaker
                if injected_chunks:
                    seen = set()
                    sources = []
                    for c in injected_chunks:
                        if c.source not in seen:
                            seen.add(c.source)
                            sources.append({"title": c.title, "source": c.source})
                    event = {**event, "sources": sources}
                if voice:
                    event = {
                        **event,
                        "timings": {
                            "asr_s": timings.get("asr_s"),
                            "retrieval_s": round(retrieval_s, 3),
                            "tool_s": round(tool_s, 3),
                            "llm_ttft_s": event.get("ttft"),
                            "tts_first_s": timings.get("tts_first_s"),
                            "end_of_speech_to_audio_s": timings.get("end_of_speech_to_audio_s"),
                            "tools": tool_names,
                        },
                    }
                    logger.info("voice timings %s", event["timings"])
                await websocket.send_json(event)
                conv_manager.add_assistant_message(session_id, event["content"])
            elif event["type"] == "error":
                await websocket.send_json(event)
    except WebSocketDisconnect:
        raise
    except Exception as e:  # noqa: BLE001
        try:
            await websocket.send_json({"type": "error", "content": f"Model error while streaming: {e}"})
        except Exception:  # noqa: BLE001
            pass
    finally:
        if speaker is not None and not speaker.done():
            audio_q.put_nowait(None)
            try:
                await speaker
            except Exception:  # noqa: BLE001
                speaker.cancel()
