"""Voice frames and prompt-based tool calls, on the real WebSocket."""

import base64
import json
import struct
import unittest
from unittest import mock

import llm_engine
from tests.test_ws_integration import Base


def _until_done(ws):
    events = []
    while True:
        ev = json.loads(ws.recv(timeout=30))
        events.append(ev)
        if ev["type"] in ("done", "error"):
            return events


def _wav() -> bytes:
    import io
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 160)
    return buf.getvalue()


class VoiceAndTools(Base):
    def test_typed_chat_still_has_no_audio_events(self):
        with self.new_ws() as ws:
            events = _until_done_send(ws, {"session_id": "", "message": "What is the overdue fine per day?"})
        self.assertEqual(events[0]["type"], "session")
        self.assertTrue(any(e["type"] == "token" for e in events))
        self.assertEqual(events[-1]["type"], "done")
        self.assertFalse(any(e["type"] in ("audio", "transcript", "audio_done") for e in events))
        self.assertNotIn("timings", events[-1])

    def test_spoken_fine_uses_the_same_turn_and_is_spoken_back(self):
        pcm = struct.pack("<" + "h" * 1600, *([8000] * 1600))
        with mock.patch("voice.asr.transcribe_pcm", return_value="How much is the fine for 4 days overdue?"), \
             mock.patch("voice.tts.synthesize", return_value=_wav()):
            with self.new_ws() as ws:
                ws.send(json.dumps({"type": "audio_start", "session_id": ""}))
                ws.send(json.dumps({
                    "type": "audio_chunk", "session_id": "", "sample_rate": 16000,
                    "audio": base64.b64encode(pcm).decode(),
                }))
                ws.send(json.dumps({"type": "audio_end", "session_id": ""}))
                events = _until_done(ws)
        kinds = [e["type"] for e in events]
        self.assertIn("transcript", kinds)
        final = [e for e in events if e["type"] == "transcript" and e.get("final")][-1]
        self.assertIn("4 days", final["content"])
        self.assertIn("speech_end", kinds)
        self.assertTrue(any(e["type"] == "token" for e in events))
        self.assertTrue(any(e["type"] == "audio" for e in events))
        self.assertEqual(events[-1]["type"], "done")
        self.assertIn("asr_s", events[-1]["timings"])
        sent = self.fake.chat_requests()[-1]["messages"][-1]["content"]
        self.assertIn("$1.00", sent)
        self.assertIn("4 days", sent)
        hist = self.main.conv_manager.get_full_history(events[0]["session_id"] if events[0]["type"] == "session" else "")
        # session event is sent on audio_start because the id was empty
        sid = next(e["session_id"] for e in events if e["type"] == "session")
        hist = self.main.conv_manager.get_full_history(sid)
        self.assertEqual(hist[0]["role"], "user")
        self.assertIn("4 days", hist[0]["content"])
        self.assertNotIn("TOOL RESULT", hist[0]["content"])

    def test_model_tool_call_is_hidden_and_the_result_is_what_gets_spoken(self):
        seen = []

        async def scripted(messages):
            seen.append(messages[-1]["content"])
            if "TOOL RESULT" in messages[-1]["content"]:
                for piece in ("The fine ", "is one dollar."):
                    yield {"type": "token", "content": piece}
                yield {"type": "done", "content": "The fine is one dollar.", "ttft": 0.01, "tokens_per_sec": 10}
                return
            line = 'TOOL_CALL: {"name":"calculate_fine","arguments":{"days":4,"items":1}}\n'
            yield {"type": "token", "content": line}
            yield {"type": "done", "content": line, "ttft": 0.01, "tokens_per_sec": 1}

        original = llm_engine.stream_chat
        llm_engine.stream_chat = scripted
        try:
            with self.new_ws() as ws:
                events = _until_done_send(ws, {"session_id": "", "message": "Please use your fine tool."})
        finally:
            llm_engine.stream_chat = original

        visible = "".join(e["content"] for e in events if e["type"] == "token")
        self.assertNotIn("TOOL_CALL", visible)
        self.assertEqual(events[-1]["content"], "The fine is one dollar.")
        self.assertTrue(any("$1.00" in text for text in seen))

    def test_audio_without_a_start_does_not_kill_the_socket(self):
        with self.new_ws() as ws:
            ws.send(json.dumps({"type": "audio_chunk", "audio": "aaaa"}))
            err = json.loads(ws.recv(timeout=10))
            self.assertEqual(err["type"], "error")
            events = _until_done_send(ws, {"session_id": "", "message": "hello"})
        self.assertEqual(events[-1]["type"], "done")


def _until_done_send(ws, payload):
    ws.send(json.dumps(payload))
    return _until_done(ws)


if __name__ == "__main__":
    unittest.main()
