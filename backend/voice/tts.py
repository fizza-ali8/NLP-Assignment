"""
Local text-to-speech. Piper (en_US-lessac-medium) on CPU.

Download the voice once with `python -m voice.setup`. Synthesis runs in a
worker thread so the event loop can keep streaming text to other patrons.
If the voice is missing, synthesize() returns None and the chat stays text-only.
"""

import io
import logging
import os
import threading
import wave

logger = logging.getLogger("voice.tts")

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
ONNX_PATH = os.path.join(_DIR, "en_US-lessac-medium.onnx")
_voice = None
_lock = threading.Lock()
_warned = False


def ready() -> bool:
    if not os.path.exists(ONNX_PATH):
        return False
    try:
        import piper  # noqa: F401
        return True
    except ImportError:
        return False


def _load():
    global _voice, _warned
    if _voice is not None:
        return _voice
    if not os.path.exists(ONNX_PATH):
        if not _warned:
            logger.warning("Piper voice not found. Run: python -m voice.setup")
            _warned = True
        return None
    try:
        from piper import PiperVoice
    except ImportError:
        if not _warned:
            logger.warning("piper-tts is not installed. Run: pip install -r requirements-voice.txt")
            _warned = True
        return None
    logger.info("loading Piper voice %s", os.path.basename(ONNX_PATH))
    _voice = PiperVoice.load(ONNX_PATH)
    return _voice


def synthesize(text: str) -> bytes | None:
    """Return a WAV file's bytes, or None if TTS is unavailable. Never raises."""
    text = " ".join((text or "").split())
    if not text:
        return None
    try:
        voice = _load()
        if voice is None:
            return None
        buf = io.BytesIO()
        with _lock:
            with wave.open(buf, "wb") as wav_file:
                if hasattr(voice, "synthesize_wav"):
                    voice.synthesize_wav(text, wav_file)
                else:
                    voice.synthesize(text, wav_file)
        return buf.getvalue()
    except Exception as e:  # noqa: BLE001 - a TTS failure must not kill the chat turn
        logger.warning("TTS failed (%r); reply stays text-only", e)
        return None


def status() -> dict:
    return {
        "engine": "piper",
        "voice": "en_US-lessac-medium",
        "model_path": ONNX_PATH,
        "ready": ready(),
        "loaded": _voice is not None,
    }
