"""
Local speech-to-text. faster-whisper on CPU, English, int8.

The model downloads once on first use (Hugging Face) and is then cached.
Import and load are lazy so text chat and the unit tests do not need it.
"""

import logging
import os
import threading

import numpy as np

logger = logging.getLogger("voice.asr")

MODEL_NAME = os.environ.get("ASR_MODEL", "tiny.en")
_model = None
_lock = threading.Lock()
_load_failed = None


def installed() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        return False


def _get_model():
    global _model, _load_failed
    if _model is not None:
        return _model
    if _load_failed:
        raise RuntimeError(_load_failed)
    from faster_whisper import WhisperModel

    logger.info("loading ASR model %s (first run downloads it)", MODEL_NAME)
    _model = WhisperModel(MODEL_NAME, device="cpu", compute_type="int8")
    return _model


def transcribe_pcm(pcm: bytes, sample_rate: int = 16000) -> str:
    """Blocking. One model, one transcription at a time, so two users share the CPU safely."""
    global _load_failed
    if not pcm:
        return ""
    try:
        model = _get_model()
    except ImportError as e:
        raise RuntimeError(
            "Local speech-to-text is not installed. From backend/: pip install -r requirements-voice.txt"
        ) from e
    except Exception as e:  # noqa: BLE001
        _load_failed = f"Could not load the speech model ({e})."
        raise RuntimeError(_load_failed) from e

    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    if sample_rate != 16000 and len(audio):
        # Linear resample. The browser already sends 16 kHz; this is a safety net.
        dest_len = max(1, int(len(audio) * 16000 / sample_rate))
        idx = (np.arange(dest_len) * (len(audio) / dest_len)).astype(np.int32)
        idx = np.clip(idx, 0, len(audio) - 1)
        audio = audio[idx]
    with _lock:
        segments, _info = model.transcribe(
            audio, language="en", vad_filter=True, beam_size=1, without_timestamps=True,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
    return text


def status() -> dict:
    return {"engine": "faster-whisper", "model": MODEL_NAME, "installed": installed(), "loaded": _model is not None}
