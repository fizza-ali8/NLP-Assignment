"""Microphone buffer and end-of-speech detection for 16 kHz PCM audio."""

import numpy as np

SAMPLE_RATE = 16000
SPEECH_RMS = 0.02          # about -34 dBFS; room noise on a laptop sits below this
SILENCE_MS = 800
MAX_SECONDS = 30


class SpeechBuffer:
    def __init__(self):
        self.pcm = bytearray()
        self.heard = False
        self.silence_ms = 0.0
        self.closed = False
        self.sample_rate = SAMPLE_RATE

    def add(self, chunk: bytes, sample_rate: int = SAMPLE_RATE) -> str:
        """
        Append one PCM s16le chunk.
        Returns 'speech', 'silence', 'endpoint' (the person stopped), or 'closed'.
        """
        if self.closed:
            return "closed"
        self.sample_rate = sample_rate or SAMPLE_RATE
        if len(chunk) < 2:
            return "silence"
        usable = chunk[: len(chunk) - (len(chunk) % 2)]
        self.pcm.extend(usable)
        samples = np.frombuffer(usable, dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(samples * samples))) / 32768.0
        dur_ms = (len(samples) / self.sample_rate) * 1000.0
        if self.duration_s() >= MAX_SECONDS and self.heard:
            self.closed = True
            return "endpoint"
        if rms >= SPEECH_RMS:
            self.heard = True
            self.silence_ms = 0.0
            return "speech"
        if self.heard:
            self.silence_ms += dur_ms
            if self.silence_ms >= SILENCE_MS:
                self.closed = True
                return "endpoint"
        return "silence"

    def duration_s(self) -> float:
        return (len(self.pcm) // 2) / float(self.sample_rate or SAMPLE_RATE)

    def close(self) -> bool:
        """Caller asked to stop (button). True if there is something to transcribe."""
        if self.closed:
            return False
        self.closed = True
        return len(self.pcm) > 0
