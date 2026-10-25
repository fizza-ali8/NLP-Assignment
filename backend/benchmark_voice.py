"""
Per-stage timings for a voice turn (Assignment 3, Phase V).

    python benchmark_voice.py

ASR and TTS print "unavailable" until the local models are installed
(pip install -r requirements-voice.txt && python -m voice.setup).
Tool time is the part that runs before the language model starts.
"""

import statistics
import time

from tools.orchestrator import detect, run_all


async def _tool_once(message: str) -> float:
    calls = detect(message, user_id="bench")
    if not calls:
        raise SystemExit(f"no tool detected for {message!r}")
    _text, _names, elapsed = await run_all(calls)
    return elapsed


def bench_tools(runs: int = 20) -> None:
    import asyncio
    samples = []
    # one warmup so the catalogue database is built before the clock starts
    asyncio.run(_tool_once("How many copies of Clean Code?"))
    for message in (
        "How much is the fine for 4 days overdue?",
        "How many copies of Clean Code?",
        "Which study rooms are free tomorrow?",
    ):
        for _ in range(runs):
            samples.append(asyncio.run(_tool_once(message)))
    ms = [s * 1000 for s in samples]
    print(f"Tool calls: n={len(ms)} avg {statistics.mean(ms):.2f} ms | max {max(ms):.2f} ms")


def bench_tts() -> None:
    from voice import tts
    if not tts.ready():
        print("TTS first-audio: unavailable (install piper-tts and run python -m voice.setup)")
        return
    tts.synthesize("warm up")
    samples = []
    for _ in range(5):
        t0 = time.perf_counter()
        wav = tts.synthesize("The overdue fine is one dollar.")
        samples.append(time.perf_counter() - t0)
        if not wav:
            print("TTS first-audio: synthesize returned nothing")
            return
    print(f"TTS first sentence: avg {statistics.mean(samples)*1000:.1f} ms | max {max(samples)*1000:.1f} ms ({len(wav)} bytes)")


def bench_asr() -> None:
    from voice import asr, tts
    if not asr.installed():
        print("ASR: unavailable (pip install faster-whisper)")
        return
    import io
    import wave
    import numpy as np
    pcm = None
    if tts.ready():
        wav = tts.synthesize("The library is open until eight.")
        if wav:
            with wave.open(io.BytesIO(wav), "rb") as handle:
                pcm = handle.readframes(handle.getnframes())
                rate = handle.getframerate()
            seconds = len(pcm) / 2 / rate
            print(f"ASR sample: Piper sentence, {seconds:.2f}s of audio")
    if pcm is None:
        pcm = np.zeros(16000, dtype=np.int16).tobytes()
        print("ASR sample: 1s of silence (Piper was not available)")
    asr.transcribe_pcm(pcm)  # load the model; not included in the average
    samples = []
    texts = []
    for _ in range(3):
        t0 = time.perf_counter()
        texts.append(asr.transcribe_pcm(pcm))
        samples.append(time.perf_counter() - t0)
    print(
        f"ASR warm: avg {statistics.mean(samples)*1000:.0f} ms | "
        f"max {max(samples)*1000:.0f} ms | transcript {texts[-1]!r}"
    )


if __name__ == "__main__":
    bench_tools()
    bench_tts()
    bench_asr()
