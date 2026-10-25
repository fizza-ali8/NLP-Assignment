"""Split streamed tokens into sentences so TTS can start before the reply is finished."""

import re

# A sentence end is punctuation followed by whitespace, or a newline.
# A period inside $0.25 does not match, because a digit follows it.
_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")
_SAFETY = 240


class SentenceSplitter:
    def __init__(self):
        self.buf = ""

    def feed(self, piece: str) -> list[str]:
        self.buf += piece
        ready = []
        while True:
            match = _BOUNDARY.search(self.buf)
            if match is None:
                break
            sentence = self.buf[: match.end()].strip()
            self.buf = self.buf[match.end():]
            if len(sentence) >= 2:
                ready.append(sentence)
        if len(self.buf) >= _SAFETY:
            cut = max(self.buf.rfind(", "), self.buf.rfind(" "))
            if cut >= 40:
                sentence = self.buf[: cut + 1].strip()
                self.buf = self.buf[cut + 1:]
                if sentence:
                    ready.append(sentence)
        return ready

    def flush(self) -> list[str]:
        sentence = self.buf.strip()
        self.buf = ""
        return [sentence] if len(sentence) >= 2 else []
