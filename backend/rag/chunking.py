"""
chunking.py
-----------
Chunking strategy (documented here and in the README):

Our corpus is not one giant document that needs arbitrary splitting - it's
already ~70 small, single-topic files (one book per file, one policy topic
per file), each written to be focused and short. Blind fixed-size chunking
would ignore that structure and risk cutting a sentence (or a book's
availability line) in half for no benefit.

Instead we chunk *document-aware*:
  1. Split each file into paragraphs (blank-line separated).
  2. Greedily pack consecutive paragraphs into a chunk until adding the
     next paragraph would exceed MAX_WORDS.
  3. If a single paragraph alone exceeds MAX_WORDS (rare in this corpus),
     it is split on sentence boundaries as a fallback.
  4. Consecutive chunks from the same long document carry OVERLAP_WORDS
     of trailing context forward, so a fact near a chunk boundary isn't
     orphaned from its surrounding context.

In practice, most of our book files (short, ~120-220 words) end up as a
single chunk; our longer policy files (e.g. overdue_fines_and_lost_items,
borrowing_and_renewals) split into 2 chunks along their natural section
breaks. This keeps each chunk topically coherent, which matters more for
retrieval quality here than hitting an exact token count.
"""
from dataclasses import dataclass

MAX_WORDS = 110
OVERLAP_WORDS = 25
MIN_CHUNK_WORDS = 20  # chunks smaller than this (usually an overlap-only
                      # fragment) get merged into a neighbor rather than
                      # kept standalone, since they carry little retrieval signal


@dataclass
class Chunk:
    text: str
    source: str       # relative file path, e.g. "books/clean_code.md"
    title: str        # human-readable title, e.g. "Clean Code" or "Overdue Fines and Lost Items"
    chunk_index: int  # 0-based index within the source document


def _split_paragraphs(text: str) -> list[str]:
    paras = [p.strip() for p in text.split("\n\n")]
    return [p for p in paras if p]


def _extract_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def _word_count(s: str) -> int:
    return len(s.split())


def chunk_document(text: str, source: str) -> list[Chunk]:
    fallback_title = source.rsplit("/", 1)[-1].replace(".md", "").replace("_", " ").title()
    title = _extract_title(text, fallback_title)
    paragraphs = _split_paragraphs(text)

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for para in paragraphs:
        pw = _word_count(para)
        if pw > MAX_WORDS:
            # Rare fallback: a single paragraph longer than the budget on
            # its own - split it on sentence boundaries instead of words,
            # so we never cut mid-sentence.
            if current:
                chunks.append("\n\n".join(current))
                current, current_words = [], 0
            sentences = [s.strip() for s in para.replace("\n", " ").split(". ") if s.strip()]
            buf, buf_words = [], 0
            for sent in sentences:
                sw = _word_count(sent)
                if buf_words + sw > MAX_WORDS and buf:
                    chunks.append(". ".join(buf) + ".")
                    buf, buf_words = [], 0
                buf.append(sent)
                buf_words += sw
            if buf:
                chunks.append(". ".join(buf) + ".")
            continue

        if current_words + pw > MAX_WORDS and current:
            chunks.append("\n\n".join(current))
            # carry overlap forward from the end of the just-closed chunk
            overlap_text = " ".join(" ".join(current).split()[-OVERLAP_WORDS:])
            current = [overlap_text] if overlap_text else []
            current_words = _word_count(overlap_text) if overlap_text else 0

        current.append(para)
        current_words += pw

    if current:
        chunks.append("\n\n".join(current))

    # Post-process: merge any too-small chunk (typically a leftover
    # overlap fragment or an isolated heading) into a neighbor, so every
    # final chunk carries enough text to be useful for retrieval. Walk
    # left-to-right by index; a small chunk merges into the previous
    # output chunk if there is one, otherwise it is prepended onto the
    # next raw chunk (handled by just carrying it forward as `pending`).
    merged: list[str] = []
    pending = ""  # text merged forward, waiting to be prepended to the next chunk
    for i, c in enumerate(chunks):
        text = (pending + "\n\n" + c) if pending else c
        pending = ""
        if _word_count(text) < MIN_CHUNK_WORDS and len(chunks) > 1:
            if merged:
                merged[-1] = merged[-1] + "\n\n" + text
            else:
                pending = text  # carry forward to merge with the next chunk
        else:
            merged.append(text)

    if pending:
        # the whole document was smaller than MIN_CHUNK_WORDS in total
        if merged:
            merged[-1] = merged[-1] + "\n\n" + pending
        else:
            merged.append(pending)

    return [
        Chunk(text=c, source=source, title=title, chunk_index=i)
        for i, c in enumerate(merged)
    ]
