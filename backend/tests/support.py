"""
Shared helpers for the test-suite.

The real embedding model (all-MiniLM-L6-v2) needs a download, so the unit
and integration tests run against a small deterministic LEXICAL embedder
instead: it hashes words into a 384-dim signed bag-of-words vector, so two
texts that share words get a high cosine score and unrelated texts score ~0.
That is enough to exercise every piece of retrieval LOGIC (chunking ->
index -> top-k -> threshold -> cache -> failure handling -> prompt
assembly -> citations) offline and fast. It says nothing about the quality
of the real model's semantic matching - that is measured separately by
`python eval_retrieval.py`, which uses the real model on your machine.
"""

import hashlib
import os
import re
import shutil
import sys
import tempfile

import numpy as np

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

DIM = 4096  # large so random hash collisions between unrelated words are negligible
# Lowered from the production 0.35 because lexical cosine scores are lower
# than semantic-model scores for the same relevant pair.
TEST_MIN_SCORE = 0.12

_STOP = {
    "the", "a", "an", "is", "are", "of", "to", "and", "or", "for", "in", "on", "at", "it", "i",
    "do", "does", "you", "your", "me", "my", "can", "what", "how", "any", "have", "be", "with",
    "that", "this", "there", "s", "if", "as", "by", "from", "about", "so", "we", "our",
}


def _tokens(text: str) -> list[str]:
    out = []
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if w in _STOP:
            continue
        if len(w) > 3 and w.endswith("s"):
            w = w[:-1]
        out.append(w)
    return out


def fake_vec(text: str) -> list[float]:
    v = np.zeros(DIM, dtype=np.float32)
    for t in _tokens(text):
        h = int(hashlib.md5(t.encode()).hexdigest(), 16)
        v[h % DIM] += 1.0 if (h >> 40) & 1 else -1.0
        # a second slot makes the vector less collision-prone
        v[(h >> 8) % DIM] += 0.5
    return v.tolist()


def fake_embed_texts(texts):
    return [fake_vec(t) for t in texts]


def fake_embed_query(text):
    return fake_vec(text)


class FakeRag:
    """
    Context manager: builds a real index of the real corpus (or a custom
    corpus dir) in a temp folder with the fake embedder and points the
    retriever at it. Restores everything on exit.
    """

    def __init__(self, corpus_dir=None, min_score=TEST_MIN_SCORE, build=True):
        self.corpus_dir = corpus_dir
        self.min_score = min_score
        self.build = build
        self.index_dir = tempfile.mkdtemp(prefix="rag_index_")

    def __enter__(self):
        import rag.embedder as emb
        import rag.ingest as ing
        import rag.retriever as ret

        self._saved = (emb.embed_texts, emb.embed_query, ing.embed_texts, ing.INDEX_DIR,
                       ing.CORPUS_DIR, ret.INDEX_DIR, ret.MIN_SCORE, ret._index)
        emb.embed_texts, emb.embed_query = fake_embed_texts, fake_embed_query
        ing.embed_texts = fake_embed_texts
        ing.INDEX_DIR = self.index_dir
        if self.corpus_dir:
            ing.CORPUS_DIR = self.corpus_dir
        ret.INDEX_DIR = self.index_dir
        ret.MIN_SCORE = self.min_score
        ret._index = None
        ret._query_cache_search.cache_clear()
        self.emb, self.ing, self.ret = emb, ing, ret
        if self.build:
            self.rebuild()
        return self

    def rebuild(self, full=False):
        import io
        from contextlib import redirect_stdout

        with redirect_stdout(io.StringIO()):
            self.ing.build_index(full=full)
        self.ret._index = None
        self.ret._query_cache_search.cache_clear()

    def __exit__(self, *exc):
        (self.emb.embed_texts, self.emb.embed_query, self.ing.embed_texts, self.ing.INDEX_DIR,
         self.ing.CORPUS_DIR, self.ret.INDEX_DIR, self.ret.MIN_SCORE, self.ret._index) = self._saved
        self.ret._query_cache_search.cache_clear()
        shutil.rmtree(self.index_dir, ignore_errors=True)
        return False


def read_text(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def write_text(path, text, mode="w"):
    with open(path, mode) as fh:
        fh.write(text)


def load_json(path):
    import json
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
