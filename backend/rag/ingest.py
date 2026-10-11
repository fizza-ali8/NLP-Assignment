"""
ingest.py
---------
Offline indexing pipeline (Assignment 2, Phase I).

Run this script whenever the document corpus changes:

    python -m rag.ingest

It walks `corpus/`, chunks every .md/.txt file (chunking.py), embeds each
chunk with the local CPU embedding model (embedder.py), and writes the
result to `rag/index/` as:
  - embeddings.npy   - a (N, 384) float32 numpy array, one row per chunk
  - chunks.json       - the chunk text + metadata (source, title, index),
                         in the same row order as embeddings.npy

Re-runnable / incremental (Assignment 2 requires updating the document set
without starting from scratch): each chunk is identified by a SHA-256 hash of
its text. On every run the script loads the previous index, reuses the stored
vector for every chunk whose hash is unchanged, and embeds ONLY chunks that
are new or edited. Chunks from deleted or edited documents simply drop out
because the index is re-assembled from the current corpus. Pass --full to
ignore the old index and re-embed everything (e.g. after changing the
embedding model).
"""

import argparse
import hashlib
import json
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.chunking import chunk_document
from rag.embedder import embed_texts

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "..", "corpus")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "index")


def load_corpus() -> list[tuple[str, str]]:
    """Returns [(relative_path, text), ...] for every .md/.txt file in corpus/."""
    files = sorted(
        glob.glob(os.path.join(CORPUS_DIR, "**", "*.md"), recursive=True)
        + glob.glob(os.path.join(CORPUS_DIR, "**", "*.txt"), recursive=True)
    )
    out = []
    for f in files:
        rel = os.path.relpath(f, CORPUS_DIR).replace(os.sep, "/")
        with open(f, "r", encoding="utf-8") as fh:
            out.append((rel, fh.read()))
    return out


def embed_text_for(title: str, text: str) -> str:
    """
    Text actually embedded for a chunk: the document title prepended to the
    chunk text. Continuation chunks of a long document (and policy chunks
    that start mid-section) otherwise lose the topic named in the title.
    """
    return f"{title}. {text}"


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_old_vectors() -> dict:
    """Map chunk-text hash -> normalized vector from the previous index, if any."""
    emb_path = os.path.join(INDEX_DIR, "embeddings.npy")
    meta_path = os.path.join(INDEX_DIR, "chunks.json")
    if not (os.path.exists(emb_path) and os.path.exists(meta_path)):
        return {}
    try:
        arr = np.load(emb_path)
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        if len(meta) != arr.shape[0]:
            return {}
        return {_hash(embed_text_for(m["title"], m["text"])): arr[i] for i, m in enumerate(meta)}
    except Exception:
        return {}


def build_index(full: bool = False):
    t0 = time.time()
    print(f"Loading corpus from {CORPUS_DIR} ...")
    docs = load_corpus()
    if not docs:
        print("No documents found in corpus/. Add .md or .txt files and re-run.")
        return
    print(f"  {len(docs)} source documents found.")

    all_chunks = []
    for rel_path, text in docs:
        all_chunks.extend(chunk_document(text, rel_path))
    print(f"  {len(all_chunks)} chunks after document-aware chunking.")

    old = {} if full else _load_old_vectors()
    embed_texts_list = [embed_text_for(c.title, c.text) for c in all_chunks]
    hashes = [_hash(t) for t in embed_texts_list]
    todo = [i for i, h in enumerate(hashes) if h not in old]
    print(f"  Reusing {len(all_chunks) - len(todo)} unchanged chunk vectors, "
          f"embedding {len(todo)} new/changed chunks.")

    new_vecs = {}
    if todo:
        print("Embedding (first run downloads the model, then it's cached) ...")
        t_embed = time.time()
        vecs = embed_texts([embed_texts_list[i] for i in todo])
        embed_time = time.time() - t_embed
        print(f"  Embedded {len(todo)} chunks in {embed_time:.2f}s.")
        for i, v in zip(todo, vecs):
            arr_v = np.array(v, dtype=np.float32)
            n = np.linalg.norm(arr_v)
            new_vecs[hashes[i]] = arr_v / (n if n else 1.0)

    arr = np.stack([old.get(h, new_vecs.get(h)) for h in hashes]).astype(np.float32)

    os.makedirs(INDEX_DIR, exist_ok=True)
    np.save(os.path.join(INDEX_DIR, "embeddings.npy"), arr)
    metadata = [
        {"text": c.text, "source": c.source, "title": c.title, "chunk_index": c.chunk_index}
        for c in all_chunks
    ]
    with open(os.path.join(INDEX_DIR, "chunks.json"), "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    print(f"Index written to {INDEX_DIR}/ "
          f"(embeddings.npy: {arr.shape}, chunks.json: {len(metadata)} entries)")
    print(f"Total pipeline time: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="re-embed everything")
    build_index(full=ap.parse_args().full)
