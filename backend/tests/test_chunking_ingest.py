"""Chunking behaviour and the re-runnable / incremental indexing pipeline."""

import glob
import json
import os
import shutil
import tempfile
import unittest

import numpy as np

from tests import support
from rag import chunking
from rag.chunking import chunk_document, MAX_WORDS, MIN_CHUNK_WORDS, OVERLAP_WORDS

CORPUS = os.path.join(support.BACKEND_DIR, "corpus")


class Chunking(unittest.TestCase):
    def test_every_real_document_produces_sane_chunks(self):
        files = glob.glob(os.path.join(CORPUS, "**", "*.md"), recursive=True)
        total = 0
        for f in files:
            rel = os.path.relpath(f, CORPUS).replace(os.sep, "/")
            chunks = chunk_document(support.read_text(f), rel)
            self.assertGreaterEqual(len(chunks), 1, rel)
            for c in chunks:
                total += 1
                self.assertEqual(c.source, rel)
                self.assertTrue(c.title)
                self.assertLessEqual(len(c.text.split()), MAX_WORDS + OVERLAP_WORDS + MIN_CHUNK_WORDS + 5, rel)
                if len(chunks) > 1:
                    self.assertGreaterEqual(len(c.text.split()), MIN_CHUNK_WORDS, f"{rel} has a tiny fragment")
            self.assertEqual([c.chunk_index for c in chunks], list(range(len(chunks))))
        self.assertGreaterEqual(total, len(files))

    def test_long_document_splits_with_overlap(self):
        paras = [" ".join(f"w{p}_{i}" for i in range(60)) for p in range(6)]
        text = "# Long Doc\n\n" + "\n\n".join(paras)
        chunks = chunk_document(text, "policies/long_doc.md")
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0].title, "Long Doc")
        # the tail of chunk i reappears at the start of chunk i+1
        tail = chunks[0].text.split()[-5:]
        self.assertTrue(all(w in chunks[1].text for w in tail))

    def test_giant_paragraph_splits_on_sentences(self):
        sentence = "This sentence has exactly eight words in it. "
        text = "# Big\n\n" + sentence * 60
        chunks = chunk_document(text, "policies/big.md")
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertTrue(c.text.rstrip().endswith("."), "chunk cut mid-sentence")

    def test_title_fallback_and_empty_input(self):
        chunks = chunk_document("Just some text without a heading but enough words to keep. " * 3, "policies/my_policy.md")
        self.assertEqual(chunks[0].title, "My Policy")
        self.assertEqual(chunk_document("", "x.md"), [])


class IncrementalIngest(unittest.TestCase):
    def setUp(self):
        self.corpus = tempfile.mkdtemp(prefix="corpus_")
        os.makedirs(os.path.join(self.corpus, "policies"))
        self.docs = {}
        for i in range(5):
            self.write(f"policies/doc{i}.md",
                       f"# Doc {i}\n\n" + " ".join(f"topic{i}word{j}" for j in range(50)))
        self.calls = []

    def tearDown(self):
        shutil.rmtree(self.corpus, ignore_errors=True)

    def write(self, rel, text):
        with open(os.path.join(self.corpus, rel), "w", encoding="utf-8") as fh:
            fh.write(text)

    def counting_rag(self):
        rag = support.FakeRag(corpus_dir=self.corpus, build=False)
        return rag

    def run_ingest(self, rag, full=False):
        calls = []
        original = rag.ing.embed_texts

        def counting(texts):
            calls.append(len(texts))
            return original(texts)

        rag.ing.embed_texts = counting
        try:
            rag.rebuild(full=full)
        finally:
            rag.ing.embed_texts = original
        return sum(calls)

    def test_first_run_rerun_edit_delete_and_full(self):
        with self.counting_rag() as rag:
            self.assertEqual(self.run_ingest(rag), 5, "first run embeds everything")
            self.assertEqual(self.run_ingest(rag), 0, "unchanged re-run embeds nothing")

            self.write("policies/doc2.md", "# Doc 2\n\n" + " ".join(f"edited{j}" for j in range(50)))
            self.assertEqual(self.run_ingest(rag), 1, "editing one doc re-embeds only that doc")

            self.write("policies/doc5.md", "# Doc 5\n\n" + " ".join(f"brandnew{j}" for j in range(50)))
            self.assertEqual(self.run_ingest(rag), 1, "adding a doc embeds only the new doc")

            os.remove(os.path.join(self.corpus, "policies/doc0.md"))
            self.assertEqual(self.run_ingest(rag), 0, "deleting a doc embeds nothing")
            meta = support.load_json(os.path.join(rag.index_dir, "chunks.json"))
            self.assertNotIn("policies/doc0.md", {m["source"] for m in meta})
            self.assertEqual(len(meta), 5)

            self.assertEqual(self.run_ingest(rag, full=True), 5, "--full re-embeds everything")

    def test_index_files_are_consistent_and_normalized(self):
        with self.counting_rag() as rag:
            rag.rebuild()
            vecs = np.load(os.path.join(rag.index_dir, "embeddings.npy"))
            meta = support.load_json(os.path.join(rag.index_dir, "chunks.json"))
            self.assertEqual(vecs.shape, (len(meta), support.DIM))
            self.assertEqual(vecs.dtype, np.float32)
            np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)

    def test_corrupt_old_index_falls_back_to_full_rebuild(self):
        with self.counting_rag() as rag:
            self.run_ingest(rag)
            support.write_text(os.path.join(rag.index_dir, "embeddings.npy"), b"garbage", "wb")
            self.assertEqual(self.run_ingest(rag), 5)


class RealCorpusIndex(unittest.TestCase):
    def test_real_corpus_builds_expected_shape(self):
        with support.FakeRag() as rag:
            meta = support.load_json(os.path.join(rag.index_dir, "chunks.json"))
            sources = {m["source"] for m in meta}
            self.assertEqual(len(sources), len(glob.glob(os.path.join(CORPUS, "**", "*.md"), recursive=True)))
            self.assertGreaterEqual(len(meta), len(sources))


if __name__ == "__main__":
    unittest.main()
