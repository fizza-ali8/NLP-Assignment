# Riverbend Library Assistant (Assignments 1–3)

A fully **local, CPU-only** chatbot for a community library. Assignment 1 gave
it a local LLM (Ollama `qwen2.5:3b`), a FastAPI WebSocket backend and a web
frontend. Assignment 2 grounds answers in a 69-document library corpus.
Assignment 3 adds a local voice pipeline (speech-to-text and text-to-speech)
and tools the model can call: overdue fines, study-room bookings, catalogue
lookup, and a patron memory record. Typed chat still uses the original
WebSocket messages.

> **Numbers policy.** Every number in this README is either a fixed design
> parameter or was measured on the machine listed in §7 with the command
> shown next to it. Nothing was estimated. Voice-model numbers are filled in
> by `python benchmark_voice.py` after the ASR and TTS models are installed.

---

## 1. Use case

Front-desk assistant for *Riverbend Community Library*: hours, membership tiers,
borrowing/renewals, fines, holds, rooms, printing, and whether a given book is
in the catalogue. It refuses off-topic requests and resists jailbreaks.

## 2. Architecture

```mermaid
flowchart LR
  FE[frontend/index.html<br/>mic + speaker] <-- WebSocket /ws/chat --> M[main.py FastAPI]
  M --> V[voice pipeline<br/>faster-whisper + Piper]
  M --> Q[build_retrieval_query]
  Q --> R[retriever.retrieve_safe<br/>thread pool + 2s timeout + LRU cache]
  R --> E[fastembed MiniLM<br/>ONNX, CPU]
  R --> I[(rag/index)]
  M --> T[tool orchestrator<br/>2s timeout]
  T --> T1[calculate_fine]
  T --> T2[study_room calendar]
  T --> T3[search_catalogue]
  T --> T4[patron_memory CRM]
  R --> C[conversation_manager]
  T --> C
  C --> L[llm_engine -> Ollama qwen2.5:3b]
  L --> M
  V --> M
  CORPUS[corpus 69 documents] -- python -m rag.ingest --> I
```

Typed or spoken, one turn does the same work: retrieve, run a tool if the
utterance needs one, build `[system][recent history][user + excerpts + tool
result]`, stream tokens. A spoken reply is also synthesised sentence by
sentence while those tokens are still arriving. The text message shapes from
Assignment 1 are unchanged.

## 3. WebSocket contract (text path unchanged)

Client → server `{"session_id": "...", "message": "..."}`; server → client
`{"type":"token"|"done"|"error", ...}` with the same required fields as
Assignment 1. Optional extras, which an old client can ignore:

- `sources` on `done` when retrieval injected a document (Assignment 2)
- `user_id` on the client message (Assignment 3, stable browser id for the CRM)
- `timings` on `done` for a voice turn only

Voice uses **extra** frame types on the same socket, never instead of the text ones:

- client → `audio_start`, `audio_chunk` (base64 PCM s16le, 16 kHz, mono), `audio_end`
- server → `transcript` (`final` false while listening or on a partial, true for the utterance), `speech_end`, `audio` (base64 WAV, `seq` in order), `audio_done`

The final transcript is stored and answered by the same conversation manager as typed text. Extra endpoints: `GET /rag/status`, `POST /rag/reload`, `GET /voice/status`, `GET /tools`.

## 4. Design choices and why

### 4.1 Embedding model: `all-MiniLM-L6-v2` via fastembed
384-dim, ~90 MB, runs on CPU through ONNX Runtime. Chosen because it is small
and fast enough for sub-second query embedding, good at short English
passages, and fastembed ships prebuilt wheels (PyTorch / sentence-transformers
repeatedly failed to build on Python 3.14 in Assignment 1). Alternative
considered: `bge-small-en-v1.5` (slightly better quality, similar size);
MiniLM was kept for the smaller footprint.

### 4.2 Vector store: flat numpy index (deliberate substitution)
The brief lists "Chroma, FAISS, etc". A flat index is *exactly* what FAISS
`IndexFlatIP` does: exact dot-product over L2-normalised vectors (= cosine).
With ~77 vectors this takes well under a millisecond, so an ANN library would
add install risk (native wheels on Python 3.14) and no speed benefit. The
whole system only touches `retrieve(query, k)`, so swapping in FAISS/Chroma is
a one-file change. Persistence: `embeddings.npy` + `chunks.json`.

### 4.3 Corpus (69 documents, within 50–100)
50 book entries + 19 policy documents, all original text (no copied blurbs).
Facts are kept consistent across documents (loan periods, pickup windows,
Premium benefits) and a test ties the static knowledge base to the corpus.

### 4.4 Chunking
Document-aware paragraph packing: `MAX_WORDS=110`, `OVERLAP_WORDS=25`,
`MIN_CHUNK_WORDS=20`; oversized paragraphs fall back to sentence splitting and
tiny fragments merge into neighbours. Documents are short and coherent, so
fixed-size splitting would cut facts in half. Result: **77 chunks** from 69
documents. The document title is prepended to the text that gets embedded, so
"Atomic Habits" in a query matches that book's chunk.

### 4.5 Top-k and threshold
`TOP_K = 4` (brief requires ≥ 3: enough to cover multi-fact questions such as
"Premium vs Standard", small enough to protect the context window).
`MIN_SCORE = 0.30` makes "no relevant match" a first-class outcome rather than
forcing weak chunks in. It was chosen from a measured sweep on the real model
(`python eval_retrieval.py --sweep`, 51 in-domain + 12 must-match-nothing
queries, `eval/retrieval_queries.json`):

| MIN_SCORE | in-domain hit@4 | no-match correctly empty |
|---|---|---|
| 0.20 | 45/48 | 7/12 |
| **0.30 (chosen)** | **45/48** | **9/12** |
| 0.35 (initial guess) | 42/48 | 9/12 |
| 0.45 | 34/48 | 11/12 |
| 0.50 | 31/48 | 12/12 |

Final setting (0.30), 51 in-domain queries: hit@1 88% (45/51), hit@4 94% (48/51),
MRR 0.91, no-match correct 9/12 (the threshold sweep below used the first 48 queries). At the initial 0.35: hit@1 79%, hit@4 88%, MRR 0.83.
0.30 dominates 0.35 on this set (more hits, same no-match score). Caveat: the threshold was tuned on the same
queries it is evaluated on, so treat the numbers as indicative.

**Relative margin (evaluated, not adopted).** An optional `REL_MARGIN` drops
chunks scoring more than M below the best. On the same 48 queries, M = 0.10
keeps hit@4 at 45/48 (MRR 0.878) and cuts the average documents from 3.17 to
1.98, but M <= 0.07 loses hits. It is left **off** (`0.0`) because the brief
requires k >= 3 and trimming below three candidates would contradict that.

**Citations match the prompt.** Retrieval returns k = 4 chunks, but the excerpt
budget (`MAX_CONTEXT_CHARS = 1000`) may fit fewer. Sources shown to the patron
are built only from the chunks that were actually injected
(`select_for_context`), so the UI never cites a document the model did not see
(a test enforces this).

**Follow-ups.** `build_retrieval_query` rewrites a short follow-up ("how many
copies of it?") by prepending the previous patron message. As a safeguard, the
session also remembers the previous turn's top document and puts its first chunks
ahead of the new matches. This safeguard is tested (even when retrieval returns
nothing for the rewritten query), but it has not been shown to be necessary on the
real model: an earlier transcript run appeared to show follow-ups failing, but that
was a bug in `collect_transcripts.py` (it did not reuse the session id), not in
the backend.

**FAQ-style section.** `membership_tiers.md` has a "Premium vs Standard" summary
section because comparison questions ("what does Premium add?") were not
retrieving the membership document on the real model (book entries mentioning
"Premium" outranked it).

**Failed experiment (reverted).** The Premium/Standard misses seemed to come
from every book entry repeating "Standard/Premium membership" words. I removed
that wording from all 50 books and re-ran the eval: the two Premium misses were
fixed, but hit@1 fell 83% -> 75%, hit@4 94% -> 88% and MRR 0.878 -> 0.806
(generic "book" questions now retrieved book entries, and author/title queries
got worse). The change was reverted; numbers above are for the original corpus.

**What the evaluation showed (honest findings):**
- No single threshold separates everything: library-adjacent questions that
  the corpus does not cover ("Does the library sell coffee?" 0.46, "Can I rent a
  car?" 0.42, "3D printer?" 0.41) score *higher* than some correct hits
  (0.30-0.32). So the threshold is a coarse filter; the second line of defence
  is the prompt rule "say you don't have that information if the excerpts
  don't answer it".
- A stand-alone question containing a filler word like "there" was once wrongly treated as a follow-up and pulled in the previous topic's document ("Is there a book club?" after a fines question). Fixed by restricting follow-up detection to real pronouns, with a test.
- The 3 remaining misses at 0.30: "lost a library book" (lost_and_found outranks the fines/lost-items policy), "Premium member take out at once" and "reserve a checked-out book" (book entries outrank the policy because they share words like Premium/book).
- Ranking weaknesses: queries about *Premium* sometimes retrieve book entries
  (they contain "Premium membership" in the loan line) ahead of the short
  `membership_tiers` bullet list. A hybrid keyword + vector re-ranker would be
  the next improvement (not implemented).

### 4.6 Incremental indexing (Phase I)
`python -m rag.ingest` hashes each chunk (SHA-256 of `"{title}. {text}"`),
reuses stored vectors for unchanged chunks and embeds only new/edited ones;
removed documents drop out. Verified: first run embedded 76 chunks (before the Premium-vs-Standard section was added; the corpus is now 77 chunks), an unchanged re-run
embeds 0, editing one file embeds 1. `--full` re-embeds everything (e.g. after
changing the model). A running server hot-reloads the index when the file
changes (or `POST /rag/reload`).

### 4.7 Context injection and overflow strategy (Phase II/IV)
Explicit priority order, enforced in `conversation_manager.build_prompt_messages`:
1. **System prompt** (static, ~10k chars) - never trimmed.
2. **Current user turn** - always kept; each message capped at 2000 chars.
3. **Older history** - dropped first when over `CONTEXT_CHAR_BUDGET = 18000`:
   whole oldest turn-pairs go first (at most 6 turns are ever kept).
4. **Retrieved excerpts** - already capped at 1000 chars by `format_context`
   (best chunk always kept); only truncated as a last resort, after all
   history is gone.
Ollama is given `num_ctx = 8192` (its default 2048 would silently cut the
system prompt). Trimming is logged. Covered by unit tests.

### 4.8 Static system prompt = prefix caching
The system prompt never changes between requests; retrieved excerpts ride on
the last user message. Ollama can therefore reuse its KV cache for the long
system prompt instead of re-processing it each turn (putting excerpts in the
system prompt would invalidate the cache every message). `keep_alive=30m` and
a start-up warm-up keep the model and embedder loaded.

### 4.9 Grounding
The prompt tells the model to answer from the excerpts, mention the source
document by name, say "I don't have that information" when they do not cover
the question, and never claim a book is available unless its copy count is > 0.

### 4.10 Performance and concurrency (Phase III)
Retrieval runs on a 2-worker thread pool (the event loop stays free for other
users' streams), results are LRU-cached (256), the query embedder is warmed at
start-up, and waiting for a worker counts toward the 2 s retrieval timeout so a
burst degrades to "no context" instead of stalling replies. Streaming code is
untouched. A unit test proves the event loop is not stalled during a 20-query burst.

### 4.11 Failure handling (Phase IV)
`retrieve_safe` never raises. Index missing/corrupt/out of sync, embedder
exception, or timeout → logged warning, the assistant continues with the static
knowledge base (Assignment 1 behaviour). Outages are never cached as "no
results", and a missing index is retried every 5 s. No relevant match → no
excerpts and the prompt tells the model to admit it. Context overflow → §4.7.
`GET /rag/status` shows index state, chunk count, cache stats.
`RAG_ENABLED=0` switches retrieval off (used for A/B benchmarking).

### 4.12 Bonus: visible citations
The frontend renders each source as a chip under the reply.

## 5. Setup

```bash
# Ollama: pull the model once; the tray app already serves on :11434
ollama pull qwen2.5:3b

cd backend
python -m venv venv && venv\Scripts\activate        # Windows
pip install -r requirements.txt
pip install -r requirements-voice.txt               # faster-whisper + piper-tts
python -m voice.setup                               # Piper voice, one download
python -m rag.ingest                                 # MiniLM embeddings, one download
uvicorn main:app --port 8000
cd ../frontend && python -m http.server 5500         # open http://localhost:5500
```

The microphone button streams audio to the backend. Typed chat does not need the voice packages. `GET /voice/status` reports whether ASR and TTS are ready.

Python 3.14 note: pinned `pydantic-core` fails to build, so dependencies are
unpinned. If `fastembed`/`onnxruntime` fails to install, use Python 3.12.
`set OLLAMA_KEEP_ALIVE=30m` is optional (the backend also sends it).

## 6. Voice, tools, and the CRM (Assignment 3)

### Models

| Piece | Choice | Why |
|---|---|---|
| LLM | Ollama `qwen2.5:3b` | Same local model as Assignments 1 and 2. The system prompt stays byte-for-byte identical every turn, so its prompt cache still works. Tool results are attached to the user message, not the system prompt. |
| ASR | faster-whisper `tiny.en`, CPU, int8 | Local. No browser speech API and no cloud. `tiny.en` is the smallest English model so the gap after the person stops talking stays short. One transcription at a time (a lock) so two patrons share the CPU without crashing the model. |
| TTS | Piper `en_US-lessac-medium` | Local neural voice. A sentence is synthesised as soon as its punctuation arrives, on a worker thread, and the WAV is sent with a sequence number. The browser plays them in order while the text is still streaming. |
| Embeddings | fastembed MiniLM | Unchanged from Assignment 2. |

### How a spoken turn works

1. The page captures the microphone, downsamples to 16 kHz PCM, and sends short base64 chunks.
2. The server watches the energy. After speech has started, about 800 ms of silence (or the Stop button, or 30 seconds) ends the utterance.
3. faster-whisper produces the transcript. A partial transcript is attempted once the audio is longer than 4 seconds. The final transcript is what enters the conversation.
4. That text goes through retrieval, tools, and token streaming, the same function typed text uses (`turn.complete_turn`).
5. Finished sentences are spoken immediately. The reply is not buffered until the model stops.

If ASR or TTS is not installed, the socket gets a clear error (ASR) or the reply stays text-only (TTS). The server does not hang.

### Tools

The orchestrator has two paths, and both end the same way: a tool runs, its text is attached to the user message, and the model writes the reply the patron sees and hears.

1. **Fast path, before generation.** Obvious sentences are turned into a call without spending a model round trip. "How much is the fine for 4 days?" is one. "Book a study room tomorrow at 2pm" is one. A vague "can I book a study room?" is left to retrieval, because it has no date or time.
2. **Prompt path.** The system prompt (`tools/specs.py`, included in `prompts.py`) tells the model that if it needs a tool it must reply with only `TOOL_CALL: {"name":"...","arguments":{...}}`. `StreamGate` hides that line. The tool runs. A second generation phrases the answer. At most one model-requested call per turn. A timeout or a crash becomes a sentence, not a dead socket. The whole tool step shares a **2 second** deadline.

| Tool | What it does | Example the model can emit |
|---|---|---|
| `calculate_fine` | $0.25/day/item, cap $10/item, block above $15, lost at 30 days. Required: `days`. Optional: `items` (default 1). | `TOOL_CALL: {"name":"calculate_fine","arguments":{"days":4,"items":1}}` → `$1.00` |
| `study_room` | Local `data/calendar.json`. `action` is `book`, `list`, or `cancel`. Book needs `date` (`YYYY-MM-DD`) and `start` (`HH:MM`). `hours` is 1 or 2, `room` is 1–4, within opening hours and 7 days ahead. | `TOOL_CALL: {"name":"study_room","arguments":{"action":"book","date":"2026-10-26","start":"14:00","hours":2,"name":"Ada"}}` |
| `search_catalogue` | SQLite built from `corpus/books`. Argument: `query` (title or author). | `TOOL_CALL: {"name":"search_catalogue","arguments":{"query":"Clean Code"}}` |
| `patron_memory` (bonus CRM) | JSON file keyed by the browser `user_id`. `action` is `recall` or `update`, with optional `name`, `email`, `phone`, `preference`. | `TOOL_CALL: {"name":"patron_memory","arguments":{"action":"update","name":"Ada","email":"ada@example.com"}}` |

Schemas are also served by `GET /tools`.

The CRM record is read at the start of every turn that has a `user_id` and injected as `PATRON MEMORY`, so a returning patron can be greeted by name and the last question can be mentioned once. "My name is Ada" updates the record on the fast path. The question itself is stored at the end of the turn, so the next visit still knows what they asked. The browser keeps `library_user_id` in localStorage, separate from the session, so New session clears the chat and keeps the person.

### Latency (Phase V)

`python benchmark_voice.py` (hardware is in the evaluation section below):

| Stage | Result |
|---|---|
| Tool calls (fine, catalogue, study-room list), 60 calls | avg **3.25 ms**, max **23.21 ms** |
| Retrieval | avg **75.3 ms** (Assignment 2 measurement, unchanged) |
| ASR | warm avg **744 ms**, max **755 ms** on a 1.86 s Piper sentence. Transcript came back as "The library is open until 8." The first launch also downloads `tiny.en` (about 3.5 minutes observed once); that download is not part of the 744 ms. |
| TTS first sentence | avg **235 ms**, max **241 ms** (`en_US-lessac-medium`, sentence "The overdue fine is one dollar.") |
| LLM first token | about **1.5 s** with RAG off and about **8.3 s** with a retrieved excerpt, from the end-to-end table below. Voice does not add a second model call on the fast path. |
| End of speech → first audio | ASR 744 ms + retrieval 75 ms + tools 3 ms + LLM first token + TTS 235 ms. Tools and retrieval are far under the 2-second budget. Adding the measured ASR and TTS to the no-RAG first token (~1.5 s) is about **2.6 s**. With a retrieved excerpt the first token is ~8.3 s, so the 1–2 second target is missed on this CPU because of the language model, not because of lookup or speech. A voice `done` event includes `timings` with each stage for that turn. |

Two patrons at once: each WebSocket is its own task. Retrieval, tools, ASR, and TTS run in threads, so one person's embed or transcription does not freeze the other's socket. ASR holds one model lock so the two calls queue instead of corrupting Whisper. Ollama still generates one reply at a time unless `OLLAMA_NUM_PARALLEL` is raised.

## 7. Tests

```bash
cd backend && python -m unittest discover -s tests -t .
```

The suite covers the Assignment 1–2 behaviour (corpus, chunking, ingest, retriever, overflow, WebSocket contract) and Assignment 3: fine maths, calendar conflicts and the file lock, catalogue copy counts, CRM recall across a reload of the file, the stream gate, tool timeouts that do not block the event loop, end-of-speech detection, sentence splitting, a spoken turn that hits `calculate_fine` and returns audio, and a model `TOOL_CALL` that the patron never sees. Tests use a lexical embedder and a fake Ollama. They do not need the voice models.

### End-to-end latency finding (Intel Core i7-8650U, 4 cores / 8 threads, qwen2.5:3b Q4, CPU only)

`benchmark_e2e.py`, same 6 questions, fresh session each, warm model, first ask of each question:

| | RAG off | RAG on (budget 1500 chars) |
|---|---|---|
| time to first token (avg) | 1.52 s | 11.62 s |
| time to first token (median) | 1.44 s | 11.65 s |
| time to first token (max) | 1.90 s | 15.51 s |
| total reply time (avg) | 8.97 s | 31.69 s |

Retrieval itself is **75 ms on average** (max 709 ms = first call; search 0.2 ms) so the
< 1 s retrieval requirement holds with a large margin. The cost is not retrieval but
**prompt processing**: the excerpts add several hundred tokens that a 15 W laptop CPU must
read before the first token. Repeating an identical question is fast (0.84 s, on and off)
because retrieval and Ollama's prompt cache both hit. Replies with excerpts were also
longer (the model restates more), which inflates total time.

Mitigation applied afterwards: `MAX_CONTEXT_CHARS` 1500 -> 1000 and `INJECT_MARGIN = 0.10`
(a chunk is injected only if it scores within 0.10 of the best one). Retrieval still returns
k = 4; only what is placed in the prompt shrinks. Re-measured, same six questions:

| first ask of each question | RAG off | RAG on, budget 1500 | RAG on, budget 1000 + margin |
|---|---|---|---|
| TTFT avg | 1.52 s | 11.62 s | **8.28 s** |
| TTFT median | 1.44 s | 11.65 s | 8.50 s |
| TTFT max | 1.90 s | 15.51 s | 10.20 s |
| total reply avg | 8.97 s | 31.69 s | 24.60 s |
| identical repeat TTFT | 0.84 s | 0.84 s | 0.90 s |

That is a 29% lower time-to-first-token, and each answer now cites exactly one document
(the one the model saw). The remaining ~7 s over the no-RAG baseline is the cost of reading
one ~650-character chunk at roughly 25 prompt tokens/s on this CPU, so it cannot be removed
without either smaller chunks or a faster machine. Trade-off accepted: a slower first token
in exchange for grounded, cited answers. The tokens/s of generation are unchanged (streaming
code was not touched).

## 8. Evaluation and benchmarks

| What | Command | Result |
|---|---|---|
| Retrieval quality + threshold | `python eval_retrieval.py --sweep` | measured at MIN_SCORE 0.30 on 51 in-domain + 12 no-match queries: hit@1 88%, hit@4 94% (48/51), MRR 0.91, no-match correct 9/12 |
| Retrieval latency (embed vs search, p50/p95, <1 s) | `python benchmark_retrieval.py --runs 10` | measured: avg **75.3 ms** end to end (embed query + search), max 708.7 ms (first call, model warm-up), vector search alone 0.2 ms; 10/10 requests under 1 s |
| LLM only | `python benchmark.py --model qwen2.5:3b --runs 5` | Assignment 1 run: TTFT 1.503 s, total 10.219 s, 6.53 tok/s |
| End-to-end, RAG on vs off | `python benchmark_e2e.py --label rag_on` / restart with `set RAG_ENABLED=0` → `--label rag_off` | see the end-to-end table in Tests: first-ask TTFT 1.52 s (off) vs 8.28 s (on) |
| Tool latency | `python benchmark_voice.py` | measured: 60 calls, avg **3.25 ms**, max **23.21 ms** |
| ASR / TTS | same command | ASR warm **744 ms** (1.86 s sentence). TTS first sentence avg **235 ms**, max **241 ms** |
| Real dialogues | `python collect_transcripts.py` → `docs/example_dialogues.md` | generated |

Hardware: Dell Latitude 5490, Intel Core i7-8650U @ 1.90 GHz (4 cores / 8 threads), 16 GB RAM, Windows, no GPU.
The first request after loading the model is slow (cold load ~50 s observed);
numbers above exclude it (warm-up request first).

## 9. Known limitations

- Retrieval quality was measured on only 60 hand-written queries; threshold
  tuned on the same set (see §4.5). Three library-adjacent but uncovered
  questions still retrieve something and rely on the model to decline.
- Flat search is O(n): fine at 77 chunks, would need FAISS/HNSW at scale.
- Citations list the documents retrieved, not a verified per-sentence
  attribution; a 3B model can still paraphrase loosely or ignore excerpts.
- Prompt-level guardrails against jailbreaks are not a hard guarantee.
- No live inventory: copy counts come from the corpus files.
- Sessions are in-memory and lost on restart; retrieval degradation is logged
  and visible in `/rag/status` but not shown to patrons.
- English only; MiniLM and `tiny.en` have no multilingual support.
- End of speech to first audio cannot hit 1–2 seconds on this CPU, because the language model's first token is already several seconds. Retrieval (75 ms) and tools (under 12 ms) are not the cause. The prompt-based tool path adds a second generation when the fast path does not recognise the sentence.
- A 3B model sometimes ignores the `TOOL_CALL` line. The fast path covers the phrasings in the tests; a paraphrased request can miss the tool and be answered from retrieval instead.
- Piper and faster-whisper are local. The first ASR run downloads `tiny.en`. After that, a short utterance transcribes in about 0.7 s and the first sentence is spoken in about 0.2 s.
- Study-room bookings and patron records are local JSON files, not a real calendar or membership database. Two patrons booking the same room are serialised by a file lock.
- Small-model answer quality (see `docs/example_dialogues.md`, real output): a follow-up
  such as "How many copies of it?" retrieved the right document but the 3B model repeated
  its previous answer instead of stating "0 of 5"; "can I book a study room?" produced a
  self-contradictory sentence ("I can't book... you can book online"); "does the library
  offer 3D printing?" cited the printing/computers policy although it correctly said no.
  Retrieval was right in all three; the weakness is generation.
- In the 10-turn conversation, first-token time rose to ~30 s on later turns (vs 5-8 s for
  single questions). Longer history plus excerpts means more prompt to read on a 15 W CPU;
  we did not isolate how much is history growth versus trimming (`MAX_TURNS=6`).

## 10. Layout

```
backend/ main.py turn.py llm_engine.py conversation_manager.py prompts.py
         rag/{chunking,embedder,ingest,retriever}.py  rag/index/ (generated)
         tools/{fine,calendar_tool,catalogue,crm,orchestrator,specs}.py
         voice/{asr,tts,endpoint,sentences,pipeline,setup}.py
         corpus/{books,policies}/  eval/retrieval_queries.json (60 labelled)
         eval_retrieval.py benchmark*.py collect_transcripts.py tests/
         requirements-voice.txt
frontend/index.html   docs/example_dialogues.md
```
