"""
conversation_manager.py
------------------------
Owns per-session dialogue state and decides what goes into the LLM's
context window on every turn.

Context-memory strategy (Assignment 1, also described in the README):
  - The full transcript is kept in memory for the life of a session (so
    the UI can always show the complete history).
  - When building the prompt actually sent to the model, a SLIDING WINDOW
    of the last MAX_TURNS user/assistant turn-pairs is used by default,
    plus the system prompt (which is always included in full).
  - Each stored message is hard-capped at MAX_CHARS_PER_MSG as a safety
    limit against one very long message blowing up the prompt.

Context-overflow strategy (Assignment 2, Phase II/IV): retrieval adds a
variable-size block of document excerpts to the patron's latest message,
on top of the fixed system prompt and the history window. format_context()
in rag/retriever.py caps the excerpts at MAX_CONTEXT_CHARS, but excerpts +
a long history could still overflow the model's context window, so
build_prompt_messages() applies an explicit priority order when the total
character budget (CONTEXT_CHAR_BUDGET) is exceeded:

  1. system prompt             - never trimmed (it holds the domain policy)
  2. the patron's current turn - never dropped; its retrieved excerpts are
                                 the LAST thing shortened (hard-truncated,
                                 only if the turn is still too big alone)
  3. older history             - shrunk first, oldest turn-pairs dropped
                                 one at a time

The system prompt stays byte-identical on every turn (retrieved text is
attached to the latest user message, not to the system prompt), so the
local model server can reuse its cached processing of that large prefix.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict
import uuid

from prompts import SYSTEM_PROMPT, format_user_turn

MAX_TURNS = 6             # default number of user+assistant turn PAIRS kept in the prompt
MAX_CHARS_PER_MSG = 2000  # hard safety cap per stored message

# CONTEXT_CHAR_BUDGET sizing: our static system prompt (domain policy +
# off-topic few-shot examples, kept verbose deliberately for robustness -
# see prompts.py) is itself ~9,100 chars (~2,300 tokens) before any
# retrieved context is added, and up to ~10,850 chars (~2,700 tokens) with
# a full-size excerpts block. We measured this directly rather
# than guessing (see docs/example_dialogues.md benchmark notes). We
# request an 8192-token context window from Ollama (llm_engine.py,
# num_ctx) specifically to give this prompt room to breathe, and size the
# character budget below (~18,000 chars / ~4,500 tokens) so that system
# prompt + retrieved context + several turns of history still leaves
# comfortable headroom (~3,700 tokens) for the model's own response -
# this is the explicit strategy required by Phase IV (context overflow
# handling): if a turn's system+context+history would exceed this budget,
# build_prompt_messages() shrinks the history window first (oldest turns
# dropped), never the system policy or the current question.
CONTEXT_CHAR_BUDGET = 18000


@dataclass
class Session:
    session_id: str
    history: List[Dict[str, str]] = field(default_factory=list)  # full transcript
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # Top retrieved document of the previous turn. Lets a short follow-up
    # ("how many copies of it?") keep the document it refers to even when the
    # embedding of the rewritten query drifts to other, similar-looking entries.
    last_source: str | None = None

    def add(self, role: str, content: str):
        self.history.append({"role": role, "content": content[:MAX_CHARS_PER_MSG]})


class ConversationManager:
    def __init__(self):
        self._sessions: Dict[str, Session] = {}

    # --- session lifecycle -------------------------------------------------
    def create_session(self) -> str:
        sid = str(uuid.uuid4())
        self._sessions[sid] = Session(session_id=sid)
        return sid

    def reset_session(self, session_id: str) -> str:
        """Wipes history but keeps (or creates) the session id."""
        self._sessions[session_id] = Session(session_id=session_id)
        return session_id

    def session_exists(self, session_id: str) -> bool:
        return session_id in self._sessions

    def ensure_session(self, session_id: str) -> str:
        if not session_id or session_id not in self._sessions:
            session_id = self.create_session()
        return session_id

    # --- message handling ----------------------------------------------------
    def add_user_message(self, session_id: str, content: str):
        self._sessions[session_id].add("user", content)

    def add_assistant_message(self, session_id: str, content: str):
        self._sessions[session_id].add("assistant", content)

    def get_full_history(self, session_id: str) -> List[Dict[str, str]]:
        return list(self._sessions[session_id].history)

    def build_prompt_messages(
        self, session_id: str, retrieved_context: str = ""
    ) -> tuple[List[Dict[str, str]], int]:
        """
        Builds the exact message list sent to the LLM:

            [system prompt] + [window of older turns] + [current user turn
            (+ retrieved excerpts)]

        The stored history is never modified: excerpts are attached only to
        the copy of the latest user message used for this request.

        Returns (messages, turns_used): turns_used is how many older
        user/assistant pairs survived the budget (MAX_TURNS if nothing had
        to be dropped), so callers can log when overflow trimming happened.
        """
        session = self._sessions[session_id]
        history = session.history
        if not history or history[-1]["role"] != "user":
            raise ValueError("build_prompt_messages needs the latest user message to be added first")

        current_raw = history[-1]["content"]
        prior = history[:-1]

        # Last resort only: if this single turn (question + excerpts) would
        # not fit next to the system prompt, shorten the excerpts.
        ctx = retrieved_context
        max_turn_chars = CONTEXT_CHAR_BUDGET - len(SYSTEM_PROMPT)
        overhead = len(format_user_turn(current_raw, "x")) - len(current_raw) - 1
        allowed_ctx = max(0, max_turn_chars - len(current_raw) - overhead)
        if len(ctx) > allowed_ctx:
            ctx = ctx[:allowed_ctx].rstrip()
        current_text = format_user_turn(current_raw, ctx)

        base_len = len(SYSTEM_PROMPT) + len(current_text)
        turns = MAX_TURNS
        while True:
            windowed = prior[-(turns * 2):] if turns > 0 else []
            history_len = sum(len(m["content"]) for m in windowed)
            if base_len + history_len <= CONTEXT_CHAR_BUDGET or turns == 0:
                break
            turns -= 1

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(windowed)
        messages.append({"role": "user", "content": current_text})
        return messages, turns

    def previous_user_message(self, session_id: str) -> str | None:
        """The patron message before the latest one (used to resolve follow-up questions for retrieval)."""
        users = [m["content"] for m in self._sessions[session_id].history if m["role"] == "user"]
        return users[-2] if len(users) >= 2 else None

    def get_last_source(self, session_id: str) -> str | None:
        return self._sessions[session_id].last_source

    def set_last_source(self, session_id: str, source: str | None):
        self._sessions[session_id].last_source = source

    def turn_count(self, session_id: str) -> int:
        return len(self._sessions[session_id].history) // 2
