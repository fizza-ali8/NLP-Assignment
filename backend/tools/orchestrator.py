"""
Decides when a tool is needed, runs it, and hands the text back to the model.

Two ways a tool runs (both explained the same way in the viva):

1. Fast path, before generation. Obvious phrasings ("fine for 4 days",
   "book a study room tomorrow at 2pm") are turned into a tool call here.
   The tool result is attached to the user message, then the model answers
   once. This stays inside the latency budget because it does not spend a
   model call on the decision.

2. Prompt path, during generation. The system prompt tells the model to
   emit a single line `TOOL_CALL: {...}` and nothing else. StreamGate hides
   that line from the patron, this module runs the tool, and a second
   generation phrases the final answer. At most one model-requested tool
   call per turn.
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field

from tools import calendar_tool, catalogue, crm, fine
from tools.specs import TOOL_NAMES

logger = logging.getLogger("tools")

TOOL_TIMEOUT_S = 2.0
PREFIX = "TOOL_CALL:"

_PRONOUNS = {"it", "that", "this", "them", "those", "these", "one", "something", "anything", "there"}
_NAME_STOP = {"me", "a", "the", "room", "two", "one", "tomorrow", "today", "hours", "hour"}


@dataclass
class ToolCall:
    name: str
    arguments: dict = field(default_factory=dict)
    user_id: str = ""
    allow_saved_memory: bool = False


def dispatch(call: ToolCall) -> str:
    """Runs one tool. Raises only if the tool itself raises; callers catch that."""
    args = dict(call.arguments)
    args.setdefault("user_id", call.user_id)
    if call.name == "calculate_fine":
        return fine.calculate(args)
    if call.name == "study_room":
        return calendar_tool.run(args)
    if call.name == "search_catalogue":
        return catalogue.search(args)
    if call.name == "patron_memory":
        if not call.allow_saved_memory:
            return (
                "Saved records from other conversations are not available in this chat. "
                "Use only what was said in this conversation, and say you do not know if it was not."
            )
        return crm.run(args)
    return f"I don't have a tool named {call.name}. I can calculate a fine, book a study room, or search the catalogue."


async def execute(call: ToolCall, timeout: float = TOOL_TIMEOUT_S) -> str:
    """Async, timed. A timeout or a crash becomes a sentence, never an exception."""
    try:
        return await asyncio.wait_for(asyncio.to_thread(dispatch, call), timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning("tool %s timed out after %.1fs", call.name, timeout)
        return f"The {call.name} tool took too long and did not finish. Please try again."
    except Exception as e:  # noqa: BLE001
        logger.warning("tool %s failed: %r", call.name, e)
        return f"The {call.name} tool could not finish ({e}). Please try again or ask at the front desk."


async def run_all(calls: list[ToolCall], timeout: float = TOOL_TIMEOUT_S) -> tuple[str, list[str], float]:
    """Run every selected tool inside one shared deadline. Returns (text, names, seconds)."""
    t0 = time.perf_counter()
    texts, names = [], []
    for call in calls:
        remaining = timeout - (time.perf_counter() - t0)
        if remaining <= 0:
            texts.append("A library tool was skipped because the tool time limit was reached.")
            break
        texts.append(await execute(call, timeout=remaining))
        names.append(call.name)
    elapsed = time.perf_counter() - t0
    return "\n".join(texts), names, elapsed


def _first_object(raw: str) -> str:
    depth = 0
    for i, ch in enumerate(raw):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return raw[: i + 1]
    return raw


def parse_tool_call(text: str, user_id: str = "", allow_saved_memory: bool = False) -> ToolCall | None:
    """Pull a TOOL_CALL JSON object out of a model reply. None if it is not one."""
    stripped = text.lstrip()
    if not stripped.startswith(PREFIX):
        return None
    raw = stripped[len(PREFIX):].strip()
    if not raw.startswith("{"):
        return None
    try:
        data = json.loads(_first_object(raw))
    except json.JSONDecodeError:
        return None
    name = data.get("name")
    args = data.get("arguments") or {}
    if name not in TOOL_NAMES or not isinstance(args, dict):
        return None
    return ToolCall(
        name=name, arguments=args, user_id=user_id, allow_saved_memory=allow_saved_memory,
    )


class StreamGate:
    """
    Buffers the start of a reply until it is clearly NOT a tool call, then
    releases the held text. A real tool call is returned whole, once the
    JSON object has closed, and is never released as patron-visible text.
    """

    def __init__(self):
        self.pending = ""
        self.streaming = False

    def feed(self, piece: str) -> tuple[str, str]:
        """Returns ('hold', ''), ('token', text), or ('tool', full_buffered_text)."""
        if self.streaming:
            return "token", piece
        self.pending += piece
        stripped = self.pending.lstrip()
        if not stripped:
            return "hold", ""
        if PREFIX.startswith(stripped) and len(stripped) < len(PREFIX):
            return "hold", ""
        if stripped.startswith(PREFIX):
            raw = stripped[len(PREFIX):].lstrip()
            if raw.startswith("{") and raw.count("{") > raw.count("}"):
                if len(stripped) < 500:
                    return "hold", ""
            return "tool", self.pending
        self.streaming = True
        text = self.pending
        self.pending = ""
        return "token", text

    def finish(self) -> tuple[str, str]:
        """End of stream while still buffering. ('token', text) or ('tool', text) or ('hold', '')."""
        if self.streaming or not self.pending:
            return "hold", ""
        stripped = self.pending.lstrip()
        if stripped.startswith(PREFIX):
            return "tool", self.pending
        self.streaming = True
        text = self.pending
        self.pending = ""
        return "token", text


def _clean_title(raw: str) -> str | None:
    title = raw.strip().strip("?.!\"' ")
    title = re.sub(r"^(the book|a book|the)\s+", "", title, flags=re.I).strip()
    if not title or len(title) < 3:
        return None
    first = title.split()[0].lower()
    if first in _PRONOUNS:
        return None
    if "study room" in title.lower():
        return None
    return title[:80]


def _parse_clock(text: str) -> str | None:
    m = re.search(r"\bat\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", text, re.I)
    if not m:
        m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", text, re.I)
    if not m:
        return None
    hour = int(m.group(1))
    minute = int(m.group(2) or 0)
    ampm = (m.group(3) or "").lower()
    if ampm == "pm" and hour < 12:
        hour += 12
    if ampm == "am" and hour == 12:
        hour = 0
    if hour > 23 or minute > 59:
        return None
    return f"{hour:02d}:{minute:02d}"


def _parse_day(text: str) -> str | None:
    from datetime import date, timedelta

    low = text.lower()
    today = date.today()
    if "tomorrow" in low:
        return (today + timedelta(days=1)).isoformat()
    if re.search(r"\btoday\b", low):
        return today.isoformat()
    m = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", text)
    return m.group(1) if m else None


_SHARE = re.compile(
    r"\b("
    r"for next time|other conversations?|other chats?|"
    r"remember me next time|save (this|my name|my email|my phone|my details) for later|"
    r"use (my |the )?saved|saved (memory|record|details|profile)|"
    r"what do you (have saved|remember) about me"
    r")\b",
    re.I,
)


def allows_cross_conversation(text: str) -> bool:
    """True only when the patron explicitly asks to use memory from outside this chat."""
    return bool(_SHARE.search(text or ""))


def detect(user_message: str, user_id: str = "") -> list[ToolCall]:
    """Fast path. Returns zero or more tool calls. Does not touch the network or the model."""
    text = " ".join(user_message.split())
    if not text:
        return []
    calls: list[ToolCall] = []

    memory_args: dict = {}
    name = re.search(r"\bmy name is\s+([A-Za-z]+(?:\s+[A-Za-z]+){0,2})", text, re.I)
    if name:
        memory_args["name"] = name.group(1).strip()
    email = re.search(r"[\w.+-]+@[\w.-]+\.\w+", text)
    if email:
        memory_args["email"] = email.group(0)
    phone = re.search(r"\b(?:\+?\d{1,2}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b", text)
    if phone:
        memory_args["phone"] = phone.group(0)
    pref = re.search(r"\b(?:i prefer|my favorite is|my favourite is|i like reading)\s+(.{3,80})", text, re.I)
    if pref:
        memory_args["preference"] = pref.group(1).strip(" .")
    if memory_args and user_id and allows_cross_conversation(text):
        memory_args["action"] = "update"
        calls.append(ToolCall("patron_memory", memory_args, user_id, allow_saved_memory=True))

    low = text.lower()
    if re.search(r"\b(fine|fines|overdue|owe|owed)\b", low):
        days_m = re.search(r"\b(\d+)\s*days?\b", low)
        if days_m:
            items_m = re.search(r"\b(\d+)\s*(?:items?|books?)\b", low)
            calls.append(ToolCall(
                "calculate_fine",
                {"days": int(days_m.group(1)), "items": int(items_m.group(1)) if items_m else 1},
                user_id,
            ))

    if "study room" in low:
        if re.search(r"\b(cancel|drop)\b", low):
            calls.append(ToolCall("study_room", {"action": "cancel"}, user_id))
        elif re.search(r"\b(book|reserve|schedule)\b", low):
            day, start = _parse_day(text), _parse_clock(text)
            if day and start:
                hours_m = re.search(r"\b(\d+)\s*hours?\b", low)
                room_m = re.search(r"\broom\s*(\d)\b", low)
                who = re.search(r"\b(?:named|name is|under the name|for)\s+([A-Za-z][A-Za-z'-]{1,30})\b", text, re.I)
                args = {"action": "book", "date": day, "start": start}
                if hours_m:
                    args["hours"] = int(hours_m.group(1))
                if room_m:
                    args["room"] = int(room_m.group(1))
                if who and who.group(1).lower() not in _NAME_STOP:
                    args["name"] = who.group(1)
                calls.append(ToolCall("study_room", args, user_id))
        elif re.search(r"\b(free|available|bookings|reserved|which rooms)\b", low):
            args = {"action": "list"}
            day = _parse_day(text)
            if day:
                args["date"] = day
            calls.append(ToolCall("study_room", args, user_id))

    cat = re.search(
        r"(?:how many copies of|copies of|search (?:the )?(?:catalogue|catalog) for|look up|do you have)\s+(.+)$",
        text, re.I,
    )
    if not cat:
        cat = re.search(r"\bis\s+(.+?)\s+available\b", text, re.I)
    if cat:
        title = _clean_title(cat.group(1))
        if title and not re.search(r"\b(card|hours|membership|printer|wifi|fine|room)\b", title, re.I):
            calls.append(ToolCall("search_catalogue", {"query": title}, user_id))

    return calls


def memory_block(user_id: str | None) -> str:
    try:
        return crm.prompt_block(user_id)
    except Exception as e:  # noqa: BLE001 - a broken CRM file must not stop the chat
        logger.warning("patron memory unreadable: %r", e)
        return ""
