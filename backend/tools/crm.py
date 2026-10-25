"""
Patron CRM. A JSON file keyed by the browser's stable user id.

recall at the start of a turn lets the assistant greet a returning patron.
update runs when they give a name, contact detail, or preference.
note_interaction records what they asked so the next session can mention it.
"""

import json
import os
from datetime import datetime, timezone

from tools.storage import data_dir, file_lock

MAX_INTERACTIONS = 5


def _path() -> str:
    return os.path.join(data_dir(), "patrons.json")


def _load() -> dict:
    path = _path()
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, dict) else {}


def _save(data: dict) -> None:
    path = _path()
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp, path)


def _blank() -> dict:
    return {
        "name": "",
        "email": "",
        "phone": "",
        "preferences": [],
        "interactions": [],
        "updated": "",
    }


def _record(data: dict, user_id: str) -> dict:
    row = data.get(user_id)
    if not isinstance(row, dict):
        row = _blank()
        data[user_id] = row
    row.setdefault("preferences", [])
    row.setdefault("interactions", [])
    return row


def prompt_block(user_id: str | None) -> str:
    """Short block injected at the start of a turn. Empty if we know nothing yet."""
    if not user_id:
        return ""
    with file_lock("crm"):
        row = _load().get(user_id)
    if not isinstance(row, dict):
        return ""
    lines = []
    if row.get("name"):
        lines.append(f"Name: {row['name']}")
    if row.get("email"):
        lines.append(f"Email: {row['email']}")
    if row.get("phone"):
        lines.append(f"Phone: {row['phone']}")
    if row.get("preferences"):
        lines.append("Preferences: " + "; ".join(row["preferences"][-3:]))
    if row.get("interactions"):
        lines.append("Last question: " + row["interactions"][-1])
    if not lines:
        return ""
    return (
        "PATRON MEMORY (saved for this person; on the first reply of a conversation, "
        "greet them by name if you know it, and you may mention the last question once):\n"
        + "\n".join(lines)
    )


def note_interaction(user_id: str | None, message: str) -> None:
    if not user_id or not message.strip():
        return
    with file_lock("crm"):
        data = _load()
        row = _record(data, user_id)
        row["interactions"] = (row["interactions"] + [message.strip()[:200]])[-MAX_INTERACTIONS:]
        row["updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _save(data)


def run(arguments: dict) -> str:
    user_id = str(arguments.get("user_id") or "").strip()
    if not user_id:
        return "I can only store a patron record when the browser sends a user id."
    action = str(arguments.get("action") or "recall").strip().lower()
    with file_lock("crm"):
        data = _load()
        row = _record(data, user_id) if action == "update" else data.get(user_id)
        if action == "recall":
            if not isinstance(row, dict) or not any(row.get(k) for k in ("name", "email", "phone", "preferences", "interactions")):
                return "No saved patron record yet."
            return _format(row)
        if action != "update":
            return "patron_memory action must be update or recall."
        changed = []
        for key in ("name", "email", "phone"):
            value = str(arguments.get(key) or "").strip()
            if value:
                row[key] = value[:80]
                changed.append(key)
        pref = str(arguments.get("preference") or "").strip()
        if pref:
            if pref[:80] not in row["preferences"]:
                row["preferences"] = (row["preferences"] + [pref[:80]])[-5:]
            changed.append("preference")
        if not changed:
            return "Tell me a name, email, phone, or preference to save."
        row["updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _save(data)
        return "Saved patron details (" + ", ".join(changed) + "). " + _format(row)


def _format(row: dict) -> str:
    bits = []
    if row.get("name"):
        bits.append(f"name {row['name']}")
    if row.get("email"):
        bits.append(f"email {row['email']}")
    if row.get("phone"):
        bits.append(f"phone {row['phone']}")
    if row.get("preferences"):
        bits.append("prefers " + "; ".join(row["preferences"]))
    if row.get("interactions"):
        bits.append("last asked: " + row["interactions"][-1])
    return "Patron record: " + "; ".join(bits) + "." if bits else "No saved patron record yet."
