"""Local study-room calendar. One JSON file, locked so two patrons can book at once."""

import json
import os
import re
import uuid
from datetime import date, datetime, timedelta

from tools.storage import data_dir, file_lock

ROOMS = (1, 2, 3, 4)
MAX_HOURS = 2
MAX_DAYS_AHEAD = 7


def _path() -> str:
    return os.path.join(data_dir(), "calendar.json")


def _load() -> dict:
    path = _path()
    if not os.path.exists(path):
        return {"bookings": []}
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    data.setdefault("bookings", [])
    return data


def _save(data: dict) -> None:
    path = _path()
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp, path)


def open_hours(day: date) -> tuple[int, int] | None:
    """Return (open_hour, close_hour) or None if the library is closed that day."""
    if day.weekday() == 6:  # Sunday
        return None
    if day.weekday() == 5:  # Saturday
        return (10, 17)
    return (9, 20)


def _parse_date(value, today: date) -> date | None:
    if not value:
        return None
    text = str(value).strip().lower()
    if text == "today":
        return today
    if text == "tomorrow":
        return today + timedelta(days=1)
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _parse_start(value) -> tuple[int, int] | None:
    if not value:
        return None
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", str(value).strip())
    if not m:
        return None
    hour, minute = int(m.group(1)), int(m.group(2))
    if hour > 23 or minute > 59:
        return None
    return hour, minute


def _overlaps(start_min: int, hours: int, other_start: str, other_hours: int) -> bool:
    parsed = _parse_start(other_start)
    if parsed is None:
        return False
    a0, a1 = start_min, start_min + hours * 60
    oh, om = parsed
    b0, b1 = oh * 60 + om, oh * 60 + om + int(other_hours) * 60
    return a0 < b1 and b0 < a1


def run(arguments: dict, today: date | None = None) -> str:
    today = today or date.today()
    action = str(arguments.get("action") or "").strip().lower()
    user_id = str(arguments.get("user_id") or "")
    with file_lock("calendar"):
        data = _load()
        if action == "list":
            return _list(data, arguments, today)
        if action == "cancel":
            return _cancel(data, arguments, user_id)
        if action == "book":
            return _book(data, arguments, today, user_id)
    return "I can book, list, or cancel a study room. Tell me which you want."


def _list(data: dict, arguments: dict, today: date) -> str:
    day = _parse_date(arguments.get("date"), today)
    rows = []
    for b in data["bookings"]:
        if day and b["date"] != day.isoformat():
            continue
        if not day and b["date"] < today.isoformat():
            continue
        rows.append(b)
    if not rows:
        when = day.isoformat() if day else "the coming week"
        return f"No study rooms are booked for {when}. Rooms 1 to 4 are free. Bookings are up to 2 hours and up to 7 days ahead."
    lines = [
        f"Room {b['room']} on {b['date']} at {b['start']} for {b['hours']} hour(s) ({b.get('name') or 'Guest'}, id {b['id']})"
        for b in rows
    ]
    return "Study room bookings: " + "; ".join(lines) + "."


def _cancel(data: dict, arguments: dict, user_id: str) -> str:
    target = None
    booking_id = str(arguments.get("booking_id") or arguments.get("id") or "")
    for b in data["bookings"]:
        if booking_id and b["id"] == booking_id:
            target = b
            break
        if user_id and b.get("user_id") == user_id:
            target = b  # last one for this patron wins
    if target is None:
        return "I could not find a study room booking to cancel."
    data["bookings"] = [b for b in data["bookings"] if b["id"] != target["id"]]
    _save(data)
    return f"Cancelled study room {target['room']} on {target['date']} at {target['start']} (id {target['id']})."


def _book(data: dict, arguments: dict, today: date, user_id: str) -> str:
    day = _parse_date(arguments.get("date"), today)
    start = _parse_start(arguments.get("start"))
    if day is None or start is None:
        return "To book a study room I need a date (YYYY-MM-DD) and a start time (HH:MM)."
    if day < today or day > today + timedelta(days=MAX_DAYS_AHEAD):
        return "Study rooms can be booked from today up to 7 days ahead."
    hours_raw = arguments.get("hours", 1)
    try:
        hours = int(hours_raw)
    except (TypeError, ValueError):
        return "Booking length must be 1 or 2 hours."
    if hours < 1 or hours > MAX_HOURS:
        return "A study room booking can be 1 or 2 hours. If nobody is waiting, the room can be booked again afterwards."
    span = open_hours(day)
    if span is None:
        return "The library is closed on Sundays, so study rooms cannot be booked that day."
    open_h, close_h = span
    start_min = start[0] * 60 + start[1]
    end_min = start_min + hours * 60
    if start_min < open_h * 60 or end_min > close_h * 60:
        return f"That time is outside library hours on that day ({open_h:02d}:00 to {close_h:02d}:00)."
    try:
        room = int(arguments["room"]) if arguments.get("room") not in (None, "") else None
    except (TypeError, ValueError):
        return "Room must be a number from 1 to 4."
    if room is not None and room not in ROOMS:
        return "Room must be a number from 1 to 4."

    same_day = [b for b in data["bookings"] if b["date"] == day.isoformat()]
    busy = set()
    for b in same_day:
        if _overlaps(start_min, hours, b["start"], b["hours"]):
            busy.add(int(b["room"]))
    if room is None:
        free = [r for r in ROOMS if r not in busy]
        if not free:
            return f"All four study rooms are busy on {day.isoformat()} at {start[0]:02d}:{start[1]:02d}."
        room = free[0]
    elif room in busy:
        free = [r for r in ROOMS if r not in busy]
        extra = f" Free then: {', '.join(str(r) for r in free)}." if free else " No room is free then."
        return f"Room {room} is already booked on {day.isoformat()} at {start[0]:02d}:{start[1]:02d}.{extra}"

    name = str(arguments.get("name") or "Guest").strip()[:40] or "Guest"
    booking = {
        "id": uuid.uuid4().hex[:8],
        "room": room,
        "date": day.isoformat(),
        "start": f"{start[0]:02d}:{start[1]:02d}",
        "hours": hours,
        "name": name,
        "user_id": user_id,
    }
    data["bookings"].append(booking)
    _save(data)
    return (
        f"Booked study room {room} on {booking['date']} at {booking['start']} "
        f"for {hours} hour(s) under the name {name}. Booking id {booking['id']}."
    )
