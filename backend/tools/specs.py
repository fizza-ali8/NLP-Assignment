"""
Tool names, descriptions, and JSON schemas.

The system prompt and the orchestrator both use this text, so the model
and the code agree on the calling convention.
"""

TOOL_INSTRUCTIONS = """
TOOLS (call one only when the patron's request needs live figures or a saved record):
Reply with ONLY this single line and nothing else when you need a tool and the
message does not already contain a TOOL RESULT:
TOOL_CALL: {"name":"<tool>","arguments":{...}}
- calculate_fine: overdue fine in dollars. arguments {"days": <int>, "items": <int, default 1>}
- study_room: book, list, or cancel a study room. arguments {"action": "book"|"list"|"cancel", "date": "YYYY-MM-DD", "start": "HH:MM", "hours": 1 or 2, "room": 1-4, "name": "<string>"}
- search_catalogue: find a book by title or author in the library database. arguments {"query": "<title or author>"}
- patron_memory: save or read this patron's name, contact, preference, and past questions. arguments {"action": "update"|"recall", "name": "", "email": "", "phone": "", "preference": ""}
If TOOL RESULT or PATRON MEMORY is already in the message, answer in normal sentences and trust those figures. Never invent a fine, a booking, or a copy count that disagrees with a TOOL RESULT. Do not call a tool for off-topic chatter.
patron_memory reaches other conversations. Call it only when the patron explicitly asks to save something for later or to use their saved record. A name said in this chat stays in this chat. If they ask for their name and this chat does not contain it, say you do not know.
""".strip()

TOOL_NAMES = (
    "calculate_fine",
    "study_room",
    "search_catalogue",
    "patron_memory",
)

# Documented schemas (also printed by GET /tools and the README).
SCHEMAS = [
    {
        "name": "calculate_fine",
        "description": "Compute the Riverbend overdue fine for a number of days and items. Rate is $0.25 per day per item, capped at $10 per item. Totals over $15 block new borrowing. Items 30 or more days late are also treated as lost.",
        "parameters": {
            "type": "object",
            "properties": {
                "days": {"type": "integer", "description": "Days past the due date, 0 or more."},
                "items": {"type": "integer", "description": "How many items are overdue. Defaults to 1."},
            },
            "required": ["days"],
        },
        "example": {"name": "calculate_fine", "arguments": {"days": 4, "items": 1}},
    },
    {
        "name": "study_room",
        "description": "Read and write study-room bookings in the local calendar. Four rooms, at most 2 hours, at most 7 days ahead, only while the library is open.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["book", "list", "cancel"]},
                "date": {"type": "string", "description": "YYYY-MM-DD. Required for book."},
                "start": {"type": "string", "description": "24-hour HH:MM. Required for book."},
                "hours": {"type": "integer", "description": "1 or 2. Defaults to 1."},
                "room": {"type": "integer", "description": "Room number 1-4. If omitted, the first free room is used."},
                "name": {"type": "string", "description": "Name to put on the booking."},
            },
            "required": ["action"],
        },
        "example": {
            "name": "study_room",
            "arguments": {"action": "book", "date": "2026-10-26", "start": "14:00", "hours": 2, "name": "Ada"},
        },
    },
    {
        "name": "search_catalogue",
        "description": "Search the local catalogue database by title or author and return copy counts and the shelf.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Title or author to look up."},
            },
            "required": ["query"],
        },
        "example": {"name": "search_catalogue", "arguments": {"query": "Clean Code"}},
    },
    {
        "name": "patron_memory",
        "description": "CRM record for this patron, keyed by their browser user id. recall reads it; update stores a name, email, phone, or preference they just gave.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["recall", "update"]},
                "name": {"type": "string"},
                "email": {"type": "string"},
                "phone": {"type": "string"},
                "preference": {"type": "string"},
            },
            "required": ["action"],
        },
        "example": {"name": "patron_memory", "arguments": {"action": "update", "name": "Ada", "email": "ada@example.com"}},
    },
]
