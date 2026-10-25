"""Overdue-fine calculator. Figures match corpus/policies/overdue_fines_and_lost_items.md."""

RATE_CENTS = 25          # $0.25 per day per item
CAP_CENTS = 1000         # $10.00 per item
BLOCK_CENTS = 1500       # account blocked above $15.00
LOST_AFTER_DAYS = 30


def _int_arg(value, default=None):
    if value is None or value == "":
        return default
    return int(value)


def calculate(arguments: dict) -> str:
    try:
        days = _int_arg(arguments.get("days"))
        items = _int_arg(arguments.get("items"), 1)
    except (TypeError, ValueError):
        return "I need a whole number of days (and optionally items) to calculate a fine."
    if days is None:
        return "I need the number of days overdue to calculate a fine."
    if days < 0 or items is None or items < 1 or days > 3650 or items > 100:
        return "Check the days overdue and the number of items, then ask me again."

    per_item = min(CAP_CENTS, RATE_CENTS * days)
    total = per_item * items
    text = (
        f"Overdue fine for {items} item(s), {days} day(s) late: ${total / 100:.2f}. "
        f"The rate is $0.25 per day per item, capped at $10.00 per item."
    )
    if days >= LOST_AFTER_DAYS:
        text += (
            f" At {days} days the item is treated as lost: staff charge the replacement cost "
            "plus a $5.00 processing fee. The overdue fine above is still capped at $10.00 per item."
        )
    if total > BLOCK_CENTS:
        text += " This total is over $15.00, so the account is blocked from new borrowing until it is paid down."
    return text
