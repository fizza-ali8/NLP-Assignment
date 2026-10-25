"""SQLite catalogue built from the markdown book files. Rebuilt when the files change."""

import glob
import os
import re
import sqlite3

from tools.storage import data_dir, file_lock

_BOOKS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "corpus", "books")


def _db_path() -> str:
    return os.path.join(data_dir(), "catalogue.sqlite")


def _heading(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return ""


def _field(text: str, label: str) -> str:
    m = re.search(rf"\*\*{re.escape(label)}:\*\*\s*(.+)", text)
    return m.group(1).strip() if m else ""


def _copies(text: str) -> tuple[int, int] | None:
    m = re.search(r"\*\*Copies:\*\*\s*(\d+)\s*total,\s*(\d+)\s*currently available", text)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _corpus_mtime() -> float:
    latest = 0.0
    for path in glob.glob(os.path.join(_BOOKS, "*.md")):
        latest = max(latest, os.path.getmtime(path))
    return latest


def ensure_db() -> str:
    """Create or refresh the database. Safe to call on every search."""
    path = _db_path()
    with file_lock("catalogue"):
        if os.path.exists(path) and os.path.getmtime(path) >= _corpus_mtime():
            return path
        rows = []
        for file_path in sorted(glob.glob(os.path.join(_BOOKS, "*.md"))):
            with open(file_path, "r", encoding="utf-8") as fh:
                text = fh.read()
            title = _heading(text)
            copies = _copies(text)
            if not title or copies is None:
                continue
            rows.append((
                title,
                _field(text, "Author"),
                _field(text, "Genre"),
                _field(text, "Shelf location"),
                copies[0],
                copies[1],
                os.path.basename(file_path),
            ))
        if os.path.exists(path):
            os.remove(path)
        conn = sqlite3.connect(path)
        try:
            conn.execute(
                """CREATE TABLE books (
                    title TEXT, author TEXT, genre TEXT, shelf TEXT,
                    copies_total INTEGER, copies_available INTEGER, source TEXT
                )"""
            )
            conn.executemany("INSERT INTO books VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
            conn.commit()
        finally:
            conn.close()
    return path


def search(arguments: dict) -> str:
    query = " ".join(str(arguments.get("query") or "").split())
    if len(query) < 2:
        return "Tell me a book title or an author to search the catalogue."
    try:
        path = ensure_db()
    except Exception as e:  # noqa: BLE001 - the orchestrator turns this into a spoken error
        return f"The catalogue database could not be opened ({e})."
    like = f"%{query.lower()}%"
    conn = sqlite3.connect(path)
    try:
        found = conn.execute(
            """SELECT title, author, shelf, copies_total, copies_available
               FROM books
               WHERE lower(title) LIKE ? OR lower(author) LIKE ?
               ORDER BY title LIMIT 4""",
            (like, like),
        ).fetchall()
    finally:
        conn.close()
    if not found:
        return f"No catalogue entry matched {query!r}."
    parts = []
    for title, author, shelf, total, available in found:
        who = f" by {author}" if author else ""
        where = f", shelf {shelf}" if shelf else ""
        parts.append(f"{title}{who}: {available} of {total} copies available{where}")
    return "Catalogue: " + "; ".join(parts) + "."
