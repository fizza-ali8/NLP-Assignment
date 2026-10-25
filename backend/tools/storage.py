"""Shared path and file lock for local tool data (calendar, CRM, catalogue)."""

import os
import threading

_LOCKS: dict[str, threading.Lock] = {}
_GUARD = threading.Lock()


def data_dir() -> str:
    path = os.environ.get("LIBRARY_DATA_DIR") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
    )
    os.makedirs(path, exist_ok=True)
    return path


def file_lock(name: str) -> threading.Lock:
    with _GUARD:
        if name not in _LOCKS:
            _LOCKS[name] = threading.Lock()
        return _LOCKS[name]
