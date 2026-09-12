"""Small shared filesystem helpers."""
from __future__ import annotations

import json
import os


def atomic_json_dump(path: str, data: dict) -> None:
    """Write JSON via a temp file + rename, so a crash mid-write never
    corrupts a checkpoint that resumable runs depend on."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
