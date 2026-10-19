"""Shared helper for the post-freeze reporting layer.

Two guarantees, both checked against the frozen MANIFEST.json (which is only read):
  1. verify_frozen(): every file listed in MANIFEST['files'] that this layer reads, plus
     every frozen script present locally, still has its recorded sha256. Any mismatch
     aborts before an output is written.
  2. register(): sha256 of each post-freeze script and output is written to a separate
     MANIFEST_post_freeze.json. MANIFEST.json is never modified.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

OUT = "MANIFEST_post_freeze.json"
NOTE = ("Post-freeze reporting layer. Scripts and outputs added after the protocol freeze. "
        "They re-express or re-plot frozen results/*.csv, results/decisions.json, protocol.yaml "
        "and constants of the frozen src/; no training, evaluation, inference or decision rule was changed or re-run. "
        "MANIFEST.json (frozen) is not modified.")
FROZEN_DIRS = ("src/", "scripts/")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _frozen(root: Path) -> dict[str, str]:
    m = root / "MANIFEST.json"
    if not m.exists():
        raise SystemExit(f"{m} not found: cannot verify frozen inputs")
    return json.loads(m.read_text())["files"]


def verify_frozen(root: Path, inputs: list[Path]) -> int:
    """Abort if any frozen input or locally present frozen script differs from MANIFEST."""
    files = _frozen(root)
    targets = {str(p.relative_to(root)) for p in inputs}
    targets |= {k for k in files if k.startswith(FROZEN_DIRS) and (root / k).exists()}
    bad, missing = [], []
    for rel in sorted(targets):
        if rel not in files:
            missing.append(rel)
        elif sha256(root / rel) != files[rel]:
            bad.append(rel)
    if missing:
        raise SystemExit(f"inputs not in frozen MANIFEST: {missing}")
    if bad:
        raise SystemExit(f"frozen files changed since freeze: {bad}")
    return len(targets)


def register(root: Path, files: list[Path]) -> None:
    out = root / OUT
    M = json.loads(out.read_text()) if out.exists() else {"note": NOTE, "files": {}}
    M["frozen_manifest_sha256"] = sha256(root / "MANIFEST.json")
    for f in files:
        M["files"][str(f.relative_to(root))] = sha256(f)
    M["updated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out.write_text(json.dumps(M, indent=2) + "\n")
    print(f"registered {len(files)} file(s) in {OUT}")
