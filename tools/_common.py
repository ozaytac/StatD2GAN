"""Shared helpers for the post-freeze release tools (tools/*.py). Standard library only."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SHA = "2d9a3b9d210df5b2b6104f6563eff583dd5642130ac863ff882b5f380b5aabd7"


# Directories the frozen code writes into without creating them first (scripts/preprocess.py ->
# data/processed; Makefile `touch data/processed/.done`), plus the other output directories of the
# frozen Makefile and scripts. git does not track empty directories, so a fresh clone (or a scratch
# copy in build/) lacks them; every tool that runs frozen code calls ensure_dirs first.
FROZEN_OUTPUT_DIRS = ("data/processed", "data/raw_redownload", "runs", "runs/_env", "results", "logs",
                      "figures", "generated", "dist")


def ensure_dirs(root: Path = ROOT) -> None:
    for d in FROZEN_OUTPUT_DIRS:
        (root / d).mkdir(parents=True, exist_ok=True)


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text())


def run(cmd: list[str], cwd: Path, log: Path | None = None) -> float:
    """Run a command, stream output (and tee to log), return wall seconds; abort on failure."""
    t = time.time()
    print("$", " ".join(map(str, cmd)), flush=True)
    with (open(log, "a") if log else open("/dev/null", "w")) as lf:
        p = subprocess.Popen(list(map(str, cmd)), cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            sys.stdout.write(line); lf.write(line)
        if p.wait():
            sys.exit(f"FAILED ({p.returncode}): {' '.join(map(str, cmd))}")
    return time.time() - t


def fresh(d: Path) -> Path:
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    return d


def copy(rel: str, dst_root: Path, src_root: Path = ROOT) -> None:
    s, d = src_root / rel, dst_root / rel
    d.parent.mkdir(parents=True, exist_ok=True)
    (shutil.copytree if s.is_dir() else shutil.copy2)(s, d)
