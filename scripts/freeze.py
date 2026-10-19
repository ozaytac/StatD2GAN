"""Freeze the protocol: refuse DRAFT/unset budget, write protocol.lock, tag if inside git."""
import json, subprocess, sys, datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s2g import config

p = config.load()
if p["status"] != "FROZEN":
    sys.exit("set `status: FROZEN` in protocol.yaml (and gpu_hours) in the commit that freezes it")
if p["budget"]["gpu_hours"] is None:
    sys.exit("budget.gpu_hours must be set before freeze")
if config.LOCK.exists():
    sys.exit("already frozen; a new protocol version needs a new package version")
git = lambda *a: subprocess.run(["git", *a], cwd=config.ROOT, capture_output=True, text=True)
commit = git("rev-parse", "HEAD").stdout.strip() if git("rev-parse", "--git-dir").returncode == 0 else None
dirty = bool(git("status", "--porcelain").stdout.strip()) if commit else None
if dirty:
    sys.exit("working tree dirty: commit protocol.yaml (and the code) before freezing")
import hashlib
code = {str(f.relative_to(config.ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
        for d in ("src", "scripts", "tests") for f in sorted((config.ROOT / d).rglob("*.py"))}
config.LOCK.write_text(json.dumps(dict(sha256=config.sha(), git_commit=commit, code_sha256=code,
                                       frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat()), indent=1))
print("frozen", config.sha()[:12], commit)
