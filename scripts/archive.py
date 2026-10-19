"""dist/statd2gan_repro_<version>.tar.gz + MANIFEST.json (sha256 of every file, code drift since freeze,
run counts, GPU hours, environment). Everything the paper reports is inside. --no-weights skips runs/*/*.pt."""
import hashlib, json, sys, tarfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s2g import config
from s2g.config import P, ROOT

sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
lock = json.loads(config.LOCK.read_text())
keep_w = "--no-weights" not in sys.argv
skip = ("__pycache__", ".git", "dist", "data/processed", "data/raw_redownload", "runs/_smoke")
files = [f for f in sorted(ROOT.rglob("*")) if f.is_file() and not any(s in f.relative_to(ROOT).as_posix() for s in skip)
         and (keep_w or f.suffix != ".pt")]
drift = [k for k, h in lock["code_sha256"].items() if not (ROOT / k).exists() or sha(ROOT / k) != h]
runs = [f for f in (ROOT / "runs").glob("*/*.json") if not f.parent.name.startswith("_")]; train_s = sum(json.loads(f.read_text()).get("train_seconds", 0) for f in runs)
man = dict(version=P["version"], protocol_sha256=lock["sha256"], git_commit=lock["git_commit"], frozen_at=lock["frozen_at"],
           code_changed_since_freeze=drift, n_run_records=len(runs), gpu_hours=round(train_s / 3600, 1),
           weights_included=keep_w, files={f.relative_to(ROOT).as_posix(): sha(f) for f in files})
(ROOT / "MANIFEST.json").write_text(json.dumps(man, indent=1))
out = ROOT / "dist" / f"statd2gan_repro_{P['version']}{'' if keep_w else '_noweights'}.tar.gz"; out.parent.mkdir(exist_ok=True)
with tarfile.open(out, "w:gz") as t:
    for f in [*files, ROOT / "MANIFEST.json"]:
        t.add(f, arcname=f"statd2gan_repro/{f.relative_to(ROOT).as_posix()}")
print(out, f"{out.stat().st_size / 1e6:.1f} MB", "| code drift since freeze:", drift or "none")
