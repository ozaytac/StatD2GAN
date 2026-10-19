#!/usr/bin/env python3
"""Build the release tree from the frozen archive plus this post-freeze kit. Run once, by the author.

  python tools/assemble_release.py --frozen-tar statd2gan_repro_3.0.0.tar.gz --out StatD2GAN \
      --weights-out weights_upload --raw-data include|exclude

1. Extract the frozen archive into --out, except runs/**/*.pt (weights).
   Every extracted file listed in MANIFEST.json is checked against its sha256; any mismatch aborts.
2. Copy the post-freeze kit (the directory holding this tools/ folder) on top. A kit file may
   replace a frozen file only for the superseded root metadata (README.md, CITATION.cff,
   .zenodo.json), and only if the kit carries the frozen original under frozen_meta/; anything else
   aborts.
3. Pack the 275 .pt files into one uncompressed tar per location in --weights-out (sorted members,
   mtime 0, uid/gid 0, mode 0644, so the archives are reproducible), check every .pt against
   MANIFEST.json and write weights_manifest.json (archive and file sizes and sha256) into --out.
4. --raw-data exclude removes data/raw/*.csv from --out (they stay in the frozen archive);
   include keeps them. This is the author's licence decision and has no default.
5. Register the post-freeze layer (tools/register_release.py) and run tools/verify.py in --out;
   build/ and __pycache__/ left by these steps are removed, so --out holds only the release.
Nothing is uploaded, pushed or tagged.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
SKIP_KIT = {".git", "build", "__pycache__", ".venv", "weights"}
SUPERSEDABLE = {"README.md", "CITATION.cff", ".zenodo.json"}


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main(a) -> int:
    out, wout = a.out.resolve(), a.weights_out.resolve()
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} exists and is not empty")
    out.mkdir(parents=True, exist_ok=True); wout.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="pt_"))
    n = 0
    with tarfile.open(a.frozen_tar) as t:
        for m in t:
            if not m.isfile():
                continue
            rel = m.name.split("/", 1)[1] if "/" in m.name else m.name
            dst = (tmp if rel.endswith(".pt") else out) / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            with t.extractfile(m) as src, open(dst, "wb") as f:
                shutil.copyfileobj(src, f, 1 << 22)
            n += 1
    man = json.loads((out / "MANIFEST.json").read_text())
    F = man["files"]
    bad = [k for k in F if not k.endswith(".pt")
           and (not (out / k).exists() or sha_bytes((out / k).read_bytes()) != F[k])]
    if bad:
        sys.exit(f"frozen archive does not match MANIFEST.json: {bad[:10]} ({len(bad)} files)")
    print(f"extracted {n} files; {len(F)} MANIFEST entries checked")

    for f in sorted(KIT.rglob("*")):
        rel = f.relative_to(KIT)
        if f.is_dir() or SKIP_KIT & set(rel.parts):
            continue
        if str(rel) in F and sha_bytes(f.read_bytes()) == F[str(rel)]:
            continue
        if str(rel) in F:
            keep = KIT / "frozen_meta" / rel
            if str(rel) not in SUPERSEDABLE or not keep.exists() or sha_bytes(keep.read_bytes()) != F[str(rel)]:
                sys.exit(f"kit file would replace frozen {rel}")
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, out / rel)

    files, archives = [], []
    pts = sorted(p for p in tmp.rglob("*.pt"))
    for loc in sorted({p.parent.name for p in pts}):
        name = f"weights_{loc}.tar"
        with tarfile.open(wout / name, "w", format=tarfile.PAX_FORMAT) as t:
            for p in [p for p in pts if p.parent.name == loc]:
                rel = str(p.relative_to(tmp))
                data = p.read_bytes(); h = sha_bytes(data)
                if F.get(rel) != h:
                    sys.exit(f"{rel}: sha256 differs from MANIFEST.json")
                ti = tarfile.TarInfo(rel); ti.size, ti.mtime, ti.mode, ti.uid, ti.gid = len(data), 0, 0o644, 0, 0
                ti.uname = ti.gname = ""
                t.addfile(ti, io.BytesIO(data))
                files.append(dict(path=rel, bytes=len(data), sha256=h, archive=name))
        ab = (wout / name).read_bytes()
        archives.append(dict(name=name, bytes=len(ab), sha256=sha_bytes(ab), members=sum(f["archive"] == name for f in files)))
    if len(files) != sum(k.endswith(".pt") for k in F):
        sys.exit(f"found {len(files)} .pt, MANIFEST.json lists {sum(k.endswith('.pt') for k in F)}")
    wm = json.loads((out / "weights_manifest.json").read_text()) if (out / "weights_manifest.json").exists() else {}
    wm.update(archives=archives, files=files, total_bytes=sum(f["bytes"] for f in files))
    (out / "weights_manifest.json").write_text(json.dumps(wm, indent=1) + "\n")
    shutil.rmtree(tmp)
    print(f"packed {len(files)} .pt into {len(archives)} archives ({wm['total_bytes'] / 1e9:.2f} GB) in {wout}")

    if a.raw_data == "exclude":
        for f in (out / "data" / "raw").glob("*.csv"):
            f.unlink()
        print("data/raw/*.csv removed from the release tree (they remain in the frozen archive)")
    subprocess.run([sys.executable, "tools/register_release.py"], cwd=out, check=True)
    rc = subprocess.run([sys.executable, "tools/verify.py"], cwd=out).returncode
    for scratch in (out / "build", *out.rglob("__pycache__")):  # verify's report and bytecode: not part of the release
        shutil.rmtree(scratch, ignore_errors=True)
    return rc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen-tar", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--weights-out", type=Path, required=True)
    ap.add_argument("--raw-data", choices=["include", "exclude"], required=True)
    sys.exit(main(ap.parse_args()))
