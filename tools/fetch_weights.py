#!/usr/bin/env python3
"""make fetch-weights: download the 275 generator weight files (about 5 GB) from Zenodo and verify them.

Zenodo allows at most 100 files per record, so the weights are stored as one uncompressed tar per
location (weights_<loc>.tar, 55 files each). weights_manifest.json lists every archive and every
.pt with its size and sha256; the .pt sha256 values are those of the frozen MANIFEST.json.
Each archive is checked after download, each extracted .pt is checked against both manifests, and
nothing outside runs/<loc>/*.pt is extracted. Existing, verified files are skipped.

The record id is read from weights_manifest.json ("record_id") or from ZENODO_RECORD=<id>.
Alternatively place the archives in weights/ (e.g. downloaded by hand) and run with --local.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, load, sha  # noqa: E402


def main(local: bool) -> int:
    wm, frozen = load("weights_manifest.json"), load("MANIFEST.json")["files"]
    rec = os.environ.get("ZENODO_RECORD") or wm.get("record_id")
    if not local and not rec:
        sys.exit("weights_manifest.json has no record_id yet: set ZENODO_RECORD=<id> or use --local with weights/*.tar")
    W = ROOT / "weights"
    W.mkdir(exist_ok=True)
    todo = [f for f in wm["files"] if not ((ROOT / f["path"]).exists() and sha(ROOT / f["path"]) == f["sha256"])]
    need = sorted({f["archive"] for f in todo})
    print(f"{len(wm['files']) - len(todo)} of {len(wm['files'])} weight files already verified; archives needed: {need or 'none'}")
    for name in need:
        meta = next(a for a in wm["archives"] if a["name"] == name)
        tar = W / name
        if not (tar.exists() and sha(tar) == meta["sha256"]):
            if local:
                sys.exit(f"--local: {tar} missing or sha256 mismatch")
            url = wm["url_template"].format(record_id=rec, name=name)
            print(f"downloading {name} ({meta['bytes'] / 1e9:.2f} GB) from {url}")
            urllib.request.urlretrieve(url, tar)
            if sha(tar) != meta["sha256"]:
                sys.exit(f"{name}: sha256 mismatch after download")
        members = {f["path"]: f for f in wm["files"] if f["archive"] == name}
        with tarfile.open(tar) as t:
            for m in t.getmembers():
                if m.name not in members or not m.isfile():
                    sys.exit(f"{name}: unexpected member {m.name}")
                t.extract(m, ROOT, filter="data")
        for p, f in members.items():
            h = sha(ROOT / p)
            if h != f["sha256"] or frozen.get(p) != h:
                sys.exit(f"{p}: sha256 does not match weights_manifest.json / MANIFEST.json")
        print(f"{name}: {len(members)} files extracted and verified")
    print("fetch-weights: all", len(wm["files"]), "weight files verified")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", action="store_true", help="use weights/*.tar already on disk")
    sys.exit(main(ap.parse_args().local))
