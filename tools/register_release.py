#!/usr/bin/env python3
"""Register every post-freeze file of the release in MANIFEST_post_freeze.json (scripts/post_freeze.py).

Post-freeze files: those matched by tools/verify.PF_GLOBS that are not in the frozen MANIFEST.json,
plus root metadata superseded post-freeze (README.md, CITATION.cff, .zenodo.json). A root file
counts as superseded only if its frozen original is kept byte-identical under frozen_meta/; it is
then registered with its new sha256 under "files" and with its frozen sha256 under "superseded".
MANIFEST.json is only read. `make verify` fails on any unregistered or changed post-freeze file.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _common import ROOT, load, sha  # noqa: E402
from post_freeze import register  # noqa: E402
from verify import PF_GLOBS, SUPERSEDABLE, UNREGISTERED_OK  # noqa: E402

frozen = load("MANIFEST.json")["files"]
files = {q for g in PF_GLOBS for q in ROOT.glob(g) if q.is_file()
         and str(q.relative_to(ROOT)) not in frozen and q.name not in UNREGISTERED_OK and "__pycache__" not in str(q)}
sup = {}
for r in sorted(SUPERSEDABLE):
    root_f, keep = ROOT / r, ROOT / "frozen_meta" / r
    if root_f.exists() and sha(root_f) != frozen[r]:
        if not (keep.exists() and sha(keep) == frozen[r]):
            sys.exit(f"{r} differs from the frozen file but frozen_meta/{r} is missing or not the frozen original")
        files.add(root_f)
        sup[r] = dict(frozen_sha256=frozen[r], kept_at=f"frozen_meta/{r}", replaced_by_sha256=sha(root_f))
register(ROOT, sorted(files))
pf = ROOT / "MANIFEST_post_freeze.json"
M = json.loads(pf.read_text())
gone = sorted(k for k in M["files"] if not (ROOT / k).exists())
for k in gone:
    del M["files"][k]
if gone:
    print("dropped registrations of files no longer in the tree:", gone)
M["superseded"] = sup
M["optional"] = {".pytest_cache/": ("Frozen by accident at freeze time (pytest's own cache). pytest rewrites it and the "
                                    "release does not track it in git; make verify notes it when absent or different.")}
M["superseded_note"] = ("Repository metadata replaced at the root after the freeze (GitHub and Zenodo read it there). "
                        "The frozen originals are kept byte-identical under frozen_meta/; the code, protocol and results are unaffected.")
pf.write_text(json.dumps(M, indent=2) + "\n")
print(f"registered {len(files)} post-freeze files; superseded: {sorted(sup) or 'none'}")
