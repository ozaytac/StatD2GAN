#!/usr/bin/env python3
"""make verify: integrity of the frozen layer and of the post-freeze layer. CPU, seconds.

Frozen layer (MANIFEST.json, never modified):
  * protocol.yaml sha256 == MANIFEST.protocol_sha256 == 2d9a3b9d210df5b2..., recorded in protocol.lock
  * every file listed in MANIFEST.json is present and byte-identical. The only tolerated gap is the
    generator weights (the 275 runs/**/*.pt of weights_manifest.json, not in git): if they are
    absent, verify passes with "weights not present, run make fetch-weights". Any other missing or
    changed frozen file is an error.
  * optional frozen files (MANIFEST_post_freeze.json "optional", i.e. .pytest_cache/, which pytest
    rewrites and which is not in git): absent or different is noted, never an error
  * superseded repository metadata (README.md, CITATION.cff, .zenodo.json): replaced at the root
    post-freeze. Tolerated only if MANIFEST_post_freeze.json lists it under "superseded" with the
    frozen sha256, the frozen original is kept byte-identical under frozen_meta/, and the new root
    file is registered.
Post-freeze layer (MANIFEST_post_freeze.json):
  * written against this MANIFEST.json (frozen_manifest_sha256)
  * every registered file present and byte-identical
  * coverage: every post-freeze file in the tree is registered (PF_GLOBS)
Weights present: each .pt matches weights_manifest.json and MANIFEST.json.

--partial: development trees only; other missing frozen files become warnings.
Exit 0 only if every check passes. Writes build/verify.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import PROTOCOL_SHA, ROOT, load, sha  # noqa: E402

PF_GLOBS = ["GNUmakefile", ".gitignore", "tools/*.py", "scripts/*.py", "tests/*.py", "generated/*.tex", "figures/*",
            "results_extra/*", "paper/*.tex", "weights_manifest.json", "requirements-*.txt",
            ".github/workflows/*.yml", "frozen_meta/*", "frozen_meta/.zenodo.json"]
UNREGISTERED_OK = {"MANIFEST_post_freeze.json"}
SUPERSEDABLE = {"README.md", "CITATION.cff", ".zenodo.json"}


def main(partial: bool) -> int:
    errors, warns, notes, info = [], [], [], {}
    man, pfm = load("MANIFEST.json"), load("MANIFEST_post_freeze.json")
    F, P, SUP, OPT = man["files"], pfm["files"], pfm.get("superseded", {}), pfm.get("optional", {})
    wm = load("weights_manifest.json") if (ROOT / "weights_manifest.json").exists() else {"files": []}
    weights = {w["path"]: w["sha256"] for w in wm["files"]}

    # -- protocol ---------------------------------------------------------------------------------
    p = sha(ROOT / "protocol.yaml")
    if not (p == man["protocol_sha256"] == PROTOCOL_SHA):
        errors.append(f"protocol.yaml sha256 {p[:12]} != frozen {PROTOCOL_SHA[:12]}")
    lock = ROOT / "protocol.lock"
    if lock.exists() and json.loads(lock.read_text()).get("sha256") != PROTOCOL_SHA:
        errors.append("protocol.lock does not record the frozen protocol sha256")

    # -- frozen files -----------------------------------------------------------------------------
    ok, miss_w, miss_other, bad, superseded = 0, [], [], [], []
    for rel, h in F.items():
        f, keep = ROOT / rel, ROOT / "frozen_meta" / rel
        if any(rel.startswith(o) for o in OPT):
            if not f.exists() or sha(f) != h:
                notes.append(f"optional frozen file {'absent' if not f.exists() else 'differs'}: {rel}")
            else:
                ok += 1
        elif rel in SUPERSEDABLE and rel in SUP:
            good = (SUP[rel].get("frozen_sha256") == h and keep.exists() and sha(keep) == h
                    and f.exists() and P.get(rel) == sha(f))
            (superseded if good else bad).append(rel)
        elif not f.exists():
            (miss_w if rel in weights and weights[rel] == h else miss_other).append(rel)
        elif sha(f) != h:
            bad.append(rel)
        else:
            ok += 1
    errors += [f"frozen file changed: {r}" for r in bad]
    (warns if partial else errors).extend(f"frozen file missing: {r}" for r in miss_other)
    if miss_w:
        warns.insert(0, f"weights not present ({len(miss_w)} of {len(weights)} .pt): run `make fetch-weights`")
    info["frozen"] = dict(listed=len(F), identical=ok, changed=len(bad), superseded=superseded,
                          weights_missing=len(miss_w), other_missing=len(miss_other))

    # -- post-freeze files ------------------------------------------------------------------------
    if pfm["frozen_manifest_sha256"] != sha(ROOT / "MANIFEST.json"):
        errors.append("MANIFEST_post_freeze.json was written against a different MANIFEST.json")
    pf_ok = 0
    for rel, h in P.items():
        f = ROOT / rel
        if not f.exists():
            errors.append(f"post-freeze file missing: {rel}")
        elif sha(f) != h:
            errors.append(f"post-freeze file differs from MANIFEST_post_freeze.json: {rel}")
        else:
            pf_ok += 1
    tree = {str(q.relative_to(ROOT)) for g in PF_GLOBS for q in ROOT.glob(g) if q.is_file()}
    unreg = sorted(t for t in tree if t not in F and t not in P and t not in UNREGISTERED_OK and "__pycache__" not in t)
    errors += [f"post-freeze file not registered in MANIFEST_post_freeze.json: {r}" for r in unreg]
    info["post_freeze"] = dict(registered=len(P), identical=pf_ok, unregistered=len(unreg))

    # -- weights present ----------------------------------------------------------------------------
    present = [r for r in weights if (ROOT / r).exists()]
    wbad = [r for r in present if sha(ROOT / r) != weights[r] or F.get(r) != weights[r]]
    errors += [f"weight file differs: {r}" for r in wbad]
    info["weights"] = dict(listed=len(weights), present=len(present), changed=len(wbad))

    # -- report -------------------------------------------------------------------------------------
    fr, pf, w = info["frozen"], info["post_freeze"], info["weights"]
    print(f"protocol.yaml               sha256 {p[:16]}...  (frozen {PROTOCOL_SHA[:16]}...)")
    print(f"MANIFEST.json               {fr['listed']} files: {fr['identical']} identical, {fr['changed']} changed, "
          f"{len(fr['superseded'])} superseded {fr['superseded']}, {fr['weights_missing']} weights missing, "
          f"{fr['other_missing']} other missing")
    print(f"MANIFEST_post_freeze.json   {pf['registered']} files: {pf['identical']} identical, {pf['unregistered']} unregistered")
    print(f"weights_manifest.json       {w['listed']} .pt: {w['present']} present, {w['changed']} changed")
    if notes:
        print(f"NOTE  {len(notes)} optional frozen file(s) absent or different ({', '.join(sorted(OPT))}); not an error")
    for m in warns[:12]:
        print("WARN ", m)
    if len(warns) > 12:
        print(f"WARN  ... {len(warns) - 12} more (build/verify.json)")
    for m in errors[:30]:
        print("ERROR", m)
    if len(errors) > 30:
        print(f"ERROR ... {len(errors) - 30} more (build/verify.json)")
    out = ROOT / "build" / "verify.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(dict(info=info, errors=errors, warnings=warns, notes=notes,
                                   missing_other=miss_other, missing_weights=miss_w), indent=1))
    print("verify:", "FAILED" if errors else ("PASSED with warnings" if warns else "PASSED") + (" (--partial)" if partial else ""))
    return 1 if errors else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", action="store_true", help="development trees: other missing frozen files are warnings")
    sys.exit(main(ap.parse_args().partial))
