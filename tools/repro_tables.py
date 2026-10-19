#!/usr/bin/env python3
"""make tables: regenerate every post-freeze number, table and figure from stored results, then check.

Runs in a scratch copy (build/tables/) so that the committed tree and MANIFEST_post_freeze.json are
never rewritten. Inputs copied: protocol.yaml, MANIFEST*.json, src/, scripts/, results/,
results_extra/, generated/{numbers,tables}.tex, runs/**/*full_model_seed*.json, runs/_env/*.json,
paper/main.tex.
Steps (CPU, about a minute):
  1. scripts/table_seq_extra.py, scripts/table_pilot_extra.py -> generated/tables_pf.tex
  2. scripts/report_extra.py     -> generated/numbers_extra.tex, generated/table_dist_pf.tex
  3. scripts/fig1_extra.py       -> figures/fig1_location_robustness_pf.{pdf,png}
  4. scripts/figs_extra.py       -> figures/fig{2,4}_*_pf.{pdf,png}
  5. compare every regenerated file with the committed one (sha256)
  6. tools/macro_check.py and tools/lint_paper.py on paper/main.tex with the regenerated macros
Text outputs (.tex) must be byte-identical; that is where every number lives. Figures depend on the
plotting stack (matplotlib, fonts; figs_extra.py PDFs also embed a creation date), so a figure that
differs is reported, and fails only with --strict-figures.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, copy, fresh, load, run, sha  # noqa: E402

PY = sys.executable
OUTPUTS = ["generated/tables_pf.tex", "generated/numbers_extra.tex", "generated/table_dist_pf.tex",
           "figures/fig1_location_robustness_pf.pdf", "figures/fig1_location_robustness_pf.png",
           "figures/fig2_dsort_mechanism_pf.pdf", "figures/fig2_dsort_mechanism_pf.png",
           "figures/fig4_variance_decomposition_pf.pdf", "figures/fig4_variance_decomposition_pf.png"]


def main(build: Path, strict_figures: bool) -> int:
    B = fresh(build)
    log = B / "tables.log"
    for rel in ["protocol.yaml", "MANIFEST.json", "MANIFEST_post_freeze.json", "src", "scripts", "results",
                "results_extra", "generated/numbers.tex", "generated/tables.tex", "paper/main.tex"]:
        copy(rel, B)
    (B / "figures").mkdir()
    runs = sorted(ROOT.glob("runs/**/*full_model_seed*.json")) + sorted(ROOT.glob("runs/_env/*.json"))
    for r in runs:
        copy(str(r.relative_to(ROOT)), B)
    t = {}
    t["table_seq_extra"] = run([PY, "scripts/table_seq_extra.py", "--root", B], B, log)
    t["table_pilot_extra"] = run([PY, "scripts/table_pilot_extra.py", "--root", B], B, log)
    t["report_extra"] = run([PY, "scripts/report_extra.py", "--root", B, "--paper", B / "paper" / "main.tex"], B, log)
    t["fig1_extra"] = run([PY, "scripts/fig1_extra.py", "--root", B], B, log)
    t["figs_extra"] = run([PY, "scripts/figs_extra.py", "--root", B], B, log)

    committed = load("MANIFEST_post_freeze.json")["files"]
    rows, fail = [], False
    for rel in OUTPUTS:
        a, b = ROOT / rel, B / rel
        same = a.exists() and b.exists() and sha(a) == sha(b)
        nodate = lambda p: re.sub(rb"/CreationDate \(D:[^)]*\)", b"", p.read_bytes())  # noqa: E731
        if not same and rel.endswith(".pdf") and a.exists() and b.exists() and nodate(a) == nodate(b):
            rows.append(dict(file=rel, status="identical except CreationDate", committed_registered=committed.get(rel) == sha(a)))
            fail |= not rows[-1]["committed_registered"]
            print(f"{'identical except date':26s} {rel}")
            continue
        reg = committed.get(rel) == (sha(a) if a.exists() else None)
        kind = "text" if rel.endswith(".tex") else "figure"
        status = "identical" if same else ("DIFFERS" if kind == "text" or strict_figures else "differs (plotting stack)")
        fail |= (not same) and (kind == "text" or strict_figures)
        fail |= not reg
        rows.append(dict(file=rel, status=status, committed_registered=reg))
        print(f"{status:26s} {rel}{'' if reg else '   [committed copy not registered!]'}")

    # macro_check expects <root>/repo (the build copy) and <root>/md (paper + its generated inputs)
    C = fresh(B / "check")
    (C / "md" / "generated").mkdir(parents=True)
    os.symlink(B, C / "repo")
    (C / "md" / "main.tex").write_bytes((B / "paper" / "main.tex").read_bytes())
    for g in ["numbers.tex", "numbers_extra.tex", "tables_pf.tex", "table_dist_pf.tex"]:
        (C / "md" / "generated" / g).write_bytes((B / "generated" / g).read_bytes())
    t["macro_check"] = run([PY, ROOT / "tools" / "macro_check.py", C], ROOT, log)
    t["lint"] = run([PY, ROOT / "tools" / "lint_paper.py", C / "md" / "main.tex", "--macros",
                     C / "md" / "generated" / "numbers.tex", C / "md" / "generated" / "numbers_extra.tex"], ROOT, log)
    lint_tail = [l for l in log.read_text().splitlines() if "errors," in l][-1]
    if not lint_tail.startswith("0 errors"):
        fail = True
    (B / "report.json").write_text(json.dumps(dict(outputs=rows, seconds=t, lint=lint_tail, runs_json=len(runs)), indent=1))
    print(f"tables: {'FAILED' if fail else 'PASSED'} ({sum(t.values()):.0f} s; lint: {lint_tail})")
    return int(fail)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=Path, default=ROOT / "build" / "tables")
    ap.add_argument("--strict-figures", action="store_true")
    a = ap.parse_args()
    sys.exit(main(a.build.resolve(), a.strict_figures))
