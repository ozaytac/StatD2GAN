#!/usr/bin/env python3
"""make paper: compile paper/main.tex with latexmk in build/paper/ (sources copied there).

Inputs: paper/main.tex, generated/{numbers,numbers_extra,tables_pf,table_dist_pf}.tex and the figure
PDFs it includes (figures/). Needs latexmk, pdflatex and the elsarticle class. If any is missing
the target prints what is missing and exits 0, so `make all` still completes.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, fresh  # noqa: E402


def main() -> int:
    tools = {t: shutil.which(t) for t in ("latexmk", "pdflatex", "kpsewhich")}
    if not all(tools.values()):
        print(f"paper: skipped, missing {[t for t, p in tools.items() if not p]} (install TeX Live or use Overleaf)")
        return 0
    missing = [f for f in ("elsarticle.cls", "lmodern.sty")
               if not subprocess.run(["kpsewhich", f], capture_output=True, text=True).stdout.strip()]
    if missing:
        print(f"paper: skipped, {missing} not found (TeX Live packages elsarticle, lm; Overleaf has both)")
        return 0
    B = fresh(ROOT / "build" / "paper")
    tex = (ROOT / "paper" / "main.tex").read_text()
    shutil.copy2(ROOT / "paper" / "main.tex", B / "main.tex")
    (B / "generated").mkdir()
    for g in ("numbers.tex", "numbers_extra.tex", "tables_pf.tex", "table_dist_pf.tex"):
        shutil.copy2(ROOT / "generated" / g, B / "generated" / g)
    for name in set(re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", tex)):
        shutil.copy2(ROOT / "figures" / Path(name).name, B / Path(name).name)
    r = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
                       cwd=B, capture_output=True, text=True, errors="replace")
    log = (B / "main.log").read_text(errors="replace") if (B / "main.log").exists() else ""
    scaled = re.findall(r"\[tabscale\] .*", log)
    print("\n".join(scaled))
    if r.returncode:
        print(r.stdout[-3000:])
        sys.exit("paper: latexmk FAILED (see build/paper/main.log)")
    undefined = re.findall(r"LaTeX Warning: (?:Reference|Citation) .* undefined", log)
    print(f"paper: build/paper/main.pdf written; {len(undefined)} undefined references")
    return 1 if undefined else 0


if __name__ == "__main__":
    sys.exit(main())
