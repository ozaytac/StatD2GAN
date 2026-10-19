#!/usr/bin/env python3
"""Post-freeze reprint of Table tab:pilot with typeset headers and rule.

The frozen tab:pilot (generated/tables.tex) prints the protocol's plain-text gate string in a
p{7cm} cell ("(mean tau-MAE of no_D_sorted / mean tau-MAE of full_model - 1) >= 0.20 ...") and the
column name "PositiveSeeds". This script replaces only the tab:pilot table environment of
generated/tables_pf.tex (written by table_seq_extra.py) with a typeset version:
  row      "Remove $\\Dsort$ (Dubai)" with the three frozen cells (Pct, PositiveSeeds, Outcome) unchanged
  headers  "Pct" -> "Change in $\\tau$ MAE (\\%)", "PositiveSeeds" -> "Positive seeds"
  rule     typeset below the tabular; its constants (location, 20 %, 4 of 5 seeds) are read from the
           frozen protocol.yaml and the frozen gate string is asserted to state the same rule
Caption and label are unchanged; every frozen cell is asserted equal before writing. No new numbers.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from post_freeze import register, verify_frozen  # noqa: E402


def block(tab: str, label: str) -> tuple[int, int, str]:
    lab = tab.index(rf"\label{{{label}}}")
    a = tab.rindex(r"\begin{table}", 0, lab)
    b = tab.index(r"\end{table}", lab) + len(r"\end{table}")
    return a, b, tab[a:b]


def cells(blk: str) -> list[str]:
    rows = [r for r in re.findall(r"^(.+?)\\\\\s*$", blk, re.M)]
    assert len(rows) == 2, rows
    return [c.strip() for c in rows[1].split("&")]


def main(root: Path) -> None:
    verify_frozen(root, [root / "generated" / "tables.tex", root / "protocol.yaml"])
    pf_path = root / "generated" / "tables_pf.tex"
    frozen, pf = (root / "generated" / "tables.tex").read_text(), pf_path.read_text()
    fa, fb, fblk = block(frozen, "tab:pilot")
    pa, pb, pblk = block(pf, "tab:pilot")
    assert pblk == fblk or "Positive seeds" in pblk, "tables_pf.tex tab:pilot is neither frozen nor this reprint"
    P = yaml.safe_load((root / "protocol.yaml").read_text())["pilot"]
    pct, pos = P["gate_min_pct"], P["gate_min_positive_seeds"]
    n = len(P["seeds"])
    gate_loc = re.match(r"(\w+):", P["gate"]).group(1)
    expect = (f"{gate_loc}: (mean tau-MAE of no_D_sorted / mean tau-MAE of full_model - 1) >= {pct / 100:.2f} "
              f"AND the paired per-seed difference (no_D_sorted - full_model) is positive in at least {pos} of {n} seeds.")
    assert " ".join(P["gate"].split()) == expect, "frozen gate string differs from the rule typeset here"
    rule_cell, pct_cell, seeds_cell, outcome = cells(fblk)
    assert rule_cell.replace("\\_", "_") == expect, rule_cell
    cap = re.search(r"\\caption\{(.*)\}\n", fblk).group(1)
    tau = r"\overline{\tau\text{-}\mathrm{MAE}}"
    new = "\n".join([
        r"\begin{table}[t]", r"\centering\small", rf"\caption{{{cap}}}", r"\label{tab:pilot}",
        r"\begin{tabular}{lrrl}", r"\toprule",
        r"Gate & Change in $\tau$ MAE (\%) & Positive seeds & Outcome \\", r"\midrule",
        rf"Remove $\Dsort$ ({gate_loc}) & {pct_cell} & {seeds_cell} & {outcome} \\",
        r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
        r"\begin{minipage}{0.9\linewidth}\footnotesize",
        rf"Rule: ${tau}(\texttt{{no\_D\_sorted}})\,/\,{tau}(\texttt{{full\_model}}) - 1 \ge {pct / 100:.2f}$, "
        rf"and the paired per-seed difference \texttt{{no\_D\_sorted}} $-$ \texttt{{full\_model}} is positive "
        rf"in at least {pos} of {n} seeds; means over the {n} pilot seeds at {gate_loc}.",
        r"\end{minipage}", r"\end{table}"])
    out = pf[:pa] + new + pf[pb:]
    assert out[:pa] == pf[:pa] and out[pa + len(new):] == pf[pb:]
    assert cells(new)[1:] == [pct_cell, seeds_cell, outcome]
    pf_path.write_text(out)
    print(f"tab:pilot: cells {pct_cell}, {seeds_cell}, {outcome} equal to frozen; rule checked against protocol.yaml; "
          f"rewrote tab:pilot in {pf_path.relative_to(root)}")
    register(root, [Path(__file__).resolve(), pf_path])


if __name__ == "__main__":
    main(Path(sys.argv[sys.argv.index("--root") + 1] if "--root" in sys.argv else ".").resolve())
