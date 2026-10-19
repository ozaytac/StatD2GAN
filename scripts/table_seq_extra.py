#!/usr/bin/env python3
"""Post-freeze reprint of Table tab:seq with short column headers.

The frozen tab:seq (generated/tables.tex) spells every column as "<metric> (<model>)", so its
natural width is about 2.5 line widths. This script rebuilds the same cells from the frozen
results/seq_level.csv (median over seeds per location and model; real_val has one value per
location) and writes generated/tables_pf.tex: a byte copy of the frozen tables.tex in which
only the tab:seq table environment is replaced. Changes in the replaced block:
  headers  "<metric> (<model>)" -> grouped: metric over model (real, StatD2GAN, TimeGAN, RCGAN)
  caption  unit "(C)" -> "(K)" (paper text and Fig. 5 use K; SDs are equal in C and K)
Label and the rest of the caption text are unchanged. Every cell is asserted equal to the frozen
cell before writing. No new inference and no new numbers.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from post_freeze import register, verify_frozen  # noqa: E402

METRICS = ("seq_mean_sd", "within_sd")
MODELS = (("real_val", "real"), ("statd2gan", "StatD2GAN"), ("timegan", "TimeGAN"), ("rcgan", "RCGAN"))


def frozen_block(tab: str) -> tuple[int, int, str]:
    """Start, end and text of the table environment that carries \\label{tab:seq}."""
    lab = tab.index(r"\label{tab:seq}")
    a = tab.rindex(r"\begin{table}", 0, lab)
    b = tab.index(r"\end{table}", lab) + len(r"\end{table}")
    return a, b, tab[a:b]


def frozen_cells(block: str) -> dict[tuple[str, str, str], str]:
    """(loc, metric, model) -> cell string, parsed from the frozen '<metric> (<model>)' header."""
    rows = [r.strip() for r in re.findall(r"^(.+?)\\\\\s*$", block, re.M)]
    head = [h.strip().replace("\\_", "_") for h in rows[0].split("&")]
    cols = [re.fullmatch(r"(\w+) \((\w+)\)", h).groups() for h in head[1:]]
    out = {}
    for r in rows[1:]:
        c = [x.strip() for x in r.split("&")]
        for (met, mod), v in zip(cols, c[1:]):
            out[(c[0], met, mod)] = v
    return out


def main(root: Path) -> None:
    ins = [root / "results" / "seq_level.csv", root / "generated" / "tables.tex", root / "protocol.yaml"]
    verify_frozen(root, ins)
    tab = (root / "generated" / "tables.tex").read_text()
    a, b, old = frozen_block(tab)
    locs = list(yaml.safe_load((root / "protocol.yaml").read_text())["data"]["locations"])
    sq = pd.read_csv(root / "results" / "seq_level.csv", comment="#")
    med = sq.groupby(["loc", "model"])[list(METRICS)].median()

    cells = {(l.capitalize(), met, mod): f"{med.loc[(l, mod), met]:.2f}" for l in locs for met in METRICS for mod, _ in MODELS}
    froz = frozen_cells(old)
    assert set(cells) == set(froz), "cell sets differ from frozen tab:seq"
    bad = {k: (cells[k], froz[k]) for k in cells if cells[k] != froz[k]}
    assert not bad, f"cells differ from frozen tab:seq: {bad}"

    cap = re.search(r"\\caption\{(.*)\}\n", old).group(1)
    assert cap.count("(C)") == 1
    cap = cap.replace("(C)", "(K)")
    n = len(MODELS)
    lines = [r"\begin{table}[t]", r"\centering\small", rf"\caption{{{cap}}}", r"\label{tab:seq}",
             r"\begin{tabular}{l" + "r" * (n * len(METRICS)) + "}", r"\toprule",
             " & ".join(["Location"] + [rf"\multicolumn{{{n}}}{{c}}{{{m.replace('_', chr(92) + '_')}}}" for m in METRICS]) + r" \\",
             "".join(rf"\cmidrule(lr){{{2 + i * n}-{1 + (i + 1) * n}}}" for i in range(len(METRICS))),
             " & ".join([""] + [lab for _ in METRICS for _, lab in MODELS]) + r" \\", r"\midrule"]
    for l in locs:
        L = l.capitalize()
        lines.append(" & ".join([L] + [cells[(L, met, mod)] for met in METRICS for mod, _ in MODELS]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    new = "\n".join(lines)

    out = root / "generated" / "tables_pf.tex"
    pf = tab[:a] + new + tab[b:]
    assert pf[:a] == tab[:a] and pf[a + len(new):] == tab[b:]
    out.write_text(pf)
    print(f"tab:seq: {len(cells)} cells equal to frozen; wrote {out.relative_to(root)}")
    register(root, [Path(__file__).resolve(), out])


if __name__ == "__main__":
    main(Path(sys.argv[sys.argv.index("--root") + 1] if "--root" in sys.argv else "."))
