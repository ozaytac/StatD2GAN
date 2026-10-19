"""Fail if the manuscript contains hand-typed statistics. Numbers must come from generated/numbers.tex macros.
usage: lint_paper.py paper/main.tex"""
import re, sys
bad = []
pats = [r"\b\d+\.\d+\b", r"\bp\s*[<=>]\s*0?\.\d", r"\d\s*\\?%", r"\b\d+(\.\d+)?\s*x\b"]
for i, line in enumerate(open(sys.argv[1]), 1):
    t = re.sub(r"%.*$", "", line)
    t = re.sub(r"\\(input|include|label|cite\w*|ref|eqref|includegraphics|bibliography\w*|usepackage|documentclass)(\[[^\]]*\])?\{[^}]*\}", "", t)
    t = re.sub(r"\\[A-Za-z]+\{[0-9.\\a-z]*(cm|pt|mm|em|textwidth|linewidth)\}", "", t)
    if any(re.search(p, t) for p in pats):
        bad.append((i, line.strip()[:110]))
for i, l in bad: print(f"{sys.argv[1]}:{i}: {l}")
sys.exit(1 if bad else 0)
