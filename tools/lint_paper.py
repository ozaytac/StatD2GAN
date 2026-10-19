#!/usr/bin/env python3
"""lint_paper.py: static checks for main.tex against the project rules.

Usage: python lint_paper.py main.tex [--until "\\section{Discussion}"] [--base accepted.tex] [--macros generated/numbers.tex generated/numbers_extra.tex]
Exit code 1 if any ERROR is found. WARN lines need a human decision.

ERROR  retracted claim / banned token, em-dash, American spelling, undefined number macro
WARN   numeric literal with a decimal point outside a macro (hard-coded number)
"""
import re, sys
from pathlib import Path

# Number-macro naming rule, shared with macro_check.py (import FAM from here).
# Family prefix followed by an upper-case letter or digit; LaTeX built-ins such as
# \phi, \cite, \medskip, \circledast, \pi, \pm never match, so no allow-list is needed.
FAM = re.compile(r"(?:(?:pHolm|pRaw|pilot|gpu|med|ks|pct|ph|proto|code|ci|dKs|phys|runNoise|seq|verdict|Within)(?=[A-Z0-9])|n[A-Z]|Seq)\w*")


def main() -> None:
    args = sys.argv[1:]
    tex = Path(args[0]).read_text()
    macro_files = []
    if "--macros" in args:
        for a in args[args.index("--macros") + 1:]:
            if a.startswith("--"): break
            macro_files.append(a)
    base = args[args.index("--base") + 1] if "--base" in args else None   # accepted .tex: its macros are known-good
    until =args[args.index("--until") + 1] if "--until" in args else None   # e.g. --until "\\section{Discussion}"
    if until:                                   # lint only the text before this marker (sections being worked on)
        cut = tex.find(until)
        assert cut > 0, f"--until marker not found: {until}"
        tex = tex[:cut]

    # strip comments, keep line numbers
    lines = [re.sub(r"(?<!\\)%.*", "", l) for l in tex.splitlines()]
    body_start = next((i for i, l in enumerate(lines) if r"\begin{frontmatter}" in l), 0)

    RETRACTED = {  # pattern -> why (all from the project's retracted-claims list)
        r"0\.080\b": "v2 tau effect",
        r"0\.085\b": "v2 run-to-run noise (v3: macro \\runNoiseDtau)",
        r"\b115\s*\\?%": "v2 Ankara~0 vs >115% regime claim",
        r"regime[- ]dependent": "retracted regime dependence",
        r"climate[- ]dependent specialis": "retracted",
        r"negligible": "D5 failed; no 'negligible cost'",
        r"\\Delta\s*\$?\s*\\?(mathrm\{)?KS\}?\s*\\leq?\s*0\.003": "dKS <= 0.003 retracted",
        r"trade-?off": "D_temporal trade-off retracted (check context)",
        r"insurance": "lambda-insurance retracted",
        r"discriminative score": "retracted",
        r"\b0\.500\b": "discriminative score 0.500 retracted",
        r"earlier (version|campaign)|previous (version|campaign)|August 2026|\bv2\b": "old versions are not mentioned",
        r"post hoc": "Lhasa 'added post hoc' retracted",
        r"essentially all": "overclaim on violations",
        r"within 2\s*\\?%": "KS within 2% retracted (median ratio is a macro)",
        r"collaps": "TimeGAN collapse retracted; allowed only as 'criterion ... not met' (check context)",
        r"isolates the mechanism|shows that the channel": "D4 is 'consistent with' only",
        r"deliberately weak": "baseline is 'under-performing'",
        r"distinct (K\\\"\{o\}ppen|climate classes)": "Ankara and Lhasa are both BSk",
        r"outperform|state[- ]of[- ]the[- ]art|superior to": "no superiority claim (allowed only in a negation, check)",
    }
    CONTEXT_OK = {  # token -> regexes that make the hit acceptable
        "collaps": [r"criterion", r"not met", r"not observed", r"we collapse", r"collapse onto"],
        "trade-?off": [r"bounded", r"no .*trade"],
        "outperform": [r"no claim"], "superior to": [r"no claim"],
        "state[- ]of[- ]the[- ]art": [r"no claim"],
    }
    AMERICAN_OK = {"visualization"}   # CRediT role name
    AMERICAN = r"\b(behavior|color|normaliz\w+|analyz\w+|generaliz\w+|optimiz\w+|regulariz\w+|visualiz\w+|summariz\w+|characteriz\w+|initializ\w+|standardiz\w+|favor|center|modeling|labeled|parameteriz\w+)\b"

    err, warn = [], []
    # reference titles keep their source spelling: the spelling rule (only) skips thebibliography
    bib = [i for i, l in enumerate(lines, 1) if r"\begin{thebibliography}" in l or r"\end{thebibliography}" in l]
    in_bib = (lambda i: bib[0] < i < bib[-1]) if len(bib) == 2 else (lambda i: False)
    for i, l in enumerate(lines, 1):
        if i < body_start:
            continue
        if "\u2014" in l or "---" in l:
            err.append((i, "em-dash", l.strip()[:90]))
        for pat, why in RETRACTED.items():
            m = re.search(pat, l, re.I)
            if m:
                key = next((k for k in CONTEXT_OK if re.search(k, pat)), None)
                window = " ".join(lines[max(0, i - 2): i + 2])
                if key and any(re.search(c, window, re.I) for c in CONTEXT_OK[key]):
                    continue
                err.append((i, why, l.strip()[:90]))
        m = re.search(AMERICAN, l, re.I)
        if m and m.group(0).lower() in AMERICAN_OK:
            m = None
        if m and not in_bib(i) and not re.search(r"\\(texttt|label|ref|cite)", l[max(0, m.start() - 12):m.start()]):
            err.append((i, f"American spelling '{m.group(0)}'", l.strip()[:90]))

    # hard-coded decimals outside math-free macro calls (WARN)
    for i, l in enumerate(lines, 1):
        if i < body_start or any(x in l for x in (r"\newcommand", r"\includegraphics", "linewidth", "textwidth")):
            continue
        t = re.sub(r"\\[A-Za-z]+\{[^}]*\}", "", l)                  # drop macro arguments
        t = re.sub(r"\\[A-Za-z]+", "", t)
        for m in re.finditer(r"(?<![\w.])\d+\.\d{2,}(?![\w.])", t):
            warn.append((i, f"hard-coded number {m.group(0)}", l.strip()[:90]))

    # undefined numeric macros
    defs = {}
    for f in macro_files:
        defs.update(dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*$", Path(f).read_text(), re.M)))
    have_frozen = any(Path(f).name.endswith("numbers.tex") and not Path(f).name.endswith("extra.tex") for f in macro_files)
    if defs:
        own = set(re.findall(r"\\newcommand\{?\\(\w+)", tex))
        called = set(re.findall(r"\\([A-Za-z]+)", "\n".join(lines[body_start:])))
        # number-macro naming rule: family prefix followed by an upper-case letter or digit
        # (\phKsRatio..., \ksFloor..., \pHolm..., \pRawNoDSortedTau, \pilotPct, \gpuHours);
        # \phi, \cite, \medskip, \circledast, \pi, \pm do not match
        fam = FAM
        for k in sorted(k for k in defs if not fam.fullmatch(k)):   # self-test: a defined macro outside fam is never checked
            warn.append((0, "defined macro outside number-macro family (undefined use not detectable)", k))
        known = set()
        if base:                                  # base text widens what is known ONLY for commands outside fam;
            known = {k for k in re.findall(r"\\([A-Za-z]+)", re.sub(r"(?<!\\)%.*", "", Path(base).read_text()))
                     if not fam.fullmatch(k)}     # a fam command is known only through a definition in the macro files
        LAYOUT_OK = {"FloatBarrier", "clearpage", "newpage"}   # layout commands, never number macros
        for k in sorted(called - own - known - LAYOUT_OK):
            if k in defs:
                continue
            if fam.fullmatch(k):                  # only the number-macro family can be "undefined"
                (err if have_frozen else warn).append((0, "undefined macro" + ("" if have_frozen else " (numbers.tex not supplied)"), k))
        unused = sorted(set(defs) - called) if len(macro_files) == 1 else []
        for k in unused:
            warn.append((0, "macro defined but never called", k))

    for tag, rows in (("ERROR", err), ("WARN ", warn)):
        for ln, why, txt in rows:
            print(f"{tag} line {ln}: {why} | {txt}")
    print(f"\n{len(err)} errors, {len(warn)} warnings")
    sys.exit(1 if err else 0)


if __name__ == "__main__":
    main()
