"""Consistency checks between the paper's macro files and the frozen v3 outputs.

1. numbers.tex and tables.tex sha256 == MANIFEST.json (frozen layer untouched).
2. numbers.tex and numbers_extra.tex define disjoint macro sets.
3. pHolmRel<X> == frozen pHolm<X> with a relation prefix.
4. Baseline family (2 contrasts x 4 metrics):
   frozen results/baseline_tests.csv == frozen tables.tex (tab:baselines), and every
   pHolmBase*/medBase*/nLocPosBase* macro in numbers_extra.tex == that table.
5. Rule 4: every numbers_extra macro is called by main.tex; every macro main.tex calls
   from either file is defined. The number-macro family is lint_paper.FAM (one regex for
   both scripts); LaTeX built-ins never match it, so no allow-list is kept here.
6. Post-freeze generated/tables_pf.tex (main.tex inputs it instead of tables.tex):
   byte-identical to the frozen tables.tex outside the tab:seq and tab:pilot table environments;
   in tab:pilot the result cells (Pct, PositiveSeeds, Outcome) and the caption equal the frozen
   ones (scripts/table_pilot_extra.py); in tab:seq
   every (location, metric, model) cell equals the frozen cell, the label is unchanged and
   the caption differs only in the unit "(C)" -> "(K)".
7. Post-freeze generated/table_dist_pf.tex (tab:dist, Appendix B) and the phDist* macros:
   results_extra/distance_cal_train.csv sha256 == MANIFEST_post_freeze.json; every gate and
   triangle check in it passed; every tab:dist cell and every numeric phDist* macro is
   recomputed here from the CSV (independently of report_extra.py) and must match.
"""
import csv, hashlib, json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lint_paper import FAM  # noqa: E402  (shared number-macro naming rule)

root = Path(sys.argv[1] if len(sys.argv) > 1 else '.')
repo, md = root / 'repo', root / 'md'
man = json.loads((repo / 'MANIFEST.json').read_text())['files']
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
for rel in ('generated/numbers.tex', 'generated/tables.tex', 'results/baseline_tests.csv'):
    assert sha(repo / rel) == man[rel], f'{rel} sha mismatch'
assert (md / 'generated/numbers.tex').read_bytes() == (repo / 'generated/numbers.tex').read_bytes()

defs = lambda t: dict(re.findall(r'\\newcommand\{\\(\w+)\}\{(.*)\}\s*$', t, re.M))
N, E = defs((repo / 'generated/numbers.tex').read_text()), defs((md / 'generated/numbers_extra.tex').read_text())
assert not set(N) & set(E), set(N) & set(E)

# 3 -- relation copies
n_rel = 0
for k, v in E.items():
    if k.startswith('pHolmRel'):
        base = 'pHolm' + k[len('pHolmRel'):]
        assert v == (N[base] if N[base][:1] in '<>=' else '=' + N[base]), (k, v, N[base]); n_rel += 1

# 4 -- baseline family
tab = (repo / 'generated/tables.tex').read_text()
blk = tab[tab.index(r'\label{tab:baselines}'):]
blk = blk[:blk.index(r'\end{tabular}')]
T = {}
for row in re.findall(r'^(\w[\w\\]*) & (rcgan|timegan) & (\S+) & (\S+) & (\S+) & (.+?) \\\\$', blk, re.M):
    met, con, med, p, ph, pos = row
    T[(met.replace('\\_', '_'), con)] = dict(med=float(med), p=p, ph=ph, pos=pos.strip())
assert len(T) == 8, T.keys()
fp = lambda x: '<0.001' if x < 0.001 else f'{x:.3f}'           # tables.tex p formatting
rows = [r for r in csv.DictReader(l for l in open(repo / 'results/baseline_tests.csv') if not l.startswith('#'))]
assert len(rows) == 8
for r in rows:
    t = T[(r['metric'], r['contrast'])]
    assert fp(float(r['p_wilcoxon'])) == t['p'] and fp(float(r['p_holm'])) == t['ph'], (r, t)
    assert f"{float(r['median_diff']):.4f}" == f"{t['med']:.4f}" and r['pos_by_loc'] == t['pos'], (r, t)
    assert r['family_size'] == '2', r
MET = {'Ks': 'ks_stat', 'Tau': 'kendall_mae', 'Acf': 'acf_mae', 'Phys': 'phys_viol_rate'}
CSV = {(r['metric'], r['contrast']): r for r in rows}
n_base = 0
for k, v in E.items():
    m = re.fullmatch(r'(pHolm|med|nLocPos)Base(Rcgan|Timegan)(Ks|Tau|Acf|Phys)', k)
    if not m:
        continue
    kind, con, met = m.group(1), m.group(2).lower(), MET[m.group(3)]
    t, r = T[(met, con)], CSV[(met, con)]
    if kind == 'pHolm':
        assert v == (t['ph'] if t['ph'][0] == '<' else '=' + t['ph']), (k, v, t['ph'])
    elif kind == 'med':
        dp = len(v.split('.')[1]) if '.' in v else 0
        assert abs(float(v) - float(r['median_diff'])) <= 0.5 * 10 ** -dp + 1e-12, (k, v, r['median_diff'])
    else:
        assert v == r['n_loc_median_pos'], (k, v, r['n_loc_median_pos'])
    n_base += 1

# 5 -- Rule 4 coverage
called = set(re.findall(r'\\([A-Za-z]+)', (md / 'main.tex').read_text()))
unused = sorted(set(E) - called)
assert not unused, f'numbers_extra macros not called by main.tex: {unused}'
undefined = sorted(k for k in called if FAM.fullmatch(k) and k not in N and k not in E)
assert not undefined, f'called but undefined: {undefined}'

# 6 -- post-freeze tab:seq reprint
pf = (md / 'generated/tables_pf.tex').read_text()
mt = (md / 'main.tex').read_text()
assert r'\input{generated/tables_pf.tex}' in mt and r'\input{generated/tables.tex}' not in mt, 'main.tex must input tables_pf.tex only'
def env(t, label='tab:seq'):
    lab = t.index(r'\label{' + label + '}')
    a = t.rindex(r'\begin{table}', 0, lab); b = t.index(r'\end{table}', lab) + len(r'\end{table}')
    return a, b, t[a:b]
seq_env = env
fa, fb, fblk = seq_env(tab); pa, pb, pblk = seq_env(pf)
qa, qb, qblk = env(tab, 'tab:pilot'); ra, rb, rblk = env(pf, 'tab:pilot')
assert fb < qa and pb < ra, 'expected tab:seq before tab:pilot'
assert pf[:pa] == tab[:fa] and pf[pb:ra] == tab[fb:qa] and pf[rb:] == tab[qb:], \
    'tables_pf.tex differs from frozen tables.tex outside tab:seq and tab:pilot'
row = lambda b: [c.strip() for c in re.findall(r'^(.+?)\\\\\s*$', b, re.M)[1].split('&')]
assert row(qblk)[1:] == row(rblk)[1:], ('tab:pilot cells differ', row(qblk)[1:], row(rblk)[1:])
pcap = lambda b: re.search(r'\\caption\{(.*)\}\n', b).group(1)
assert pcap(qblk) == pcap(rblk), 'tab:pilot caption changed'
cap = lambda b: re.search(r'\\caption\{(.*)\}\n', b).group(1)
assert cap(fblk).replace('(C)', '(K)') == cap(pblk) and cap(fblk).count('(C)') == 1, 'caption changed beyond the unit'
body = lambda b: [[c.strip() for c in r.split('&')] for r in re.findall(r'^(.+?)\\\\\s*$', b, re.M)]
F = body(fblk); fcols = [re.fullmatch(r'(\w+) \((\w+)\)', h.replace('\\_', '_')).groups() for h in F[0][1:]]
fcell = {(r[0], m, o): v for r in F[1:] for (m, o), v in zip(fcols, r[1:])}
P = body(pblk)
groups = [re.fullmatch(r'\\multicolumn\{(\d+)\}\{c\}\{(.+)\}', h).groups() for h in P[0][1:]]
pmet = [g[1].replace('\\_', '_') for g in groups for _ in range(int(g[0]))]
NAME = {'real': 'real_val', 'StatD2GAN': 'statd2gan', 'TimeGAN': 'timegan', 'RCGAN': 'rcgan'}
pmod = [NAME[h] for h in P[1][1:]]
pcell = {(r[0], m, o): v for r in P[2:] for m, o, v in zip(pmet, pmod, r[1:])}
assert set(pcell) == set(fcell) and len(fcell) == 40, (len(fcell), set(pcell) ^ set(fcell))
diff = {k: (fcell[k], pcell[k]) for k in fcell if fcell[k] != pcell[k]}
assert not diff, f'tab:seq cells differ: {diff}'

# 7 -- post-freeze tab:dist and phDist* macros, recomputed from the CSV
import numpy as np, pandas as pd
pfm = json.loads((repo / 'MANIFEST_post_freeze.json').read_text())['files']
dc = repo / 'results_extra/distance_cal_train.csv'
assert sha(dc) == pfm['results_extra/distance_cal_train.csv'], 'distance CSV sha mismatch'
assert r'\input{generated/table_dist_pf.tex}' in mt, 'main.tex must input table_dist_pf.tex'
dz = pd.read_csv(dc, comment='#')
assert dz.gate_ok.all() and dz.triangle_ok.all() and len(dz) == 250
gap = dz.ks_cal_val - dz.ks_floor; r = dz.d_cal_train / dz.ks_floor; g = gap.abs() / dz.ks_floor
bl = gap < 0; base = dz.config.isin(['rcgan', 'timegan'])
assert np.allclose(np.abs(gap), np.minimum(np.abs(gap), dz.d_cal_train))        # Proposition on every run
f3, f4 = (lambda x: f'{x:.3f}'), (lambda x: f'{x:.4f}')
want = dict(phDistNRuns=str(len(dz)), phDistRatioMed=f3(r.median()), phDistRatioPninety=f3(np.quantile(r, 0.9)),
            phDistGapMed=f3(g.median()), phDistGapOverDMed=f3((gap.abs() / dz.d_cal_train).median()),
            phDistBelowMaxRatio=f3((-gap[bl] / dz.d_cal_train[bl]).max()), phDistBelowAll=str(int(bl.sum())),
            phDistTopOneRatio=f3(r.max()), phDistGateMaxDiff=f'{max(dz.gate_raw_diff.max(), dz.gate_cal_diff.max()):g}')
assert int((bl & ~base).sum()) == int(E['phKsRatioCellsBelowOne']), 'below-floor arm count'
for k, v in want.items():
    if k in E: assert E[k] == v, (k, E[k], v)
n_dm = sum(k in E for k in want)
dt = (md / 'generated/table_dist_pf.tex').read_text()
trows = {c[0]: c[1:] for c in ([x.strip() for x in row.split('&')] for row in re.findall(r'^(.+?)\\\\\s*$', dt, re.M)) if c[0]}  # header rows start with an empty cell
grp = [(l.capitalize(), dz[~base & (dz['loc'] == l)]) for l in ['ankara', 'bergen', 'dubai', 'lhasa', 'yakutsk']] + \
      [('RCGAN', dz[dz.config == 'rcgan']), ('TimeGAN', dz[dz.config == 'timegan']), ('All runs', dz)]
assert list(trows) == [k for k, _ in grp], list(trows)
for lab, x in grp:
    rx = x.d_cal_train / x.ks_floor
    exp = [str(len(x)), f4(x.ks_floor.iloc[0]) if x['loc'].nunique() == 1 else '--', f4(x.d_cal_train.median()),
           f3(rx.median()), f3(rx.max()), f3(((x.ks_cal_val - x.ks_floor).abs() / x.ks_floor).median()), f3(x.d_raw_train.median())]
    assert trows[lab] == exp, (lab, trows[lab], exp)

print(f'frozen sha256 OK (numbers.tex, tables.tex, baseline_tests.csv)')
print(f'{n_rel} pHolmRel copies == frozen pHolm values')
print(f'baseline_tests.csv == tables.tex tab:baselines (8 rows); {n_base} baseline macros == table')
print(f'Rule 4: {len(E)} numbers_extra macros, all called; no undefined macro calls')
print(f'tables_pf.tex == tables.tex outside tab:seq/tab:pilot; tab:seq {len(pcell)} cells == frozen, caption differs only (C)->(K); tab:pilot cells {row(rblk)[1:]} == frozen')
print(f'tab:dist {len(grp)} rows x 7 cells == recomputed from distance_cal_train.csv (sha OK); {n_dm} phDist macros == recomputed; Proposition holds on all {len(dz)} runs')
