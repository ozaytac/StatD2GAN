"""Integration: fake runs -> make_* -> analyze -> figures -> report, in a temporary copy of the package.
Checks the CPU layers end to end (no torch needed)."""
import json, os, shutil, subprocess, sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))


def sh(cwd, *cmd):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(cwd / "src"), os.environ.get("PYTHONPATH", "")]))
    r = subprocess.run([sys.executable, *cmd], cwd=cwd, capture_output=True, text=True, env=env)
    assert r.returncode == 0, f"{cmd}\n{r.stdout[-1500:]}\n{r.stderr[-2500:]}"
    return r.stdout


@pytest.fixture(scope="module")
def pkg(tmp_path_factory):
    import fake_runs, yaml
    d = tmp_path_factory.mktemp("pkg") / "p"
    shutil.copytree(ROOT, d, ignore=shutil.ignore_patterns("runs", "results", "figures", "generated", "logs", "__pycache__", "data", ".git", "dist", "MANIFEST.json"))
    (d / "protocol.lock").unlink(missing_ok=True)      # the copy is frozen on its own below
    for sub in ("runs", "results", "figures", "generated"): (d / sub).mkdir()
    (d / "data").mkdir(); shutil.copytree(ROOT / "data/raw", d / "data/raw"); (d / "data/processed").mkdir()
    p = yaml.safe_load((d / "protocol.yaml").read_text()); p["status"], p["budget"]["gpu_hours"] = "FROZEN", 100
    (d / "protocol.yaml").write_text(yaml.safe_dump(p, sort_keys=False)); sh(d, "scripts/freeze.py")
    sh(d, "scripts/preprocess.py"); sh(d, "scripts/make_floor.py")
    from s2g.config import LOCS, ARMS, SEEDS
    import pandas as pd
    fl = pd.read_csv(d / "results/ks_floor.csv", comment="#").groupby("loc").ks_floor.mean().to_dict()
    fake_runs.write(d, LOCS, ARMS, SEEDS, fl)
    (d / "runs/pilot_gate.json").write_text(json.dumps(dict(pct=31.0, positive_seeds=5, n_seeds=5, passed=True)))
    return d


def test_full_cpu_chain(pkg):
    for s in ("make_abl_perseed", "make_baselines", "make_projection", "make_seq_level", "analyze", "figures", "report"):
        sh(pkg, f"scripts/{s}.py")
    res = pkg / "results"
    for f in ("abl_perseed", "baselines", "projection", "seq_level", "ks_ratio", "summary", "pct_change", "wilcoxon", "bootstrap", "baseline_tests", "run_noise"):
        assert (res / f"{f}.csv").read_text().startswith("# produced by scripts/"), f
    assert len(open(res / "abl_perseed.csv").readlines()) == 2 + 5 * 9 * 5   # header comment + columns + rows
    dec = json.loads((res / "decisions.json").read_text())
    assert dec["D1"]["holds"] and dec["D2"]["no_D_sorted"]["holds"] and not dec["D2"]["no_D_stat"]["holds"]
    assert dec["D4"]["holds"] and dec["D5"]["holds"] and dec["D6"]["timegan"]["holds"] and not dec["D6"]["rcgan"]["holds"]
    for n in ("fig1_ks_floor", "fig2_dsort_mechanism", "fig3_injection_projection", "fig4_variance_decomposition"):
        assert (pkg / f"figures/{n}.pdf").stat().st_size > 5000
    nums = (pkg / "generated/numbers.tex").read_text()
    assert r"\newcommand{\nPairs}{25}" in nums and "verdictComponentNoDSorted}{effective}" in nums
    tex = (pkg / "generated/tables.tex").read_text()
    assert "not a mathematical lower bound" in tex and "Holm" in tex and tex.count(r"\begin{table}") == tex.count(r"\end{table}")


def test_write_once_is_enforced(pkg):
    r = subprocess.run([sys.executable, "scripts/analyze.py"], cwd=pkg, capture_output=True, text=True,
                       env=dict(os.environ, PYTHONPATH=str(pkg / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")))
    assert r.returncode != 0 and "written once" in (r.stderr + r.stdout)


def test_frozen_protocol_edit_is_refused(pkg):
    (pkg / "protocol.yaml").write_text((pkg / "protocol.yaml").read_text() + "\n# tampered\n")
    r = subprocess.run([sys.executable, "scripts/make_floor.py"], cwd=pkg, capture_output=True, text=True,
                       env=dict(os.environ, PYTHONPATH=str(pkg / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")))
    assert r.returncode != 0 and "changed after freeze" in (r.stderr + r.stdout)
