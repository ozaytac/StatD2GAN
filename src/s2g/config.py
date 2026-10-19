"""Single source of truth: protocol.yaml. Frozen (sha256 lock) before any run."""
import hashlib, json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
PROTO, LOCK = ROOT / "protocol.yaml", ROOT / "protocol.lock"


def load() -> dict:
    return yaml.safe_load(PROTO.read_text())


def sha() -> str:
    return hashlib.sha256(PROTO.read_bytes()).hexdigest()


def assert_frozen() -> str:
    """Every run/analysis script calls this first."""
    if not LOCK.exists():
        raise SystemExit("protocol not frozen: run `make freeze`")
    if json.loads(LOCK.read_text())["sha256"] != sha():
        raise SystemExit("protocol.yaml changed after freeze; refusing to run")
    return sha()


P = load()
LOCS = list(P["data"]["locations"])
ARMS = list(P["arms"])
SEEDS = P["seeds"]
FEATS = P["data"]["features"]
SC_IDX = P["data"]["scalable_idx"]
PHY = P["physics"]


def header(script: str, inputs: list[str]) -> str:
    """Provenance header written at the top of every derived CSV (read with comment='#')."""
    return f"# produced by scripts/{script}; inputs: {', '.join(inputs)}; protocol sha256 {sha()[:12]}\n"


def write_csv(df, path, script, inputs):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if Path(path).exists():
        raise SystemExit(f"{path} exists: intermediates are written once (make clean-<step> to redo)")
    with open(path, "w") as f:
        f.write(header(script, inputs))
        df.to_csv(f, index=False)


def read_csv(path):
    import pandas as pd
    return pd.read_csv(path, comment="#")
