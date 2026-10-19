"""runs/pilot_gate.json: evaluate the pre-declared gate on the pilot runs (values only)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s2g import decide, runs
from s2g.config import P, ROOT, assert_frozen

assert_frozen()
out = ROOT / "runs/pilot_gate.json"
if out.exists():
    sys.exit("pilot_gate.json exists (written once)")
r = runs.load_runs(names=P["pilot"]["arms"], locs=["dubai"])
assert len(r) == len(P["pilot"]["arms"]) * len(P["pilot"]["seeds"]), f"Dubai pilot incomplete ({len(r)} runs)"
g = decide.pilot_gate(r); g["rule"] = P["pilot"]["gate"]
out.write_text(json.dumps(g, indent=1)); print(g)
