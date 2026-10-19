"""runs/_env/<host>.json: package versions (pip freeze), hardware, git commit, protocol hash."""
import json, platform, socket, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s2g import config

out = config.ROOT / "runs/_env"; out.mkdir(parents=True, exist_ok=True)
sh = lambda *a: subprocess.run(a, capture_output=True, text=True).stdout.strip()
info = dict(host=socket.gethostname(), python=sys.version, platform=platform.platform(),
            pip_freeze=sh(sys.executable, "-m", "pip", "freeze").split("\n"),
            nvidia_smi=sh("nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"),
            git_commit=sh("git", "-C", str(config.ROOT), "rev-parse", "HEAD"), protocol_sha256=config.sha())
try:
    import torch
    info.update(torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version())
except ImportError:
    pass
(out / f"{info['host']}.json").write_text(json.dumps(info, indent=1))
print("wrote", out / f"{info['host']}.json")
