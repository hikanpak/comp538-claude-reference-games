"""What a result file must carry: git commit, package versions, hardware. Files written before the parent session commits name the commit they started from and say `git_dirty`."""
from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PACKAGES = ["alfworld", "textworld", "fast-downward-textworld", "kev", "torch", "transformers", "peft", "numpy"]


def _text(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip()


def git_state() -> dict:
    status = _text(["git", "status", "--porcelain", "--", "lab/src"])
    return {"git_commit": _text(["git", "rev-parse", "HEAD"]), "git_dirty_lab_src": bool(status)}


def source_digest() -> dict:
    """SHA-256 of the code that made this file, because the parent session commits it afterwards."""
    files = sorted((ROOT / "lab/src/dataset").glob("*.py")) + [ROOT / "lab/src/menu.py", ROOT / "lab/src/alf/env.py", ROOT / "lab/src/harness/prompt.py", ROOT / "lab/src/harness/control.py", ROOT / "lab/src/train/export_kev.py"]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def packages() -> dict:
    out = {"python": platform.python_version()}
    for name in PACKAGES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def hardware() -> dict:
    return {"machine": platform.machine(), "platform": platform.platform(), "cpu": _text(["sysctl", "-n", "machdep.cpu.brand_string"]), "cores": int(_text(["sysctl", "-n", "hw.ncpu"]) or 0),
            "memory_bytes": int(_text(["sysctl", "-n", "hw.memsize"]) or 0)}


def provenance() -> dict:
    return {**git_state(), "packages": packages(), "hardware": hardware(), "source_sha256": source_digest()}


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
