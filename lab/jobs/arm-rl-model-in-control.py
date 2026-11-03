"""Train Sys1 by PPO with Sys1's sampled pick running at every step (plan label A4), with no LLM, inside a Colab kernel. `lab/src/train/ppo.py` holds the method and states every default.

The script runs as one kernel cell (`colab exec -f`), so there is no sys.argv and no __file__, and every argument arrives as an environment variable. The kernel imports only the
standard library and all work runs in child processes, as in `lab/jobs/arm-classification.py`. Failure raises, the cell stops, and `colab_job.sh` sees that its end-of-script line never printed.

Required: SEED. STEP_BUDGET in MODE=full (commands run in training games, Open until a speed test measures the rate).
Modes: MODE=full runs the budget. MODE=speedtest plays one iteration and writes rates and a projection. MODE=smoke is a path check on a few games and its result says so. DRY_RUN=1 prints the plan and stops.
Inputs, by default under WORK: TRAIN_JSONL DEV_JSONL (the export's rows), MENU, TRAIN_GAMES DEV_GAMES (game lists), CODE_TAR, INIT_TAR (the classification arm's checkpoint tar), RESUME_TAR (a state tar pulled from an earlier VM).
Options: INSTALL=0 uses PYTHON, INSTALL_FLA=0 skips Kev's fast kernels, OUT_DIR, INIT_FROM, CODE_DIR, MAX_SECONDS (from the start of this cell), and any field of `Config` in `lab/src/train/ppo.py`: REWARD, FALLBACK, CLIP, LR, KL, and so on.
Outputs under OUT_DIR, with STEM seed<SEED>-<REWARD>: STEM.json, STEM.tar (the final checkpoint with its refit temperature), STEM-state.tar (replaced atomically, for a pull and a resume), STEM-curve.json.
"""
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

T0 = time.time()
ENV = os.environ
TOPIC = "arm-rl-model-in-control"
TRAINER = "train.ppo"
FLA_PINS = ["flash-linear-attention==0.5.2", "triton>=3.7.1"]

SEED = int(ENV["SEED"])
MODE = ENV.get("MODE", "full")
if MODE not in ("full", "speedtest", "smoke"):
    raise ValueError(f"MODE must be full, speedtest or smoke, got {MODE!r}")
WORK = Path(ENV.get("WORK", "/content"))
CODE_TAR = Path(ENV.get("CODE_TAR", WORK / "sys1_code.tar.gz"))
CODE_DIR = Path(ENV.get("CODE_DIR", WORK / "sys1_code"))
OUT_DIR = Path(ENV.get("OUT_DIR", WORK / "out"))
INIT_TAR = Path(ENV.get("INIT_TAR", WORK / "init.tar"))
RESUME_TAR = Path(ENV.get("RESUME_TAR", WORK / "resume.tar"))
INSTALL = ENV.get("INSTALL", "1") == "1"
INSTALL_FLA = ENV.get("INSTALL_FLA", "1") == "1"
DRY_RUN = ENV.get("DRY_RUN", "0") == "1"
MAX_SECONDS = float(ENV.get("MAX_SECONDS", "27000"))   # 7.5 hours of the 8 a VM run may take


def run(cmd, env=None):
    """Run a child, echoing its output as it comes, because `colab exec` shows only what the kernel prints."""
    print("+", " ".join(str(c) for c in cmd), flush=True)
    with subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", env={**ENV, **(env or {})}) as child:
        for line in child.stdout:
            print(line, end="", flush=True)
    if child.returncode:
        raise RuntimeError(f"exit {child.returncode}: {cmd[:4]}")


def install():
    """The steps of `lab/ENVIRONMENT.md`: a separate environment, so Colab's own packages do not matter. No VM has run these yet."""
    python = WORK / "venv" / "bin" / "python"
    if python.exists():
        return python
    run(["bash", "-c", "command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh"])
    uv = shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")
    run([uv, "venv", "--python", "3.12", WORK / "venv"])
    run([uv, "pip", "install", "--python", python, "-r", CODE_DIR / "requirements.txt"])
    if INSTALL_FLA:
        for pin in FLA_PINS:   # separate steps, as `lab/ENVIRONMENT.md` explains
            run([uv, "pip", "install", "--python", python, pin])
    run([WORK / "venv" / "bin" / "alfworld-download"])
    return python


def extract(tar_path, dest):
    with tarfile.open(tar_path) as tar:
        tar.extractall(dest, filter="data")


def inputs():
    """Paths the trainer reads, each from the environment or from its place under WORK."""
    found = {"TRAIN_JSONL": WORK / "train.jsonl.gz", "DEV_JSONL": WORK / "dev.jsonl.gz", "MENU": WORK / "menu.json", "TRAIN_GAMES": WORK / "train_games.txt", "DEV_GAMES": WORK / "dev_games.txt"}
    return {name: ENV.get(name, str(default)) for name, default in found.items()}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if CODE_TAR.exists():
        extract(CODE_TAR, CODE_DIR)
    if not (CODE_DIR / "train" / "ppo.py").exists():
        raise FileNotFoundError(f"no training code under {CODE_DIR}")
    paths = inputs()
    if ENV.get("KL") == "0" and not Path(paths["TRAIN_JSONL"]).is_file():
        paths["TRAIN_JSONL"] = ""   # with the KL term off, this arm reads no supervised rows
    missing = [f"{name}={value}" for name, value in paths.items() if value and not Path(value).is_file()]
    if missing:
        raise FileNotFoundError(f"missing inputs: {missing}")
    resume, init_from = "", ENV.get("INIT_FROM", "")
    if RESUME_TAR.exists():
        extract(RESUME_TAR, OUT_DIR / "resume-from")
        resume = str(OUT_DIR / "resume-from" / "state")
    elif "RESUME_TAR" in ENV:   # a named state that is missing must stop the job, because starting over would look like a resume in the logs of a long run
        raise FileNotFoundError(f"RESUME_TAR={RESUME_TAR} does not exist")
    elif not init_from:
        if not INIT_TAR.exists():
            raise FileNotFoundError(f"RL starts from the classification arm's weights, and there is no {INIT_TAR} and no INIT_FROM")
        extract(INIT_TAR, WORK / "init")
        init_from = str(WORK / "init" / "checkpoint")
    plan = {"topic": TOPIC, "trainer": TRAINER, "seed": SEED, "mode": MODE, "paths": paths, "init_from": init_from, "resume": resume, "out_dir": str(OUT_DIR)}
    if DRY_RUN:
        print(json.dumps(plan), flush=True)
        return
    python = install() if INSTALL else Path(ENV.get("PYTHON", sys.executable))
    extra = ENV.get("PYTHONPATH", "")
    child = {"PYTHONPATH": os.pathsep.join(p for p in (extra, str(CODE_DIR)) if p), "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false", "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1",
             "OUT_DIR": str(OUT_DIR), "INIT_FROM": init_from, "RESUME": resume, "MAX_SECONDS": str(max(0.0, MAX_SECONDS - (time.time() - T0))), **paths}
    run([python, "-m", TRAINER], child)
    print(f"{TOPIC} seed {SEED} {MODE} done in {time.time() - T0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
