"""Train Sys1 by classification on the expert's commands, inside a Colab kernel: `kev.train --init_from`, then a temperature refit on dev rows.

The script runs as one kernel cell (`colab exec -f`), so there is no sys.argv and no __file__, and every argument arrives as an
environment variable. The kernel itself imports only the standard library: pip may replace numpy or torch, and a kernel
holding the old modules would break, so all work runs in child processes. Failure raises, the cell stops, and
`colab_job.sh` sees that its end-of-script line never printed. No result file is written on failure, because
`lab/tools/check_gates.py trained` counts any JSON with a seed in `lab/results/arm-classification/`.

Required: SEED, MAX_STATE (`max_state_needed` from the export's report: kev.train drops rows over its limit, and those are the
late steps of long games).
Modes: MODE=full trains on every row. MODE=speedtest trains on SPEED_ROWS rows and only measures. MODE=smoke is a Mac check of
this file on a few rows. DRY_RUN=1 validates the data, prints the kev.train command and stops.
Optional, with defaults below: TRAIN_JSONL DEV_JSONL CODE_TAR CODE_DIR OUT_DIR EPOCHS LR BATCH ACCUM DTYPE DEVICE INIT_FROM BASE BASE_REVISION SPEED_ROWS
PARTIAL_EVERY_MIN EVAL_BATCH INSTALL INSTALL_FLA GIT_COMMIT GIT_DIRTY.
"""
import gzip
import json
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

T0 = time.time()
PY = sys.executable   # install() replaces this with the separate environment
ENV = os.environ
KEV_SOURCE = "git+https://github.com/jaredpalmer/kev@2074cc44256200bb37540103fa4572901faa10f3"   # the commit in the Mac environment
# The Mac's versions (lab/requirements.txt). Kev's bound is kev-0.1.0.dist-info/METADATA, and the kernels' pin is kev/fused_qwen35.py:53.
PINS = ["torch==2.8.0", "transformers==5.22.1", "peft==0.21.2", "numpy==2.5.4"]
FLA_PINS = ["flash-linear-attention==0.5.2", "triton>=3.7.1"]

SEED = int(ENV["SEED"])
MODE = ENV.get("MODE", "full")
if MODE not in ("full", "speedtest", "smoke"):
    raise ValueError(f"MODE must be full, speedtest or smoke, got {MODE!r}")
MAX_STATE = int(ENV["MAX_STATE"])
WORK = Path(ENV.get("WORK", "/content"))
TRAIN_JSONL = Path(ENV.get("TRAIN_JSONL", WORK / "train.jsonl.gz"))
DEV_JSONL = Path(ENV.get("DEV_JSONL", WORK / "dev.jsonl.gz"))
CODE_TAR = Path(ENV.get("CODE_TAR", WORK / "sys1_code.tar.gz"))
CODE_DIR = Path(ENV.get("CODE_DIR", WORK / "sys1_code"))
OUT_DIR = Path(ENV.get("OUT_DIR", WORK / "out"))
EPOCHS = ENV.get("EPOCHS", "1")
LR = ENV.get("LR", "2e-5")   # the README's start value for --init_from (kev README, Fine-Tune on Your Own Data)
BATCH = ENV.get("BATCH", "8")
ACCUM = ENV.get("ACCUM", "1")
DEVICE = ENV.get("DEVICE", "cuda")
DTYPE = ENV.get("DTYPE", "bf16" if DEVICE == "cuda" else "fp32")   # bf16 autocast over fp32 weights: the Kev-0.8B card's training precision
INIT_FROM = ENV.get("INIT_FROM", "jaredpalmer/kev-0.8b@bf75a6a8848ea6960ff2ed108d9ed44c2941174f")   # the Hub revision the Mac check loaded
BASE = ENV.get("BASE", "Qwen/Qwen3.5-0.8B-Base")
BASE_REVISION = ENV.get("BASE_REVISION", "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68")   # the base_revision in Kev-0.8B's head.pt
SPEED_ROWS = int(ENV.get("SPEED_ROWS", "200"))
PARTIAL_EVERY_MIN = ENV.get("PARTIAL_EVERY_MIN", "10")
EVAL_BATCH = ENV.get("EVAL_BATCH", "16")
INSTALL = ENV.get("INSTALL", "1") == "1"
INSTALL_FLA = ENV.get("INSTALL_FLA", "1") == "1"
DRY_RUN = ENV.get("DRY_RUN", "0") == "1"
# kev.train by default deletes the labelled option in 10% of records and adds wrong options in 27% more (kev/train.py:400-402,
# kev/data.py:335-340). Here the options are the menu and the fallback, and the label is the rule of the export, so this stays off.
AUGMENTATION_OFF = ["--p_none", "0", "--p_none_distract", "0", "--p_distract", "0", "--p_none_pair", "0"]
STEM = f"seed{SEED}"
RUN_DIR = OUT_DIR / f"{STEM}-{MODE}"
RESULT = OUT_DIR / f"{STEM}.json"
CHECKPOINT_TAR = OUT_DIR / f"{STEM}.tar"
PARTIAL_TAR = OUT_DIR / f"{STEM}-partial.tar"


def run(cmd, env=None, abort_on=None):
    """Run a child, echoing its output as it comes, because `colab exec` shows only what the kernel prints. A line matching
    `abort_on` stops the child at once, so a run that is already wrong spends no more compute units."""
    print("+", " ".join(str(c) for c in cmd), flush=True)
    with subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", env={**ENV, **(env or {})}) as child:
        for line in child.stdout:
            print(line, end="", flush=True)
            if abort_on and re.search(abort_on, line):
                child.kill()
                raise RuntimeError(f"stopped on this line: {line.strip()}")
    if child.returncode:
        raise RuntimeError(f"exit {child.returncode}: {cmd[:4]}")


def install():
    """A separate environment with the Mac's versions, so Colab's own packages do not matter. In Colab's Python, Kev's bound moved torch
    from 2.11 to 2.8 and left torchvision, torchaudio and torchao behind, and each of them broke an import in turn."""
    global PY
    python = WORK / "venv" / "bin" / "python"
    if not python.exists():
        run(["bash", "-c", "command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh"])
        uv = shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")
        run([uv, "venv", "--python", "3.12", WORK / "venv"])
        run([uv, "pip", "install", "--python", python, f"kev @ {KEV_SOURCE}", *PINS])
        if INSTALL_FLA:
            for pin in FLA_PINS:   # separate steps, because torch pins an older triton (`lab/ENVIRONMENT.md`)
                run([uv, "pip", "install", "--python", python, pin])
    PY = str(python)


def read_lines(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield line


def prepare(source, dest, allowed, limit=None):
    """Check every row's split, then write the plain JSONL `kev.data.load_records` reads. -> (rows in the source, rows written)."""
    lines = []
    for line in read_lines(source):
        meta = json.loads(line).get("_meta", {})
        if meta.get("split") not in allowed:
            raise AssertionError(f"{source}: split must be one of {allowed}, got {meta.get('split')!r} at {meta.get('id')}")
        if "valid_unseen" in str(meta.get("game_id", "")):
            raise AssertionError(f"{source}: test game {meta.get('game_id')}")
        lines.append(line)
    total = len(lines)
    if not total:
        raise ValueError(f"{source}: no rows")
    if limit and limit < total:
        lines = random.Random(SEED).sample(lines, limit)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return total, len(lines)


def kev_train_args(data):
    return ["--data", data, "--base", BASE, "--base_revision", BASE_REVISION, "--init_from", INIT_FROM, "--epochs", EPOCHS, "--lr", LR, "--batch", BATCH, "--accum", ACCUM,
            "--dtype", DTYPE, "--device", DEVICE, "--max_state", MAX_STATE, "--seed", SEED, "--out", RUN_DIR, *AUGMENTATION_OFF]


def child_json(code, env):
    return json.loads(subprocess.run([PY, "-c", code], capture_output=True, text=True, env={**ENV, **env}, check=True).stdout)


def environment(env):
    code = ("import importlib.metadata as m, json, platform, torch\n"
            "names = ['kev', 'torch', 'transformers', 'peft', 'flash-linear-attention', 'fla-core', 'triton', 'causal-conv1d', 'numpy']\n"
            "def v(n):\n"
            "    try: return m.version(n)\n"
            "    except m.PackageNotFoundError: return None\n"
            "print(json.dumps({'python': platform.python_version(), 'torch_cuda': torch.version.cuda, 'packages': {n: v(n) for n in names}, 'deterministic_algorithms': False}))\n")
    found = child_json(code, env)
    found["kev_source"] = KEV_SOURCE
    return found


def hardware():
    out = {"machine": platform.machine(), "platform": platform.platform(), "cpus": os.cpu_count()}
    try:
        query = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], capture_output=True, text=True, check=True).stdout.strip()
        out["nvidia_smi"] = query
    except (OSError, subprocess.CalledProcessError) as error:
        out["nvidia_smi"] = f"unavailable: {error}"
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        out["memory_total_kb"] = int(re.search(r"MemTotal:\s+(\d+)", meminfo.read_text()).group(1))
    return out


def speeds(metrics, records_per_step):
    """Rows per second over the whole run, and with the first tenth of steps left out, which hold the first-call kernel compile."""
    seconds = metrics["step_seconds"]
    skip = max(1, len(seconds) // 10) if len(seconds) > 1 else 0
    rest = seconds[skip:]
    return {"train_rows_per_second": round(metrics["records_seen"] / metrics["wall_seconds"], 3),
            "train_rows_per_second_after_first_tenth": round(records_per_step * len(rest) / sum(rest), 3) if rest and sum(rest) > 0 else None,
            "optimizer_steps": metrics["optimizer_steps"], "steps_left_out_as_warmup": skip, "wall_seconds": round(metrics["wall_seconds"], 1)}


def tar_checkpoint():
    with tarfile.open(CHECKPOINT_TAR, "w") as tar:
        for name in ("adapter_config.json", "adapter_model.safetensors", "head.pt", "training_config.json", "training_metrics.json", "runner_metrics.json"):
            tar.add(RUN_DIR / name, arcname=f"checkpoint/{name}")


def main():
    stages = {}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if RUN_DIR.exists():
        raise RuntimeError(f"{RUN_DIR} exists, and kev.train refuses to overwrite a run")
    if CODE_TAR.exists():
        with tarfile.open(CODE_TAR) as tar:
            tar.extractall(CODE_DIR, filter="data")
    if not (CODE_DIR / "train" / "kev_train_runner.py").exists():
        raise FileNotFoundError(f"no training code under {CODE_DIR}")

    t = time.time()
    train_plain, dev_plain = WORK / f"{STEM}-train.jsonl", WORK / f"{STEM}-dev.jsonl"
    limit = None if MODE == "full" else SPEED_ROWS
    train_total, train_rows = prepare(TRAIN_JSONL, train_plain, ("train",), limit)
    dev_total, dev_rows = prepare(DEV_JSONL, dev_plain, ("dev", "dev_heldout_rooms"), None if MODE == "full" else SPEED_ROWS)
    stages["prepare_seconds"] = round(time.time() - t, 1)
    if DRY_RUN:
        print(json.dumps({"kev_train_args": [str(x) for x in kev_train_args(train_plain)], "train_rows": train_rows, "dev_rows": dev_rows, "mode": MODE}), flush=True)
        return

    t = time.time()
    if INSTALL:
        install()
    stages["install_seconds"] = round(time.time() - t, 1)
    child_env = {"PYTHONPATH": str(CODE_DIR), "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false", "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1"}

    t = time.time()
    train_env = {**child_env, "PARTIAL_TAR": "" if MODE == "speedtest" else str(PARTIAL_TAR), "PARTIAL_EVERY_MIN": PARTIAL_EVERY_MIN}
    # kev.train prints this line at the start when rows do not fit (kev/train.py:108). Raise MAX_STATE from the export's report.
    run([PY, "-m", "train.kev_train_runner", *kev_train_args(train_plain)], train_env, abort_on=r"dropped \d+ of \d+ records")
    stages["train_stage_seconds"] = round(time.time() - t, 1)
    metrics = json.loads((RUN_DIR / "training_metrics.json").read_text())
    runner = json.loads((RUN_DIR / "runner_metrics.json").read_text())
    if metrics["records_seen"] != int(EPOCHS) * train_rows:
        raise RuntimeError(f"kev.train saw {metrics['records_seen']} rows, expected {int(EPOCHS) * train_rows}")

    t = time.time()
    refit_json = OUT_DIR / f"{STEM}-{MODE}-refit.json"
    refit_cmd = [PY, "-m", "train.refit_temperature", "--run", RUN_DIR, "--dev", dev_plain, "--device", DEVICE, "--seed", SEED, "--out", refit_json, "--batch", EVAL_BATCH]
    if MODE == "speedtest":
        refit_cmd.append("--measure-only")
    run(refit_cmd, child_env)
    stages["refit_stage_seconds"] = round(time.time() - t, 1)
    refit = json.loads(refit_json.read_text())

    env = environment(child_env)
    machine = hardware()
    records_per_step = int(BATCH) * int(ACCUM)
    speed = speeds(metrics, records_per_step)
    result = {
        "topic": "arm-classification", "mode": MODE, "seed": SEED, "git_commit": ENV.get("GIT_COMMIT", "unknown"), "git_dirty": ENV.get("GIT_DIRTY", "unknown"),
        "environment": env, "hardware": machine, "precision": {"training": f"{DTYPE} autocast over fp32 weights" if DTYPE == "bf16" else "fp32", "temperature_refit": refit["precision"]},
        "config": {"init_from": INIT_FROM, "base": BASE, "base_revision": BASE_REVISION, "epochs": int(EPOCHS), "lr": float(LR), "batch": int(BATCH), "accum": int(ACCUM), "max_state": MAX_STATE,
                   "device": DEVICE, "augmentation": "off", "partial_every_min": PARTIAL_EVERY_MIN, "kev_train_args": [str(x) for x in kev_train_args(train_plain)]},
        "rows": {"train_in_file": train_total, "train_used": train_rows, "dev_in_file": dev_total, "dev_used": dev_rows},
        "train_speed": speed, "infer_rows_per_second": refit["infer_rows_per_second"],
        "peak_memory": {"device_bytes": metrics.get("peak_device_bytes"), "rss_bytes": metrics.get("peak_rss_bytes"), **runner["peak_memory"]},
        "grad_norm": metrics.get("grad_norm"), "stage_seconds": stages, "scope": f"Measured with device {DEVICE} on {machine.get('nvidia_smi')}.",
    }
    if MODE != "speedtest":
        result |= {"temperature": refit["temperature_fit"]["temperature"], "temperature_fit": refit["temperature_fit"], "per_game": refit["per_game"], "aggregates": refit["aggregates"]}
        tar_checkpoint()
        result["checkpoint_tar_bytes"] = CHECKPOINT_TAR.stat().st_size
    if MODE == "speedtest":
        # A projection for planning: rows per second measured on SPEED_ROWS rows, applied to every row of the file. It is no measurement of a full run.
        rate = speed["train_rows_per_second_after_first_tenth"] or speed["train_rows_per_second"]
        result["projection"] = {"train_hours_one_epoch": round(train_total / rate / 3600, 2), "from_rows_per_second": rate, "train_rows": train_total, "cu_at_1_54_per_hour": round(train_total / rate / 3600 * 1.54, 2)}
    result["total_seconds"] = round(time.time() - T0, 1)
    RESULT.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print("RESULT", json.dumps({k: result[k] for k in ("seed", "mode", "train_speed", "infer_rows_per_second", "peak_memory", "total_seconds")}, default=str), flush=True)
    shutil.rmtree(RUN_DIR / "resume", ignore_errors=True)


if __name__ == "__main__":
    main()
