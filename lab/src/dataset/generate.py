"""Play every rollout of every game into the pool, one gzip file per game, in at most 4 worker processes.

    nice -n 10 python -m dataset.generate --pool POOL --recovery 3 --loop-break 2

A game's file appears by rename only after all its rollouts ran, so a crash loses the games in flight and the next start skips what exists. A
start with settings other than those in `POOL/config.json` is refused. The driver submits a game only while this job stays under `--max-gb` and the
machine under `--max-used-share`, and it stops if the pool passes `--max-pool-gb`.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import get_context
from pathlib import Path

from alf.env import MAX_STEPS, _make_stack, game_spec

from .rollouts import Game, play_game
from .splits import SPLITS, assert_no_leakage, load_splits

_stack = None
GB = 1024**3


def game_file(pool: Path, split: str, uid: str) -> Path:
    return pool / split / (uid.split("/", 1)[1].replace("/", "__") + ".jsonl.gz")


def read_runs(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pool_files(pool, split: str) -> list[Path]:
    return sorted((Path(pool) / split).glob("*.jsonl.gz"))


def work_one_game(job: dict) -> dict:
    """In a worker process: load the game, play its rollouts, write its file."""
    global _stack
    t0 = time.perf_counter()
    if _stack is None:
        _stack = _make_stack(MAX_STEPS)
    spec = game_spec(job["uid"])
    _stack.load(spec.path)
    game = Game(spec, job["split"], _stack)
    runs, notes = play_game(game, job["seed"], job["expert_seeds"], job["recovery"], job["loop_break"])
    out = Path(job["path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_name(out.name + f".{os.getpid()}.tmp")
    with open(temp, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as f:
        for run in runs:
            f.write((json.dumps(run, ensure_ascii=False) + "\n").encode("utf-8"))
    os.replace(temp, out)
    return {"uid": job["uid"], "split": job["split"], "seconds": time.perf_counter() - t0, "runs": len(runs), "steps": sum(len(r["steps"]) for r in runs),
            "won": dict(Counter(r["family"] for r in runs if r["won"])), "lost": dict(Counter(r["family"] for r in runs if not r["won"])), "pid": os.getpid(), **notes}


def job_memory_gb() -> float:
    """Resident memory of this process and its children."""
    table = subprocess.run(["ps", "-ax", "-o", "pid=,ppid=,rss="], capture_output=True, text=True).stdout
    me, total = os.getpid(), 0
    for line in table.splitlines():
        pid, ppid, rss = (int(x) for x in line.split())
        if pid == me or ppid == me:
            total += rss
    return total * 1024 / GB


def machine_used_share() -> float:
    """Active, wired and compressed pages over all memory, from `vm_stat`. The system's own count leaves out file cache it can drop."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    size = int(out.split("page size of ")[1].split()[0])
    pages = {line.split(":")[0].strip(): int(line.split(":")[1].strip(" .")) for line in out.splitlines()[1:] if ":" in line}
    used = pages["Pages active"] + pages["Pages wired down"] + pages.get("Pages occupied by compressor", 0)
    total = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout)
    return used * size / total


def pool_gb(pool: Path) -> float:
    return sum(p.stat().st_size for p in pool.rglob("*.jsonl.gz")) / GB


def log(path: Path, record: dict) -> None:
    line = json.dumps(record)
    print(line, flush=True)
    with open(path, "a") as f:
        f.write(line + "\n")


def run_jobs(jobs: list[dict], a, pool: Path) -> tuple[Counter, int]:
    """Submit games as memory allows. A dead worker raises BrokenProcessPool, and the caller starts again, skipping what exists."""
    totals: Counter = Counter()
    done = 0
    t0 = last = time.time()
    context = get_context("spawn")
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=context, max_tasks_per_child=a.tasks_per_child) as executor:
        waiting = list(reversed(jobs))
        running = set()
        while waiting or running:
            over = job_memory_gb() >= a.max_gb or machine_used_share() >= a.max_used_share
            while waiting and len(running) < a.workers * 2 and not over:
                running.add(executor.submit(work_one_game, waiting.pop()))
            if not running:
                log(pool / "progress.jsonl", {"event": "paused for memory", "job_gb": round(job_memory_gb(), 2), "machine_used_share": round(machine_used_share(), 3), "time": time.time()})
                time.sleep(30)
                continue
            finished, running = wait(running, timeout=30, return_when=FIRST_COMPLETED)
            for future in finished:
                result = future.result()
                done += 1
                totals.update({"runs": result["runs"], "steps": result["steps"], "skipped_rollouts": result["skipped_rollouts"]})
                totals.update({f"won/{k}": v for k, v in result["won"].items()})
                totals.update({f"lost/{k}": v for k, v in result["lost"].items()})
                with open(pool / "timing.jsonl", "a") as f:
                    f.write(json.dumps({k: result[k] for k in ("uid", "split", "seconds", "runs", "steps", "pid")}) + "\n")
            now = time.time()
            if now - last >= a.log_every:
                last = now
                rate = done / (now - t0)
                size = pool_gb(pool)
                log(pool / "progress.jsonl", {"time": now, "session_seconds": round(now - t0, 1), "games_done_this_session": done, "games_left": len(waiting) + len(running), "games_per_second": round(rate, 3),
                                              "eta_hours": round((len(waiting) + len(running)) / max(rate, 1e-9) / 3600, 2), "job_gb": round(job_memory_gb(), 2), "machine_used_share": round(machine_used_share(), 3),
                                              "pool_gb": round(size, 3), **totals})
                if size > a.max_pool_gb:
                    raise SystemExit(f"the pool passed {a.max_pool_gb} GB")
    return totals, done


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pool", required=True)
    ap.add_argument("--splits-dir", default="lab/results/sys1-training-set")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--expert-seeds", type=int, default=3)
    ap.add_argument("--recovery", type=int, default=3)
    ap.add_argument("--loop-break", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit-per-split", type=int, default=0, help="only the first N games of each split, for a speed test")
    ap.add_argument("--tasks-per-child", type=int, default=200, help="a worker is replaced after this many games, which bounds its memory")
    ap.add_argument("--max-gb", type=float, default=8.0)
    ap.add_argument("--max-used-share", type=float, default=0.8)
    ap.add_argument("--max-pool-gb", type=float, default=5.0)
    ap.add_argument("--log-every", type=float, default=60.0)
    ap.add_argument("--retries", type=int, default=5)
    a = ap.parse_args(argv)
    if a.workers > 4:
        raise SystemExit("at most 4 worker processes, from Halil's limit on this Mac")
    pool = Path(a.pool).expanduser()
    pool.mkdir(parents=True, exist_ok=True)
    splits = load_splits(a.splits_dir)
    assert_no_leakage(splits)
    config = {"seed": a.seed, "expert_seeds": list(range(a.expert_seeds)), "recovery": a.recovery, "loop_break": a.loop_break, "tracking": "always", "max_steps": MAX_STEPS,
              "limit_per_split": a.limit_per_split, "splits_seed": splits["seed"], "held_out_rooms": splits["held_out_rooms"]}
    config_path = pool / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise SystemExit(f"{config_path} holds other settings, so this pool cannot be resumed with these")
    config_path.write_text(json.dumps(config, indent=1) + "\n")

    def jobs_left() -> list[dict]:
        jobs = []
        for split in ("dev", "dev_heldout_rooms", "train"):  # dev first, so the menu can be checked early. Each list is in sorted uid order.
            uids = splits[split][:a.limit_per_split] if a.limit_per_split else splits[split]
            for uid in uids:
                path = game_file(pool, split, uid)
                if not path.exists():
                    jobs.append({"uid": uid, "split": split, "path": str(path), "seed": a.seed, "expert_seeds": config["expert_seeds"], "recovery": a.recovery, "loop_break": a.loop_break})
        return jobs

    os.environ.setdefault("PYTHONHASHSEED", str(a.seed))  # The expert's label can depend on Python's string hash order, so workers inherit a fixed one.
    session = {"started": time.time(), "workers": a.workers, "platform": platform.platform(), "python": sys.version.split()[0], "python_hash_seed": os.environ["PYTHONHASHSEED"], "games_at_start": len(jobs_left())}
    for attempt in range(a.retries):
        jobs = jobs_left()
        if not jobs:
            break
        try:
            totals, done = run_jobs(jobs, a, pool)
            session.update(dict(totals), games_done=session.get("games_done", 0) + done)
        except BrokenProcessPool as error:
            log(pool / "progress.jsonl", {"event": "a worker died, starting again", "attempt": attempt, "error": str(error), "time": time.time()})
    session["ended"], session["wall_seconds"], session["games_left"] = time.time(), time.time() - session["started"], len(jobs_left())
    with open(pool / "sessions.jsonl", "a") as f:
        f.write(json.dumps(session) + "\n")
    print(json.dumps(session, indent=1), flush=True)
    if session["games_left"]:
        raise SystemExit(f"{session['games_left']} games left after {a.retries} attempts")


if __name__ == "__main__":
    main()
