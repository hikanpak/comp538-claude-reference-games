#!/usr/bin/env python
"""One ALFWorld game played by one reference model, for the reference rows and the saved traces. Never training data.

    python lab/jobs/reference-games.py --player elm start --game UID --model MODEL     # what the model's process calls
    python lab/jobs/reference-games.py --player elm step --game UID --command "go to shelf 1"
    python lab/jobs/reference-games.py --player elm status --list test [--next 4]
    python lab/jobs/reference-games.py gather --list test                              # the neat tables, from the result files
    python lab/jobs/reference-games.py replay-check --list test                        # replays every saved game in the real game

Every question was a fresh session: each game ran in its own command-line agent process, in a new empty folder with no settings, so the process
saw no project file and nothing of another game. The launcher that started those processes names the models and their command-line tool, so it stays
in the private folder `lab/data/reference-games-private/`, with the processes' whole output. This file holds what a process called and what reads the results.

The process sees only what Bonn's state holds (`jev_benchmarking/sequential/envs/alfworld.py`): the task, the observations, the steps left and
the admissible commands without `help`. A process holds no game between calls: each call loads the game, replays the commands in the result file, checks
that every observation repeats, then runs one command. A command off the list costs no step, because Bonn's models choose from the list.

`--player` is elm, birch, cypress or ironwood, and a longer name is a larger model. Each keeps its own results folder. Three lists run here: `dev`, `dev_heldout_rooms`, and `test`, the 134 test games,
which Halil opened for these models on 2026-11-04. While games run, this program prints no win count for the test list, and only `gather` does, because Halil asked for the results gathered.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lab" / "src"))

ALL_RESULTS = Path(os.environ.get("REFERENCE_GAMES_RESULTS", ROOT / "lab" / "results" / "reference-games"))
LISTS = ROOT / "lab" / "results" / "sys1-training-set"
LIST_NAMES = ("dev", "dev_heldout_rooms", "test")
TEST_SPLIT = "valid_unseen"
MODELS = ("elm", "birch", "cypress", "ironwood")  # Real names only in lab/data/reference-games-private/names.json.
MAX_FAILED_TRIES = 2
PROMPT = ROOT / "lab" / "jobs" / "reference-games.prompt.md"
NOT_FOR_TRAINING = ("These commands are a benchmark reference. The provider's usage policy lists using outputs to train an AI model "
                    "without prior authorization, so nothing here may train Kev or any other model, as labels, features or fallback steps.")
PACKAGES = ("alfworld", "textworld", "fast-downward-textworld", "numpy")
PLAYER = None  # Set in main from --player, with RESULTS its folder.
RESULTS = None


def listed_games(name: str) -> list[str]:
    path = ROOT / "lab" / "results" / "reference-games" / "test-games.txt" if name == "test" else LISTS / f"{name}.txt"
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def result_path(uid: str) -> Path:
    return RESULTS / uid.split("/")[0] / (uid.replace("/", "__") + ".json")


def is_test(uid: str) -> bool:
    return uid.split("/")[0] == TEST_SPLIT


def write(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(record, indent=1) + "\n")
    os.replace(tmp, path)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True).stdout.strip()


def provenance() -> dict:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain", "--", "lab/src", "lab/jobs")),
            "python": platform.python_version(), "packages": versions, "machine": platform.platform(), "cpus": os.cpu_count(),
            "prompt_sha256": hashlib.sha256(PROMPT.read_bytes()).hexdigest(), "python_hash_seed": os.environ.get("PYTHONHASHSEED"), "seed": None,
            "seed_note": "The game has no randomness and the model's sampling cannot be seeded from here."}


def open_game(record: dict):
    """-> (slot, view) with the record's commands replayed. Raises if the game does not repeat."""
    from alf.env import MAX_STEPS, _Slot, game_spec
    slot = _Slot(MAX_STEPS, False, "foreign")
    view = slot.load(0, game_spec(record["uid"], allow_test_games=is_test(record["uid"])))
    for number, entry in enumerate(record["trace"]):
        view = slot.step(entry["command"])
        if view.obs != entry["observation"]:
            raise SystemExit(f"REPLAY FAILED at step {number + 1} of {record['uid']}: the game did not repeat")
    return slot, view


def show(view, record: dict) -> None:
    from alf.env import MAX_STEPS
    if view.done:
        print(f"Current observation: {view.obs}\nGAME OVER won={str(view.won).lower()} steps={view.step}")
        return
    print(f"Task: {view.goal}\nInitial observation: {view.first_obs}\nCurrent observation: {view.obs}")
    print(f"Steps taken: {view.step}, steps left: {MAX_STEPS - view.step}\nAdmissible commands:")
    print("\n".join(view.admissible))


def start(args) -> None:
    from alf.env import MAX_STEPS
    allowed = {uid for name in LIST_NAMES for uid in listed_games(name)}
    if args.game not in allowed:
        raise SystemExit(f"{args.game} is on none of the lists, so it does not run here")
    path = result_path(args.game)
    earlier, kept = [], {}
    if path.exists():
        old = json.loads(path.read_text())
        if old["finished"]:
            raise SystemExit(f"{args.game} has been played. A game runs once")
        kept = {k: old[k] for k in ("failed_processes", "processes") if k in old}
        earlier = old["abandoned_attempts"] + ([{k: old[k] for k in ("started_at", "model_reported", "refused_commands", "trace")}] if old["started_at"] else [])
    record = {"topic": "reference-games", "player": PLAYER, "not_for_training": NOT_FOR_TRAINING, "uid": args.game, "split": args.game.split("/")[0],
              "task_type": args.game.split("/")[1].split("-")[0], "model_reported": args.model, "max_steps": MAX_STEPS,
              **provenance(), "started_at": now(), "finished_at": None, "finished": False, "ended_early": False, "won": False, "steps": 0,
              "refused_commands": [], "abandoned_attempts": earlier, "task": None, "initial_observation": None, "trace": [], **kept}
    slot, view = open_game(record)
    record["task"], record["initial_observation"] = view.goal, view.first_obs
    write(path, record)
    show(view, record)
    slot.close()


def step(args) -> None:
    path = result_path(args.game)
    if not path.exists():
        raise SystemExit(f"{args.game} has not been started")
    record = json.loads(path.read_text())
    if record["finished"]:
        raise SystemExit(f"{args.game} is over: won={str(record['won']).lower()} steps={record['steps']}")
    slot, view = open_game(record)
    if args.command not in view.admissible:
        record["refused_commands"].append({"after_step": view.step, "command": args.command})
        write(path, record)
        print(f"'{args.command}' is not an admissible command, and no step was used. Choose one from this list.")
        show(view, record)
        slot.close()
        return
    view = slot.step(args.command)
    record["trace"].append({"command": args.command, "observation": view.obs, "commands_on_offer": len(view.admissible)})
    record["steps"], record["won"] = view.step, view.won
    if view.done:
        record["finished"], record["finished_at"] = True, now()
    write(path, record)
    show(view, record)
    slot.close()


def counts(name: str) -> dict:
    uids = listed_games(name)
    records = [json.loads(result_path(uid).read_text()) for uid in uids if result_path(uid).exists()]
    found = {"list": name, "games": len(uids), "finished": sum(r["finished"] for r in records), "gave_up_after_failures": sum(r.get("failed_processes", 0) >= MAX_FAILED_TRIES and not r["finished"] for r in records),
             "restarted": sum(bool(r["abandoned_attempts"]) for r in records),
             "last_error": next((p["error"][:300] for r in reversed(records) for p in reversed(r.get("processes", [])) if p.get("error")), None)}
    if name != "test":
        found["won"] = sum(r["won"] for r in records)
    return found


def waiting(name: str, retry_gave_up: bool = False) -> list[str]:
    left = []
    for uid in listed_games(name):
        path = result_path(uid)
        record = json.loads(path.read_text()) if path.exists() else {}
        if not record.get("finished") and (retry_gave_up or record.get("failed_processes", 0) < MAX_FAILED_TRIES):
            left.append(uid)
    return left


def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    """The 95% Wilson interval of a share, in percent."""
    if not n:
        return None, None
    share, half = wins / n, z * math.sqrt(wins * (n - wins) / n + z * z / 4) / n
    centre, width = (share + z * z / (2 * n)) / (1 + z * z / n), half / (1 + z * z / n)
    return round(100 * (centre - width), 1), round(100 * (centre + width), 1)


def share(wins: int, n: int) -> float | None:
    return round(100 * wins / n, 1) if n else None


def mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def table(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plan_steps(uid: str) -> int | None:
    """Steps in the plan that ships with the game, which knows where everything is. None when the game's plan file is not on this machine."""
    from alf.env import _json_root
    path = Path(_json_root()) / uid / "traj_data.json"
    if not path.exists():
        return None
    return sum(step["discrete_action"]["action"] != "NoOp" for step in json.loads(path.read_text())["plan"]["high_pddl"])


def gather(args) -> None:
    """Writes `summary/<list>-by-model.csv`, `-by-task.csv` and `-by-game.csv` from the result files of every model, and prints the first."""
    from harness.metrics import repeat_metrics
    uids = listed_games(args.list)
    plans = {uid: plan_steps(uid) for uid in uids}
    by_model, by_task, by_game = [], [], []
    for player in MODELS:
        files = {uid: ALL_RESULTS / player / uid.split("/")[0] / (uid.replace("/", "__") + ".json") for uid in uids}
        records = [json.loads(path.read_text()) for path in files.values() if path.exists()]
        played = [r for r in records if r["finished"]]
        processes = [p for r in records for p in r.get("processes", [])]
        logs = [SimpleNamespace(steps=[SimpleNamespace(command=e["command"], obs=e["observation"]) for e in r["trace"]]) for r in played]
        wins = sum(r["won"] for r in played)
        low, high = wilson(wins, len(played))
        by_model.append({"model": player, "games_on_list": len(uids), "played": len(played), "not_played": len(uids) - len(played), "won": wins, "win_percent": share(wins, len(played)),
                         "ci95_low": low, "ci95_high": high, "ended_early": sum(bool(r.get("ended_early")) for r in played), "restarted": sum(bool(r["abandoned_attempts"]) for r in played),
                         "mean_steps": mean([r["steps"] for r in played]), "mean_steps_won": mean([r["steps"] for r in played if r["won"]]), "mean_plan_steps": mean([plans[r["uid"]] for r in played if plans.get(r["uid"]) is not None]), **repeat_metrics(logs),
                         "refused_commands": sum(len(r.get("refused_commands", [])) for r in played),
                         "mean_seconds_per_process": mean([p["duration_ms"] / 1000 for p in processes if p.get("duration_ms")]),
                         "prompt_hashes": len({r.get("prompt_sha256") for r in played}), "git_commits": " ".join(sorted({str(r.get("git_commit"))[:9] for r in played}))})
        for task in sorted({uid.split("/")[1].split("-")[0] for uid in uids}):
            of_task = [r for r in played if r["task_type"] == task]
            won = sum(r["won"] for r in of_task)
            by_task.append({"model": player, "task_type": task, "played": len(of_task), "won": won, "win_percent": share(won, len(of_task))})
        for r in records:
            by_game.append({"model": player, "uid": r["uid"], "task_type": r.get("task_type"), "finished": r["finished"], "won": r["won"], "steps": r["steps"], "plan_steps": plans.get(r["uid"]), "ended_early": bool(r.get("ended_early")),
                            "refused_commands": len(r.get("refused_commands", [])), "restarted": bool(r["abandoned_attempts"])})
    for ending, rows in (("by-model", by_model), ("by-task", by_task), ("by-game", by_game)):
        if rows:
            table(ALL_RESULTS / "summary" / f"{args.list}-{ending}.csv", rows)
    for row in by_model:
        print(json.dumps({k: row[k] for k in ("model", "played", "not_played", "won", "win_percent", "ci95_low", "ci95_high", "mean_steps", "repeat_rate")}))


def status(args) -> None:
    print(json.dumps(counts(args.list)))
    if args.next:
        print("\n".join(waiting(args.list)[:args.next]))


def replay_check(args) -> None:
    """Replays every saved game in the real game and compares each observation, the win and the step count with the record. Prints REPLAY_OK or REPLAY_FAILED."""
    failures = 0
    for player in MODELS:
        checked, bad = 0, []
        for uid in listed_games(args.list):
            path = ALL_RESULTS / player / uid.split("/")[0] / (uid.replace("/", "__") + ".json")
            if not path.exists():
                continue
            record = json.loads(path.read_text())
            checked += 1
            try:
                _, view = open_game(record)
            except SystemExit as error:
                bad.append(str(error))
                continue
            if (view.done and view.won) != record["won"] or view.step != record["steps"]:
                bad.append(f"{uid}: replay won={view.won} step={view.step}, record won={record['won']} steps={record['steps']}")
        print(json.dumps({"model": player, "replayed": checked, "failed": len(bad), "first_failures": bad[:3]}))
        failures += len(bad)
    print("REPLAY_FAILED" if failures else "REPLAY_OK")


def main() -> None:
    # The game's text can depend on Python's hash order, and every call replays the game in a new process, so all of them use one order.
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable, *sys.argv])
    global PLAYER, RESULTS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--player", choices=MODELS)
    commands = parser.add_subparsers(dest="action", required=True)
    p = commands.add_parser("start")
    p.add_argument("--game", required=True)
    p.add_argument("--model", required=True)
    p.set_defaults(run=start)
    p = commands.add_parser("step")
    p.add_argument("--game", required=True)
    p.add_argument("--command", required=True)
    p.set_defaults(run=step)
    p = commands.add_parser("gather")
    p.add_argument("--list", required=True, choices=LIST_NAMES)
    p.set_defaults(run=gather)
    p = commands.add_parser("replay-check")
    p.add_argument("--list", required=True, choices=LIST_NAMES)
    p.set_defaults(run=replay_check)
    p = commands.add_parser("status")
    p.add_argument("--list", required=True, choices=LIST_NAMES)
    p.add_argument("--next", type=int, default=0)
    p.set_defaults(run=status)
    args = parser.parse_args()
    if args.action not in ("gather", "replay-check") and not args.player:
        parser.error(f"{args.action} needs --player")
    PLAYER, RESULTS = args.player, ALL_RESULTS / str(args.player)
    args.run(args)


if __name__ == "__main__":
    main()
