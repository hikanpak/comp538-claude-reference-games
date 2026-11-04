"""The games of each split, and the rooms held out of the training games.

`train` and `dev` (`valid_seen`) come from `AlfredTWEnv.collect_game_files`, ALFWorld's own game filter, so the counts can be compared with
the 3,553 and 140 papers report. `valid_unseen` holds the test games. Nothing here names that folder as a path to read, and every list is
checked against it.

`dev_heldout_rooms` takes whole rooms out of the training games. A room is the floorplan number at the end of a game's task folder. Rooms are
shuffled by `seed` inside each of ALFWorld's four room types and taken one type at a time, kitchen first, until the games reach `MIN_GAMES`
and each of the 6 task types has at least `MIN_PER_TASK` games. Rooms of all four types are always taken, because the task types live in
different room types (heat and cool only in kitchens, looking under a lamp only in bedrooms and living rooms).

    python -m dataset.splits --seed 0 --out lab/results/sys1-training-set
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import random
import re
from collections import Counter
from pathlib import Path

from alf.env import TEST_SPLIT, _json_root, game_spec

SPLITS = ("train", "dev", "dev_heldout_rooms")
TASK_TYPE_IDS = [1, 2, 3, 4, 5, 6]
MIN_GAMES = 140
MIN_PER_TASK = 10
ROOM_TYPES = ("kitchen", "bedroom", "living", "bathroom")
ROOM = re.compile(r"-(\d+)/trial_")  # The same pattern as `lab/tools/check_gates.py dataset`.


class LeakageError(AssertionError):
    """A game or a room sits where a split rule forbids it. An AssertionError that `python -O` keeps, because it is raised by hand."""


def room_of(uid: str) -> int:
    match = ROOM.search(uid)
    if not match:
        raise ValueError(f"no room number in {uid}")
    return int(match.group(1))


def room_type(room: int) -> str:
    if room < 100:
        return "kitchen"
    if room < 300:
        return "living"
    if room < 400:
        return "bedroom"
    return "bathroom"


def task_type_of(uid: str) -> str:
    return uid.split("/")[1].split("-")[0]


def filtered_games(folder: str) -> list[str]:
    """Uids of `folder`, `train` or `valid_seen`, as `AlfredTWEnv` filters them: no movable or sliced game, a known task type, a game file marked solvable."""
    if folder == TEST_SPLIT:
        raise LeakageError("the test games are never listed")
    from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
    path = os.path.join(_json_root(), folder)
    config = {"env": {"goal_desc_human_anns_prob": 0, "task_types": TASK_TYPE_IDS},
              "dataset": {"data_path": path, "eval_id_data_path": path, "num_train_games": -1, "num_eval_games": -1}}
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        env = AlfredTWEnv(config, train_eval="train" if folder == "train" else "eval_in_distribution")
    return sorted(game_spec(os.path.dirname(p)).uid for p in env.game_files)


def choose_held_out_rooms(train_uids: list[str], seed: int, min_games: int = MIN_GAMES, min_per_task: int = MIN_PER_TASK) -> list[int]:
    by_room: dict[int, list[str]] = {}
    for uid in train_uids:
        by_room.setdefault(room_of(uid), []).append(uid)
    rng = random.Random(f"held-out-rooms/{seed}")
    queues = {t: sorted(r for r in by_room if room_type(r) == t) for t in ROOM_TYPES}
    for queue in queues.values():
        rng.shuffle(queue)
    chosen: list[int] = []
    task_types = {task_type_of(u) for u in train_uids}

    def enough() -> bool:
        uids = [u for r in chosen for u in by_room[r]]
        counts = Counter(task_type_of(u) for u in uids)
        return len(uids) >= min_games and all(counts[t] >= min_per_task for t in task_types) and {room_type(r) for r in chosen} >= {t for t, q in queues.items() if q or any(room_type(c) == t for c in chosen)}

    turn = 0
    while not enough():
        if not any(queues.values()):
            raise ValueError("not enough rooms for the held-out split")
        queue = queues[ROOM_TYPES[turn % len(ROOM_TYPES)]]
        if queue:
            chosen.append(queue.pop(0))
        turn += 1
    return sorted(chosen)


def make_splits(train_all: list[str], dev: list[str], seed: int, min_games: int = MIN_GAMES, min_per_task: int = MIN_PER_TASK) -> dict:
    rooms = choose_held_out_rooms(train_all, seed, min_games, min_per_task)
    held = set(rooms)
    splits = {"train": sorted(u for u in train_all if room_of(u) not in held), "dev": sorted(dev),
              "dev_heldout_rooms": sorted(u for u in train_all if room_of(u) in held), "held_out_rooms": rooms, "seed": seed}
    assert_no_leakage(splits)
    return splits


def assert_no_leakage(splits: dict) -> None:
    """No test game anywhere, no game in two splits, and no held-out room among the training games."""
    train, dev, held = set(splits["train"]), set(splits["dev"]), set(splits["dev_heldout_rooms"])
    for name, uids in (("train", train), ("dev", dev), ("dev_heldout_rooms", held)):
        for uid in uids:
            if TEST_SPLIT in uid:
                raise LeakageError(f"test game {uid} in {name}")
    for one, two, a, b in (("train", "dev", train, dev), ("train", "dev_heldout_rooms", train, held), ("dev", "dev_heldout_rooms", dev, held)):
        shared = a & b
        if shared:
            raise LeakageError(f"{len(shared)} games in both {one} and {two}, for example {sorted(shared)[0]}")
    rooms_shared = {room_of(u) for u in train} & {room_of(u) for u in held}
    if rooms_shared:
        raise LeakageError(f"rooms {sorted(rooms_shared)} are in both train and dev_heldout_rooms")
    declared = set(splits.get("held_out_rooms", []))
    if declared and {room_of(u) for u in held} != declared:
        raise LeakageError(f"dev_heldout_rooms holds rooms {sorted({room_of(u) for u in held})}, the list says {sorted(declared)}")


def assert_row_allowed(record: dict, splits_as_sets: dict) -> None:
    """One stored row against the split lists: the row's game is in the list of the split the row names, and a train row is outside the held-out rooms."""
    uid, split = record["game_id"], record["split"]
    if TEST_SPLIT in uid:
        raise LeakageError(f"test game in a stored row: {uid}")
    if split not in SPLITS:
        raise LeakageError(f"unknown split {split!r} at {uid}")
    if uid not in splits_as_sets[split]:
        raise LeakageError(f"{uid} is not a game of split {split}")
    if split == "train" and room_of(uid) in splits_as_sets["held_out_rooms"]:
        raise LeakageError(f"train row from held-out room {room_of(uid)}: {uid}")


def as_sets(splits: dict) -> dict:
    return {name: set(splits[name]) for name in (*SPLITS, "held_out_rooms")}


def load_splits(directory) -> dict:
    return json.loads((Path(directory) / "splits.json").read_text())


def summary(splits: dict) -> dict:
    out = {"held_out_rooms": splits["held_out_rooms"], "seed": splits["seed"], "filtered_counts": splits.get("filtered_counts")}
    for name in SPLITS:
        out[name] = {"games": len(splits[name]), "by_task_type": dict(sorted(Counter(task_type_of(u) for u in splits[name]).items())),
                     "by_room_type": dict(sorted(Counter(room_type(room_of(u)) for u in splits[name]).items())), "rooms": len({room_of(u) for u in splits[name]})}
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="lab/results/sys1-training-set")
    a = ap.parse_args(argv)
    train_all = filtered_games("train")
    dev = filtered_games("valid_seen")
    splits = make_splits(train_all, dev, a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "splits.json").write_text(json.dumps({**splits, "filtered_counts": {"train_folder": len(train_all), "valid_seen_folder": len(dev)}}, indent=0) + "\n")
    for name in SPLITS:
        (out / f"{name}.txt").write_text("\n".join(splits[name]) + "\n")
    print(json.dumps({"filtered_counts": {"train_folder": len(train_all), "valid_seen_folder": len(dev)}, **summary(splits)}, indent=1))


if __name__ == "__main__":
    main()
