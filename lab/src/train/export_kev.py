"""Turn stored step records into Kev's training JSONL (`kev.data.load_records` reads it, `kev.train --data` trains on it).

    python lab/src/train/export_kev.py --records 'SHARD*.jsonl.gz' --menu MENU --k 10 --trace-window 5 --dedupe hard --out train.jsonl.gz

One row per step: the state is `render_prompt`'s text, one question asks which command to run next, the options are the
menu with the fallback option last, and the label is the expert's command if it is on the menu and the fallback otherwise.

What the stored record does not say, and what this file assumes (`lab/INTERFACES.md` gives the fields, not these):

- `obs` at a step is the observation the expert's command answers, and the observation after a command is the next
  record's `obs`. So at step t the trace holds (command_0, obs_1) up to (command_t-1, obs_t), and `first_obs` is step 0's `obs`.
- The command that ran at a step is the expert's, unless the record carries the optional field `command`, which a
  recovery state or a loop-breaking state must carry because its next state did not come from the expert.
- A new trajectory starts where `game_id` changes or `step` does not rise by one. A trajectory must start at step 0 or 1,
  and otherwise the shard is refused, because interleaved shards would give silently wrong state text.
- The control state a row shows is what the harness would hold had Sys1 matched every earlier label: Sys1 controls, no
  LLM rounds left, and `fallbacks_in_row` counts earlier fallback labels in a row, restarting at `fallbacks_before_handover`
  (a value of 0 switches the restart off, as in `harness/control.py`).
- The trace entries carry no actor, so `render_prompt` shows them unmarked, as it shows a teacher's commands. No exported row
  shows the mark of an LLM command.
- Rows are kept only from trajectories the checker marks won (`plan/results_plan.tex`, training labels). Rows flagged
  `label_random` stay out of training, and their commands stay in the trace because they ran. Both are counted.

A row's `_meta.split` travels with it, and `lab/jobs/arm-classification.py` checks it again before training.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

LAB_SRC = Path(__file__).resolve().parents[1]
if str(LAB_SRC) not in sys.path:
    sys.path.insert(0, str(LAB_SRC))

from sys1.base import FALLBACK_TEXT, QUESTION_TEXT  # noqa: E402

SPLITS = ("train", "dev", "dev_heldout_rooms")
TEST_MARKER = "valid_unseen"
QUESTION_ID = "command"
SOURCE = "alfworld_expert"
BASE = "Qwen/Qwen3.5-0.8B-Base"
BASE_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
MAX_OPTIONS = 255   # kev/api.py:14


def check_split(record: dict, required: str = "train") -> None:
    """The assert `lab/INTERFACES.md` asks of every training loader, written as a raise so `python -O` keeps it."""
    game = str(record.get("game_id", ""))
    if TEST_MARKER in game:
        raise AssertionError(f"test game in the export: {game}")
    if record.get("split") != required:
        raise AssertionError(f"split must be {required!r}, got {record.get('split')!r} at {game} step {record.get('step')}")


def trajectories(records):
    """Consecutive records of one game, with `step` rising by one, form one trajectory."""
    current: list[dict] = []
    for record in records:
        if current and (record["game_id"] != current[-1]["game_id"] or record["step"] != current[-1]["step"] + 1):
            yield current
            current = []
        if not current and record["step"] not in (0, 1):
            raise ValueError(f"trajectory of {record['game_id']} starts at step {record['step']}: shards must keep each game's steps in order")
        current.append(record)
    if current:
        yield current


def default_control_state(fallbacks_in_row: int):
    """The control state of a teacher-run row. The harness's own class is used when it exists, because `render_prompt` reads it."""
    try:
        from harness.prompt import ControlState
    except ImportError:
        return SimpleNamespace(controller="sys1", fallbacks_in_row=fallbacks_in_row, llm_rounds_left=0, menu_run=0)
    return ControlState(controller="sys1", fallbacks_in_row=fallbacks_in_row, llm_rounds_left=0, menu_run=0)


def load_render_prompt():
    from harness.prompt import render_prompt   # the harness session owns this module, so it is imported only when needed
    return render_prompt


def load_menu(path):
    from menu import MenuModel   # the alf-data session owns this module, so it is imported only when needed
    return MenuModel.load(path)


def kev_row(state: str, options: list[str], label: str, record: dict, trajectory: int) -> dict:
    return {
        "state": state,
        "questions": {QUESTION_ID: {"type": "choice", "instructions": QUESTION_TEXT, "criteria": {option: None for option in options}, "label": label, "src": SOURCE}},
        "_meta": {"source": SOURCE, "id": f"{record['game_id']}#{trajectory}#{record['step']}", "group_id": record["game_id"], "split": record["split"],
                  "game_id": record["game_id"], "task_type": record.get("task_type"), "step": record["step"], "off_menu": label == FALLBACK_TEXT},
    }


def export_rows(records, menu, k, *, render=None, control_state=default_control_state, required_split="train", trace_window=5, fallbacks_before_handover=2, require_won=True):
    """-> (Kev rows, counts). `menu` has `.menu(task_type, admissible, k)`. `render` defaults to the harness's `render_prompt`."""
    if required_split not in SPLITS:
        raise ValueError(f"unknown split {required_split!r}")
    render = render or load_render_prompt()
    rows, counts = [], Counter()
    checked = []
    for record in records:   # every record is checked before any is used, so one bad record stops the whole export
        check_split(record, required_split)
        checked.append(record)
    counts["records_in"] = len(checked)
    for index, trajectory in enumerate(trajectories(checked)):
        counts["trajectories"] += 1
        if require_won and not any(r.get("won") for r in trajectory):
            counts["trajectories_not_won"] += 1
            counts["rows_dropped_not_won"] += len(trajectory)
            continue
        first_obs, trace, fallbacks = trajectory[0]["obs"], [], 0
        for position, record in enumerate(trajectory):
            expert = record.get("expert")
            usable = True
            if record.get("label_random"):
                counts["rows_dropped_label_random"] += 1
                usable = False
            elif not expert:
                counts["rows_dropped_no_expert"] += 1
                usable = False
            options = None
            if usable:
                menu_commands = list(menu.menu(record["task_type"], record["admissible"], k))
                if FALLBACK_TEXT in menu_commands:
                    raise ValueError(f"a menu holds the fallback text at {record['game_id']} step {record['step']}")
                if not menu_commands:
                    counts["rows_dropped_empty_menu"] += 1
                    usable = False
                else:
                    options = menu_commands + [FALLBACK_TEXT]
                    if len(set(options)) != len(options):
                        raise ValueError(f"repeated command on the menu at {record['game_id']} step {record['step']}: {options}")
                    if len(options) > MAX_OPTIONS:
                        raise ValueError(f"{len(options)} options at {record['game_id']} step {record['step']}, Kev allows {MAX_OPTIONS}")
            on_menu = False
            if usable:
                on_menu = expert in options[:-1]
                label = expert if on_menu else FALLBACK_TEXT
                counts["rows"] += 1
                counts["labels_fallback" if not on_menu else "labels_on_menu"] += 1
                if expert not in record["admissible"]:
                    counts["expert_not_admissible"] += 1
                rows.append(kev_row(render(record["goal"], first_obs, trace, control_state(fallbacks), window=trace_window), options, label, record, index))
            ran = record.get("command") or expert
            if position + 1 < len(trajectory):
                if not ran:
                    raise ValueError(f"{record['game_id']} step {record['step']} has no command to put in the trace of the next step")
                trace.append((ran, trajectory[position + 1]["obs"]))
            fallbacks = fallbacks + 1 if usable and not on_menu else 0
            if fallbacks_before_handover > 0 and fallbacks >= fallbacks_before_handover:   # 0 switches the hand-over off, as in harness/control.py
                fallbacks = 0
    return rows, counts


def _options(row):
    return list(row["questions"][QUESTION_ID]["criteria"])


def dedupe(rows):
    """Drop rows with identical state text, options and label, keeping the first and recording `_meta.copies`. Option order
    does not count, because `kev.data.augment` shuffles options anyway (kev/data.py:341)."""
    kept, by_key, conflicts = [], {}, defaultdict(set)
    for row in rows:
        options = tuple(sorted(_options(row)))
        label = row["questions"][QUESTION_ID]["label"]
        conflicts[(row["state"], options)].add(label)
        key = (row["state"], options, label)
        if key in by_key:
            by_key[key]["_meta"]["copies"] += 1
            continue
        row = {**row, "_meta": {**row["_meta"], "copies": 1}}
        by_key[key] = row
        kept.append(row)
    return kept, {"rows_before_dedupe": len(rows), "rows_after_dedupe": len(kept), "duplicates_removed": len(rows) - len(kept),
                  "states_with_more_than_one_label": sum(1 for labels in conflicts.values() if len(labels) > 1)}


def soft_labels(rows):
    """One row per (state text, options). Where several expert seeds gave different labels the row carries `target`, the
    share of each label, which `kev.data.materialize` passes to `kev.train.question_loss` (kev/data.py:404-409,
    kev/train.py:47-49). `label` stays as the most frequent label, the earliest on the menu on a tie, because
    `load_records` and `materialize` require it (kev/data.py:379, 402)."""
    groups: dict = {}
    for row in rows:
        groups.setdefault((row["state"], tuple(sorted(_options(row)))), []).append(row)
    out, soft = [], 0
    for group in groups.values():
        votes = Counter(row["questions"][QUESTION_ID]["label"] for row in group)
        first = group[0]
        options = _options(first)
        label = max(options, key=lambda option: (votes[option], -options.index(option)))
        question = {**first["questions"][QUESTION_ID], "label": label}
        if len(votes) > 1:
            total = sum(votes.values())
            question["target"] = {option: count / total for option, count in votes.items()}
            soft += 1
        out.append({**first, "questions": {QUESTION_ID: question}, "_meta": {**first["_meta"], "copies": len(group), "off_menu": label == FALLBACK_TEXT}})
    return out, {"rows_before_dedupe": len(rows), "rows_after_dedupe": len(out), "duplicates_removed": len(rows) - len(out), "rows_with_soft_target": soft}


def token_report(rows, tok) -> dict:
    """State and row lengths as Kev counts them, and the `--max_state` that keeps every row: Kev drops a row that does not fit
    (kev/train.py:105-110), and the long rows are the late steps of long games."""
    from kev.data import materialize
    from kev.model import MAX_BRANCH, MAX_STATE, encode, rows_of
    states, branches = [], []
    for row in rows:
        state, _, parts = rows_of(encode(tok, materialize(row), max_state=10**7, max_branch=10**7))
        states.append(len(state))
        branches.append(len(parts[0]["ids"]))
    if not states:
        return {"rows": 0}
    ordered = sorted(states)
    needed = max(MAX_STATE, max(states))
    extra = needed - MAX_STATE
    return {"rows": len(states), "state_tokens_p50": ordered[len(ordered) // 2], "state_tokens_p99": ordered[min(len(ordered) - 1, int(len(ordered) * 0.99))],
            "state_tokens_max": ordered[-1], "branch_tokens_max": max(branches), "rows_over_default_max_state": sum(1 for s in states if s > MAX_STATE),
            "max_state_needed": needed, "rows_over_row_limit_at_max_state_needed": sum(1 for s, b in zip(states, branches) if s + b > MAX_BRANCH + extra)}


def read_step_records(paths, select_split=None, counts=None):
    """Records from gzip or plain JSONL shards. `select_split` keeps one split of a mixed shard and counts what it skipped, and
    a test game stops it whatever the split says."""
    for path in paths:
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                if TEST_MARKER in str(record.get("game_id", "")):
                    raise AssertionError(f"test game in {path}: {record.get('game_id')}")
                if select_split is not None and record.get("split") != select_split:
                    if counts is not None:
                        counts["records_skipped_other_split"] += 1
                    continue
                yield record


def sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(rows, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        with open(path, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as f:
            for row in rows:
                f.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
    else:
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def report_path(out) -> Path:
    out = Path(out)
    stem = out.name.removesuffix(".gz").removesuffix(".jsonl")
    return out.with_name(f"{stem}.report.json")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--records", required=True, nargs="+", help="step record shards, or glob patterns")
    ap.add_argument("--menu", required=True, help="a file MenuModel.load reads")
    ap.add_argument("--k", required=True, help="menu length, or 'all' for every admissible command")
    ap.add_argument("--out", required=True, help="output .jsonl or .jsonl.gz; the report goes beside it")
    ap.add_argument("--split", default="train", choices=SPLITS, help="the split this export holds; training takes only 'train'")
    ap.add_argument("--select-split", action="store_true", help="the shards mix splits: keep --split and skip the rest, counting them")
    ap.add_argument("--trace-window", type=int, default=5)
    ap.add_argument("--fallbacks-before-handover", type=int, default=2)
    ap.add_argument("--dedupe", default="none", choices=["none", "hard", "soft"])
    ap.add_argument("--keep-not-won", action="store_true", help="keep rows of trajectories the checker did not mark won")
    ap.add_argument("--no-token-report", action="store_true")
    a = ap.parse_args(argv)
    k = None if a.k == "all" else int(a.k)
    paths = sorted(p for pattern in a.records for p in glob.glob(pattern))
    if not paths:
        raise SystemExit(f"no shard matches {a.records}")
    counts = Counter()
    records = read_step_records(paths, a.split if a.select_split else None, counts)
    rows, export_counts = export_rows(records, load_menu(a.menu), k, required_split=a.split, trace_window=a.trace_window,
                                      fallbacks_before_handover=a.fallbacks_before_handover, require_won=not a.keep_not_won)
    counts.update(export_counts)
    dedupe_counts = {}
    if a.dedupe == "hard":
        rows, dedupe_counts = dedupe(rows)
    elif a.dedupe == "soft":
        rows, dedupe_counts = soft_labels(rows)
    write_jsonl(rows, a.out)
    report = {"split": a.split, "k": k, "trace_window": a.trace_window, "fallbacks_before_handover": a.fallbacks_before_handover, "dedupe": a.dedupe,
              "require_won": not a.keep_not_won, "fallback_text": FALLBACK_TEXT, "question_text": QUESTION_TEXT, "counts": dict(counts), "dedupe_counts": dedupe_counts,
              "fallback_label_share": counts["labels_fallback"] / max(1, counts["rows"]), "rows_written": len(rows),
              "shards": [{"path": p, "sha256": sha256(p)} for p in paths], "menu": {"path": a.menu, "sha256": sha256(a.menu)}, "out_sha256": sha256(a.out),
              "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=LAB_SRC).stdout.strip()}
    if not a.no_token_report:
        from kev.model import load_tokenizer
        report["tokens"] = token_report(rows, load_tokenizer(BASE, revision=BASE_REVISION))
    report_path(a.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k_: report[k_] for k_ in ("split", "k", "rows_written", "counts", "dedupe_counts", "fallback_label_share")} | {"tokens": report.get("tokens")}, indent=1))


if __name__ == "__main__":
    main()
