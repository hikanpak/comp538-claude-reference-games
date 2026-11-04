"""The training set's audit, by script. Every stored shard is read again, a random sample of stored sequences is replayed in the real game, and the result goes to `audit.json`.

    nice -n 10 python -m dataset.audit --pool POOL --dir lab/results/sys1-training-set --export-dir lab/data/sys1-kev

A check that fails is written with what failed. The file's `failures` list stays empty only when every check passed. This script never changes a shard.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

from alf.env import MAX_STEPS, _make_stack, _no_help, game_spec, split_intro
from menu import MenuModel, command_form
from sys1.base import FALLBACK_TEXT

from .generate import pool_files, read_runs
from .provenance import git_state, sha256_file
from .rollouts import FAMILIES, SeededTeacher, label_fields
from .rows import dedupe_key, render_row, state_hashes
from .splits import as_sets, assert_no_leakage, assert_row_allowed, load_splits

REPLAY_PER_STRATUM = 150  # Rollout family by split.
REPLAY_GAMES_PER_WORKER = 50


def replay_game(job: tuple[str, list[dict]]) -> dict:
    """Replay stored rows of one game in a fresh game: each command's output, the state's observation and admissible commands, and the teacher's label under the same expert seed."""
    uid, rows = job
    spec = game_spec(uid)
    stack = _make_stack(MAX_STEPS)
    stack.load(spec.path)
    result = Counter()
    problems = []
    for row in rows:
        state = stack.reset()
        teacher = SeededTeacher(spec, row["expert_seed"])
        goal, first_obs = split_intro(state["feedback"])
        label = teacher.at_reset(state)
        ok = goal == row["goal"] and first_obs == row["first_obs"]
        outputs_ok, done = True, False
        for entry in row["trace"]:
            if done:
                outputs_ok = False
                break
            state, _, done = stack.step(entry["command"])
            output = state["feedback"].strip()
            outputs_ok = outputs_ok and output == entry["output"]
            if not done:
                label = teacher.after_step(state, entry["command"])
        admissible = _no_help(state["admissible_commands"])
        obs = state["feedback"].strip() if row["trace"] else first_obs
        fields = label_fields(label, admissible)
        checks = {"first_text": ok, "outputs": outputs_ok, "observation": obs == row["obs"], "admissible": admissible == row["admissible"], "label": fields["label"] == row["expert"], "label_source": fields["label_source"] == row["label_source"], "not_ended_early": not done}
        result["rows"] += 1
        for name, passed in checks.items():
            result[f"{name}_ok"] += int(passed)
        if not all(checks.values()) and len(problems) < 3:
            problems.append({"run_id": row["run_id"], "step": row["step"], "failed": [k for k, v in checks.items() if not v]})
    stack.close()
    return {"counts": dict(result), "problems": problems}


def read_jsonl_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def ratio(top, bottom):
    return top / bottom if bottom else None


def seed_disagreement(pool, split: str) -> dict:
    """On the expert's own won games: games whose seeds played different commands, and states two or more seeds reached where they gave different labels."""
    games = differ = 0
    by_task = defaultdict(lambda: [0, 0])
    states = disagree = 0
    for path in pool_files(pool, split):
        runs = [r for r in read_runs(path) if r["family"] == "expert-path" and r["won"]]
        if not runs:
            continue
        games += 1
        task = runs[0]["task_type"]
        by_task[task][0] += 1
        if len({tuple(s["command"] for s in r["steps"]) for r in runs}) > 1:
            differ += 1
            by_task[task][1] += 1
        seen: dict[bytes, tuple[set, set]] = {}
        for run in runs:
            for digest, s in zip(state_hashes(run), run["steps"]):
                if s["label"] is not None:
                    labels, seeds = seen.setdefault(digest, (set(), set()))
                    labels.add(s["label"])
                    seeds.add(run["expert_seed"])
        for labels, seeds in seen.values():
            if len(seeds) >= 2:
                states += 1
                disagree += len(labels) > 1
    return {"games_with_a_won_expert_run": games, "games_with_seeds_playing_different_commands": differ, "share_of_games": ratio(differ, games), "by_task_type": {t: {"games": v[0], "seeds_differ": v[1], "share": ratio(v[1], v[0])} for t, v in sorted(by_task.items())},
            "states_reached_by_two_or_more_seeds": states, "of_which_seeds_gave_different_labels": disagree, "share_of_states": ratio(disagree, states)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pool", required=True)
    ap.add_argument("--dir", default="lab/results/sys1-training-set")
    ap.add_argument("--export-dir", default="lab/data/sys1-kev")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--replay-per-stratum", type=int, default=REPLAY_PER_STRATUM)
    a = ap.parse_args(argv)
    pool, directory, export_dir = Path(a.pool).expanduser(), Path(a.dir), Path(a.export_dir)
    manifest = json.loads((directory / "manifest.json").read_text())
    k = manifest["k"]
    menu = MenuModel.load(manifest["menu"]["path"])
    splits = load_splits(directory)
    failures: list[str] = []

    def check(condition: bool, text: str) -> None:
        if not condition:
            failures.append(text)

    try:
        assert_no_leakage(splits)
    except AssertionError as error:
        failures.append(f"splits: {error}")
    sets = as_sets(splits)
    check(sha256_file(manifest["menu"]["path"]) == manifest["menu"]["sha256"], "the menu file differs from the manifest")

    rng = random.Random(f"audit/{a.seed}")
    reservoirs: dict[tuple, list] = defaultdict(list)
    seen_stratum = Counter()
    counts = {"train": Counter(), "dev": Counter(), "dev_heldout_rooms": Counter()}
    coverage: dict[str, Counter] = defaultdict(Counter)
    tally = Counter()
    control = Counter()
    core_plain_keys, core_text_keys = Counter(), Counter()
    expected_states = {}
    for shard in manifest["shards"]:
        path = directory / shard["path"]
        actual = sha256_file(path)
        check(actual == shard["sha256"], f"{shard['path']}: SHA-256 differs from the manifest")
        check(path.stat().st_size < 50 * 1024 * 1024, f"{shard['path']}: 50 MB or more")
        split = shard["split"]
        rows_here = 0
        for row in read_jsonl_gz(path):
            rows_here += 1
            label = row["expert"]
            family, task = row["family"], row["task_type"]
            try:
                assert_row_allowed(row, sets)
            except AssertionError as error:
                failures.append(f"{shard['path']}: {error}")
            counts[split]["rows"] += 1
            counts[split][f"family/{family}"] += 1
            counts[split][f"task_type/{task}"] += 1
            counts[split][f"family_task/{family}/{task}"] += 1
            if row["split"] != split:
                failures.append(f"{shard['path']}: row of split {row['split']}")
            check(row["tracking"] == "always", f"{row['run_id']}: tracking {row['tracking']}")
            check(row.get("step") == len(row["trace"]), f"{row['run_id']} step {row['step']}: trace has {len(row['trace'])} entries")
            if row["trace"]:
                check(row["trace"][-1]["output"] == row["obs"], f"{row['run_id']} step {row['step']}: the last output is not the observation")
            else:
                check(row["obs"] == row["first_obs"], f"{row['run_id']}: step 0 observation is not the first observation")
            tally["rows_read"] += 1
            if label is None:
                tally["rows_flagged"] += 1
                check(split != "train", f"{row['run_id']} step {row['step']}: a flagged label in a training row")
                check(row.get("menu_rank") is None, f"{row['run_id']} step {row['step']}: a flagged label with a menu rank")
            else:
                tally["labels_checked"] += 1
                if label not in row["admissible"]:
                    tally["labels_not_admissible"] += 1
                    failures.append(f"{row['run_id']} step {row['step']}: label {label!r} is not admissible")
                rank = menu.rank(task, row["admissible"], label)
                check(rank == row["menu_rank"], f"{row['run_id']} step {row['step']}: menu_rank {row['menu_rank']}, the menu says {rank}")
                if row["won"]:
                    on = rank is not None and rank <= k
                    for key in (f"{split}/{family}", f"{split}/{family}/{task}", f"{split}/all", f"{split}/task/{task}"):
                        coverage[key]["rows"] += 1
                        coverage[key]["on_menu"] += int(on)
            if split == "train":
                check(row["won"], f"{row['run_id']}: a training row from a lost rollout")
                text = render_row(row)
                core_text_keys[hashlib.blake2b(repr((text, sorted(row["admissible"]), label)).encode(), digest_size=12).digest()] += 1
                state = row["control_state"]
                check(state["controller"] == "sys1" and state["llm_rounds_left"] == 0 and state["menu_run"] == 0, f"{row['run_id']} step {row['step']}: Sys1 is not asked in control state {state}")
                varied = row["control_varied"]
                control["varied" if varied else "plain"] += 1
                control[f"fallbacks_in_row={state['fallbacks_in_row']}"] += 1
                shown = row["trace"][-5:]
                control["rows_showing_an_llm_mark"] += int(any(e["actor"] == "llm" for e in shown))
                if not varied:
                    check(state["fallbacks_in_row"] == 0 and all(e["actor"] != "llm" for e in row["trace"]), f"{row['run_id']} step {row['step']}: plain row with a varied state")
                expected_states[(row["run_id"], row["step"])] = hashlib.blake2b(text.encode(), digest_size=12).digest()
            key = (split, family)
            seen_stratum[key] += 1
            pick = {f: row[f] for f in ("game_id", "goal", "first_obs", "trace", "obs", "admissible", "expert", "expert_seed", "label_source", "run_id", "step")}
            if len(reservoirs[key]) < a.replay_per_stratum:
                reservoirs[key].append(pick)
            else:
                j = rng.randrange(seen_stratum[key])
                if j < a.replay_per_stratum:
                    reservoirs[key][j] = pick
        check(rows_here == shard["rows"], f"{shard['path']}: {rows_here} rows, the manifest says {shard['rows']}")
    # Counts against the manifest.
    check(counts["train"]["rows"] == manifest["core"]["rows"], "core rows differ from the manifest")
    check(counts["train"]["rows"] <= 100_000, "core is over 100,000 rows")
    for split in ("dev", "dev_heldout_rooms"):
        check(counts[split]["rows"] == manifest["dev_splits"][split]["rows"], f"{split} rows differ from the manifest")
    duplicates = sum(v - 1 for v in core_text_keys.values() if v > 1)

    # Replay.
    by_game: dict[str, list[dict]] = defaultdict(list)
    for key, rows in reservoirs.items():
        for row in rows:
            by_game[row["game_id"]].append(row)
    replay = Counter()
    problems = []
    # A fresh pool per chunk keeps workers from growing. `max_tasks_per_child` with `map` hangs on Python 3.12: a worker that exits is replaced only when the pool counts no idle worker, and
    # every task finished before it raises that count, so all workers leave and the queue never drains.
    jobs = sorted(by_game.items())
    for start in range(0, len(jobs), a.workers * REPLAY_GAMES_PER_WORKER):
        with ProcessPoolExecutor(max_workers=a.workers, mp_context=get_context("spawn")) as executor:
            for result in executor.map(replay_game, jobs[start:start + a.workers * REPLAY_GAMES_PER_WORKER]):
                replay.update(result["counts"])
                problems += result["problems"]
        print(json.dumps({"replayed_games": min(start + a.workers * REPLAY_GAMES_PER_WORKER, len(jobs)), "of": len(jobs)}), flush=True)
    for name in ("first_text", "outputs", "observation", "admissible", "label", "label_source", "not_ended_early"):
        check(replay[f"{name}_ok"] == replay["rows"], f"replay: {name} matched {replay[f'{name}_ok']} of {replay['rows']}")

    # The export against the stored rows.
    exports = {}
    report_path = export_dir / "train.report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        rows = 0
        for row in read_jsonl_gz(export_dir / "train.jsonl.gz"):
            rows += 1
            meta = row["_meta"]
            if meta["split"] != "train":
                failures.append(f"export row of split {meta['split']}")
            run_id = f"{meta['game_id']}#{meta['id'][len(meta['game_id']) + 1:].rsplit('#', 1)[0]}"
            want = expected_states.get((run_id, meta["step"]))
            if want is None:
                failures.append(f"export row {meta['id']} has no stored row")
            elif hashlib.blake2b(row["state"].encode(), digest_size=12).digest() != want:
                failures.append(f"export row {meta['id']}: state text differs from render_prompt on the stored row")
            question = row["questions"]["command"]
            options = list(question["criteria"])
            check(options[-1] == FALLBACK_TEXT, f"export row {meta['id']}: fallback is not last")
            check(len(options) <= k + 1, f"export row {meta['id']}: {len(options)} options")
            check(question["label"] in options, f"export row {meta['id']}: label not an option")
        check(rows == report["rows_written"], f"train.jsonl.gz has {rows} rows, its report says {report['rows_written']}")
        check(report["counts"]["rows"] == counts["train"]["rows"], f"the export made {report['counts']['rows']} rows from core's {counts['train']['rows']}: it dropped rows")
        merged = report["dedupe_counts"]["duplicates_removed"]
        check(rows + merged == counts["train"]["rows"], f"the export has {rows} rows, merged {merged}, and core has {counts['train']['rows']}")
        for name in ("train", "dev", "dev_heldout_rooms", "speedtest-200"):
            path = export_dir / f"{name}.report.json"
            if path.exists():
                r = json.loads(path.read_text())
                exports[name] = {"rows_written": r["rows_written"], "fallback_label_share": r["fallback_label_share"], "tokens": r.get("tokens"), "counts": r["counts"], "dedupe_counts": r.get("dedupe_counts"), "out_sha256": r["out_sha256"]}
    else:
        failures.append(f"no export at {report_path}")

    def cov(prefix: str) -> dict:
        return {key[len(prefix):]: {"rows": v["rows"], "on_menu_share": ratio(v["on_menu"], v["rows"])} for key, v in sorted(coverage.items()) if key.startswith(prefix)}

    audit = {"topic": "sys1-training-set-audit", "seed": a.seed, "k": k, **git_state(), "failures": failures[:200], "failures_total": len(failures), "passed": not failures,
             "counts": {split: dict(sorted(c.items())) for split, c in counts.items()},
             "labels": {"checked": tally["labels_checked"], "not_admissible": tally["labels_not_admissible"], "rows_flagged_in_dev_splits": tally["rows_flagged"]},
             "flagged_labels_in_train_won_runs": manifest["train_pool"]["flagged"], "failed_rollouts": {"train_by_family_and_outcome": manifest["train_pool"]["runs"], "train_by_task": manifest["train_pool"]["runs_by_task"]},
             "replay": {"rows": replay["rows"], "per_stratum_asked": a.replay_per_stratum, "strata": {f"{s}/{f}": len(v) for (s, f), v in sorted(reservoirs.items())}, "matched": {k_: v for k_, v in sorted(replay.items()) if k_.endswith("_ok")}, "problems_shown": problems[:10]},
             "label_coverage_at_k": {split: {"all": cov(f"{split}/all").get("", None), "by_family_and_task_type": cov(f"{split}/"), "by_task_type": cov(f"{split}/task/")} for split in counts},
             "expert_seed_disagreement": {split: seed_disagreement(pool, split) for split in ("train", "dev", "dev_heldout_rooms")},
             "duplicates": {"train_pool_rows_before_dedupe": manifest["train_pool"]["rows_candidate"], "train_pool_rows_after_dedupe": manifest["train_pool"]["rows_after_dedupe"], "train_pool_duplicate_rate_by_family": {f: ratio(manifest["train_pool"]["copies_dropped_by_dedupe"].get(f, 0), manifest["train_pool"]["rows_candidate"].get(f, 0)) for f in FAMILIES}, "core_rows_identical_in_text_commands_and_label": duplicates},
             "control_state": dict(sorted(control.items())), "control_varied_share": ratio(control["varied"], control["varied"] + control["plain"]), "exports": exports, "repo_bytes": manifest["bytes_total"]}
    (directory / "audit.json").write_text(json.dumps(audit, indent=1) + "\n")
    print(json.dumps({k_: audit[k_] for k_ in ("passed", "failures_total", "labels", "replay", "duplicates", "control_state", "control_varied_share", "repo_bytes")}, indent=1)[:6000])
    if failures:
        print("FAILURES", *failures[:30], sep="\n")


if __name__ == "__main__":
    main()
