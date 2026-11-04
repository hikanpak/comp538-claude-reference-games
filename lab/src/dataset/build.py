"""Build the delivered files from the pool: the `core` training rows, the two dev splits in full, and the manifest.

    python -m dataset.build --pool POOL --menu lab/results/menu-statistics/menu.json --k K --out lab/results/sys1-training-set

Rule for a training row. Its rollout was won, its label is usable (not flagged random, inadmissible or error), and its game is in `train`. Rows are deduplicated on the plain text Sys1 sees,
the admissible commands and the label, the first copy kept in the order expert-path, recovery, loop-break. Then each family takes its share of `--core-size`, and inside a family each task type
takes the share of `train` games it has, drawn uniformly by `--seed`. A task type short of rows leaves its quota to the others of that family. `CONTROL_SHARE` of the drawn rows show a varied
control state (`dataset/rows.py`), taken by hash among the rows where a run could show one.

A dev split holds every row of every rollout, with flags and lost rollouts kept and marked, in the plain control state a Sys1-alone evaluation shows.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

from menu import MenuModel

from .generate import pool_files, read_runs
from .provenance import provenance, sha256_file
from .rollouts import FAMILIES
from .rows import CONTROL_SHARE, choose_varied, dedupe_key, family_steps, make_row, menu_flags, outputs_of, render_row, state_hashes, varied_path, varied_row, varied_value
from .splits import as_sets, assert_no_leakage, assert_row_allowed, load_splits, summary, task_type_of

DEFAULT_SHARES = {"expert-path": 0.5, "recovery": 0.3, "loop-break": 0.2}
ROWS_PER_SHARD = 20_000
SHARD_LIMIT_BYTES = 50 * 1024 * 1024


class ShardWriter:
    """Gzip JSONL shards with no timestamp in the header, so equal rows give equal files. A new shard starts every `rows_per_shard` rows."""

    def __init__(self, directory: Path, stem: str, rows_per_shard: int = ROWS_PER_SHARD):
        self.directory, self.stem, self.rows_per_shard = directory, stem, rows_per_shard
        self.paths: list[Path] = []
        self.rows_in_shard: list[int] = []
        self._raw = self._gz = None

    def _open(self) -> None:
        self.close()
        path = self.directory / f"{self.stem}-{len(self.paths):04d}.jsonl.gz"
        self.paths.append(path)
        self.rows_in_shard.append(0)
        self._raw = open(path, "wb")
        self._gz = gzip.GzipFile(filename="", mode="wb", fileobj=self._raw, mtime=0, compresslevel=9)

    def write(self, row: dict) -> None:
        if self._gz is None or self.rows_in_shard[-1] >= self.rows_per_shard:
            self._open()
        self._gz.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
        self.rows_in_shard[-1] += 1

    def close(self) -> None:
        if self._gz is not None:
            self._gz.close()
            self._raw.close()
            self._gz = self._raw = None

    def manifest_entries(self, split: str) -> list[dict]:
        return [{"path": p.name, "sha256": sha256_file(p), "split": split, "rows": n, "bytes": p.stat().st_size} for p, n in zip(self.paths, self.rows_in_shard)]


def scan_pool(pool, split: str, sets: dict) -> dict:
    """First pass over a split's rollouts: the labels each state got, what is flagged, what was lost, and one light record per candidate row."""
    state_labels: dict[bytes, tuple[set, set]] = {}
    candidates = []
    counts = {"runs": Counter(), "runs_by_task": Counter(), "rows_candidate": Counter(), "flagged": Counter(), "flagged_by_task": Counter(), "rows_of_lost_runs": Counter(), "formed": Counter()}
    files = pool_files(pool, split)
    for file_index, path in enumerate(files):
        for run_index, run in enumerate(read_runs(path)):
            assert_row_allowed({"split": run["split"], "game_id": run["game_id"]}, sets)
            family, task = run["family"], run["task_type"]
            outcome = "won" if run["won"] else "lost"
            counts["runs"][f"{family}/{outcome}"] += 1
            counts["runs_by_task"][f"{family}/{task}/{outcome}"] += 1
            if family == "loop-break" and not run["perturb"]["formed"]:
                counts["formed"]["loop-break/no repeat formed"] += 1
            steps = family_steps(run)
            if not run["won"]:
                counts["rows_of_lost_runs"][family] += len(steps)
                continue
            hashes = state_hashes(run)
            for step, s in enumerate(run["steps"]):
                if s["label"] is not None:
                    labels, seeds = state_labels.setdefault(hashes[step], (set(), set()))
                    labels.add(s["label"])
                    seeds.add(run["expert_seed"])
            outputs = outputs_of(run)
            for step in steps:
                s = run["steps"][step]
                if s["label"] is None:
                    counts["flagged"][f"{family}/{s['label_source']}"] += 1
                    counts["flagged_by_task"][f"{task}/{s['label_source']}"] += 1
                    continue
                counts["rows_candidate"][family] += 1
                candidates.append((FAMILIES.index(family), task, file_index, run_index, step, dedupe_key(run, step, outputs)))
    return {"files": files, "state_labels": state_labels, "candidates": candidates, "counts": counts}


def acceptable_at(state_labels, digest: bytes, label: str | None):
    labels, seeds = state_labels.get(digest, (set(), set()))
    if len(seeds) < 2:
        return None
    return sorted(labels - {label})


def deduplicate(candidates: list[tuple]) -> tuple[list[tuple], Counter, Counter]:
    """Keep the first row of each key, with families in order. -> (kept rows, rows kept per family, copies dropped per family)."""
    seen = set()
    kept, kept_counts, dropped = [], Counter(), Counter()
    for row in sorted(candidates, key=lambda c: c[:5]):
        family = FAMILIES[row[0]]
        if row[5] in seen:
            dropped[family] += 1
            continue
        seen.add(row[5])
        kept.append(row)
        kept_counts[family] += 1
    return kept, kept_counts, dropped


def proportional_quotas(total: int, shares: dict[str, float]) -> dict[str, int]:
    """Whole numbers that sum to `total`, largest remainders first, ties by name."""
    norm = sum(shares.values())
    raw = {k: total * v / norm for k, v in shares.items()}
    out = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: (-(raw[k] - out[k]), k))[:total - sum(out.values())]:
        out[k] += 1
    return out


def sample_core(deduped: list[tuple], core_size: int, family_shares: dict[str, float], task_games: dict[str, int], seed: int) -> tuple[list[tuple], dict]:
    """The stated rule. Each family has a quota, each task type a share of that quota by its number of train games, and a short task type's rest goes to the others of the family."""
    strata: dict[tuple, list] = defaultdict(list)
    for row in deduped:
        strata[(FAMILIES[row[0]], row[1])].append(row)
    rng = random.Random(f"core/{seed}")
    picked, plan = [], {}
    family_quotas = proportional_quotas(core_size, family_shares)
    for family in FAMILIES:
        left = family_quotas[family]
        tasks = {t: n for t, n in task_games.items() if strata[(family, t)]}
        taken: dict[str, int] = {}
        while left > 0 and tasks:
            quotas = proportional_quotas(left, tasks)
            short = {t for t, q in quotas.items() if len(strata[(family, t)]) - taken.get(t, 0) <= q}
            if not short:
                for t, q in quotas.items():
                    taken[t] = taken.get(t, 0) + q
                left = 0
                break
            for t in short:
                extra = len(strata[(family, t)]) - taken.get(t, 0)
                taken[t] = taken.get(t, 0) + extra
                left -= extra
                del tasks[t]
        for task in sorted(taken):
            rows = sorted(strata[(family, task)], key=lambda r: r[5])
            picked += rng.sample(rows, taken[task])
        plan[family] = {"quota": family_quotas[family], "taken": sum(taken.values()), "by_task_type": dict(sorted(taken.items())), "unfilled": family_quotas[family] - sum(taken.values())}
    return picked, plan


def write_core(pool: Path, scan: dict, picked: list[tuple], menu: MenuModel, k, seed: int, out: Path, sets: dict) -> tuple[ShardWriter, dict]:
    by_file: dict[int, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    for row in picked:
        by_file[row[2]][row[3]].add(row[4])
    possible = []  # Rows a live run could show with a varied control state.
    for file_index in sorted(by_file):
        runs = read_runs(scan["files"][file_index])
        for run_index in sorted(by_file[file_index]):
            run = runs[run_index]
            flags = menu_flags(run, menu, k)
            for step in by_file[file_index][run_index]:
                if varied_path(run, step, menu, k, seed, flags=flags) is not None:
                    possible.append((varied_value(run["run_id"], step, seed), file_index, run_index, step))
    varied = {entry[1:] for entry in choose_varied(possible, len(picked))}
    writer = ShardWriter(out, "core")
    counts = Counter(control_varied_possible=len(possible))
    for file_index in sorted(by_file):
        runs = read_runs(scan["files"][file_index])
        for run_index in sorted(by_file[file_index]):
            run = runs[run_index]
            hashes, outputs = state_hashes(run), outputs_of(run)
            for step in sorted(by_file[file_index][run_index]):
                s = run["steps"][step]
                row = make_row(run, step, outputs, acceptable_at(scan["state_labels"], hashes[step], s["label"]), menu.rank(run["task_type"], s["admissible"], s["label"]))
                if row["split"] != "train" or row["expert"] is None or not row["won"]:
                    raise AssertionError(f"not a training row: {row['run_id']} step {step}")
                assert_row_allowed(row, sets)
                if (file_index, run_index, step) in varied:
                    row = varied_row(row, run, menu, k, seed)
                    if not row["control_varied"]:
                        raise AssertionError(f"no varied path on the second pass: {row['run_id']} step {step}")
                writer.write(row)
                counts[f"family/{row['family']}"] += 1
                counts[f"task_type/{row['task_type']}"] += 1
                counts[f"family_task/{row['family']}/{row['task_type']}"] += 1
                counts["control_varied"] += int(row["control_varied"])
                counts["rows"] += 1
                counts["off_menu_at_k"] += int(k is not None and (row["menu_rank"] is None or row["menu_rank"] > k))
    writer.close()
    return writer, dict(sorted(counts.items()))


def write_dev(pool: Path, split: str, menu: MenuModel, scan: dict, out: Path, sets: dict) -> tuple[ShardWriter, dict]:
    writer = ShardWriter(out, split)
    counts = Counter()
    for path in scan["files"]:
        for run in read_runs(path):
            hashes, outputs = state_hashes(run), outputs_of(run)
            for step in family_steps(run):
                s = run["steps"][step]
                row = make_row(run, step, outputs, acceptable_at(scan["state_labels"], hashes[step], s["label"]), menu.rank(run["task_type"], s["admissible"], s["label"]))
                assert_row_allowed(row, sets)
                writer.write(row)
                counts["rows"] += 1
                counts[f"family/{row['family']}"] += 1
                counts[f"task_type/{row['task_type']}"] += 1
                counts[f"rows_of_lost_runs"] += int(not row["won"])
                counts["flagged"] += int(row["expert"] is None)
    writer.close()
    return writer, dict(sorted(counts.items()))


def pool_digest(pool: Path) -> str:
    digest = hashlib.sha256()
    for split in ("dev", "dev_heldout_rooms", "train"):
        for path in pool_files(pool, split):
            digest.update(f"{split}/{path.name} {sha256_file(path)}\n".encode())
    return digest.hexdigest()


def generation_seconds(pool: Path) -> dict:
    sessions = []
    path = pool / "sessions.jsonl"
    if path.exists():
        sessions = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    timing = pool / "timing.jsonl"
    game_seconds = sum(json.loads(line)["seconds"] for line in timing.read_text().splitlines()) if timing.exists() else None
    return {"wall_seconds": sum(s.get("wall_seconds", 0) for s in sessions), "sessions": len(sessions), "worker_seconds_summed": game_seconds, "workers": sessions[-1]["workers"] if sessions else None, "python_hash_seed": sessions[-1].get("python_hash_seed") if sessions else None}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pool", required=True)
    ap.add_argument("--menu", default="lab/results/menu-statistics/menu.json")
    ap.add_argument("--k", required=True, type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--core-size", type=int, default=100_000)
    ap.add_argument("--splits-dir", default="lab/results/sys1-training-set")
    ap.add_argument("--out", default="lab/results/sys1-training-set")
    ap.add_argument("--shares", default=",".join(f"{k}={v}" for k, v in DEFAULT_SHARES.items()))
    ap.add_argument("--allow-partial-pool", action="store_true")
    a = ap.parse_args(argv)
    t0 = time.time()
    pool, out = Path(a.pool).expanduser(), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for stale in [*out.glob("core-*.jsonl.gz"), *out.glob("dev-*.jsonl.gz"), *out.glob("dev_heldout_rooms-*.jsonl.gz")]:
        stale.unlink()
    splits = load_splits(a.splits_dir)
    assert_no_leakage(splits)
    sets = as_sets(splits)
    done = {name: len(pool_files(pool, name)) for name in ("train", "dev", "dev_heldout_rooms")}
    partial = any(done[name] != len(splits[name]) for name in done)
    if partial and not a.allow_partial_pool:
        raise SystemExit(f"the pool is not complete: {done}")
    menu = MenuModel.load(a.menu)
    if menu.rule != "form":
        raise SystemExit("the training set is built on the plan's rule, `form`")
    shares = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in a.shares.split(",")}
    if set(shares) != set(FAMILIES):
        raise SystemExit(f"--shares must name {FAMILIES}")
    config = json.loads((pool / "config.json").read_text())

    scan = scan_pool(pool, "train", sets)
    deduped, kept, dropped = deduplicate(scan["candidates"])
    task_games = Counter(task_type_of(u) for u in splits["train"])
    picked, plan = sample_core(deduped, a.core_size, shares, dict(task_games), a.seed)
    print(json.dumps({"seconds": round(time.time() - t0), "candidates": scan["counts"]["rows_candidate"], "kept": kept, "dropped": dropped, "plan": plan}, indent=1, default=dict), flush=True)
    writer, core_counts = write_core(pool, scan, picked, menu, a.k, a.seed, out, sets)
    shards = writer.manifest_entries("train")
    train_counts = {k: dict(sorted(v.items())) for k, v in scan["counts"].items()}
    state_counts = len(scan["state_labels"])
    del deduped, picked
    candidates = scan.pop("candidates")
    labels_per_state = Counter(len(v[0]) for v in scan["state_labels"].values() if len(v[1]) >= 2)
    del candidates

    dev_counts = {}
    for split in ("dev", "dev_heldout_rooms"):
        dev_scan = scan_pool(pool, split, sets)
        dev_writer, dev_counts[split] = write_dev(pool, split, menu, dev_scan, out, sets)
        dev_counts[split]["runs"] = dict(sorted(dev_scan["counts"]["runs"].items()))
        dev_counts[split]["runs_by_task"] = dict(sorted(dev_scan["counts"]["runs_by_task"].items()))
        shards += dev_writer.manifest_entries(split)
    too_big = [s for s in shards if s["bytes"] >= SHARD_LIMIT_BYTES]
    if too_big:
        raise SystemExit(f"shards over 50 MB: {too_big}")
    manifest = {"topic": "sys1-training-set", "created": time.strftime("%Y-%m-%d %H:%M:%S %z"), "partial_pool": partial, "seed": a.seed, "k": a.k, "core_size_asked": a.core_size, "family_shares_asked": shares, "control_share_asked": CONTROL_SHARE,
                "shards": shards, "bytes_total": sum(s["bytes"] for s in shards), "splits": summary(splits), "generation": {"config": config, **generation_seconds(pool), "pool_files": done, "pool_sha256": pool_digest(pool), "pool_bytes": sum(p.stat().st_size for s in done for p in pool_files(pool, s))},
                "menu": {"path": a.menu, "sha256": sha256_file(a.menu), "rule": menu.rule},
                "train_pool": {**train_counts, "rows_after_dedupe": dict(sorted(kept.items())), "copies_dropped_by_dedupe": dict(sorted(dropped.items())), "states_with_labels": state_counts, "states_reached_by_two_or_more_seeds_by_number_of_labels": {str(k): v for k, v in sorted(labels_per_state.items())}},
                "core": {**core_counts, "sampling_plan": plan}, "dev_splits": dev_counts, **provenance()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({"seconds": round(time.time() - t0), "core": core_counts, "dev": {k: {x: y for x, y in v.items() if x != "runs_by_task"} for k, v in dev_counts.items()}, "bytes_total": manifest["bytes_total"], "shards": len(shards)}, indent=1), flush=True)


if __name__ == "__main__":
    main()
