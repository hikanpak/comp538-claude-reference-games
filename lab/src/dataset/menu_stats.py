"""Fit the menus on the pool's training games and measure them on the dev splits.

    python -m dataset.menu_stats --pool POOL --out lab/results/menu-statistics

The menu is fitted on `expert-path` rows of won rollouts on `train` games, which is what "how often the expert used it" means. `recovery` and `loop-break` rows are measured and not
fitted on. A row counts when its rollout was won and its label is usable, which is the rule of the training rows.

Writes `menu.json` (the plan's rule), `menu-kind.json` (the weaker rule), `stats.json` and, with `--k`, `REPORT.md`, which holds the tables and the reasons read from `--why`.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from menu import MenuModel, command_form, fit

from .generate import pool_files, read_runs
from .provenance import provenance
from .rollouts import FAMILIES
from .rows import family_steps
from .splits import as_sets, assert_no_leakage, assert_row_allowed, load_splits

KS = (5, 10, 20, None)
CURVE_KS = (5, 8, 10, 12, 15, 18, 20, 22, 25, 30, 35, 40, None)  # A finer range for rule `form`, because coverage does not grow evenly with k.
MEASURED = ("dev", "dev_heldout_rooms")


def k_name(k) -> str:
    return "all" if k is None else str(k)


def fit_records(pool, sets: dict):
    """Light records of the expert's own won games on training games, checked one by one against the split lists."""
    for path in pool_files(pool, "train"):
        for run in read_runs(path):
            if run["family"] != "expert-path" or not run["won"]:
                continue
            for step, s in enumerate(run["steps"]):
                record = {"split": run["split"], "game_id": run["game_id"], "task_type": run["task_type"], "step": step, "expert": s["label"]}
                assert_row_allowed(record, sets)
                yield record


class Tally:
    """Coverage and slots for one rule at one k, under any grouping."""

    def __init__(self):
        self.steps, self.on_menu, self.slots, self.go_to_slots, self.menu_length = Counter(), Counter(), Counter(), Counter(), Counter()

    def add(self, keys, rank: int, k, ordered: list[str]) -> None:
        top = ordered if k is None else ordered[:k]
        go = sum(c.startswith("go to ") for c in top)
        for key in keys:
            self.steps[key] += 1
            self.on_menu[key] += int(k is None or rank <= k)
            self.slots[key] += len(top)
            self.go_to_slots[key] += go

    def row(self, key) -> dict:
        n = self.steps[key]
        if not n:
            return {"steps": 0}
        return {"steps": n, "coverage": self.on_menu[key] / n, "fallback_rate_implied": 1 - self.on_menu[key] / n, "mean_menu_length": self.slots[key] / n,
                "go_to_slot_share": self.go_to_slots[key] / self.slots[key] if self.slots[key] else None}

    def table(self, prefix: str) -> dict:
        out = {}
        for key in sorted(k for k in self.steps if isinstance(k, tuple) and k[0] == prefix):
            name = "/".join(key[1:])
            out[name] = self.row(key)
        return out


def measure(pool, split: str, menus: dict[str, MenuModel], sets: dict) -> dict:
    tallies = {rule: {k: Tally() for k in KS} for rule in menus}
    whole_runs = {rule: {k: Counter() for k in KS} for rule in menus}
    inserted = {rule: {k_name(k): Counter() for k in KS} for rule in menus}
    curve = {k_name(k): Counter() for k in CURVE_KS}
    runs = Counter()
    admissible = Counter()
    largest = total = 0
    label_kinds, admissible_kinds, flagged = Counter(), Counter(), Counter()
    for path in pool_files(pool, split):
        for run in read_runs(path):
            task, family = run["task_type"], run["family"]
            runs[f"{family}/{'won' if run['won'] else 'lost'}"] += 1
            if not run["won"]:
                continue
            on_whole = {rule: {k: True for k in KS} for rule in menus}
            curve_whole = {k: True for k in CURVE_KS}
            if run["perturb"]:
                for t in run["steps"][run["perturb"]["start"]:run["perturb"]["end"]]:
                    for rule, model in menus.items():
                        ordered = model.menu(task, t["admissible"], None)
                        rank = ordered.index(t["command"]) + 1 if t["command"] in ordered else None
                        for k in KS:
                            inserted[rule][k_name(k)][f"{family}/commands"] += 1
                            inserted[rule][k_name(k)][f"{family}/on_menu"] += int(rank is not None and (k is None or rank <= k))
            for step in family_steps(run):
                s = run["steps"][step]
                assert_row_allowed({"split": run["split"], "game_id": run["game_id"]}, sets)
                n = len(s["admissible"])
                total += n
                largest = max(largest, n)
                admissible["steps"] += 1
                for c in s["admissible"]:
                    admissible_kinds[command_form(c)[0]] += 1
                label = s["label"]
                if label is None:
                    flagged[f"{family}/{s['label_source']}"] += 1
                    continue
                kind = command_form(label)[0]
                label_kinds[f"{family}/{kind}"] += 1
                for rule, model in menus.items():
                    ordered = model.menu(task, s["admissible"], None)
                    rank = ordered.index(label) + 1
                    if rule == "form":
                        for k in CURVE_KS:
                            on = k is None or rank <= k
                            c = curve[k_name(k)]
                            c["rows"] += 1
                            c["rows_on_menu"] += int(on)
                            c["menu_slots"] += len(ordered) if k is None else min(k, len(ordered))
                            if family == "expert-path":
                                c["expert_rows"] += 1
                                c["expert_rows_on_menu"] += int(on)
                                curve_whole[k] = curve_whole[k] and on
                    keys = [("overall",), ("task", task), ("family", family), ("task_family", task, family), ("label_kind", kind)]
                    for k in KS:
                        tallies[rule][k].add(keys, rank, k, ordered)
                        if k is not None and rank > k:
                            on_whole[rule][k] = False
            if family == "expert-path":
                for k in CURVE_KS:
                    c = curve[k_name(k)]
                    c["expert_runs"] += 1
                    c["expert_runs_all_on_menu"] += int(curve_whole[k])
                    c[f"expert_runs/{task}"] += 1
                    c[f"expert_runs_all_on_menu/{task}"] += int(curve_whole[k])
                for rule in menus:
                    for k in KS:
                        whole_runs[rule][k]["runs"] += 1
                        whole_runs[rule][k]["all_on_menu"] += int(on_whole[rule][k])
    out = {"games": len(pool_files(pool, split)), "runs": dict(sorted(runs.items())), "rows_counted": sum(tallies[next(iter(menus))][None].steps[("family", f)] for f in FAMILIES),
           "admissible_per_step": {"mean": total / max(1, admissible["steps"]), "max": largest, "steps": admissible["steps"]},
           "admissible_kind_share": {k: v / max(1, total) for k, v in admissible_kinds.most_common()},
           "label_kinds": dict(sorted(label_kinds.items())), "flagged_labels_won_runs": dict(sorted(flagged.items())),
           "inserted_commands_on_menu": {rule: {k: dict(sorted(c.items())) for k, c in by_k.items()} for rule, by_k in inserted.items()},
           "form_curve": {k: dict(sorted(c.items())) for k, c in curve.items()}, "rules": {}}
    for rule in menus:
        out["rules"][rule] = {}
        for k in KS:
            t = tallies[rule][k]
            whole = whole_runs[rule][k]
            out["rules"][rule][k_name(k)] = {"overall": t.row(("overall",)), "by_task_type": t.table("task"), "by_family": t.table("family"), "by_task_type_and_family": t.table("task_family"),
                                             "by_label_kind": t.table("label_kind"), "expert_runs_with_every_label_on_menu": whole["all_on_menu"] / max(1, whole["runs"]), "expert_runs": whole["runs"]}
    return out


def pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}"


def report(stats: dict, k_choice, why: str) -> str:
    lines = ["# Menu statistics", "", why.strip(), "", f"Empirical. Every number comes from `stats.json` beside this file, rollouts of seed {stats['pool_config']['seed']}, expert under `tracking=always`, git commit `{stats['git_commit']}` (dirty: {stats['git_dirty_lab_src']}). A row counts when its rollout was won and its label is usable. Coverage is the share of steps with the expert's command on the menu, and the implied fallback rate is one minus coverage. `form` is the plan's rule and `kind` the weaker one.", ""]
    for split in MEASURED:
        s = stats["splits"][split]
        lines += [f"## {split}: {s['games']} games, {s['rows_counted']} rows", "", f"Admissible commands per step: mean {s['admissible_per_step']['mean']:.1f}, maximum {s['admissible_per_step']['max']}.", "",
                  "| Rule | k | Coverage % | Implied fallback % | `go to` share of menu slots % | Expert games with every label on the menu % |", "| --- | --- | --- | --- | --- | --- |"]
        for rule in s["rules"]:
            for k in s["rules"][rule]:
                r = s["rules"][rule][k]
                lines.append(f"| {rule} | {k} | {pct(r['overall']['coverage'])} | {pct(r['overall']['fallback_rate_implied'])} | {pct(r['overall']['go_to_slot_share'])} | {pct(r['expert_runs_with_every_label_on_menu'])} |")
        lines += ["", "Coverage % by family, rule `form`:", "", "| Family | k=5 | k=10 | k=20 | k=all |", "| --- | --- | --- | --- | --- |"]
        form = s["rules"]["form"]
        for family in FAMILIES:
            lines.append("| " + family + " | " + " | ".join(pct(form[k]["by_family"].get(family, {}).get("coverage")) for k in ("5", "10", "20", "all")) + " |")
        lines += ["", "Coverage % by family, rule `kind`:", "", "| Family | k=5 | k=10 | k=20 | k=all |", "| --- | --- | --- | --- | --- |"]
        weak = s["rules"]["kind"]
        for family in FAMILIES:
            lines.append("| " + family + " | " + " | ".join(pct(weak[k]["by_family"].get(family, {}).get("coverage")) for k in ("5", "10", "20", "all")) + " |")
        lines += ["", "Coverage % by task type, rule `form`:", "", "| Task type | k=5 | k=10 | k=20 | k=all |", "| --- | --- | --- | --- | --- |"]
        for task in form["all"]["by_task_type"]:
            lines.append("| " + task + " | " + " | ".join(pct(form[k]["by_task_type"].get(task, {}).get("coverage")) for k in ("5", "10", "20", "all")) + " |")
        lines += ["", "Coverage % by the kind of the expert's command, rule `form`, with how many rows have that kind:", "", "| Command kind | Rows | k=5 | k=10 | k=20 |", "| --- | --- | --- | --- | --- |"]
        for kind, row in sorted(form["all"]["by_label_kind"].items(), key=lambda kv: -kv[1]["steps"]):
            lines.append("| " + kind + f" | {row['steps']} | " + " | ".join(pct(form[k]["by_label_kind"].get(kind, {}).get("coverage")) for k in ("5", "10", "20")) + " |")
        tasks = sorted(form["all"]["by_task_type"])
        lines += ["", "Rule `form` along a finer range of k. Coverage counts all families, the expert columns count the expert's own won games, and a game counts as whole when every usable label of it is on the menu at its state. Those games are the ones a Sys1 restricted to the menu could finish by following the expert:", "",
                  "| k | Mean menu length | Coverage % | Expert-path coverage % | Whole expert games % | " + " | ".join(f"Whole {t} %" for t in tasks) + " |", "| " + " | ".join("---" for _ in range(5 + len(tasks))) + " |"]
        for k, c in s["form_curve"].items():
            cells = [f"{c['menu_slots'] / max(1, c['rows']):.1f}", pct(c["rows_on_menu"] / max(1, c["rows"])), pct(c["expert_rows_on_menu"] / max(1, c["expert_rows"])), pct(c["expert_runs_all_on_menu"] / max(1, c["expert_runs"]))]
            cells += [pct(c.get(f"expert_runs_all_on_menu/{t}", 0) / c[f"expert_runs/{t}"] if c.get(f"expert_runs/{t}") else None) for t in tasks]
            lines.append("| " + k + " | " + " | ".join(cells) + " |")
        lines += ["", "Share of random or repeated commands inserted into won rollouts that were on the menu, rule `form`, so a live Sys1 could have picked them %:", "", "| Family | k=5 | k=10 | k=20 | k=all |", "| --- | --- | --- | --- | --- |"]
        for family in ("recovery", "loop-break"):
            cells = [s["inserted_commands_on_menu"]["form"][k] for k in ("5", "10", "20", "all")]
            lines.append("| " + family + " | " + " | ".join(pct(c.get(f"{family}/on_menu", 0) / c[f"{family}/commands"] if c.get(f"{family}/commands") else None) for c in cells) + " |")
        lines += ["", "Share of admissible commands by kind %: " + ", ".join(f"{k} {100 * v:.1f}" for k, v in s["admissible_kind_share"].items()) + ".", ""]
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pool", required=True)
    ap.add_argument("--splits-dir", default="lab/results/sys1-training-set")
    ap.add_argument("--out", default="lab/results/menu-statistics")
    ap.add_argument("--why", help="a text file with the choice of k, in plain words, for the report")
    ap.add_argument("--allow-partial-pool", action="store_true", help="for a look before generation ends. The result says so")
    a = ap.parse_args(argv)
    pool, out = Path(a.pool).expanduser(), Path(a.out)
    splits = load_splits(a.splits_dir)
    assert_no_leakage(splits)
    sets = as_sets(splits)
    done = {name: len(pool_files(pool, name)) for name in ("train", *MEASURED)}
    partial = any(done[name] != len(splits[name]) for name in done)
    if partial and not a.allow_partial_pool:
        raise SystemExit(f"the pool is not complete: {done}")
    records = list(fit_records(pool, sets))
    menus = {rule: fit(iter(records), rule=rule, meta={"fitted_on": "expert-path rows of won rollouts on train games", "pool_config": json.loads((pool / "config.json").read_text())}) for rule in ("form", "kind")}
    del records
    out.mkdir(parents=True, exist_ok=True)
    menus["form"].save(out / "menu.json")
    menus["kind"].save(out / "menu-kind.json")
    stats = {"topic": "menu-statistics", "seed": json.loads((pool / "config.json").read_text())["seed"], "partial_pool": partial, "games_in_pool": done, "pool_config": json.loads((pool / "config.json").read_text()), **provenance(),
             "menu_records_used": menus["form"].meta["records_used"], "menu_records_skipped_flagged": menus["form"].meta["records_skipped_flagged"], "splits": {}}
    for split in MEASURED:
        stats["splits"][split] = measure(pool, split, menus, sets)
    (out / "stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    if a.why:
        (out / "REPORT.md").write_text(report(stats, None, Path(a.why).read_text()))
    for split in MEASURED:
        s = stats["splits"][split]
        print(split, s["games"], "rows", s["rows_counted"], "adm", s["admissible_per_step"])
        for rule in s["rules"]:
            for k, r in s["rules"][rule].items():
                print(" ", rule, k, {f: round(v.get("coverage", 0), 3) for f, v in r["by_family"].items()}, "all", round(r["overall"]["coverage"], 3), "goto", round(r["overall"]["go_to_slot_share"], 3), "whole", round(r["expert_runs_with_every_label_on_menu"], 3))


if __name__ == "__main__":
    main()
