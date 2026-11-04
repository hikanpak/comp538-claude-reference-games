"""Write `CARD.md` from the manifest and the audit, so its numbers are the files' numbers.

    python -m dataset.card --dir lab/results/sys1-training-set --export-dir lab/data/sys1-kev
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .rollouts import FAMILIES


def pct(top, bottom) -> str:
    return f"{100 * top / bottom:.1f}" if bottom else "n/a"


def share(value) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}"


def table(header: list[str], rows: list[list]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |", *["| " + " | ".join(str(c) for c in row) + " |" for row in rows]]


def family_task_table(core: dict, tasks: list[str]) -> list[str]:
    rows = []
    total = core["rows"]
    for task in tasks:
        cells = [core.get(f"family_task/{f}/{task}", 0) for f in FAMILIES]
        rows.append([task, *cells, sum(cells), pct(sum(cells), total)])
    cells = [core.get(f"family/{f}", 0) for f in FAMILIES]
    rows.append(["All", *cells, sum(cells), "100.0"])
    rows.append(["Share %", *[pct(c, total) for c in cells], "", ""])
    return table(["Task type", *FAMILIES, "Rows", "Share %"], rows)


def outcome_table(by_task: dict, tasks: list[str]) -> list[str]:
    rows = []
    for task in tasks:
        row = [task]
        for family in FAMILIES:
            won, lost = by_task.get(f"{family}/{task}/won", 0), by_task.get(f"{family}/{task}/lost", 0)
            row.append(f"{lost} of {won + lost} ({pct(lost, won + lost)}%)")
        rows.append(row)
    return table(["Task type", *[f"{f} lost" for f in FAMILIES]], rows)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", default="lab/results/sys1-training-set")
    ap.add_argument("--export-dir", default="lab/data/sys1-kev")
    ap.add_argument("--menu-stats", default="lab/results/menu-statistics/stats.json")
    a = ap.parse_args(argv)
    directory = Path(a.dir)
    m = json.loads((directory / "manifest.json").read_text())
    audit = json.loads((directory / "audit.json").read_text())
    verify_path = directory / "verify-kev-batch.json"
    verify = json.loads(verify_path.read_text()) if verify_path.exists() else None
    menu_stats = json.loads(Path(a.menu_stats).read_text())
    g, tp, core = m["generation"], m["train_pool"], m["core"]
    config = g["config"]
    tasks = sorted(m["splits"]["train"]["by_task_type"])
    k = m["k"]
    disagreement = audit["expert_seed_disagreement"]["train"]
    train_tokens = audit["exports"].get("train", {}).get("tokens") or {}
    train_export = audit["exports"].get("train", {})
    flagged = defaultdict(dict)
    for key, value in tp["flagged"].items():
        family, source = key.split("/")
        flagged[family][source] = value
    candidates, after = tp["rows_candidate"], tp["rows_after_dedupe"]
    plan = core["sampling_plan"]
    no_winner = m["splits"]["train"]["games"] - disagreement["games_with_a_won_expert_run"]
    no_winner_by_task = ", ".join(f"{t} {m['splits']['train']['by_task_type'][t] - disagreement['by_task_type'].get(t, {}).get('games', 0)}" for t in tasks)
    won = lambda f: tp["runs"].get(f"{f}/won", 0)   # noqa: E731
    lost = lambda f: tp["runs"].get(f"{f}/lost", 0)   # noqa: E731

    lines = ["# Sys1 training set on ALFWorld", ""]
    lines += [f"`core` holds {core['rows']:,} deduplicated training rows from {m['splits']['train']['games']:,} `train` games, taught only by ALFWorld's handcoded expert. The rows belong to three families: the expert's clean games, states after random mistakes, and states inside and just after repeats. Two dev splits are stored in full. The audit reports {audit['failures_total']} failures. Nothing has been trained on it, so whether it gives the Sys1 of `lib/sys1-acceptance.md` is Open.", ""]
    lines += ["Claude built this set on Halil's brief. Family shares, the sampling rule, the control-state mix, the held-out rooms' floors and the starting k are Claude's choices and none is Halil's. The reference models' play supplies no label, because their provider's usage policy lists using outputs to train an AI model without prior authorization (`lib/work-log.md`, 2026-11-04).", ""]

    lines += ["## What is here", ""]
    lines += [f"`core-*.jsonl.gz` are the training rows, `dev-*.jsonl.gz` and `dev_heldout_rooms-*.jsonl.gz` the dev splits, `manifest.json` the hash of every shard with seeds, package versions and hardware, `audit.json` the audit, `verify-kev-batch.json` the check through Kev's loader and model, `splits.json` and the three `.txt` files the games. The shards total {m['bytes_total'] / 1e6:.1f} MB. The menu is `lab/results/menu-statistics/menu.json`, chosen as `k = {k}` in `lab/results/menu-statistics/REPORT.md`. Kev's JSONL, which is not committed, is in `lab/data/sys1-kev/`: `train.jsonl.gz` with `train.report.json`, whose `tokens.max_state_needed` is the job's `MAX_STATE`, the two dev exports, and `speedtest-200.jsonl.gz`. Rows are stored by game, not shuffled, because `kev.train` shuffles them (`kev/train.py`, `rng.shuffle(reqs)`).", ""]
    lines += ["Rows of `core` by family and task type:", "", *family_task_table(core, tasks), ""]

    lines += ["## Games and splits", ""]
    fc = m["splits"].get("filtered_counts") or {}
    lines += [f"`AlfredTWEnv.collect_game_files`, ALFWorld's own filter, gives {fc.get('train_folder')} games in `train` and {fc.get('valid_seen_folder')} in `valid_seen`, which match what papers report. `valid_unseen` was never opened, and every stored row and list is checked against it.", ""]
    lines += [f"`dev_heldout_rooms` holds {m['splits']['dev_heldout_rooms']['games']} games from rooms {', '.join(str(r) for r in m['splits']['held_out_rooms'])} of the training games, split seed {m['splits']['seed']}. Rooms were shuffled inside each of the four room types and taken one type at a time until there were at least 140 games and at least 10 of each of the 6 task types. `train` is the other {m['splits']['train']['games']:,} games in {m['splits']['train']['rooms']} rooms. `dev` is the {m['splits']['dev']['games']} games of `valid_seen`, in {m['splits']['dev']['rooms']} rooms, and those rooms are shared with `train`, which is why `lib/sys1-acceptance.md` puts its gates on the held-out rooms. No game is in two splits and no room is in both `train` and `dev_heldout_rooms`.", ""]
    rows = [[split, m["splits"][split]["games"], m["splits"][split]["rooms"], *[m["splits"][split]["by_task_type"].get(t, 0) for t in tasks]] for split in ("train", "dev", "dev_heldout_rooms")]
    lines += table(["Split", "Games", "Rooms", *tasks], rows) + [""]

    lines += ["## Labels", ""]
    lines += ["Every label comes from `policy.act` of the handcoded expert under `tracking=always`, which repairs the expert's three tracking faults at every step (`lib/issues/2026-11-04-claude-installed-expert-faults.md`). That choice is Claude's for Halil to rule on. A consequence is that this teacher is not the expert Bonn's 90.3 measures.", ""]
    lines += ["A label the shipped wrappers would have replaced by a random draw, a label the game does not admit, and a policy that raises are flagged. The row keeps its command in the traces of later rows and stays out of training. These counts cover rows of won rollouts on `train` games:", ""]
    rows = [[f, candidates.get(f, 0), *[flagged[f].get(s, 0) for s in ("random-draw", "inadmissible", "error")]] for f in FAMILIES]
    lines += table(["Family", "Usable rows", "Flagged random draw", "Flagged inadmissible", "Flagged error"], rows) + [""]
    lines += ["The first step of every expert game is the expert's fixed `look`, which `policy.act` did not choose. Those rows stay in training with `label_source` `initial-look`, because a Sys1-alone run starts from that text, and they give `look` a high rank on the menu.", ""]

    lines += ["## How each family was made", ""]
    lines += [f"The rollout seed is {config['seed']}. Rollouts ran on Halil's Mac in {g.get('workers')} worker processes, in {g['wall_seconds'] / 3600:.2f} hours of wall time and {g['sessions']} {'session' if g['sessions'] == 1 else 'sessions'}. The pool, {g['pool_bytes'] / 1e6:.1f} MB, is outside the repo at `~/.cache/comp538-sys1/pool-seed0`, and its digest is in the manifest.", ""]
    lines += [f"`expert-path`: the expert plays every game under {len(config['expert_seeds'])} seeds. Seed 0 is the stream `lab/src/alf/env.py` gives a game, so it repeats a harness run under `tracking=always`, and a test holds that. Every state of a won game is a row.", ""]
    lines += [f"`recovery`: {config['recovery']} rollouts per game. The expert plays a won seed's game up to a uniform random step, 1 to 5 uniform random admissible commands run, and the expert continues to the end. Each state after the first random command is labelled by the expert, and a row carries `command` where the command that ran differs from the label.", ""]
    lines += [f"`loop-break`: {config['loop_break']} rollouts per game. At a uniform random step, one command runs 2 to 4 times, then the expert continues. With equal chance the command is one with no effect (`look`, `inventory`, `examine`), or one already used in this game that is still admissible. An already-used command the game does not admit twice in a row alternates with its undo (`go to A` and back, `open` and `close`, `take` and `move`), so at most 7 commands are inserted. A row is a state after the first repeat ran, while the last two inserted commands are still in the 5 trace entries Sys1 sees. A rollout where the command ran fewer than twice gave no row: {tp['formed'].get('loop-break/no repeat formed', 0)} such rollouts.", ""]
    lines += ["Only rollouts the checker marks won give training rows. Rollouts lost, by task type, on `train` games:", "", *outcome_table(tp["runs_by_task"], tasks), ""]
    lines += [f"Overall the expert lost {lost('expert-path'):,} of {won('expert-path') + lost('expert-path'):,} of its own games on `train`, recovery rollouts lost {lost('recovery'):,} of {won('recovery') + lost('recovery'):,}, and loop-break rollouts lost {lost('loop-break'):,} of {won('loop-break') + lost('loop-break'):,}. {no_winner} of {m['splits']['train']['games']:,} games had no expert seed that won ({no_winner_by_task}), and those games got no perturbed rollout.", ""]

    lines += ["## Sampling rule", ""]
    asked = m["family_shares_asked"]
    lines += [f"Rows are deduplicated on the plain text Sys1 sees, the admissible commands and the label, keeping the first copy in the order expert-path, recovery, loop-break. Then each family takes a fixed share of {m['core_size_asked']:,} rows ({', '.join(f'{f} {100 * asked[f]:.0f}%' for f in FAMILIES)}). Inside a family each task type takes the share of `train` games it has, drawn uniformly by seed {m['seed']}, and a task type short of rows leaves its rest to the others of that family.", ""]
    rows = [[f, f"{candidates.get(f, 0):,}", f"{after.get(f, 0):,}", share(audit["duplicates"]["train_pool_duplicate_rate_by_family"].get(f)), f"{plan[f]['quota']:,}", f"{plan[f]['taken']:,}", plan[f]["unfilled"]] for f in FAMILIES]
    lines += table(["Family", "Usable rows in the pool", "After dedupe", "Duplicate rate %", "Quota", "Taken", "Unfilled"], rows) + [""]

    held = menu_stats["splits"]["dev_heldout_rooms"]
    form_k = held["rules"]["form"][str(k)]
    kinds = form_k["by_label_kind"]
    curve_k = held["form_curve"][str(k)]
    lines += [f"## Menu at k = {k}", ""]
    lines += [f"`lab/results/menu-statistics/REPORT.md` holds the coverage tables and the reasons for k. k = {k} is Claude's starting value and departs from the start of 10 that Halil named. Empirical, on `dev_heldout_rooms`: the expert's command is on the menu at {share(form_k['overall']['coverage'])}% of steps, and {share(form_k['expert_runs_with_every_label_on_menu'])}% of the expert's games keep every command on the menu. By command kind the share is {', '.join(f'`{c}` {share(kinds[c]["coverage"])}%' for c in ('go to', 'open', 'close', 'take', 'move', 'clean', 'heat', 'cool'))}. Whole expert games by task type: {', '.join(t + ' ' + pct(curve_k.get(f'expert_runs_all_on_menu/{t}', 0), curve_k.get(f'expert_runs/{t}', 0)) + '%' for t in tasks)}. A whole game is one a Sys1 confined to the menu could finish by following the expert. A step with the expert's command off the menu is a fallback row in `core`.", ""]
    lines += [f"Of the stored fields only the varied control states depend on k, because `menu_rank` holds the full ordering. Another k or another menu rule needs `dataset.menu_stats`, `dataset.build` and the export, and no new rollouts.", ""]

    lines += ["## Control state", ""]
    control = audit["control_state"]
    lines += [f"About 80% of training rows show the plain state a Sys1-alone run shows: Sys1 controls, no fallbacks, no LLM marks. The other rows vary so that Sys1 has seen some of the text a run with Gemma shows, because `harness/episode.py` leaves the control state still when there is no LLM (`lab/src/sys1/README.md`). In `core`, {control.get('varied', 0):,} rows of {control.get('varied', 0) + control.get('plain', 0):,} are varied ({share(audit['control_varied_share'])}%), taken by hash among the {core.get('control_varied_possible', 0):,} rows where a simulated path exists, {control.get('fallbacks_in_row=1', 0):,} show one fallback in a row, and {control.get('rows_showing_an_llm_mark', 0):,} show at least one `LLM>` entry in their 5 trace entries. Rows at step 0 cannot vary, since there is nothing to mark.", ""]
    lines += ["A varied row's control state and marks come from a simulation through the harness's own rules at the plan's start values (2 fallbacks, 10 LLM rounds, a run of 3): a command off that step's menu is the LLM's, a menu command is the LLM's with chance 0.3, and a step under the LLM's control gives no row, because Sys1 is not asked there. The commands stay the rollout's, so the expert stands in for the LLM. So no row shows an LLM's mistake, and only a random command inserted into a recovery rollout can carry an `LLM>` mark and be wrong. Dev rows are all plain, since the gates run Sys1 alone. `lab/src/tests/test_dataset.py` plays the real harness through the same fallbacks and asserts `render_prompt` on a stored row gives the text the live run gave, in plain and varied rows.", ""]

    lines += ["## Audit", ""]
    replay = audit["replay"]
    lines += [f"`audit.json` is written by `dataset/audit.py`. {'It passed' if audit['passed'] else 'It failed'}, with {audit['failures_total']} failures. It read every shard against its hash. All {audit['labels']['checked']:,} usable labels were checked against their admissible commands, and {audit['labels']['not_admissible']} were not admissible. {replay['rows']:,} stored sequences, up to {replay['per_stratum_asked']} per family and split, were replayed in the real game under the same expert seed, and every command's output, the observation, the admissible commands and the label were compared: {json.dumps(replay['matched'])}. {len(replay['labels_that_differed_and_returned_under_another_hash_seed'])} labels differed and returned under another of {replay['hash_seeds_tried']} hash seeds, and {len(replay['labels_that_never_returned'])} never returned, which would be a failure. `{audit['duplicates']['core_rows_identical_in_text_commands_and_label']}` rows of `core` are identical to another in rendered text, commands and label. Every exported state equals `render_prompt` on its stored row.", ""]
    lines += [f"Empirical, on the expert's won games on `train`: seeds played different commands in {disagreement['games_with_seeds_playing_different_commands']} of {disagreement['games_with_a_won_expert_run']} games ({share(disagreement['share_of_games'])}%), and {disagreement['of_which_seeds_gave_different_labels']:,} of {disagreement['states_reached_by_two_or_more_seeds']:,} states reached by two or more seeds got different labels ({share(disagreement['share_of_states'])}%). The expert's code draws at random only when it picks which object of a class to act on, and in the wrapper's draw, so seeds can differ only through those.", ""]
    rows = []
    coverage = audit["label_coverage_at_k"]["train"]["by_family_and_task_type"]
    for task in tasks:
        rows.append([task, *[share(coverage.get(f"{f}/{task}", {}).get("on_menu_share")) for f in FAMILIES]])
    rows.append(["All", *[share(coverage.get(f, {}).get("on_menu_share")) for f in FAMILIES]])
    lines += [f"Share % of `core` labels on the menu at k = {k}:", "", *table(["Task type", *FAMILIES], rows), ""]
    if train_tokens:
        lines += [f"Tokens of the state text as Kev counts them, on `train.jsonl.gz`: median {train_tokens['state_tokens_p50']}, 99th percentile {train_tokens['state_tokens_p99']}, maximum {train_tokens['state_tokens_max']}. Kev's default limit is 384, which {train_tokens['rows_over_default_max_state']:,} rows exceed, and a non-strict encode would cut the end of those states, where the last trace entries sit. So the job must run with `MAX_STATE = {train_tokens['max_state_needed']}`, which `train.report.json` holds. At that value {train_tokens['rows_over_row_limit_at_max_state_needed']} rows are over Kev's row limit. {share(train_export.get('fallback_label_share'))}% of exported rows have fallback as their label.", ""]
    if verify:
        lines += [f"`verify-kev-batch.json`: `kev.train.training_requests` read all {verify['rows_in_file']:,} exported rows and dropped {verify['rows_dropped_by_kev']} at `--max_state {verify['max_state']}`. One batch of {verify['batch']} went through Kev's own `microbatch_plan`, `encode_batch` and `batch_loss` into `DecisionModel` warm-started from Kev-0.8B on the Mac CPU, with a finite loss and one logit per option. No optimizer ran. Peak footprint was {verify['peak_footprint_gb']} GB.", ""]

    lines += ["## Known weaknesses, and what this set cannot teach", ""]
    spread = replay["under_other_hash_seeds"]
    moved = ", ".join(f"{v['labels_differ']} under hash seed {h}" for h, v in spread.items())
    weak = [
        "Fallback's positive label is only \"the teacher's command is off the menu\". It teaches nothing about steps where the command is on the menu and Sys1 should still hand over because it is unsure, and nothing about what Gemma would do with the step. The acceptance file reports a confidence-threshold curve for this reason.",
        "The expert decides from memory the trace window does not show: what it saw at places it visited earlier, and facts of the game. Two rows can show the same text and different labels, and the export's report counts them. A `go to` label among places not yet searched follows the expert's list: the places it expects that object class in, last listed first, minus the places already searched. So step match has a ceiling below 1 (Heuristic, from the expert's code in `alfworld/agents/expert/handcoded_expert.py`).",
        "Recovery commands are uniform random admissible commands, and Sys1's own mistakes need not be uniform (Heuristic), so these rows are not the states Sys1 itself reaches. A random command can be off the menu, which Sys1 could not pick in a live run, and the menu report gives that share. Loop-break repeats are Claude's sketch of a repeat, not repeats observed from Sys1.",
        "Only won rollouts give rows. States the expert does not recover from are absent, so Sys1 learns nothing about them, and every perturbation is short: at most 5 random commands or 7 inserted commands, once per game.",
        "The text of a run with Gemma is approximated. No row shows a command the LLM really wrote, or the game's \"Nothing happens\" to a command it rejected. The fallbacks in varied rows are drawn at random, independent of the row's label.",
        "The menu does not read the goal, so it cannot prefer the object the task names. Commands that move the game forward (`take`, `move`, `heat`, `cool`, `clean`) compete with `go to` by their counts over all games of a task type, so they rank low and many are off the menu, as the section on the menu shows. Those steps teach fallback and never teach the command itself.",
        "`look` is on the menu at every step where the fit ranks it in the top k, because the expert's fixed first step counts in the fit. `look` changes nothing in the game, so a Sys1 picking it spends a step.",
        "The held-out rooms are few, so their games are correlated, and a bootstrap over games understates the uncertainty about a new room (Heuristic). All look-at-object games of the held-out split are in room 316.",
        "Under `tracking=always` this teacher wins more than the shipped expert (95.71 against 83.57 on `valid_seen`, `lib/issues/2026-11-04-claude-installed-expert-faults.md`), and a label here can differ from what the shipped expert plays. A step-match number measured against the shipped expert measures agreement with another teacher.",
        f"The expert's label can depend on Python's string hash order, which changes per process unless it is pinned. Empirical: the audit replayed {replay['rows']:,} sampled sequences under hash seed {replay['hash_seed']} and again under other hash seeds, and the labels that differed from the stored ones number {len(replay['labels_that_differed_and_returned_under_another_hash_seed'])} under hash seed {replay['hash_seed']}, {moved}, with every output, observation and admissible list equal. An earlier audit with unpinned hash seeds found 2 of 1,350 labels different, and both stored labels returned under other hash seeds (that run's result was overwritten and the count is in `lib/work-log.md`). Both labels are the expert's own choice, and a stored rollout came from one process, so it is consistent inside itself. So a label is one draw of the expert.",
        "The rollouts ran before the last code changes, which are the audit's and the generator's. `manifest.json` names the commit, says `git_dirty_lab_src`, and holds the SHA-256 of each source file.",
    ]
    lines += [f"{i}. {text}" for i, text in enumerate(weak, 1)] + [""]

    lines += ["## Rebuild", "", "```bash", "export PYTHONPATH=lab/src HF_HUB_OFFLINE=1", "python -m dataset.splits --seed 0", f"nice -n 10 python -m dataset.generate --pool ~/.cache/comp538-sys1/pool-seed0 --seed {config['seed']} --expert-seeds {len(config['expert_seeds'])} --recovery {config['recovery']} --loop-break {config['loop_break']}",
              "python -m dataset.menu_stats --pool ~/.cache/comp538-sys1/pool-seed0 --why lab/results/menu-statistics/k-choice.txt", f"python -m dataset.build --pool ~/.cache/comp538-sys1/pool-seed0 --k {k}",
              f"python lab/src/train/export_kev.py --records 'lab/results/sys1-training-set/core-*.jsonl.gz' --menu lab/results/menu-statistics/menu.json --k {k} --dedupe hard --out lab/data/sys1-kev/train.jsonl.gz",
              f"python lab/src/train/export_kev.py --records 'lab/results/sys1-training-set/dev_heldout_rooms-*.jsonl.gz' --split dev_heldout_rooms --menu lab/results/menu-statistics/menu.json --k {k} --dedupe hard --out lab/data/sys1-kev/dev_heldout_rooms.jsonl.gz",
              f"python lab/src/train/export_kev.py --records 'lab/results/sys1-training-set/dev-*.jsonl.gz' --split dev --menu lab/results/menu-statistics/menu.json --k {k} --dedupe hard --out lab/data/sys1-kev/dev.jsonl.gz",
              f"python lab/src/train/export_kev.py --records 'lab/results/sys1-training-set/core-*.jsonl.gz' --menu lab/results/menu-statistics/menu.json --k {k} --dedupe hard --sample 200 --out lab/data/sys1-kev/speedtest-200.jsonl.gz",
              "python -m dataset.verify_batch", "python -m dataset.audit --pool ~/.cache/comp538-sys1/pool-seed0", "python -m dataset.card", "```", "",
              f"A rebuild is close to this pool and not byte-identical. The generator's workers for this pool {'ran under `PYTHONHASHSEED` ' + str(g['python_hash_seed']) if g.get('python_hash_seed') else 'had unpinned string hash seeds'}, and the expert's label can depend on that order, as the weaknesses say. The generator now pins it to `--seed`, and two pinned runs gave identical game files on 18 games (Empirical, Mac, scratch run with no result file).", ""]
    (directory / "CARD.md").write_text("\n".join(lines))
    print("wrote", directory / "CARD.md", len(lines), "lines")


if __name__ == "__main__":
    main()
