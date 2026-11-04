# Sys1 training set on ALFWorld

`core` holds 100,000 deduplicated training rows from 3,385 `train` games, taught only by ALFWorld's handcoded expert. The rows belong to three families: the expert's clean games, states after random mistakes, and states inside and just after repeats. Two dev splits are stored in full. The audit reports 0 failures. Nothing has been trained on it, so whether it gives the Sys1 of `lib/sys1-acceptance.md` is Open.

Claude built this set on Halil's brief. Family shares, the sampling rule, the control-state mix, the held-out rooms' floors and the starting k are Claude's choices and none is Halil's. The reference models' play supplies no label, because their provider's usage policy lists using outputs to train an AI model without prior authorization (`lib/work-log.md`, 2026-11-04).

## What is here

`core-*.jsonl.gz` are the training rows, `dev-*.jsonl.gz` and `dev_heldout_rooms-*.jsonl.gz` the dev splits, `manifest.json` the hash of every shard with seeds, package versions and hardware, `audit.json` the audit, `verify-kev-batch.json` the check through Kev's loader and model, `splits.json` and the three `.txt` files the games. The shards total 9.3 MB. The menu is `lab/results/menu-statistics/menu.json`, chosen as `k = 20` in `lab/results/menu-statistics/REPORT.md`. Kev's JSONL, which is not committed, is in `lab/data/sys1-kev/`: `train.jsonl.gz` with `train.report.json`, whose `tokens.max_state_needed` is the job's `MAX_STATE`, the two dev exports, and `speedtest-200.jsonl.gz`. Rows are stored by game, not shuffled, because `kev.train` shuffles them (`kev/train.py`, `rng.shuffle(reqs)`).

Rows of `core` by family and task type:

| Task type | expert-path | recovery | loop-break | Rows | Share % |
| --- | --- | --- | --- | --- | --- |
| look_at_obj_in_light | 4581 | 2597 | 1731 | 8909 | 8.9 |
| pick_and_place_simple | 8876 | 6691 | 4461 | 20028 | 20.0 |
| pick_clean_then_place_in_recep | 9882 | 5601 | 3734 | 19217 | 19.2 |
| pick_cool_then_place_in_recep | 7756 | 4396 | 2931 | 15083 | 15.1 |
| pick_heat_then_place_in_recep | 6755 | 3829 | 2552 | 13136 | 13.1 |
| pick_two_obj_and_place | 12150 | 6886 | 4591 | 23627 | 23.6 |
| All | 50000 | 30000 | 20000 | 100000 | 100.0 |
| Share % | 50.0 | 30.0 | 20.0 |  |  |

## Games and splits

`AlfredTWEnv.collect_game_files`, ALFWorld's own filter, gives 3553 games in `train` and 140 in `valid_seen`, which match what papers report. `valid_unseen` was never opened, and every stored row and list is checked against it.

`dev_heldout_rooms` holds 168 games from rooms 7, 207, 316, 409 of the training games, split seed 0. Rooms were shuffled inside each of the four room types and taken one type at a time until there were at least 140 games and at least 10 of each of the 6 task types. `train` is the other 3,385 games in 104 rooms. `dev` is the 140 games of `valid_seen`, in 76 rooms, and those rooms are shared with `train`, which is why `lib/sys1-acceptance.md` puts its gates on the held-out rooms. No game is in two splits and no room is in both `train` and `dev_heldout_rooms`.

| Split | Games | Rooms | look_at_obj_in_light | pick_and_place_simple | pick_clean_then_place_in_recep | pick_cool_then_place_in_recep | pick_heat_then_place_in_recep | pick_two_obj_and_place |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 3385 | 104 | 293 | 755 | 632 | 496 | 432 | 777 |
| dev | 140 | 76 | 13 | 35 | 27 | 25 | 16 | 24 |
| dev_heldout_rooms | 168 | 4 | 15 | 35 | 18 | 37 | 27 | 36 |

## Labels

Every label comes from `policy.act` of the handcoded expert under `tracking=always`, which repairs the expert's three tracking faults at every step (`lib/issues/2026-11-04-claude-installed-expert-faults.md`). That choice is Claude's for Halil to rule on. A consequence is that this teacher is not the expert Bonn's 90.3 measures.

A label the shipped wrappers would have replaced by a random draw, a label the game does not admit, and a policy that raises are flagged. The row keeps its command in the traces of later rows and stays out of training. These counts cover rows of won rollouts on `train` games:

| Family | Usable rows | Flagged random draw | Flagged inadmissible | Flagged error |
| --- | --- | --- | --- | --- |
| expert-path | 148422 | 3159 | 3034 | 0 |
| recovery | 159752 | 1877 | 2046 | 2 |
| loop-break | 57696 | 559 | 1012 | 0 |

The first step of every expert game is the expert's fixed `look`, which `policy.act` did not choose. Those rows stay in training with `label_source` `initial-look`, because a Sys1-alone run starts from that text, and they give `look` a high rank on the menu.

## How each family was made

The rollout seed is 0. Rollouts ran on Halil's Mac in 4 worker processes, in 2.91 hours of wall time and 1 session. The pool, 17.4 MB, is outside the repo at `~/.cache/comp538-sys1/pool-seed0`, and its digest is in the manifest.

`expert-path`: the expert plays every game under 3 seeds. Seed 0 is the stream `lab/src/alf/env.py` gives a game, so it repeats a harness run under `tracking=always`, and a test holds that. Every state of a won game is a row.

`recovery`: 5 rollouts per game. The expert plays a won seed's game up to a uniform random step, 1 to 5 uniform random admissible commands run, and the expert continues to the end. Each state after the first random command is labelled by the expert, and a row carries `command` where the command that ran differs from the label.

`loop-break`: 3 rollouts per game. At a uniform random step, one command runs 2 to 4 times, then the expert continues. With equal chance the command is one with no effect (`look`, `inventory`, `examine`), or one already used in this game that is still admissible. An already-used command the game does not admit twice in a row alternates with its undo (`go to A` and back, `open` and `close`, `take` and `move`), so at most 7 commands are inserted. A row is a state after the first repeat ran, while the last two inserted commands are still in the 5 trace entries Sys1 sees. A rollout where the command ran fewer than twice gave no row: 2 such rollouts.

Only rollouts the checker marks won give training rows. Rollouts lost, by task type, on `train` games:

| Task type | expert-path lost | recovery lost | loop-break lost |
| --- | --- | --- | --- |
| look_at_obj_in_light | 17 of 879 (1.9%) | 27 of 1455 (1.9%) | 25 of 873 (2.9%) |
| pick_and_place_simple | 17 of 2265 (0.8%) | 28 of 3760 (0.7%) | 18 of 2256 (0.8%) |
| pick_clean_then_place_in_recep | 71 of 1896 (3.7%) | 66 of 3055 (2.2%) | 39 of 1833 (2.1%) |
| pick_cool_then_place_in_recep | 103 of 1488 (6.9%) | 60 of 2340 (2.6%) | 43 of 1404 (3.1%) |
| pick_heat_then_place_in_recep | 52 of 1296 (4.0%) | 50 of 2095 (2.4%) | 37 of 1257 (2.9%) |
| pick_two_obj_and_place | 237 of 2331 (10.2%) | 132 of 3555 (3.7%) | 83 of 2133 (3.9%) |

Overall the expert lost 497 of 10,155 of its own games on `train`, recovery rollouts lost 363 of 16,260, and loop-break rollouts lost 245 of 9,756. 133 of 3,385 games had no expert seed that won (look_at_obj_in_light 2, pick_and_place_simple 3, pick_clean_then_place_in_recep 21, pick_cool_then_place_in_recep 28, pick_heat_then_place_in_recep 13, pick_two_obj_and_place 66), and those games got no perturbed rollout.

## Sampling rule

Rows are deduplicated on the plain text Sys1 sees, the admissible commands and the label, keeping the first copy in the order expert-path, recovery, loop-break. Then each family takes a fixed share of 100,000 rows (expert-path 50%, recovery 30%, loop-break 20%). Inside a family each task type takes the share of `train` games it has, drawn uniformly by seed 0, and a task type short of rows leaves its rest to the others of that family.

| Family | Usable rows in the pool | After dedupe | Duplicate rate % | Quota | Taken | Unfilled |
| --- | --- | --- | --- | --- | --- | --- |
| expert-path | 148,422 | 58,388 | 60.7 | 50,000 | 50,000 | 0 |
| recovery | 159,752 | 113,774 | 28.8 | 30,000 | 30,000 | 0 |
| loop-break | 57,696 | 52,563 | 8.9 | 20,000 | 20,000 | 0 |

## Menu at k = 20

`lab/results/menu-statistics/REPORT.md` holds the coverage tables and the reasons for k. k = 20 is Claude's starting value and departs from the start of 10 that Halil named. Empirical, on `dev_heldout_rooms`: the expert's command is on the menu at 83.7% of steps, and 51.6% of the expert's games keep every command on the menu. By command kind the share is `go to` 91.3%, `open` 92.9%, `close` 86.3%, `take` 59.5%, `move` 57.2%, `clean` 45.0%, `heat` 19.5%, `cool` 0.0%. Whole expert games by task type: look_at_obj_in_light 100.0%, pick_and_place_simple 87.6%, pick_clean_then_place_in_recep 41.2%, pick_cool_then_place_in_recep 0.0%, pick_heat_then_place_in_recep 0.0%, pick_two_obj_and_place 91.7%. A whole game is one a Sys1 confined to the menu could finish by following the expert. A step with the expert's command off the menu is a fallback row in `core`.

Of the stored fields only the varied control states depend on k, because `menu_rank` holds the full ordering. Another k or another menu rule needs `dataset.menu_stats`, `dataset.build` and the export, and no new rollouts.

## Control state

About 80% of training rows show the plain state a Sys1-alone run shows: Sys1 controls, no fallbacks, no LLM marks. The other rows vary so that Sys1 has seen some of the text a run with Gemma shows, because `harness/episode.py` leaves the control state still when there is no LLM (`lab/src/sys1/README.md`). In `core`, 20,000 rows of 100,000 are varied (20.0%), taken by hash among the 85,547 rows where a simulated path exists, 7,166 show one fallback in a row, and 20,000 show at least one `LLM>` entry in their 5 trace entries. Rows at step 0 cannot vary, since there is nothing to mark.

A varied row's control state and marks come from a simulation through the harness's own rules at the plan's start values (2 fallbacks, 10 LLM rounds, a run of 3): a command off that step's menu is the LLM's, a menu command is the LLM's with chance 0.3, and a step under the LLM's control gives no row, because Sys1 is not asked there. The commands stay the rollout's, so the expert stands in for the LLM. So no row shows an LLM's mistake, and only a random command inserted into a recovery rollout can carry an `LLM>` mark and be wrong. Dev rows are all plain, since the gates run Sys1 alone. `lab/src/tests/test_dataset.py` plays the real harness through the same fallbacks and asserts `render_prompt` on a stored row gives the text the live run gave, in plain and varied rows.

## Audit

`audit.json` is written by `dataset/audit.py`. It passed, with 0 failures. It read every shard against its hash. All 139,176 usable labels were checked against their admissible commands, and 0 were not admissible. 1,350 stored sequences, up to 150 per family and split, were replayed in the real game under the same expert seed, and every command's output, the observation, the admissible commands and the label were compared: {"admissible_ok": 1350, "first_text_ok": 1350, "label_ok": 1350, "label_source_ok": 1350, "not_ended_early_ok": 1350, "observation_ok": 1350, "outputs_ok": 1350}. 0 labels differed and returned under another of 16 hash seeds, and 0 never returned, which would be a failure. `0` rows of `core` are identical to another in rendered text, commands and label. Every exported state equals `render_prompt` on its stored row.

Empirical, on the expert's won games on `train`: seeds played different commands in 955 of 3252 games (29.4%), and 563 of 42,872 states reached by two or more seeds got different labels (1.3%). The expert's code draws at random only when it picks which object of a class to act on, and in the wrapper's draw, so seeds can differ only through those.

Share % of `core` labels on the menu at k = 20:

| Task type | expert-path | recovery | loop-break |
| --- | --- | --- | --- |
| look_at_obj_in_light | 89.5 | 90.8 | 92.2 |
| pick_and_place_simple | 90.6 | 90.0 | 90.5 |
| pick_clean_then_place_in_recep | 77.5 | 76.5 | 80.3 |
| pick_cool_then_place_in_recep | 70.9 | 66.4 | 70.4 |
| pick_heat_then_place_in_recep | 73.4 | 69.6 | 75.2 |
| pick_two_obj_and_place | 88.3 | 89.4 | 91.6 |
| All | 82.0 | 81.4 | 84.1 |

Tokens of the state text as Kev counts them, on `train.jsonl.gz`: median 337, 99th percentile 557, maximum 844. Kev's default limit is 384, which 30,863 rows exceed, and a non-strict encode would cut the end of those states, where the last trace entries sit. So the job must run with `MAX_STATE = 844`, which `train.report.json` holds. At that value 0 rows are over Kev's row limit. 17.8% of exported rows have fallback as their label.

`verify-kev-batch.json`: `kev.train.training_requests` read all 99,820 exported rows and dropped 0 at `--max_state 844`. One batch of 4 went through Kev's own `microbatch_plan`, `encode_batch` and `batch_loss` into `DecisionModel` warm-started from Kev-0.8B on the Mac CPU, with a finite loss and one logit per option. No optimizer ran. Peak footprint was 3.97 GB.

## Known weaknesses, and what this set cannot teach

1. Fallback's positive label is only "the teacher's command is off the menu". It teaches nothing about steps where the command is on the menu and Sys1 should still hand over because it is unsure, and nothing about what Gemma would do with the step. The acceptance file reports a confidence-threshold curve for this reason.
2. The expert decides from memory the trace window does not show: what it saw at places it visited earlier, and facts of the game. Two rows can show the same text and different labels, and the export's report counts them. A `go to` label among places not yet searched follows the expert's list: the places it expects that object class in, last listed first, minus the places already searched. So step match has a ceiling below 1 (Heuristic, from the expert's code in `alfworld/agents/expert/handcoded_expert.py`).
3. Recovery commands are uniform random admissible commands, and Sys1's own mistakes need not be uniform (Heuristic), so these rows are not the states Sys1 itself reaches. A random command can be off the menu, which Sys1 could not pick in a live run, and the menu report gives that share. Loop-break repeats are Claude's sketch of a repeat, not repeats observed from Sys1.
4. Only won rollouts give rows. States the expert does not recover from are absent, so Sys1 learns nothing about them, and every perturbation is short: at most 5 random commands or 7 inserted commands, once per game.
5. The text of a run with Gemma is approximated. No row shows a command the LLM really wrote, or the game's "Nothing happens" to a command it rejected. The fallbacks in varied rows are drawn at random, independent of the row's label.
6. The menu does not read the goal, so it cannot prefer the object the task names. Commands that move the game forward (`take`, `move`, `heat`, `cool`, `clean`) compete with `go to` by their counts over all games of a task type, so they rank low and many are off the menu, as the section on the menu shows. Those steps teach fallback and never teach the command itself.
7. `look` is on the menu at every step where the fit ranks it in the top k, because the expert's fixed first step counts in the fit. `look` changes nothing in the game, so a Sys1 picking it spends a step.
8. The held-out rooms are few, so their games are correlated, and a bootstrap over games understates the uncertainty about a new room (Heuristic). All look-at-object games of the held-out split are in room 316.
9. Under `tracking=always` this teacher wins more than the shipped expert (95.71 against 83.57 on `valid_seen`, `lib/issues/2026-11-04-claude-installed-expert-faults.md`), and a label here can differ from what the shipped expert plays. A step-match number measured against the shipped expert measures agreement with another teacher.
10. The expert's label can depend on Python's string hash order, which changes per process unless it is pinned. Empirical: the audit replayed 1,350 sampled sequences under hash seed 0 and again under other hash seeds, and the labels that differed from the stored ones number 0 under hash seed 0, 0 under hash seed 1, 1 under hash seed 2, with every output, observation and admissible list equal. An earlier audit with unpinned hash seeds found 2 of 1,350 labels different, and both stored labels returned under other hash seeds (that run's result was overwritten and the count is in `lib/work-log.md`). Both labels are the expert's own choice, and a stored rollout came from one process, so it is consistent inside itself. So a label is one draw of the expert.
11. The rollouts ran before the last code changes, which are the audit's and the generator's. `manifest.json` names the commit, says `git_dirty_lab_src`, and holds the SHA-256 of each source file.

## Rebuild

```bash
export PYTHONPATH=lab/src HF_HUB_OFFLINE=1
python -m dataset.splits --seed 0
nice -n 10 python -m dataset.generate --pool ~/.cache/comp538-sys1/pool-seed0 --seed 0 --expert-seeds 3 --recovery 5 --loop-break 3
python -m dataset.menu_stats --pool ~/.cache/comp538-sys1/pool-seed0 --why lab/results/menu-statistics/k-choice.txt
python -m dataset.build --pool ~/.cache/comp538-sys1/pool-seed0 --k 20
python lab/src/train/export_kev.py --records 'lab/results/sys1-training-set/core-*.jsonl.gz' --menu lab/results/menu-statistics/menu.json --k 20 --dedupe hard --out lab/data/sys1-kev/train.jsonl.gz
python lab/src/train/export_kev.py --records 'lab/results/sys1-training-set/dev_heldout_rooms-*.jsonl.gz' --split dev_heldout_rooms --menu lab/results/menu-statistics/menu.json --k 20 --dedupe hard --out lab/data/sys1-kev/dev_heldout_rooms.jsonl.gz
python lab/src/train/export_kev.py --records 'lab/results/sys1-training-set/dev-*.jsonl.gz' --split dev --menu lab/results/menu-statistics/menu.json --k 20 --dedupe hard --out lab/data/sys1-kev/dev.jsonl.gz
python lab/src/train/export_kev.py --records 'lab/results/sys1-training-set/core-*.jsonl.gz' --menu lab/results/menu-statistics/menu.json --k 20 --dedupe hard --sample 200 --out lab/data/sys1-kev/speedtest-200.jsonl.gz
python -m dataset.verify_batch
python -m dataset.audit --pool ~/.cache/comp538-sys1/pool-seed0
python -m dataset.card
```

A rebuild is close to this pool and not byte-identical. The generator's workers for this pool had unpinned string hash seeds, and the expert's label can depend on that order, as the weaknesses say. The generator now pins it to `--seed`, and two pinned runs gave identical game files on 18 games (Empirical, Mac, scratch run with no result file).
