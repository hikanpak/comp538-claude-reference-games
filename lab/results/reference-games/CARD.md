# Reference games

These files record four reference models, elm, birch, cypress and ironwood, playing ALFWorld. A longer name is a larger model. Halil asked for the names to be anonymized on 2026-11-05, for research and confidentiality, so the real names are only in the private folder `lab/data/reference-games-private/`, which no clone holds. The files are a benchmark reference and a record for reproducing the runs. They never train a model.

The provider's usage policy, read 2026-11-04, lists using outputs to train an AI model without its prior authorization. Nothing here may become labels, features, rewards or fallback steps for Kev or any other model. Halil set this on 2026-11-04: these runs are for benchmarking and reproducibility, and not for training by any means. `lab/src/tests/test_reference_games.py` fails if training code names this folder.

Halil on 2026-11-04: "first create test data using cloud sessions with [the four models] all", then "run all the benchmark questions we have on these models for test sets and neatly gather the test results, note that in for each question it should be a fresh session". ALFWorld is the one benchmark the project holds, and its test set is 134 games.

## How a game runs

Each game ran as a fresh session: its own process of one command-line agent tool, started by a launcher, in a new empty folder with no settings and no saved session. So no process saw another game or a project file. One cloud session per model ran that model's processes, so a fresh session here is not a fresh cloud container. The process could run the player's `start` and `step` commands and no other shell command, so it could not read a game file.

The launcher read the model from the process's first event and would have stopped the list if it was not the model asked for. It names the models and their command-line tool, so it stays private with the processes' whole output.

## What a row can be compared with

A row is one model through one command-line agent tool in a cloud session, with that tool's system prompt and a shell tool, and with thinking and effort at the tool's default for that model, which is not the same across the four. So it is not one forward pass with thinking off, which is the setting of Bonn's published rows for Gemma and Qwen 27B. Compare it with those rows only with that difference stated.

- The game stack, the 50-step limit, the admissible commands without `help`, and the instruction sentence are Bonn's.
- The process sees the whole game so far in its own context, as Bonn's `history_window = None` gives its models.
- Bonn's models give a probability per command with no text. A reference model here sends one command through a tool call.
- `model_reported` and `processes[].model` hold the anonymized name. Before anonymizing, all 134 records of each model held that model's own id in both fields.
- A process that stops before GAME OVER is a lost game with `ended_early`. A process that dies is retried once from the first step, and the abandoned commands stay in `abandoned_attempts`. A game whose process dies twice is not counted as played, and the row must state how many there are. On the test list no process died and none was retried.

## Files

`summary/` holds the gathered tables. The win interval is a 95% Wilson interval, and `win_percent` is over the games played, with `not_played` beside it. One JSON per game, `<player>/valid_unseen/<uid with slashes as __>.json`, on the branch below. `test-games.txt` is the `test` list of 134 games. Each file holds the commands, the observations, the outcome, the refused commands, any abandoned attempt, the git commit, the package versions and the machine. `lab/jobs/reference-games.prompt.md` is the fixed prompt, and each file records its SHA-256.

Cost, tokens and raw transcripts are not in these files, because a price or a transcript identifies a model. They stay in the private folder. `summary/test-total-cost.csv` holds the total over all four models, and `summary/test-transcript-audit.csv` holds what was counted from the transcripts, so a clone cannot recompute that table.

## The test list

Halil opened the test games for these models on 2026-11-04: "make it play games and save the traces". Their choices are not Kev's choices, so this spends no test game for Kev. While games ran, the runner printed no win count for this list. Halil then asked for the results to be gathered. Kev's choices must still be frozen on dev without reading these files, because a per-game outcome says which test games are hard.

## Where the results are

Per-game records stay out of `main`, on branch `data-reference-games`, commit `1c6d984`, in `hkanpak21/comp538-sys1-harness` and `hikanpak/comp538-reference-games`. Halil on 2026-11-04: "Open a repo under hikanpak and push results there I dont care".

## Commit ids after anonymizing

The history of both repositories was rewritten on 2026-11-05 to remove the names, so every git commit named in a result file here is the rewritten form of the commit the run used, and the originals are private. The records name `263f2a3`, the rewritten form of the commit the games started from. Its job file was written for the rewrite: it holds the game commands and the table commands and leaves out the launcher, and the prompt is byte for byte the one the games used, so `prompt_sha256` still checks. The commits the runner made while it played only added results, apart from two log files on ironwood's branch, and they are not kept.

## Gathering

Each record's `seed` is empty, and its `seed_note` says why.

1. `git fetch origin data-reference-games`, then `git archive FETCH_HEAD lab/results/reference-games | tar -x -C <empty folder>`.
2. `REFERENCE_GAMES_RESULTS=<empty folder>/lab/results/reference-games python lab/jobs/reference-games.py gather --list test` writes the three tables under `summary/` in that folder.
3. `replay-check --list test`, with the same variable set, replays every saved game in the real game and compares each observation, the win and the step count. It needs the test game files (`data-test-game-files` branch) under `ALFWORLD_DATA`.

`plan_steps` and `mean_plan_steps` need the games' `traj_data.json`, and they stay empty without it. A row in these tables is through one agent tool with default thinking, so it is not one forward pass with thinking off. Kev's choices do not read these files.
