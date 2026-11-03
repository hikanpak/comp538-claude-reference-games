# When Sys1 counts as good enough to try on Gemma

Halil's instruction: "when you are sure that we have indeed a nice system 1 model that we
dream of then proceed and start trying it on gemma". This file turns that into written
gates, fixed before any training run. Claude drafted them while Halil was away. They are
not Halil's, and every threshold is a guess he should accept or rewrite. A threshold moves
only by a dated line in the changes section at the end, with the reason, and never after
looking at a result in order to pass it.

All gates are measured with no Gemma, on the dev games of held-out rooms (at least 140 games, carved from the training games), on the mean of 3 seeds, with the
worst seed reported beside it. The 140 `valid_seen` games are reported beside them and carry no gate, because they share rooms with training games. Test games are not used. Sys1 alone means fallback is masked,
so Sys1 must pick a menu command at every step, which the plan already defines for this
table.

| Gate | What is measured | Passes at | Published rows for scale |
| --- | --- | --- | --- |
| Solves games | Sys1 alone, success on dev | at least 54.5 | Bonn on test games: Jev 1.13 at 35.1, Qwen 27B at 54.5, expert at 90.3 |
| Does not loop | Sys1 alone, repeat rate | at most 0.10 | Bonn on test games: Jev 0.447, Qwen 27B 0.272, expert 0.018 |
| Holds up in its own games | Exposure gap, step match on the teacher's states minus step match on its own states | at most 0.10 | None read |
| Confidence is honest | ECE on its own games | at most 0.05 | Bonn on test games: Jev 0.247, Qwen 27B 0.388, Gemma 0.647 |
| Confidence ranks right above wrong | AUROC of confidence on its own games | at least 0.80 | Not read |
| Falls back when it should | With fallback unmasked, share of steps with the teacher's command off the menu where Sys1 falls back | at least 0.70 | None read |
| Does not fall back needlessly | With fallback unmasked, share of steps with the teacher's command on the menu where Sys1 falls back | at most 0.15 | None read |
| Seeds agree | Worst seed's dev success against the mean | within 10 points | None read |

Dev is 140 games, so a single success rate near 50% has an interval of about plus or minus 8
points (Heuristic, binomial). Passing by less than that is a pass on the written rule, and the
report must say the margin is inside the noise.

Step match on the teacher's states is reported and carries no gate. Bonn reports Qwen 27B at
54.5 success with 0.222 step match on its setup, so step match and success can separate. The
"solves games" gate is on the verifiable return, the game's own checker.

Reported beside the gates, with no threshold: the curve of step match against the share of steps Sys1 keeps as a confidence threshold on Sys1's top pick rises. This shows whether a threshold could hand hard on-menu steps to Gemma, which the off-menu fallback label cannot.

## Rule for moving on

All gates pass on 3 seeds: an agent review checks the results and the code path behind them,
and only then does Gemma + Sys1 start. Some gates fail with Sys1's Colab budget left: change
the training set or the arm and record why in `lib/work-log.md`. The Sys1 budget is 70 CU. If
60 CU are spent and gates still fail, stop, spend nothing on Gemma, and report to Halil what
passed, what failed, and the options. Reaching Gemma is not a goal in itself.

## Changes to this file

- 2026-11-03, before any result: gates moved from `valid_seen` to held-out rooms, from a review finding that `valid_seen` shares rooms with train.
- 2026-11-03, before any result: added a row below for confidence threshold, reported and not gated, from a review finding on fallback labels.
