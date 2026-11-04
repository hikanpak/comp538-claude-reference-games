# sys1-harness

COMP438/538 Reinforcement Learning course project, Koç University, Fall 2026.
Instructor Barış Akgün. Started from the proposal "System One Control of LLM
Agent Harnesses with Verifiable Rewards". The destination is a course report, so
the `neurips` value in `paper/.paper-meta.yml` is only what the scaffold requires.

## What this is

A System One model (Sys1) picks one option from a short list and gives a
probability for each option. It writes no text. We put a trained Sys1 in front of
a small LLM in an agent harness. At each step Sys1 picks a command from a menu or
falls back to the LLM, and the hand-over rules decide who keeps control.

## The target, in Halil's words

"The ideal I have in my mind is to make Gemma-4-E4B to match about qwen 3.8 27B
model in terms of success."

On Bonn's 134 unseen ALFWorld games, one forward pass with thinking off, the
published success is Gemma 6.7 [3.0, 11.2] and Qwen 27B 54.5 [46.3, 62.7]. No row
for Gemma + Sys1 exists anywhere we read. The plan is `plan/results_plan.tex`,
and every choice with who made it is `plan/choices.tex`.

## People

| Handle | Name | Role |
| --- | --- | --- |
| halil | Halil Kanpak | Sole author. Chose the topic, the target and the design |
| akgun | Barış Akgün | Instructor |
| claude | Claude | Tooling, disclosed in `lib/llm-use-log.md` |

## Terminology ledger

| Term | Means | Never called |
| --- | --- | --- |
| Sys1 | The System One model we train. Halil also says Jev or System One | decision model, overseer, router |
| Jev | TypeSafe's closed Sys1, used only as a published reference row | our model |
| Gemma | `google/gemma-4-E4B-it`, the small LLM we try to lift | worker |
| Qwen 27B | `Qwen/Qwen3.8-27B`, whose success is the bar | large model, teacher |
| LLM | Whichever of Gemma or Qwen 27B takes the steps Sys1 hands over | fallback model |
| Gemma + Sys1 | The harness with Sys1 in control, handing steps to Gemma | hybrid |
| harness | The loop that decides who acts, runs the command and records the output | framework |
| menu | The top k admissible commands by teacher use in similar situations | action list, shortlist |
| admissible commands | The commands the game allows at this step | legal commands, actions |
| fallback | The menu option meaning "LLM, you take this step" | defer, abstain |
| control state | Who controls, fallbacks in a row, LLM rounds left | |
| teacher | Whoever supplies the right command in training. ALFWorld's handcoded expert here | oracle, checker |
| checker | The game's own success test | verifier, teacher |
| Reference games | Birch, Cypress, Elm or Ironwood playing ALFWorld games in a cloud session for benchmarking. Never training data | reference-model labels, reference-model teacher |
| arm | One way of training Sys1 | variant |
| step match | How often Sys1's pick equals the teacher's | accuracy |
| handover | One step given to Qwen 27B, one call. Only in the Qwen question of the plan | fallback |

## Plan labels and code names

The plan uses short labels because Halil reads it. Jobs, results and issues use
slugs, as `CLAUDE.md` requires.

| Plan label | Slug in `lab/jobs/` and `lab/results/` |
| --- | --- |
| G | `gate-published-numbers` |
| Q1 | `menu-statistics`, `llm-alone-dev` |
| A0 | `arm-untrained` |
| A1 | `arm-classification` |
| A2 | `arm-rl-teacher-forced` |
| A3 | `arm-rl-scheduled` |
| A4 | `arm-rl-model-in-control` |
| Q4 rows | `test-<row-name>`, row names listed in `lib/execution-plan.md` |
| Q7 rows | `ablation-<what-changed>` |

## Symbols

| Symbol | Means | Defined in |
| --- | --- | --- |
| k | Menu length, a hyperparameter starting at 10 | `plan/choices.tex` |
| w | Reward lost on a wrong pick | `plan/results_plan.tex` |
| c | Reward lost on a needless fallback | `plan/results_plan.tex` |
| beta | Reward at a successful end | `plan/results_plan.tex` |
| R1 to R4 | Named settings of w, c and beta | `plan/results_plan.tex` |

## Compute and money

Colab has about 100 compute units on Halil's account: Gemma at most 30, Sys1 the
other 70. L4 cost 1.54 CU per hour on his account on 2026-10-28, so about 19 and
45 hours. Sys1 trains on Colab. Cloud session credits are $250 and expire on
2026-11-05 at 10:59 GMT+3. On the Mac, use at most 80% of memory and keep storage
small. Cloud sessions cannot reach Colab or the clusters, because the credentials
stay on the Mac. MN5 and AWS are not in use for this project.

## Current state

Nothing has been measured yet. The live state is in `lib/execution-plan.md`.

## Recorded deferrals

Halil's project-setup interview was skipped because he was away, and his
instruction was to proceed. `paper/` stays empty until he writes the report's
claims, so no paper prose is drafted by an agent. Live Qwen 27B runs are not
funded under the Colab split above, so Qwen rows come from Bonn's published
numbers and stored responses unless compute units remain.
