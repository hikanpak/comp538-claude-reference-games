# Execution plan

Written 2026-11-03 by Claude while Halil was away, on his instruction: "First criteria
here is not to finish it but instead finish the necessary experimentation we need." All
costs are Heuristic until the first run replaces them. Actuals go in the ledger below.

## What is necessary, in order

The chain to Halil's target is: the setup reproduces published numbers, the menu covers the
teacher, Sys1 trains, then Gemma + Sys1 plays the test games. Everything else depends on it.

| Stage | Plan label | What | Where | Cost |
| --- | --- | --- | --- | --- |
| 1 | G, Q1 | Environment probe, expert dataset, expert gate, menu statistics | Cloud, `alf-data` | $15 to $25 |
| 2 | G | Parse Bonn's stored responses, reproduce Gemma and Qwen 27B per game, port Bonn's pick format to `lab/src/llm/` | Cloud, `bonn-responses` | $15 to $25 |
| 3 | | Harness, reward, control state, metrics, bootstrap, tests on real ALFWorld with stubs | Cloud, `harness` | $20 to $35 |
| 4 | | Sys1 adapter on real Kev weights, arm training code, tests, Colab job scripts | Cloud, `sys1-train` | $25 to $40 |
| 5 | | Paper notes with page pointers | Cloud, `lit-notes` | $10 to $20 |
| 6 | | First end-to-end run on L4: Gemma bf16 beside Kev-0.8B, memory, speed, cost per step | Colab | 2 CU Gemma cap |
| 7 | G | Gemma picks on the 134 games, Bonn's way, compare with Bonn | Colab | 2 CU Gemma cap |
| 8 | Q1 | Gemma's stored picks at expert states, a subset if the measured rate needs it | Colab | at most 10 CU Gemma cap |
| 9 | Q2, Q3 | A1 classification, A2, A3, A4 on R3 and R4, 1 seed each on dev | Colab | about 25 CU Sys1 |
| 10 | Q2, Q3 | 3 seeds of A1 and the best arm | Colab | about 15 CU Sys1 |
| 11 | Q4 | Frozen choices. Test rows: Gemma alone two ways, Sys1 alone, Gemma + untrained Sys1, Gemma + best of A1 and A2, Gemma + best arm, fine-tuned Gemma if CU remain | Colab | at most 16 CU Gemma cap |
| 12 | Q7 | Tier 1 ablations in the order hand-over, control state, menu length, reward | Colab | what Sys1's cap has left, about 25 CU |
| 13 | | Report for Halil with all result tables | Local | |

Cloud total for stages 1 to 5 is about $85 to $145. The rest of the $250 is held for fixes,
for reviews of results, and for the second tasks if the core finishes.

## Not funded unless budget remains

Live Qwen 27B rows, since Halil capped Gemma at 30 CU and gave the rest to Sys1. This means
the target bar is Bonn's published 54.5 and not our own thinking-off row, which is option
(b) of decision D6 and not Claude's earlier suggestion (a). Also not funded: the handover
question Q5, which needs Qwen 27B live, MiniGrid, AgentDojo, and Tier 2 ablations. They run
in that reverse order of need only after stage 12: MiniGrid first, since it is cheap, then
AgentDojo. Cut order from the plan otherwise holds.

## Model choice forced by the budget

Kev-0.8B is the primary Sys1 instead of Kev-4B. Only Kev-0.8B fits an L4 beside Gemma at
Bonn's precision (17.7 GB of weights, against 25.3 GB). This is option (a) of decision D5,
and it replaces the dagger on Kev-4B in the plan. Kev-4B moves to the ablation list if CU
remain.

## Rules while spending

The Colab VM is released on every exit path, through the wrapper modelled on Halil's
`colab_job.sh`. Before each VM starts, its expected cost goes in the ledger, and its real
cost after. Training scripts checkpoint under `/content` and the Mac downloads the latest
only, because Drive mounting cannot run unattended. Nothing compiles on L4, which hung for
72 minutes in Halil's earlier work. Timing numbers from Colab are not claims.

No tool here reads the cloud credit balance, and the Mac cannot see what a cloud session has
spent. Only Halil's usage card shows it. So each cloud session gets a token-sparing brief and
a narrow scope, and the costs above stay estimates.

## Compute ledger

| When | Job | GPU | Hours | CU | Charged to |
| --- | --- | --- | --- | --- | --- |
