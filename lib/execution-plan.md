# Execution plan

Written 2026-11-03 by Claude while Halil was away, on his instruction: "First criteria
here is not to finish it but instead finish the necessary experimentation we need." All
costs are Heuristic until the first run replaces them. Actuals go in the ledger below.

## What is necessary, in order

Halil's order: "first let us have a model specifically trained for our defined task". So the
chain is: training set, Sys1 alone against `lib/sys1-acceptance.md`, and only after those gates pass,
Gemma. Costs were corrected after the agent review in `lib/issues/2026-11-03-claude-execution-critique.md`.

| Stage | Plan label | What | Where | Cost |
| --- | --- | --- | --- | --- |
| 1 | G, Q1 | Environment probe, expert dataset, expert gate, menu statistics | Cloud, `alf-data` | $15 to $25 |
| 2 | | One Python 3.12 environment for ALFWorld and Kev. Batched harness, control state, reward, metrics, bootstrap, Sys1-alone evaluation job | Cloud, `harness` | $20 to $35 |
| 3 | | Sys1 adapter on real Kev-0.8B weights, classification export and job, Colab wrapper | Cloud, `sys1-train` | $15 to $25 |
| 4 | | The training set: several expert seeds per state, recovery states, loop-breaking states, flagged random labels removed, held-out rooms, card and audit | Cloud, `sys1-dataset` | $25 to $40 |
| 5 | | Independent audit of the training set by a second session that did not build it | Cloud, `sys1-dataset-audit` | $10 to $20 |
| 6 | A1 | First Colab run: rows per second and memory on 200 rows, then classification, 1 seed, then Sys1 alone on dev | Colab, Mac for online if light | about 8 CU Sys1 |
| 7 | A3, A4 | RL code in `lab/src/train/`, then the DAgger-style arm and the PPO arm, 1 seed each. Step budgets set from stage 6's speed | Cloud for code, Colab to train | about 30 CU Sys1 |
| 8 | | Change the training set or arm where gates fail, each change logged with its reason | Cloud and Colab | about 15 CU Sys1 |
| 9 | Q2, Q3 | 3 seeds of the best arm and classification. Measure the gates | Colab | about 15 CU Sys1 |
| 10 | | An agent reviews Sys1's results and code path. Halil's gate decides whether Gemma starts | Local | |

Cloud total for stages 1 to 5 is about $85 to $145. The rest is held for stage 7's code, for fixes and for
reviews, and any left over is not a reason to spend. Stage 6 is the first to spend Colab units.

After stage 10, and only if gates pass, the Gemma stage: Gemma on L4 within its 30 CU, at Bonn's precision,
batched. Gate on Gemma alone with Bonn's method (review finding 14), Gemma alone two ways, Gemma + untrained
Sys1, Gemma + best Sys1, on the test games with choices frozen. Hand-over numbers stay at 2, 10 and 3. The
report prints the reasoning row beside Gemma + Sys1, gives generated tokens per game, and says the bar is
one-pass and thinking-off (review finding 15).

## Not funded unless budget remains

Live Qwen 27B rows. Halil capped Gemma at 30 CU and gave the rest to Sys1, so the target bar is Bonn's
published 54.5, which is option (b) of decision D6. If CU remain, one live row: Qwen 27B one-pass at 4-bit,
with its precision stated. Also: stored Gemma picks and A2, the handover question Q5, MiniGrid, AgentDojo,
Tier 2 ablations. Tier 1 ablations come after the Gemma stage in the order hand-over, control state, menu
length, reward.

## Teacher

The handcoded expert is the only teacher, which is option (b) of decision D1. The training set records its
label source so a stronger teacher can be added. The reference models' own play is not used as labels, because their provider's
terms restrict using its outputs to train other models, and reading those terms is Halil's call, not Claude's.

## Model choice forced by the budget

Kev-0.8B is the primary Sys1 instead of Kev-4B, because only it fits an L4 beside Gemma at Bonn's precision,
which is option (a) of decision D5. Kev-4B moves to the ablation list if CU remain.

## Rules while spending

The Colab VM is released on every exit path, through the wrapper modelled on Halil's
`colab_job.sh`. Before each VM starts, its expected cost goes in the ledger, and its real
cost after. Training scripts checkpoint under `/content` and the Mac pulls the latest with `colab download` every few
minutes, because Drive mounting cannot run unattended and a token on the VM would be a new secret. No source builds of CUDA extensions on a VM, and no `torch.compile` on L4, which hung for 72 minutes
in Halil's earlier work. Kev's fast kernels may compile at first call, and stage 6 times that. The wrapper
asserts the VM really has an L4, and every VM run stays under 8 hours. Timing numbers from Colab are not claims.

No tool here reads the cloud credit balance, and the Mac cannot see what a cloud session has
spent. Only Halil's usage card shows it. So each cloud session gets a token-sparing brief and
a narrow scope, and the costs above stay estimates.

## Compute ledger

| When | Job | GPU | Hours | CU | Charged to |
| --- | --- | --- | --- | --- | --- |
