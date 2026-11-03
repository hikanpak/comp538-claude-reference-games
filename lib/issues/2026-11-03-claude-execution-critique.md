# Agent review of the execution path, and what was done with it

Opened 2026-11-03. The reviewer read the execution plan, interfaces, plan, Bonn's code, the Kev card
and the expert code, before Halil's Sys1-first instruction reached it. The local session re-checked the code
facts it could: Kev needs Python 3.12 or newer, `PointerHead.temperature` divides logits only in
eval mode, there is no value head, `kev.train --init_from` exists, the Kev-0.8B card's temperature
is 2.35, and `HandCodedAgent.act` returns `random.choice` when the policy repeats its last action.
Speed figures for L4 are the reviewer's inference (Heuristic), to be replaced by a measurement.

| # | Finding | Decision |
| --- | --- | --- |
| 1 | Single-episode harness makes text-generating Gemma rows unaffordable, about 7.5 s per Gemma step on L4 (Heuristic) | Adopted. `run_episodes` over batches of games in the interfaces |
| 2 | Kev has no RL hooks. PPO needs a value head, and A3's teacher-run steps break PPO's ratio | Adopted. Value head on the `<decide>` hidden state, PPO loss only on steps Sys1 sampled, temperature 1 in training and refit after |
| 3 | Nothing stops test expert data entering the menu or training | Adopted. `split` field, and asserts in `menu.fit` and the loader. `alf-data` stores no test states |
| 4 | Kev and ALFWorld dependencies never resolved together, and Kev's fast kernels compile at first call | Adopted. The `harness` session resolves one Python 3.12 environment. The no-compile rule narrows to source builds of CUDA extensions |
| 5 | Stage table needs 95 CU in series, 61.7 L4 hours, against 34 wall hours | Adopted in part. The 34 hours bound the cloud credits, not Colab, so Colab work may continue after. Stages were re-cut for Sys1 first, and each VM run stays under 8 hours |
| 6 | VM death loses `/content`, and GPU type is not guaranteed | Adopted in part. The wrapper asserts the L4. Not adopted: a Hugging Face token on the VM. The Mac pulls checkpoints with `colab download` every few minutes, so no new secret exists |
| 7 | RL step budget is Open, so live arms cannot be costed | Adopted. Set from the measured speed after the first Colab run, written in the execution plan, and the ratio to field practice is reported with scope |
| 8 | Classification training undercosted by 2 to 3 times (Heuristic) | Adopted. Measure rows per second on 200 rows first, dedupe, default one epoch |
| 9 | Hand-over numbers tuned on dev have no budget | Adopted. Start values 2, 10 and 3 stay untuned at the Gemma stage unless CU remain |
| 10 | Training memory is above the inference number | Moved to the Gemma stage, because Sys1 trains alone first |
| 11 | Expert labels at off-path states contain random draws | Adopted. Call `policy.act` directly, flag and exclude such labels, report the share |
| 12 | "Teacher off menu" fallback labels call Gemma where the menu fails, not where the step is hard. With all admissible commands fallback never has a positive label | Adopted. A confidence threshold rule moves to Tier 1, and the all-admissible row is named Sys1 alone |
| 13 | Best arm chosen on one seed and 140 dev games, and `valid_seen` shares rooms with train | Adopted. Choice labelled Heuristic. A dev split of held-out rooms is added from the training games, and the gates read it |
| 14 | Bonn's repo holds summaries only, so per-game checks need its Zenodo databases and its environment | Moved to the Gemma stage, with this method |
| 15 | Report must print the reasoning row beside Gemma + Sys1 and generated tokens, and say the bar is one-pass thinking-off. One extra live row should be Qwen 27B one-pass at 4-bit, about 5.5 CU (Heuristic) | Adopted for the report. The row runs only if CU remain, and its precision is stated |
| 16 | Drop stored picks and A2 first, then A4 on R4, then third seeds. The plan's cut order wrongly drops A3 before A2, while A3 has precedent on this task (BUTLER) | Adopted. Cut order to be corrected in the plan at the next edit |

Scope note the reviewer raised and the local session checked: the Kev-0.8B card reports When2Call accuracy below chance
(0.167 and 0.133) and advises against tool-call routing. That is a different task from ours, and it
trains no fallback option on it, so this is recorded with its scope and is no forecast.

Field practice as the reviewer reports it: GLAM and TWOSOME score all valid actions with no menu, use a value
head on the LLM, and run 1e5 to 1.5e6 steps on 3 to 4 seeds. BUTLER ran DAgger with this expert for 50K
episodes. Our menu, fallback label and far fewer steps are departures forced by budget, and the report
must say so.
