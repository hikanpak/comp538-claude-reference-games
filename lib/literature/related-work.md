# Related work

Numbers below were read from the sources by Claude on 2026-11-03, on the setup each
row states, and none of them predicts our setting. A note under `notes/` is written for
each paper by the `lit-notes` session with page pointers. A row without a note is read
at the level of this table only.

## Axis: small and large LLMs on ALFWorld, one model, no controller

| Work | Setup | Results |
| --- | --- | --- |
| Bonn, arXiv 2609.37647 | 134 unseen games, 50 steps, one forward pass over admissible commands, thinking off, no text | Gemma 6.7 [3.0, 11.2], steps 47.8, repeat 0.541, step match 0.119. Qwen 27B 54.5 [46.3, 62.7], steps 31.6, repeat 0.272, step match 0.222. Jev 1.13 35.1 [27.6, 43.3], step match 0.179. Expert 90.3 [85.1, 95.5], steps 20.7, repeat 0.018 |
| ReDAct, arXiv 2604.07036 | 400 episodes, no split stated, Qwen3-Next-80B small model, reason then command, thinking setting not stated | Qwen3-80B alone 68.3. GPT-5.2 alone 78.3. With 3.6 handovers per episode 80.8. Random handovers 77.0 |

Only Bonn uses both Gemma and Qwen 27B on one task, and only Bonn uses either on
ALFWorld, among the work we read. Bonn's raw responses are at Zenodo
10.5281/zenodo.23039006 (one database per model, 311 MB each).

## Axis: training on ALFWorld, ReAct or commands, no controller

| Work | Setup | Results |
| --- | --- | --- |
| ETO, arXiv 2403.02502 | 134 unseen games, 3,321 training instances, ReAct with one example. Column order inferred and checked against its Table 7 | Llama-2-7B-Chat untuned 0.0, supervised 67.2, RFT 66.4, PPO on final reward 29.1, ETO 72.4. GPT-4 38.1, GPT-3.5 10.5 |
| AgentTuning, arXiv 2310.12823 | AgentBench's ALFWorld, 50 instances, 1 shot. Train 954 filtered to 336 successes | Llama-2 chat 7B 2.0, AgentLM-7B 84.0, 13B 76.0, 70B 86.0, GPT-4 78.0, GPT-3.5 14.0 |
| AgentGym, arXiv 2406.04151 | 6 task types, 200 evaluation episodes, 500 and 2,420 trajectories | Llama-2-7B 2.0, BC 77.5 and 83.0, AgentEvol 88.0, GPT-4-Turbo 67.5. ALFWorld column inferred from order |
| BUTLER, arXiv 2010.03768 | 3,553 train, 140 seen, 134 unseen. DAgger with handcoded expert, share falls 100% to 1% over 50K episodes, 8 seeds, text generation | Read at setup level only |

## Axis: RL over a list of options with an LLM body

| Work | Setup | Results |
| --- | --- | --- |
| TWOSOME, arXiv 2401.14151 | PPO with LoRA on frozen LLaMA-7B, valid actions scored by token likelihood with word normalization, 1e5 to 5e5 steps, 3 seeds, VirtualHome and Overcooked | Reports better sample efficiency than plain PPO and SayCan |
| GLAM, arXiv 2302.02662 | PPO on Flan-T5 780M over valid actions, 1.5M steps, 4 seeds, BabyAI-Text | Read at setup level only |
| decider, `github.com/Mapika/decider`, `docs/RL.md` | v10 is v8 continued for 384 optimizer steps on outcome rewards, PPO clip 0.2 on the served softmax, KL to v8 on 8 supervised rows per step, peak LR 1e-6, cosine over 576 steps, 96 transitions per iteration, temperature 1.30. Training code kept in a separate repo | Vendor report |
| Kev, decider-2b, Clef cards | Supervised cross entropy with a fitted temperature, as the cards describe. Kev-0.8B card: T 2.35, LoRA rank 16, reports When2Call accuracy below chance (0.167, 0.133) on tool-call routing, a different task | Vendor reports |
| RelayLLM, arXiv 2601.06051 | Qwen3-0.6B and 1.7B call Qwen3-8B. Supervised warm-up then GRPO | 42.50 to 49.52 with 1.07% of tokens, teacher alone 54.12 |

## Axis: a small model hands steps to a large one, or to a controller

| Work | Setup | Results |
| --- | --- | --- |
| When2Ask, arXiv 2306.03604 | PPO ask policy, 5 seeds, penalty when the plan does not change | Reports LLM interaction counts |
| Bellman Language Models, arXiv 2610.24871 | Controller solved from logs, tau-squared telecom, 114 tasks, preregistered, no code link | Qwen 27B alone then with controller: 52 to 94, 31 to 102, 33 to 109. Fine-tuned Qwen 100 of 114 (not registered). SymPy 14 to 19 of 63, not significant |

No source we read keeps the large model in control for several rounds. ReDAct and
When2Ask hand over one step, RelayLLM one token span, Bellman Language Models one move.
Halil's hand-over rules are distinctive for that reason, and whether they help is Open.

## Axis: prompt injection

| Work | Setup | Results |
| --- | --- | --- |
| AgentDojo, arXiv 2406.13352 | 97 user tasks, 629 security cases | `important_instructions` targeted attack success 57.7 |
| Task Shield, arXiv 2412.16682 | AgentDojo, GPT-4o, checks each call against the task | ASR 2.07, utility 69.79 |
| CaMeL, arXiv 2503.18813 | AgentDojo, control flow fixed from the trusted query | 77% with provable security, 84% undefended |

## Gemma and Qwen 27B elsewhere

Gemma: small coding agents arXiv 2610.33958 (SWE-bench Verified localization, Acc@10 16.7
with grep tools, 29.7 with all tools). Not Every Call Needs a Frontier Model arXiv
2610.09021 (7 local models, 4-bit, overall 78.6). arXiv 2604.07035 (mean VRAM 14.9 GB,
bf16 where supported). MobilePA-Bench arXiv 2608.23035 ran GRPO on Qwen 27B, 66.94 to
69.25, one exploratory checkpoint. Qwen 27B is also the base of Clef, a Kev variant and
JEV-27B.
