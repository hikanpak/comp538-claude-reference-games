# Agent review rounds on the results plan

Opened 2026-11-03. An agent reviewed `plan/results_plan.tex` twice, and the local session checked each
finding against sources before applying it.

Round 1, applied: replay made fallback look like expert play, so LLM rounds now score
Gemma's stored pick. Added the GPU fit table and Kev-0.8B. Q5 counts Qwen 27B calls.
Statistics use a 10-point margin and a per-game gate. Q2 splits teacher's games and own
games. Q3 uses the training menu with fallback masked. AgentDojo's main attack is
`important_instructions`. Thinking pinned off with a ceiling row. Fine-tuned Gemma row
added from ETO. Task Shield and CaMeL added. Colab hours caveat added.

Round 2, applied: menu rule stated at command level. Fine-tuned row specified. GPU row for
live arms and an estimate for stored picks. Replay stated as a proxy. Gate rule rewritten.
ETO rows added to the RL table. Bootstrap wording fixed. Hand-over row moved to Tier 1.
Decisions and draft fail conditions moved to `plan/choices.tex`.

Not adopted, with the reason: "H6 is a lower bound" is unproven. "Expect R4 to need ..." is
a forecast from one benchmark, which Halil asked not to do. Dropping the Page column was a
matter of taste.
