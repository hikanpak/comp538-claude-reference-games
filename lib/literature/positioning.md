# Positioning

Course project, so the claim stays Halil's. This file holds the facts around it.

## 1. Setting

An agent harness on ALFWorld. Sys1 picks commands from a menu, and hands steps to Gemma
when it falls back. Hand-over rules keep the LLM in control for several rounds. The
teacher in training is ALFWorld's handcoded expert, and the game's own checker gives the
verifiable return.

## 2. What exists

See `related-work.md`. Closest are Bonn (arXiv 2609.37647) for the numbers we line up
with, ETO (arXiv 2403.02502) for fine-tuning and PPO on the same 134 games, and ReDAct
(arXiv 2604.07036) for handing steps to a large model on uncertainty.

## 3. The gap

Found by us from what we read, not stated by the authors: no source reports a trained
Sys1 in front of Gemma on ALFWorld, and no source keeps the large model in control for
several rounds. A paper we did not read could still exist (Open).

## 4. The claim

Halil's words: "make Gemma-4-E4B to match about qwen 3.8 27B model in terms of success".
He writes the report's claim after the results exist.

## 5. Why it is hard

Bonn reports Qwen 27B at 54.5 success with 0.222 step match, so a model can solve games
while disagreeing with the expert at most steps, on that setup. Step match alone therefore
does not settle success. Under teacher forcing the game's outcome does not depend on Sys1,
so the checker's return carries signal only when Sys1's own picks run.
