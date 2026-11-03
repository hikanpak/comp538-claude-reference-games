# Sys1: Kev-0.8B behind the interface of `lab/INTERFACES.md`

`base.py` holds `FALLBACK_TEXT`, `QUESTION_TEXT` and the `Sys1` interface. `stub.py` holds `StubSys1`, a small CPU model for tests. `kev_adapter.py` wraps `kev.model.DecisionModel`. `check_real_weights.py` is the check on real weights, and its result files are in `real-weights-check-mac/`. Training sits in `lab/src/train/` and `lab/jobs/arm-classification.*`.

Pointers below are relative to the installed package, `.venv/lib/python3.12/site-packages/kev/`, from git commit `2074cc44256200bb37540103fa4572901faa10f3`. "README" means Kev's README on GitHub at that commit. Everything in the next section was read in the source and not measured, unless a test is named.

## What Kev's source says

### Record format

`load_records` reads plain JSONL, one labelled request per line, and does not read gzip (`data.py:362-386`). A row is `{"state", "questions": {id: {"type", "instructions", "criteria": {option: description}, "label", "src"}}, "_meta"}`. A `label` is required on every question (`data.py:379`), `_meta` is merged into the record, and `src` takes a default. `materialize` turns a request into the internal record through the serving path (`data.py:397-410`, `api.py:102-117`), so training and serving share one encoding. A description of `None` makes the option's text its name alone (`api.py:58-59`). README lines 185-212 document the format. `lab/src/tests/test_sys1_interface.py` checks that a row from our export and a request from `KevSys1` encode to the same token ids.

### Context limits

Defaults are 384 state tokens, 1024 per row of state and branch, and 2048 packed (`model.py:14`). Serving allows 65536 (`model.py:18-20`). `training_context(max_state)` raises all three together (`model.py:31-38`), and `--max_state` sets it (`train.py:413`).

Two behaviours matter for our text. `kev.train` drops every row that does not fit and prints one line about it (`train.py:105-110`, `model.py:145-151`). A non-strict `encode` keeps the first `max_state` tokens (`model.py:98-104`), which cuts the end of the state, where the last trace entries and the control state sit. So `export_kev.py` counts tokens and writes `max_state_needed` to its report, the job requires `MAX_STATE`, and the job stops if Kev's dropped-rows line appears. `KevSys1` encodes strictly at serving limits, so it raises instead of cutting.

### Option limits

A question takes 1 to 255 options (`api.py:14`, `api.py:28-31`). The pointer head scores any number of options (`model.py:216-218`), so the count may change at every step. Empirical: the real-weights check below ran 4 then 8 options with matching logit shapes.

### Precision

`--dtype bf16` means autocast over fp32 weights and is CUDA only (`train.py:391`, `train.py:443-444`). `--weights_dtype bf16` makes the frozen backbone bf16 (`train.py:392`). Training turns TF32 on on CUDA (`train.py:532-533`). Loading defaults to fp32, while `kev.serve` defaults to bf16 on CUDA and MPS, with probabilities reported within about 0.01 (`checkpoint.py:89-92`). The model card (line 81) says Kev-0.8B was trained with bf16 autocast over fp32 weights and evaluated in fp32. The job does the same on an L4 and records its precision in the result file.

### Soft labels

Kev's trainer takes them, through a `target` dict keyed by option name beside the required `label`. The README does not document this. `question_loss` uses the cross entropy against the target (`train.py:47-49`), `materialize` normalises it (`data.py:404-409`), `augment` only permutes such a question (`data.py:330-331`), and the option-pair step skips it (`data.py:352`). `lab/src/tests/test_export_kev.py` passes a target through `load_records`, `augment`, `materialize` and `question_loss` and compares the loss to a hand computation. `export_kev.py --dedupe soft` writes this form for states where expert seeds gave different commands.

### Temperature

`PointerHead.temperature` divides logits in eval mode only (`model.py:211-218`), and loading a checkpoint sets it (`checkpoint.py:241`). `warm_start` does not carry it (`checkpoint.py:318-331`), and a new run's `Meta` starts at 1.0 (`train.py:550`). So a fine-tuned checkpoint has temperature 1.0 until a refit, whatever the released 2.351 was. The fit minimises mean NLL over a 121-point log grid from 0.25 to 4 (`metrics.py:272-294`). `scripts/calibrate_checkpoint.py` is not in the wheel and refuses rows it cannot place, so `lab/src/train/refit_temperature.py` calls `kev.metrics.fit_temperature` directly. A naive wrapper would divide twice, so `KevSys1` sets the head's attribute to 1.0 and holds the fitted value itself. A test with the real `PointerHead` pins this.

### Augmentation

The defaults are `--p_none 0.1`, `--p_none_distract 0.12` and `--p_distract 0.15` (`train.py:400-402`). `p_none` deletes the labelled option and relabels the row to an option meaning none of these (`data.py:335-336`), which for us would teach a different fallback text. Options are always shuffled (`data.py:341`), so menu order carries no signal in training. The job passes all three as 0, and also `--p_none_pair 0`. A test pins Kev's behaviour and another pins the flags.

### Saving during a run

`--save_every_*`, resume and snapshots are refused unless `--full_ft 1` (`train.py:472-477`), and a LoRA run saves only at its end (`train.py:663`). `lab/src/train/kev_train_runner.py` therefore adds a global optimizer hook that writes a partial checkpoint on a timer, so `colab_job.sh -p` has something to pull.

### Other traps

The default `--base` is `Qwen/Qwen3-0.6B-Base`, so the job passes it (`train.py:372`). `kev.train` refuses an existing `--out` (`train.py:481`). `kev.suite` builds Kev's own frozen benchmark, and `--suite` (`train.py:387`) trains on its partition, which this project does not use.

### Resource use as Kev documents it

README line 212 says `--batch 1 --accum 8` in bf16 fits a 4 GB GPU, and README line 338 gives a recipe with `--batch 8 --dtype bf16`. The model card (line 257) reports 3.8 GB of resident GPU memory when serving on an L4, which is no training number. The job's defaults take one epoch from the brief, `--lr 2e-5` from the README's `--init_from` recipe (lines 200-203), and `--batch 8` from line 338, which was Claude's choice and which the speed test can overrule through `BATCH`. Kernels without `flash-linear-attention` run through the reference PyTorch path, and the runtime warns "much slower". `modal_app.py:54-65` installs `flash-linear-attention==0.5.2` and `triton>=3.7.1` as separate steps. The job does the same, and no VM has run those steps yet. Kev requires Python at least 3.12 and under 3.14, and torch under 2.9.

## Measured on this Mac

Empirical. Scope for every number here: Halil's Mac, invented ALFWorld-shaped text, fp32, seed 0, 6 threads, git HEAD `665ba62` with uncommitted code, no `flash-linear-attention`. These are Mac numbers and not L4 numbers. No L4 number exists yet. Footprint is the macOS physical footprint. RSS also counts the mapped weight file.

| File in `real-weights-check-mac/` | Stage | Batch, tokens per row | Rows per second | Footprint GB | RSS high-water GB | Lowest system free |
| --- | --- | --- | --- | --- | --- | --- |
| `mac-cpu-infer.json` | Inference, CPU | 4, 378.5 | 0.679 | 3.78 | 4.4 | 35% |
| `mac-mps-infer.json` | Inference, MPS | 8, 385.9 | 1.781 | 5.56 | 4.72 | 29% |
| `mac-cpu-train-b1.json` | Training step, CPU | 1, 349 to 407 | 0.182 after the first step | 6.76 | 8.05 | 49% |
| `mac-mps-train-b1-t2.json` | Training step, MPS | 1, 244 to 302 | 0.467 after the first step | 7.82 | 4.76 | 31% |

Gradients. On CPU at batch 1 the first-step loss was 2.3005, and gradients were non-zero on 372 of 372 LoRA tensors and on 4 of 4 pointer head tensors, with norm 153.68 on the LoRA and 10.69 on the head. On MPS with 2 trace entries they were non-zero on the same counts, with norm 66.85 and 4.50. There are 11.35 M trainable parameters, and the warm start loaded 372 tensors. The option count ran 4 then 8 within one run.

Agreement. In `mac-cpu-infer.json` the fitted temperature read 2.351 with the head's attribute at 1.0, and the largest logit difference between a batch and single rows was 1.38e-5. In `mac-cpu-roundtrip.json` the largest logit difference between the live LoRA model and the same model saved, reloaded and merged was 1.84e-5, and a temperature of 1.7 survived the round trip.

Three runs went past or to the limit. CPU training at batch 4 reached a footprint of 8.22 GB and the watchdog killed it. MPS training at batch 1 with 5 trace entries (about 380 tokens) reached 8.00 GB and the watchdog killed it, so MPS training at that length has no measurement inside the limit. CPU training at batch 1 finished, and its RSS high-water of 8.05 GB is slightly over the 8 GB limit, though its footprint stayed at 6.76 GB.

`job-smoke-cpu.json` is one run of `lab/jobs/arm-classification.py` with `MODE=smoke` on CPU, on 6 invented training rows and 6 invented dev rows, batch 1. It measured 0.343 training rows per second and 0.503 inference rows per second, on short rows, with a resident high-water of 6.03 GB. It shows that the path runs: a partial checkpoint and a final one were written, both loaded through `KevSys1`, and the final one read its refit temperature. The temperature came out at 4.0, at the grid's edge, and the result flagged it. That value and the step match say nothing about Sys1 on six invented rows.

## What the export assumes

`lab/INTERFACES.md` gives the stored fields, not these rules. Claude chose them, and the docstring of `lab/src/train/export_kev.py` states them.

- `obs` at a step is what the command answers, and the observation after a command is the next record's `obs`. `lab/src/harness/episode.py` builds its trace the same way.
- A recovery or loop-breaking state needs an optional field `command` in the stored record, because the command that ran there did not come from the expert. `lab/INTERFACES.md` has no such field yet.
- Rows flagged `label_random` stay out of training, and their commands stay in the trace.
- Only trajectories the checker marks won are kept, following the training labels row of `plan/results_plan.tex`.
- The control state moves as if Sys1 had matched every earlier label. So training rows show `fallbacks_in_row` above 0, and a Sys1-alone run with no LLM does not, because `harness/episode.py` leaves the control state still when `llm` is `None`.
- Trace entries carry no actor, so no exported row shows the `LLM>` mark that `render_prompt` puts on an LLM's command.

`lab/src/menu.py` did not exist when this was written. `export_kev.py` imports it lazily and tests use a stand-in, so the export has not run on the real menu or on real stored records.
