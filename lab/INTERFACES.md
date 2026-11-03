# Interfaces between the pieces of `lab/src/`

Sessions build these pieces on separate branches, so keep to these names and
shapes. Change one only by editing this file in the same commit and saying so in
your final commit message.

## Who owns what

| Path | Owner |
| --- | --- |
| `lab/src/menu.py`, `lab/results/alfworld-expert-data/`, `lab/results/menu-statistics/` | `alf-data` session |
| `lab/src/llm/`, `lab/results/bonn-responses/` | `bonn-responses` session |
| `lab/src/alf/`, `lab/src/harness/` | `harness` session |
| `lab/results/sys1-training-set/`, `lab/src/dataset/` | `sys1-dataset` session |
| `lab/src/sys1/`, `lab/src/train/`, `lab/jobs/arm-*`, `lab/jobs/colab_job.sh` | `sys1-train` session |
| `plan/`, `CONTEXT.md`, `CLAUDE.md`, `lib/execution-plan.md` | the local session |

`lab/requirements.txt` and `lib/experiments.md` are shared and append-only.

## Stored step record

One JSON object per line, gzip shards, as `alf-data` writes them:

`{"split", "game_id", "task_type", "goal", "step", "obs", "admissible": [str], "expert": str, "done", "won"}`

`split` is `train`, `dev`, or `dev_heldout_rooms`. Test games never appear in stored records.
`alf-data`'s first shards have no `split` because they cover training and dev files only, and
`sys1-dataset` adds it. `menu.fit` and every training loader assert `split == "train"`.

Added 2026-11-04 by the build: `obs` matches the harness's `GameView` form, which is `first_obs` without the
welcome text and task, and later observations stripped. `command` is optional and gives the command that actually
ran, for recovery and loop-breaking rows. `family`, `label_source`, `tracking`, `menu_rank`, `acceptable` and
`control_state` are written by the training set.

## Departures the build made, recorded by the agents

`HarnessConfig.match` added, because R4 pays only beta. `ControlState` and `HarnessConfig` are frozen with defaults.
`render_prompt` has `window=5`. `run_episodes` has keyword-only `batch_size`, `num_workers`, `tracking`,
`allow_test_games`, `keep_prompts`, `timing`. `teacher` takes None or `"expert"`. An LLM's history opens with
`("", first_obs)`. Labels carry `label_inadmissible`, `label_error`, `label_initial`. With no LLM and fallback open,
the step runs Sys1's top menu command and control stays still. Those last two are Claude's choices, for Halil to rewrite.

## Menu

`menu.fit(records) -> MenuModel`, `MenuModel.save(path)`, `MenuModel.load(path)`,
`MenuModel.menu(task_type, admissible, k) -> list[str]`. With `k=None` it returns all
admissible commands in score order. Deterministic.

## LLM

```python
class LLM:
    name: str
    precision: str
    def pick(self, goals, histories, admissibles) -> list[Pick]  # batched, one forward pass, Bonn's format, no text
    def reason_then_command(self, goals, histories, admissibles) -> list[Pick]  # batched, two calls, thinking off, greedy, fixed max_new_tokens
@dataclass
class Pick: command: str; probs: dict[str, float] | None; tokens_in: int; tokens_out: int; seconds: float
```

`history` is a list of `(command, observation)` pairs. Implementations: `StubLLM` for
tests, `StoredPicksLLM` for replay, and `HFLLM` on a GPU.

## Sys1

```python
class Sys1:
    def score(self, prompts: list[str], options: list[list[str]]) -> list[Tensor]
        # one logit tensor per prompt, one logit per option, with gradients when training
    temperature: float  # 1.0 during training, refit on dev after
    def probs(self, prompts, options) -> list[Tensor]   # softmax of logits / temperature
    def trainable_parameters(self)
    def save(self, path); @classmethod def load(cls, path, device)
```

Kev's own `kev.train`, its JSONL records and `--init_from` are the training path for classification,
and `lab/src/sys1/kev_adapter.py` wraps `kev.model.DecisionModel` for the online arms. RL adds a value
head on the `<decide>` hidden state. The PPO loss uses only steps where Sys1's sampled pick ran.

`options` always ends with the fallback option, whose text is `FALLBACK_TEXT` in
`lab/src/sys1/base.py`. `StubSys1` is a small CPU model for tests. `render_prompt(goal,
first_obs, trace, control_state)` in `lab/src/harness/prompt.py` builds the text Sys1 sees:
task text, first observation, last trace entries and the control state.

## Harness

```python
@dataclass
class HarnessConfig: k: int | None; fallbacks_before_handover: int; llm_rounds: int;
    menu_run: int; trace_window: int; w: float; c: float; beta: float; max_steps: int = 50
@dataclass
class ControlState: controller: str; fallbacks_in_row: int; llm_rounds_left: int; menu_run: int
def run_episodes(games, sys1, llm, menu, cfg, mode, teacher=None, teacher_prob=0.0, seed=0, mask_fallback=False) -> list[EpisodeLog]
```

Games run in a batch, because Sys1 and the LLM score lists. `llm=None` with `mask_fallback=True` is Sys1
alone. The teacher label at a reached state comes from `policy.act` directly. A label that the expert's
wrapper would have replaced by a random draw is flagged `label_random`, kept out of training and counted.

`mode` is one of `teacher`, `scheduled`, `model`, `test`. `EpisodeLog` holds per-step who
acted, the menu, Sys1's probabilities, the command, the teacher's command when known, the
reward, the control state, and the outcome, tokens and seconds.

## Metrics

`lab/src/harness/metrics.py` takes `EpisodeLog`s and returns success, mean steps, repeat
rate, Gemma calls, tokens, Sys1 share of steps, step match, ECE, multi-class Brier, NLL,
AUROC of confidence and exposure gap. `lab/src/harness/stats.py` holds the bootstrap that
resamples games, paired across rows, on each game's mean over seeds.

## Results

`lab/results/<topic>/<run>.json` with `config`, `seed`, `git_commit`, `environment`,
`hardware`, `precision`, `per_game` outcomes and aggregates.
