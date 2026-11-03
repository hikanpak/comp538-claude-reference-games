"""`arm-rl-scheduled` (plan label A3): a teacher whose share of the commands falls from 1 to 0, in two forms kept apart so that each can be measured alone.

    SEED=0 STEP_BUDGET=20000 OBJECTIVE=ce ... python -m train.dagger

`OBJECTIVE=ce`, the default and the supervised form of the idea. Games run in batches through `run_episodes` in `scheduled` mode. At each step the teacher's command runs with
probability `teacher_prob`, otherwise Sys1's sampled pick runs. The teacher's label is taken at every state reached, under `tracking=always`, and new labelled states collect in a buffer.
Sys1 trains on the buffer with cross entropy on the same options the state showed, mixed with rows from the core training set. There is no value head and no reward in this loss.

`OBJECTIVE=ppo`, the plan's form. The same play, then PPO from `train/ppo.py` on the steps Sys1's sampled pick ran, on R2 and on R3 once Sys1 acts. The plan does not say when Sys1 acts, and Claude
reads it as the teacher's probability at or below `R3_BELOW_TEACHER_PROB`, 0.5 by default. While the teacher's probability is 1 no step is sampled, so such an iteration trains nothing and is counted.

The schedule is Claude's choice, and the plan says only that it falls from 1 to 0. `teacher_prob` falls linearly in commands run: 1 at the start, 0 at `SCHEDULE_FRACTION` of the step budget, 0.8 by default, and 0 after,
so the last fifth of the budget has Sys1 alone at the wheel. One play holds one value, read at its start. BUTLER's schedule on this task, from 100% to 1%, is the precedent the plan names.

The labels. A label flagged `label_random`, `label_inadmissible` or `label_error` stays out of the buffer and is counted, and those commands still run and stay in the trace. The expert's fixed first `look`, `label_initial`,
stays in, because `export_kev` keeps it too. The target is the teacher's command when it is on the menu and fallback otherwise, which is R2's rule and the export's. A state with an identical prompt, options and label enters once.

Other choices by Claude: 2 rows trained per new state, half of each batch from the core set, batch 8, learning rate 2e-5 from the classification job with the warm-up and cosine of `train/ppo.py` over the planned updates, and dropout on as in
`kev.train`. Rows keep the option order the harness shows, while `kev.train` shuffles options, because `Sys1.score` requires fallback last. `train/ppo.py` states the fallback rule, the cadence, the checkpoints and the modes, which are the same here.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

LAB_SRC = Path(__file__).resolve().parents[1]
if str(LAB_SRC) not in sys.path:
    sys.path.append(str(LAB_SRC))

from sys1.base import FALLBACK_TEXT  # noqa: E402
from train.ppo import Config, PPOTrainer, build_transitions, chunks, harness_config, release_cache, run_job  # noqa: E402

TOPIC = "arm-rl-scheduled"
FLAGS = ("label_random", "label_inadmissible", "label_error")


def teacher_prob(env_steps: int, schedule_steps: float) -> float:
    """The probability that the teacher's command runs at this position, linear from 1 at command 0 to 0 at `schedule_steps`, and 0 after."""
    if schedule_steps <= 0:
        return 0.0
    return max(0.0, 1.0 - env_steps / schedule_steps)


def reward_at(probability: float, r3_below: float) -> str:
    """The reward of the PPO form: R2, and R3 once Sys1 acts."""
    return "R3" if probability <= r3_below else "R2"


def flag_of(s) -> str | None:
    """Why a step's label must stay out of training, or None."""
    if s.label_random:
        return "label_random"
    if s.label_inadmissible:
        return "label_inadmissible"
    if s.label_error is not None:
        return "label_error"
    if s.teacher_command is not None and s.teacher_command not in s.admissible:   # `help`, which the policy may answer and the harness's list does not hold. The dataset flags it the same way.
        return "label_inadmissible"
    return None


class LabelBuffer:
    """The states the games reached, with the teacher's label on the options they showed."""

    def __init__(self):
        self.rows: list[dict] = []
        self.keys: set[str] = set()
        self.counts: Counter = Counter()

    def add_step(self, s, uid: str, teacher_probability: float) -> bool:
        """-> whether the step added a row. A step with no question or no label adds none, and a flagged one is counted under its flag."""
        self.counts["steps_seen"] += 1
        flag = flag_of(s)
        if flag:
            self.counts[f"dropped_{flag}"] += 1
            return False
        if s.options is None or s.teacher_command is None:
            self.counts["dropped_no_question_or_no_label"] += 1
            return False
        if FALLBACK_TEXT in s.menu:
            raise ValueError(f"{uid} step {s.step}: a menu holds the fallback text")
        label = s.teacher_command if s.teacher_command in s.menu else FALLBACK_TEXT
        key = hashlib.sha256(json.dumps([s.prompt, s.options, label]).encode()).hexdigest()
        if key in self.keys:
            self.counts["duplicates"] += 1
            return False
        self.keys.add(key)
        self.rows.append({"state": s.prompt, "options": list(s.options), "label": label, "target": None, "game": uid, "step": s.step, "source": "buffer", "actor": s.actor, "teacher_prob": teacher_probability})
        self.counts["rows"] += 1
        self.counts["rows_label_fallback"] += label == FALLBACK_TEXT
        self.counts["rows_label_initial"] += s.label_initial
        return True

    def add_logs(self, logs, teacher_probability: float) -> int:
        return sum(self.add_step(s, log.uid, teacher_probability) for log in logs for s in log.steps)

    def state(self) -> dict:
        return {"rows": self.rows, "keys": sorted(self.keys), "counts": dict(self.counts)}

    def load(self, saved: dict) -> None:
        self.rows, self.keys, self.counts = saved["rows"], set(saved["keys"]), Counter(saved["counts"])


def row_loss(logits: torch.Tensor, row: dict) -> torch.Tensor:
    """Cross entropy on one row's options, against its soft target when the core set gives one, as `kev.train.question_loss` does."""
    logp = F.log_softmax(logits, -1)
    if row.get("target"):
        return -sum(share * logp[row["options"].index(option)] for option, share in row["target"].items())
    return -logp[row["options"].index(row["label"])]


class ScheduledTrainer(PPOTrainer):
    """Both forms. The PPO form uses what it inherits, and the cross entropy form never builds a value head."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer = LabelBuffer()
        self.schedule_steps = self.cfg.schedule_fraction * self.cfg.step_budget

    def extra_state(self) -> dict:
        return {"buffer": self.buffer.state()}

    def load_extra_state(self, extra: dict) -> None:
        self.buffer.load(extra["buffer"])

    def prepare(self) -> None:
        if self.cfg.objective == "ppo":
            super().prepare()

    def probability_now(self) -> float:
        fixed = self.cfg.teacher_prob_fixed
        if fixed < 0 and self.cfg.mode == "speedtest":
            fixed = 0.5   # at a probability of 1 nothing is sampled, so a speed test would measure no PPO step
        return fixed if fixed >= 0 else teacher_prob(self.pos["env_steps"], self.schedule_steps)

    def sample_rows(self, n: int) -> list[dict]:
        """`n` rows with replacement, `core_share` of them from the core set while both sources hold rows."""
        core, new = self.core, self.buffer.rows
        if not core and not new:
            return []
        n_core = n if not new else 0 if not core else round(n * self.cfg.core_share)
        return [self.rng.choice(core) for _ in range(n_core)] + [self.rng.choice(new) for _ in range(n - n_core)]

    def ce_step(self, rows: list[dict]) -> dict:
        self.set_lr()
        self.optimizer.zero_grad(set_to_none=True)
        losses, matches = [], 0
        for part in chunks(rows, self.cfg.micro_batch):
            logits = self.sys1.score([r["state"] for r in part], [r["options"] for r in part])
            values = [row_loss(z, r) for z, r in zip(logits, part)]
            (torch.stack(values).sum() / len(rows)).backward()
            losses += [float(v.detach()) for v in values]
            matches += sum(r["options"][int(z.argmax())] == r["label"] for z, r in zip(logits, part))
        norm, _ = self.clip_and_step()
        return {"loss": float(np.mean(losses)), "matches": matches, "grad_norm": norm}

    def ce_update(self, n_new: int) -> dict:
        cfg = self.cfg
        n_updates = math.ceil(cfg.train_rows_per_new_row * n_new / cfg.batch)
        if not n_updates:
            self.counts["iterations_without_new_rows"] += 1
            return {"ce_updates": 0}
        started = time.perf_counter()
        self.sys1.train()   # dropout stays on, as in kev.train
        stats = [self.ce_step(self.sample_rows(cfg.batch)) for _ in range(n_updates)]
        self.sys1.eval()
        self.seconds["update"] += time.perf_counter() - started
        self.counts["ce_rows_trained"] += n_updates * cfg.batch
        return {"ce_updates": n_updates, "ce_loss": float(np.mean([s["loss"] for s in stats])), "ce_batch_step_match": sum(s["matches"] for s in stats) / (n_updates * cfg.batch), "grad_norm": float(np.mean([s["grad_norm"] for s in stats]))}

    def iteration(self) -> dict:
        cfg = self.cfg
        probability = self.probability_now()
        reward = reward_at(probability, cfg.r3_below_teacher_prob) if cfg.objective == "ppo" else "R2"
        harness = harness_config(cfg, reward)
        started = self.pos["env_steps"]
        if self.collector is not None:
            self.collector.values.clear()
        logs = self.play(self.next_games(cfg.games_per_collect), harness, "scheduled", teacher_prob=probability)
        summary = self.iteration_summary(logs, started, reward, probability)
        if cfg.objective == "ppo":
            return summary | self.ppo_update(build_transitions(logs, harness, cfg.gamma, cfg.lam, self.collector.values))
        new_rows = self.buffer.add_logs(logs, probability)
        return summary | {"new_rows": new_rows, "buffer_rows": len(self.buffer.rows)} | self.ce_update(new_rows)

    def base_result(self) -> dict:
        return super().base_result() | {"schedule": {"form": "linear in commands run, from 1 to 0", "schedule_steps": self.schedule_steps, "fixed": self.cfg.teacher_prob_fixed if self.cfg.teacher_prob_fixed >= 0 else None},
                                        "buffer": {"rows": len(self.buffer.rows), "counts": dict(self.buffer.counts)}, "core_rows": len(self.core)}


def main() -> None:
    env = dict(os.environ)
    run_job(Config.from_env(env, topic=TOPIC, objective="ce", reward="R2"), ScheduledTrainer)


if __name__ == "__main__":
    main()
