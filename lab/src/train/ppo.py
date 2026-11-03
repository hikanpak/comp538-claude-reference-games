"""PPO for Sys1 on ALFWorld with no LLM, and what both RL arms share. `arm-rl-model-in-control` (plan label A4) runs this file, and `train/dagger.py` builds
`arm-rl-scheduled` (plan label A3) on it.

    SEED=0 STEP_BUDGET=4608 REWARD=R3 ... python -m train.ppo

Every setting arrives as an environment variable named after a field of `Config`, because a Colab kernel has no `sys.argv` (`lab/jobs/arm-rl-model-in-control.py`).

The arm. Games run in batches through `run_episodes` in `model` mode, so Sys1's sampled pick runs at every step. RL starts from the weights the classification arm
wrote (`INIT_FROM`), at temperature 1, and the temperature is refit on dev rows at the end with the function the classification job uses.

The fallback rule at this stage is in `FALLBACK_RULE`, and it is Claude's choice and not Halil's. `FALLBACK=masked` plays with fallback removed from Sys1's list, which
is what `lab/jobs/eval-sys1-alone.py` calls Sys1 alone, and with `REWARD=R4` it needs no teacher at all. Two things follow from the stand-in, and the report must say them:
a win reached through the teacher's stand-in steps still pays beta, which makes fallback look better than it will with Gemma (Heuristic), and R4 with the stand-in is not
purely verifiable, because a fallback step pays R2's rule. `games_won_with_stand_in` counts the first and `teacher_stand_in_steps` the second.

The harness does not offer this rule. With `llm=None` its fallback step runs Sys1's top menu command, and an LLM is not told the teacher's label, so `teacher_stand_in`
wraps the harness's private `_ask_llm` for the length of a play. `lab/src/tests/test_rl.py` fails if the wrap stops working.

What the PPO ratio reads. Only steps where `StepLog.sampled` is true, which are steps where Sys1's drawn pick decided what ran (`ratio_mask`). A fallback sampled by Sys1
counts, as the harness says for an LLM, because Sys1's choice decided the step. Steps Sys1 did not decide pay it nothing except a terminal beta, and they sit inside the
discount between its decisions, so returns and advantages run over Sys1's decisions as a semi-Markov chain, which is ordinary GAE when every step is a decision. The
50-step cap counts as the end of a game with no bootstrap, because a game at the cap pays nothing more.

Defaults. Read from decider's `docs/RL.md` at github.com/Mapika/decider, as data about what worked on decider's tasks and not as a claim about ALFWorld: clip 0.2, peak
learning rate 1e-6 with 16 warm-up steps and a cosine over the run, 96 transitions per iteration in minibatches of 8 with one pass, AdamW betas 0.9 and 0.95, weight decay 0.01,
gradient norm 1.0, actor coefficient 0.1, KL(supervised || current) with coefficient 0.7 on 8 replayed supervised rows per step, and its guard that drops the actor term and
zeroes the first moment above 0.01 nats on the mean or 0.05 on a row. `LR_TOTAL_UPDATES` comes from the step budget and equals decider's 576 at `STEP_BUDGET=4608`. Decider serves at
temperature 1.30 and this arm trains at 1, as the brief says. Its document gives no value head, gamma, lambda or entropy term, so these are Claude's choices: the value head of
`sys1/value_head.py` on a detached hidden state with its own learning rate 1e-3, value coefficient 0.5, gamma 0.99, lambda 0.95, entropy coefficient 0. Also Claude's: 4 games per collection,
a pool of 256 supervised rows whose reference distributions are computed once from the starting weights, so no second model sits in memory, advantages normalized per
iteration, dropout off in PPO steps, hand-over numbers 2, 0 and 3 so that no LLM rounds exist and `fallbacks_in_row` restarts at 2 as in `export_kev`, and `tracking=always`.

Cadence. A dev point at the start, every `DEV_EVERY_STEPS` commands (a tenth of the budget by default) and at the end of a finished run, but not at a stop a resume will continue: Sys1 alone with fallback masked on `DEV_GAMES`, at temperature 1,
which is the learning curve for Figure 1. The state tar is replaced atomically every `CHECKPOINT_EVERY_MIN` minutes at an iteration boundary, for `colab_job.sh -p`, and it holds weights,
optimizer, buffer, schedule position and random state. The loop stops by itself at `MAX_SECONDS`, 7.5 hours by default, so a VM run stays under 8.

`MODE=speedtest` plays one iteration and writes rows per second and a projection. `MODE=smoke` is a path check on a few games and its result says so. Failure raises.
"""
from __future__ import annotations

import contextlib
import dataclasses
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import resource
import subprocess
import sys
import tarfile
import tempfile
import time
import typing
from collections import Counter
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

LAB_SRC = Path(__file__).resolve().parents[1]
if str(LAB_SRC) not in sys.path:
    sys.path.append(str(LAB_SRC))   # appended, so a PYTHONPATH entry can stand in for a module that is not written yet

from alf.env import TRACKING, load_game_list  # noqa: E402
from harness import episode as harness_episode  # noqa: E402
from harness.control import HarnessConfig  # noqa: E402
from harness.episode import EXPERT, run_episodes  # noqa: E402
from harness.metrics import success_by_task_type, summarize  # noqa: E402
from harness.reward import R2, REWARDS, step_reward, terminal_reward, with_reward  # noqa: E402
from sys1.base import FALLBACK_TEXT  # noqa: E402
from sys1.value_head import ValueHead, decide_dim, dropout_off, score_and_decide  # noqa: E402
from train.export_kev import QUESTION_ID  # noqa: E402
from train.refit_temperature import DEV_SPLITS, fit as fit_on_rows, per_game as per_game_rows, score_rows  # noqa: E402

TOPIC = "arm-rl-model-in-control"
MODES = ("full", "speedtest", "smoke")
STAND_IN_NAME = "teacher-stand-in"
FALLBACK_RULE = ("Claude's choice, not Halil's. With no LLM at this stage the fallback option stays in the option list and pays by R2's rule under every reward setting: 1 when the "
                 "teacher's command is off the menu, minus 0.1 when it is on the menu. When Sys1's sampled pick is fallback the teacher's command runs as a stand-in for the LLM, "
                 "and the step is counted in teacher_stand_in_steps.")
RUNTIME_FIELDS = ("train_games", "dev_games", "menu", "init_from", "train_jsonl", "dev_jsonl", "out_dir", "resume", "device", "num_workers", "max_seconds", "checkpoint_every_min",
               "stop_after_iterations", "git_commit", "git_dirty", "max_gb", "min_free_pct", "torch_threads", "mode", "speed_dev_games", "eval_batch")   # may differ after a resume
PACKAGES = ["alfworld", "textworld", "fast-downward-textworld", "kev", "torch", "transformers", "peft", "numpy", "flash-linear-attention", "triton"]


@dataclass
class Config:
    seed: int = 0
    topic: str = TOPIC
    mode: str = "full"
    stem: str = ""
    objective: str = "ppo"   # "ppo", or "ce" in arm-rl-scheduled
    step_budget: int = 0   # Commands run in training games, required in full mode.
    train_games: str = ""
    dev_games: str = ""
    menu: str = ""
    init_from: str = ""
    train_jsonl: str = ""
    dev_jsonl: str = ""
    out_dir: str = "out"
    resume: str = ""   # A folder holding an extracted state tar.
    device: str = "cuda"
    dtype: str = ""   # bf16 autocast over fp32 weights on CUDA, as the classification job, and fp32 elsewhere.
    # The harness.
    k: int = 10
    trace_window: int = 5
    max_steps: int = 50
    fallbacks_before_handover: int = 2
    llm_rounds: int = 0
    menu_run: int = 3
    fallback: str = "stand_in"
    tracking: str = "always"
    reward: str = "R3"
    # Collection.
    games_per_collect: int = 4
    batch_games: int = 8
    num_workers: int = 0
    # The optimizer. A zero means the arm's default, filled in by `resolve`.
    lr: float = 0.0
    lr_warmup: int = 16
    lr_total_updates: int = 0
    beta1: float = 0.9
    beta2: float = 0.0
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    micro_batch: int = 8   # Rows in one forward and backward pass. A minibatch accumulates over these.
    # PPO.
    clip: float = 0.2
    transitions_per_iter: int = 96
    minibatch: int = 8
    ppo_epochs: int = 1
    gamma: float = 0.99
    lam: float = 0.95
    actor_coef: float = 0.1
    vf_coef: float = 0.5
    entropy_coef: float = 0.0
    value_lr: float = 1e-3
    value_detach: bool = True
    normalize_advantages: bool = True
    kl: bool = True
    kl_coef: float = 0.7
    kl_rows: int = 8
    kl_pool_rows: int = 256
    kl_guard: bool = True
    kl_guard_mean: float = 0.01
    kl_guard_row: float = 0.05
    # arm-rl-scheduled.
    schedule_fraction: float = 0.8
    teacher_prob_fixed: float = -1.0   # Any value in [0, 1] replaces the schedule, for a speed test.
    batch: int = 8
    core_share: float = 0.5
    train_rows_per_new_row: float = 2.0
    r3_below_teacher_prob: float = 0.5
    # The run.
    dev_every_steps: int = 0
    dev_games_limit: int = 0
    eval_batch: int = 16
    checkpoint_every_min: float = 10.0
    max_seconds: float = 27000.0
    stop_after_iterations: int = 0
    speed_dev_games: int = 4
    max_gb: float = 0.0   # Above 0 on the Mac, the watchdog of sys1/check_real_weights.py kills the process at this many GB.
    min_free_pct: int = 20
    torch_threads: int = 0
    git_commit: str = ""
    git_dirty: str = ""

    @classmethod
    def from_env(cls, env, **defaults) -> "Config":
        hints = typing.get_type_hints(cls)
        values = dict(defaults)
        for f in fields(cls):
            name = f.name.upper()
            if name in env and env[name] != "":
                kind = hints[f.name]
                values[f.name] = env[name] in ("1", "true", "True") if kind is bool else kind(env[name])
        return cls(**values).resolve()

    def resolve(self) -> "Config":
        """Fills the values that depend on the arm, then refuses a setting that cannot run."""
        if self.mode not in MODES:
            raise ValueError(f"MODE must be one of {MODES}, got {self.mode!r}")
        if self.objective not in ("ppo", "ce"):
            raise ValueError(f"OBJECTIVE must be ppo or ce, got {self.objective!r}")
        if self.fallback not in ("stand_in", "masked"):
            raise ValueError(f"FALLBACK must be stand_in or masked, got {self.fallback!r}")
        if self.tracking not in TRACKING:
            raise ValueError(f"TRACKING must be one of {TRACKING}, got {self.tracking!r}")
        if self.reward not in REWARDS:
            raise ValueError(f"REWARD must be one of {sorted(REWARDS)}, got {self.reward!r}")
        if self.topic == TOPIC and (self.reward not in ("R3", "R4") or self.objective != "ppo"):
            raise ValueError(f"{TOPIC} is PPO on R3 or R4, got {self.objective} on {self.reward}")
        if self.mode == "full":
            if self.step_budget <= 0:
                raise ValueError("full mode needs STEP_BUDGET, which is Open until a speed test measures the rate")
            if self.init_from == "stub" or self.init_from.startswith("jaredpalmer/"):
                raise ValueError("full mode starts from the classification arm's weights, as the plan says, so INIT_FROM must be that checkpoint")
        if self.mode == "speedtest" and self.step_budget <= 0:
            self.step_budget = 10**9   # One iteration runs and the projection takes the budget from its reader.
        self.lr = self.lr or (1e-6 if self.objective == "ppo" else 2e-5)   # 2e-5 is the classification job's value, from Kev's README for --init_from
        self.beta2 = self.beta2 or (0.95 if self.objective == "ppo" else 0.999)   # 0.999 is torch's default, which kev.train uses
        self.dtype = self.dtype or ("bf16" if self.device.startswith("cuda") else "fp32")
        if not self.lr_total_updates:
            self.lr_total_updates = math.ceil(self.step_budget * self.ppo_epochs / self.minibatch) if self.objective == "ppo" else math.ceil(self.step_budget * self.train_rows_per_new_row / self.batch)
        self.dev_every_steps = self.dev_every_steps or max(1, self.step_budget // 10)
        self.stem = self.stem or f"seed{self.seed}-{self.reward if self.topic == TOPIC else self.objective}"
        return self

    def fingerprint(self) -> dict:
        """What must hold across a resume."""
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name not in RUNTIME_FIELDS}


def harness_config(cfg: Config, reward: str) -> HarnessConfig:
    base = HarnessConfig(k=cfg.k or None, fallbacks_before_handover=cfg.fallbacks_before_handover, llm_rounds=cfg.llm_rounds, menu_run=cfg.menu_run, trace_window=cfg.trace_window,
                         w=0.0, c=0.0, beta=0.0, max_steps=cfg.max_steps)
    return with_reward(base, reward)


# The teacher's stand-in for the LLM.

@dataclass
class Pick:
    """The shape of `Pick` in `lab/INTERFACES.md`. `lab/src/llm/` belongs to another session, and nothing here calls an LLM."""
    command: str
    probs: dict | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    seconds: float = 0.0


class TeacherStandIn:
    """Takes the LLM's place in `run_episodes`, inside `teacher_stand_in`. It has no text and no model, and `metrics.summarize` counts it under `llm_calls` and under no LLM's name."""
    name = STAND_IN_NAME
    precision = "none"

    def pick(self, goals, histories, admissibles):
        raise RuntimeError("the teacher's stand-in sees the teacher's label only inside teacher_stand_in()")

    reason_then_command = pick


@contextlib.contextmanager
def teacher_stand_in():
    """For the length of a play, a `TeacherStandIn` answers with what a teacher-run step would run, `label.played`. The LLM interface has no place for the label, so this wraps the
    harness's private `_ask_llm`, which holds each game's view. Any other LLM goes through as before."""
    real = harness_episode._ask_llm

    def ask(games, llm, mode):
        if not isinstance(llm, TeacherStandIn):
            return real(games, llm, mode)
        for g in games:
            if g.view.label is None:
                raise RuntimeError(f"{g.view.uid}: the stand-in needs the teacher, so run with teacher='expert'")
            g.llm_pick = Pick(g.view.label.played)

    harness_episode._ask_llm = ask
    try:
        yield
    finally:
        harness_episode._ask_llm = real


class ValueCollector:
    """What `run_episodes` calls in place of Sys1 when PPO will read values: the same probabilities, and the value of each state from the same forward pass, so no second pass over the prompts is needed."""

    def __init__(self, sys1, value_head):
        self.sys1, self.value_head = sys1, value_head
        self.values: dict[tuple, float] = {}

    @property
    def temperature(self) -> float:
        return self.sys1.temperature

    def probs(self, prompts, options):
        logits, decide = score_and_decide(self.sys1, prompts, options)
        with torch.no_grad():
            values = self.value_head(decide).tolist()
        for prompt, opts, value in zip(prompts, options, values):
            self.values[(prompt, tuple(opts))] = value
        return [torch.softmax(z / self.sys1.temperature, -1) for z in logits]


# Transitions, returns and advantages.

@dataclass
class Transition:
    uid: str
    step: int
    prompt: str
    options: list[str]
    action: int   # Index into the distribution the pick was drawn from: the options, or the menu when fallback was masked.
    masked: bool
    old_log_prob: float
    value: float
    reward: float   # This step's pay plus, discounted, what the steps up to Sys1's next decision paid.
    discount: float   # Gamma to the number of steps until Sys1's next decision.
    advantage: float = 0.0
    ret: float = 0.0
    weight: float = 0.0   # The advantage the loss reads, normalized when asked.


def ratio_mask(log) -> list[bool]:
    """Per step, whether the PPO ratio may read it: only steps where Sys1's sampled pick ran. Teacher-run steps carry a greedy pick that never ran, so they are out."""
    return [s.sampled for s in log.steps]


def decision_reward(s, harness) -> float:
    """What a step Sys1 decided pays. A fallback pays R2's rule under every setting (`FALLBACK_RULE`), which differs from `step_reward` under R4 only."""
    if s.pick == FALLBACK_TEXT:
        return step_reward(s.pick, s.teacher_command, s.menu, R2)
    return step_reward(s.pick, s.teacher_command, s.menu, harness)


def returns_and_advantages(rewards: list[float], values: list[float], discounts: list[float], lam: float) -> tuple[list[float], list[float]]:
    """GAE over one game's decisions, ending the game after the last one. -> (advantages, returns), where a return is the advantage plus the value."""
    advantages = [0.0] * len(rewards)
    following = 0.0
    for t in reversed(range(len(rewards))):
        terminal = t == len(rewards) - 1
        next_value = 0.0 if terminal else values[t + 1]
        delta = rewards[t] + (0.0 if terminal else discounts[t]) * next_value - values[t]
        following = delta + (0.0 if terminal else discounts[t] * lam * following)
        advantages[t] = following
    return advantages, [a + v for a, v in zip(advantages, values)]


def build_transitions(logs, harness, gamma: float, lam: float, values: dict) -> list[Transition]:
    out = []
    for log in logs:
        mask = ratio_mask(log)
        game: list[Transition] = []
        for i, s in enumerate(log.steps):
            pay = decision_reward(s, harness) if mask[i] else 0.0
            if i == len(log.steps) - 1:
                pay += terminal_reward(log.won, harness)
            if mask[i]:
                if s.actor == "teacher":
                    raise RuntimeError(f"{log.uid} step {s.step}: a teacher-run step is marked sampled")
                names = s.menu if s.masked else s.options
                game.append(Transition(log.uid, s.step, s.prompt, list(s.options), names.index(s.pick), s.masked, s.log_prob, values[(s.prompt, tuple(s.options))], pay, gamma))
            elif game:
                game[-1].reward += game[-1].discount * pay
                game[-1].discount *= gamma
        if game:
            advantages, returns = returns_and_advantages([t.reward for t in game], [t.value for t in game], [t.discount for t in game], lam)
            for t, a, r in zip(game, advantages, returns):
                t.advantage, t.ret, t.weight = a, r, a
        out.extend(game)
    return out


def normalize_weights(transitions: list[Transition]) -> None:
    """Mean 0 and standard deviation 1 over one iteration."""
    if len(transitions) < 2:
        return
    a = np.array([t.advantage for t in transitions], dtype=float)
    spread = a.std()
    for t, value in zip(transitions, (a - a.mean()) / (spread if spread > 0 else 1.0)):
        t.weight = float(value)


# Losses.

def policy_loss(new_log_probs: torch.Tensor, old_log_probs: torch.Tensor, advantages: torch.Tensor, clip: float) -> tuple[torch.Tensor, torch.Tensor]:
    """PPO's clipped surrogate. -> (loss, ratios)."""
    ratio = torch.exp(new_log_probs - old_log_probs)
    return -torch.min(ratio * advantages, torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantages).mean(), ratio


def value_loss(values: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    return 0.5 * (values - returns).pow(2).mean()


def log_prob_and_entropy(logits: torch.Tensor, action: int, masked: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Under the distribution the pick was drawn from. Masked, the last option is out of the softmax, which is the harness's `masked_probs`."""
    logp = F.log_softmax(logits[:-1] if masked else logits, -1)
    return logp[action], -(logp.exp() * logp).sum()


def kl_to_reference(reference_log_probs: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """KL(reference || current) in nats for one row."""
    return (reference_log_probs.exp() * (reference_log_probs - F.log_softmax(logits, -1))).sum()


def cosine_factor(update: int, warmup: int, total: int) -> float:
    """The share of the peak learning rate at this optimizer step: a linear rise over `warmup` steps, then a cosine reaching 0 at `total`, and 0 after."""
    if total <= 0:
        return 1.0
    if update < warmup:
        return (update + 1) / warmup
    progress = min(1.0, (update - warmup) / max(1, total - warmup))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


# Rows, files and provenance.

def read_kev_rows(path, allowed_splits: tuple[str, ...]) -> list[dict]:
    """Rows of the export's JSONL, plain or gzip. Every row's split is checked, and a test game stops the read, as every training loader must."""
    opener = gzip.open if str(path).endswith(".gz") else open
    rows = []
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            meta = row.get("_meta", {})
            if meta.get("split") not in allowed_splits:
                raise AssertionError(f"{path}: split must be one of {allowed_splits}, got {meta.get('split')!r} at {meta.get('id')}")
            if "valid_unseen" in str(meta.get("game_id", "")):
                raise AssertionError(f"{path}: test game {meta.get('game_id')}")
            rows.append(row)
    return rows


def labelled_row(kev_row: dict) -> dict:
    """One row of the core training set in the form both the buffer and the KL pool hold."""
    question = kev_row["questions"][QUESTION_ID]
    meta = kev_row.get("_meta", {})
    return {"state": kev_row["state"], "options": list(question["criteria"]), "label": question["label"], "target": question.get("target"), "game": meta.get("game_id"), "step": meta.get("step"), "source": "core"}


def chunks(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def params_sha256(sys1) -> str:
    """Of the trainable parameters, by name. A resume compares it, so a state that did not restore exactly stops the run."""
    digest = hashlib.sha256()
    for name, p in sys1.named_parameters():
        if p.requires_grad:
            digest.update(name.encode())
            digest.update(p.detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()


def release_cache(device) -> None:
    if str(device) == "mps":   # kev.train does this per step (kev/train.py:638)
        torch.mps.synchronize()
        torch.mps.empty_cache()


def file_sha256(path) -> str | None:
    if not path or not Path(path).is_file():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, default=str) + "\n")
    os.replace(tmp, path)


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment() -> dict:
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": {n: _version(n) for n in PACKAGES}, "alfworld_data": os.environ.get("ALFWORLD_DATA"),
            "torch_cuda": torch.version.cuda, "torch_threads": torch.get_num_threads(), "deterministic_algorithms": False}


def hardware(device: str) -> dict:
    out = {"device": device, "machine": platform.machine(), "cpus": os.cpu_count()}
    if device.startswith("cuda") and torch.cuda.is_available():
        out["gpu"] = torch.cuda.get_device_name(0)
    if sys.platform == "darwin":
        with contextlib.suppress(OSError, subprocess.CalledProcessError):
            out["memory_bytes"] = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True).stdout.strip())
    return out


def git_state(cfg: Config) -> tuple[str, str]:
    if cfg.git_commit:
        return cfg.git_commit, cfg.git_dirty or "unknown"
    root = LAB_SRC.parents[1]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "lab/src", "lab/jobs"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        return commit, str(int(bool(dirty)))
    except (OSError, subprocess.CalledProcessError):
        return "unknown", "unknown"


def peak_memory(watchdog=None) -> dict:
    scale = 1 if sys.platform == "darwin" else 1024   # ru_maxrss is bytes on macOS and kilobytes on Linux
    out = {"ru_maxrss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale, "largest_child_ru_maxrss_bytes": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * scale}
    if torch.cuda.is_available():
        out |= {"cuda_max_allocated_bytes": torch.cuda.max_memory_allocated(), "cuda_max_reserved_bytes": torch.cuda.max_memory_reserved(), "cuda_total_bytes": torch.cuda.get_device_properties(0).total_memory}
    if watchdog is not None:
        out |= {"mac_physical_footprint_peak_bytes": watchdog.peak, "mac_lowest_system_free_pct": watchdog.lowest_free_pct}
    return out


# The training loop both arms share.

class Trainer:
    """One arm's run. `iteration` is the arm's own, and the rest is what both arms promise: games in order, state in one tar, the learning curve, the result file."""

    def __init__(self, cfg: Config, sys1, menu, train_games, dev_games, core_rows=(), dev_rows=(), inputs: dict | None = None, watchdog=None):
        self.cfg, self.sys1, self.menu, self.watchdog = cfg, sys1, menu, watchdog
        self.train_games, self.dev_games = list(train_games), list(dev_games)[:cfg.dev_games_limit or None]
        self.core = [labelled_row(r) for r in core_rows]
        self.dev_rows = list(dev_rows)
        self.inputs = inputs or {}
        self.pos = {"iteration": 0, "plays": 0, "env_steps": 0, "games_played": 0, "updates": 0, "cursor": 0, "epoch": 0, "next_dev_at": cfg.dev_every_steps}
        self.counts: Counter = Counter()
        self.seconds: Counter = Counter()
        self.rng = random.Random(f"{cfg.topic}/{cfg.seed}")
        self.curve: list[dict] = []
        self.iterations: list[dict] = []
        self.last_dev: dict | None = None
        self.kl_pool: list[dict] = []
        self.wall_before = 0.0
        self.clock = time.time()
        self.masked = cfg.fallback == "masked"
        self.llm = None if self.masked else TeacherStandIn()
        self.teacher = None if self.masked and cfg.reward == "R4" and cfg.objective == "ppo" else EXPERT
        self.value_head, self.collector = None, None
        trainable = list(sys1.trainable_parameters())
        if not trainable:
            raise ValueError("this Sys1 has nothing to train: a KevSys1 must load with trainable=True")
        groups = [{"params": trainable, "lr": cfg.lr}]
        if cfg.objective == "ppo":
            self.value_head = ValueHead(decide_dim(sys1), seed=cfg.seed).to(sys1.device)
            self.value_head.eval()
            self.collector = ValueCollector(sys1, self.value_head)
            groups.append({"params": list(self.value_head.parameters()), "lr": cfg.value_lr})
        self.optimizer = torch.optim.AdamW(groups, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay)

    # Games.

    def next_games(self, n: int) -> list:
        """The next games of this epoch's order, which comes from the seed and the epoch alone. A slice stops at the end of an epoch, so no game appears twice in a batch."""
        order = list(range(len(self.train_games)))
        random.Random(f"{self.cfg.seed}/epoch/{self.pos['epoch']}").shuffle(order)
        taken = order[self.pos["cursor"]:self.pos["cursor"] + n]
        self.pos["cursor"] += len(taken)
        if self.pos["cursor"] >= len(order):
            self.pos["cursor"], self.pos["epoch"] = 0, self.pos["epoch"] + 1
        return [self.train_games[i] for i in taken]

    def play_seed(self) -> int:
        """Per play, so a game seen again later gets new draws."""
        return (self.cfg.seed + 1) * 1_000_003 + self.pos["plays"]

    def play(self, games, harness, mode: str, teacher_prob: float = 0.0) -> list:
        self.sys1.eval()
        if self.sys1.temperature != 1.0:
            raise RuntimeError(f"Sys1 trains at temperature 1 and reads {self.sys1.temperature}: refit only after training")
        started = time.perf_counter()
        with teacher_stand_in():
            logs = run_episodes(games, self.collector or self.sys1, self.llm, self.menu, harness, mode, teacher=self.teacher, teacher_prob=teacher_prob, seed=self.play_seed(), mask_fallback=self.masked,
                                batch_size=self.cfg.batch_games, num_workers=self.cfg.num_workers, tracking=self.cfg.tracking, keep_prompts=True)
        self.seconds["collect"] += time.perf_counter() - started
        self.pos["plays"] += 1
        self.count_steps(logs)
        return logs

    def count_steps(self, logs) -> None:
        for log in logs:
            self.pos["games_played"] += 1
            self.pos["env_steps"] += log.n_steps
            self.counts["games_won"] += log.won
            stand_in = 0
            for s in log.steps:
                self.counts["sys1_questions"] += s.probs is not None
                self.seconds["sys1_inference"] += s.seconds_sys1
                if s.llm_name == STAND_IN_NAME:
                    stand_in += 1
                    self.counts["teacher_stand_in_steps"] += 1
                    self.counts["teacher_stand_in_steps_with_flagged_label"] += s.teacher_command is None
                self.counts[f"{s.actor}_run_steps"] += 1
                self.counts["sampled_steps"] += s.sampled
                self.counts["sampled_fallback_picks"] += s.sampled and s.pick == FALLBACK_TEXT
                self.counts["steps_with_label_random"] += s.label_random
                self.counts["steps_with_label_inadmissible"] += s.label_inadmissible and not s.label_random
                self.counts["steps_with_label_error"] += s.label_error is not None
            self.counts["games_won_with_stand_in"] += log.won and stand_in > 0

    # Dev.

    def dev_point(self, label: str) -> None:
        """One point of the learning curve: Sys1 alone with fallback masked, greedy, at temperature 1. The gates are read later by `lab/jobs/eval-sys1-alone.py`, after the refit."""
        if not self.dev_games:
            return
        self.sys1.eval()
        started = time.perf_counter()
        logs = run_episodes(self.dev_games, self.sys1, None, self.menu, harness_config(self.cfg, "R2"), "test", teacher=EXPERT, seed=self.cfg.seed, mask_fallback=True,
                            batch_size=self.cfg.batch_games, num_workers=self.cfg.num_workers, tracking=self.cfg.tracking, keep_prompts=False)
        spent = time.perf_counter() - started
        self.seconds["dev"] += spent
        summary = {k: v for k, v in summarize(logs).items() if k != "coverage_curve"}
        point = {"label": label, "env_steps": self.pos["env_steps"], "games_played": self.pos["games_played"], "updates": self.pos["updates"], "wall_seconds": round(self.wall(), 1), "dev_seconds": round(spent, 1),
                 "temperature": float(self.sys1.temperature), "n_games": len(logs), **{k: summary.get(k) for k in ("success", "mean_steps", "repeat_rate", "step_match", "ece", "brier", "nll", "auroc", "fallback_when_off_menu", "fallback_when_on_menu", "teacher_on_menu_share")},
                 "success_by_task_type": success_by_task_type(logs)}
        self.curve.append(point)
        self.last_dev = {"aggregates": summary, "per_game": [e.outcome() for e in logs]}
        print("dev", json.dumps({k: point[k] for k in ("label", "env_steps", "success", "step_match", "ece", "repeat_rate", "dev_seconds")}, default=str), flush=True)

    # State.

    def wall(self) -> float:
        return self.wall_before + (time.time() - self.clock)

    def extra_state(self) -> dict:
        return {}

    def load_extra_state(self, extra: dict) -> None:
        pass

    def state(self) -> dict:
        rng = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "trainer": self.rng.getstate()}
        if torch.cuda.is_available():
            rng["cuda"] = torch.cuda.get_rng_state_all()
        if torch.backends.mps.is_available():
            rng["mps"] = torch.mps.get_rng_state()
        return {"config": self.cfg.fingerprint(), "inputs": self.inputs, "pos": dict(self.pos), "counts": dict(self.counts), "seconds": dict(self.seconds), "curve": self.curve, "iterations": self.iterations, "last_dev": self.last_dev,
                "kl_pool": self.kl_pool, "wall_seconds": self.wall(), "params_sha256": params_sha256(self.sys1), "rng": rng, "extra": self.extra_state()}

    def save_state(self) -> Path:
        """Replaces `<stem>-state.tar` atomically, so a pull never sees half a file. Extracted, it holds `state/` for `load_state`."""
        started = time.perf_counter()
        out = Path(self.cfg.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        tar_path = out / f"{self.cfg.stem}-state.tar"
        tmp = tar_path.with_name(tar_path.name + ".tmp")
        with tempfile.TemporaryDirectory(dir=out) as d:
            folder = Path(d)
            self.sys1.save(folder / "sys1")
            torch.save(self.optimizer.state_dict(), folder / "optimizer.pt")
            if self.value_head is not None:
                torch.save(self.value_head.state_dict(), folder / "value_head.pt")
            torch.save(self.state(), folder / "trainer.pt")
            with tarfile.open(tmp, "w") as tar:
                tar.add(folder, arcname="state")
        os.replace(tmp, tar_path)
        self.seconds["checkpoint"] += time.perf_counter() - started
        self.counts["checkpoints_written"] += 1
        return tar_path

    def load_state(self, folder) -> None:
        """`folder` holds what `save_state` wrote, and Sys1's weights in `folder/sys1` were already loaded into this trainer's Sys1."""
        folder = Path(folder)
        saved = torch.load(folder / "trainer.pt", map_location="cpu", weights_only=False)   # this run's own file
        now = self.cfg.fingerprint()
        differ = sorted(k for k in now if saved["config"].get(k) != now[k])
        if differ:
            raise ValueError(f"the state was written under other settings: {differ}")
        if self.inputs != saved["inputs"]:
            raise ValueError(f"the state was written on other inputs: {saved['inputs']} against {self.inputs}")
        if params_sha256(self.sys1) != saved["params_sha256"]:
            raise ValueError("Sys1's weights differ from the ones this state saved")
        self.optimizer.load_state_dict(torch.load(folder / "optimizer.pt", map_location=self.sys1.device, weights_only=True))
        if self.value_head is not None:
            self.value_head.load_state_dict(torch.load(folder / "value_head.pt", map_location=self.sys1.device, weights_only=True))
        self.pos, self.counts, self.seconds = saved["pos"], Counter(saved["counts"]), Counter(saved["seconds"])
        self.curve, self.iterations, self.last_dev, self.kl_pool = saved["curve"], saved["iterations"], saved["last_dev"], saved["kl_pool"]
        self.wall_before, self.clock = saved["wall_seconds"], time.time()
        rng = saved["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        self.rng.setstate(rng["trainer"])
        if "cuda" in rng and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
        if "mps" in rng and torch.backends.mps.is_available():
            torch.mps.set_rng_state(rng["mps"])
        self.load_extra_state(saved["extra"])
        self.counts["resumes"] += 1
        print(f"resumed at env step {self.pos['env_steps']}, iteration {self.pos['iteration']}, update {self.pos['updates']}", flush=True)

    # The arm's work.

    def prepare(self) -> None:
        pass

    def iteration(self) -> dict:
        raise NotImplementedError

    def run(self) -> dict:
        cfg = self.cfg
        self.prepare()
        if not self.curve:
            self.dev_point("start")
        saved_at = time.time()
        done_here = 0
        stopped = None
        while self.pos["env_steps"] < cfg.step_budget:
            if cfg.stop_after_iterations and done_here >= cfg.stop_after_iterations:
                stopped = "stop_after_iterations"
                break
            if time.time() - self.clock >= cfg.max_seconds:
                stopped = "max_seconds"
                break
            self.iterations.append(self.iteration())
            self.pos["iteration"] += 1
            done_here += 1
            print("iteration", json.dumps(self.iterations[-1], default=str), flush=True)
            if self.pos["env_steps"] >= self.pos["next_dev_at"]:
                self.dev_point("cadence")
                while self.pos["next_dev_at"] <= self.pos["env_steps"]:
                    self.pos["next_dev_at"] += cfg.dev_every_steps
            self.write_curve()
            if time.time() - saved_at >= cfg.checkpoint_every_min * 60:
                self.save_state()
                saved_at = time.time()
        if stopped is None and self.curve[-1]["env_steps"] != self.pos["env_steps"]:   # a stop that a resume will continue adds no point, so a resumed curve equals a straight one
            self.dev_point("end")
        self.save_state()
        self.write_curve()
        return self.finish(stopped)

    def write_curve(self) -> None:
        write_json(Path(self.cfg.out_dir) / f"{self.cfg.stem}-curve.json", {"seed": self.cfg.seed, "topic": self.cfg.topic, "mode": self.cfg.mode, "config": dataclasses.asdict(self.cfg), "learning_curve": self.curve, "iterations": self.iterations})

    def refit(self) -> dict:
        """The classification job's fit (`train.refit_temperature`) on dev rows, on this process's model, then the final checkpoint, which `lab/jobs/eval-sys1-alone.py` loads."""
        cfg = self.cfg
        self.sys1.eval()
        scored, spent = score_rows(self.sys1, self.dev_rows, cfg.eval_batch)
        fitted = fit_on_rows(scored, cfg.seed)
        self.sys1.temperature = fitted["temperature"]
        out = Path(cfg.out_dir)
        tar_path, tmp = out / f"{cfg.stem}.tar", out / f"{cfg.stem}.tar.tmp"
        with tempfile.TemporaryDirectory(dir=out) as d:
            self.sys1.save(Path(d) / "checkpoint")
            with tarfile.open(tmp, "w") as tar:
                tar.add(Path(d) / "checkpoint", arcname="checkpoint")
        os.replace(tmp, tar_path)
        return {"temperature": fitted["temperature"], "temperature_fit": fitted, "dev_rows_per_game": per_game_rows(scored, fitted["temperature"]), "dev_rows_score_seconds": round(spent, 2),
                "teacher_state_step_match": float(np.mean([np.argmax(r["p"]) == r["label"] for r in scored])) if scored else None, "checkpoint_tar_bytes": tar_path.stat().st_size}

    def base_result(self) -> dict:
        cfg = self.cfg
        commit, dirty = git_state(cfg)
        steps, wall = self.pos["env_steps"], self.wall()
        questions = self.counts["sys1_questions"]
        menu_module = sys.modules.get("menu")
        out = {"topic": cfg.topic, "mode": cfg.mode, "seed": cfg.seed, "git_commit": commit, "git_dirty": dirty, "environment": environment(), "hardware": hardware(cfg.device), "precision": getattr(self.sys1, "precision", str(next(self.sys1.parameters()).dtype)),
               "config": dataclasses.asdict(cfg), "inputs": self.inputs, "menu_module_file": getattr(menu_module, "__file__", None), "fallback_rule": FALLBACK_RULE, "games_played": self.pos["games_played"], "steps": steps, "steps_per_second": round(steps / wall, 4) if wall else None,
               "updates": self.pos["updates"], "iterations_run": self.pos["iteration"], "wall_seconds": round(wall, 1), "seconds": {k: round(v, 2) for k, v in self.seconds.items()},
               "sys1_inference_rows_per_second": round(questions / self.seconds["sys1_inference"], 3) if self.seconds["sys1_inference"] else None, "counts": dict(self.counts),
               "teacher_stand_in_share_of_steps": round(self.counts["teacher_stand_in_steps"] / steps, 4) if steps else None, "peak_memory": peak_memory(self.watchdog), "learning_curve": self.curve, "iterations": self.iterations,
               "aggregates": (self.last_dev or {}).get("aggregates"), "per_game": (self.last_dev or {}).get("per_game"), "provenance_of_weights": getattr(self.sys1, "provenance", None), "deterministic_algorithms": False}
        if cfg.mode == "smoke":
            out["scope"] = "A path check on a few games. It says nothing about Sys1's quality, and no number here may be cited for it."
        else:
            out["scope"] = f"Measured with device {cfg.device}. The gates are read by lab/jobs/eval-sys1-alone.py, not by this job."
        return out

    def finish(self, stopped: str | None) -> dict:
        result = self.base_result()
        result |= {"completed": stopped is None, "stopped_because": stopped}
        if stopped is None:
            result |= self.refit()
        else:
            result["temperature"] = None
        return result

    def run_speedtest(self) -> dict:
        """One iteration, a dev play on a few games and one checkpoint. The projection is for planning and is no measurement of a full run."""
        cfg = self.cfg
        n_dev_all = len(self.dev_games)
        self.dev_games = self.dev_games[:cfg.speed_dev_games]
        self.prepare()
        self.dev_point("speedtest")
        self.iterations.append(self.iteration())
        self.pos["iteration"] += 1
        self.save_state()
        result = self.base_result()
        steps = self.pos["env_steps"]
        per_step = (self.seconds["collect"] + self.seconds["update"]) / steps if steps else None
        dev_play = self.seconds["dev"] / max(1, len(self.dev_games)) * n_dev_all
        projection = {"seconds_per_command_collect_and_update": round(per_step, 4) if per_step else None, "dev_play_seconds_all_games": round(dev_play, 1), "dev_games_all": n_dev_all, "note": "For planning only, from one iteration."}
        if per_step:
            eight_hours = 8 * 3600 - 11 * dev_play
            projection |= {"step_budget_fitting_8_hours_with_11_dev_plays": int(max(0.0, eight_hours) / per_step), "commands_per_second_collect_and_update": round(1 / per_step, 3)}
            projection["cu_per_1000_commands_at_1_54_per_hour"] = round(1000 * per_step / 3600 * 1.54, 4)
        result |= {"completed": False, "stopped_because": "speedtest", "temperature": None, "projection": projection, "state_tar_bytes": (Path(cfg.out_dir) / f"{cfg.stem}-state.tar").stat().st_size}
        return result


class PPOTrainer(Trainer):
    """`arm-rl-model-in-control`: Sys1's sampled pick at every step, PPO on R3 or R4."""

    def prepare(self) -> None:
        if self.cfg.kl and not self.kl_pool:
            self.kl_pool = self.build_kl_pool()

    def build_kl_pool(self) -> list[dict]:
        """Supervised rows with their distributions under the starting weights, computed once. A resume reads them from the state."""
        if not self.core:
            raise ValueError("the KL term replays supervised rows, so it needs TRAIN_JSONL, or KL=0")
        cfg = self.cfg
        started = time.perf_counter()
        self.sys1.eval()
        rows = self.rng.sample(self.core, min(cfg.kl_pool_rows, len(self.core)))
        pool = []
        for part in chunks(rows, cfg.eval_batch):
            with torch.no_grad():
                logits = self.sys1.score([r["state"] for r in part], [r["options"] for r in part])
            pool += [{**r, "reference": F.log_softmax(z, -1).tolist()} for r, z in zip(part, logits)]
        self.seconds["kl_reference"] += time.perf_counter() - started
        return pool

    def set_lr(self) -> float:
        factor = cosine_factor(self.pos["updates"], self.cfg.lr_warmup, self.cfg.lr_total_updates)
        self.optimizer.param_groups[0]["lr"] = self.cfg.lr * factor
        if self.value_head is not None:
            self.optimizer.param_groups[1]["lr"] = self.cfg.value_lr * factor
        return factor

    def clip_and_step(self) -> tuple[float, float]:
        cfg = self.cfg
        norm = float(torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]["params"], cfg.max_grad_norm, error_if_nonfinite=True))
        value_norm = 0.0
        if self.value_head is not None:   # apart, so the value's scale cannot shrink Sys1's gradient
            value_norm = float(torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[1]["params"], cfg.max_grad_norm, error_if_nonfinite=True))
        self.optimizer.step()
        self.pos["updates"] += 1
        release_cache(self.sys1.device)
        return norm, value_norm

    def zero_first_moment(self) -> None:
        for p in self.optimizer.param_groups[0]["params"]:
            state = self.optimizer.state.get(p)
            if state and "exp_avg" in state:
                state["exp_avg"].zero_()

    def kl_backward(self) -> list[float]:
        """Backward on the KL term, a pass per micro batch. -> each replayed row's KL."""
        cfg = self.cfg
        rows = [self.rng.choice(self.kl_pool) for _ in range(cfg.kl_rows)]
        per_row: list[float] = []
        for part in chunks(rows, cfg.micro_batch):
            logits = self.sys1.score([r["state"] for r in part], [r["options"] for r in part])
            values = [kl_to_reference(torch.tensor(r["reference"], dtype=z.dtype, device=z.device), z) for r, z in zip(part, logits)]
            (cfg.kl_coef * torch.stack(values).sum() / len(rows)).backward()
            per_row += [float(v.detach()) for v in values]
        return per_row

    def ppo_step(self, batch: list[Transition]) -> dict:
        """One optimizer step on one minibatch. The KL term runs first because its size decides whether the guard drops the actor term."""
        cfg = self.cfg
        factor = self.set_lr()
        self.optimizer.zero_grad(set_to_none=True)
        stats: dict = {"lr_factor": factor}
        tripped = False
        if cfg.kl:
            per_row = self.kl_backward()
            stats |= {"kl_mean": float(np.mean(per_row)), "kl_max": max(per_row)}
            tripped = cfg.kl_guard and (stats["kl_mean"] > cfg.kl_guard_mean or stats["kl_max"] > cfg.kl_guard_row)
        stats["kl_guard_tripped"] = tripped
        if tripped:
            self.zero_first_moment()
            self.counts["kl_guard_steps"] += 1
        else:
            pi, v, entropy, ratios, kl_old = [], [], [], [], []
            for part in chunks(batch, cfg.micro_batch):
                logits, decide = score_and_decide(self.sys1, [t.prompt for t in part], [t.options for t in part])
                pairs = [log_prob_and_entropy(z, t.action, t.masked) for z, t in zip(logits, part)]
                new = torch.stack([a for a, _ in pairs])
                part_entropy = torch.stack([b for _, b in pairs]).mean()
                like = {"dtype": new.dtype, "device": new.device}
                old = torch.tensor([t.old_log_prob for t in part], **like)
                pi_loss, ratio = policy_loss(new, old, torch.tensor([t.weight for t in part], **like), cfg.clip)
                predicted = self.value_head(decide.detach() if cfg.value_detach else decide)
                v_loss = value_loss(predicted, torch.tensor([t.ret for t in part], dtype=predicted.dtype, device=predicted.device))
                loss = (cfg.actor_coef * pi_loss + cfg.vf_coef * v_loss - cfg.entropy_coef * part_entropy) * len(part) / len(batch)
                loss.backward()
                share = len(part) / len(batch)
                pi.append(float(pi_loss.detach()) * share)
                v.append(float(v_loss.detach()) * share)
                entropy.append(float(part_entropy.detach()) * share)
                ratios += ratio.detach().float().cpu().tolist()
                kl_old += (old - new.detach()).float().cpu().tolist()
            stats |= {"policy_loss": sum(pi), "value_loss": sum(v), "entropy": sum(entropy), "ratio_mean": float(np.mean(ratios)), "ratio_deviation_mean": float(np.mean(np.abs(np.array(ratios) - 1.0))),
                      "clip_fraction": float(np.mean(np.abs(np.array(ratios) - 1.0) > cfg.clip)), "approx_kl": float(np.mean(kl_old))}
        stats["grad_norm"], stats["value_grad_norm"] = self.clip_and_step()
        return stats

    def ppo_update(self, transitions: list[Transition]) -> dict:
        cfg = self.cfg
        if not transitions:
            self.counts["iterations_without_transitions"] += 1
            return {"transitions": 0}
        started = time.perf_counter()
        if cfg.normalize_advantages:
            normalize_weights(transitions)
        self.sys1.train()
        dropout_off(self.sys1)
        self.value_head.train()
        stats = []
        for _ in range(cfg.ppo_epochs):
            order = list(range(len(transitions)))
            self.rng.shuffle(order)
            for part in chunks(order, cfg.minibatch):
                stats.append(self.ppo_step([transitions[i] for i in part]))
        self.sys1.eval()
        self.value_head.eval()
        self.seconds["update"] += time.perf_counter() - started
        returns, values = np.array([t.ret for t in transitions]), np.array([t.value for t in transitions])
        spread = returns.var()

        def mean(key):
            present = [s[key] for s in stats if key in s]
            return float(np.mean(present)) if present else None

        rows = len(transitions) * cfg.ppo_epochs
        self.counts["ppo_rows_trained"] += rows
        self.counts["kl_rows_trained"] += len(stats) * cfg.kl_rows if cfg.kl else 0
        first = next((s for s in stats if "ratio_deviation_mean" in s), None)
        return {"transitions": len(transitions), "minibatches": len(stats), "mean_return_at_start": float(returns.mean()), "explained_variance": float(1 - np.var(returns - values) / spread) if spread > 0 else None,
                "policy_loss": mean("policy_loss"), "value_loss": mean("value_loss"), "entropy": mean("entropy"), "clip_fraction": mean("clip_fraction"), "approx_kl": mean("approx_kl"), "kl_mean": mean("kl_mean"), "kl_max": mean("kl_max"),
                "kl_guard_steps": sum(s["kl_guard_tripped"] for s in stats), "grad_norm": mean("grad_norm"), "value_grad_norm": mean("value_grad_norm"), "lr_factor_last": stats[-1]["lr_factor"],
                "first_ratio_deviation": first["ratio_deviation_mean"] if first and stats[0] is first else None}

    def iteration(self) -> dict:
        cfg = self.cfg
        harness = harness_config(cfg, cfg.reward)
        started = self.pos["env_steps"]
        logs, transitions = [], []
        while len(transitions) < cfg.transitions_per_iter and self.pos["env_steps"] < cfg.step_budget:
            self.collector.values.clear()
            played = self.play(self.next_games(cfg.games_per_collect), harness, "model")
            transitions += build_transitions(played, harness, cfg.gamma, cfg.lam, self.collector.values)
            logs += played
        update = self.ppo_update(transitions)
        return self.iteration_summary(logs, started, cfg.reward, 0.0) | update

    def iteration_summary(self, logs, started: int, reward: str, teacher_prob: float) -> dict:
        steps = [s for log in logs for s in log.steps]
        return {"iteration": self.pos["iteration"], "env_steps": self.pos["env_steps"], "games_played": self.pos["games_played"], "updates": self.pos["updates"], "reward": reward, "teacher_prob": teacher_prob, "new_steps": self.pos["env_steps"] - started,
                "games": len(logs), "train_success": sum(log.won for log in logs) / len(logs) if logs else None, "mean_game_steps": len(steps) / len(logs) if logs else None,
                "teacher_stand_in_steps": sum(s.llm_name == STAND_IN_NAME for s in steps), "teacher_run_steps": sum(s.actor == "teacher" for s in steps), "sys1_run_steps": sum(s.actor == "sys1" for s in steps), "wall_seconds": round(self.wall(), 1)}


# Running a job from environment variables.

def load_sys1(spec: str, cfg: Config):
    """`stub`, or a checkpoint folder or Hub id. RL always loads with the LoRA unmerged, whose fitted temperature starts at 1, so a classification refit does not reach training."""
    from sys1.stub import StubSys1
    if spec == "stub":
        return StubSys1(seed=cfg.seed, device=cfg.device)
    if (Path(spec) / "stub.pt").exists():
        return StubSys1.load(spec, cfg.device)
    from sys1.kev_adapter import KevSys1
    return KevSys1.load(spec, cfg.device, trainable=True, autocast_bf16=cfg.dtype == "bf16")


def load_menu(path: str):
    from menu import MenuModel   # the alf-data session owns this module, so it is imported only when needed
    return MenuModel.load(path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_job(cfg: Config, trainer_class) -> dict:
    """Reads the inputs, starts or resumes the arm, and writes `<stem>.json`. A run stopped by `MAX_SECONDS` writes its result with `completed` false, and a later VM resumes it."""
    seed_everything(cfg.seed)
    if cfg.torch_threads:
        torch.set_num_threads(cfg.torch_threads)
    watchdog = None
    if cfg.max_gb > 0 and sys.platform == "darwin":
        from sys1.check_real_weights import Watchdog
        watchdog = Watchdog(cfg.max_gb, cfg.min_free_pct)
        watchdog.start()
    train_games, dev_games = load_game_list(cfg.train_games), load_game_list(cfg.dev_games)
    wrong_split = [g.uid for g in train_games if g.split != "train"]
    if wrong_split:
        raise AssertionError(f"RL trains on training games only, got {wrong_split[:3]}")
    shared = {g.uid for g in train_games} & {g.uid for g in dev_games}
    if shared:
        raise AssertionError(f"{len(shared)} games sit in both the training list and the dev list, for example {sorted(shared)[0]}")
    menu = load_menu(cfg.menu)
    core_rows = read_kev_rows(cfg.train_jsonl, ("train",)) if cfg.train_jsonl else []
    dev_rows = read_kev_rows(cfg.dev_jsonl, DEV_SPLITS) if cfg.dev_jsonl else []
    if not dev_rows and cfg.mode != "speedtest":
        raise ValueError("the temperature refit needs DEV_JSONL")
    inputs = {name: file_sha256(getattr(cfg, name)) for name in ("train_games", "dev_games", "menu", "train_jsonl", "dev_jsonl")}
    resume = Path(cfg.resume) if cfg.resume else None
    started = time.perf_counter()
    sys1 = load_sys1(str(resume / "sys1") if resume else cfg.init_from, cfg)
    load_seconds = time.perf_counter() - started
    trainer = trainer_class(cfg, sys1, menu, train_games, dev_games, core_rows, dev_rows, inputs, watchdog)
    if resume:
        trainer.load_state(resume)
    result = trainer.run_speedtest() if cfg.mode == "speedtest" else trainer.run()
    result["sys1_load_seconds"] = round(load_seconds, 1)
    write_json(Path(cfg.out_dir) / f"{cfg.stem}.json", result)
    print("RESULT", json.dumps({k: result.get(k) for k in ("topic", "mode", "seed", "completed", "games_played", "steps", "steps_per_second", "sys1_inference_rows_per_second", "teacher_stand_in_share_of_steps", "temperature", "peak_memory", "wall_seconds")}, default=str), flush=True)
    return result


def main() -> None:
    run_job(Config.from_env(os.environ, topic=TOPIC, objective="ppo", reward="R3"), PPOTrainer)


if __name__ == "__main__":
    main()
