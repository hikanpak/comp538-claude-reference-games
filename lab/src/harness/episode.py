"""`run_episodes`: games in a batch, Sys1 and the LLM scoring lists, the hand-over rules deciding who acts.

Whose command runs, by `mode`:

  - `teacher`: the teacher's, at every step. Sys1 is scored greedily where it controls, and nothing it says runs. `sys1=None`
    is allowed here, which plays the expert through the harness alone.
  - `scheduled`: the teacher's with probability `teacher_prob` at each step, otherwise Sys1's sampled pick.
  - `model`: Sys1's sampled pick.
  - `test`: Sys1's greedy pick.

The teacher's label is logged when `teacher="expert"`, which `teacher` and `scheduled` assume by default. `model` and `test`
default to no teacher, so a test game never touches the expert.

`mask_fallback=True` removes fallback from Sys1's distribution and renormalizes the rest. `llm=None` with `mask_fallback=True`
is Sys1 alone. With `llm=None` and fallback unmasked there is nobody to take a fallback, so the step runs Sys1's top menu
command and the control state stays as it was. Under a greedy pick that game is the masked game, with the fallback decisions
recorded on the same states.

The LLM is called through `reason_then_command`, except in `teacher` mode, where only a stored pick is scored, through `pick`.
`history` for the LLM opens with `("", first observation)`, because the LLM interface has no other place for it.

A random draw is per game and per step, from `seed`, the game's uid and the step, so a game's picks do not depend on which
other games share its batch.
"""
from __future__ import annotations

import hashlib
import math
import random
import time
from dataclasses import asdict, dataclass, field

import torch

from alf.env import AlfBatchEnv, GameView, TeacherLabel, resolve_games
from sys1.base import FALLBACK_TEXT

from .control import LLM, SYS1, ControlState, HarnessConfig, after_fallback, after_llm_round, after_pick, start_state
from .metrics import masked_probs
from .prompt import TraceEntry, render_prompt
from .reward import step_reward, terminal_reward

MODES = ("teacher", "scheduled", "model", "test")
TEACHER = "teacher"
EXPERT = "expert"

__all__ = ["run_episodes", "EpisodeLog", "StepLog", "HarnessConfig", "ControlState", "MODES", "EXPERT", "masked_probs", "FALLBACK_TEXT"]


@dataclass
class StepLog:
    step: int
    actor: str  # Whose command ran: "sys1", "llm" or "teacher".
    command: str
    obs: str  # What the command produced.
    admissible: list[str]  # At the state the command ran in, with `help` removed.
    control: ControlState  # Before the step.
    control_after: ControlState
    menu: list[str] | None = None
    options: list[str] | None = None  # The menu plus fallback, when Sys1 decided.
    probs: list[float] | None = None  # Sys1's probabilities over options, fallback included, before masking.
    masked: bool = False  # Whether this run masked fallback.
    pick: str | None = None  # Sys1's decision under this run's mask and mode. None when Sys1 did not decide.
    masked_pick: str | None = None  # Sys1's top menu command.
    unmasked_pick: str | None = None  # Sys1's top option, fallback included.
    log_prob: float | None = None  # Of `pick` under the distribution it was drawn from.
    sampled: bool = False  # `pick` was drawn and decided this step, the only steps the PPO loss may use.
    teacher_command: str | None = None  # The label. None when unknown, or when flagged random or error.
    teacher_raw: str | None = None  # What policy.act returned.
    teacher_on_menu: bool | None = None
    label_random: bool = False
    label_inadmissible: bool = False
    label_error: str | None = None
    label_initial: bool = False
    has_teacher: bool = False
    reward: float = 0.0
    prompt: str | None = None
    llm_name: str | None = None
    llm_command: str | None = None
    llm_on_menu: bool | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    seconds_sys1: float = 0.0
    seconds_llm: float = 0.0
    seconds_env: float = 0.0  # This game's share of the batched env step.


@dataclass
class EpisodeLog:
    uid: str
    task_type: str
    goal: str
    first_obs: str
    mode: str
    seed: int
    masked: bool
    steps: list[StepLog] = field(default_factory=list)
    won: bool = False
    seconds_load: float = 0.0

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def episode_return(self) -> float:
        return sum(s.reward for s in self.steps)

    def outcome(self) -> dict:
        """One game's line in a result file."""
        return {
            "uid": self.uid,
            "task_type": self.task_type,
            "won": self.won,
            "steps": self.n_steps,
            "return": self.episode_return,
            "llm_calls": sum(s.llm_command is not None for s in self.steps),
            "tokens_in": sum(s.tokens_in for s in self.steps),
            "tokens_out": sum(s.tokens_out for s in self.steps),
            "seconds_agent": sum(s.seconds_sys1 + s.seconds_llm for s in self.steps),
            "seconds_env": sum(s.seconds_env for s in self.steps),
        }

    def to_dict(self, prompts: bool = False) -> dict:
        d = asdict(self)
        if not prompts:
            for s in d["steps"]:
                s["prompt"] = None
        return d


def _top(probs: list[float]) -> int:
    return max(range(len(probs)), key=lambda i: (probs[i], -i))


def _rng(kind: str, seed: int, uid: str, step: int) -> random.Random:
    return random.Random(hashlib.sha256(f"{kind}/{seed}/{uid}/{step}".encode()).hexdigest())


def _as_list(x) -> list[float]:
    return x.detach().float().cpu().tolist() if hasattr(x, "detach") else [float(v) for v in x]


class _Game:
    """One live game inside `run_episodes`."""

    def __init__(self, view: GameView, mode: str, seed: int, masked: bool, seconds_load: float):
        self.view = view
        self.log = EpisodeLog(view.uid, view.task_type, view.goal, view.first_obs, mode, seed, masked, seconds_load=seconds_load)
        self.control = start_state()
        self.trace: list[TraceEntry] = []
        self.history: list[tuple[str, str]] = [("", view.first_obs)]
        self.clear()

    def clear(self) -> None:
        self.menu = None
        self.options = None
        self.prompt = None
        self.probs = None
        self.pick = None
        self.masked_pick = None
        self.unmasked_pick = None
        self.log_prob = None
        self.draw_pick = False
        self.teacher_runs = False
        self.llm_pick = None
        self.seconds_sys1 = 0.0
        self.seconds_llm = 0.0


def _validate(sys1, llm, menu, cfg, mode, teacher, teacher_prob) -> str | None:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if mode in ("teacher", "scheduled") and teacher is None:
        teacher = EXPERT
    if teacher not in (None, EXPERT):
        raise ValueError(f"teacher must be None or {EXPERT!r}, got {teacher!r}")
    if sys1 is None and mode != "teacher":
        raise ValueError(f"mode {mode!r} needs a Sys1")
    if sys1 is not None and menu is None:
        raise ValueError("Sys1 needs a menu")
    if llm is not None and menu is None:
        raise ValueError("the hand-over rules need a menu")
    if not 0.0 <= teacher_prob <= 1.0:
        raise ValueError(f"teacher_prob must be in [0, 1], got {teacher_prob}")
    if mode == "teacher" and teacher_prob not in (0.0, 1.0):
        raise ValueError("teacher mode runs the teacher at every step, so teacher_prob has no meaning there")
    if cfg.max_steps < 1:
        raise ValueError("max_steps must be positive")
    return teacher


def _ask_sys1(games: list[_Game], sys1, cfg, mask_fallback: bool, sampling: bool, seed: int) -> None:
    prompts = [g.prompt for g in games]
    options = [g.options for g in games]
    t0 = time.perf_counter()
    with torch.no_grad():
        out = sys1.probs(prompts, options)
    share = (time.perf_counter() - t0) / len(games)
    if len(out) != len(games):
        raise ValueError(f"Sys1 returned {len(out)} answers for {len(games)} prompts")
    for g, tensor in zip(games, out):
        probs = _as_list(tensor)
        if len(probs) != len(g.options):
            raise ValueError(f"Sys1 returned {len(probs)} probabilities for {len(g.options)} options")
        g.probs, g.seconds_sys1 = probs, share
        masked = masked_probs(probs)
        g.masked_pick = g.menu[_top(masked)]
        g.unmasked_pick = g.options[_top(probs)]
        dist = masked if mask_fallback else probs
        names = g.menu if mask_fallback else g.options
        if sampling and not g.teacher_runs:
            index = _rng("sys1", seed, g.view.uid, g.view.step).choices(range(len(dist)), weights=dist, k=1)[0]
            g.draw_pick = True
        else:
            index = _top(dist)
        g.pick = names[index]
        g.log_prob = math.log(dist[index]) if dist[index] > 0 else -math.inf


def _ask_llm(games: list[_Game], llm, mode: str) -> None:
    call = llm.pick if mode == TEACHER else llm.reason_then_command
    goals = [g.view.goal for g in games]
    histories = [list(g.history) for g in games]
    admissibles = [list(g.view.admissible) for g in games]
    t0 = time.perf_counter()
    picks = call(goals, histories, admissibles)
    share = (time.perf_counter() - t0) / len(games)
    if len(picks) != len(games):
        raise ValueError(f"the LLM returned {len(picks)} picks for {len(games)} games")
    for g, p in zip(games, picks):
        g.llm_pick = p
        g.seconds_llm = p.seconds if p.seconds else share


def _decide(g: _Game, sys1, llm, cfg, teacher_label: TeacherLabel | None) -> tuple[str, str, ControlState]:
    """Who acts this step, what runs, and the next control state."""
    control = g.control
    llm_command = g.llm_pick.command if g.llm_pick is not None else None
    if control.controller == LLM:
        next_control = after_llm_round(control, llm_command in g.menu, cfg)
        if g.teacher_runs:
            return TEACHER, teacher_label.played, next_control
        return LLM, llm_command, next_control
    if g.pick is None:  # No Sys1: the expert alone.
        return TEACHER, teacher_label.played, control
    if g.pick == FALLBACK_TEXT:
        next_control = after_fallback(control, cfg) if llm is not None else control
        if g.teacher_runs:
            return TEACHER, teacher_label.played, next_control
        if llm is not None:
            return LLM, llm_command, next_control
        return SYS1, g.masked_pick, next_control
    next_control = after_pick(control)
    if g.teacher_runs:
        return TEACHER, teacher_label.played, next_control
    return SYS1, g.pick, next_control


def run_episodes(games, sys1, llm, menu, cfg: HarnessConfig, mode: str, teacher=None, teacher_prob: float = 0.0, seed: int = 0, mask_fallback: bool = False,
                 *, batch_size: int = 8, num_workers: int = 0, tracking: str = "foreign", allow_test_games: bool = False, keep_prompts: bool = True,
                 timing: dict | None = None) -> list[EpisodeLog]:
    """Play `games`, in list order of the returned logs. The keyword-only arguments extend `lab/INTERFACES.md`, and their
    defaults keep its signature. `timing`, when given, is filled with the wall times of the batched env steps and loads."""
    teacher = _validate(sys1, llm, menu, cfg, mode, teacher, teacher_prob)
    specs = resolve_games(games, allow_test_games)
    logs: dict[int, EpisodeLog] = {}
    sampling = mode in ("scheduled", "model")
    with AlfBatchEnv(specs, max_steps=cfg.max_steps, batch_size=batch_size, num_workers=num_workers, with_teacher=teacher is not None,
                     tracking=tracking, allow_test_games=allow_test_games) as env:
        views = env.reset()
        load_share = env.load_seconds[-1] / len(views) if views else 0.0
        live = {v.index: _Game(v, mode, seed, mask_fallback, load_share) for v in views}
        while live:
            order = [live[i] for i in sorted(live)]
            for g in order:
                g.clear()
                if menu is not None:
                    g.menu = list(menu.menu(g.view.task_type, g.view.admissible, cfg.k))
                g.teacher_runs = mode == "teacher" or (mode == "scheduled" and _rng("coin", seed, g.view.uid, g.view.step).random() < teacher_prob)
            ask = [g for g in order if sys1 is not None and g.control.controller == SYS1]
            for g in ask:
                g.options = g.menu + [FALLBACK_TEXT]
                g.prompt = render_prompt(g.view.goal, g.view.first_obs, g.trace, g.control, window=cfg.trace_window)
            if ask:
                _ask_sys1(ask, sys1, cfg, mask_fallback, sampling, seed)
            need_llm = [g for g in order if llm is not None and (g.control.controller == LLM or (g.pick == FALLBACK_TEXT and not g.teacher_runs))]
            if need_llm:
                _ask_llm(need_llm, llm, mode)

            commands: dict[int, str] = {}
            pending: dict[int, StepLog] = {}
            for g in order:
                label = g.view.label
                actor, command, next_control = _decide(g, sys1, llm, cfg, label)
                commands[g.view.index] = command
                pending[g.view.index] = _step_log(g, actor, command, next_control, label, llm, mask_fallback, cfg, keep_prompts)

            stepped, started = env.step(commands)
            share = env.step_seconds[-1] / len(stepped)
            for v in stepped:
                g, s = live[v.index], pending[v.index]
                s.obs, s.seconds_env = v.obs, share
                if v.done:
                    s.reward += terminal_reward(v.won, cfg)
                g.log.steps.append(s)
                g.control = s.control_after
                g.trace.append(TraceEntry(s.actor, s.command, v.obs))
                g.history.append((s.command, v.obs))
                g.view = v
                if v.done:
                    g.log.won = v.won
                    logs[v.index] = g.log
                    del live[v.index]
            load_share = env.load_seconds[-1] / len(started) if started else 0.0
            for v in started:
                live[v.index] = _Game(v, mode, seed, mask_fallback, load_share)
        if timing is not None:
            timing.update(batched_step_seconds=list(env.step_seconds), load_seconds=list(env.load_seconds), batch_size=env.batch_size, num_workers=env.num_workers)
    return [logs[i] for i in sorted(logs)]


def _step_log(g: _Game, actor, command, next_control, label: TeacherLabel | None, llm, mask_fallback, cfg, keep_prompts) -> StepLog:
    v = g.view
    s = StepLog(step=v.step, actor=actor, command=command, obs="", admissible=list(v.admissible), control=g.control, control_after=next_control,
                menu=g.menu, options=g.options, probs=g.probs, masked=mask_fallback, pick=g.pick, masked_pick=g.masked_pick, unmasked_pick=g.unmasked_pick,
                log_prob=g.log_prob, prompt=g.prompt if keep_prompts else None, seconds_sys1=g.seconds_sys1, seconds_llm=g.seconds_llm)
    # A fallback handed to the LLM counts as Sys1's sampled pick running, because Sys1's choice decided the step.
    s.sampled = g.draw_pick and not g.teacher_runs and (g.pick != FALLBACK_TEXT or llm is not None)
    if label is not None:
        s.has_teacher = True
        s.teacher_command, s.teacher_raw = label.command, label.raw
        s.label_random, s.label_inadmissible, s.label_error, s.label_initial = label.label_random, label.label_inadmissible, label.label_error, label.label_initial
        if g.menu is not None and label.command is not None:
            s.teacher_on_menu = label.command in g.menu
    if g.llm_pick is not None:
        s.llm_name = getattr(llm, "name", None)
        s.llm_command = g.llm_pick.command
        s.llm_on_menu = g.llm_pick.command in g.menu
        s.tokens_in, s.tokens_out = g.llm_pick.tokens_in, g.llm_pick.tokens_out
    s.reward = step_reward(g.pick, s.teacher_command, g.menu, cfg)
    return s
