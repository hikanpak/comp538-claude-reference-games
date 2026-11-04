"""Play one rollout of a game and record every state: its observation, its admissible commands, the teacher's label, and the command that ran.

Three families, all labelled by ALFWorld's handcoded expert with `tracking="always"` (`lib/issues/2026-11-04-claude-installed-expert-faults.md`):

  - `expert-path`: the expert plays the whole game under one seed. Its ties use random draws, so a seed is a separate stream.
  - `recovery`: the expert plays up to a random step, 1 to 5 uniform random admissible commands run, and the expert continues.
  - `loop-break`: the expert plays up to a random step, one command runs 2 to 4 times, and the expert continues. The command is one that changes
    nothing (`look`, `inventory`, `examine ...`), or one already used in this game. An already-used command that cannot run twice in a row, because the
    game no longer admits it, alternates with the command that undoes it (`go to A` and back, `open X` and `close X`, `take` and `move`), so the
    repeat is what `lab/src/harness/metrics.py` counts, a (command, observation) pair equal to one of the previous two.

The teacher's `policy.act` runs after every command, whoever chose it, because its memory is built there (`lab/src/alf/env.py`). A label flagged
random, inadmissible or error has `label` None and never reaches a training row.

Seed 0 of the expert is the stream `lab/src/alf/env.py` gives every game, so an `expert-path` run under seed 0 repeats a harness run under
`tracking="always"` command by command, and a test holds that.
"""
from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass

from alf.env import ExpertTeacher, GameSpec, TeacherLabel, _no_help, split_intro

TRACKING = "always"
FAMILIES = ("expert-path", "recovery", "loop-break")
NO_EFFECT = ("look", "inventory", "examine")
_TAKE = re.compile(r"^take (.+) from (.+)$")
_MOVE = re.compile(r"^move (.+) to (.+)$")


class SeededTeacher(ExpertTeacher):
    """The expert with its random stream chosen by `seed`. Seed 0 keeps the stream `ExpertTeacher` already has."""

    def __init__(self, spec: GameSpec, seed: int, tracking: str = TRACKING):
        super().__init__(spec, tracking)
        if seed != 0:
            generator = random.Random()
            generator.seed(int(hashlib.sha256(f"alfworld-expert/{spec.uid}/{seed}".encode()).hexdigest()[:16], 16))
            self._stream = generator.getstate()


def label_fields(label: TeacherLabel, admissible: list[str]) -> dict:
    """What a stored state keeps of the teacher's answer. `label_inadmissible` is also set when the answer is `help`, which stored lists do not hold."""
    fields: dict = {"label": label.command, "label_source": "policy"}
    if label.label_error:
        fields.update(label=None, label_source="error", label_error=label.label_error)
    elif label.label_random:
        fields.update(label=None, label_source="random-draw", label_random=True)
    elif label.label_inadmissible or (label.command is not None and label.command not in admissible):
        fields.update(label=None, label_source="inadmissible", label_inadmissible=True)
    elif label.label_initial:
        fields.update(label_source="initial-look", label_initial=True)
    return fields


class RandomCommands:
    """`count` uniform random admissible commands from step `start`."""

    kind = "random"

    def __init__(self, rng: random.Random, start: int, count: int):
        self.rng, self.start, self.count = rng, start, count
        self.commands: list[str] = []
        self.end = start

    def next(self, step: int, admissible: list[str], history: list[str]) -> str | None:
        if not self.start <= step < self.start + self.count:
            return None
        command = self.rng.choice(admissible)
        self.commands.append(command)
        self.end = step + 1
        return command

    @property
    def formed(self) -> bool:
        return bool(self.commands)


def undo_of(command: str, place_before: str | None) -> str | None:
    """The command that makes `command` admissible again."""
    if command.startswith("go to "):
        return f"go to {place_before}" if place_before else None
    if command.startswith("open "):
        return "close " + command[len("open "):]
    if command.startswith("close "):
        return "open " + command[len("close "):]
    taken = _TAKE.match(command)
    if taken:
        return f"move {taken.group(1)} to {taken.group(2)}"
    moved = _MOVE.match(command)
    if moved:
        return f"take {moved.group(1)} from {moved.group(2)}"
    return None


class RepeatedCommand:
    """One command runs `repeats` times from step `start`, with its undo between runs when the game needs it."""

    def __init__(self, rng: random.Random, start: int, repeats: int):
        self.rng, self.start, self.repeats = rng, start, repeats
        self.kind: str | None = None
        self.command: str | None = None
        self.undo: str | None = None
        self.runs = 0
        self.finished = False
        self.commands: list[str] = []
        self.end = start

    def _choose(self, admissible: list[str], history: list[str]) -> None:
        used = [c for c in dict.fromkeys(history) if c in admissible and c.split()[0] not in NO_EFFECT]
        if used and self.rng.random() < 0.5:
            self.kind, self.command = "already-used", self.rng.choice(used)
            place = next((c[len("go to "):] for c in reversed(history) if c.startswith("go to ")), None)
            self.undo = undo_of(self.command, place)
        else:
            quiet = [c for c in admissible if c.split()[0] in NO_EFFECT]
            self.kind, self.command = "no-effect", self.rng.choice(quiet)

    def next(self, step: int, admissible: list[str], history: list[str]) -> str | None:
        if step < self.start or self.finished:
            return None
        if self.command is None:
            self._choose(admissible, history)
        if self.runs >= self.repeats:
            self.finished = True
            return None
        if self.command in admissible:
            chosen = self.command
            self.runs += 1
        elif self.undo in admissible:
            chosen = self.undo
        else:
            self.finished = True
            return None
        self.commands.append(chosen)
        self.end = step + 1
        return chosen

    @property
    def formed(self) -> bool:
        return self.runs >= 2


@dataclass
class Game:
    """A loaded game stack and its spec."""

    spec: GameSpec
    split: str
    stack: object


def play(game: Game, family: str, expert_seed: int, rollout: int = 0, plan=None) -> dict:
    """One rollout from the game's reset to its end. `plan.next` may choose the command of a step, and otherwise the expert's runs."""
    state = game.stack.reset()
    teacher = SeededTeacher(game.spec, expert_seed)
    goal, first_obs = split_intro(state["feedback"])
    label = teacher.at_reset(state)
    obs, admissible = first_obs, _no_help(state["admissible_commands"])
    steps, history = [], []
    done = won = False
    while not done:
        command = plan.next(len(steps), admissible, history) if plan else None
        command = command or label.played
        state, _, done = game.stack.step(command)
        steps.append({"obs": obs, "admissible": admissible, "command": command, **label_fields(label, admissible)})
        history.append(command)
        obs, admissible = state["feedback"].strip(), _no_help(state["admissible_commands"])
        won = bool(state["won"])
        if not done:
            label = teacher.after_step(state, command)
    run = {"run_id": f"{game.spec.uid}#{family}#{rollout}", "split": game.split, "game_id": game.spec.uid, "task_type": game.spec.task_type, "family": family,
           "expert_seed": expert_seed, "rollout": rollout, "tracking": TRACKING, "goal": goal, "first_obs": first_obs, "won": won, "final_obs": obs, "steps": steps, "perturb": None}
    if plan is not None:
        run["perturb"] = {"kind": plan.kind, "start": plan.start, "end": plan.end, "commands": plan.commands, "formed": plan.formed}
    return run


def rollout_rng(kind: str, seed: int, uid: str, index: int) -> random.Random:
    return random.Random(hashlib.sha256(f"{kind}/{seed}/{uid}/{index}".encode()).hexdigest())


def play_game(game: Game, seed: int, expert_seeds: list[int], recovery_rollouts: int, loop_rollouts: int) -> tuple[list[dict], dict]:
    """Every rollout of one game. A game no expert seed wins gets no recovery or loop-break rollout, because no clean path gives their random step."""
    runs = [play(game, "expert-path", s, rollout=s) for s in expert_seeds]
    won = [r for r in runs if r["won"]]
    notes = {"expert_seeds_won": len(won), "skipped_rollouts": 0}
    if not won:
        notes["skipped_rollouts"] = recovery_rollouts + loop_rollouts
        return runs, notes
    for i in range(recovery_rollouts):
        rng = rollout_rng("recovery", seed, game.spec.uid, i)
        base = won[i % len(won)]
        plan = RandomCommands(rng, rng.randrange(len(base["steps"])), rng.randint(1, 5))
        runs.append(play(game, "recovery", base["expert_seed"], rollout=i, plan=plan))
    for i in range(loop_rollouts):
        rng = rollout_rng("loop-break", seed, game.spec.uid, i)
        base = won[i % len(won)]
        plan = RepeatedCommand(rng, rng.randrange(len(base["steps"])), rng.randint(2, 4))
        runs.append(play(game, "loop-break", base["expert_seed"], rollout=i, plan=plan))
    return runs, notes
