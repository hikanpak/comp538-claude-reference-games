"""Batched ALFWorld over an explicit list of games, with the handcoded expert available as the teacher at any state.

The game stack is the one Bonn plays (`jev_benchmarking/sequential/envs/alfworld.py`): TextWorld's generic environment,
`Limit` at 50 steps, `AlfredDemangler(shuffle=False)`, `AlfredInfos`. `AlfBatchEnv` subclasses `AlfredTWEnv` so its game
filter and wrappers stay the reference. It does not call `AlfredTWEnv.init_env`, because that registers a shuffled,
cycling pool with one process per game. Here at most `num_workers` processes hold `batch_size` games, and a free slot
loads the next game of the list.

The teacher. `ExpertTeacher` calls `policy.act` directly and works out by itself what the two shipped wrappers would have
done with the answer:

  - `HandCodedAgent.act` returns `random.choice(admissible)` when the policy repeats the last command, or after `look`
    when the policy's command is not admissible. Such a label is flagged `label_random`, and `command` is None.
  - `AlfredExpert._gather_infos` replaces a command that is not admissible by `look`. Such a label is flagged
    `label_inadmissible`, and `command` is `look`, which is what the expert really plays.
  - At reset `AlfredExpert` only calls `policy.observe` and plays `look`. `at_reset` does the same.
  - Bonn's `ExpertBackend` removes `help` from the list and plays the first remaining command when the expert says `help`.
    `played` does the same, so a teacher-run game repeats Bonn's expert command by command.

Tracking. The policy keeps its memory (where it is, what it holds, what it saw, what it should still close) inside
`policy.act`, from `last_action` and the observation. So `policy.act` runs after every step, whoever chose the command.
The installed expert has three faults that matter once somebody else chooses commands, each checked in
`lab/src/tests/test_alf_teacher.py`:

  - `update_state_tracking` looks for `put`, but ALFWorld 0.4.2's command is `move X to Y`. The policy's `inventory` keeps
    a moved object for good, and `inventory[0]` then names an object the agent no longer holds.
  - `go to` moves `curr_recep` even when the game answered "Nothing happens", as it does for an LLM's unknown command.
  - `action_backlog` keeps `close X` after the agent left X, so the next `find` pops a command that is not admissible.

`tracking` chooses what happens. `shipped` leaves the expert alone. `foreign`, the default, leaves it alone until a command
the expert did not play has run in this episode, then repairs the three faults at every later step from the game's facts,
so a teacher-only game is still Bonn's expert. `always` repairs from the first step, which changes the expert's own games.
"""
from __future__ import annotations

import contextlib
import glob
import hashlib
import json
import multiprocessing as mp
import os
import random
import re
import time
import traceback
from dataclasses import dataclass

import textworld
from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos, AlfredTWEnv
from alfworld.agents.expert import HandCodedAgentFailed, HandCodedAgentTimeout, HandCodedTWAgent
from alfworld.info import ALFWORLD_DATA
from textworld.envs.wrappers import GenericEnvironment, Limit

MAX_STEPS = 50
EXPERT_MAX_STEPS = 200  # What AlfredExpert.load gives HandCodedTWAgent. The timeout cannot fire inside 50 steps.
TEST_SPLIT = "valid_unseen"
TRACKING = ("shipped", "foreign", "always")
INTRO = re.compile(r"^-= Welcome to TextWorld, ALFRED! =-\s*", re.M)


@dataclass(frozen=True)
class GameSpec:
    path: str  # The game.tw-pddl file.
    uid: str  # "<split>/<task folder>/<trial folder>", Bonn's episode id.
    split: str
    task_type: str


def _json_root() -> str:
    return os.path.join(os.environ.get("ALFWORLD_DATA", ALFWORLD_DATA), "json_2.1.1")


def game_spec(game: str | GameSpec, allow_test_games: bool = False) -> GameSpec:
    """A GameSpec from a game file, its folder, or a uid. Decided from the path alone, so a test game is never opened."""
    if isinstance(game, GameSpec):
        spec = game
    else:
        text = str(game).rstrip("/")
        path = text if text.endswith("game.tw-pddl") else os.path.join(text, "game.tw-pddl")
        if not os.path.isabs(path):
            path = os.path.join(_json_root(), path)
        parts = os.path.normpath(path).split(os.sep)
        split, folder, trial = parts[-4], parts[-3], parts[-2]
        spec = GameSpec(path, f"{split}/{folder}/{trial}", split, folder.split("-")[0])
    if spec.split == TEST_SPLIT and not allow_test_games:
        raise ValueError(f"{spec.uid} is a test game. Test games run once, after every choice is frozen on dev, with allow_test_games=True")
    return spec


def resolve_games(games, allow_test_games: bool = False) -> list[GameSpec]:
    specs = [game_spec(g, allow_test_games) for g in games]
    if len({s.uid for s in specs}) != len(specs):
        raise ValueError("the same game appears twice in the list")
    return specs


def list_games(split: str, limit: int | None = None, allow_test_games: bool = False) -> list[GameSpec]:
    """Games of a split in Bonn's hash order, so a smaller limit is a subset of a larger one and holds mixed task types."""
    if split == TEST_SPLIT and not allow_test_games:
        raise ValueError("refusing to list the test games")
    root = os.path.join(_json_root(), split)
    specs = resolve_games(sorted(glob.glob(os.path.join(root, "*", "*", "game.tw-pddl"))), allow_test_games)
    specs.sort(key=lambda s: hashlib.sha256(f"alfworld/{s.uid}".encode()).hexdigest())
    return specs[:limit] if limit is not None else specs


def load_game_list(path: str, allow_test_games: bool = False) -> list[GameSpec]:
    """A game list file: a JSON list, or one game per line. Each entry is a uid, a folder, or a game file."""
    with open(path) as f:
        text = f.read()
    if text.lstrip().startswith("["):
        entries = json.loads(text)
    else:
        entries = [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]
    return resolve_games(entries, allow_test_games)


def expert_seed(uid: str) -> int:
    """Bonn's per-episode seed for the expert's ties."""
    return int(hashlib.sha256(f"alfworld-expert/{uid}".encode()).hexdigest()[:16], 16)


def split_intro(feedback: str) -> tuple[str, str]:
    """The task text and the first observation, cut from the reset feedback as Bonn cuts them."""
    text = INTRO.sub("", feedback).strip()
    intro, _, task = text.partition("Your task is to:")
    return task.strip(), intro.strip()


@dataclass(frozen=True)
class TeacherLabel:
    command: str | None  # The label. None when flagged label_random or label_error, so nothing can train on it by accident.
    raw: str | None  # What policy.act returned. None at reset and on an error.
    played: str  # What the shipped chain plus Bonn's help rule would run. A teacher-run step runs this.
    label_random: bool = False  # HandCodedAgent.act would have replaced raw by a random draw.
    label_inadmissible: bool = False  # raw is not admissible.
    label_error: str | None = None  # policy.act raised, because its memory no longer fits the game.
    label_initial: bool = False  # The expert's fixed first `look`, which no policy.act call produced.


def held_objects(facts: list[str]) -> list[str]:
    """Objects the agent really holds, from the game's facts, as `holds agent1 newspaper 1`."""
    return [" ".join(f.split()[2:]) for f in facts if f.startswith("holds ")]


def receptacles_at_agent(facts: list[str]) -> set[str]:
    """Receptacles at the agent's real location, from `atlocation agent1 loc 15` and `receptacleatlocation desk 1 loc 15`."""
    here = [" ".join(f.split()[2:]) for f in facts if f.startswith("atlocation ")]
    out = set()
    for f in facts:
        if f.startswith("receptacleatlocation "):
            words = f.split()
            if " ".join(words[-2:]) in here:
                out.add(" ".join(words[1:-2]))
    return out


class ExpertTeacher:
    """One game's handcoded expert. Holds its own random stream, so episodes in one process do not disturb each other."""

    def __init__(self, spec: GameSpec, tracking: str = "foreign", max_steps: int = EXPERT_MAX_STEPS):
        if tracking not in TRACKING:
            raise ValueError(f"tracking must be one of {TRACKING}")
        self.spec, self.tracking = spec, tracking
        agent = HandCodedTWAgent(max_steps=max_steps)
        agent.reset(spec.path)
        self.policy = agent.policy
        generator = random.Random()
        generator.seed(expert_seed(spec.uid))
        self._stream = generator.getstate()
        self.foreign = False  # True once a command the expert did not play has run.
        self._last_played = None
        self._last_location = None
        self._shipped_tracking = self.policy.update_state_tracking
        if tracking != "shipped":
            self.policy.update_state_tracking = self._update_state_tracking

    @contextlib.contextmanager
    def _own_stream(self):
        saved = random.getstate()
        random.setstate(self._stream)
        try:
            yield
        finally:
            self._stream = random.getstate()
            random.setstate(saved)

    def at_reset(self, state) -> TeacherLabel:
        with self._own_stream():
            self.policy.observe(state["feedback"])
        self._last_played = "look"
        return TeacherLabel("look", None, "look", label_initial=True)

    def after_step(self, state, last_command: str) -> TeacherLabel:
        """The teacher's label at the state `last_command` led to. Call it after every step, whoever chose the command."""
        admissible = state["admissible_commands"]
        self.foreign = self.foreign or last_command != self._last_played
        error = None
        with self._own_stream():
            try:
                raw = self.policy.act(state, last_command)
            except (HandCodedAgentFailed, HandCodedAgentTimeout, IndexError, KeyError) as e:
                raw, error = None, type(e).__name__
            label_random = raw is not None and (raw == last_command or (last_command == "look" and raw not in admissible))
            wrapped = random.choice(admissible) if label_random else raw
        plan = wrapped if wrapped in admissible else "look"  # AlfredExpert._gather_infos
        played = plan if plan != "help" else next(c for c in admissible if c != "help")  # Bonn's ExpertBackend
        label_inadmissible = raw is not None and raw not in admissible
        if error or label_random:
            command = None
        else:
            command = raw if not label_inadmissible else "look"
        self._last_played = played
        return TeacherLabel(command, raw, played, label_random, label_inadmissible, error)

    def _repairing(self) -> bool:
        return self.tracking == "always" or (self.tracking == "foreign" and self.foreign)

    def _update_state_tracking(self, game_state, last_action) -> None:
        """Runs inside policy.act, between the shipped tracking and the decision. Repairs the faults in the module docstring."""
        p = self.policy
        before = (p.curr_recep, p.checked_inside_curr_recep)
        self._shipped_tracking(game_state, last_action)
        if not self._repairing():
            return
        failed = "Nothing happens" in game_state["feedback"]
        if failed and last_action.startswith("go to"):
            p.curr_recep, p.checked_inside_curr_recep = before
        facts = p.get_facts(game_state)
        if not failed:
            words = last_action.split()
            if words and words[0] == "move" and "to" in words:
                self._after_move(" ".join(words[1:words.index("to")]), " ".join(words[words.index("to") + 1:]))
            elif words and words[0] == "take" and "from" in words:
                taken = " ".join(words[1:words.index("from")])
                if taken in p.object_blacklist:
                    p.object_blacklist.remove(taken)
        p.inventory = held_objects(facts)
        if not p.inventory:
            p.got_inventory_from_recep = ""
        p.action_backlog = [c for c in p.action_backlog if c in game_state["admissible_commands"]]

    def _after_move(self, obj: str, recep: str) -> None:
        """What the shipped `put` branch means to do for `move`. An object left at the wrong place is not blacklisted, so the
        expert can still find it again."""
        p = self.policy
        p.got_inventory_from_recep = ""
        p.blacklist_obj(obj)
        target, parent = p.task_params.get("object_target"), p.task_params.get("parent_target")
        if target and parent and p.object_id_to_cls(obj) == target and p.object_id_to_cls(recep) == parent:
            p.object_blacklist.append(obj)


def _make_stack(max_steps: int):
    """Bonn's stack without its `AlfredExpert` wrapper, which ExpertTeacher replaces, and without `Filter`, because the
    policy reads `feedback` and `facts` from the raw game state."""
    infos = textworld.EnvInfos(won=True, admissible_commands=True, facts=True, extras=["gamefile"])
    env = Limit(GenericEnvironment(infos), max_episode_steps=max_steps)
    return AlfredInfos(AlfredDemangler(shuffle=False)(env))


@dataclass
class GameView:
    """One game at one moment, as the harness sees it. Crosses the process boundary, so it carries no facts."""

    index: int  # Position in the game list.
    uid: str
    task_type: str
    goal: str
    first_obs: str
    step: int  # Commands run so far.
    obs: str
    admissible: list[str]  # With `help` removed, as Bonn does.
    won: bool = False
    done: bool = False
    label: TeacherLabel | None = None
    last_command: str | None = None


class _Slot:
    """One reused game stack. Reuse keeps one libdownward copy per slot, where a new stack per game grows memory."""

    def __init__(self, max_steps: int, with_teacher: bool, tracking: str):
        self.stack = _make_stack(max_steps)
        self.with_teacher, self.tracking = with_teacher, tracking
        self.teacher: ExpertTeacher | None = None
        self.view: GameView | None = None
        self.state = None  # The raw game state, for tests.

    def load(self, index: int, spec: GameSpec) -> GameView:
        self.stack.load(spec.path)
        state = self.state = self.stack.reset()
        self.teacher = ExpertTeacher(spec, self.tracking) if self.with_teacher else None
        goal, first = split_intro(state["feedback"])
        label = self.teacher.at_reset(state) if self.teacher else None
        self.view = GameView(index, spec.uid, spec.task_type, goal, first, 0, first, _no_help(state["admissible_commands"]), label=label)
        return self.view

    def step(self, command: str) -> GameView:
        state, _, done = self.stack.step(command)
        self.state = state
        v = self.view
        label = None
        if self.teacher and not done:
            label = self.teacher.after_step(state, command)
        self.view = GameView(v.index, v.uid, v.task_type, v.goal, v.first_obs, v.step + 1, state["feedback"].strip(), _no_help(state["admissible_commands"]),
                             bool(state["won"]), bool(done), label, command)
        return self.view

    def close(self) -> None:
        self.stack.close()


def _no_help(commands: list[str]) -> list[str]:
    return [c for c in commands if c != "help"]


def _serve(ops: list[tuple], slots: dict, cfg: tuple) -> list[GameView]:
    out = []
    for op in ops:
        if op[0] == "load":
            _, slot, index, spec = op
            if slot not in slots:
                slots[slot] = _Slot(*cfg)
            out.append(slots[slot].load(index, spec))
        else:
            _, slot, command = op
            out.append(slots[slot].step(command))
    return out


def _worker(conn, cfg: tuple) -> None:
    slots: dict[int, _Slot] = {}
    try:
        while True:
            ops = conn.recv()
            if ops is None:
                break
            try:
                conn.send(_serve(ops, slots, cfg))
            except Exception:  # noqa: BLE001 - the parent raises it, with the worker's traceback.
                conn.send(traceback.format_exc())
    finally:
        for s in slots.values():
            s.close()
        conn.close()


class AlfBatchEnv(AlfredTWEnv):
    """`batch_size` live games out of an explicit list. `num_workers=0` plays them in this process, which tests need to
    reach `teacher_of`. At most 4 workers, from Halil's limit on this Mac."""

    def __init__(self, games, max_steps: int = MAX_STEPS, batch_size: int = 8, num_workers: int = 0, with_teacher: bool = True,
                 tracking: str = "foreign", allow_test_games: bool = False):
        if not 0 <= num_workers <= 4:
            raise ValueError("at most 4 worker processes")
        if tracking not in TRACKING:
            raise ValueError(f"tracking must be one of {TRACKING}")
        self.specs = resolve_games(games, allow_test_games)
        self.max_steps, self.batch_size, self.num_workers = max_steps, min(batch_size, max(1, len(self.specs))), num_workers
        self._cfg = (max_steps, with_teacher, tracking)
        self.config, self.train_eval = {"dataset": {}, "env": {}, "general": {}}, "train"
        self.collect_game_files()
        self._next = 0
        self._free = list(range(self.batch_size))
        self._slot_of: dict[int, int] = {}
        self._slots: dict[int, _Slot] = {}
        self._pipes, self._procs = [], []
        self.step_seconds: list[float] = []  # Wall time of each batched env step.
        self.load_seconds: list[float] = []  # Wall time of each batch of game loads.
        if num_workers:
            ctx = mp.get_context("spawn")
            for _ in range(num_workers):
                parent, child = ctx.Pipe()
                proc = ctx.Process(target=_worker, args=(child, self._cfg), daemon=True)
                proc.start()
                child.close()
                self._pipes.append(parent)
                self._procs.append(proc)

    def collect_game_files(self, verbose: bool = False) -> None:
        """The explicit list replaces AlfredTWEnv's walk over the data folder, so no other game file is ever read."""
        self.game_files = [s.path for s in self.specs]
        self.num_games = len(self.game_files)

    def init_env(self, batch_size: int):
        raise NotImplementedError("AlfBatchEnv plays its own batch: use reset and step")

    def _run(self, ops: list[tuple]) -> list[GameView]:
        if not ops:
            return []
        if not self.num_workers:
            return _serve(ops, self._slots, self._cfg)
        per_worker: dict[int, list[tuple]] = {}
        for op in ops:
            per_worker.setdefault(op[1] % self.num_workers, []).append(op)
        for w, batch in per_worker.items():
            self._pipes[w].send(batch)
        out = []
        for w in per_worker:
            answer = self._pipes[w].recv()
            if isinstance(answer, str):
                raise RuntimeError(f"ALFWorld worker {w} failed:\n{answer}")
            out.extend(answer)
        return out

    def _fill(self) -> list[GameView]:
        ops = []
        while self._free and self._next < len(self.specs):
            slot = self._free.pop(0)
            self._slot_of[self._next] = slot
            ops.append(("load", slot, self._next, self.specs[self._next]))
            self._next += 1
        t0 = time.perf_counter()
        views = self._run(ops)
        if ops:
            self.load_seconds.append(time.perf_counter() - t0)
        return sorted(views, key=lambda v: v.index)

    def reset(self) -> list[GameView]:
        """Start the first `batch_size` games of the list."""
        return self._fill()

    def step(self, commands: dict[int, str]) -> tuple[list[GameView], list[GameView]]:
        """Run one command in each listed game, keyed by game index. Returns the games stepped, in index order, and the
        games newly started in the slots that finished games left free."""
        ops = [("step", self._slot_of[i], c) for i, c in sorted(commands.items())]
        t0 = time.perf_counter()
        stepped = sorted(self._run(ops), key=lambda v: v.index)
        self.step_seconds.append(time.perf_counter() - t0)
        for v in stepped:
            if v.done:
                self._free.append(self._slot_of.pop(v.index))
        self._free.sort()
        return stepped, self._fill()

    def teacher_of(self, index: int) -> ExpertTeacher:
        """The live teacher of a game, in-process only. For tests."""
        if self.num_workers:
            raise RuntimeError("the teacher lives in a worker process")
        return self._slots[self._slot_of[index]].teacher

    def game_state_of(self, index: int):
        """The raw game state of a live game, in-process only. For tests."""
        if self.num_workers:
            raise RuntimeError("the game lives in a worker process")
        return self._slots[self._slot_of[index]].state

    def close(self) -> None:
        for s in self._slots.values():
            s.close()
        self._slots.clear()
        for pipe in self._pipes:
            with contextlib.suppress(BrokenPipeError, OSError):
                pipe.send(None)
                pipe.close()
        for proc in self._procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
        self._pipes, self._procs = [], []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
