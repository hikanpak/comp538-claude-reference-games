"""Stand-ins for tests only: a small menu, and scripted Sys1 and LLM behind their interfaces. No test here."""
from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass

SRC = pathlib.Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch  # noqa: E402

from sys1.base import FALLBACK_TEXT  # noqa: E402


class StandInMenu:
    """`MenuModel.menu`'s shape: a deterministic order of the admissible commands, cut to k. Task-solving kinds first."""

    ORDER = ("take", "move", "heat", "cool", "clean", "use", "open", "go to", "close", "examine", "look", "inventory")

    def menu(self, task_type, admissible, k):
        def rank(command):
            for i, kind in enumerate(self.ORDER):
                if command.startswith(kind):
                    return i, command
            return len(self.ORDER), command

        ordered = sorted(admissible, key=rank)
        return ordered if k is None else ordered[:k]


class ScriptedSys1:
    """Sys1's `probs`, deciding by a function of the prompt, the options and the call count. `choose` returns the option index to favour."""

    temperature = 1.0

    def __init__(self, choose):
        self.choose = choose
        self.calls = 0

    def probs(self, prompts, options):
        out = []
        for prompt, opts in zip(prompts, options):
            assert opts[-1] == FALLBACK_TEXT
            favoured = self.choose(prompt, opts, self.calls)
            p = torch.full((len(opts),), 0.1 / max(1, len(opts) - 1))
            p[favoured] = 0.9
            out.append(p / p.sum())
            self.calls += 1
        return out


@dataclass
class Pick:
    command: str
    probs: dict | None = None
    tokens_in: int = 7
    tokens_out: int = 3
    seconds: float = 0.0


class ScriptedLLM:
    """The LLM interface. `choose(goal, history, admissible)` returns a command. Records which call ran."""

    name = "stand-in-gemma"
    precision = "none"

    def __init__(self, choose):
        self.choose = choose
        self.pick_calls = 0
        self.reason_calls = 0
        self.histories = []

    def _answer(self, goals, histories, admissibles):
        self.histories.extend(histories)
        return [Pick(self.choose(g, h, a)) for g, h, a in zip(goals, histories, admissibles)]

    def pick(self, goals, histories, admissibles):
        self.pick_calls += len(goals)
        return self._answer(goals, histories, admissibles)

    def reason_then_command(self, goals, histories, admissibles):
        self.reason_calls += len(goals)
        return self._answer(goals, histories, admissibles)
