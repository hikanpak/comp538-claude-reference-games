"""The reward of `plan/results_plan.tex`: `match` for a match, a fallback counts as a match when the teacher's command is off
the menu, minus w for a wrong pick, minus c for a needless fallback, and beta at a successful end.

A step with no usable teacher label pays 0, whatever Sys1 did, so a `label_random` label cannot pay or cost anything. A step
Sys1 did not decide pays 0 as well, because an LLM round is judged through the return.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from sys1.base import FALLBACK_TEXT


@dataclass(frozen=True)
class RewardSetting:
    name: str
    match: float
    w: float
    c: float
    beta: float


R1 = RewardSetting("R1", 1.0, 0.0, 0.0, 0.0)  # Agreement only.
R2 = RewardSetting("R2", 1.0, 1.0, 0.1, 0.0)  # The default.
R3 = RewardSetting("R3", 1.0, 1.0, 0.1, 5.0)  # R2 plus the end.
R4 = RewardSetting("R4", 0.0, 0.0, 0.0, 1.0)  # Only the checker's verdict, so no pay for a match.
REWARDS = {r.name: r for r in (R1, R2, R3, R4)}


def with_reward(cfg, name: str):
    """A copy of a HarnessConfig with a named reward."""
    r = REWARDS[name]
    return replace(cfg, match=r.match, w=r.w, c=r.c, beta=r.beta)


def step_reward(pick: str | None, teacher_command: str | None, menu: list[str] | None, cfg) -> float:
    if pick is None or teacher_command is None:
        return 0.0
    if pick == FALLBACK_TEXT:
        return cfg.match if teacher_command not in menu else -cfg.c
    return cfg.match if pick == teacher_command else -cfg.w


def terminal_reward(won: bool, cfg) -> float:
    return cfg.beta if won else 0.0
