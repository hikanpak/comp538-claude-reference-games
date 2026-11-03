"""Control state and hand-over rules, as pure functions.

Sys1 controls until it has fallen back `fallbacks_before_handover` times in a row. Each of those fallbacks gives one step to
the LLM, and control returns after it. The fallback that reaches the count also hands control over, so the LLM then keeps
control for the next `llm_rounds` steps. Control returns to Sys1 when those rounds run out, or as soon as `menu_run` LLM
commands in a row were all on the menu. A menu command picked by Sys1 resets the count of fallbacks.

A value of 0 switches a rule off: `fallbacks_before_handover=0` means a fallback never hands control over, which is what the
plan's Qwen 27B question wants, and `menu_run=0` means no early return.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

SYS1 = "sys1"
LLM = "llm"


@dataclass(frozen=True)
class ControlState:
    controller: str = SYS1
    fallbacks_in_row: int = 0
    llm_rounds_left: int = 0
    menu_run: int = 0


@dataclass(frozen=True)
class HarnessConfig:
    k: int | None
    fallbacks_before_handover: int
    llm_rounds: int
    menu_run: int
    trace_window: int
    w: float
    c: float
    beta: float
    max_steps: int = 50
    match: float = 1.0  # Pay for a match. Added to the interface because R4 pays only beta, so it needs 0 here.


# The plan's start values: 2 fallbacks, 10 rounds, a run of 3, the last 5 trace entries, reward R2, and a menu of 10.
DEFAULT_CONFIG = HarnessConfig(k=10, fallbacks_before_handover=2, llm_rounds=10, menu_run=3, trace_window=5, w=1.0, c=0.1, beta=0.0)


def start_state() -> ControlState:
    return ControlState()


def after_pick(state: ControlState) -> ControlState:
    """Sys1 picked a menu command."""
    if state.controller != SYS1:
        raise ValueError(f"Sys1 cannot pick while {state.controller} controls")
    return replace(state, fallbacks_in_row=0)


def after_fallback(state: ControlState, cfg: HarnessConfig) -> ControlState:
    """Sys1 fell back, and the LLM took this one step."""
    if state.controller != SYS1:
        raise ValueError(f"Sys1 cannot fall back while {state.controller} controls")
    count = state.fallbacks_in_row + 1
    if cfg.fallbacks_before_handover > 0 and count >= cfg.fallbacks_before_handover:
        if cfg.llm_rounds > 0:
            return ControlState(LLM, 0, cfg.llm_rounds, 0)
        return start_state()
    return replace(state, fallbacks_in_row=count)


def after_llm_round(state: ControlState, on_menu: bool, cfg: HarnessConfig) -> ControlState:
    """The LLM, in control, ran one command. `on_menu` says whether that command was on this step's menu."""
    if state.controller != LLM:
        raise ValueError("the LLM has no round while Sys1 controls")
    run = state.menu_run + 1 if on_menu else 0
    left = state.llm_rounds_left - 1
    if left <= 0 or (cfg.menu_run > 0 and run >= cfg.menu_run):
        return start_state()
    return ControlState(LLM, 0, left, run)
