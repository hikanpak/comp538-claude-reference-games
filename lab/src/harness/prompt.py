"""The text Sys1 sees: task text, first observation, last trace entries, control state.

Commands run by Sys1 and by the teacher render the same, so a prompt from a teacher-run game has the form of a prompt from
Sys1's own game. Only the LLM's commands carry a mark, because the plan wants its commands and outputs visible to Sys1.
"""
from __future__ import annotations

from dataclasses import dataclass

from .control import LLM, ControlState

DEFAULT_TRACE_WINDOW = 5


@dataclass(frozen=True)
class TraceEntry:
    actor: str  # "sys1", "llm" or "teacher".
    command: str
    output: str


def _entry(item) -> TraceEntry:
    """A trace entry may be a TraceEntry, a mapping, a (command, output) pair, or an (actor, command, output) triple."""
    if isinstance(item, TraceEntry):
        return item
    if isinstance(item, dict):
        output = item.get("output", item.get("observation", item.get("obs", "")))
        return TraceEntry(item.get("actor", "sys1"), item["command"], output)
    if len(item) == 2:
        return TraceEntry("sys1", item[0], item[1])
    return TraceEntry(*item)


def _control_line(state: ControlState | dict) -> str:
    s = ControlState(**state) if isinstance(state, dict) else state
    who = "the LLM" if s.controller == LLM else "Sys1"
    return (f"Control: {who} controls, {s.fallbacks_in_row} fallbacks in a row, {s.llm_rounds_left} LLM rounds left, "
            f"{s.menu_run} LLM commands on the menu in a row.")


def render_prompt(goal: str, first_obs: str, trace, control_state, window: int = DEFAULT_TRACE_WINDOW) -> str:
    """`trace` runs oldest first and only its last `window` entries are shown. `window=0` shows none."""
    entries = [_entry(t) for t in trace]
    recent = entries[-window:] if window > 0 else []
    lines = [f"Task: {goal}", "", f"First observation: {first_obs}", ""]
    if recent:
        lines.append("Last steps, oldest first:")
        for e in recent:
            mark = "LLM> " if e.actor == LLM else "> "
            lines.append(f"{mark}{e.command}")
            lines.append(e.output)
    else:
        lines.append("Last steps: none yet.")
    lines += ["", _control_line(control_state)]
    return "\n".join(lines)
