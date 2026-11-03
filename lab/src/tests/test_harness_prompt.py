"""`render_prompt` shows the task, the first observation, the last trace entries with the LLM's marked, and the control state."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # noqa: E401,E702

from harness.control import LLM, SYS1, ControlState
from harness.prompt import TraceEntry, render_prompt

GOAL, FIRST = "put a pen on desk.", "You are in the middle of a room."


def test_exact_text():
    trace = [TraceEntry("sys1", "go to desk 1", "You arrive at desk 1."), TraceEntry("llm", "examine desk 1", "On the desk 1, you see a pen 1.")]
    text = render_prompt(GOAL, FIRST, trace, ControlState(SYS1, 1, 0, 0))
    assert text == (
        "Task: put a pen on desk.\n\nFirst observation: You are in the middle of a room.\n\n"
        "Last steps, oldest first:\n> go to desk 1\nYou arrive at desk 1.\nLLM> examine desk 1\nOn the desk 1, you see a pen 1.\n\n"
        "Control: Sys1 controls, 1 fallbacks in a row, 0 LLM rounds left, 0 LLM commands on the menu in a row."
    )


def test_empty_trace_and_llm_control():
    text = render_prompt(GOAL, FIRST, [], ControlState(LLM, 0, 7, 2))
    assert "Last steps: none yet." in text
    assert "Control: the LLM controls, 0 fallbacks in a row, 7 LLM rounds left, 2 LLM commands on the menu in a row." in text


def test_window_keeps_the_last_entries():
    trace = [TraceEntry("sys1", f"c{i}", f"o{i}") for i in range(8)]
    text = render_prompt(GOAL, FIRST, trace, ControlState())
    assert "> c2\n" not in text and "> c3\no3" in text and "> c7\no7" in text
    assert "> c6" in render_prompt(GOAL, FIRST, trace, ControlState(), window=2) and "> c5" not in render_prompt(GOAL, FIRST, trace, ControlState(), window=2)
    assert "none yet" in render_prompt(GOAL, FIRST, trace, ControlState(), window=0)


def test_teacher_and_sys1_commands_render_the_same():
    # A teacher-forced training prompt must have the form of a prompt from Sys1's own game.
    a = render_prompt(GOAL, FIRST, [TraceEntry("teacher", "look", "x")], ControlState())
    b = render_prompt(GOAL, FIRST, [TraceEntry("sys1", "look", "x")], ControlState())
    assert a == b


def test_trace_entry_shapes_give_the_same_text():
    want = render_prompt(GOAL, FIRST, [TraceEntry("sys1", "look", "x")], ControlState())
    assert render_prompt(GOAL, FIRST, [("look", "x")], ControlState()) == want
    assert render_prompt(GOAL, FIRST, [("sys1", "look", "x")], ControlState()) == want
    assert render_prompt(GOAL, FIRST, [{"command": "look", "observation": "x"}], {"controller": "sys1", "fallbacks_in_row": 0, "llm_rounds_left": 0, "menu_run": 0}) == want
