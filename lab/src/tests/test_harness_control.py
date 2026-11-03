"""Control state and hand-over rules, against sequences computed by hand from the plan's wording."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # noqa: E401,E702

import pytest

from harness.control import LLM, SYS1, ControlState, HarnessConfig, after_fallback, after_llm_round, after_pick, start_state


def cfg(n=2, m=10, run=3):
    return HarnessConfig(k=10, fallbacks_before_handover=n, llm_rounds=m, menu_run=run, trace_window=5, w=1.0, c=0.1, beta=0.0)


def test_start():
    assert start_state() == ControlState(SYS1, 0, 0, 0)


def test_two_fallbacks_hand_over_and_a_pick_resets_the_count():
    c = cfg(n=2, m=10)
    s = after_fallback(start_state(), c)
    assert s == ControlState(SYS1, 1, 0, 0)
    s = after_pick(s)
    assert s == ControlState(SYS1, 0, 0, 0)
    s = after_fallback(s, c)
    assert s == ControlState(SYS1, 1, 0, 0)
    s = after_fallback(s, c)
    assert s == ControlState(LLM, 0, 10, 0)


def test_one_fallback_hands_over_when_n_is_one():
    assert after_fallback(start_state(), cfg(n=1, m=4)) == ControlState(LLM, 0, 4, 0)


def test_rounds_run_out_and_control_returns():
    c = cfg(n=2, m=4, run=3)
    s = ControlState(LLM, 0, 4, 0)
    # On, off, on, off: the run never reaches 3, so all four rounds are used.
    s = after_llm_round(s, True, c)
    assert s == ControlState(LLM, 0, 3, 1)
    s = after_llm_round(s, False, c)
    assert s == ControlState(LLM, 0, 2, 0)
    s = after_llm_round(s, True, c)
    assert s == ControlState(LLM, 0, 1, 1)
    s = after_llm_round(s, False, c)
    assert s == start_state()


def test_run_of_menu_commands_returns_control_early():
    c = cfg(n=2, m=10, run=3)
    s = ControlState(LLM, 0, 10, 0)
    s = after_llm_round(s, True, c)
    assert s == ControlState(LLM, 0, 9, 1)
    s = after_llm_round(s, True, c)
    assert s == ControlState(LLM, 0, 8, 2)
    s = after_llm_round(s, True, c)
    assert s == start_state()


def test_off_menu_command_breaks_the_run():
    c = cfg(n=2, m=10, run=3)
    s = ControlState(LLM, 0, 10, 0)
    for on_menu in (True, True, False, True, True):
        s = after_llm_round(s, on_menu, c)
    assert s == ControlState(LLM, 0, 5, 2)
    s = after_llm_round(s, True, c)
    assert s == start_state()


def test_run_equal_to_the_rounds_left_ends_on_the_same_step():
    c = cfg(n=2, m=3, run=3)
    s = ControlState(LLM, 0, 3, 0)
    s = after_llm_round(after_llm_round(s, True, c), True, c)
    assert s == ControlState(LLM, 0, 1, 2)
    assert after_llm_round(s, True, c) == start_state()


def test_zero_switches_rules_off():
    # n=0: a fallback never hands over, and the count keeps climbing.
    s = start_state()
    for expected in (1, 2, 3):
        s = after_fallback(s, cfg(n=0))
        assert s == ControlState(SYS1, expected, 0, 0)
    # run=0: no early return, however long the run.
    c = cfg(n=2, m=4, run=0)
    s = ControlState(LLM, 0, 4, 0)
    for left, run in ((3, 1), (2, 2), (1, 3)):
        s = after_llm_round(s, True, c)
        assert s == ControlState(LLM, 0, left, run)
    assert after_llm_round(s, True, c) == start_state()
    # m=0: a handover with no rounds leaves Sys1 in control with the count cleared.
    assert after_fallback(ControlState(SYS1, 1, 0, 0), cfg(n=2, m=0)) == start_state()


def test_wrong_controller_is_refused():
    with pytest.raises(ValueError):
        after_pick(ControlState(LLM, 0, 3, 0))
    with pytest.raises(ValueError):
        after_fallback(ControlState(LLM, 0, 3, 0), cfg())
    with pytest.raises(ValueError):
        after_llm_round(start_state(), True, cfg())
