"""Metrics on a small constructed log. The numbers below were computed by hand, and the comments show the working."""
import math
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # noqa: E401,E702

import pytest

from harness.control import ControlState
from harness.episode import EpisodeLog, StepLog
from harness.metrics import auroc, coverage_curve, decisions, ece, exposure_gap, fallback_rates, masked_probs, repeat_metrics, step_match, summarize
from sys1.base import FALLBACK_TEXT as F

C = ControlState()
MENU = ["a", "b"]
OPTIONS = MENU + [F]


def step(i, pa, pb, pf, teacher, *, label_random=False, label_inadmissible=False, actor="sys1", command=None, obs=None, masked=True, llm=None):
    probs = [pa, pb, pf]
    menu_probs = masked_probs(probs)
    masked_pick = MENU[0] if menu_probs[0] >= menu_probs[1] else MENU[1]
    unmasked_pick = OPTIONS[max(range(3), key=lambda j: probs[j])]
    return StepLog(step=i, actor=actor, command=command or masked_pick, obs=obs or f"o{i}", admissible=MENU, control=C, control_after=C, menu=MENU, options=OPTIONS, probs=probs,
                   masked=masked, pick=masked_pick if masked else unmasked_pick, masked_pick=masked_pick, unmasked_pick=unmasked_pick, teacher_command=None if label_random else teacher,
                   teacher_raw=teacher, teacher_on_menu=None if label_random else teacher in MENU, label_random=label_random, label_inadmissible=label_inadmissible, has_teacher=True,
                   llm_name=llm, llm_command="a" if llm else None, tokens_in=10 if llm else 0, tokens_out=4 if llm else 0)


def log(steps, won=False):
    return EpisodeLog("u", "t", "g", "f", "test", 0, True, steps=steps, won=won)


def constructed():
    return [log([
        step(0, 0.62, 0.28, 0.10, "a"),                    # On menu, masked and unmasked pick a, right.
        step(1, 0.20, 0.70, 0.10, "a"),                    # On menu, picks b, wrong.
        step(2, 0.50, 0.10, 0.40, "c"),                    # Off menu, unmasked pick a is wrong.
        step(3, 0.10, 0.22, 0.68, "z"),                    # Off menu, unmasked pick is fallback, which is right.
        step(4, 0.90, 0.05, 0.05, "a", label_random=True),  # Would match, but the label is flagged and stays out.
    ], won=True)]


def test_step_match_leaves_out_label_random():
    assert step_match(constructed(), "unmasked") == pytest.approx(1 / 4)
    assert step_match(constructed(), "masked") == pytest.approx(1 / 4)
    assert summarize(constructed())["label_random_share"] == pytest.approx(1 / 5)


def test_ece_unmasked():
    # Confidence .62 right in bin 9, .70 wrong and .68 right in bin 10, .50 wrong in bin 7 of 15.
    # (|.62-1| + 2*|.69-.5| + |.50-0|) / 4 = (.38 + .38 + .50) / 4 = .315
    ds = decisions(constructed(), "unmasked")
    assert ece([d.confidence for d in ds], [d.right for d in ds]) == pytest.approx(0.315)


def test_ece_masked():
    # Masked confidences 31/45 right, 7/9 wrong, 5/6 wrong, 11/16 wrong. 31/45 and 11/16 share bin 10.
    # (2*|(31/45+11/16)/2 - 1/2| + 7/9 + 5/6) / 4 = (0.376389 + 0.777778 + 0.833333) / 4 = 0.496875
    ds = decisions(constructed(), "masked")
    assert ece([d.confidence for d in ds], [d.right for d in ds]) == pytest.approx(0.496875)


def test_brier_unmasked_and_masked():
    # Unmasked, target is the teacher's option or else fallback: .2328, 1.14, .62, .1608, mean .5384.
    assert summarize(constructed(), "unmasked")["brier"] == pytest.approx(0.5384)
    # Masked, an off-menu teacher adds 1: 0.193580, 1.209877, 1.722222, 1.570313, mean 1.173998.
    assert summarize(constructed(), "masked")["brier"] == pytest.approx(1.173998, abs=1e-5)


def test_nll_unmasked_and_masked():
    # Unmasked: -ln .62, -ln .20, -ln .40, -ln .68.
    expected = (-math.log(0.62) - math.log(0.20) - math.log(0.40) - math.log(0.68)) / 4
    assert expected == pytest.approx(0.847357, abs=1e-5)
    s = summarize(constructed(), "unmasked")
    assert s["nll"] == pytest.approx(expected) and s["nll_excluded"] == 0
    # Masked: only the two on-menu steps have a target: -ln(31/45), -ln(2/9).
    s = summarize(constructed(), "masked")
    assert s["nll"] == pytest.approx(0.938376, abs=1e-5) and s["nll_excluded"] == 2


def test_auroc():
    # Unmasked: right .62, .68 against wrong .70, .50. Two of four pairs are ordered right.
    assert summarize(constructed(), "unmasked")["auroc"] == pytest.approx(0.5)
    # Masked: right 31/45 against wrong 7/9, 5/6, 11/16. One of three.
    assert summarize(constructed(), "masked")["auroc"] == pytest.approx(1 / 3)
    assert auroc([0.5, 0.5], [True, False]) == pytest.approx(0.5)  # A tie counts one half.
    assert auroc([0.9, 0.1], [True, False]) == 1.0
    assert auroc([0.9, 0.1], [False, True]) == 0.0
    assert auroc([0.9, 0.1], [True, True]) is None  # One class only: not measured.


def test_repeat_rate():
    # Game 1: (x,1) (y,2) (x,1) (x,1) (z,3): step 2 equals step 0, step 3 equals step 2. Game 2: no repeat. 2 of 8.
    a = log([step(i, .5, .3, .2, "a", command=c, obs=o) for i, (c, o) in enumerate([("x", "1"), ("y", "2"), ("x", "1"), ("x", "1"), ("z", "3")])])
    b = log([step(i, .5, .3, .2, "a", command=c, obs=o) for i, (c, o) in enumerate([("u", "9"), ("v", "8"), ("w", "7")])])
    r = repeat_metrics([a, b])
    assert r["repeat_rate"] == pytest.approx(0.25)
    assert r["episodes_with_loop"] == 0.0


def test_repeat_needs_the_same_observation_and_a_loop_is_three_in_a_row():
    # The same command with a new observation is no repeat.
    assert repeat_metrics([log([step(0, .5, .3, .2, "a", command="x", obs="1"), step(1, .5, .3, .2, "a", command="x", obs="2")])])["repeat_rate"] == 0.0
    # Five equal steps: steps 1 to 4 repeat, a run of 4, so the game has a loop.
    r = repeat_metrics([log([step(i, .5, .3, .2, "a", command="x", obs="1") for i in range(5)])])
    assert r["repeat_rate"] == pytest.approx(4 / 5) and r["episodes_with_loop"] == 1.0


def test_exposure_gap():
    # Teacher's states: 3 of 4 match. Own states: 1 of 4. The gap is .75 - .25.
    on_teacher = [log([step(0, .8, .1, .1, "a"), step(1, .8, .1, .1, "a"), step(2, .1, .8, .1, "b"), step(3, .8, .1, .1, "b")])]
    own = [log([step(0, .8, .1, .1, "a"), step(1, .8, .1, .1, "b"), step(2, .1, .8, .1, "a"), step(3, .8, .1, .1, "b")])]
    assert exposure_gap(on_teacher, own) == pytest.approx(0.5)
    assert exposure_gap([], own) is None


def test_coverage_curve():
    # Unmasked confidences .62 (matches), .70, .50, .68. The fallback at .68 is right but is no step match.
    rows = coverage_curve(decisions(constructed(), "unmasked"), [0.0, 0.6, 0.65, 0.75])
    assert [r["kept_share"] for r in rows] == pytest.approx([1.0, 0.75, 0.5, 0.0])
    assert rows[0]["step_match"] == pytest.approx(1 / 4)
    assert rows[1]["step_match"] == pytest.approx(1 / 3)
    assert rows[2]["step_match"] == 0.0
    assert rows[3]["step_match"] is None


def test_fallback_rates():
    r = fallback_rates(constructed())
    assert r["n_off_menu"] == 2 and r["n_on_menu"] == 2
    assert r["fallback_when_off_menu"] == pytest.approx(1 / 2)
    assert r["fallback_when_on_menu"] == 0.0


def test_fallback_rates_with_nothing_off_menu_are_not_filled_in():
    r = fallback_rates([log([step(0, .8, .1, .1, "a")])])
    assert r["fallback_when_off_menu"] is None and r["n_off_menu"] == 0


def test_summary_of_counts_and_shares():
    a = log([step(0, .8, .1, .1, "a"), step(1, .8, .1, .1, "a", actor="llm", llm="Gemma-4"), step(2, .8, .1, .1, "a", actor="teacher")], won=True)
    b = log([step(0, .8, .1, .1, "a", llm="Qwen3.8")])
    s = summarize([a, b])
    assert s["n_games"] == 2 and s["success"] == 0.5 and s["mean_steps"] == 2.0
    assert s["sys1_share_of_steps"] == pytest.approx(2 / 4)
    assert s["llm_calls"] == 2 and s["gemma_calls"] == 1 and s["qwen_calls"] == 1
    assert s["tokens_in"] == 20 and s["tokens_out"] == 8


def test_step_match_without_inadmissible_labels():
    # The `look` stand-in for an inadmissible command stays in `step_match` and leaves `step_match_excl_inadmissible`.
    logs = [log([step(0, .8, .1, .1, "a"), step(1, .8, .1, .1, "b", label_inadmissible=True)])]
    s = summarize(logs)
    assert s["step_match"] == pytest.approx(1 / 2)
    assert s["step_match_excl_inadmissible"] == 1.0
    assert s["label_inadmissible_share"] == pytest.approx(1 / 2)
