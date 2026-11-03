"""R1 to R4 as `plan/results_plan.tex` defines them, and the plan's one-line expectation."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # noqa: E401,E702

import pytest

from harness.control import DEFAULT_CONFIG
from harness.reward import REWARDS, step_reward, terminal_reward, with_reward
from sys1.base import FALLBACK_TEXT as F

MENU = ["go to desk 1", "take pen 1 from desk 1"]


@pytest.mark.parametrize("name, match, wrong, needless, off_menu_fallback, end", [
    ("R1", 1.0, 0.0, 0.0, 1.0, 0.0),
    ("R2", 1.0, -1.0, -0.1, 1.0, 0.0),
    ("R3", 1.0, -1.0, -0.1, 1.0, 5.0),
    ("R4", 0.0, 0.0, 0.0, 0.0, 1.0),
])
def test_each_named_reward(name, match, wrong, needless, off_menu_fallback, end):
    cfg = with_reward(DEFAULT_CONFIG, name)
    assert step_reward(MENU[0], MENU[0], MENU, cfg) == match
    assert step_reward(MENU[1], MENU[0], MENU, cfg) == wrong
    assert step_reward(F, MENU[0], MENU, cfg) == needless
    assert step_reward(F, "go to bed 1", MENU, cfg) == off_menu_fallback
    assert step_reward(MENU[0], "go to bed 1", MENU, cfg) == wrong  # A pick when the teacher is off the menu is a wrong pick.
    assert terminal_reward(True, cfg) == end
    assert terminal_reward(False, cfg) == 0.0


def test_no_label_or_no_decision_pays_nothing():
    cfg = with_reward(DEFAULT_CONFIG, "R2")
    assert step_reward(MENU[0], None, MENU, cfg) == 0.0
    assert step_reward(F, None, MENU, cfg) == 0.0
    assert step_reward(None, MENU[0], MENU, cfg) == 0.0


def test_settings_named_in_the_plan():
    assert (REWARDS["R1"].w, REWARDS["R1"].c, REWARDS["R1"].beta) == (0.0, 0.0, 0.0)
    assert (REWARDS["R2"].w, REWARDS["R2"].c, REWARDS["R2"].beta) == (1.0, 0.1, 0.0)
    assert (REWARDS["R3"].w, REWARDS["R3"].c, REWARDS["R3"].beta) == (1.0, 0.1, 5.0)
    assert (REWARDS["R4"].match, REWARDS["R4"].beta) == (0.0, 1.0)


def test_expected_pay_matches_the_plans_inequality():
    # Picking pays p(1+w)-w in expectation, and falling back pays pf(1+c)-c, for true probabilities p and pf.
    cfg = with_reward(DEFAULT_CONFIG, "R2")
    p, pf = 0.7, 0.2
    picking = p * step_reward(MENU[0], MENU[0], MENU, cfg) + (1 - p) * step_reward(MENU[0], MENU[1], MENU, cfg)
    falling_back = pf * step_reward(F, "off the menu", MENU, cfg) + (1 - pf) * step_reward(F, MENU[0], MENU, cfg)
    assert picking == pytest.approx(p * (1 + cfg.w) - cfg.w)
    assert falling_back == pytest.approx(pf * (1 + cfg.c) - cfg.c)
