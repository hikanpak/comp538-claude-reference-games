"""The bootstrap of the plan on known cases: pairing, the mean over seeds, and a distribution small enough to enumerate."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # noqa: E401,E702

import pytest

from harness.stats import game_means, paired_bootstrap


def test_constant_difference_has_no_width_only_when_paired():
    # Games differ a lot, and row b is row a plus .25 on every game. Pairing leaves the difference exact.
    a = {f"g{i}": v for i, v in enumerate([0.0, 1.0, 0.2, 0.8, 0.4, 0.6])}
    b = {g: v + 0.25 for g, v in a.items()}
    out = paired_bootstrap({"a": a, "b": b}, n_boot=2000, seed=1)
    d = out["differences"]["a minus b"]
    assert d["mean"] == pytest.approx(-0.25)
    assert d["interval"][0] == pytest.approx(-0.25) and d["interval"][1] == pytest.approx(-0.25)
    width = out["rows"]["a"]["interval"][1] - out["rows"]["a"]["interval"][0]
    assert width > 0.3  # Each row alone keeps the spread between games.


def test_known_distribution_of_two_games():
    # Differences 0 and 1 on two games. Resampling two with replacement gives mean difference 0, .5, 1 with chance 1/4, 1/2, 1/4.
    # A 40% interval runs from the 30th to the 70th percentile, both inside the middle mass at .5.
    # A 95% interval runs from the 2.5th to the 97.5th percentile, which sit at 0 and 1.
    rows = {"a": {"g0": 0.0, "g1": 1.0}, "b": {"g0": 0.0, "g1": 0.0}}
    narrow = paired_bootstrap(rows, n_boot=20000, seed=2, ci=0.40)["differences"]["a minus b"]
    wide = paired_bootstrap(rows, n_boot=20000, seed=2, ci=0.95)["differences"]["a minus b"]
    assert narrow["mean"] == pytest.approx(0.5)
    assert narrow["interval"] == pytest.approx((0.5, 0.5))
    assert wide["interval"] == pytest.approx((0.0, 1.0))


def test_mean_over_seeds_comes_first():
    # Game g0 has seeds 1, 0 and game g1 has seeds 1, 1, so the game means are .5 and 1, and the row mean is .75.
    assert game_means({"g0": [1, 0], "g1": [1, 1], "g2": 0.25}) == {"g0": 0.5, "g1": 1.0, "g2": 0.25}
    out = paired_bootstrap({"a": {"g0": [1, 0], "g1": [1, 1]}}, n_boot=100, seed=0)
    assert out["rows"]["a"]["mean"] == pytest.approx(0.75)


def test_same_seed_same_answer_and_unmatched_games_are_refused():
    rows = {"a": {"g0": 0.0, "g1": 1.0, "g2": 0.5}, "b": {"g0": 1.0, "g1": 1.0, "g2": 0.0}}
    assert paired_bootstrap(rows, n_boot=500, seed=3) == paired_bootstrap(rows, n_boot=500, seed=3)
    with pytest.raises(ValueError):
        paired_bootstrap({"a": {"g0": 0.0}, "b": {"g1": 0.0}})
    with pytest.raises(ValueError):
        paired_bootstrap({})
