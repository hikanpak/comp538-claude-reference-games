"""The bootstrap of the plan: it resamples games, the same games for every row, on each game's mean over seeds.

The interval is the percentile interval, as in Bonn's `_bootstrap`. Bonn draws 1000 resamples, and the default here is 10000.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

N_BOOT = 10000


def game_means(row: dict) -> dict:
    """A row maps a game to one value or to one value per seed. The result maps a game to its mean over seeds."""
    out = {}
    for game, values in row.items():
        array = np.atleast_1d(np.asarray(values, dtype=float))
        if array.size == 0:
            raise ValueError(f"no value for game {game}")
        out[game] = float(array.mean())
    return out


def paired_bootstrap(rows: dict, n_boot: int = N_BOOT, seed: int = 0, ci: float = 0.95) -> dict:
    """Intervals for each row's mean over games and for each pair's difference, `first - second` in the order of `rows`.

    Every row must cover the same games, because a pair's difference is taken inside each resample of games.
    """
    means = {name: game_means(row) for name, row in rows.items()}
    names = list(means)
    if not names:
        raise ValueError("no rows")
    games = sorted(means[names[0]])
    for name in names:
        if sorted(means[name]) != games:
            raise ValueError(f"row {name!r} does not cover the same games as row {names[0]!r}")
    table = np.array([[means[name][g] for g in games] for name in names])
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(games), (n_boot, len(games)))
    resampled = table[:, picks].mean(axis=2)  # rows x resamples
    low, high = 100 * (1 - ci) / 2, 100 * (1 + ci) / 2

    def interval(values: np.ndarray) -> tuple[float, float]:
        return float(np.percentile(values, low)), float(np.percentile(values, high))

    out = {"n_games": len(games), "n_boot": n_boot, "seed": seed, "ci": ci, "rows": {}, "differences": {}}
    for i, name in enumerate(names):
        out["rows"][name] = {"mean": float(table[i].mean()), "interval": interval(resampled[i])}
    for i, j in combinations(range(len(names)), 2):
        out["differences"][f"{names[i]} minus {names[j]}"] = {"mean": float((table[i] - table[j]).mean()), "interval": interval(resampled[i] - resampled[j])}
    return out
