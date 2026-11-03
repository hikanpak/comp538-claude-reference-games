"""Metrics over `EpisodeLog`s. The definitions copy Bonn's (`jev_benchmarking/sequential/metrics.py`, `metrics.py`) where Bonn has one.

Which of Sys1's decisions a metric reads depends on `view`:

  - None: the decision this run really made, its `pick`, under the distribution it came from.
  - "masked": Sys1's top menu command, with fallback removed and the rest renormalized.
  - "unmasked": Sys1's top option, fallback included.

A step counts toward step match, ECE, Brier, NLL and AUROC when Sys1 decided it and the teacher's label is usable, which leaves
out `label_random` and `label_error` steps. A label of `look` that stands in for a command that was not admissible stays in,
because it is what the expert plays, and `step_match_excl_inadmissible` gives the number without those steps.

Step match is strict: a fallback never equals a command. For ECE and AUROC, a fallback counts as right when the teacher's command
is off the menu, as in the reward. Under the masked view, a teacher's command off the menu makes every pick wrong, and Brier treats
it as a further class of probability 0. NLL is undefined at such a step, so it is left out and `nll_excluded` counts it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from sys1.base import FALLBACK_TEXT

N_BINS = 15  # Bonn's.
LOOP_PERIOD = 2  # Bonn's: a step repeats if it equals one of the previous two.
FEEDBACK_CUT = 500  # Bonn cuts feedback to this length before comparing.


def masked_probs(probs: list[float]) -> list[float]:
    """Sys1's probabilities with the last option, fallback, removed and the rest renormalized."""
    menu = probs[:-1]
    total = sum(menu)
    if total <= 0:
        return [1.0 / len(menu)] * len(menu)
    return [p / total for p in menu]


@dataclass(frozen=True)
class Decision:
    pick: str
    confidence: float
    options: list[str]
    probs: list[float]
    teacher: str
    on_menu: bool
    right: bool  # The pick equals the teacher's command, or falls back when that is off the menu.
    strict: bool  # The pick equals the teacher's command.
    inadmissible: bool


def _distribution(s, view) -> tuple[str, list[str], list[float]]:
    if view == "masked" or (view is None and s.masked):
        probs = masked_probs(s.probs)
        pick = s.masked_pick if view == "masked" else s.pick
        return pick, s.menu, probs
    if view == "unmasked":
        return s.unmasked_pick, s.options, s.probs
    if view is None:
        return s.pick, s.options, s.probs
    raise ValueError(f"unknown view {view!r}")


def decisions(logs, view=None) -> list[Decision]:
    out = []
    for log in logs:
        for s in log.steps:
            if s.probs is None or s.teacher_command is None:
                continue
            pick, options, probs = _distribution(s, view)
            teacher = s.teacher_command
            on_menu = teacher in s.menu
            strict = pick == teacher
            right = strict or (pick == FALLBACK_TEXT and not on_menu)
            confidence = probs[options.index(pick)]
            out.append(Decision(pick, confidence, options, probs, teacher, on_menu, right, strict, s.label_inadmissible))
    return out


def ece(confidence: list[float], right: list[bool], n_bins: int = N_BINS) -> float | None:
    """Expected calibration error with equal-width bins, Bonn's `ece`."""
    if not confidence:
        return None
    bins: dict[int, list[int]] = {}
    for i, c in enumerate(confidence):
        bins.setdefault(min(int(c * n_bins), n_bins - 1), []).append(i)
    total = 0.0
    for members in bins.values():
        mean_conf = sum(confidence[i] for i in members) / len(members)
        mean_right = sum(right[i] for i in members) / len(members)
        total += len(members) * abs(mean_conf - mean_right)
    return total / len(confidence)


def brier(ds: list[Decision]) -> float | None:
    """Multi-class Brier, summed over options and averaged over steps, Bonn's form. The right option's target is the teacher's
    command, or fallback when that is off the menu and fallback is an option. With no such option, it adds 1."""
    if not ds:
        return None
    total = 0.0
    for d in ds:
        target = d.teacher if d.on_menu else (FALLBACK_TEXT if FALLBACK_TEXT in d.options else None)
        total += sum((p - (o == target)) ** 2 for o, p in zip(d.options, d.probs))
        if target is None:
            total += 1.0
    return total / len(ds)


def nll(ds: list[Decision]) -> tuple[float | None, int]:
    """Mean negative log probability of the right option, and how many steps had no such option."""
    values, excluded = [], 0
    for d in ds:
        target = d.teacher if d.on_menu else (FALLBACK_TEXT if FALLBACK_TEXT in d.options else None)
        if target is None:
            excluded += 1
            continue
        p = d.probs[d.options.index(target)]
        values.append(-math.log(p) if p > 0 else math.inf)
    return (sum(values) / len(values) if values else None), excluded


def auroc(confidence: list[float], right: list[bool]) -> float | None:
    """Chance a right pick has higher confidence than a wrong one, with ties at one half. None with one class only."""
    n_right = sum(right)
    n_wrong = len(right) - n_right
    if not n_right or not n_wrong:
        return None
    order = sorted(range(len(confidence)), key=lambda i: confidence[i])
    rank_sum, i = 0.0, 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and confidence[order[j + 1]] == confidence[order[i]]:
            j += 1
        midrank = (i + j) / 2 + 1
        rank_sum += midrank * sum(right[order[m]] for m in range(i, j + 1))
        i = j + 1
    return (rank_sum - n_right * (n_right + 1) / 2) / (n_right * n_wrong)


def repeat_metrics(logs, period: int = LOOP_PERIOD) -> dict:
    """Bonn's `repeat_metrics`: a step whose (command, observation) equals one of the previous `period` steps added nothing."""
    n_steps = repeats = loops = 0
    for log in logs:
        pairs = [(s.command, s.obs[:FEEDBACK_CUT]) for s in log.steps]
        flags = [any(t - k >= 0 and pairs[t] == pairs[t - k] for k in range(1, period + 1)) for t in range(len(pairs))]
        n_steps += len(pairs)
        repeats += sum(flags)
        run = best = 0
        for f in flags:
            run = run + 1 if f else 0
            best = max(best, run)
        loops += best >= 3
    return {"repeat_rate": repeats / n_steps if n_steps else None, "episodes_with_loop": loops / len(logs) if logs else None}


def step_match(logs, view=None) -> float | None:
    ds = decisions(logs, view)
    return sum(d.strict for d in ds) / len(ds) if ds else None


def exposure_gap(teacher_logs, own_logs, view="masked") -> float | None:
    """Step match on the teacher's states minus step match on Sys1's own states."""
    on_teacher, on_own = step_match(teacher_logs, view), step_match(own_logs, view)
    return None if on_teacher is None or on_own is None else on_teacher - on_own


def coverage_curve(ds: list[Decision], thresholds: list[float] | None = None) -> list[dict]:
    """Step match against the share of steps kept, as a threshold on the confidence in Sys1's top pick rises."""
    if thresholds is None:
        thresholds = [i / 20 for i in range(21)]
    rows = []
    for t in thresholds:
        kept = [d for d in ds if d.confidence >= t]
        rows.append({"threshold": t, "kept_share": len(kept) / len(ds) if ds else None, "step_match": sum(d.strict for d in kept) / len(kept) if kept else None, "n_kept": len(kept)})
    return rows


def fallback_rates(logs) -> dict:
    """Share of steps where Sys1's top option is fallback, on steps with the teacher's command off the menu and on it. These read the
    unmasked decision, so they apply to any run that logged probabilities."""
    off = on = fell_off = fell_on = 0
    for log in logs:
        for s in log.steps:
            if s.probs is None or s.teacher_command is None:
                continue
            fell = s.unmasked_pick == FALLBACK_TEXT
            if s.teacher_command in s.menu:
                on += 1
                fell_on += fell
            else:
                off += 1
                fell_off += fell
    return {
        "fallback_when_off_menu": fell_off / off if off else None,
        "fallback_when_on_menu": fell_on / on if on else None,
        "n_off_menu": off,
        "n_on_menu": on,
        "teacher_on_menu_share": on / (on + off) if on + off else None,
    }


def summarize(logs, view=None) -> dict:
    n_games = len(logs)
    steps = [s for log in logs for s in log.steps]
    n_steps = len(steps)
    ds = decisions(logs, view)
    confidence = [d.confidence for d in ds]
    right = [d.right for d in ds]
    labelled = [s for s in steps if s.has_teacher]
    kept = [d for d in ds if not d.inadmissible]
    llm_steps = [s for s in steps if s.llm_command is not None]
    nll_value, nll_excluded = nll(ds)

    def llm_calls_named(word: str) -> int:
        return sum(word in (s.llm_name or "").lower() for s in llm_steps)

    out = {
        "n_games": n_games,
        "n_steps": n_steps,
        "success": sum(log.won for log in logs) / n_games if n_games else None,
        "mean_steps": n_steps / n_games if n_games else None,
        **repeat_metrics(logs),
        "llm_calls": len(llm_steps),
        "gemma_calls": llm_calls_named("gemma"),
        "qwen_calls": llm_calls_named("qwen"),
        "tokens_in": sum(s.tokens_in for s in steps),
        "tokens_out": sum(s.tokens_out for s in steps),
        "sys1_share_of_steps": sum(s.actor == "sys1" for s in steps) / n_steps if n_steps else None,
        "n_decisions_with_label": len(ds),
        "step_match": sum(d.strict for d in ds) / len(ds) if ds else None,
        "step_match_excl_inadmissible": sum(d.strict for d in kept) / len(kept) if kept else None,
        "label_random_share": sum(s.label_random for s in labelled) / len(labelled) if labelled else None,
        "label_inadmissible_share": sum(s.label_inadmissible and not s.label_random for s in labelled) / len(labelled) if labelled else None,
        "label_error_share": sum(s.label_error is not None for s in labelled) / len(labelled) if labelled else None,
        "ece": ece(confidence, right),
        "brier": brier(ds),
        "nll": nll_value,
        "nll_excluded": nll_excluded,
        "auroc": auroc(confidence, right),
        "mean_confidence": sum(confidence) / len(confidence) if confidence else None,
        "coverage_curve": coverage_curve(ds),
        "seconds_sys1": sum(s.seconds_sys1 for s in steps),
        "seconds_llm": sum(s.seconds_llm for s in steps),
        "seconds_env": sum(s.seconds_env for s in steps),
    }
    out.update(fallback_rates(logs))
    return out


def success_by_task_type(logs) -> dict:
    groups: dict[str, list[bool]] = {}
    for log in logs:
        groups.setdefault(log.task_type, []).append(log.won)
    return {k: {"n": len(v), "success": sum(v) / len(v)} for k, v in sorted(groups.items())}
