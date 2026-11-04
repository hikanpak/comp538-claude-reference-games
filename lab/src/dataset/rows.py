"""From a stored rollout to stored rows. A row stands alone: it carries everything `render_prompt` needs, so rows can be sampled and deduplicated.

The fields of `lab/INTERFACES.md`, with these meanings:

  - `obs`, `admissible`: the state, as `GameView` shows it. `expert`: the teacher's label there, None when flagged. `step`: commands run so far.
  - `done`: the command run at this state ended the game. `won`: the checker marked this rollout won.
  - `family`, `label_source` (`policy`, `initial-look`, or the flag `random-draw`, `inadmissible`, `error`), `tracking`.
  - `command`: the command that ran here, written only when it differs from the label.
  - `menu_rank`: the 1-based position of the label in the fitted menu's full ordering. On a menu of length k when `menu_rank <= k`. None if the label is absent.
  - `acceptable`: other commands expert seeds gave at this same state, which means the same game and the same commands so far. `[]` when two or more seeds
    reached the state and agreed, None when only one seed reached it.
  - `control_state`: the fields of `harness.control.ControlState`.

Beyond them: `first_obs` and `trace` (every earlier command, oldest first, as `{"actor", "command", "output"}`) so `render_prompt` runs on the row alone, `run_id`, `expert_seed`, the flags, and
`control_varied`. The replay in `dataset/audit.py` needs only `trace`.

Which rows a rollout gives:

  - `expert-path`: every state.
  - `recovery`: every state after the first random command ran, so no row repeats the clean path up to that point.
  - `loop-break`: the states after the first repeat ran, while the last two inserted commands are still in the 5 trace entries Sys1 sees. A later state shows no sign of
    the repeat. A rollout whose command ran fewer than twice formed no repeat and gives no row.

A trace entry's actor is `teacher` for an expert's command and `sys1` for an inserted command, which stands for Sys1's own mistake. Both render the same. `varied_row` marks entries `llm`.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import asdict

from harness.control import DEFAULT_CONFIG, LLM, SYS1, ControlState, HarnessConfig, after_fallback, after_llm_round, after_pick, start_state
from harness.prompt import DEFAULT_TRACE_WINDOW, render_prompt

LOOP_VISIBLE_AFTER_END = 3  # The last two inserted commands sit at end-1 and end-2. With a window of 5 they are both shown up to step end+3.
FLAG_FIELDS = ("label_random", "label_inadmissible", "label_error", "label_initial")
PLAIN_CONTROL = asdict(start_state())
CONTROL_SHARE = 0.2  # Share of training rows whose control state and trace marks vary.
P_FALLBACK = 0.3  # The chance that Sys1 falls back on a menu command when a varied row's control is simulated. Off-menu commands always fall back.
ATTEMPTS = 12


def family_steps(run: dict) -> list[int]:
    n = len(run["steps"])
    perturb = run["perturb"]
    if run["family"] == "expert-path":
        return list(range(n))
    if run["family"] == "recovery":
        return list(range(perturb["start"] + 1, n))
    if run["family"] == "loop-break":
        if not perturb["formed"]:
            return []
        return list(range(perturb["start"] + 1, min(n, perturb["end"] + LOOP_VISIBLE_AFTER_END + 1)))
    raise ValueError(f"unknown family {run['family']!r}")


def _actor(run: dict, step: int) -> str:
    perturb = run["perturb"]
    return SYS1 if perturb and perturb["start"] <= step < perturb["end"] else "teacher"


def outputs_of(run: dict) -> list[str]:
    """What each command answered: the next state's `obs`, and at the end the final observation."""
    return [s["obs"] for s in run["steps"][1:]] + [run["final_obs"]]


def state_hashes(run: dict) -> list[bytes]:
    """One digest per state, from the game and the commands so far, so two rollouts share a digest only at the same state with the same history."""
    digest = hashlib.blake2b(run["game_id"].encode(), digest_size=16).digest()
    out = [digest]
    for s in run["steps"][:-1]:
        digest = hashlib.blake2b(digest + s["command"].encode(), digest_size=16).digest()
        out.append(digest)
    return out


def dedupe_key(run: dict, step: int, outputs: list[str] | None = None) -> bytes:
    """Equal when the text Sys1 sees in a plain row, the admissible commands and the label are equal."""
    outputs = outputs if outputs is not None else outputs_of(run)
    s = run["steps"][step]
    start = max(0, step - DEFAULT_TRACE_WINDOW)
    shown = [(run["steps"][i]["command"], outputs[i]) for i in range(start, step)]
    text = repr((run["goal"], run["first_obs"], shown, sorted(s["admissible"]), s["label"]))
    return hashlib.blake2b(text.encode(), digest_size=16).digest()


def make_row(run: dict, step: int, outputs: list[str] | None = None, acceptable: list[str] | None = None, menu_rank: int | None = None) -> dict:
    outputs = outputs if outputs is not None else outputs_of(run)
    s = run["steps"][step]
    label = s["label"]
    row = {"split": run["split"], "game_id": run["game_id"], "task_type": run["task_type"], "goal": run["goal"], "step": step, "obs": s["obs"], "admissible": s["admissible"],
           "expert": label, "done": step == len(run["steps"]) - 1, "won": run["won"], "family": run["family"], "label_source": s["label_source"], "tracking": run["tracking"],
           "menu_rank": menu_rank, "acceptable": acceptable, "control_state": dict(PLAIN_CONTROL), "control_varied": False, "first_obs": run["first_obs"],
           "trace": [{"actor": _actor(run, i), "command": run["steps"][i]["command"], "output": outputs[i]} for i in range(step)], "run_id": run["run_id"], "expert_seed": run["expert_seed"]}
    if s["command"] != label:
        row["command"] = s["command"]
    for flag in FLAG_FIELDS:
        if s.get(flag):
            row[flag] = s[flag]
    return row


def render_row(row: dict, window: int = DEFAULT_TRACE_WINDOW) -> str:
    """The text Sys1 sees at this row, from the harness's own function."""
    return render_prompt(row["goal"], row["first_obs"], row["trace"], row["control_state"], window=window)


def simulate_control(on_menu: list[bool], rng: random.Random, cfg: HarnessConfig = DEFAULT_CONFIG, p_fallback: float = P_FALLBACK) -> list[tuple[ControlState, str]]:
    """Who ran each command and the control state before it, through the harness's own rules, as if the rollout's commands were picks of Sys1 and of an LLM. A command off this
    step's menu cannot be Sys1's pick, so Sys1 fell back there and the LLM ran it. Under Sys1's control a menu command falls back with chance `p_fallback`. While the LLM
    controls, Sys1 is not asked. The rollout's commands stay what they were, so the expert stands in for the LLM."""
    control = start_state()
    out = []
    for listed in on_menu:
        before = control
        if control.controller == LLM:
            actor, control = LLM, after_llm_round(control, listed, cfg)
        elif not listed or rng.random() < p_fallback:
            actor, control = LLM, after_fallback(control, cfg)
        else:
            actor, control = SYS1, after_pick(control)
        out.append((before, actor))
    return out


def varied_value(run_id: str, step: int, seed: int) -> float:
    """A row's place in the order the builder tries rows for a varied control state. A hash in [0, 1), so it does not depend on the order rows are read in."""
    return int.from_bytes(hashlib.sha256(f"control/{seed}/{run_id}/{step}".encode()).digest()[:8], "big") / 2**64


def choose_varied(entries: list[tuple], total: int, share: float = CONTROL_SHARE) -> list[tuple]:
    """The rows to vary, from `entries`, each starting with its `varied_value` and holding only rows that have a path. The lowest values are taken up to `share` of `total` rows, because a
    fixed draw per row would fall short of the share by the rows with no path, whose number depends on the menu's coverage."""
    return sorted(entries)[:round(share * total)]


def menu_flags(run: dict, menu, k: int | None) -> list[bool]:
    """Whether each command that ran was on its state's menu."""
    return [s["command"] in menu.menu(run["task_type"], s["admissible"], k) for s in run["steps"]]


def varied_path(run: dict, step: int, menu, k: int | None, seed: int, cfg: HarnessConfig = DEFAULT_CONFIG, flags: list[bool] | None = None) -> list[tuple[ControlState, str]] | None:
    """A simulated control path up to `step`, in which Sys1 is asked at `step` and the row shows a fallback count or an LLM mark. None when twelve seeded tries find none. A try that leaves
    the LLM in control at this step is refused, because Sys1 is not asked there, and so is one that leaves nothing to see."""
    if step == 0:
        return None
    on_menu = (flags if flags is not None else menu_flags(run, menu, k))[:step] + [True]
    for attempt in range(ATTEMPTS):
        rng = random.Random(hashlib.sha256(f"control-path/{seed}/{run['run_id']}/{step}/{attempt}".encode()).hexdigest())
        path = simulate_control(on_menu, rng, cfg)
        before = path[step][0]
        shown = [actor for _, actor in path[:step]][-cfg.trace_window:] if cfg.trace_window > 0 else []
        if before.controller == SYS1 and (before.fallbacks_in_row > 0 or LLM in shown):
            return path
    return None


def varied_row(row: dict, run: dict, menu, k: int | None, seed: int, cfg: HarnessConfig = DEFAULT_CONFIG) -> dict:
    """The same row with a control state and LLM marks a live run of Gemma + Sys1 at `cfg` could show, or the row unchanged when none is found. Its commands, outputs and label do not change."""
    path = varied_path(run, row["step"], menu, k, seed, cfg)
    if path is None:
        return row
    trace = [{**entry, "actor": LLM if actor == LLM else entry["actor"]} for entry, (_, actor) in zip(row["trace"], path)]
    return {**row, "trace": trace, "control_state": asdict(path[row["step"]][0]), "control_varied": True}
