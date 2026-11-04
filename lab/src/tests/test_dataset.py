"""The training set: game counts, splits and their leakage assertions on real files, the label rule, flagged labels, row families, and the text a live run shows."""
import gzip
import json
import random
from dataclasses import asdict
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # noqa: E401,E702

import pytest
from test_harness_standins import ScriptedLLM, ScriptedSys1

from alf.env import MAX_STEPS, TeacherLabel, _make_stack, list_games
from dataset import build, menu_stats, rows, splits
from dataset.generate import game_file
from dataset.rollouts import Game, RepeatedCommand, SeededTeacher, label_fields, play_game, undo_of
from dataset.rows import dedupe_key, family_steps, make_row, render_row, varied_path, varied_row
from dataset.splits import LeakageError, assert_no_leakage, assert_row_allowed, choose_held_out_rooms, filtered_games, make_splits, room_of, room_type, task_type_of
from harness.control import DEFAULT_CONFIG, LLM, SYS1, HarnessConfig
from harness.episode import run_episodes
from harness.reward import with_reward
from menu import fit
from sys1.base import FALLBACK_TEXT
from sys1.stub import StubSys1
from train.export_kev import export_rows

TASKS = {"pick_and_place_simple", "look_at_obj_in_light", "pick_clean_then_place_in_recep", "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep", "pick_two_obj_and_place"}


def uid(task, room, trial=1, folder="train"):
    return f"{folder}/{task}-Thing-None-Place-{room}/trial_T{trial}"


def small_game_list():
    out = []
    for room, tasks in ((1, ["pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep", "pick_clean_then_place_in_recep"]), (2, ["pick_heat_then_place_in_recep"]), (301, ["look_at_obj_in_light", "pick_two_obj_and_place"]), (201, ["pick_and_place_simple"]), (401, ["pick_and_place_simple", "pick_clean_then_place_in_recep"])):
        for task in tasks:
            for trial in range(3):
                out.append(uid(task, room, trial))
    return out


# Games, splits and leakage.

@pytest.fixture(scope="module")
def filtered():
    return {"train": filtered_games("train"), "valid_seen": filtered_games("valid_seen")}


def test_alfworlds_own_filter_gives_the_counts_papers_report(filtered):
    assert len(filtered["train"]) == 3553
    assert len(filtered["valid_seen"]) == 140
    assert {task_type_of(u) for u in filtered["train"]} == TASKS


def test_the_filter_refuses_the_test_folder():
    with pytest.raises(LeakageError):
        filtered_games("valid_unseen")


def test_held_out_rooms_are_whole_rooms_with_every_task_type_and_seed_fixed(filtered):
    made = make_splits(filtered["train"], filtered["valid_seen"], seed=0)
    assert len(made["dev_heldout_rooms"]) >= 140
    assert {task_type_of(u) for u in made["dev_heldout_rooms"]} == TASKS
    assert {room_of(u) for u in made["dev_heldout_rooms"]} == set(made["held_out_rooms"])
    assert not {room_of(u) for u in made["train"]} & set(made["held_out_rooms"])
    assert len(made["train"]) + len(made["dev_heldout_rooms"]) == 3553
    assert {room_type(r) for r in made["held_out_rooms"]} == {"kitchen", "bedroom", "living", "bathroom"}
    assert make_splits(filtered["train"], filtered["valid_seen"], seed=0)["held_out_rooms"] == made["held_out_rooms"]
    assert make_splits(filtered["train"], filtered["valid_seen"], seed=1)["held_out_rooms"] != made["held_out_rooms"]


def test_the_saved_lists_are_what_the_filter_and_the_seed_give(filtered):
    saved = splits.load_splits(pathlib.Path(__file__).resolve().parents[2] / "results" / "sys1-training-set")
    made = make_splits(filtered["train"], filtered["valid_seen"], seed=saved["seed"])
    for name in ("train", "dev", "dev_heldout_rooms", "held_out_rooms"):
        assert saved[name] == made[name]
    assert_no_leakage(saved)


def good_splits():
    return make_splits(small_game_list(), [uid("pick_and_place_simple", 1, 99, "valid_seen")], seed=0, min_games=3, min_per_task=0)


def test_the_good_input_passes():
    assert_no_leakage(good_splits())


@pytest.mark.parametrize("change, message", [
    (lambda s: s["train"].append(s["dev_heldout_rooms"][0]), "both train and dev_heldout_rooms"),
    (lambda s: s["train"].append(uid("pick_and_place_simple", s["held_out_rooms"][0], 77)), "rooms"),
    (lambda s: s["dev"].append(s["train"][0]), "both train and dev"),
    (lambda s: s["dev"].append(uid("pick_and_place_simple", 10, 1, "valid_unseen")), "test game"),
    (lambda s: s["train"].append(uid("pick_and_place_simple", 10, 1, "valid_unseen")), "test game"),
    (lambda s: s["dev_heldout_rooms"].append(uid("pick_and_place_simple", 999, 1)), "the list says"),
])
def test_a_deliberately_leaky_input_is_caught(change, message):
    leaky = good_splits()
    change(leaky)
    with pytest.raises(LeakageError, match=message):
        assert_no_leakage(leaky)


def test_a_stored_row_from_the_wrong_place_is_caught():
    made = good_splits()
    sets = splits.as_sets(made)
    assert_row_allowed({"split": "train", "game_id": made["train"][0]}, sets)
    assert_row_allowed({"split": "dev_heldout_rooms", "game_id": made["dev_heldout_rooms"][0]}, sets)
    with pytest.raises(LeakageError, match="not a game of split"):
        assert_row_allowed({"split": "train", "game_id": made["dev_heldout_rooms"][0]}, sets)
    with pytest.raises(LeakageError, match="test game"):
        assert_row_allowed({"split": "dev", "game_id": uid("pick_and_place_simple", 10, 1, "valid_unseen")}, sets)
    with pytest.raises(LeakageError, match="unknown split"):
        assert_row_allowed({"split": "test", "game_id": made["train"][0]}, sets)
    hidden = dict(sets)
    hidden["train"] = sets["train"] | {uid("pick_and_place_simple", made["held_out_rooms"][0], 5)}
    with pytest.raises(LeakageError, match="held-out room"):
        assert_row_allowed({"split": "train", "game_id": uid("pick_and_place_simple", made["held_out_rooms"][0], 5)}, hidden)


def test_choose_held_out_rooms_reaches_the_floors_or_says_it_cannot():
    games = small_game_list()
    rooms = choose_held_out_rooms(games, seed=3, min_games=5, min_per_task=1)
    chosen = [u for u in games if room_of(u) in rooms]
    assert len(chosen) >= 5 and {task_type_of(u) for u in chosen} == {task_type_of(u) for u in games}
    with pytest.raises(ValueError):
        choose_held_out_rooms(games, seed=3, min_games=10_000)


# The label rule and flags.

@pytest.mark.parametrize("label, source, field", [
    (TeacherLabel("go to desk 1", "go to desk 1", "go to desk 1"), "policy", None),
    (TeacherLabel("look", None, "look", label_initial=True), "initial-look", "label_initial"),
    (TeacherLabel(None, "go to desk 1", "look", label_random=True, label_inadmissible=False), "random-draw", "label_random"),
    (TeacherLabel("look", "go to mars 1", "look", label_inadmissible=True), "inadmissible", "label_inadmissible"),
    (TeacherLabel(None, None, "look", label_error="IndexError"), "error", "label_error"),
])
def test_label_fields_keep_only_usable_labels(label, source, field):
    fields = label_fields(label, ["go to desk 1", "look"])
    assert fields["label_source"] == source
    if field in ("label_random", "label_inadmissible", "label_error"):
        assert fields["label"] is None and fields[field]
    else:
        assert fields["label"] == label.command
    if field == "label_initial":
        assert fields["label_initial"]


def test_a_label_the_stored_list_does_not_hold_is_flagged():
    fields = label_fields(TeacherLabel("help", "help", "look"), ["look"])
    assert fields["label"] is None and fields["label_inadmissible"]


def fake_run(family="expert-path", won=True, perturb=None, split="train", game=None, labels=None):
    game = game or uid("pick_and_place_simple", 3)
    labels = labels or ["look", "go to a 1", "take x 1 from a 1", "go to b 1", "move x 1 to b 1", "look"]
    steps = []
    for t, label in enumerate(labels):
        command = label or "inventory"
        steps.append({"obs": f"obs{t}", "admissible": ["look", "inventory", "go to a 1", "go to b 1", "take x 1 from a 1", "move x 1 to b 1"], "command": command, "label": label, "label_source": "policy" if label else "random-draw", **({} if label else {"label_random": True})})
    return {"run_id": f"{game}#{family}#0", "split": split, "game_id": game, "task_type": "pick_and_place_simple", "family": family, "expert_seed": 0, "rollout": 0, "tracking": "always", "goal": "put x in b.", "first_obs": "obs0", "won": won, "final_obs": "end", "steps": steps, "perturb": perturb}


def fitted_menu():
    return fit([{"split": "train", "game_id": uid("pick_and_place_simple", 3), "task_type": "pick_and_place_simple", "step": 0, "expert": c} for c in ["go to a 1", "go to a 1", "go to b 1", "take x 1 from a 1", "move x 1 to b 1", "look"]])


def test_row_rule_of_each_family():
    assert family_steps(fake_run()) == [0, 1, 2, 3, 4, 5]
    assert family_steps(fake_run("recovery", perturb={"kind": "random", "start": 2, "end": 4, "commands": [], "formed": True})) == [3, 4, 5]
    loop = fake_run("loop-break", perturb={"kind": "no-effect", "start": 0, "end": 2, "commands": [], "formed": True}, labels=["look"] * 12)
    assert family_steps(loop) == [1, 2, 3, 4, 5]
    assert family_steps(fake_run("loop-break", perturb={"kind": "no-effect", "start": 0, "end": 1, "commands": [], "formed": False})) == []


def test_flagged_labels_stay_out_of_training_rows_and_are_counted_by_family(tmp_path):
    pool = tmp_path / "pool"
    (pool / "train").mkdir(parents=True)
    made = {"train": [uid("pick_and_place_simple", 3)], "dev": [], "dev_heldout_rooms": [], "held_out_rooms": []}
    runs = [fake_run(labels=["look", None, "go to a 1"]), fake_run("recovery", perturb={"kind": "random", "start": 0, "end": 1, "commands": ["inventory"], "formed": True}, labels=["look", None, "go to b 1"]),
            fake_run("loop-break", won=False, perturb={"kind": "no-effect", "start": 0, "end": 2, "commands": ["look", "look"], "formed": True}, labels=["look", "look", "go to b 1"])]
    with gzip.open(pool / "train" / "game.jsonl.gz", "wt") as f:
        for run in runs:
            f.write(json.dumps(run) + "\n")
    scan = build.scan_pool(pool, "train", splits.as_sets(made))
    assert scan["counts"]["flagged"] == {"expert-path/random-draw": 1, "recovery/random-draw": 1}
    assert scan["counts"]["rows_of_lost_runs"]["loop-break"] == 2
    assert scan["counts"]["rows_candidate"] == {"expert-path": 2, "recovery": 1}
    assert scan["counts"]["runs_by_task"]["loop-break/pick_and_place_simple/lost"] == 1
    assert all(row[3] != 2 for row in scan["candidates"]), "a lost rollout gave a candidate"


def test_export_of_a_stored_row_applies_the_label_rule():
    menu = fitted_menu()
    run = fake_run()
    on_menu = make_row(run, 1)
    off_menu = make_row(run, 4)
    flagged = make_row(fake_run(labels=[None, "look"]), 0)
    got, counts = export_rows([on_menu, off_menu, flagged], menu, 1, trace_window=5)
    assert counts["rows_dropped_label_random"] == 1 and len(got) == 2
    top = menu.menu("pick_and_place_simple", on_menu["admissible"], 1)
    assert list(got[0]["questions"]["command"]["criteria"]) == top + [FALLBACK_TEXT]
    assert got[0]["questions"]["command"]["label"] == (on_menu["expert"] if on_menu["expert"] in top else FALLBACK_TEXT)
    assert got[1]["questions"]["command"]["label"] == (off_menu["expert"] if off_menu["expert"] in top else FALLBACK_TEXT)
    assert got[0]["state"] == render_row(on_menu) and got[1]["state"] == render_row(off_menu)
    assert got[0]["_meta"]["family"] == "expert-path" and len({r["_meta"]["id"] for r in got}) == 2


def test_export_drops_a_row_of_a_lost_rollout_and_refuses_other_splits_and_mixed_records():
    run = fake_run(won=False)
    assert export_rows([make_row(run, 1)], fitted_menu(), 3)[1]["rows_dropped_not_won"] == 1
    with pytest.raises(AssertionError, match="split must be 'train'"):
        export_rows([make_row(fake_run(split="dev"), 1)], fitted_menu(), 3)
    old_style = {k: v for k, v in make_row(fake_run(), 1).items() if k not in ("trace", "first_obs", "control_state")}
    with pytest.raises(ValueError, match="one export takes one kind"):
        export_rows([make_row(fake_run(), 1), old_style], fitted_menu(), 3)


def test_make_row_writes_command_only_when_it_differs_and_keeps_the_interface_fields():
    run = fake_run("recovery", perturb={"kind": "random", "start": 1, "end": 2, "commands": ["inventory"], "formed": True})
    run["steps"][1]["command"] = "inventory"
    row = make_row(run, 1)
    assert row["command"] == "inventory" and row["expert"] == "go to a 1"
    assert "command" not in make_row(run, 2)
    for field in ("split", "game_id", "task_type", "goal", "step", "obs", "admissible", "expert", "done", "won", "family", "label_source", "tracking", "menu_rank", "acceptable", "control_state"):
        assert field in row
    assert [e["actor"] for e in make_row(run, 3)["trace"]] == ["teacher", "sys1", "teacher"]
    assert make_row(run, 5)["done"] and not make_row(run, 4)["done"]


def test_acceptable_says_what_other_seeds_gave_and_stays_unknown_with_one_seed():
    labels = {b"s": ({"a", "b"}, {0, 1}), b"t": ({"a"}, {0})}
    assert build.acceptable_at(labels, b"s", "a") == ["b"]
    assert build.acceptable_at(labels, b"t", "a") is None
    assert build.acceptable_at({b"s": ({"a"}, {0, 2})}, b"s", "a") == []


# Sampling.

def test_deduplicate_keeps_the_first_family():
    row = lambda family, key, step=0: (family, "t", 0, family, step, key)  # noqa: E731
    kept, counts, dropped = build.deduplicate([row(2, b"k"), row(0, b"k"), row(1, b"l"), row(1, b"l", 1)])
    assert [r[0] for r in kept] == [0, 1]
    assert counts == {"expert-path": 1, "recovery": 1} and dropped == {"loop-break": 1, "recovery": 1}


def test_quotas_sum_to_the_total_and_a_short_task_type_gives_its_rest_to_the_others():
    assert sum(build.proportional_quotas(101, {"a": 0.5, "b": 0.3, "c": 0.2}).values()) == 101
    rows_ = [(0, "rich", 0, 0, i, str(i).encode()) for i in range(1000)] + [(0, "poor", 0, 1, i, f"p{i}".encode()) for i in range(5)]
    picked, plan = build.sample_core(rows_, 100, {"expert-path": 1.0, "recovery": 0.0, "loop-break": 0.0}, {"rich": 1, "poor": 1}, seed=0)
    assert len(picked) == 100 and plan["expert-path"]["by_task_type"] == {"poor": 5, "rich": 95}
    again, _ = build.sample_core(list(reversed(rows_)), 100, {"expert-path": 1.0, "recovery": 0.0, "loop-break": 0.0}, {"rich": 1, "poor": 1}, seed=0)
    assert sorted(picked) == sorted(again)


def test_dedupe_key_sees_the_trace_window_and_the_label_and_not_older_history():
    a, b = fake_run(labels=["look"] * 8), fake_run(labels=["look"] * 8)
    assert dedupe_key(a, 7) == dedupe_key(b, 7)
    b["steps"][0]["command"] = "inventory"
    assert dedupe_key(a, 7) == dedupe_key(b, 7), "step 0 is older than the 5 entries Sys1 sees"
    b["steps"][6]["command"] = "inventory"
    assert dedupe_key(a, 7) != dedupe_key(b, 7)


# Plans.

def test_undo_of_each_kind():
    assert undo_of("go to a 1", "b 1") == "go to b 1"
    assert undo_of("go to a 1", None) is None
    assert undo_of("open a 1", None) == "close a 1"
    assert undo_of("close a 1", None) == "open a 1"
    assert undo_of("take x 1 from a 1", None) == "move x 1 to a 1"
    assert undo_of("move x 1 to a 1", None) == "take x 1 from a 1"
    assert undo_of("heat x 1 with a 1", None) is None


def test_repeated_command_alternates_with_its_undo_and_counts_runs():
    plan = RepeatedCommand(random.Random(1), 2, 3)
    plan.kind, plan.command, plan.undo = "already-used", "open a 1", "close a 1"
    assert plan.next(1, ["open a 1"], []) is None
    assert plan.next(2, ["open a 1", "look"], []) == "open a 1"
    assert plan.next(3, ["close a 1", "look"], []) == "close a 1"
    assert plan.next(4, ["open a 1"], []) == "open a 1"
    assert plan.next(5, ["close a 1"], []) == "close a 1"
    assert plan.next(6, ["open a 1"], []) == "open a 1"
    assert plan.next(7, ["close a 1"], []) is None
    assert plan.formed and plan.start == 2 and plan.end == 7


def test_repeated_command_that_cannot_run_twice_forms_no_repeat():
    plan = RepeatedCommand(random.Random(1), 0, 3)
    plan.kind, plan.command, plan.undo = "already-used", "go to a 1", None
    assert plan.next(0, ["go to a 1"], []) == "go to a 1"
    assert plan.next(1, ["look"], []) is None
    assert not plan.formed


# Real games.

def one_game_per_task(n=6):
    picked, seen = [], set()
    for spec in list_games("train", limit=400):
        if spec.task_type not in seen:
            seen.add(spec.task_type)
            picked.append(spec)
    return picked[:n]


GAMES = one_game_per_task()


@pytest.fixture(scope="module")
def stack():
    made = _make_stack(MAX_STEPS)
    yield made
    made.close()


@pytest.fixture(scope="module")
def game_runs(stack):
    out = {}
    for spec in GAMES:
        stack.load(spec.path)
        runs, _ = play_game(Game(spec, "train", stack), 0, [0, 1, 2], 2, 2)
        out[spec.uid] = runs
    return out


def test_expert_seed_zero_repeats_a_harness_teacher_run_command_by_command(game_runs):
    cfg = HarnessConfig(k=None, fallbacks_before_handover=2, llm_rounds=10, menu_run=3, trace_window=5, w=0, c=0, beta=0)
    logs = run_episodes(GAMES, None, None, None, cfg, "teacher", batch_size=len(GAMES), tracking="always")
    for spec, log in zip(GAMES, logs):
        run = game_runs[spec.uid][0]
        assert run["expert_seed"] == 0
        assert [s["command"] for s in run["steps"]] == [s.command for s in log.steps]
        assert run["won"] == log.won


def test_perturbed_rollouts_ran_admissible_commands_and_a_seed_gives_its_own_stream(game_runs):
    for runs in game_runs.values():
        families = {r["family"] for r in runs}
        assert families == {"expert-path", "recovery", "loop-break"}
        for run in runs:
            for s in run["steps"]:
                if s["label"] is not None:
                    assert s["label"] in s["admissible"]
            if run["family"] == "recovery":
                p = run["perturb"]
                assert 1 <= len(p["commands"]) <= 5 and [s["command"] for s in run["steps"][p["start"]:p["end"]]] == p["commands"]
                assert all(c in s["admissible"] for c, s in zip(p["commands"], run["steps"][p["start"]:]))
            if run["family"] == "loop-break" and run["perturb"]["formed"]:
                p = run["perturb"]
                assert p["commands"].count(p["commands"][0]) >= 2


def test_a_teacher_with_another_seed_has_another_stream():
    one, two = SeededTeacher(GAMES[0], 0), SeededTeacher(GAMES[0], 1)
    assert one._stream != two._stream
    assert SeededTeacher(GAMES[0], 1)._stream == two._stream


def test_labels_replay_from_the_stored_commands(stack, game_runs):
    spec = GAMES[1]
    run = next(r for r in game_runs[spec.uid] if r["family"] == "recovery")
    stack.load(spec.path)
    state = stack.reset()
    teacher = SeededTeacher(spec, run["expert_seed"])
    label = teacher.at_reset(state)
    for s in run["steps"]:
        assert label_fields(label, s["admissible"])["label"] == s["label"]
        state, _, done = stack.step(s["command"])
        if not done:
            label = teacher.after_step(state, s["command"])


# Text a live run shows.

def real_menu(game_runs):
    records = [{"split": "train", "game_id": spec.uid, "task_type": spec.task_type, "step": t, "expert": s["label"]} for spec in GAMES for r in game_runs[spec.uid] if r["family"] == "expert-path" for t, s in enumerate(r["steps"])]
    return fit(records)


def live_config(k):
    return with_reward(HarnessConfig(k=k, fallbacks_before_handover=DEFAULT_CONFIG.fallbacks_before_handover, llm_rounds=DEFAULT_CONFIG.llm_rounds, menu_run=DEFAULT_CONFIG.menu_run, trace_window=DEFAULT_CONFIG.trace_window, w=0, c=0, beta=0), "R2")


def test_a_plain_stored_row_renders_the_text_of_a_live_run_with_no_llm(game_runs):
    """A harness run with the expert running every command and no LLM holds the plain control state, as a Sys1-alone run does."""
    menu = real_menu(game_runs)
    sys1 = StubSys1(seed=0)
    sys1.eval()
    logs = run_episodes(GAMES, sys1, None, menu, live_config(10), "teacher", batch_size=len(GAMES), tracking="always", keep_prompts=True)
    checked = 0
    for spec, log in zip(GAMES, logs):
        run = game_runs[spec.uid][0]
        assert len(run["steps"]) == log.n_steps
        for t, s in enumerate(log.steps):
            row = make_row(run, t)
            assert render_row(row) == s.prompt
            assert row["obs"] == (log.first_obs if t == 0 else log.steps[t - 1].obs) and row["admissible"] == s.admissible
            assert row["expert"] == s.teacher_command or row["expert"] is None
            checked += 1
    assert checked > 40


def test_a_stored_row_of_a_perturbed_rollout_renders_the_text_of_a_live_sys1_alone_run(game_runs):
    """Sys1 alone, fallback masked, no LLM, scripted to the rollout's commands. Every command is on a menu of all admissible commands."""
    menu = real_menu(game_runs)
    spec = GAMES[2]
    run = next(r for r in game_runs[spec.uid] if r["family"] == "recovery")
    commands = [s["command"] for s in run["steps"]]

    class Script:
        def choose(self, prompt, options, call):
            return options.index(commands[call])

    sys1 = ScriptedSys1(Script().choose)
    logs = run_episodes([spec], sys1, None, menu, live_config(None), "test", mask_fallback=True, batch_size=1, tracking="always", keep_prompts=True)
    log = logs[0]
    assert [s.command for s in log.steps] == commands
    for step in family_steps(run):
        assert render_row(make_row(run, step)) == log.steps[step].prompt


def play_with_fallbacks(spec, run, menu, k, falls):
    """A live Gemma + Sys1 harness run on the rollout's commands. Sys1 falls back at the steps in `falls`, and the LLM's commands are the rollout's, so the expert stands in for the LLM."""
    commands = [s["command"] for s in run["steps"]]
    position = {"next": 0}

    class StepCounter:
        """The harness asks for one menu per step, so this keeps the step for the two scripts."""

        def menu(self, task_type, admissible, k_):
            position["t"] = position["next"]
            position["next"] += 1
            return menu.menu(task_type, admissible, k_)

    def sys1_choose(prompt, options, call):
        t = position["t"]
        return len(options) - 1 if t in falls or commands[t] not in options else options.index(commands[t])

    sys1 = ScriptedSys1(sys1_choose)
    llm = ScriptedLLM(lambda goal, history, admissible: commands[position["t"]])
    logs = run_episodes([spec], sys1, llm, StepCounter(), live_config(k), "test", batch_size=1, tracking="always", keep_prompts=True)
    assert [s.command for s in logs[0].steps] == commands, "the live run left the rollout's commands"
    return logs[0]


@pytest.mark.parametrize("task_index, k", [(0, 3), (2, 5), (4, 10)])
def test_a_varied_stored_row_renders_the_text_of_a_live_gemma_plus_sys1_run(game_runs, task_index, k):
    """The real harness, driven through the fallbacks of a row's simulated control path, shows the text `varied_row` stores, for an expert game and for a perturbed one."""
    menu = real_menu(game_runs)
    spec = GAMES[task_index]
    checked = 0
    for run in (game_runs[spec.uid][0], next(r for r in game_runs[spec.uid] if r["family"] == "recovery")):
        for step in family_steps(run)[1::3]:
            path = varied_path(run, step, menu, k, seed=task_index)
            if path is None:
                continue
            falls = {t for t, (before, actor) in enumerate(path[:step]) if before.controller == SYS1 and actor == LLM}
            log = play_with_fallbacks(spec, run, menu, k, falls)
            row = varied_row(make_row(run, step), run, menu, k, seed=task_index)
            assert row["control_varied"]
            assert render_row(row) == log.steps[step].prompt, f"{run['run_id']} step {step}"
            assert asdict(log.steps[step].control) == row["control_state"]
            for t in range(step):
                assert (log.steps[t].actor == LLM) == (path[t][1] == LLM), f"step {t}"
            checked += 1
    assert checked >= 1


def test_varied_row_shows_a_reachable_state_and_keeps_label_and_commands(game_runs):
    menu = real_menu(game_runs)
    spec = next(s for s in GAMES if len(game_runs[s.uid][0]["steps"]) >= 6)
    run = game_runs[spec.uid][0]
    found = 0
    for step in range(1, len(run["steps"])):
        row = make_row(run, step)
        varied = varied_row(row, run, menu, 10, seed=0)
        if not varied["control_varied"]:
            continue
        found += 1
        state = varied["control_state"]
        assert state["controller"] == SYS1 and state["llm_rounds_left"] == 0 and state["menu_run"] == 0
        assert state["fallbacks_in_row"] in (0, 1)
        assert state["fallbacks_in_row"] > 0 or any(e["actor"] == LLM for e in varied["trace"][-5:])
        assert [e["command"] for e in varied["trace"]] == [e["command"] for e in row["trace"]]
        assert varied["expert"] == row["expert"] and varied["obs"] == row["obs"]
        text = render_row(varied)
        assert ("LLM> " in text) or state["fallbacks_in_row"] > 0
    assert found >= 1
    assert varied_row(make_row(run, 0), run, menu, 10, seed=0)["control_varied"] is False


def test_choose_varied_takes_the_asked_share_of_all_rows_when_some_rows_have_no_path():
    values = {step: rows.varied_value("run", step, 0) for step in range(5000)}
    assert 0.47 < sum(v < 0.5 for v in values.values()) / len(values) < 0.53
    assert rows.varied_value("run", 3, 0) == values[3]
    with_path = [(values[step], step) for step in range(5000) if step % 2]
    chosen = rows.choose_varied(with_path, total=5000, share=0.2)
    assert len(chosen) == 1000 and all(step % 2 for _, step in chosen)
    assert chosen == rows.choose_varied(list(reversed(with_path)), total=5000, share=0.2)
    assert len(rows.choose_varied(with_path[:10], total=5000, share=0.2)) == 10


def test_stored_rows_go_through_the_export_with_the_harness_text(game_runs):
    menu = real_menu(game_runs)
    spec = GAMES[3]
    run = game_runs[spec.uid][0]
    records = [varied_row(make_row(run, t), run, menu, 5, seed=1) if t % 2 else make_row(run, t) for t in range(len(run["steps"]))]
    records = [r for r in records if r["expert"] is not None]
    got, counts = export_rows(records, menu, 5, trace_window=5)
    assert len(got) == len(records) and counts["expert_not_admissible"] == 0
    for row, record in zip(got, records):
        assert row["state"] == render_row(record)
        assert row["questions"]["command"]["label"] == (record["expert"] if record["expert"] in menu.menu(spec.task_type, record["admissible"], 5) else FALLBACK_TEXT)


def test_menu_statistics_agree_with_a_count_by_hand(tmp_path):
    game = uid("pick_and_place_simple", 4)
    sets = splits.as_sets({"train": [uid("pick_and_place_simple", 3)], "dev": [], "dev_heldout_rooms": [game], "held_out_rooms": [4]})
    expert = fake_run(split="dev_heldout_rooms", game=game)
    recovery = fake_run("recovery", split="dev_heldout_rooms", game=game, perturb={"kind": "random", "start": 1, "end": 2, "commands": ["go to a 1"], "formed": True})
    lost = fake_run("expert-path", won=False, split="dev_heldout_rooms", game=game)
    pool = tmp_path / "pool"
    path = game_file(pool, "dev_heldout_rooms", game)
    path.parent.mkdir(parents=True)
    with gzip.open(path, "wt") as f:
        for run in (expert, recovery, lost):
            f.write(json.dumps(run) + "\n")
    menu = fitted_menu()
    measured = menu_stats.measure(pool, "dev_heldout_rooms", {"form": menu, "kind": menu}, sets)
    ranks = [menu.rank(expert["task_type"], s["admissible"], s["label"]) for s in expert["steps"]]
    recovery_ranks = [ranks[t] for t in family_steps(recovery)]
    everything = ranks + recovery_ranks
    assert measured["runs"] == {"expert-path/won": 1, "expert-path/lost": 1, "recovery/won": 1}
    assert measured["rows_counted"] == len(everything)
    for name in menu_stats.k_name(5), menu_stats.k_name(10), menu_stats.k_name(None):
        curve, rule = measured["form_curve"][name], measured["rules"]["form"][name]
        assert curve["rows"] == len(everything) and curve["expert_rows"] == len(ranks)
        assert curve["rows_on_menu"] / curve["rows"] == rule["overall"]["coverage"]
        assert curve["expert_runs_all_on_menu"] / curve["expert_runs"] == rule["expert_runs_with_every_label_on_menu"]
    assert measured["form_curve"]["5"]["expert_rows_on_menu"] == sum(r <= 5 for r in ranks)
    assert measured["form_curve"]["all"]["rows_on_menu"] == len(everything)
    assert measured["inserted_commands_on_menu"]["form"]["all"] == {"recovery/commands": 1, "recovery/on_menu": 1}
