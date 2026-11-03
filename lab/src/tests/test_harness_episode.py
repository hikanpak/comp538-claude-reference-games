"""`run_episodes` on real ALFWorld training games: a batch with a stub Sys1, the expert to a win through the harness, every mode, and the
hand-over rules against sequences computed by hand."""
import math
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # noqa: E401,E702

import pytest
from test_harness_standins import ScriptedLLM, ScriptedSys1, StandInMenu

from alf.env import ExpertTeacher, list_games
from harness.control import LLM, SYS1, ControlState, HarnessConfig, start_state
from harness.episode import run_episodes
from harness.metrics import summarize
from harness.prompt import TraceEntry, render_prompt
from harness.reward import step_reward, with_reward
from sys1.base import FALLBACK_TEXT
from sys1.stub import StubSys1

MENU = StandInMenu()


def config(k=10, n=2, m=10, run=3, max_steps=50, reward="R2", window=5):
    cfg = HarnessConfig(k=k, fallbacks_before_handover=n, llm_rounds=m, menu_run=run, trace_window=window, w=0.0, c=0.0, beta=0.0, max_steps=max_steps)
    return with_reward(cfg, reward)


def per_task_type(n_each=1):
    picked, counts = [], {}
    for spec in list_games("train", limit=400):
        if counts.get(spec.task_type, 0) < n_each:
            counts[spec.task_type] = counts.get(spec.task_type, 0) + 1
            picked.append(spec)
    return picked


GAMES = per_task_type()
ONE_GAME = [next(g for g in GAMES if g.task_type == "pick_and_place_simple")]
ALWAYS_FALLBACK = lambda prompt, options, call: len(options) - 1  # noqa: E731
ALWAYS_FIRST = lambda prompt, options, call: 0  # noqa: E731


@pytest.fixture(scope="module")
def stub():
    model = StubSys1(seed=0)
    model.eval()
    return model


@pytest.fixture(scope="module")
def stub_logs(stub):
    return run_episodes(GAMES, stub, None, MENU, config(), "test", teacher="expert", seed=0, mask_fallback=True, batch_size=len(GAMES))


@pytest.fixture(scope="module")
def expert_logs():
    timing = {}
    logs = run_episodes(GAMES, None, None, None, config(), "teacher", seed=0, batch_size=len(GAMES), timing=timing)
    return logs, timing


def test_stub_sys1_plays_at_least_five_real_games_in_a_batch(stub_logs):
    assert len(stub_logs) == 6
    assert [log.uid for log in stub_logs] == [g.uid for g in GAMES]
    assert {log.task_type for log in stub_logs} == {g.task_type for g in GAMES}
    for log in stub_logs:
        assert 1 <= log.n_steps <= 50
        assert log.won or log.n_steps == 50
        assert log.goal and log.first_obs
        for t, s in enumerate(log.steps):
            assert s.step == t and s.actor == SYS1 and s.masked
            assert s.command == s.pick == s.masked_pick
            assert s.command in s.menu and set(s.menu) <= set(s.admissible) and len(s.menu) <= 10
            assert s.options == s.menu + [FALLBACK_TEXT]
            assert math.isclose(sum(s.probs), 1.0, abs_tol=1e-5)
            assert log.goal in s.prompt and log.first_obs in s.prompt and "Control: Sys1 controls" in s.prompt
            assert s.obs and s.control == start_state() and s.control_after == start_state()
            assert s.has_teacher and (s.teacher_command is not None or s.label_random or s.label_error)
            assert s.log_prob <= 0 and not s.sampled


def test_prompt_shows_the_last_commands_and_outputs(stub_logs):
    log = max(stub_logs, key=lambda e: e.n_steps)
    s = log.steps[7]
    trace = [TraceEntry(e.actor, e.command, e.obs) for e in log.steps[:7]]
    assert s.prompt == render_prompt(log.goal, log.first_obs, trace, s.control, window=5)
    assert [line for line in s.prompt.splitlines() if line.startswith("> ")] == [f"> {e.command}" for e in log.steps[2:7]]


def test_same_seed_repeats_and_batch_size_does_not_change_a_game(stub, stub_logs):
    again = run_episodes(GAMES, stub, None, MENU, config(), "test", teacher="expert", seed=0, mask_fallback=True, batch_size=2)
    assert [[s.command for s in log.steps] for log in again] == [[s.command for s in log.steps] for log in stub_logs]


def test_expert_plays_through_the_harness_to_a_win(expert_logs):
    logs, timing = expert_logs
    assert all(s.actor == "teacher" for log in logs for s in log.steps)
    assert all(s.probs is None and s.pick is None for log in logs for s in log.steps)
    won = [log for log in logs if log.won]
    assert len(won) >= 5  # A probe of 180 training games had the expert winning 78.9%, and pick_two_obj_and_place at 25%.
    pick_place = next(log for log in logs if log.task_type == "pick_and_place_simple")
    assert pick_place.won and pick_place.n_steps < 50
    assert pick_place.steps[0].command == "look" and pick_place.steps[-1].command.startswith("move ")
    assert summarize(logs)["success"] == pytest.approx(len(won) / 6)
    assert timing["batch_size"] == 6 and len(timing["batched_step_seconds"]) >= 1


def test_teacher_mode_scores_sys1_and_ignores_it(stub, expert_logs):
    logs = run_episodes(GAMES[:3], stub, None, MENU, config(), "teacher", seed=0, batch_size=3)
    expert = expert_logs[0]
    for log, alone in zip(logs, expert):
        assert [s.command for s in log.steps] == [s.command for s in alone.steps]
    assert all(s.probs is not None and s.actor == "teacher" and not s.sampled for log in logs for s in log.steps)


def test_scheduled_at_the_two_ends_and_in_the_middle(stub):
    games = GAMES[:2]
    all_teacher = run_episodes(games, stub, None, MENU, config(), "scheduled", teacher_prob=1.0, seed=1, mask_fallback=True, batch_size=2)
    plain = run_episodes(games, stub, None, MENU, config(), "teacher", seed=1, batch_size=2)
    assert [[s.command for s in x.steps] for x in all_teacher] == [[s.command for s in x.steps] for x in plain]
    assert not any(s.sampled for x in all_teacher for s in x.steps)

    no_teacher = run_episodes(games, stub, None, MENU, config(), "scheduled", teacher_prob=0.0, seed=1, mask_fallback=True, batch_size=2)
    assert all(s.actor == SYS1 and s.sampled and s.has_teacher for x in no_teacher for s in x.steps)

    mixed = run_episodes(games, stub, None, MENU, config(), "scheduled", teacher_prob=0.5, seed=1, mask_fallback=True, batch_size=2)
    actors = {s.actor for x in mixed for s in x.steps}
    assert actors == {SYS1, "teacher"}
    for x in mixed:
        for s in x.steps:
            assert s.sampled == (s.actor == SYS1)  # The PPO loss may use exactly the steps where Sys1's sampled pick ran.
            if s.actor == SYS1:
                assert s.command == s.pick


def test_model_mode_samples_from_sys1(stub):
    a = run_episodes(GAMES[:2], stub, None, MENU, config(), "model", teacher="expert", seed=1, mask_fallback=True, batch_size=2)
    b = run_episodes(GAMES[:2], stub, None, MENU, config(), "model", teacher="expert", seed=1, mask_fallback=True, batch_size=1)
    c = run_episodes(GAMES[:2], stub, None, MENU, config(), "model", teacher="expert", seed=2, mask_fallback=True, batch_size=2)
    commands = lambda logs: [[s.command for s in x.steps] for x in logs]  # noqa: E731
    assert commands(a) == commands(b)
    assert commands(a) != commands(c)
    assert all(s.sampled and s.command == s.pick for x in a for s in x.steps)


def test_test_mode_with_no_teacher_never_builds_the_expert(stub, monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("the expert was built")

    monkeypatch.setattr(ExpertTeacher, "__init__", refuse)
    logs = run_episodes(ONE_GAME, stub, None, MENU, config(max_steps=3), "test", seed=0, mask_fallback=True, batch_size=1)
    assert all(not s.has_teacher and s.teacher_command is None and s.reward == 0.0 for s in logs[0].steps)


def test_hand_over_sequence_by_hand():
    # N=2, M=3, no early return, Sys1 always falls back, the LLM always runs an off-menu command.
    # Step 0: fallback, the LLM takes it, one fallback in a row.
    # Step 1: fallback, the LLM takes it, count reaches 2, so the LLM controls with 3 rounds.
    # Steps 2, 3, 4: LLM rounds, no Sys1 question. Rounds left 2, 1, 0, and control returns.
    # Step 5: Sys1 falls back again, one in a row.
    llm = ScriptedLLM(lambda g, h, a: "inventory")
    sys1 = ScriptedSys1(ALWAYS_FALLBACK)
    logs = run_episodes(ONE_GAME, sys1, llm, MENU, config(k=1, n=2, m=3, run=0, max_steps=6), "test", seed=0, batch_size=1)
    steps = logs[0].steps
    assert len(steps) == 6
    assert [s.actor for s in steps] == [LLM] * 6
    assert [s.probs is None for s in steps] == [False, False, True, True, True, False]
    assert [s.pick for s in steps] == [FALLBACK_TEXT, FALLBACK_TEXT, None, None, None, FALLBACK_TEXT]
    assert [s.control for s in steps] == [
        ControlState(SYS1, 0, 0, 0), ControlState(SYS1, 1, 0, 0), ControlState(LLM, 0, 3, 0), ControlState(LLM, 0, 2, 0), ControlState(LLM, 0, 1, 0), ControlState(SYS1, 0, 0, 0)]
    assert steps[-1].control_after == ControlState(SYS1, 1, 0, 0)
    assert llm.reason_calls == 6 and llm.pick_calls == 0 and sys1.calls == 3
    assert all(s.tokens_in == 7 and s.tokens_out == 3 and s.llm_name == "stand-in-gemma" for s in steps)
    assert summarize(logs)["sys1_share_of_steps"] == 0.0 and summarize(logs)["llm_calls"] == 6


def test_run_on_the_menu_returns_control_early_by_hand():
    # N=1, M=10, run=2. Step 0: fallback hands over at once. Steps 1, 2: LLM runs menu commands, so the run reaches 2 and control returns.
    # Step 3: Sys1 decides again.
    llm = ScriptedLLM(lambda g, h, a: MENU.menu("", a, 1)[0])
    sys1 = ScriptedSys1(lambda p, o, c: 0 if c >= 1 else len(o) - 1)  # Falls back once, then picks the first menu command.
    steps = run_episodes(ONE_GAME, sys1, llm, MENU, config(k=3, n=1, m=10, run=2, max_steps=4), "test", seed=0, batch_size=1)[0].steps
    assert [s.actor for s in steps] == [LLM, LLM, LLM, SYS1]
    assert [s.control for s in steps] == [ControlState(SYS1, 0, 0, 0), ControlState(LLM, 0, 10, 0), ControlState(LLM, 0, 9, 1), ControlState(SYS1, 0, 0, 0)]
    assert steps[1].llm_on_menu and steps[2].llm_on_menu


def test_llm_sees_the_first_observation_and_the_history():
    llm = ScriptedLLM(lambda g, h, a: "look")
    logs = run_episodes(ONE_GAME, ScriptedSys1(ALWAYS_FALLBACK), llm, MENU, config(max_steps=3), "test", seed=0, batch_size=1)
    first = logs[0].first_obs
    assert llm.histories[0] == [("", first)]
    assert llm.histories[1] == [("", first), ("look", logs[0].steps[0].obs)]


def test_llms_unknown_command_is_answered_by_the_game():
    llm = ScriptedLLM(lambda g, h, a: "dance wildly")
    steps = run_episodes(ONE_GAME, ScriptedSys1(ALWAYS_FALLBACK), llm, MENU, config(max_steps=2), "test", teacher="expert", seed=0, batch_size=1)[0].steps
    assert steps[0].obs == "Nothing happens." and steps[0].llm_on_menu is False


def test_mask_stops_a_sys1_that_wants_to_fall_back():
    llm = ScriptedLLM(lambda g, h, a: "look")
    steps = run_episodes(ONE_GAME, ScriptedSys1(ALWAYS_FALLBACK), llm, MENU, config(max_steps=4), "test", seed=0, mask_fallback=True, batch_size=1)[0].steps
    assert all(s.actor == SYS1 and s.pick in s.menu and s.unmasked_pick == FALLBACK_TEXT for s in steps)
    assert llm.reason_calls == 0


def test_no_llm_with_fallback_open_runs_the_top_menu_command_on_the_masked_path():
    sys1 = ScriptedSys1(ALWAYS_FALLBACK)
    open_ = run_episodes(ONE_GAME, sys1, None, MENU, config(max_steps=6), "test", teacher="expert", seed=0, batch_size=1)[0].steps
    masked = run_episodes(ONE_GAME, ScriptedSys1(ALWAYS_FALLBACK), None, MENU, config(max_steps=6), "test", teacher="expert", seed=0, mask_fallback=True, batch_size=1)[0].steps
    assert all(s.pick == FALLBACK_TEXT and s.actor == SYS1 and s.control_after == start_state() for s in open_)
    assert [s.command for s in open_] == [s.command for s in masked]
    assert all(s.command == s.masked_pick for s in open_)


def test_teacher_mode_with_an_llm_scores_stored_picks_and_runs_the_teacher():
    llm = ScriptedLLM(lambda g, h, a: "inventory")
    logs = run_episodes(ONE_GAME, ScriptedSys1(ALWAYS_FALLBACK), llm, MENU, config(k=1, n=1, m=2, run=0), "teacher", seed=0, batch_size=1)
    expert = run_episodes(ONE_GAME, None, None, None, config(), "teacher", seed=0, batch_size=1)
    assert [s.command for s in logs[0].steps] == [s.command for s in expert[0].steps]
    assert all(s.actor == "teacher" for s in logs[0].steps)
    # Round 0 Sys1 falls back while the teacher runs, so no LLM is needed. Rounds 1 and 2 are LLM rounds, scored through pick.
    assert llm.pick_calls >= 2 and llm.reason_calls == 0
    assert logs[0].steps[1].control == ControlState(LLM, 0, 2, 0) and logs[0].steps[1].llm_command == "inventory"


def test_rewards_and_beta_on_real_steps():
    cfg = config(reward="R3")
    logs = run_episodes(ONE_GAME, ScriptedSys1(ALWAYS_FIRST), None, MENU, cfg, "teacher", seed=0, batch_size=1)
    log = logs[0]
    assert log.won
    for s in log.steps[:-1]:
        assert s.reward == step_reward(s.pick, s.teacher_command, s.menu, cfg)
    last = log.steps[-1]
    assert last.reward == step_reward(last.pick, last.teacher_command, last.menu, cfg) + 5.0
    assert any(s.reward == 1.0 for s in log.steps) and any(s.reward == -1.0 for s in log.steps)
    r4 = run_episodes(ONE_GAME, ScriptedSys1(ALWAYS_FIRST), None, MENU, config(reward="R4"), "teacher", seed=0, batch_size=1)[0]
    assert [s.reward for s in r4.steps] == [0.0] * (r4.n_steps - 1) + [1.0]


def test_flagged_labels_pay_nothing_and_stay_out_of_step_match(stub):
    # In a pick_two game the shipped expert flails, so flags appear on the teacher's own path.
    game = [g for g in GAMES if g.task_type == "pick_two_obj_and_place"]
    log = run_episodes(game, stub, None, MENU, config(), "teacher", seed=0, batch_size=1)[0]
    flagged = [s for s in log.steps if s.label_random]
    assert flagged, "this game was picked because its expert run has flagged labels"
    assert all(s.teacher_command is None and s.reward == 0.0 for s in flagged)
    s = summarize([log])
    assert s["label_random_share"] == pytest.approx(len(flagged) / log.n_steps)
    assert s["n_decisions_with_label"] == log.n_steps - len(flagged)


def test_bad_arguments_are_refused(stub):
    with pytest.raises(ValueError):
        run_episodes(ONE_GAME, stub, None, MENU, config(), "fly")
    with pytest.raises(ValueError):
        run_episodes(ONE_GAME, None, None, None, config(), "test")
    with pytest.raises(ValueError):
        run_episodes(ONE_GAME, stub, None, None, config(), "test", teacher="expert")
    with pytest.raises(ValueError):
        run_episodes(ONE_GAME, stub, None, MENU, config(), "scheduled", teacher_prob=1.5)
    with pytest.raises(ValueError):
        run_episodes(ONE_GAME, stub, None, MENU, config(), "test", teacher="oracle")


def test_workers_give_the_same_episodes(stub, stub_logs):
    logs = run_episodes(GAMES, stub, None, MENU, config(), "test", teacher="expert", seed=0, mask_fallback=True, batch_size=6, num_workers=2)
    assert [[(s.command, s.teacher_command, s.obs) for s in x.steps] for x in logs] == [[(s.command, s.teacher_command, s.obs) for s in x.steps] for x in stub_logs]
