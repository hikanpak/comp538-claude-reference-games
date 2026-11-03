"""The batched env on real training games: explicit list, step limit, test-game refusal, rolling batch, worker parity."""
import os
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # noqa: E401,E702

import pytest
from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv

from alf.env import AlfBatchEnv, game_spec, list_games, load_game_list, resolve_games, split_intro

UNSEEN = "/nowhere/json_2.1.1/valid_unseen/pick_and_place_simple-Pen-None-Desk-1/trial_T1/game.tw-pddl"


@pytest.fixture(scope="module")
def games():
    return list_games("train", limit=5)


def test_test_games_are_refused_from_the_path_alone():
    with pytest.raises(ValueError):
        game_spec(UNSEEN)
    with pytest.raises(ValueError):
        game_spec("valid_unseen/pick_and_place_simple-Pen-None-Desk-1/trial_T1")
    with pytest.raises(ValueError):
        resolve_games([UNSEEN])
    with pytest.raises(ValueError):
        AlfBatchEnv([UNSEEN])
    with pytest.raises(ValueError):
        list_games("valid_unseen")
    assert game_spec(UNSEEN, allow_test_games=True).split == "valid_unseen"  # The file does not exist, and nothing opened it.


def test_three_spellings_of_a_game_give_one_spec(games):
    spec = games[0]
    assert game_spec(spec.uid) == spec
    assert game_spec(os.path.dirname(spec.path)) == spec
    assert game_spec(spec.path) == spec
    assert spec.uid.startswith("train/") and spec.task_type == spec.uid.split("/")[1].split("-")[0]


def test_game_list_file(tmp_path, games):
    lines = tmp_path / "games.txt"
    lines.write_text("# dev games\n" + "\n".join(s.uid for s in games[:2]) + "\n")
    assert load_game_list(str(lines)) == games[:2]
    listed = tmp_path / "games.json"
    listed.write_text('["%s"]' % games[0].path)
    assert load_game_list(str(listed)) == games[:1]
    with pytest.raises(ValueError):
        resolve_games([games[0], games[0]])  # The same game twice.


def test_is_an_alfred_env_and_never_walks_the_data_folder(games, monkeypatch):
    monkeypatch.setattr(os, "walk", lambda *a, **k: pytest.fail("walked the data folder"))
    with AlfBatchEnv(games[:2], batch_size=2) as env:
        assert isinstance(env, AlfredTWEnv)
        assert env.game_files == [s.path for s in games[:2]]


def test_step_limit_ends_the_game(games):
    with AlfBatchEnv(games[:1], max_steps=50, batch_size=1, with_teacher=False) as env:
        v = env.reset()[0]
        n = 0
        while not v.done:
            (v,), started = env.step({0: "look"})
            n += 1
            assert not started
        assert n == 50 and v.step == 50 and v.won is False


def test_view_cuts_task_and_room_and_removes_help(games):
    with AlfBatchEnv(games[:1], batch_size=1) as env:
        v = env.reset()[0]
        assert v.goal and "Your task is to" not in v.first_obs and "Welcome" not in v.first_obs
        assert v.first_obs.startswith("You are in the middle of a room")
        assert "help" not in v.admissible and "look" in v.admissible
        assert v.label.label_initial and v.label.command == "look"
    goal, first = split_intro("-= Welcome to TextWorld, ALFRED! =-\n\nA room.\n\nYour task is to: put a pen on desk.")
    assert (goal, first) == ("put a pen on desk.", "A room.")


def test_unknown_command_is_answered_and_uses_a_step(games):
    with AlfBatchEnv(games[:1], batch_size=1) as env:
        env.reset()
        (v,), _ = env.step({0: "dance wildly"})
        assert v.obs == "Nothing happens." and v.step == 1 and not v.done


def test_rolling_batch_plays_every_game_and_keeps_at_most_batch_size_live(games):
    with AlfBatchEnv(games, max_steps=3, batch_size=2, with_teacher=False) as env:
        live = {v.index: v for v in env.reset()}
        assert sorted(live) == [0, 1]
        finished = []
        while live:
            assert len(live) <= 2
            stepped, started = env.step({i: "look" for i in live})
            live = {v.index: v for v in stepped if not v.done}
            finished += [v.index for v in stepped if v.done]
            live.update({v.index: v for v in started})
        assert sorted(finished) == [0, 1, 2, 3, 4]
        assert len(env.step_seconds) > 0 and len(env.load_seconds) > 0


def test_workers_give_the_same_game_as_this_process(games):
    def play(num_workers):
        commands = {}
        with AlfBatchEnv(games, batch_size=5, num_workers=num_workers) as env:
            live = {v.index: v for v in env.reset()}
            while live:
                order = {i: v.label.played for i, v in live.items()}
                for i, c in order.items():
                    commands.setdefault(i, []).append(c)
                stepped, _ = env.step(order)
                live = {v.index: v for v in stepped if not v.done}
        return commands

    assert play(2) == play(0)


def test_more_than_four_workers_are_refused(games):
    with pytest.raises(ValueError):
        AlfBatchEnv(games, num_workers=5)
