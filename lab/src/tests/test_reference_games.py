"""lab/jobs/reference-games.py on a real dev game, with the expert's commands standing in for a reference model's."""
import csv
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from alf.env import MAX_STEPS, _Slot, game_spec

ROOT = Path(__file__).resolve().parents[3]
JOB = ROOT / "lab" / "jobs" / "reference-games.py"
GAME = "valid_seen/look_at_obj_in_light-AlarmClock-None-DeskLamp-323/trial_T20190909_044715_250790"


def run(results, *args, player="birch"):
    return subprocess.run([sys.executable, str(JOB), "--player", player, *args], capture_output=True, text=True, env={**os.environ, "REFERENCE_GAMES_RESULTS": str(results)})


def record_of(results, uid=GAME, player="birch"):
    return json.loads((Path(results) / player / uid.split("/")[0] / (uid.replace("/", "__") + ".json")).read_text())


def expert_commands(uid=GAME):
    slot = _Slot(MAX_STEPS, True, "foreign")
    view = slot.load(0, game_spec(uid))
    commands = []
    while not view.done:
        commands.append(view.label.played)
        view = slot.step(commands[-1])
    slot.close()
    assert view.won
    return commands


def test_a_whole_game_is_recorded_and_runs_once(tmp_path):
    assert "Admissible commands:" in run(tmp_path, "start", "--game", GAME, "--model", "birch").stdout
    commands = expert_commands()
    for command in commands:
        done = run(tmp_path, "step", "--game", GAME, "--command", command)
        assert done.returncode == 0, done.stderr
    assert f"GAME OVER won=true steps={len(commands)}" in done.stdout
    record = record_of(tmp_path)
    assert record["finished"] and record["won"] and record["steps"] == len(commands)
    assert [e["command"] for e in record["trace"]] == commands
    assert record["not_for_training"] and record["git_commit"] and record["prompt_sha256"]
    assert run(tmp_path, "start", "--game", GAME, "--model", "birch").returncode != 0
    assert run(tmp_path, "step", "--game", GAME, "--command", commands[0]).returncode != 0


def test_every_call_uses_one_hash_order_whatever_the_shell_sets(tmp_path):
    env = {**os.environ, "REFERENCE_GAMES_RESULTS": str(tmp_path), "PYTHONHASHSEED": "random"}
    subprocess.run([sys.executable, str(JOB), "--player", "birch", "start", "--game", GAME, "--model", "birch"], capture_output=True, text=True, env=env)
    assert record_of(tmp_path)["python_hash_seed"] == "0"


def test_a_command_off_the_list_costs_no_step_and_is_recorded(tmp_path):
    run(tmp_path, "start", "--game", GAME, "--model", "birch")
    done = run(tmp_path, "step", "--game", GAME, "--command", "fly to the moon")
    assert "not an admissible command" in done.stdout and "Steps taken: 0" in done.stdout
    record = record_of(tmp_path)
    assert record["steps"] == 0 and record["refused_commands"] == [{"after_step": 0, "command": "fly to the moon"}]


def test_a_restart_keeps_the_abandoned_attempt(tmp_path):
    run(tmp_path, "start", "--game", GAME, "--model", "birch")
    run(tmp_path, "step", "--game", GAME, "--command", "look")
    run(tmp_path, "start", "--game", GAME, "--model", "birch")
    record = record_of(tmp_path)
    assert record["steps"] == 0 and len(record["abandoned_attempts"]) == 1 and record["abandoned_attempts"][0]["trace"][0]["command"] == "look"


def test_a_changed_game_is_caught_by_the_replay(tmp_path):
    run(tmp_path, "start", "--game", GAME, "--model", "birch")
    run(tmp_path, "step", "--game", GAME, "--command", "look")
    path = Path(tmp_path) / "birch" / "valid_seen" / (GAME.replace("/", "__") + ".json")
    record = json.loads(path.read_text())
    record["trace"][0]["observation"] = "something else"
    path.write_text(json.dumps(record))
    assert "REPLAY FAILED" in run(tmp_path, "step", "--game", GAME, "--command", "look").stderr


@pytest.mark.parametrize("game", ["valid_unseen/some-game/trial_1", "train/some-game/trial_1"])
def test_games_off_the_lists_are_refused(tmp_path, game):
    done = run(tmp_path, "start", "--game", game, "--model", "birch")
    assert done.returncode != 0 and "none of the lists" in done.stderr


def test_the_test_list_is_134_test_games_and_prints_no_win_count(tmp_path):
    games = (ROOT / "lab" / "results" / "reference-games" / "test-games.txt").read_text().split()
    assert len(games) == len(set(games)) == 134 and all(g.startswith("valid_unseen/") for g in games)
    path = tmp_path / "birch" / "valid_unseen" / (games[0].replace("/", "__") + ".json")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"uid": games[0], "finished": True, "won": True, "abandoned_attempts": []}))
    shown = json.loads(run(tmp_path, "status", "--list", "test").stdout)
    assert shown["finished"] == 1 and "won" not in shown
    path = tmp_path / "birch" / "valid_seen" / (GAME.replace("/", "__") + ".json")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"uid": GAME, "finished": True, "won": True, "abandoned_attempts": []}))
    assert json.loads(run(tmp_path, "status", "--list", "dev").stdout)["won"] == 1


def test_replay_check_passes_a_true_record_and_catches_a_false_one(tmp_path):
    run(tmp_path, "start", "--game", GAME, "--model", "birch")
    run(tmp_path, "step", "--game", GAME, "--command", "look")
    def check():
        return subprocess.run([sys.executable, str(JOB), "replay-check", "--list", "dev"], capture_output=True, text=True, env={**os.environ, "REFERENCE_GAMES_RESULTS": str(tmp_path)})
    assert "REPLAY_OK" in check().stdout
    path = Path(tmp_path) / "birch" / "valid_seen" / (GAME.replace("/", "__") + ".json")
    record = json.loads(path.read_text())
    record["won"] = True
    path.write_text(json.dumps(record))
    assert "REPLAY_FAILED" in check().stdout


def test_gather_writes_the_three_tables_from_every_model(tmp_path):
    games = (ROOT / "lab" / "results" / "reference-games" / "test-games.txt").read_text().split()
    task = games[0].split("/")[1].split("-")[0]
    def put(player, uid, won, trace):
        path = tmp_path / player / "valid_unseen" / (uid.replace("/", "__") + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"uid": uid, "player": player, "task_type": uid.split("/")[1].split("-")[0], "finished": True, "won": won, "steps": len(trace), "abandoned_attempts": [], "refused_commands": [], "ended_early": False,
                                    "trace": [{"command": c, "observation": o} for c, o in trace], "prompt_sha256": "abc", "git_commit": "1234567890",
                                    "processes": [{"model": "elm", "duration_ms": 2000}]}))
    put("elm", games[0], True, [("look", "room"), ("look", "room"), ("go", "desk"), ("take", "pen")])
    put("elm", games[1], False, [("go", "desk")] * 2)
    done = subprocess.run([sys.executable, str(JOB), "gather", "--list", "test"], capture_output=True, text=True, env={**os.environ, "REFERENCE_GAMES_RESULTS": str(tmp_path), "PYTHONPATH": str(ROOT / "lab" / "src")})
    assert done.returncode == 0, done.stderr
    rows = {r["model"]: r for r in csv.DictReader((tmp_path / "summary" / "test-by-model.csv").open())}
    assert set(rows) == {"elm", "birch", "cypress", "ironwood"} and rows["birch"]["played"] == "0" and rows["birch"]["win_percent"] == ""
    elm = rows["elm"]
    assert (elm["games_on_list"], elm["played"], elm["not_played"], elm["won"], elm["win_percent"]) == ("134", "2", "132", "1", "50.0")
    assert elm["mean_steps"] == "3.0" and elm["mean_steps_won"] == "4.0" and float(elm["repeat_rate"]) == pytest.approx(2 / 6)
    tasks = [r for r in csv.DictReader((tmp_path / "summary" / "test-by-task.csv").open()) if r["model"] == "elm" and r["task_type"] == task]
    assert len(tasks) == 1 and int(tasks[0]["played"]) >= 1
    assert sum(int(r["played"]) for r in csv.DictReader((tmp_path / "summary" / "test-by-task.csv").open()) if r["model"] == "elm") == 2
    assert len(list(csv.DictReader((tmp_path / "summary" / "test-by-game.csv").open()))) == 2
    assert '"win_percent": 50.0' in done.stdout


def test_the_wilson_interval_matches_a_known_value():
    spec = importlib.util.spec_from_file_location("reference_games", JOB)
    job = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(job)
    assert job.wilson(50, 100) == (40.4, 59.6) and job.wilson(0, 0) == (None, None)


def test_no_training_code_reads_these_games():
    """The provider's usage policy: these runs are a benchmark and never train a model without the provider's authorization."""
    code = [p for folder in ("dataset", "train", "sys1", "harness", "alf") for p in (ROOT / "lab" / "src" / folder).rglob("*.py")]
    code += [p for p in (ROOT / "lab" / "jobs").glob("*") if p.is_file() and not p.name.startswith("reference-games")]
    assert code
    assert [p.name for p in code if "reference-games" in p.read_text() or "REFERENCE_GAMES_RESULTS" in p.read_text()] == []
