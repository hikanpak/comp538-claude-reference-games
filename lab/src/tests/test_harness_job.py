"""`lab/jobs/eval-sys1-alone.py`: every row of the acceptance file gets a measurement, the result file has the promised fields, and the
verdict over seeds works. On real training games, with a stand-in menu that exists only here."""
import importlib.util
import json
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # noqa: E401,E702
import types

import pytest
import yaml
from test_harness_standins import StandInMenu

from alf.env import list_games
from harness.control import DEFAULT_CONFIG
from sys1.stub import StubSys1

ROOT = pathlib.Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def job():
    spec = importlib.util.spec_from_file_location("eval_sys1_alone", ROOT / "lab" / "jobs" / "eval-sys1-alone.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_sys1_alone"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def games():
    return list_games("train", limit=3)


def test_every_row_of_the_acceptance_file_has_a_measurement(job):
    rows = job.read_gates()
    assert len(rows) == 8
    by_name = {r["gate"]: r for r in rows}
    assert by_name["Solves games"]["direction"] == ">=" and by_name["Solves games"]["threshold"] == 54.5
    assert by_name["Does not loop"]["direction"] == "<=" and by_name["Does not loop"]["threshold"] == 0.10
    assert by_name["Seeds agree"]["direction"] == "<=" and by_name["Seeds agree"]["threshold"] == 10


def test_judge_keeps_not_measured_apart_from_failed(job):
    assert job.judge(None, ">=", 1.0) is None
    assert job.judge(1.0, ">=", 1.0) is True
    assert job.judge(0.9, ">=", 1.0) is False
    assert job.judge(0.9, "<=", 1.0) is True


def test_evaluate_on_real_games_measures_each_row(job, games):
    model = StubSys1(seed=0)
    model.eval()
    result = job.evaluate(model, StandInMenu(), games, DEFAULT_CONFIG, seed=0, batch_size=3)
    gates = {g["gate"]: g for g in result["gates"]}
    for name, gate in gates.items():
        if name == "Seeds agree":
            assert gate["value"] is None and gate["passed"] is None  # One seed cannot measure it, and it does not pretend to.
        elif name == "Falls back when it should" and gate["value"] is None:
            assert result["aggregates"]["own_unmasked"]["n_off_menu"] == 0
        elif name == "Confidence ranks right above wrong" and gate["value"] is None:
            assert result["aggregates"]["own_masked"]["step_match"] in (0.0, 1.0)  # One class only, so it stays not measured.
            assert gate["passed"] is None
        else:
            assert isinstance(gate["value"], float), name
            assert isinstance(gate["passed"], bool), name
    a = result["aggregates"]
    assert set(a) == {"teacher_states", "teacher_states_fallback_open", "own_masked", "own_unmasked"}
    assert gates["Solves games"]["value"] == a["own_masked"]["success_pct"]
    assert result["exposure_gap"] == pytest.approx(a["teacher_states"]["step_match"] - a["own_masked"]["step_match"])
    assert len(a["own_masked"]["coverage_curve"]) == 21
    assert a["teacher_states"]["sys1_share_of_steps"] == 0.0 and a["own_masked"]["sys1_share_of_steps"] == 1.0
    assert len(result["per_game"]["own_masked"]) == 3
    assert result["timing"]["batched_env_step_seconds_mean"] > 0


def test_command_line_writes_a_result_file_then_a_verdict(job, games, tmp_path, monkeypatch):
    # `menu` is replaced for this test, so it runs whether or not lab/src/menu.py exists yet.
    standin = types.ModuleType("menu")

    class MenuModel(StandInMenu):
        @classmethod
        def load(cls, path):
            return cls()

    standin.MenuModel = MenuModel
    monkeypatch.setitem(sys.modules, "menu", standin)
    listing = tmp_path / "games.txt"
    listing.write_text("\n".join(g.uid for g in games))
    menu_file = tmp_path / "menu.json"
    menu_file.write_text("{}")
    paths = []
    for seed in (0, 1, 2):
        out = job.main(["--sys1", "stub", "--menu", str(menu_file), "--games", str(listing), "--seed", str(seed), "--run", f"t-seed{seed}", "--device", "cpu", "--results-root", str(tmp_path), "--max-steps", "8"])
        paths.append(str(out))
    data = json.loads(pathlib.Path(paths[0]).read_text())
    for key in ("config", "seed", "git_commit", "environment", "hardware", "precision", "per_game", "aggregates", "gates", "timing", "peak_memory_bytes"):
        assert key in data, key
    assert data["seed"] == 0 and data["config"]["max_steps"] == 8 and data["hardware"]["device"] == "cpu"
    assert data["environment"]["packages"]["alfworld"] == "0.4.2"

    verdict = tmp_path / "verdict.json"
    job.main(["--aggregate", *paths, "--out", str(verdict)])
    v = json.loads(verdict.read_text())
    assert v["seeds"] == [0, 1, 2] and isinstance(v["passed"], bool)
    gates = {g["gate"]: g for g in v["gates"]}
    assert set(gates) == set(job.GATE_MEASURES)
    success = gates["Solves games"]["per_seed"]
    assert gates["Solves games"]["value"] == pytest.approx(sum(success) / 3)
    assert gates["Solves games"]["worst_seed"] == min(success)
    assert gates["Seeds agree"]["value"] == pytest.approx(sum(success) / 3 - min(success))
    assert v["bootstrap_on_games"]["n_games"] == 3


def test_verdict_needs_every_gate_true_and_refuses_a_repeated_seed(job, tmp_path):
    run = {"seed": 0, "git_commit": "x", "exposure_gap": 0.0, "aggregates": {
        "own_masked": {"success_pct": 100.0, "repeat_rate": 0.0, "ece": 0.0, "auroc": 1.0},
        "own_unmasked": {"fallback_when_off_menu": None, "fallback_when_on_menu": 0.0}},
        "per_game": {p: [{"uid": "g", "won": True}] for p in ("own_masked", "own_unmasked", "teacher_states")}}
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps(run))
    b.write_text(json.dumps({**run, "seed": 1}))
    v = job.aggregate([str(a), str(b)], n_boot=100)
    assert v["passed"] is False  # One gate was not measured, so the verdict cannot pass.
    assert v["not_measured"] == ["Falls back when it should"]
    with pytest.raises(ValueError):
        job.aggregate([str(a), str(a)], n_boot=100)


def test_test_games_are_refused_by_the_job(job, tmp_path):
    listing = tmp_path / "games.txt"
    listing.write_text("valid_unseen/pick_and_place_simple-Pen-None-Desk-1/trial_T1\n")
    with pytest.raises(ValueError):
        job.load_game_list(str(listing))


def test_yaml_has_the_fields_and_names_the_falsification_source():
    data = yaml.safe_load((ROOT / "lab" / "jobs" / "eval-sys1-alone.yaml").read_text())
    for field in ("claim", "falsified_by", "axis", "controls", "metrics", "run", "compute"):
        assert field in data, field
    assert "lib/sys1-acceptance.md" in data["falsified_by"]
