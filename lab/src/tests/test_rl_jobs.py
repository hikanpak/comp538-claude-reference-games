"""The RL jobs as a Colab kernel runs them, on a stub Sys1 and real training games, plus their yaml and their Mac-side scripts. `COLAB_JOB` points at a recorder, so no VM can start."""
import json
import os
import pathlib
import stat
import subprocess
import sys
import tarfile

import pytest
import yaml

SRC = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from alf.env import list_games  # noqa: E402
from sys1.base import FALLBACK_TEXT  # noqa: E402
from train.export_kev import kev_row, write_jsonl  # noqa: E402
from train.ppo import read_kev_rows  # noqa: E402

JOBS = SRC.parent / "jobs"
TOPICS = ["arm-rl-scheduled", "arm-rl-model-in-control"]
MENU_STAND_IN = """class MenuModel:
    \"\"\"A stand-in for lab/src/menu.py's class, in this test's PYTHONPATH only.\"\"\"
    @classmethod
    def load(cls, path):
        return cls()
    def menu(self, task_type, admissible, k):
        return sorted(admissible)[:k]
"""


@pytest.fixture
def inputs(tmp_path):
    games = list_games("train", limit=3)
    (tmp_path / "train_games.txt").write_text(games[0].uid + "\n" + games[1].uid + "\n")
    (tmp_path / "dev_games.txt").write_text(games[2].uid + "\n")
    rows = lambda split, n: [kev_row(f"state {i}", [f"go to desk {i}", "take mug 1", FALLBACK_TEXT], "take mug 1", {"game_id": f"train/room/{i % 2}", "split": split, "step": i, "task_type": "t"}, 0) for i in range(n)]  # noqa: E731
    write_jsonl(rows("train", 4), tmp_path / "train.jsonl.gz")
    write_jsonl(rows("dev_heldout_rooms", 4), tmp_path / "dev.jsonl.gz")
    (tmp_path / "menu.json").write_text("{}\n")
    stand_ins = tmp_path / "stand_ins"
    stand_ins.mkdir()
    (stand_ins / "menu.py").write_text(MENU_STAND_IN)
    return tmp_path


def run_job(topic, work, **env):
    full = {"PATH": os.environ["PATH"], "HOME": os.environ["HOME"], "SEED": "3", "MODE": "smoke", "WORK": str(work), "CODE_DIR": str(SRC), "OUT_DIR": str(work / "out"), "INSTALL": "0", "PYTHON": sys.executable, "PYTHONPATH": str(work / "stand_ins"),
            "DEVICE": "cpu", "INIT_FROM": "stub", "STEP_BUDGET": "12", "MAX_STEPS": "5", "GAMES_PER_COLLECT": "2", "BATCH_GAMES": "2", "TRANSITIONS_PER_ITER": "4", "MINIBATCH": "2", "MICRO_BATCH": "2", "KL_ROWS": "1", "KL_POOL_ROWS": "2", "EVAL_BATCH": "4", "BATCH": "2", "TRAIN_ROWS_PER_NEW_ROW": "0.5", "K": "4", "LR": "0.01", **env}
    return subprocess.run([sys.executable, str(JOBS / f"{topic}.py")], capture_output=True, text=True, env=full, cwd=work)


def stem_of(topic):
    return "seed3-R3" if topic == "arm-rl-model-in-control" else "seed3-ce"


@pytest.mark.parametrize("topic", TOPICS)
def test_job_runs_from_environment_variables_alone_and_writes_the_result(topic, inputs):
    done = run_job(topic, inputs)
    assert done.returncode == 0, done.stdout[-1500:] + done.stderr[-1500:]
    result = json.loads((inputs / "out" / f"{stem_of(topic)}.json").read_text())
    for key in ("seed", "git_commit", "git_dirty", "environment", "hardware", "precision", "config", "games_played", "steps", "steps_per_second", "peak_memory", "learning_curve", "per_game", "aggregates", "fallback_rule", "temperature", "completed", "inputs", "menu_module_file"):
        assert key in result, key
    assert result["topic"] == topic and result["seed"] == 3 and result["completed"] and result["config"]["tracking"] == "always"
    assert result["menu_module_file"].endswith("stand_ins/menu.py")   # a stand-in shows in the result
    assert result["inputs"]["train_games"] and result["inputs"]["menu"]
    assert (inputs / "out" / f"{stem_of(topic)}.tar").stat().st_size > 0 and (inputs / "out" / f"{stem_of(topic)}-curve.json").exists()
    assert "train/" not in result["aggregates"]   # plain metrics, no paths
    for row in read_kev_rows(inputs / "train.jsonl.gz", ("train",)):
        assert row["_meta"]["split"] == "train"


@pytest.mark.parametrize("topic", TOPICS)
def test_job_stops_and_a_second_kernel_resumes_from_the_pulled_tar(topic, inputs):
    done = run_job(topic, inputs, STEP_BUDGET="1000000", STOP_AFTER_ITERATIONS="1")
    assert done.returncode == 0, done.stdout[-1500:] + done.stderr[-1500:]
    stem = stem_of(topic)
    first = json.loads((inputs / "out" / f"{stem}.json").read_text())
    assert not first["completed"] and first["stopped_because"] == "stop_after_iterations"
    # The VM dies. Its out folder goes, and the Mac uploads the state tar it pulled, as resume.tar.
    (inputs / "out" / f"{stem}-state.tar").rename(inputs / "resume.tar")
    subprocess.run(["rm", "-rf", str(inputs / "out")], check=True)
    done = run_job(topic, inputs, STEP_BUDGET="1000000", STOP_AFTER_ITERATIONS="1")
    assert done.returncode == 0, done.stdout[-1500:] + done.stderr[-1500:]
    assert "resumed at env step" in done.stdout
    second = json.loads((inputs / "out" / f"{stem}.json").read_text())
    assert second["iterations_run"] == 2 and second["games_played"] > first["games_played"] and second["counts"]["resumes"] == 1
    assert [i["iteration"] for i in second["iterations"]] == [0, 1]


@pytest.mark.parametrize("topic", TOPICS)
def test_a_named_resume_tar_that_is_missing_stops_the_job(topic, inputs):
    done = run_job(topic, inputs, RESUME_TAR=str(inputs / "no-such-state.tar"))
    assert done.returncode != 0 and "RESUME_TAR" in done.stderr and not (inputs / "out" / f"{stem_of(topic)}.json").exists()


def test_job_refuses_a_game_in_both_lists(inputs):
    (inputs / "dev_games.txt").write_text((inputs / "train_games.txt").read_text())
    done = run_job("arm-rl-model-in-control", inputs)
    assert done.returncode != 0 and "both the training list and the dev list" in done.stdout


def test_job_refuses_a_test_game(inputs):
    (inputs / "dev_games.txt").write_text("valid_unseen/look_at_obj_in_light-AlarmClock-None-DeskLamp-301/trial_T1\n")
    done = run_job("arm-rl-model-in-control", inputs)
    assert done.returncode != 0 and "is a test game" in done.stdout


def test_job_refuses_dev_games_as_training_games(inputs):
    (inputs / "train_games.txt").write_text(list_games("valid_seen", limit=1)[0].uid + "\n")
    done = run_job("arm-rl-model-in-control", inputs)
    assert done.returncode != 0 and "training games only" in done.stdout


def test_job_stops_on_a_missing_input(inputs):
    (inputs / "menu.json").unlink()
    done = run_job("arm-rl-scheduled", inputs)
    assert done.returncode != 0 and "missing inputs" in done.stderr


def test_full_mode_needs_a_budget_and_the_classification_weights(inputs):
    done = run_job("arm-rl-model-in-control", inputs, MODE="full", STEP_BUDGET="")
    assert done.returncode != 0 and "STEP_BUDGET" in done.stdout
    done = run_job("arm-rl-model-in-control", inputs, MODE="full", STEP_BUDGET="5", INIT_FROM="jaredpalmer/kev-0.8b")
    assert done.returncode != 0 and "classification arm" in done.stdout
    env = {"PATH": os.environ["PATH"], "HOME": str(inputs), "SEED": "3", "WORK": str(inputs), "CODE_DIR": str(SRC), "OUT_DIR": str(inputs / "out"), "INSTALL": "0"}
    done = subprocess.run([sys.executable, str(JOBS / "arm-rl-scheduled.py")], capture_output=True, text=True, env=env, cwd=inputs)
    assert done.returncode != 0 and "classification arm" in done.stderr


def test_dry_run_prints_the_plan_and_reads_the_init_tar(inputs):
    ckpt = inputs / "ckpt"
    ckpt.mkdir()
    (ckpt / "head.pt").write_text("x")
    with tarfile.open(inputs / "init.tar", "w") as tar:
        tar.add(ckpt, arcname="checkpoint")
    env = {"PATH": os.environ["PATH"], "HOME": str(inputs), "SEED": "3", "WORK": str(inputs), "CODE_DIR": str(SRC), "OUT_DIR": str(inputs / "out"), "INSTALL": "0", "DRY_RUN": "1"}
    done = subprocess.run([sys.executable, str(JOBS / "arm-rl-scheduled.py")], capture_output=True, text=True, env=env, cwd=inputs)
    assert done.returncode == 0, done.stderr
    plan = json.loads(done.stdout.strip().splitlines()[-1])
    assert plan["trainer"] == "train.dagger" and plan["init_from"].endswith("init/checkpoint") and plan["resume"] == "" and (inputs / "init" / "checkpoint" / "head.pt").exists()


@pytest.mark.parametrize("topic", TOPICS)
def test_kernel_imports_only_the_standard_library(topic):
    source = (JOBS / f"{topic}.py").read_text()
    imported = {line.split()[1].split(".")[0] for line in source.splitlines() if line.startswith(("import ", "from "))}
    assert imported <= {"json", "os", "shutil", "subprocess", "sys", "tarfile", "time", "pathlib"}
    assert source.rstrip().endswith("main()")   # falls off its last line, because colab_job.sh appends the end token


@pytest.mark.parametrize("topic", TOPICS)
def test_yaml_has_every_field_and_points_at_the_acceptance_file(topic):
    doc = yaml.safe_load((JOBS / f"{topic}.yaml").read_text())
    assert {"claim", "falsified_by", "axis", "controls", "metrics", "run", "compute"} <= set(doc)
    assert doc["topic"] == topic and "lib/sys1-acceptance.md" in doc["falsified_by"]
    assert doc["compute"]["cluster"] == "colab" and doc["compute"]["charged_to"] == "sys1"
    assert (SRC.parent.parent / "lib" / "sys1-acceptance.md").exists()


def driver_env(tmp_path, recorder, **extra):
    return {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "COLAB_JOB": str(recorder), "TRAIN_JSONL": str(tmp_path / "train.jsonl.gz"), "DEV_JSONL": str(tmp_path / "dev.jsonl.gz"), "MENU": str(tmp_path / "menu.json"), "TRAIN_GAMES": str(tmp_path / "train_games.txt"), "DEV_GAMES": str(tmp_path / "dev_games.txt"),
            "INIT_TAR": str(tmp_path / "init.tar"), "RESULTS_DIR": str(tmp_path / "results"), "PACK_DIR": str(tmp_path / "pack"), "PYTHON": sys.executable, **extra}


@pytest.fixture
def driver(inputs):
    (inputs / "init.tar").write_text("weights")
    recorder = inputs / "recorder.sh"
    recorder.write_text('#!/bin/sh\nfor a in "$@"; do printf "%s\\n" "$a"; done\n')
    recorder.chmod(recorder.stat().st_mode | stat.S_IEXEC)
    return inputs, recorder


@pytest.mark.parametrize("topic,stem", [("arm-rl-scheduled", "seed0-ce"), ("arm-rl-model-in-control", "seed0-R3")])
def test_mac_script_speedtest_uploads_inputs_and_starts_from_classification_weights(driver, topic, stem):
    work, recorder = driver
    done = subprocess.run(["bash", str(JOBS / f"{topic}.sh"), "0", "speedtest"], capture_output=True, text=True, env=driver_env(work, recorder))
    assert done.returncode == 0, done.stderr
    args = done.stdout.splitlines()
    assert args[args.index("-c") + 1] == "sys1" and args[args.index("-g") + 1] == "L4" and args[args.index("-s") + 1].endswith(f"{topic}.py")
    assert f"{work}/init.tar:/content/init.tar" in args and f"{work}/menu.json:/content/menu.json" in args and f"SEED=0" in args and "MODE=speedtest" in args and f"STEM={stem}" in args
    assert f"/content/out/{stem}.json:{work}/results/speedtest/{stem}.json" in args
    assert any(a.startswith("GIT_COMMIT=") for a in args)
    with tarfile.open(work / "pack" / f"code-{topic}-speedtest-{stem}.tar.gz") as tar:
        names = tar.getnames()
    assert "train/ppo.py" in names and "train/dagger.py" in names and "sys1/value_head.py" in names and "harness/episode.py" in names and "alf/env.py" in names and "menu.py" in names and "requirements.txt" in names
    assert not any("__pycache__" in n for n in names)


@pytest.mark.parametrize("topic", TOPICS)
def test_mac_script_full_needs_a_budget_and_clean_code_and_pulls_state(driver, topic):
    work, recorder = driver
    env = driver_env(work, recorder)
    no_budget = subprocess.run(["bash", str(JOBS / f"{topic}.sh"), "0", "full"], capture_output=True, text=True, env=env)
    assert no_budget.returncode == 2 and "STEP_BUDGET" in no_budget.stderr
    done = subprocess.run(["bash", str(JOBS / f"{topic}.sh"), "0", "full"], capture_output=True, text=True, env={**env, "STEP_BUDGET": "100", "ALLOW_DIRTY": "1"})
    assert done.returncode == 0, done.stderr
    args = done.stdout.splitlines()
    assert "STEP_BUDGET=100" in args and any(a.endswith("-state.tar:300") for a in args) and any(a.endswith("-curve.json:300") for a in args)
    assert int(args[args.index("-t") + 1]) <= 28800


@pytest.mark.parametrize("topic,stem", [("arm-rl-scheduled", "seed0-ce"), ("arm-rl-model-in-control", "seed0-R3")])
def test_mac_script_uploads_a_pulled_state_for_a_resume(driver, topic, stem):
    work, recorder = driver
    state = work / "results" / "weights" / f"{stem}-state.tar"
    state.parent.mkdir(parents=True)
    state.write_text("state")
    env = driver_env(work, recorder, ALLOW_DIRTY="1", STEP_BUDGET="100")
    args = subprocess.run(["bash", str(JOBS / f"{topic}.sh"), "0", "full"], capture_output=True, text=True, env=env, check=True).stdout.splitlines()
    assert f"{state}:/content/resume.tar" in args and not any(a.endswith(":/content/init.tar") for a in args)
    args = subprocess.run(["bash", str(JOBS / f"{topic}.sh"), "0", "full"], capture_output=True, text=True, env={**env, "FRESH": "1"}, check=True).stdout.splitlines()
    assert not any(a.endswith(":/content/resume.tar") for a in args) and any(a.endswith(":/content/init.tar") for a in args)


def test_mac_script_passes_the_reward_and_refuses_another(driver):
    work, recorder = driver
    env = driver_env(work, recorder, REWARD="R4")
    args = subprocess.run(["bash", str(JOBS / "arm-rl-model-in-control.sh"), "1", "speedtest"], capture_output=True, text=True, env=env, check=True).stdout.splitlines()
    assert "REWARD=R4" in args and "STEM=seed1-R4" in args
    done = subprocess.run(["bash", str(JOBS / "arm-rl-model-in-control.sh"), "1", "speedtest"], capture_output=True, text=True, env={**env, "REWARD": "R2"})
    assert done.returncode == 2 and "R3 or R4" in done.stderr
