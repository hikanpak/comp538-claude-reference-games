"""The arm-classification job with no weights: its checks on the data, the kev.train command it builds, the temperature refit, and the partial checkpoint hook."""
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

LAB_SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_SRC))

from sys1.base import FALLBACK_TEXT  # noqa: E402
from sys1.stub import StubSys1  # noqa: E402
from train.export_kev import QUESTION_ID, kev_row, write_jsonl  # noqa: E402
from train.kev_train_runner import install_partial_checkpoints  # noqa: E402
from train.refit_temperature import fit, per_game, read_rows, score_rows  # noqa: E402

JOB = LAB_SRC.parent / "jobs" / "arm-classification.py"


def rows(split, n=6, game="pick_and_place/room-1/trial_1"):
    return [kev_row(f"state {i}", [f"go to desk {i}", "take mug 1", FALLBACK_TEXT], "take mug 1", {"game_id": game, "split": split, "step": i, "task_type": "t"}, 0) for i in range(n)]


def run_job(tmp_path, train_rows=None, dev_rows=None, **env):
    train, dev = tmp_path / "train.jsonl.gz", tmp_path / "dev.jsonl.gz"
    write_jsonl(rows("train") if train_rows is None else train_rows, train)
    write_jsonl(rows("dev_heldout_rooms") if dev_rows is None else dev_rows, dev)
    full = {"PATH": os.environ["PATH"], "SEED": "7", "MAX_STATE": "384", "WORK": str(tmp_path), "CODE_DIR": str(LAB_SRC), "OUT_DIR": str(tmp_path / "out"), "DRY_RUN": "1", "INSTALL": "0", **env}
    return subprocess.run([sys.executable, str(JOB)], capture_output=True, text=True, env=full, cwd=tmp_path)


def dry_run_args(tmp_path, **env):
    done = run_job(tmp_path, **env)
    assert done.returncode == 0, done.stderr
    line = [x for x in done.stdout.splitlines() if x.startswith("{")][-1]
    return json.loads(line)


def value(args, flag):
    return args[args.index(flag) + 1]


def test_job_runs_kev_train_from_the_released_checkpoint_with_the_seed(tmp_path):
    args = dry_run_args(tmp_path)["kev_train_args"]
    assert value(args, "--init_from").startswith("jaredpalmer/kev-0.8b")
    assert value(args, "--base") == "Qwen/Qwen3.5-0.8B-Base"
    assert value(args, "--seed") == "7" and value(args, "--epochs") == "1"
    assert value(args, "--max_state") == "384"
    assert value(args, "--data").endswith("seed7-train.jsonl")


def test_job_turns_off_kev_augmentation(tmp_path):
    args = dry_run_args(tmp_path)["kev_train_args"]
    for flag in ("--p_none", "--p_none_distract", "--p_distract", "--p_none_pair"):
        assert value(args, flag) == "0"


def test_arguments_arrive_by_environment_variables(tmp_path):
    args = dry_run_args(tmp_path, EPOCHS="2", LR="1e-5", BATCH="4", ACCUM="2", MAX_STATE="900", SEED="3")["kev_train_args"]
    assert (value(args, "--epochs"), value(args, "--lr"), value(args, "--batch"), value(args, "--accum"), value(args, "--max_state"), value(args, "--seed")) == ("2", "1e-5", "4", "2", "900", "3")


def test_job_refuses_to_start_without_seed_or_max_state(tmp_path):
    for missing in ("SEED", "MAX_STATE"):
        env = {"PATH": os.environ["PATH"], "SEED": "7", "MAX_STATE": "384", "WORK": str(tmp_path), "DRY_RUN": "1"}
        del env[missing]
        done = subprocess.run([sys.executable, str(JOB)], capture_output=True, text=True, env=env, cwd=tmp_path)
        assert done.returncode != 0 and missing in done.stderr


def test_job_refuses_non_train_rows_in_the_training_file(tmp_path):
    done = run_job(tmp_path, train_rows=rows("train") + rows("dev", 1))
    assert done.returncode != 0 and "split must be one of ('train',)" in done.stderr


def test_job_refuses_train_rows_in_the_dev_file(tmp_path):
    done = run_job(tmp_path, dev_rows=rows("train", 1))
    assert done.returncode != 0 and "split must be one of ('dev', 'dev_heldout_rooms')" in done.stderr


def test_job_refuses_test_games(tmp_path):
    done = run_job(tmp_path, dev_rows=rows("dev", 1, game="valid_unseen/room-1/trial_1"))
    assert done.returncode != 0 and "test game" in done.stderr


def test_speed_test_mode_samples_its_rows_and_the_sample_follows_the_seed(tmp_path):
    a = dry_run_args(tmp_path, MODE="speedtest", SPEED_ROWS="4", train_rows=None)
    assert a["train_rows"] == 4 and a["dev_rows"] == 4
    first = (tmp_path / "seed7-train.jsonl").read_text()
    (tmp_path / "seed7-train.jsonl").unlink()
    dry_run_args(tmp_path, MODE="speedtest", SPEED_ROWS="4")
    assert (tmp_path / "seed7-train.jsonl").read_text() == first
    dry_run_args(tmp_path, MODE="speedtest", SPEED_ROWS="4", SEED="8")
    assert (tmp_path / "seed8-train.jsonl").read_text() != first


def test_full_mode_uses_every_row(tmp_path):
    assert dry_run_args(tmp_path, MODE="full")["train_rows"] == 6


def test_unknown_mode_is_refused(tmp_path):
    done = run_job(tmp_path, MODE="both")
    assert done.returncode != 0 and "MODE must be" in done.stderr


# The temperature refit.


def kev_metric_rows(n, right_share, scale, seed=0, games=10):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        k = 4
        label = i % k
        z = rng.normal(0, 0.1, k)
        top = label if rng.random() < right_share else (label + 1) % k
        z[top] += scale
        p = np.exp(z) / np.exp(z).sum()
        out.append({"p": p.tolist(), "logits": z.tolist(), "label": label, "type": "choice", "keys": list("abcd"), "task": "t", "source": "alfworld_expert", "group": f"g{i % games}", "variant": "clean", "inference_temperature": 1.0})
    return out


def test_refit_lowers_confidence_of_an_overconfident_model():
    fitted = fit(kev_metric_rows(600, 0.6, 6.0), seed=0)
    assert fitted["temperature"] > 1.5
    assert fitted["fitted"]["ece"] < fitted["raw"]["ece"]
    assert fitted["fitted"]["acc"] == fitted["raw"]["acc"]
    assert "cross_validation" in fitted and fitted["n"] == 600 and not fitted["at_grid_edge"]


def test_refit_flags_a_temperature_at_the_edge_of_the_grid():
    assert fit(kev_metric_rows(200, 0.25, 40.0), seed=0)["at_grid_edge"]


def test_refit_skips_the_out_of_fold_estimate_with_too_few_games():
    assert "cross_validation" not in fit(kev_metric_rows(40, 0.8, 2.0, games=3), seed=0)


def test_refit_reads_dev_rows_only(tmp_path):
    path = tmp_path / "dev.jsonl"
    write_jsonl(rows("train", 2), path)
    with pytest.raises(AssertionError, match="dev rows only"):
        read_rows(path)
    write_jsonl(rows("dev", 5), path)
    assert len(read_rows(path)) == 5 and len(read_rows(path, limit=2)) == 2


def test_scoring_keeps_row_order_and_gives_per_game_step_match(tmp_path):
    stub = StubSys1()
    stub.eval()
    source = rows("dev", 5)
    scored, seconds = score_rows(stub, source, batch=2)
    assert seconds > 0 and len(scored) == 5
    for row, out in zip(source, scored):
        options = list(row["questions"][QUESTION_ID]["criteria"])
        direct = stub.score([row["state"]], [options])[0]
        assert out["logits"] == pytest.approx(direct.tolist())
        assert out["keys"][out["label"]] == "take mug 1"
    games = per_game(scored, 1.0)
    assert games["pick_and_place/room-1/trial_1"]["rows"] == 5
    assert 0 <= games["pick_and_place/room-1/trial_1"]["step_match"] <= 1
    stub.train()
    with pytest.raises(RuntimeError):
        score_rows(stub, source)


# The partial checkpoint hook. A LoRA run of kev.train saves nothing until its end, so this is what the Mac can pull during a run.


class TinyLm(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(2))

    def save_pretrained(self, path):
        (Path(path) / "adapter_config.json").write_text("{}")
        torch.save(self.state_dict(), Path(path) / "adapter_model.pt")


class TinyDecisionModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.lm, self.head = TinyLm(), nn.Linear(2, 2)


def test_partial_checkpoint_is_written_between_optimizer_steps(tmp_path):
    from kev.checkpoint import Meta, read_meta
    fake_kev_train = SimpleNamespace(DecisionModel=TinyDecisionModel, Meta=Meta)
    tar_path = tmp_path / "out" / "partial.tar"
    handle = install_partial_checkpoints(fake_kev_train, tar_path, every_seconds=0.0)
    try:
        model = fake_kev_train.DecisionModel("base", None, "cpu")
        fake_kev_train.Meta(base="b", lora=16, head_dim=256)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
        loss = model.lm.w.sum()
        loss.backward()
        optimizer.step()
    finally:
        handle.remove()
    assert tar_path.exists() and not (tmp_path / "out" / "partial.tar.tmp").exists()
    with tarfile.open(tar_path) as tar:
        tar.extractall(tmp_path / "x", filter="data")
    meta = read_meta(tmp_path / "x" / "checkpoint")
    assert meta.temperature == 1.0 and meta.base == "b" and meta.extra["partial"]["optimizer_steps"] == 1
    assert set(meta.head) == {"weight", "bias"}


def test_partial_checkpoint_waits_for_its_interval(tmp_path):
    from kev.checkpoint import Meta
    fake_kev_train = SimpleNamespace(DecisionModel=TinyDecisionModel, Meta=Meta)
    tar_path = tmp_path / "partial.tar"
    handle = install_partial_checkpoints(fake_kev_train, tar_path, every_seconds=3600.0)
    try:
        model = fake_kev_train.DecisionModel("base", None, "cpu")
        fake_kev_train.Meta(base="b")
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
        model.lm.w.sum().backward()
        optimizer.step()
    finally:
        handle.remove()
    assert not tar_path.exists()
