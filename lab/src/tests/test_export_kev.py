"""The export to Kev's JSONL: labels on hand-built rows, the fallback when off menu, the split assert, and Kev's own readers."""
import gzip
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sys1.base import FALLBACK_TEXT, QUESTION_TEXT  # noqa: E402
from train import export_kev  # noqa: E402
from train.export_kev import QUESTION_ID, check_split, dedupe, export_rows, read_step_records, soft_labels, token_report, write_jsonl  # noqa: E402


class StandInMenu:
    """Commands in the order given, cut to k, as `MenuModel.menu` is specified: k=None gives all."""

    def menu(self, task_type, admissible, k):
        return list(admissible) if k is None else list(admissible)[:k]


def stand_in_render(goal, first_obs, trace, control_state, window=5):
    recent = trace[-window:] if window > 0 else []
    lines = [f"goal={goal}", f"first={first_obs}"] + [f"> {c} -> {o}" for c, o in recent]
    return "\n".join(lines + [f"fallbacks={control_state.fallbacks_in_row}"])


def stand_in_control_state(fallbacks_in_row):
    return SimpleNamespace(fallbacks_in_row=fallbacks_in_row)


def step(step_number, expert, admissible=("go to a", "go to b", "take x"), *, game="pick_and_place/room-1/trial_1", split="train", won=True, **extra):
    return {"split": split, "game_id": game, "task_type": "pick_and_place", "goal": "put x on y", "step": step_number, "obs": f"obs{step_number}",
            "admissible": list(admissible), "expert": expert, "done": False, "won": won, **extra}


def export(records, k=2, **kwargs):
    return export_rows(records, StandInMenu(), k, render=stand_in_render, control_state=stand_in_control_state, **kwargs)


def question(row):
    return row["questions"][QUESTION_ID]


def test_label_is_the_expert_command_when_it_is_on_the_menu():
    rows, counts = export([step(0, "go to b")], k=2)
    assert question(rows[0])["label"] == "go to b"
    assert not rows[0]["_meta"]["off_menu"]
    assert counts["labels_on_menu"] == 1 and counts["labels_fallback"] == 0


def test_options_are_the_menu_then_the_fallback_last():
    rows, _ = export([step(0, "go to a")], k=2)
    assert list(question(rows[0])["criteria"]) == ["go to a", "go to b", FALLBACK_TEXT]
    assert all(v is None for v in question(rows[0])["criteria"].values())
    assert question(rows[0])["instructions"] == QUESTION_TEXT


def test_label_is_the_fallback_when_the_expert_command_is_off_the_menu():
    rows, counts = export([step(0, "take x")], k=2)
    assert question(rows[0])["label"] == FALLBACK_TEXT
    assert rows[0]["_meta"]["off_menu"]
    assert counts["labels_fallback"] == 1


def test_all_admissible_menu_never_labels_the_fallback():
    rows, counts = export([step(0, "take x")], k=None)
    assert question(rows[0])["label"] == "take x"
    assert counts["labels_fallback"] == 0


def test_expert_command_that_is_not_admissible_goes_to_the_fallback_and_is_counted():
    rows, counts = export([step(0, "open z")], k=2)
    assert question(rows[0])["label"] == FALLBACK_TEXT
    assert counts["expert_not_admissible"] == 1


@pytest.mark.parametrize("split", ["dev", "dev_heldout_rooms", "test", None])
def test_split_assert_fires_on_rows_that_are_not_train(split):
    with pytest.raises(AssertionError, match="split must be 'train'"):
        export([step(0, "go to a"), step(1, "go to b", split=split)])


def test_split_assert_fires_when_the_split_field_is_missing():
    record = step(0, "go to a")
    del record["split"]
    with pytest.raises(AssertionError):
        check_split(record)


def test_split_assert_fires_before_any_row_is_made():
    calls = []

    def counting_render(*args):
        calls.append(args)
        return "text"
    with pytest.raises(AssertionError):
        export_rows([step(0, "go to a"), step(1, "go to b", split="dev")], StandInMenu(), 2, render=counting_render, control_state=stand_in_control_state)
    assert not calls


def test_test_games_stop_the_export_whatever_their_split_says(tmp_path):
    with pytest.raises(AssertionError, match="test game"):
        check_split(step(0, "go to a", game="valid_unseen/room-9/trial_1"))
    shard = tmp_path / "s.jsonl"
    shard.write_text(json.dumps(step(0, "go to a", game="valid_unseen/room-9/trial_1")) + "\n")
    with pytest.raises(AssertionError, match="test game"):
        list(read_step_records([shard], select_split="dev"))


def test_a_dev_export_needs_its_split_named_and_keeps_it_in_meta():
    rows, _ = export([step(0, "go to a", split="dev_heldout_rooms")], required_split="dev_heldout_rooms")
    assert rows[0]["_meta"]["split"] == "dev_heldout_rooms"
    with pytest.raises(ValueError):
        export([step(0, "go to a")], required_split="test")


def test_trace_holds_the_commands_and_the_observations_that_followed():
    rows, _ = export([step(0, "go to a"), step(1, "go to b"), step(2, "take x")], trace_window=5)
    assert "first=obs0" in rows[2]["state"]
    assert "> go to a -> obs1" in rows[2]["state"]
    assert "> go to b -> obs2" in rows[2]["state"]
    assert "> " not in rows[0]["state"]


def test_trace_window_keeps_the_last_entries():
    rows, _ = export([step(0, "go to a"), step(1, "go to b"), step(2, "go to a"), step(3, "go to b")], trace_window=1)
    assert "go to b -> obs1" not in rows[3]["state"] and "> go to a -> obs3" in rows[3]["state"]


def test_command_field_overrides_the_expert_in_the_trace_for_recovery_states():
    rows, _ = export([step(0, "go to a", command="go to b"), step(1, "go to b")])
    assert question(rows[0])["label"] == "go to a"
    assert "> go to b -> obs1" in rows[1]["state"]


def test_label_random_rows_stay_out_and_their_commands_stay_in_the_trace():
    rows, counts = export([step(0, None, label_random=True, command="go to b"), step(1, "go to a")])
    assert counts["rows_dropped_label_random"] == 1 and len(rows) == 1
    assert "> go to b -> obs1" in rows[0]["state"]


def test_rows_of_games_the_checker_did_not_mark_won_are_dropped_by_default():
    won = [step(0, "go to a", game="g-won", won=False), step(1, "go to b", game="g-won", won=True)]
    lost = [step(0, "go to a", game="g-lost", won=False), step(1, "go to b", game="g-lost", won=False)]
    rows, counts = export(won + lost)
    assert {r["_meta"]["game_id"] for r in rows} == {"g-won"}
    assert counts["trajectories_not_won"] == 1 and counts["rows_dropped_not_won"] == 2
    rows, _ = export(won + lost, require_won=False)
    assert len(rows) == 4


def test_a_second_expert_seed_of_one_game_starts_a_new_trajectory():
    rows, counts = export([step(0, "go to a"), step(1, "go to b"), step(0, "go to b"), step(1, "go to a")])
    assert counts["trajectories"] == 2
    assert "> go to a -> obs1" in rows[1]["state"] and "> go to b -> obs1" in rows[3]["state"]
    assert len({r["_meta"]["id"] for r in rows}) == 4


def test_interleaved_shards_are_refused():
    with pytest.raises(ValueError, match="starts at step 5"):
        export([step(5, "go to a")])


def test_control_state_counts_fallback_labels_and_restarts_at_handover():
    records = [step(0, "take x"), step(1, "take x"), step(2, "take x"), step(3, "go to a"), step(4, "take x")]
    rows, _ = export(records, k=2, fallbacks_before_handover=2)
    assert [r["state"].rsplit("fallbacks=", 1)[1] for r in rows] == ["0", "1", "0", "1", "0"]


def test_control_state_never_restarts_when_the_hand_over_is_switched_off():
    rows, _ = export([step(0, "take x"), step(1, "take x"), step(2, "take x"), step(3, "go to a"), step(4, "take x")], k=2, fallbacks_before_handover=0)
    assert [r["state"].rsplit("fallbacks=", 1)[1] for r in rows] == ["0", "1", "2", "3", "0"]


def test_trace_window_zero_shows_no_trace_as_the_harness_does():
    rows, _ = export([step(0, "go to a"), step(1, "go to b")], trace_window=0)
    assert "> " not in rows[1]["state"]


def test_exported_state_is_the_text_the_harness_builds_in_a_teacher_run():
    prompt = pytest.importorskip("harness.prompt")
    control = pytest.importorskip("harness.control")
    records = [step(0, "take x"), step(1, "go to a"), step(2, "go to b")]
    rows, _ = export_rows(records, StandInMenu(), 2, trace_window=5, fallbacks_before_handover=2)
    teacher = [prompt.TraceEntry("teacher", "take x", "obs1"), prompt.TraceEntry("teacher", "go to a", "obs2")]
    assert rows[2]["state"] == prompt.render_prompt("put x on y", "obs0", teacher, control.ControlState(), window=5)
    shown = control.ControlState(fallbacks_in_row=1)
    assert rows[1]["state"] == prompt.render_prompt("put x on y", "obs0", teacher[:1], shown, window=5)


def test_dedupe_removes_only_identical_state_options_and_label():
    a = step(0, "go to a")
    rows, _ = export([a, step(0, "go to a"), step(0, "go to b")])
    kept, counts = dedupe(rows)
    assert counts == {"rows_before_dedupe": 3, "rows_after_dedupe": 2, "duplicates_removed": 1, "states_with_more_than_one_label": 1}
    assert sorted(question(r)["label"] for r in kept) == ["go to a", "go to b"]
    assert [r["_meta"]["copies"] for r in kept] == [2, 1]


def test_dedupe_ignores_option_order():
    rows, _ = export([step(0, "go to a", admissible=("go to a", "go to b")), step(0, "go to a", admissible=("go to b", "go to a"))], k=None)
    kept, counts = dedupe(rows)
    assert len(kept) == 1 and counts["duplicates_removed"] == 1


def test_soft_labels_give_each_label_its_share_and_keep_the_majority_as_label():
    rows, _ = export([step(0, "go to a"), step(0, "go to a"), step(0, "go to b"), step(0, "take x")], k=2)
    merged, counts = soft_labels(rows)
    assert len(merged) == 1 and counts["rows_with_soft_target"] == 1
    q = question(merged[0])
    assert q["label"] == "go to a"
    assert q["target"] == {"go to a": 0.5, "go to b": 0.25, FALLBACK_TEXT: 0.25}


def test_soft_labels_leave_states_with_one_label_hard_and_break_ties_by_menu_order():
    rows, _ = export([step(0, "go to a"), step(0, "go to a")], k=2)
    merged, counts = soft_labels(rows)
    assert "target" not in question(merged[0]) and counts["rows_with_soft_target"] == 0
    rows, _ = export([step(0, "go to b"), step(0, "go to a")], k=2)
    merged, _ = soft_labels(rows)
    assert question(merged[0])["label"] == "go to a"


def kev_read(rows, tmp_path):
    from kev.data import load_records
    path = tmp_path / "rows.jsonl"
    write_jsonl(rows, path)
    return load_records(path)


def test_kev_reads_the_export_and_keeps_the_label_under_its_option_shuffle(tmp_path):
    from kev.data import augment, materialize
    rows, _ = export([step(0, "go to b"), step(1, "take x")], k=2)
    for record, expected in zip(kev_read(rows, tmp_path), ["go to b", FALLBACK_TEXT]):
        assert record["_meta"]["split"] == "train"
        for seed in range(5):
            shuffled = materialize(augment(record, random.Random(seed), p_none=0, p_none_distract=0, p_distract=0))
            q = shuffled["questions"][0]
            assert q["options"][q["label"]] == expected
            assert sorted(q["options"]) == sorted(record["questions"][QUESTION_ID]["criteria"])
            assert q["instr"] == QUESTION_TEXT


def test_kev_default_augmentation_would_delete_the_labelled_option(tmp_path):
    """Why the job passes --p_none 0: this pins Kev's behaviour, and the job's own test pins the flags."""
    from kev.data import augment
    rows, _ = export([step(0, "go to a")], k=None)
    record = kev_read(rows, tmp_path)[0]
    seen = [augment(record, random.Random(seed))["questions"][QUESTION_ID] for seed in range(200)]
    assert any("go to a" not in q["criteria"] for q in seen)
    assert any(FALLBACK_TEXT in q["criteria"] and q["label"] not in ("go to a", FALLBACK_TEXT) for q in seen)


def test_kev_trainer_takes_the_soft_target_in_its_loss(tmp_path):
    from kev.data import augment, materialize
    from kev.train import question_loss
    rows, _ = export([step(0, "go to a"), step(0, "go to a"), step(0, "go to b"), step(0, "take x")], k=2)
    merged, _ = soft_labels(rows)
    record = kev_read(merged, tmp_path)[0]
    for seed in range(5):
        shuffled = materialize(augment(record, random.Random(seed)))   # default probabilities: soft-target rows are only shuffled (kev/data.py:330)
        q = shuffled["questions"][0]
        assert len(q["options"]) == 3
        shares = {"go to a": 0.5, "go to b": 0.25, FALLBACK_TEXT: 0.25}
        assert q["target"] == pytest.approx([shares[o] for o in q["options"]])
        z = torch.tensor([1.0, -0.5, 0.3])
        loss = question_loss(z, q, "cpu", 0.0)
        assert loss.item() == pytest.approx(-(torch.tensor(q["target"]) * torch.log_softmax(z, -1)).sum().item())
        assert loss.item() != pytest.approx(torch.nn.functional.cross_entropy(z[None], torch.tensor([q["label"]])).item())


class WordTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[hash(w) % 1000 for w in text.split()])

    def convert_tokens_to_ids(self, token):
        return 1


def test_token_report_gives_the_max_state_that_keeps_every_row():
    short = {"state": "a b c", "questions": {QUESTION_ID: {"type": "choice", "instructions": "which", "criteria": {"x": None, FALLBACK_TEXT: None}, "label": "x", "src": "t"}}}
    long = {**short, "state": " ".join(["w"] * 500)}
    report = token_report([short, long], WordTokenizer())
    assert report["state_tokens_max"] == 501
    assert report["max_state_needed"] == 501
    assert report["rows_over_default_max_state"] == 1
    assert report["rows_over_row_limit_at_max_state_needed"] == 0
    assert token_report([short], WordTokenizer())["max_state_needed"] == 384


def test_command_line_writes_gzip_and_a_report_with_hashes(tmp_path, monkeypatch):
    monkeypatch.setattr(export_kev, "load_menu", lambda path: StandInMenu())
    monkeypatch.setattr(export_kev, "load_render_prompt", lambda: stand_in_render)
    monkeypatch.setattr(export_kev, "default_control_state", stand_in_control_state)
    monkeypatch.setattr(export_kev.export_rows, "__kwdefaults__", {**export_kev.export_rows.__kwdefaults__, "control_state": stand_in_control_state})
    shard = tmp_path / "shard.jsonl.gz"
    with gzip.open(shard, "wt") as f:
        for record in [step(0, "go to a"), step(1, "take x"), step(0, "go to a", split="dev", game="dev-game")]:
            f.write(json.dumps(record) + "\n")
    menu_file = tmp_path / "menu.json"
    menu_file.write_text("{}")
    out = tmp_path / "train.jsonl.gz"
    argv = ["--records", str(tmp_path / "shard*.gz"), "--menu", str(menu_file), "--k", "2", "--out", str(out), "--select-split", "--no-token-report", "--dedupe", "hard"]
    export_kev.main(argv)
    with gzip.open(out, "rt") as f:
        written = [json.loads(line) for line in f]
    assert [question(r)["label"] for r in written] == ["go to a", FALLBACK_TEXT]
    report = json.loads((tmp_path / "train.report.json").read_text())
    assert report["rows_written"] == 2 and report["counts"]["records_skipped_other_split"] == 1
    assert report["fallback_label_share"] == 0.5 and report["shards"][0]["sha256"] == export_kev.sha256(shard)
    with pytest.raises(AssertionError):   # without --select-split the dev row in the shard stops the export
        export_kev.main(argv[:8] + ["--no-token-report"])
