"""The menu: command forms, the fit's assert on split, the ranking rules, determinism, and saving."""
import json
import random
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # noqa: E401,E702

import pytest

from menu import MenuModel, class_of, command_form, fit, natural_key

TASK = "pick_heat_then_place_in_recep"


def record(label, task=TASK, split="train", game="train/pick_heat_then_place_in_recep-Apple-None-Fridge-1/trial_1", **extra):
    return {"split": split, "game_id": game, "task_type": task, "step": 3, "expert": label, **extra}


@pytest.mark.parametrize("command, form", [
    ("go to cabinet 12", ("go to", None, "cabinet")),
    ("take apple 3 from fridge 1", ("take", "apple", "fridge")),
    ("move apple 3 to countertop 2", ("move", "apple", "countertop")),
    ("put apple 3 in/on countertop 2", ("put", "apple", "countertop")),
    ("open fridge 1", ("open", None, "fridge")),
    ("close fridge 1", ("close", None, "fridge")),
    ("examine desk 1", ("examine", None, "desk")),
    ("use desklamp 2", ("use", None, "desklamp")),
    ("heat apple 3 with microwave 1", ("heat", "apple", "microwave")),
    ("cool mug 2 with fridge 1", ("cool", "mug", "fridge")),
    ("clean plate 1 with sinkbasin 1", ("clean", "plate", "sinkbasin")),
    ("slice apple 1 with knife 2", ("slice", "apple", "knife")),
    ("look", ("look", None, None)),
    ("inventory", ("inventory", None, None)),
])
def test_command_form_strips_instance_numbers(command, form):
    assert command_form(command) == form


def test_class_of_keeps_every_word_but_the_number():
    assert class_of("alarmclock 3") == "alarmclock"
    assert natural_key("go to cabinet 2") < natural_key("go to cabinet 10")


def test_fit_refuses_a_row_that_is_not_train():
    """The deliberately leaky input: one dev row among training rows stops the fit."""
    for split in ("dev", "dev_heldout_rooms", None):
        with pytest.raises(AssertionError, match="split 'train'"):
            fit([record("go to fridge 1"), record("go to fridge 1", split=split)])


def test_fit_refuses_a_test_game_whatever_its_split_says():
    with pytest.raises(AssertionError, match="test game"):
        fit([record("go to fridge 1", game="valid_unseen/pick_heat_then_place_in_recep-Apple-None-Fridge-10/trial_1")])


def test_fit_counts_forms_within_a_task_type_and_skips_flagged_labels():
    model = fit([record("go to fridge 1"), record("go to fridge 2"), record("go to desk 1", task="pick_two_obj_and_place"), record(None, label_random=True), record("go to desk 1", label_inadmissible=True)])
    assert model.score(TASK, "go to fridge 9") == 2
    assert model.score(TASK, "go to desk 1") == 0
    assert model.score("pick_two_obj_and_place", "go to desk 3") == 1
    assert model.meta["records_used"] == 3 and model.meta["records_skipped_flagged"] == 2


def test_form_rule_ranks_by_kind_object_and_place_with_numbers_stripped():
    records = [record("take apple 1 from fridge 1")] * 5 + [record("go to fridge 1")] * 3 + [record("go to desk 1")]
    model = fit(records)
    admissible = ["go to desk 2", "go to fridge 4", "take apple 7 from fridge 2", "take mug 1 from fridge 2", "look"]
    assert model.menu(TASK, admissible, 3) == ["take apple 7 from fridge 2", "go to fridge 4", "go to desk 2"]


def test_ties_go_to_coarser_counts_and_then_natural_order_whatever_the_input_order():
    model = fit([record("go to cabinet 1")] * 2 + [record("take apple 1 from cabinet 1")] * 4)
    admissible = [f"go to cabinet {i}" for i in (10, 2, 1)] + ["take mug 1 from cabinet 2", "examine cabinet 2", "look"]
    expected = ["go to cabinet 1", "go to cabinet 2", "go to cabinet 10", "take mug 1 from cabinet 2", "examine cabinet 2", "look"]
    for seed in range(5):
        shuffled = list(admissible)
        random.Random(seed).shuffle(shuffled)
        assert model.menu(TASK, shuffled, None) == expected


def test_kind_rule_ranks_by_kind_alone():
    model = fit([record("go to fridge 1")] * 5 + [record("take apple 1 from fridge 1")] * 3, rule="kind")
    admissible = ["take apple 1 from fridge 1", "go to desk 1", "go to cabinet 2", "go to cabinet 10", "look"]
    assert model.menu(TASK, admissible, 3) == ["go to cabinet 2", "go to cabinet 10", "go to desk 1"]
    assert model.score(TASK, "go to anything 1") == 5


def test_k_none_gives_all_admissible_commands_and_k_cuts():
    model = fit([record("go to fridge 1")])
    admissible = ["look", "go to fridge 1", "inventory"]
    assert sorted(model.menu(TASK, admissible, None)) == sorted(admissible)
    assert model.menu(TASK, admissible, 1) == ["go to fridge 1"]
    with pytest.raises(ValueError):
        model.menu(TASK, admissible, 0)


def test_rank_is_one_based_and_none_when_absent():
    model = fit([record("go to fridge 1")])
    assert model.rank(TASK, ["look", "go to fridge 1"], "go to fridge 1") == 1
    assert model.rank(TASK, ["look", "go to fridge 1"], "look") == 2
    assert model.rank(TASK, ["look"], "go to fridge 1") is None
    assert model.rank(TASK, ["look"], None) is None


def test_save_and_load_give_the_same_menus(tmp_path):
    model = fit([record("go to fridge 1"), record("take apple 1 from fridge 1"), record("take apple 2 from fridge 1")], meta={"note": "x"})
    path = tmp_path / "menu.json"
    model.save(path)
    again = MenuModel.load(path)
    admissible = ["go to fridge 1", "take apple 2 from fridge 1", "look", "go to desk 1"]
    assert again.menu(TASK, admissible, None) == model.menu(TASK, admissible, None)
    assert again.rule == "form" and json.loads(path.read_text())["meta"]["note"] == "x"


def test_a_task_type_with_no_fit_is_refused():
    with pytest.raises(ValueError, match="no menu"):
        fit([record("look")]).menu("other", ["look"], 1)
