"""The Sys1 interface of `lab/INTERFACES.md`, on StubSys1, and the parts of KevSys1 that need no weights."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sys1.base import FALLBACK_TEXT, QUESTION_TEXT, Sys1, check_options  # noqa: E402
from sys1.stub import StubSys1  # noqa: E402

PROMPTS = ["put a mug on the desk", "heat an egg", "look at a book under the lamp"]
OPTIONS = [["go to desk 1", FALLBACK_TEXT], ["go to fridge 1", "open fridge 1", "go to sinkbasin 1", FALLBACK_TEXT], ["take book 1", "use lamp 1", "go to shelf 1", "go to shelf 2", "go to shelf 3", FALLBACK_TEXT]]


def test_score_gives_one_logit_tensor_per_prompt_and_one_logit_per_option():
    logits = StubSys1().score(PROMPTS, OPTIONS)
    assert [z.shape for z in logits] == [torch.Size([len(o)]) for o in OPTIONS]


def test_score_has_gradients_when_training_and_none_otherwise():
    stub = StubSys1()
    stub.train()
    loss = sum(z.sum() for z in stub.score(PROMPTS, OPTIONS))
    loss.backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in stub.trainable_parameters())
    stub.eval()
    assert not any(z.requires_grad for z in stub.score(PROMPTS, OPTIONS))


def test_probs_are_softmax_of_logits_over_the_fitted_temperature():
    stub = StubSys1()
    stub.temperature = 2.5
    stub.eval()
    assert stub.temperature == 2.5
    for p, z in zip(stub.probs(PROMPTS, OPTIONS), stub.score(PROMPTS, OPTIONS)):
        assert float(p.sum()) == pytest.approx(1)
        assert torch.allclose(p, torch.softmax(z / 2.5, -1))


def test_temperature_reads_one_during_training_and_returns_after():
    stub = StubSys1()
    stub.temperature = 2.5
    stub.train()
    assert stub.temperature == 1.0
    for p, z in zip(stub.probs(PROMPTS, OPTIONS), stub.score(PROMPTS, OPTIONS)):
        assert torch.allclose(p, torch.softmax(z, -1))
    stub.eval()
    assert stub.temperature == 2.5


def test_temperature_must_be_positive():
    with pytest.raises(ValueError):
        StubSys1().temperature = 0.0


def test_option_count_may_change_between_calls():
    stub = StubSys1()
    assert len(stub.score(["a"], [["x", FALLBACK_TEXT]])[0]) == 2
    assert len(stub.score(["a"], [["x", "y", "z", FALLBACK_TEXT]])[0]) == 4


@pytest.mark.parametrize("bad", [[FALLBACK_TEXT], ["x", "y"], [FALLBACK_TEXT, "x"], ["x", "x", FALLBACK_TEXT], []])
def test_option_lists_must_end_with_the_fallback_and_not_repeat(bad):
    with pytest.raises(ValueError):
        check_options(bad)
    with pytest.raises(ValueError):
        StubSys1().score(["a"], [bad])


def test_prompts_and_option_lists_must_pair_up():
    with pytest.raises(ValueError):
        StubSys1().score(["a", "b"], [["x", FALLBACK_TEXT]])


def test_equal_seeds_give_equal_outputs_and_a_save_load_round_trip_holds(tmp_path):
    a, b = StubSys1(seed=3), StubSys1(seed=3)
    a.temperature = 1.5
    assert all(torch.equal(x, y) for x, y in zip(a.score(PROMPTS, OPTIONS), b.score(PROMPTS, OPTIONS)))
    a.save(tmp_path)
    loaded = StubSys1.load(tmp_path, "cpu")
    assert loaded.temperature == 1.5 and not loaded.training
    assert all(torch.equal(x, y) for x, y in zip(a.score(PROMPTS, OPTIONS), loaded.score(PROMPTS, OPTIONS)))


def test_the_base_class_is_abstract():
    with pytest.raises(NotImplementedError):
        Sys1().score(PROMPTS, OPTIONS)


# KevSys1, with a stand-in for kev.model.DecisionModel: the temperature rule, and agreement with what training sees.


class StandInDecisionModel(nn.Module):
    """Calls the real PointerHead as `DecisionModel.forward_batch` does, so the head's own temperature step runs."""

    def __init__(self):
        from kev.model import PointerHead
        super().__init__()
        self.head = PointerHead(8, dp=8)
        self.device = "cpu"
        self.body = nn.Linear(8, 8)

    @property
    def dtype(self):
        return "float32"

    def encode(self, tok, rec, **kw):
        return {"ids": [0] * (len(rec["state"]) + 1), "k": len(rec["questions"][0]["options"]), "tag": len(rec["state"])}

    def forward_batch(self, encs):
        out = []
        for enc in encs:
            torch.manual_seed(enc["tag"])
            h = self.body(torch.randn(enc["k"] + 1, 8))
            out.append([self.head(h[0], h[1:])])
        return out


def stand_in_kev(temperature=2.35, trainable=True):
    from sys1.kev_adapter import KevSys1
    return KevSys1(StandInDecisionModel(), None, SimpleNamespace(), temperature=temperature, trainable=trainable)


def test_kev_adapter_applies_the_temperature_once():
    sys1 = stand_in_kev(temperature=2.0)
    assert sys1.model.head.temperature == 1.0
    assert not sys1.training and sys1.temperature == 2.0
    for p, z in zip(sys1.probs(PROMPTS, OPTIONS), sys1.score(PROMPTS, OPTIONS)):
        assert torch.allclose(p, torch.softmax(z / 2.0, -1))
        head_raw = sys1.model.head.training or sys1.model.head.temperature == 1.0
        assert head_raw


def test_kev_adapter_keeps_batch_order_and_has_gradients_only_when_training():
    sys1 = stand_in_kev()
    logits = sys1.score(PROMPTS, OPTIONS)
    assert [len(z) for z in logits] == [len(o) for o in OPTIONS]
    assert not any(z.requires_grad for z in logits)
    single = sys1.score(PROMPTS[1:2], OPTIONS[1:2])[0]
    assert torch.allclose(single, logits[1])
    sys1.train()
    assert sys1.temperature == 1.0 and all(z.requires_grad for z in sys1.score(PROMPTS, OPTIONS))


def test_a_merged_kev_model_refuses_to_train_and_to_save(tmp_path):
    sys1 = stand_in_kev(trainable=False)
    with pytest.raises(RuntimeError):
        sys1.train()
    with pytest.raises(RuntimeError):
        sys1.save(tmp_path)
    assert sys1.trainable_parameters() == []


def test_bf16_autocast_is_refused_off_cuda():
    from sys1.kev_adapter import KevSys1
    with pytest.raises(ValueError):
        KevSys1(StandInDecisionModel(), None, SimpleNamespace(), autocast_bf16=True)


def test_inference_record_equals_what_training_reads_from_the_export(tmp_path):
    from kev.data import load_records, materialize
    from sys1.kev_adapter import to_record
    from train.export_kev import kev_row, write_jsonl
    record = {"game_id": "g", "split": "train", "step": 0, "task_type": "t"}
    state, options = "a state: with colons\nand lines", ["go to desk 1", "take mug 1", FALLBACK_TEXT]
    path = tmp_path / "rows.jsonl"
    write_jsonl([kev_row(state, options, "take mug 1", record, 0)], path)
    trained = materialize(load_records(path)[0])
    served = to_record(state, options)
    assert trained["state"] == served["state"]
    assert trained["questions"][0]["instr"] == served["questions"][0]["instr"] == QUESTION_TEXT
    assert trained["questions"][0]["options"] == served["questions"][0]["options"]
    assert trained["questions"][0]["label"] == 1


def test_inference_and_training_encode_to_the_same_token_ids_on_the_real_tokenizer():
    from kev.data import materialize
    from kev.model import encode
    from sys1.kev_adapter import to_record
    from train.export_kev import BASE, BASE_REVISION, kev_row
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(BASE, revision=BASE_REVISION, local_files_only=True)
    except OSError:
        pytest.skip("the Qwen3.5-0.8B-Base tokenizer is not in the local cache")
    state = "Your task is to: put a hot mug in shelf 1.\nYou are in the middle of a room."
    options = ["go to shelf 1", "go to countertop 2", FALLBACK_TEXT]
    row = kev_row(state, options, "go to shelf 1", {"game_id": "g", "split": "train", "step": 0, "task_type": "t"}, 0)
    a, b = encode(tok, materialize(row)), encode(tok, to_record(state, options))
    assert a["ids"] == b["ids"] and a["opt_idx"] == b["opt_idx"] and a["decide_idx"] == b["decide_idx"]


def test_real_weights_check_file_is_valid_json_when_present():
    folder = Path(__file__).resolve().parents[1] / "sys1" / "real-weights-check-mac"
    for path in folder.glob("mac-*.json"):
        result = json.loads(path.read_text())
        assert result["seed"] == 0 and "Not an L4 number" in result["scope"]
