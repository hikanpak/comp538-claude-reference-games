"""The two RL arms on a stub Sys1 and real ALFWorld training games, with hand computations for the arithmetic. No weights are read, no VM starts, and `valid_unseen` is never opened."""
import copy
import math
import pathlib
import sys
import tarfile
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from test_harness_standins import StandInMenu  # noqa: E402

from alf.env import list_games  # noqa: E402
from harness.control import ControlState, HarnessConfig  # noqa: E402
from harness.episode import EpisodeLog, StepLog, run_episodes  # noqa: E402
from harness.reward import R3, step_reward, with_reward  # noqa: E402
from sys1.base import FALLBACK_TEXT  # noqa: E402
from sys1.stub import StubSys1  # noqa: E402
from sys1.value_head import ValueHead, decide_dim, dropout_off, score_and_decide  # noqa: E402
from train.dagger import LabelBuffer, ScheduledTrainer, flag_of, reward_at, row_loss, teacher_prob  # noqa: E402
from train.ppo import (FALLBACK_RULE, STAND_IN_NAME, Config, PPOTrainer, TeacherStandIn, ValueCollector, build_transitions, cosine_factor, harness_config, kl_to_reference, labelled_row, log_prob_and_entropy, normalize_weights,  # noqa: E402
                       params_sha256, policy_loss, ratio_mask, read_kev_rows, returns_and_advantages, teacher_stand_in, value_loss)
from train.export_kev import kev_row, write_jsonl  # noqa: E402

MENU = StandInMenu()
GAMES = list_games("train", limit=12)
DEV_GAME = list_games("train", limit=13)[12:]


def config(**kw):
    base = dict(seed=3, mode="smoke", device="cpu", step_budget=24, max_steps=6, games_per_collect=2, batch_games=2, transitions_per_iter=6, minibatch=2, micro_batch=2, kl_rows=2, kl_pool_rows=4, eval_batch=4, lr=0.01, lr_warmup=0, dev_every_steps=12, out_dir="unused", k=4, checkpoint_every_min=1e6, batch=2, train_rows_per_new_row=0.5)
    return Config(**{**base, **kw}).resolve()


def core_rows(n=6):
    return [kev_row(f"Task: put a mug on a desk. State {i}", [f"go to desk {i}", "take mug 1", FALLBACK_TEXT], "take mug 1", {"game_id": f"train/x/{i}", "split": "train", "step": i, "task_type": "t"}, 0) for i in range(n)]


def dev_rows(n=4):
    return [kev_row(f"state {i}", [f"go to desk {i}", "take mug 1", FALLBACK_TEXT], "take mug 1", {"game_id": f"train/room/{i % 2}", "split": "dev_heldout_rooms", "step": i, "task_type": "t"}, 0) for i in range(n)]


def ppo_trainer(tmp_path=None, games=GAMES, **kw):
    cfg = config(topic="arm-rl-model-in-control", reward="R3", **({"out_dir": str(tmp_path)} if tmp_path else {}), **kw)
    return PPOTrainer(cfg, StubSys1(seed=cfg.seed), MENU, games, DEV_GAME, core_rows(), dev_rows())


def scheduled_trainer(tmp_path=None, objective="ce", **kw):
    cfg = config(topic="arm-rl-scheduled", objective=objective, reward="R2", **({"out_dir": str(tmp_path)} if tmp_path else {}), **kw)
    return ScheduledTrainer(cfg, StubSys1(seed=cfg.seed), MENU, GAMES, DEV_GAME, core_rows(), dev_rows())


def step_log(step=0, actor="sys1", pick="a", teacher="a", menu=("a", "b"), sampled=True, log_prob=-0.5, prompt=None, **kw):
    options = list(menu) + [FALLBACK_TEXT]
    return StepLog(step=step, actor=actor, command=pick or "look", obs="", admissible=list(menu), control=ControlState(), control_after=ControlState(), menu=list(menu), options=options, probs=[0.5, 0.25, 0.25], pick=pick,
                   log_prob=log_prob, sampled=sampled, teacher_command=teacher, has_teacher=True, prompt=prompt or f"prompt {step}", **kw)


def r3_harness():
    return with_reward(HarnessConfig(k=4, fallbacks_before_handover=2, llm_rounds=0, menu_run=3, trace_window=5, w=0, c=0, beta=0, max_steps=50), "R3")


def game_log(steps, won):
    log = EpisodeLog("train/x/y", "t", "g", "o", "model", 0, False, steps=steps)
    log.won = won
    return log


# The ratio and the clip.

def test_policy_loss_by_hand():
    # Ratios: 0.6/0.4 = 1.5, 0.3/0.6 = 0.5, 0.45/0.5 = 0.9. Clip 0.2, so the clipped ratios are 1.2, 0.8, 0.9.
    # Advantages 2, 1, -2. Row 1: min(3.0, 2.4) = 2.4. Row 2: min(0.5, 0.8) = 0.5. Row 3: min(-1.8, -1.8) = -1.8. Mean is 1.1/3, and the loss is its negative.
    new = torch.log(torch.tensor([0.6, 0.3, 0.45], dtype=torch.float64))
    old = torch.log(torch.tensor([0.4, 0.6, 0.5], dtype=torch.float64))
    loss, ratio = policy_loss(new, old, torch.tensor([2.0, 1.0, -2.0], dtype=torch.float64), 0.2)
    assert ratio.tolist() == pytest.approx([1.5, 0.5, 0.9])
    assert loss.item() == pytest.approx(-(2.4 + 0.5 - 1.8) / 3)


def test_clip_binds_on_the_side_that_gains_and_not_on_the_side_that_loses():
    # Advantage -2 at ratio 1.5: unclipped -3.0, clipped -2.4, the min is -3.0, so the loss is +3.0 and the clip does not help the step.
    new, old = torch.log(torch.tensor([0.6], dtype=torch.float64)), torch.log(torch.tensor([0.4], dtype=torch.float64))
    loss, _ = policy_loss(new, old, torch.tensor([-2.0], dtype=torch.float64), 0.2)
    assert loss.item() == pytest.approx(3.0)
    # Advantage 2 at ratio 1.5 with a wide clip: the unclipped 3.0 stands.
    wide, _ = policy_loss(new, old, torch.tensor([2.0], dtype=torch.float64), 1.0)
    assert wide.item() == pytest.approx(-3.0)


def test_log_prob_uses_the_distribution_the_pick_was_drawn_from():
    logits = torch.log(torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64))
    open_list, entropy = log_prob_and_entropy(logits, 1, masked=False)
    assert open_list.item() == pytest.approx(math.log(0.3))
    assert entropy.item() == pytest.approx(-(0.5 * math.log(0.5) + 0.3 * math.log(0.3) + 0.2 * math.log(0.2)))
    masked, _ = log_prob_and_entropy(logits, 1, masked=True)   # fallback out, so 0.3 / 0.8, as harness.metrics.masked_probs
    assert masked.item() == pytest.approx(math.log(0.3 / 0.8))


def test_value_loss_by_hand():
    assert value_loss(torch.tensor([1.0, 3.0]), torch.tensor([2.0, 0.0])).item() == pytest.approx(0.5 * (1 + 9) / 2)


def test_ratio_at_the_start_of_an_update_is_one():
    trainer = ppo_trainer(kl=False)
    logs = trainer.play(GAMES[:2], harness_config(trainer.cfg, "R3"), "model")
    transitions = build_transitions(logs, harness_config(trainer.cfg, "R3"), 0.99, 0.95, trainer.collector.values)
    trainer.sys1.train()
    logits, _ = score_and_decide(trainer.sys1, [t.prompt for t in transitions], [t.options for t in transitions])
    new = torch.stack([log_prob_and_entropy(z, t.action, t.masked)[0] for z, t in zip(logits, transitions)]).detach()
    old = torch.tensor([t.old_log_prob for t in transitions])
    assert torch.allclose(torch.exp(new - old), torch.ones_like(old), atol=1e-4)


# Which steps the ratio may read.

def test_ratio_mask_excludes_teacher_run_steps_on_a_real_mixed_game():
    trainer = scheduled_trainer(objective="ppo")
    harness = harness_config(trainer.cfg, "R2")
    logs = trainer.play(GAMES[:3], harness, "scheduled", teacher_prob=0.5)
    actors = {s.actor for log in logs for s in log.steps}
    assert "teacher" in actors and "sys1" in actors   # both kinds exist, or the test says nothing
    for log in logs:
        assert ratio_mask(log) == [s.actor != "teacher" for s in log.steps]
    picked = {(t.uid, t.step) for t in build_transitions(logs, harness, 0.99, 0.95, trainer.collector.values)}
    assert picked == {(log.uid, s.step) for log in logs for s in log.steps if s.actor != "teacher"}


def test_a_teacher_run_step_cannot_change_the_loss():
    # A teacher-run step with an absurd log probability and pay: the transitions the loss reads are the same either way, beyond the discount it adds to the step before it.
    harness = r3_harness()
    values = {(f"prompt {i}", ("a", "b", FALLBACK_TEXT)): 0.0 for i in range(3)}
    plain = [step_log(0), step_log(1, actor="teacher", sampled=False, log_prob=-50.0), step_log(2)]
    other = [step_log(0), step_log(1, actor="teacher", pick="b", sampled=False, log_prob=-1.0), step_log(2)]
    a = build_transitions([game_log(plain, False)], harness, 0.9, 0.8, values)
    b = build_transitions([game_log(other, False)], harness, 0.9, 0.8, values)
    assert [t.step for t in a] == [0, 2]
    assert [(t.old_log_prob, t.reward, t.discount, t.advantage) for t in a] == [(t.old_log_prob, t.reward, t.discount, t.advantage) for t in b]


def test_a_teacher_run_step_marked_sampled_stops_the_build():
    log = game_log([step_log(0, actor="teacher", sampled=True)], False)
    with pytest.raises(RuntimeError, match="teacher-run"):
        build_transitions([log], r3_harness(), 0.9, 0.8, {("prompt 0", ("a", "b", FALLBACK_TEXT)): 0.0})


def test_a_teacher_only_iteration_trains_nothing_and_says_so():
    trainer = scheduled_trainer(objective="ppo", teacher_prob_fixed=1.0)
    before = params_sha256(trainer.sys1)
    trainer.prepare()
    summary = trainer.iteration()
    assert summary["transitions"] == 0 and trainer.counts["iterations_without_transitions"] == 1 and trainer.pos["updates"] == 0
    assert params_sha256(trainer.sys1) == before


# Returns and advantages.

def test_returns_and_advantages_by_hand_with_a_terminal_beta():
    # R3 pays 1, -1, and at the end 1 + 5. Gamma 0.9, lambda 0.8. Values 0.5, 1, 2.
    # Last: delta = 6 - 2 = 4, A = 4.
    # Middle: delta = -1 + 0.9 * 2 - 1 = -0.2, A = -0.2 + 0.72 * 4 = 2.68.
    # First: delta = 1 + 0.9 * 1 - 0.5 = 1.4, A = 1.4 + 0.72 * 2.68 = 3.3296. A return is A plus the value.
    harness = r3_harness()
    steps = [step_log(0, pick="a", teacher="a"), step_log(1, pick="b", teacher="a"), step_log(2, pick="a", teacher="a")]
    values = {("prompt 0", ("a", "b", FALLBACK_TEXT)): 0.5, ("prompt 1", ("a", "b", FALLBACK_TEXT)): 1.0, ("prompt 2", ("a", "b", FALLBACK_TEXT)): 2.0}
    out = build_transitions([game_log(steps, True)], harness, 0.9, 0.8, values)
    assert [t.reward for t in out] == pytest.approx([1.0, -1.0, 6.0])
    assert [t.advantage for t in out] == pytest.approx([3.3296, 2.68, 4.0])
    assert [t.ret for t in out] == pytest.approx([3.8296, 3.68, 6.0])


def test_a_game_at_the_cap_pays_no_beta_and_does_not_bootstrap():
    harness = r3_harness()
    values = {("prompt 0", ("a", "b", FALLBACK_TEXT)): 10.0}
    out = build_transitions([game_log([step_log(0)], False)], harness, 0.9, 0.8, values)
    assert out[0].reward == pytest.approx(1.0) and out[0].advantage == pytest.approx(1.0 - 10.0) and out[0].ret == pytest.approx(1.0)


def test_returns_across_a_teacher_run_step_by_hand():
    # Steps 0 and 2 are Sys1's, 1 and 3 the teacher's, and the teacher wins at step 3, so beta 5 lands there. Gamma 0.9, lambda 0.8. Values 0.5 and 1.
    # Decision at step 0: pay 1 + 0.9 * 0 = 1, two steps to the next decision, so discount 0.81.
    # Decision at step 2: pay -1 + 0.9 * 5 = 3.5, and the game ends, so A = 3.5 - 1 = 2.5.
    # Then delta = 1 + 0.81 * 1 - 0.5 = 1.31, and A = 1.31 + 0.81 * 0.8 * 2.5 = 2.93.
    harness = r3_harness()
    steps = [step_log(0), step_log(1, actor="teacher", sampled=False), step_log(2, pick="b"), step_log(3, actor="teacher", sampled=False)]
    values = {("prompt 0", ("a", "b", FALLBACK_TEXT)): 0.5, ("prompt 2", ("a", "b", FALLBACK_TEXT)): 1.0}
    out = build_transitions([game_log(steps, True)], harness, 0.9, 0.8, values)
    assert [t.step for t in out] == [0, 2]
    assert [t.reward for t in out] == pytest.approx([1.0, 3.5])
    assert out[0].discount == pytest.approx(0.81)
    assert [t.advantage for t in out] == pytest.approx([2.93, 2.5])
    assert [t.ret for t in out] == pytest.approx([3.43, 3.5])


def test_returns_function_alone_matches_ordinary_gae():
    advantages, returns = returns_and_advantages([1.0, 2.0], [0.0, 0.0], [0.5, 0.5], 1.0)
    assert advantages == pytest.approx([2.0, 2.0]) and returns == pytest.approx([2.0, 2.0])


def test_rewards_from_transitions_match_what_the_harness_logged_under_r3():
    trainer = ppo_trainer()
    harness = harness_config(trainer.cfg, "R3")
    logs = trainer.play(GAMES[:2], harness, "model")
    collected = build_transitions(logs, harness, 1.0, 0.0, trainer.collector.values)
    assert [t.reward for t in collected] == pytest.approx([s.reward for log in logs for s in log.steps])


def test_a_fallback_pays_r2s_rule_under_r4_and_nothing_else_does():
    r4 = with_reward(r3_harness(), "R4")
    menu = ("a", "b")
    fallback_on = step_log(0, pick=FALLBACK_TEXT, teacher="a")
    fallback_off = step_log(1, pick=FALLBACK_TEXT, teacher="c")
    pick = step_log(2, pick="a", teacher="a")
    values = {(f"prompt {i}", (*menu, FALLBACK_TEXT)): 0.0 for i in range(3)}
    out = build_transitions([game_log([fallback_on, fallback_off, pick], True)], r4, 1.0, 1.0, values)
    assert [t.reward for t in out] == pytest.approx([-0.1, 1.0, 1.0])   # the last is beta 1, because R4 pays no match
    assert step_reward(FALLBACK_TEXT, "a", list(menu), r4) == 0.0   # the harness alone would pay 0 under R4
    assert "Claude's choice, not Halil's" in FALLBACK_RULE


def test_normalized_weights_have_mean_zero_and_unit_spread():
    harness = r3_harness()
    values = {(f"prompt {i}", ("a", "b", FALLBACK_TEXT)): 0.0 for i in range(3)}
    out = build_transitions([game_log([step_log(0), step_log(1, pick="b"), step_log(2)], True)], harness, 0.9, 0.8, values)
    raw = [t.advantage for t in out]
    normalize_weights(out)
    assert np.mean([t.weight for t in out]) == pytest.approx(0.0, abs=1e-12)
    assert np.std([t.weight for t in out]) == pytest.approx(1.0)
    assert [t.advantage for t in out] == raw


# The schedule.

@pytest.mark.parametrize("steps,expected", [(0, 1.0), (200, 0.75), (400, 0.5), (800, 0.0), (1000, 0.0)])
def test_schedule_by_hand(steps, expected):
    assert teacher_prob(steps, 800) == pytest.approx(expected)   # a budget of 1000 at the default fraction 0.8


def test_trainer_reads_the_schedule_from_commands_run():
    trainer = scheduled_trainer(step_budget=100, schedule_fraction=0.5)
    assert trainer.probability_now() == 1.0
    trainer.pos["env_steps"] = 25
    assert trainer.probability_now() == pytest.approx(0.5)
    trainer.pos["env_steps"] = 60
    assert trainer.probability_now() == 0.0


def test_one_play_uses_the_probability_of_its_start():
    trainer = scheduled_trainer(teacher_prob_fixed=0.0)
    summary = trainer.iteration()
    assert summary["teacher_prob"] == 0.0 and summary["teacher_run_steps"] == 0
    trainer = scheduled_trainer(teacher_prob_fixed=1.0)
    summary = trainer.iteration()
    assert summary["teacher_prob"] == 1.0 and summary["sys1_run_steps"] == 0 and summary["teacher_stand_in_steps"] == 0


def test_reward_moves_to_r3_once_sys1_acts():
    assert [reward_at(p, 0.5) for p in (1.0, 0.75, 0.5, 0.25, 0.0)] == ["R2", "R2", "R3", "R3", "R3"]


@pytest.mark.parametrize("update,factor", [(0, 1 / 16), (15, 1.0), (16, 1.0), (296, 0.5), (576, 0.0), (600, 0.0)])
def test_learning_rate_factor_by_hand(update, factor):
    # A rise over 16 steps, then a cosine over 560 more: step 296 is its midpoint, where the cosine is 0.
    assert cosine_factor(update, 16, 576) == pytest.approx(factor)


def test_the_planned_updates_equal_deciders_at_its_budget():
    cfg = Config(topic="arm-rl-model-in-control", seed=0, step_budget=4608, device="cpu", init_from="x").resolve()
    assert cfg.lr_total_updates == 576 and cfg.lr == 1e-6 and cfg.clip == 0.2 and cfg.transitions_per_iter == 96 and cfg.kl_rows == 8 and cfg.minibatch == 8 and cfg.beta2 == 0.95
    assert cosine_factor(576, cfg.lr_warmup, cfg.lr_total_updates) == 0.0


# The buffer.

@pytest.mark.parametrize("flags,name", [(dict(label_random=True, teacher="a"), "label_random"), (dict(label_inadmissible=True, teacher="look"), "label_inadmissible"), (dict(label_error="KeyError", teacher=None), "label_error"),
                                        (dict(teacher="help"), "label_inadmissible")])
def test_buffer_drops_flagged_labels(flags, name):
    buffer = LabelBuffer()
    s = step_log(0, **flags)
    assert flag_of(s) == name
    assert not buffer.add_step(s, "train/x/y", 0.5)
    assert buffer.rows == [] and buffer.counts[f"dropped_{name}"] == 1 and buffer.counts["steps_seen"] == 1


def test_buffer_keeps_good_labels_on_the_options_shown_and_labels_fallback_off_the_menu():
    buffer = LabelBuffer()
    on = step_log(0, teacher="b", pick="a")
    off = step_log(1, teacher="c", pick="a")
    off.admissible = ["a", "b", "c"]
    assert buffer.add_step(on, "u", 0.25) and buffer.add_step(off, "u", 0.25)
    assert buffer.rows[0]["label"] == "b" and buffer.rows[0]["options"] == ["a", "b", FALLBACK_TEXT] and buffer.rows[0]["state"] == "prompt 0" and buffer.rows[0]["teacher_prob"] == 0.25
    assert buffer.rows[1]["label"] == FALLBACK_TEXT and buffer.counts["rows_label_fallback"] == 1
    assert not buffer.add_step(step_log(0, teacher="b", pick="b"), "other", 0.0) and buffer.counts["duplicates"] == 1   # same prompt, options and label
    no_question = step_log(2)
    no_question.options = None
    assert not buffer.add_step(no_question, "u", 0.0) and buffer.counts["dropped_no_question_or_no_label"] == 1


def test_buffer_on_real_labels_holds_no_flagged_row_and_counts_each_drop():
    trainer = scheduled_trainer(teacher_prob_fixed=0.0, max_steps=30, tracking="always", batch_games=3)
    logs = trainer.play([GAMES[1], GAMES[10], GAMES[11]], harness_config(trainer.cfg, "R2"), "scheduled", teacher_prob=0.0)   # on these three a random Sys1 reached the expert's `help` when this test was written
    buffer = LabelBuffer()
    buffer.add_logs(logs, 0.0)
    states = {s.prompt: s for log in logs for s in log.steps}
    flagged = [s for log in logs for s in log.steps if flag_of(s)]
    assert buffer.rows and all(flag_of(states[r["state"]]) is None for r in buffer.rows)
    assert sum(buffer.counts[f"dropped_{f}"] for f in ("label_random", "label_inadmissible", "label_error")) == len(flagged)
    assert flagged   # or this test says nothing about real labels


def test_row_loss_by_hand():
    logits = torch.log(torch.tensor([0.5, 0.3, 0.2]))
    row = {"options": ["a", "b", FALLBACK_TEXT], "label": "b"}
    assert row_loss(logits, row).item() == pytest.approx(-math.log(0.3), rel=1e-5)
    assert row_loss(logits, {**row, "target": {"a": 0.5, "b": 0.5}}).item() == pytest.approx(-(0.5 * math.log(0.5) + 0.5 * math.log(0.3)), rel=1e-5)


def test_batches_mix_the_core_set_and_the_buffer():
    trainer = scheduled_trainer(core_share=0.5)
    assert {r["source"] for r in trainer.sample_rows(8)} == {"core"}   # an empty buffer leaves the core set
    trainer.buffer.add_step(step_log(0), "u", 1.0)
    rows = trainer.sample_rows(8)
    assert sum(r["source"] == "core" for r in rows) == 4 and sum(r["source"] == "buffer" for r in rows) == 4


def test_a_core_row_keeps_its_options_and_label_from_the_export():
    row = labelled_row(core_rows(1)[0])
    assert row["options"] == ["go to desk 0", "take mug 1", FALLBACK_TEXT] and row["label"] == "take mug 1" and row["source"] == "core"


def test_the_row_reader_refuses_the_wrong_split_and_a_test_game(tmp_path):
    good = tmp_path / "rows.jsonl.gz"
    write_jsonl(core_rows(2), good)
    assert len(read_kev_rows(good, ("train",))) == 2
    with pytest.raises(AssertionError, match="split"):
        read_kev_rows(good, ("dev",))
    bad = tmp_path / "bad.jsonl.gz"
    write_jsonl([kev_row("s", ["a", FALLBACK_TEXT], "a", {"game_id": "valid_unseen/x/y", "split": "train", "step": 0, "task_type": "t"}, 0)], bad)
    with pytest.raises(AssertionError, match="test game"):
        read_kev_rows(bad, ("train",))


# The value head and gradients.

def test_value_head_gets_gradients_and_pointer_head_still_gets_them():
    trainer = ppo_trainer(entropy_coef=0.01)
    trainer.prepare()
    harness = harness_config(trainer.cfg, "R3")
    logs = trainer.play(GAMES[:2], harness, "model")
    batch = build_transitions(logs, harness, 0.99, 0.95, trainer.collector.values)[:2]
    trainer.sys1.train()
    dropout_off(trainer.sys1)
    trainer.value_head.train()
    trainer.optimizer.step = lambda: None   # keeps the gradients for reading
    trainer.optimizer.zero_grad = lambda set_to_none=True: None
    stats = trainer.ppo_step(batch)
    assert not stats["kl_guard_tripped"]
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in trainer.value_head.parameters())
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in trainer.sys1.trainable_parameters())


def test_detaching_keeps_the_value_loss_out_of_sys1():
    sys1 = StubSys1(seed=0)
    sys1.train()
    head = ValueHead(decide_dim(sys1), seed=0)
    _, decide = score_and_decide(sys1, ["p"], [["a", FALLBACK_TEXT]])
    value_loss(head(decide.detach()), torch.tensor([1.0])).backward()
    assert all(p.grad is None for p in sys1.parameters()) and all(p.grad is not None for p in head.parameters())
    value_loss(head(decide), torch.tensor([1.0])).backward()
    assert all(p.grad is not None for p in sys1.parameters())   # the head reads Sys1's hidden state, and VALUE_DETACH=0 opens this path


def test_a_stub_score_and_decide_equals_score():
    sys1 = StubSys1(seed=1)
    sys1.eval()
    prompts, options = ["put a mug on the desk", "heat an egg"], [["go to desk 1", FALLBACK_TEXT], ["go to fridge 1", "open fridge 1", FALLBACK_TEXT]]
    logits, decide = score_and_decide(sys1, prompts, options)
    for a, b in zip(logits, sys1.score(prompts, options)):
        assert torch.equal(a, b)
    assert decide.shape == (2, sys1.dim) and not decide.requires_grad


def test_kev_path_restores_the_callers_order_and_reads_decide_from_the_real_head():
    from kev.model import PointerHead

    head = PointerHead(4, dp=4)

    class FakeModel:
        def __init__(self):
            self.head = head

        def forward_batch(self, encs):
            out = []
            for e in encs:
                h = torch.full((3, 4), float(len(e["ids"])))
                out.append([self.head(h[0], h[1:])])
            return out

    fake = SimpleNamespace(model=FakeModel(), training=False, autocast_bf16=False, _encode=lambda p, o: {"ids": list(range(len(p)))})
    prompts = ["long prompt", "a", "mid"]   # lengths 11, 1, 3, so the pass runs them as 1, 3, 11
    logits, decide = score_and_decide(fake, prompts, [["x", "y"]] * 3)
    assert decide[:, 0].tolist() == [11.0, 1.0, 3.0]
    for z, p in zip(logits, prompts):
        assert torch.equal(z, head(torch.full((4,), float(len(p))), torch.full((2, 4), float(len(p)))))
    assert not head._forward_pre_hooks   # the hook does not stay


def test_dropout_off_survives_only_until_the_next_train_call():
    net = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.5))
    net.train()
    dropout_off(net)
    assert not net[1].training and net[0].training
    net.train()
    assert net[1].training


def test_collector_gives_the_same_probabilities_and_one_value_per_state():
    sys1 = StubSys1(seed=2)
    sys1.eval()
    head = ValueHead(decide_dim(sys1), seed=0)
    collector = ValueCollector(sys1, head)
    prompts, options = ["p one", "p two"], [["a", FALLBACK_TEXT], ["b", "c", FALLBACK_TEXT]]
    for a, b in zip(collector.probs(prompts, options), sys1.probs(prompts, options)):
        assert torch.allclose(a, b)
    assert set(collector.values) == {("p one", ("a", FALLBACK_TEXT)), ("p two", ("b", "c", FALLBACK_TEXT))}
    _, decide = score_and_decide(sys1, prompts, options)
    assert collector.values[("p one", ("a", FALLBACK_TEXT))] == pytest.approx(head(decide)[0].item())


def test_a_fitted_temperature_cannot_reach_collection():
    trainer = ppo_trainer()
    trainer.sys1.temperature = 2.0
    with pytest.raises(RuntimeError, match="temperature 1"):
        trainer.play(GAMES[:1], harness_config(trainer.cfg, "R3"), "model")


def test_micro_batches_give_the_same_step_as_one_pass():
    def after(micro):
        trainer = ppo_trainer(kl=False, micro_batch=micro, kl_guard=False, normalize_advantages=False)
        harness = harness_config(trainer.cfg, "R3")
        batch = build_transitions(trainer.play(GAMES[:1], harness, "model"), harness, 0.99, 0.95, trainer.collector.values)[:4]
        trainer.sys1.train()
        trainer.value_head.train()
        trainer.ppo_step(batch)
        return [p.detach().clone() for p in trainer.sys1.trainable_parameters()]

    for a, b in zip(after(1), after(4)):
        assert torch.allclose(a, b, atol=1e-6)


# The KL term and its guard.

def test_kl_by_hand():
    # KL((0.5, 0.5) || (0.75, 0.25)) = 0.5 ln(0.5 / 0.75) + 0.5 ln(0.5 / 0.25).
    reference = torch.log(torch.tensor([0.5, 0.5], dtype=torch.float64))
    current = torch.log(torch.tensor([0.75, 0.25], dtype=torch.float64))
    assert kl_to_reference(reference, current).item() == pytest.approx(0.5 * math.log(0.5 / 0.75) + 0.5 * math.log(2.0))
    assert kl_to_reference(reference, reference).item() == pytest.approx(0.0)


def test_kl_pool_holds_the_starting_distributions_and_is_zero_before_any_update():
    trainer = ppo_trainer()
    trainer.prepare()
    assert len(trainer.kl_pool) == 4
    trainer.sys1.train()
    assert max(trainer.kl_backward()) == pytest.approx(0.0, abs=1e-5)


def test_the_kl_term_can_be_switched_off():
    trainer = ppo_trainer(kl=False)
    trainer.prepare()
    assert trainer.kl_pool == []
    harness = harness_config(trainer.cfg, "R3")
    batch = build_transitions(trainer.play(GAMES[:1], harness, "model"), harness, 0.99, 0.95, trainer.collector.values)[:2]
    trainer.sys1.train()
    stats = trainer.ppo_step(batch)
    assert "kl_mean" not in stats and not stats["kl_guard_tripped"]


def test_guard_drops_the_actor_term_and_zeroes_the_first_moment():
    trainer = ppo_trainer(kl_guard_mean=-1.0)   # any KL trips it
    trainer.prepare()
    harness = harness_config(trainer.cfg, "R3")
    batch = build_transitions(trainer.play(GAMES[:1], harness, "model"), harness, 0.99, 0.95, trainer.collector.values)[:2]
    trainer.sys1.train()
    trainer.value_head.train()
    params = trainer.optimizer.param_groups[0]["params"]
    for p in params:   # a moment left from an earlier step, which the guard must clear
        trainer.optimizer.state[p] = {"step": torch.tensor(5.0), "exp_avg": torch.full_like(p, 9.0), "exp_avg_sq": torch.ones_like(p)}
    value_before = [p.detach().clone() for p in trainer.value_head.parameters()]
    stats = trainer.ppo_step(batch)
    assert stats["kl_guard_tripped"] and "policy_loss" not in stats and trainer.counts["kl_guard_steps"] == 1
    assert all(trainer.optimizer.state[p]["exp_avg"].abs().max() < 1.0 for p in params)   # 0.1 of a clipped gradient, and no trace of the 9
    assert all(torch.equal(a, b) for a, b in zip(value_before, trainer.value_head.parameters()))


# The fallback rule with no LLM.

class AlwaysFallback(StubSys1):
    def score(self, prompts, options):
        return [torch.tensor([-20.0] * (len(o) - 1) + [20.0]) for o in options]


def test_a_sampled_fallback_runs_the_teachers_command_and_is_counted():
    cfg = config(topic="arm-rl-model-in-control", reward="R3", kl=False, max_steps=8)
    harness = harness_config(cfg, "R3")
    with teacher_stand_in():
        logs = run_episodes(GAMES[:1], AlwaysFallback(), TeacherStandIn(), MENU, harness, "model", teacher="expert", seed=0, batch_size=1, tracking="always")
    expert = run_episodes(GAMES[:1], None, None, None, harness, "teacher", seed=0, batch_size=1, tracking="always")
    steps = logs[0].steps
    assert [s.command for s in steps] == [s.command for s in expert[0].steps]   # the teacher's game, command by command
    assert all(s.actor == "llm" and s.llm_name == STAND_IN_NAME and s.pick == FALLBACK_TEXT and s.sampled for s in steps)
    assert all(s.reward == step_reward(FALLBACK_TEXT, s.teacher_command, s.menu, harness) for s in steps[:-1])   # R2's rule
    assert any(line.startswith("LLM> ") for line in steps[-1].prompt.splitlines())   # the mark an LLM's command has at the Gemma stage


def test_trainer_counts_stand_in_steps_and_wins_they_touched():
    # The untrained stub samples fallback on some steps and a menu command on others, and the counts must match the log step by step.
    trainer = ppo_trainer(kl=False, max_steps=50)
    logs = trainer.play(GAMES[:1], harness_config(trainer.cfg, "R3"), "model")
    stand_in = sum(s.llm_name == STAND_IN_NAME for s in logs[0].steps)
    assert 0 < stand_in < logs[0].n_steps   # both kinds exist, or the test says nothing
    assert trainer.pos["env_steps"] == logs[0].n_steps
    assert trainer.counts["teacher_stand_in_steps"] == stand_in == trainer.counts["llm_run_steps"] == trainer.counts["sampled_fallback_picks"]
    assert trainer.counts["sys1_run_steps"] == logs[0].n_steps - stand_in and trainer.counts["sampled_steps"] == logs[0].n_steps
    assert trainer.counts["games_won_with_stand_in"] == int(logs[0].won)


def test_the_wrap_leaves_the_harness_as_it_was_and_other_llms_untouched():
    from harness import episode
    real = episode._ask_llm
    with teacher_stand_in():
        assert episode._ask_llm is not real
        called = []

        class Other:
            name = "other"

            def reason_then_command(self, goals, histories, admissibles):
                called.append(len(goals))
                return [SimpleNamespace(command="look", probs=None, tokens_in=0, tokens_out=0, seconds=0.0) for _ in goals]

        games = [SimpleNamespace(view=SimpleNamespace(goal="g", admissible=["look"], uid="u", label=None), history=[("", "o")], llm_pick=None, seconds_llm=0.0)]
        episode._ask_llm(games, Other(), "model")
        assert called == [1] and games[0].llm_pick.command == "look"
    assert episode._ask_llm is real


def test_the_stand_in_refuses_a_call_outside_the_wrap():
    with pytest.raises(RuntimeError, match="teacher_stand_in"):
        TeacherStandIn().reason_then_command(["g"], [[]], [["look"]])


def test_masked_training_never_falls_back_and_needs_no_teacher_under_r4():
    trainer = ppo_trainer(fallback="masked", kl=False)
    trainer.cfg.reward = "R4"
    trainer.teacher = None
    harness = harness_config(trainer.cfg, "R4")
    logs = trainer.play(GAMES[:1], harness, "model")
    assert not any(s.has_teacher for s in logs[0].steps) and all(s.pick != FALLBACK_TEXT and s.masked and s.sampled for s in logs[0].steps)
    out = build_transitions(logs, harness, 1.0, 1.0, trainer.collector.values)
    assert all(t.masked and t.action < len(t.options) - 1 for t in out)
    trainer.sys1.train()
    logits, _ = score_and_decide(trainer.sys1, [t.prompt for t in out], [t.options for t in out])
    new = torch.stack([log_prob_and_entropy(z, t.action, t.masked)[0] for z, t in zip(logits, out)]).detach()
    assert torch.allclose(new, torch.tensor([t.old_log_prob for t in out]), atol=1e-4)   # the ratio reads the masked distribution


# Games, dev and refit.

def test_games_come_in_an_order_from_the_seed_with_no_repeat_in_a_batch():
    trainer = ppo_trainer()
    n = len(GAMES)
    first_epoch = []
    while trainer.pos["epoch"] == 0:
        batch = trainer.next_games(5)
        assert len({g.uid for g in batch}) == len(batch) <= 5
        first_epoch += [g.uid for g in batch]
    assert sorted(first_epoch) == sorted(g.uid for g in GAMES) and len(first_epoch) == n
    second = trainer.next_games(n)
    assert [g.uid for g in second] != first_epoch   # a new order
    assert [g.uid for g in ppo_trainer().next_games(5)] == first_epoch[:5]


def test_each_play_gets_new_draws():
    trainer = ppo_trainer()
    harness = harness_config(trainer.cfg, "R3")
    a = trainer.play(GAMES[:1], harness, "model")
    b = trainer.play(GAMES[:1], harness, "model")
    assert [s.command for s in a[0].steps] != [s.command for s in b[0].steps]


def test_dev_point_is_sys1_alone_and_records_the_curve():
    trainer = ppo_trainer(kl=False)
    trainer.dev_point("start")
    point = trainer.curve[-1]
    assert point["label"] == "start" and point["env_steps"] == 0 and point["temperature"] == 1.0 and point["n_games"] == 1
    assert 0.0 <= point["success"] <= 1.0 and point["step_match"] is not None and point["success_by_task_type"]
    assert trainer.pos["env_steps"] == 0   # dev games do not spend the budget
    assert trainer.last_dev["per_game"][0]["uid"] == DEV_GAME[0].uid
    assert trainer.counts["teacher_stand_in_steps"] == 0


def test_a_full_ppo_run_finishes_with_a_refit_and_a_loadable_checkpoint(tmp_path):
    trainer = ppo_trainer(tmp_path)
    result = trainer.run()
    assert result["completed"] and result["steps"] >= 24 and result["games_played"] >= 2 and result["updates"] > 0
    assert 0.25 <= result["temperature"] <= 4.0 and trainer.sys1.temperature == result["temperature"]
    assert result["learning_curve"][0]["label"] == "start" and result["learning_curve"][-1]["env_steps"] == result["steps"]
    for key in ("seed", "git_commit", "environment", "hardware", "config", "games_played", "steps", "steps_per_second", "peak_memory", "precision", "per_game", "aggregates", "fallback_rule", "teacher_stand_in_share_of_steps"):
        assert key in result
    assert "says nothing about Sys1's quality" in result["scope"]
    extract = tmp_path / "x"
    with tarfile.open(tmp_path / f"{trainer.cfg.stem}.tar") as tar:
        tar.extractall(extract, filter="data")
    loaded = StubSys1.load(extract / "checkpoint", "cpu")
    assert loaded.temperature == result["temperature"]
    assert all(torch.equal(a, b) for a, b in zip(loaded.parameters(), trainer.sys1.parameters()))
    first = result["iterations"][0]
    assert first["first_ratio_deviation"] == pytest.approx(0.0, abs=1e-4)


def test_a_full_scheduled_run_fills_the_buffer_and_finishes(tmp_path):
    trainer = scheduled_trainer(tmp_path)
    result = trainer.run()
    assert result["completed"] and result["buffer"]["rows"] > 0 and result["counts"]["ce_rows_trained"] > 0
    probabilities = [i["teacher_prob"] for i in result["iterations"]]
    assert probabilities[0] == 1.0 and probabilities == sorted(probabilities, reverse=True)
    assert 0.25 <= result["temperature"] <= 4.0 and "says nothing" in result["scope"]


def test_ppo_form_of_the_scheduled_arm_reads_only_what_sys1_ran(tmp_path):
    trainer = scheduled_trainer(tmp_path, objective="ppo", teacher_prob_fixed=0.5, kl=True)
    result = trainer.run()
    assert result["completed"] and result["updates"] > 0
    assert sum(i["transitions"] for i in result["iterations"]) == result["counts"]["sampled_steps"]
    assert result["counts"]["sampled_steps"] < result["steps"] and result["counts"]["teacher_run_steps"] > 0
    assert all(i["reward"] == "R3" for i in result["iterations"])   # 0.5 is the threshold


def test_ce_form_has_no_value_head_and_no_reward_in_its_loss():
    trainer = scheduled_trainer()
    assert trainer.value_head is None and trainer.collector is None and len(trainer.optimizer.param_groups) == 1


def test_training_changes_sys1_and_the_dev_curve_exists_at_the_stated_cadence(tmp_path):
    trainer = ppo_trainer(tmp_path, step_budget=30, dev_every_steps=10, lr=0.01)
    before = params_sha256(trainer.sys1)
    result = trainer.run()
    assert params_sha256(trainer.sys1) != before
    steps = [p["env_steps"] for p in result["learning_curve"]]
    assert steps[0] == 0 and steps[-1] == result["steps"] and all(b >= a for a, b in zip(steps, steps[1:]))
    assert len(steps) >= 3


# Resume.

def comparable(state):
    """What must match after a resume, without clocks. A dev point and an iteration each carry their wall time."""
    out = copy.deepcopy({k: state[k] for k in ("pos", "counts", "curve", "iterations", "kl_pool", "params_sha256", "extra")})
    for row in out["curve"] + out["iterations"]:
        row.pop("wall_seconds", None)
        row.pop("dev_seconds", None)
    return out


def same_rng(a, b):
    return a["python"] == b["python"] and a["trainer"] == b["trainer"] and torch.equal(a["torch"], b["torch"]) and all(np.array_equal(x, y) for x, y in zip(a["numpy"], b["numpy"]))


@pytest.mark.parametrize("make,kw", [(ppo_trainer, {}), (scheduled_trainer, {}), (scheduled_trainer, {"objective": "ppo", "teacher_prob_fixed": 0.5})], ids=["ppo", "ce", "scheduled-ppo"])
def test_resume_restores_identical_state(tmp_path, make, kw):
    seed_all = lambda: (__import__("random").seed(0), np.random.seed(0), torch.manual_seed(0))  # noqa: E731

    seed_all()
    fast = dict(games_per_collect=1, batch_games=1, max_steps=4, dev_every_steps=6)   # a dev point falls inside the resumed part, and this test stays short
    straight = make(tmp_path / "straight", stop_after_iterations=3, step_budget=10**6, **fast, **kw)
    straight.run()
    expected = straight.state()

    seed_all()
    first = make(tmp_path / "first", stop_after_iterations=1, step_budget=10**6, **fast, **kw)
    first.run()
    saved_position = dict(first.pos)
    tar_path = tmp_path / "first" / f"{first.cfg.stem}-state.tar"
    with tarfile.open(tar_path) as tar:
        tar.extractall(tmp_path / "resume", filter="data")
    state_dir = tmp_path / "resume" / "state"

    # The random streams move in between, as they do on a new VM, so only the saved state can bring them back.
    __import__("random").seed(99), np.random.seed(99), torch.manual_seed(99)
    second = make(tmp_path / "second", stop_after_iterations=2, step_budget=10**6, **fast, **kw)
    second.sys1 = StubSys1.load(state_dir / "sys1", "cpu")
    second.optimizer = torch.optim.AdamW([{"params": second.sys1.trainable_parameters(), "lr": second.cfg.lr}] + ([{"params": list(second.value_head.parameters()), "lr": second.cfg.value_lr}] if second.value_head is not None else []), betas=(second.cfg.beta1, second.cfg.beta2), weight_decay=second.cfg.weight_decay)
    if second.collector is not None:
        second.collector.sys1 = second.sys1
    second.load_state(state_dir)

    # Restored at once: the position, the buffer, the random state, the optimizer, the weights.
    assert second.pos == saved_position
    if hasattr(second, "buffer"):
        assert second.buffer.rows == first.buffer.rows and second.buffer.keys == first.buffer.keys and second.buffer.counts == first.buffer.counts
    assert same_rng(second.state()["rng"], first.state()["rng"])
    for a, b in zip(second.optimizer.state_dict()["state"].values(), first.optimizer.state_dict()["state"].values()):
        assert all(torch.equal(a[k], b[k]) for k in a)
    assert params_sha256(second.sys1) == params_sha256(first.sys1)
    assert second.counts["resumes"] == 1
    second.counts["resumes"] = 0   # the one count a straight run does not have

    second.run()
    got = second.state()
    expect, have = comparable(expected), comparable(got)
    have["counts"].pop("resumes")
    expect["counts"].pop("checkpoints_written", None), have["counts"].pop("checkpoints_written", None)
    assert have == expect
    assert same_rng(got["rng"], expected["rng"])
    for a, b in zip(second.sys1.parameters(), straight.sys1.parameters()):
        assert torch.equal(a, b)
    for a, b in zip(second.optimizer.state_dict()["state"].values(), straight.optimizer.state_dict()["state"].values()):
        assert all(torch.equal(a[k], b[k]) for k in a)
    if second.value_head is not None:
        for a, b in zip(second.value_head.parameters(), straight.value_head.parameters()):
            assert torch.equal(a, b)


def test_resume_refuses_other_settings_and_other_weights(tmp_path):
    trainer = ppo_trainer(tmp_path, stop_after_iterations=1, step_budget=10**6)
    trainer.run()
    with tarfile.open(tmp_path / f"{trainer.cfg.stem}-state.tar") as tar:
        tar.extractall(tmp_path / "r", filter="data")
    state = tmp_path / "r" / "state"
    other = ppo_trainer(tmp_path / "o", stop_after_iterations=1, step_budget=10**6, clip=0.3)
    other.sys1 = StubSys1.load(state / "sys1", "cpu")
    with pytest.raises(ValueError, match="clip"):
        other.load_state(state)
    fresh = ppo_trainer(tmp_path / "f", stop_after_iterations=1, step_budget=10**6)   # weights that were not loaded from the state
    with pytest.raises(ValueError, match="weights differ"):
        fresh.load_state(state)


def test_the_state_tar_is_replaced_whole(tmp_path):
    trainer = ppo_trainer(tmp_path, kl=False)
    path = trainer.save_state()
    assert path.exists() and not path.with_name(path.name + ".tmp").exists()
    with tarfile.open(path) as tar:
        names = tar.getnames()
    assert {"state/trainer.pt", "state/optimizer.pt", "state/value_head.pt", "state/sys1/stub.pt"} <= set(names)


def test_a_time_limit_stops_with_a_state_and_an_unfinished_result(tmp_path):
    trainer = ppo_trainer(tmp_path, max_seconds=0.0)
    result = trainer.run()
    assert not result["completed"] and result["stopped_because"] == "max_seconds" and result["temperature"] is None
    assert (tmp_path / f"{trainer.cfg.stem}-state.tar").exists() and not (tmp_path / f"{trainer.cfg.stem}.tar").exists()


# Speed test mode and settings.

def test_speedtest_measures_and_projects_without_a_temperature(tmp_path):
    trainer = ppo_trainer(tmp_path, mode="speedtest")
    result = trainer.run_speedtest()
    assert result["stopped_because"] == "speedtest" and not result["completed"] and result["temperature"] is None
    assert result["steps"] > 0 and result["steps_per_second"] > 0 and result["sys1_inference_rows_per_second"] > 0 and result["state_tar_bytes"] > 0
    assert result["projection"]["commands_per_second_collect_and_update"] > 0 and result["projection"]["step_budget_fitting_8_hours_with_11_dev_plays"] > 0


def test_speedtest_of_the_scheduled_arm_mixes_commands(tmp_path):
    result = scheduled_trainer(tmp_path, mode="speedtest").run_speedtest()
    assert result["iterations"][0]["teacher_prob"] == 0.5


def test_settings_come_from_the_environment_and_bad_ones_stop():
    cfg = Config.from_env({"SEED": "7", "STEP_BUDGET": "50", "REWARD": "R4", "KL": "0", "CLIP": "0.1", "MODE": "smoke", "DEVICE": "cpu"}, topic="arm-rl-model-in-control")
    assert (cfg.seed, cfg.step_budget, cfg.reward, cfg.kl, cfg.clip, cfg.stem, cfg.dtype) == (7, 50, "R4", False, 0.1, "seed7-R4", "fp32")
    assert Config.from_env({"SEED": "1", "MODE": "smoke", "DEVICE": "cuda"}, topic="arm-rl-model-in-control", step_budget=5).dtype == "bf16"
    for bad in ({"REWARD": "R2", "MODE": "smoke"}, {"MODE": "other"}, {"FALLBACK": "other", "MODE": "smoke"}, {"TRACKING": "other", "MODE": "smoke"}, {"MODE": "full"}, {"MODE": "full", "STEP_BUDGET": "5", "INIT_FROM": "jaredpalmer/kev-0.8b"}, {"MODE": "full", "STEP_BUDGET": "5", "INIT_FROM": "stub"}):
        with pytest.raises(ValueError):
            Config.from_env(bad, topic="arm-rl-model-in-control")


def test_a_budget_of_commands_ends_the_run(tmp_path):
    trainer = ppo_trainer(tmp_path, step_budget=7, transitions_per_iter=1, games_per_collect=1)
    result = trainer.run()
    assert result["completed"] and 7 <= result["steps"] <= 7 + trainer.cfg.max_steps


def test_constants_reward_r3_beta_stays_as_the_plan_names_it():
    assert R3.beta == 5.0 and R3.match == 1.0 and R3.w == 1.0 and R3.c == 0.1
