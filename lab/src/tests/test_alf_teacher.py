"""The teacher: labels from `policy.act` directly, flags for what the wrappers would have replaced, and tracking that stays true when
somebody else chooses the commands. Played on real training games. The faults of the installed expert are checked here, not assumed."""
import hashlib
import random
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # noqa: E401,E702

import pytest
import textworld
import textworld.gym
from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredExpert, AlfredExpertType, AlfredInfos
from alfworld.agents.expert.handcoded_expert import HandCodedAgent

from alf.env import AlfBatchEnv, ExpertTeacher, _make_stack, expert_seed, held_objects, list_games, receptacles_at_agent

TASK_TYPES = 6


def per_task_type(n_each):
    picked, counts = [], {}
    for spec in list_games("train", limit=400):
        if counts.get(spec.task_type, 0) < n_each:
            counts[spec.task_type] = counts.get(spec.task_type, 0) + 1
            picked.append(spec)
    assert len(counts) == TASK_TYPES
    return picked


@pytest.fixture(scope="module")
def six_games():
    return per_task_type(1)


def bonns_expert(spec):
    """Bonn's stack and seed, with the real AlfredExpert and HandCodedAgent.act, and its ExpertBackend's rule for `help`."""
    random.seed(int(hashlib.sha256(f"alfworld-expert/{spec.uid}".encode()).hexdigest()[:16], 16))
    infos = textworld.EnvInfos(won=True, admissible_commands=True, facts=True, extras=["gamefile", "expert_plan"])
    env_id = textworld.gym.register_games([spec.path], infos, batch_size=1, asynchronous=False, max_episode_steps=50,
                                          wrappers=[AlfredDemangler(shuffle=False), AlfredInfos, AlfredExpert(AlfredExpertType.HANDCODED)])
    env = textworld.gym.make(env_id)
    obs, info = env.reset()
    commands, won = [], False
    for _ in range(50):
        plan = info["extra.expert_plan"][0][0]
        allowed = [c for c in info["admissible_commands"][0] if c != "help"]
        command = plan if plan in allowed else allowed[0]
        commands.append(command)
        obs, _, done, info = env.step([command])
        won = bool(info["won"][0])
        if done[0]:
            break
    env.close()
    return commands, won


def teacher_plays(specs, tracking="foreign", batch_size=6, num_workers=0):
    commands, won = {}, {}
    with AlfBatchEnv(specs, batch_size=batch_size, num_workers=num_workers, tracking=tracking) as env:
        live = {v.index: v for v in env.reset()}
        while live:
            order = {i: v.label.played for i, v in live.items()}
            for i, c in order.items():
                commands.setdefault(i, []).append(c)
            stepped, started = env.step(order)
            live = {}
            for v in stepped:
                if v.done:
                    won[v.index] = v.won
                else:
                    live[v.index] = v
            for v in started:
                live[v.index] = v
    return commands, won


def test_batched_teacher_repeats_bonns_expert_command_by_command(six_games):
    # Six games in one process, so the expert's ties would collide without each game's own random stream.
    commands, won = teacher_plays(six_games)
    for i, spec in enumerate(six_games):
        bonn_commands, bonn_won = bonns_expert(spec)
        assert commands[i] == bonn_commands, spec.uid
        assert won[i] == bonn_won, spec.uid


def test_foreign_tracking_leaves_a_teacher_only_game_alone(six_games):
    assert teacher_plays(six_games, "foreign")[0] == teacher_plays(six_games, "shipped")[0]


def test_the_wrapper_is_never_called(six_games, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("HandCodedAgent.act was called")

    monkeypatch.setattr(HandCodedAgent, "act", refuse)
    _, won = teacher_plays(six_games[:1], batch_size=1)
    assert len(won) == 1


class ScriptedPolicy:
    """Stands for the policy, so a test can state what `act` answers."""

    def __init__(self, answer):
        self.answer = answer
        self.observed = []

    def act(self, state, last_action):
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer

    def observe(self, obs):
        self.observed.append(obs)


ADMISSIBLE = ["examine desk 1", "go to bed 1", "help", "inventory", "look"]


def scripted_teacher(six_games, answer, last_played=None):
    teacher = ExpertTeacher(six_games[0], tracking="shipped")
    teacher.policy = ScriptedPolicy(answer)
    teacher._last_played = last_played
    return teacher


def shipped_chain(answer, last, stream):
    """The real wrappers on the same answer and random stream: HandCodedAgent.act, then AlfredExpert's look."""
    agent = HandCodedAgent()
    agent.policy = ScriptedPolicy(answer)
    saved = random.getstate()
    random.setstate(stream)
    wrapped = agent.act({"admissible_commands": ADMISSIBLE, "feedback": ""}, 0, False, last)
    random.setstate(saved)
    return wrapped if wrapped in ADMISSIBLE else "look"


@pytest.mark.parametrize("answer, last, random_, inadmissible, command", [
    ("go to bed 1", "examine desk 1", False, False, "go to bed 1"),         # A plain admissible answer.
    ("go to bed 1", "go to bed 1", True, False, None),                      # The policy repeats its last command.
    ("go to desk 7", "look", True, True, None),                             # After look, an answer that is not admissible.
    ("go to bed 1", "look", False, False, "go to bed 1"),                   # After look, an admissible answer stands.
    ("close fridge 1", "examine desk 1", False, True, "look"),              # Not admissible and not after look: AlfredExpert's look.
    ("look", "look", True, False, None),                                    # Repeating look is a repeat.
])
def test_flags_agree_with_the_shipped_wrappers(six_games, answer, last, random_, inadmissible, command):
    teacher = scripted_teacher(six_games, answer)
    stream = teacher._stream
    label = teacher.after_step({"admissible_commands": ADMISSIBLE, "feedback": ""}, last)
    assert label.raw == answer
    assert label.label_random is random_
    assert label.label_inadmissible is inadmissible
    assert label.command == command
    shipped = shipped_chain(answer, last, stream)
    assert label.played == (shipped if shipped != "help" else "examine desk 1")


def test_a_draw_of_help_plays_the_first_other_command_as_bonn_does(six_games):
    for attempt in range(200):
        teacher = scripted_teacher(six_games, "look")
        generator = random.Random(attempt)
        teacher._stream = generator.getstate()
        if shipped_chain("look", "look", teacher._stream) == "help":
            assert teacher.after_step({"admissible_commands": ADMISSIBLE, "feedback": ""}, "look").played == "examine desk 1"
            return
    pytest.fail("no stream drew help")


def test_error_in_the_policy_gives_no_label(six_games):
    teacher = scripted_teacher(six_games, IndexError("pop from empty list"))
    label = teacher.after_step({"admissible_commands": ADMISSIBLE, "feedback": ""}, "examine desk 1")
    assert label.command is None and label.raw is None and label.label_error == "IndexError" and label.played == "look"


def test_first_label_is_the_experts_fixed_look_and_only_observes(six_games):
    teacher = scripted_teacher(six_games, "go to bed 1")
    label = teacher.at_reset({"feedback": "-= Welcome to TextWorld, ALFRED! =-\n"})
    assert (label.command, label.played, label.label_initial, label.raw) == ("look", "look", True, None)
    assert len(teacher.policy.observed) == 1


def test_teacher_keeps_the_global_random_stream_untouched(six_games):
    teacher = scripted_teacher(six_games, "go to bed 1")
    random.seed(123)
    before = random.getstate()
    teacher.after_step({"admissible_commands": ADMISSIBLE, "feedback": ""}, "go to bed 1")  # Draws from its own stream.
    assert random.getstate() == before


def test_seed_is_bonns_hash():
    uid = "train/x/y"
    assert expert_seed(uid) == int(hashlib.sha256(b"alfworld-expert/train/x/y").hexdigest()[:16], 16)


def foreign_game(spec, tracking, n_steps=40, seed=0, check=None):
    """Plays a real game with commands chosen by a script that favours take and move, reporting the tracking at every step."""
    stack = _make_stack(50)
    stack.load(spec.path)
    state = stack.reset()
    teacher = ExpertTeacher(spec, tracking)
    teacher.at_reset(state)
    rng = random.Random(f"{seed}/{spec.uid}")
    seen = {"take": 0, "move": 0, "go": 0}
    try:
        for t in range(n_steps):
            allowed = [c for c in state["admissible_commands"] if c != "help"]
            favoured = [c for c in allowed if c.startswith(("take", "move"))]
            command = rng.choice(favoured) if favoured and rng.random() < 0.5 else rng.choice(allowed)
            state, _, done = stack.step(command)
            if done:
                break
            teacher.after_step(state, command)
            for word, key in (("take", "take"), ("move", "move"), ("go to", "go")):
                seen[key] += command.startswith(word)
            if check:
                check(teacher, state, command, seen)
    finally:
        stack.close()
    return seen


def test_fact_readers_agree_with_the_games_own_words(six_games):
    # The repair trusts these two readers, so they are checked against what the game says, on the expert's way through a game.
    spec = next(s for s in six_games if s.task_type == "pick_and_place_simple")
    stack = _make_stack(50)
    stack.load(spec.path)
    state = stack.reset()
    teacher = ExpertTeacher(spec)
    label = teacher.at_reset(state)
    took = moved = False
    for _ in range(50):
        command = label.played
        state, _, done = stack.step(command)
        facts = teacher.policy.get_facts(state)
        if command.startswith("go to"):
            assert command.replace("go to ", "") in receptacles_at_agent(facts)
        if command.startswith("take"):
            taken = " ".join(command.split()[1:3])
            assert held_objects(facts) == [taken] and "You pick up" in state["feedback"]
            took = True
        if command.startswith("move"):
            assert held_objects(facts) == []
            moved = True
        if done:
            break
        label = teacher.after_step(state, command)
    stack.close()
    assert took and moved and state["won"]
    # The `inventory` command's own text names what `holds` names.
    stack = _make_stack(50)
    stack.load(spec.path)
    state = stack.reset()
    assert stack.step("inventory")[0]["feedback"] == "You are not carrying anything."
    stack.close()


def test_tracking_stays_true_under_foreign_commands(six_games):
    totals = {"take": 0, "move": 0, "go": 0, "checked": 0}

    def check(teacher, state, command, seen):
        p = teacher.policy
        facts = p.get_facts(state)
        assert p.inventory == held_objects(facts), (command, state["feedback"])
        if command.startswith("take"):
            assert " ".join(command.split()[1:3]) in p.inventory, (command, state["feedback"])
        if command.startswith("move"):
            assert " ".join(command.split()[1:3]) not in p.inventory, (command, state["feedback"])
        if seen["go"]:
            assert p.curr_recep in receptacles_at_agent(facts), (command, state["feedback"])
        assert all(c in state["admissible_commands"] for c in p.action_backlog)
        if "you see" in state["feedback"] and "you see nothing" not in state["feedback"]:
            assert set(p.visible_objects) <= set(p.get_objects_and_classes(state["feedback"])), command
        totals["checked"] += 1

    for spec in six_games:
        seen = foreign_game(spec, "foreign", check=check)
        for key in seen:
            totals[key] += seen[key]
    # The test has teeth only if the commands that broke the tracking really ran.
    assert totals["take"] >= 6 and totals["move"] >= 6 and totals["go"] >= 6 and totals["checked"] >= 60


def test_shipped_expert_keeps_a_moved_object_in_its_inventory(six_games):
    # Why `tracking` exists: ALFWorld 0.4.2 says `move`, and update_state_tracking looks for `put`.
    stale = []

    def check(teacher, state, command, seen):
        if teacher.policy.inventory != held_objects(teacher.policy.get_facts(state)):
            stale.append(command)

    for spec in six_games:
        foreign_game(spec, "shipped", check=check)
    assert stale, "the installed expert no longer goes stale, so `tracking` should be reviewed"


def test_skipping_a_step_loses_the_tracking(six_games):
    # Why after_step must run at every step: the policy learns of a move only through policy.act.
    spec = six_games[0]
    stack = _make_stack(50)
    stack.load(spec.path)
    state = stack.reset()
    teacher = ExpertTeacher(spec)
    teacher.at_reset(state)
    goes = [c for c in state["admissible_commands"] if c.startswith("go to")]
    state, _, _ = stack.step(goes[0])
    teacher.after_step(state, goes[0])
    assert teacher.policy.curr_recep in receptacles_at_agent(teacher.policy.get_facts(state))
    state, _, _ = stack.step(goes[1])  # The teacher is not told.
    assert teacher.policy.curr_recep not in receptacles_at_agent(teacher.policy.get_facts(state))
    stack.close()


def test_a_failed_go_to_does_not_move_the_tracked_place(six_games):
    def play(tracking):
        spec = six_games[0]
        stack = _make_stack(50)
        stack.load(spec.path)
        state = stack.reset()
        teacher = ExpertTeacher(spec, tracking)
        teacher.at_reset(state)
        go = next(c for c in state["admissible_commands"] if c.startswith("go to"))
        state, _, _ = stack.step(go)
        teacher.after_step(state, go)
        state, _, _ = stack.step("go to nowhere 9")
        assert state["feedback"] == "Nothing happens."
        teacher.after_step(state, "go to nowhere 9")
        place = teacher.policy.curr_recep
        stack.close()
        return place, go.replace("go to ", "")

    place, expected = play("foreign")
    assert place == expected
    place, _ = play("shipped")
    assert place == "nowhere 9"  # The fault this repairs.


def test_foreign_flag_rises_only_on_a_command_the_expert_did_not_play(six_games):
    spec = six_games[0]
    stack = _make_stack(50)
    stack.load(spec.path)
    state = stack.reset()
    teacher = ExpertTeacher(spec)
    label = teacher.at_reset(state)
    for _ in range(3):
        state, _, done = stack.step(label.played)
        label = teacher.after_step(state, label.played)
        assert teacher.foreign is False
    other = next(c for c in state["admissible_commands"] if c not in (label.played, "help"))
    state, _, _ = stack.step(other)
    teacher.after_step(state, other)
    assert teacher.foreign is True
    stack.close()


def test_object_moved_to_the_wrong_place_can_still_be_found(six_games):
    spec = next(s for s in six_games if s.task_type == "pick_and_place_simple")
    teacher = ExpertTeacher(spec)
    target = teacher.policy.task_params["object_target"]
    parent = teacher.policy.task_params["parent_target"]
    teacher._after_move(f"{target} 1", "elsewhere 1")
    assert teacher.policy.object_blacklist == []
    teacher._after_move(f"{target} 1", f"{parent} 1")
    assert teacher.policy.object_blacklist == [f"{target} 1"]


def recovery(specs, tracking, k):
    """k scripted random commands, then the teacher plays. Returns the games won."""
    won = 0
    with AlfBatchEnv(specs, batch_size=8, num_workers=2, tracking=tracking) as env:
        live = {v.index: v for v in env.reset()}
        while live:
            order = {}
            for i, v in live.items():
                order[i] = random.Random(f"{v.uid}/{v.step}").choice(v.admissible) if v.step < k else v.label.played
            stepped, started = env.step(order)
            live = {}
            for v in stepped:
                if v.done:
                    won += v.won
                else:
                    live[v.index] = v
            for v in started:
                live[v.index] = v
    return won


def test_expert_recovers_from_foreign_commands():
    # The outcome that matters for labels on Sys1's own states: from where a stranger left the game, following the label wins.
    specs = per_task_type(2)
    assert len(specs) == 12
    repaired = recovery(specs, "foreign", 10)
    shipped = recovery(specs, "shipped", 10)
    assert repaired >= 10, f"foreign won {repaired} of 12"
    assert repaired > shipped, f"foreign won {repaired}, shipped {shipped}"
