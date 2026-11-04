"""The menu: the top k admissible commands by how often the teacher used commands of that form on `train` games of the same task type.

    model = fit(records)                            # records of split `train` only, which the fit asserts
    model.menu("pick_heat_then_place_in_recep", admissible, 10)
    model.save(path); MenuModel.load(path)

A command is reduced to its form: its kind (`go to`, `take`, `move`, `open`, `heat`, ...), its object class and its place class,
with instance numbers stripped, so `take apple 3 from fridge 1` becomes (`take`, `apple`, `fridge`). ALFWorld 0.4.2 places an object
with `move X to Y`, and the parser reads that form.

Two ranking rules, named by `rule`:

  - `form`, the plan's rule: the score is the count of that kind with that object class and place class. Ties go to the count of the
    kind with the place class, then the count of the kind alone, then the command's natural order, so `cabinet 2` comes before `cabinet 10`.
  - `kind`, the weaker variant kept for comparison: the score is the count of the kind alone in that task type. Ties go to natural order.

The menu sees only the task type and the admissible commands. It does not read the goal text, so it cannot prefer the object the task names.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

RULES = ("form", "kind")
TEST_MARKER = "valid_unseen"
FLAGS = ("label_random", "label_inadmissible", "label_error")
_NUMBER = re.compile(r"(\d+)")
_SEPARATOR = "|"
_TWO_PART = {"take": " from ", "move": " to ", "put": " in/on ", "heat": " with ", "cool": " with ", "clean": " with ", "slice": " with "}
_ONE_PART = ("open", "close", "examine", "use")


def class_of(name: str) -> str:
    """`apple 3` to `apple`. The instance number is the only thing removed."""
    return " ".join(w for w in name.split() if not w.isdigit())


@lru_cache(maxsize=200_000)
def command_form(command: str) -> tuple[str, str | None, str | None]:
    """(kind, object class, place class). A command this parser does not know keeps its first word as the kind."""
    if command.startswith("go to "):
        return "go to", None, class_of(command[len("go to "):])
    first, _, rest = command.partition(" ")
    if first == "put":
        for word in (" in/on ", " in ", " on "):
            if word in rest:
                obj, place = rest.split(word, 1)
                return "put", class_of(obj), class_of(place)
    separator = _TWO_PART.get(first)
    if separator and separator in rest:
        obj, place = rest.split(separator, 1)
        return first, class_of(obj), class_of(place)
    if first in _ONE_PART and rest:
        return first, None, class_of(rest)
    return first, None, None


def natural_key(command: str) -> tuple:
    return tuple(int(part) if part.isdigit() else part for part in _NUMBER.split(command))


def _encode(form: tuple) -> str:
    return _SEPARATOR.join(part or "" for part in form)


def check_train(record: dict) -> None:
    """The assert `lab/INTERFACES.md` asks of `menu.fit`, written as a raise so `python -O` keeps it."""
    game = str(record.get("game_id", ""))
    if TEST_MARKER in game:
        raise AssertionError(f"test game in the menu fit: {game}")
    if record.get("split") != "train":
        raise AssertionError(f"menu.fit reads split 'train' only, got {record.get('split')!r} at {game} step {record.get('step')}")


class MenuModel:
    def __init__(self, counts: dict[str, dict[str, int]], rule: str = "form", meta: dict | None = None):
        if rule not in RULES:
            raise ValueError(f"rule must be one of {RULES}, got {rule!r}")
        self.rule, self.counts, self.meta = rule, counts, meta or {}
        self._tables = {task: self._tables_of(table) for task, table in counts.items()}

    @staticmethod
    def _tables_of(table: dict[str, int]) -> tuple[dict, Counter, Counter]:
        by_place, by_kind = Counter(), Counter()
        for key, count in table.items():
            kind, _, place = key.split(_SEPARATOR)
            by_place[(kind, place)] += count
            by_kind[kind] += count
        return table, by_place, by_kind

    def _sort_key(self, tables, command: str) -> tuple:
        table, by_place, by_kind = tables
        form = command_form(command)
        if self.rule == "kind":
            return (-by_kind[form[0]], natural_key(command))
        return (-table.get(_encode(form), 0), -by_place[(form[0], form[2] or "")], -by_kind[form[0]], natural_key(command))

    def score(self, task_type: str, command: str) -> int:
        """The count the rule ranks by."""
        table, _, by_kind = self._lookup(task_type)
        form = command_form(command)
        return by_kind[form[0]] if self.rule == "kind" else table.get(_encode(form), 0)

    def _lookup(self, task_type: str):
        try:
            return self._tables[task_type]
        except KeyError:
            raise ValueError(f"no menu was fitted for task type {task_type!r}") from None

    def menu(self, task_type: str, admissible, k: int | None) -> list[str]:
        """The top `k` admissible commands in score order. `k=None` gives all of them. Deterministic, and the input order does not matter."""
        if k is not None and k < 1:
            raise ValueError(f"k must be positive or None, got {k}")
        tables = self._lookup(task_type)
        ordered = sorted(set(admissible), key=lambda c: self._sort_key(tables, c))
        return ordered if k is None else ordered[:k]

    def rank(self, task_type: str, admissible, command: str | None) -> int | None:
        """The 1-based position of `command` in the full ordering, so it is on a menu of length k when rank <= k. None when it is not admissible."""
        if command is None:
            return None
        ordered = self.menu(task_type, admissible, None)
        return ordered.index(command) + 1 if command in ordered else None

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        body = {"rule": self.rule, "meta": self.meta, "counts": {task: dict(sorted(table.items())) for task, table in sorted(self.counts.items())}}
        path.write_text(json.dumps(body, indent=1, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path) -> "MenuModel":
        body = json.loads(Path(path).read_text())
        return cls(body["counts"], body["rule"], body.get("meta"))


def fit(records, rule: str = "form", meta: dict | None = None) -> MenuModel:
    """Count the teacher's commands in `records`. Every record must have split `train`. A record with no label, or a label flagged random,
    inadmissible or error, adds nothing and is counted in `meta`."""
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}, got {rule!r}")
    counts: dict[str, Counter] = {}
    used = skipped = 0
    for record in records:
        check_train(record)
        label = record.get("expert")
        if not label or any(record.get(flag) for flag in FLAGS):
            skipped += 1
            continue
        counts.setdefault(record["task_type"], Counter())[_encode(command_form(label))] += 1
        used += 1
    if not counts:
        raise ValueError("menu.fit got no usable record")
    return MenuModel({task: dict(table) for task, table in counts.items()}, rule, {**(meta or {}), "records_used": used, "records_skipped_flagged": skipped})
