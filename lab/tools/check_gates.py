"""Checks behind GATES.md. Each subcommand prints a token only after every assertion passes."""
import csv
import gzip
import hashlib
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS = ROOT / "lab" / "results"
ACCEPTANCE_RESULT = RESULTS / "sys1-acceptance" / "verdict.json"
LEDGER = RESULTS / "colab-ledger.csv"
SYS1_CU_CAP, GEMMA_CU_CAP = 70.0, 30.0


def fail(msg):
    print(f"FAILED: {msg}")
    sys.exit(1)


def knowledge():
    for name in ["CONTEXT.md", "lib/execution-plan.md", "lib/work-log.md", "lib/llm-use-log.md",
                 "lib/literature/related-work.md", "lib/literature/positioning.md", "lab/INTERFACES.md"]:
        p = ROOT / name
        if not p.exists():
            fail(f"missing {name}")
        if re.search(r"\bFILL\b", p.read_text()):
            fail(f"placeholder left in {name}")
    print("knowledge recorded")


def critique():
    p = ROOT / "lib/issues/2026-11-03-claude-execution-critique.md"
    if not p.exists():
        fail("no critique file")
    rows = [l for l in p.read_text().splitlines() if re.match(r"\| \d+ \|", l)]
    if len(rows) != 16 or any(not re.search(r"Adopted|Moved", r) for r in rows):
        fail(f"{len(rows)} rows, each needs a decision")
    print("critique recorded with decisions")


def first_commit(path):
    out = subprocess.run(["git", "log", "--reverse", "--format=%ct", "--", path], cwd=ROOT,
                         capture_output=True, text=True).stdout.split()
    return int(out[0]) if out else None


def gates_first():
    a = first_commit("lib/sys1-acceptance.md")
    t = first_commit("lab/results/arm-classification")
    if a is None:
        fail("acceptance file not committed")
    if t is None:
        fail("no training result committed yet")
    if t < a:
        fail("training result older than the acceptance gates")
    print("gates predate training")


def dataset():
    d = RESULTS / "sys1-training-set"
    m = json.loads((d / "manifest.json").read_text())
    rows = {"train": 0}
    rooms = {"train": set(), "dev_heldout_rooms": set()}
    room_re = re.compile(r"-(\d+)/trial_")
    for shard in m["shards"]:
        p = d / shard["path"]
        if hashlib.sha256(p.read_bytes()).hexdigest() != shard["sha256"]:
            fail(f"hash mismatch {p.name}")
        with gzip.open(p, "rt") as f:
            for line in f:
                r = json.loads(line)
                split = r["split"]
                if "valid_unseen" in r["game_id"]:
                    fail(f"test game in {p.name}")
                rows[split] = rows.get(split, 0) + 1
                g = room_re.search(r["game_id"])
                if g and split in rooms:
                    rooms[split].add(g.group(1))
    if rooms["train"] & rooms["dev_heldout_rooms"]:
        fail("room overlap between train and dev_heldout_rooms")
    if rows["train"] < 1000 or rows.get("dev_heldout_rooms", 0) == 0:
        fail(f"row counts too small {rows}")
    if not (d / "CARD.md").exists() or not (d / "audit.json").exists():
        fail("card or audit missing")
    print("training set verified", rows)


def harness_tests():
    r = subprocess.run([str(ROOT / ".venv/bin/python"), "-m", "pytest", "-q", "lab/src/tests"], cwd=ROOT,
                       capture_output=True, text=True)
    if r.returncode != 0:
        fail(r.stdout[-800:])
    if not re.search(r"\d+ passed", r.stdout):
        fail("no tests ran")
    print("code tests passed")


def trained():
    runs = list((RESULTS / "arm-classification").glob("*.json"))
    seeds = {json.loads(p.read_text()).get("seed") for p in runs}
    if len(seeds - {None}) < 1:
        fail("no classification run with a seed")
    print("sys1 trained", sorted(seeds - {None}))


def verdict():
    if not ACCEPTANCE_RESULT.exists():
        fail("no verdict file")
    v = json.loads(ACCEPTANCE_RESULT.read_text())
    if len(v.get("seeds", [])) < 3 or "gates" not in v or "passed" not in v:
        fail("verdict needs 3 seeds, the gates and a pass flag")
    print("acceptance evaluated, passed =", v["passed"])


def ledger():
    sys1 = gemma = 0.0
    gemma_seen_at = None
    if LEDGER.exists():
        with LEDGER.open() as f:
            for i, row in enumerate(csv.DictReader(f)):
                if row["charged_to"] == "gemma":
                    gemma += float(row["cu"])
                    gemma_seen_at = i if gemma_seen_at is None else gemma_seen_at
                else:
                    sys1 += float(row["cu"])
    if sys1 > SYS1_CU_CAP or gemma > GEMMA_CU_CAP:
        fail(f"over cap sys1={sys1} gemma={gemma}")
    if gemma_seen_at is not None:
        if not ACCEPTANCE_RESULT.exists() or not json.loads(ACCEPTANCE_RESULT.read_text())["passed"]:
            fail("Gemma spending before Sys1 passed its gates")
    print(f"ledger within caps sys1={sys1:.1f} gemma={gemma:.1f}")


def report():
    reports = list((ROOT / "lib/reports").glob("*.pdf"))
    if not reports:
        fail("no report PDF")
    print("report written", reports[-1].name)


if __name__ == "__main__":
    {"knowledge": knowledge, "critique": critique, "gates_first": gates_first, "dataset": dataset,
     "tests": harness_tests, "trained": trained, "verdict": verdict, "ledger": ledger,
     "report": report}[sys.argv[1]]()
