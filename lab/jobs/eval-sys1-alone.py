"""Sys1 alone on a list of games, with no LLM, measured against every row of `lib/sys1-acceptance.md`.

One run is one Sys1, one menu, one game list and one seed. It plays the games three ways:

  - `teacher_states`: the handcoded expert runs every command and Sys1 is scored on the expert's states.
  - `own_masked`: Sys1 runs every command, with fallback masked.
  - `own_unmasked`: Sys1 decides with fallback open. With no LLM to take a fallback, the step runs Sys1's top menu command.

`--aggregate` reads the result files of several seeds and writes the verdict: each gate on the mean over seeds, the worst seed beside
it, the seeds-agree gate, and game-level bootstrap intervals. The thresholds are read from `lib/sys1-acceptance.md` at run time, so the
file stays their only source.

    python lab/jobs/eval-sys1-alone.py --sys1 stub --menu lab/results/menu-statistics/menu.json \\
        --games lab/results/sys1-training-set/dev_heldout_rooms.txt --seed 0 --run stub-seed0
    python lab/jobs/eval-sys1-alone.py --aggregate lab/results/eval-sys1-alone/a-seed*.json --out lab/results/sys1-acceptance/verdict.json

On this Mac keep to `nice -n 10`, at most 4 workers. Test games are refused here, whatever the list says.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import re
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lab" / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from alf.env import GameSpec, load_game_list  # noqa: E402
from harness.control import HarnessConfig  # noqa: E402
from harness.episode import run_episodes  # noqa: E402
from harness.metrics import exposure_gap, success_by_task_type, summarize  # noqa: E402
from harness.reward import REWARDS  # noqa: E402
from harness.stats import paired_bootstrap  # noqa: E402

TOPIC = "eval-sys1-alone"
ACCEPTANCE = ROOT / "lib" / "sys1-acceptance.md"
PACKAGES = ["alfworld", "textworld", "kev", "torch", "transformers", "numpy", "fast-downward-textworld"]

# Each row of the acceptance table, by its name there, and the measurement that fills it.
GATE_MEASURES = {
    "Solves games": "own_masked.success_pct",
    "Does not loop": "own_masked.repeat_rate",
    "Holds up in its own games": "exposure_gap",
    "Confidence is honest": "own_masked.ece",
    "Confidence ranks right above wrong": "own_masked.auroc",
    "Falls back when it should": "own_unmasked.fallback_when_off_menu",
    "Does not fall back needlessly": "own_unmasked.fallback_when_on_menu",
    "Seeds agree": "seeds_agree_points",
}


def read_gates(path: Path = ACCEPTANCE) -> list[dict]:
    """The gate rows of the acceptance file. A row with no measurement, or a measurement with no row, stops the job."""
    rows = []
    in_table = False
    for line in path.read_text().splitlines():
        if line.startswith("| Gate |"):
            in_table = True
            continue
        if in_table:
            if not line.startswith("|"):
                break
            cells = [c.strip() for c in line.strip("|").split("|")]
            if set(cells[0]) <= {"-", " "}:
                continue
            match = re.fullmatch(r"(at least|at most|within) ([0-9.]+)( points)?", cells[2])
            if not match:
                raise ValueError(f"cannot read the threshold of gate {cells[0]!r}: {cells[2]!r}")
            direction = ">=" if match.group(1) == "at least" else "<="
            rows.append({"gate": cells[0], "measures": cells[1], "direction": direction, "threshold": float(match.group(2))})
    names = {r["gate"] for r in rows}
    if names != set(GATE_MEASURES):
        raise ValueError(f"acceptance rows and measurements differ: {sorted(names ^ set(GATE_MEASURES))}")
    return rows


def _lookup(values: dict, dotted: str):
    node = values
    for part in dotted.split("."):
        if node is None or part not in node:
            return None
        node = node[part]
    return node


def judge(value, direction: str, threshold: float) -> bool | None:
    """True or False, or None when the value was not measured. A gate with no measurement never passes silently."""
    if value is None:
        return None
    return value >= threshold if direction == ">=" else value <= threshold


def gates_from(values: dict) -> list[dict]:
    out = []
    for row in read_gates():
        value = _lookup(values, GATE_MEASURES[row["gate"]])
        out.append({**row, "source": GATE_MEASURES[row["gate"]], "value": value, "passed": judge(value, row["direction"], row["threshold"])})
    return out


def _summary(logs, view=None) -> dict:
    s = summarize(logs, view)
    s["success_pct"] = None if s["success"] is None else 100.0 * s["success"]
    return s


def evaluate(sys1, menu, games: list, cfg: HarnessConfig, seed: int, batch_size: int = 8, num_workers: int = 0, tracking: str = "foreign") -> dict:
    """The three plays, and everything the acceptance table and the plan's Q2 and Q3 tables read."""
    common = dict(seed=seed, batch_size=batch_size, num_workers=num_workers, tracking=tracking, keep_prompts=False)
    timings: dict[str, dict] = {"teacher_states": {}, "own_masked": {}, "own_unmasked": {}}
    t0 = time.perf_counter()
    on_teacher = run_episodes(games, sys1, None, menu, cfg, "teacher", teacher="expert", timing=timings["teacher_states"], **common)
    own_masked = run_episodes(games, sys1, None, menu, cfg, "test", teacher="expert", mask_fallback=True, timing=timings["own_masked"], **common)
    own_unmasked = run_episodes(games, sys1, None, menu, cfg, "test", teacher="expert", mask_fallback=False, timing=timings["own_unmasked"], **common)
    wall = time.perf_counter() - t0

    aggregates = {
        # The masked view matches own_masked, and it is the one the exposure gap reads. The expert's own win rate sits in `success`.
        "teacher_states": _summary(on_teacher, "masked"),
        "teacher_states_fallback_open": _summary(on_teacher, "unmasked"),
        "own_masked": _summary(own_masked),
        "own_unmasked": _summary(own_unmasked),
    }
    values = {**aggregates, "exposure_gap": exposure_gap(on_teacher, own_masked, "masked"), "seeds_agree_points": None}
    step_seconds = [t for timing in timings.values() for t in timing["batched_step_seconds"]]
    return {
        "aggregates": aggregates,
        "exposure_gap": values["exposure_gap"],
        "gates": gates_from(values),
        "success_by_task_type": {"teacher_states": success_by_task_type(on_teacher), "own_masked": success_by_task_type(own_masked), "own_unmasked": success_by_task_type(own_unmasked)},
        "per_game": {"teacher_states": [e.outcome() for e in on_teacher], "own_masked": [e.outcome() for e in own_masked], "own_unmasked": [e.outcome() for e in own_unmasked]},
        "timing": {
            "wall_seconds": wall,
            "batch_size": timings["own_masked"]["batch_size"],
            "num_workers": timings["own_masked"]["num_workers"],
            "batched_env_step_seconds_mean": float(np.mean(step_seconds)) if step_seconds else None,
            "batched_env_step_seconds_p95": float(np.percentile(step_seconds, 95)) if step_seconds else None,
            "n_batched_env_steps": len(step_seconds),
            "game_load_batches": sum(len(timing["load_seconds"]) for timing in timings.values()),
            "game_load_seconds_total": sum(sum(timing["load_seconds"]) for timing in timings.values()),
        },
    }


def expert_through_harness(games: list, cfg: HarnessConfig, seed: int, batch_size: int = 8, num_workers: int = 0, tracking: str = "foreign") -> dict:
    """The teacher alone, with no Sys1 and no menu. Reads the harness itself: what the expert wins, and what a batched step costs."""
    timing: dict = {}
    t0 = time.perf_counter()
    logs = run_episodes(games, None, None, None, cfg, "teacher", seed=seed, batch_size=batch_size, num_workers=num_workers, tracking=tracking, timing=timing)
    steps = timing["batched_step_seconds"]
    return {
        "aggregates": {"expert": _summary(logs)},
        "success_by_task_type": {"expert": success_by_task_type(logs)},
        "per_game": {"expert": [e.outcome() for e in logs]},
        "timing": {
            "wall_seconds": time.perf_counter() - t0,
            "batch_size": timing["batch_size"],
            "num_workers": timing["num_workers"],
            "batched_env_step_seconds_mean": float(np.mean(steps)),
            "batched_env_step_seconds_p95": float(np.percentile(steps, 95)),
            "n_batched_env_steps": len(steps),
            "game_load_batches": len(timing["load_seconds"]),
            "game_load_seconds_total": sum(timing["load_seconds"]),
        },
    }


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def provenance(device: str) -> dict:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    hardware = {"device": device, "machine": platform.machine(), "processor": platform.processor(), "cpu_count": os.cpu_count()}
    if device.startswith("cuda") and torch.cuda.is_available():
        hardware["gpu"] = torch.cuda.get_device_name(0)
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--", "lab/src", "lab/jobs")),
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "packages": versions, "alfworld_data": os.environ.get("ALFWORLD_DATA")},
        "hardware": hardware,
    }


def peak_memory_bytes() -> dict:
    """ru_maxrss is bytes on macOS and kilobytes on Linux."""
    scale = 1 if sys.platform == "darwin" else 1024
    return {"self": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale, "largest_child": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * scale}


def resolve_device(requested: str) -> str:
    if requested != "auto":
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise SystemExit("cuda was asked for, and torch sees no CUDA device here")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_sys1(spec: str, device: str, seed: int):
    """`stub`, or a checkpoint folder. A StubSys1 checkpoint holds stub.pt, and anything else goes to KevSys1."""
    if spec == "stub":
        from sys1.stub import StubSys1

        model = StubSys1(seed=seed, device=device)
        model.eval()
        return model
    if (Path(spec) / "stub.pt").exists():
        from sys1.stub import StubSys1

        return StubSys1.load(spec, device)
    from sys1.kev_adapter import KevSys1

    return KevSys1.load(spec, device)


def load_menu(path: str):
    try:
        from menu import MenuModel
    except ImportError as e:
        raise SystemExit(f"lab/src/menu.py is not available yet, and this job needs its MenuModel: {e}")
    return MenuModel.load(path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def file_sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, default=str) + "\n")
    tmp.replace(path)


def aggregate(paths: list[str], n_boot: int = 10000) -> dict:
    """The verdict over seeds: mean, worst seed, the seeds-agree gate, and intervals from the paired bootstrap on games."""
    runs = [json.loads(Path(p).read_text()) for p in paths]
    seeds = [r["seed"] for r in runs]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"repeated seed in {seeds}")
    gates = []
    means = {}
    for row in read_gates():
        source = GATE_MEASURES[row["gate"]]
        if source == "seeds_agree_points":
            continue
        per_seed = [_lookup({**r["aggregates"], "exposure_gap": r["exposure_gap"]}, source) for r in runs]
        measured = [v for v in per_seed if v is not None]
        value = float(np.mean(measured)) if len(measured) == len(runs) else None
        worst = None if value is None else (min(measured) if row["direction"] == ">=" else max(measured))
        means[row["gate"]] = value
        gates.append({**row, "source": source, "value": value, "per_seed": per_seed, "worst_seed": worst, "passed": judge(value, row["direction"], row["threshold"])})
    success = [r["aggregates"]["own_masked"]["success_pct"] for r in runs]
    points = float(np.mean(success) - np.min(success))
    row = next(r for r in read_gates() if r["gate"] == "Seeds agree")
    gates.append({**row, "source": "seeds_agree_points", "value": points, "per_seed": success, "worst_seed": float(np.min(success)), "passed": judge(points, row["direction"], row["threshold"])})

    def wins(play: str) -> dict:
        table: dict[str, list[float]] = {}
        for r in runs:
            for game in r["per_game"][play]:
                table.setdefault(game["uid"], []).append(float(game["won"]))
        return table

    intervals = paired_bootstrap({"own_masked": wins("own_masked"), "own_unmasked": wins("own_unmasked"), "teacher_states": wins("teacher_states")}, n_boot=n_boot)
    return {
        "topic": TOPIC,
        "seeds": seeds,
        "runs": [str(p) for p in paths],
        "git_commit": [r.get("git_commit") for r in runs],
        "gates": gates,
        "passed": all(g["passed"] is True for g in gates),
        "not_measured": [g["gate"] for g in gates if g["passed"] is None],
        "bootstrap_on_games": intervals,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sys1", help="`stub`, or a checkpoint folder")
    p.add_argument("--expert-only", action="store_true", help="play the expert through the harness, with no Sys1 and no menu")
    p.add_argument("--menu", help="a file MenuModel.load reads")
    p.add_argument("--games", help="a game list: JSON list, or one uid or path per line")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run", help="the result's file name, without .json")
    p.add_argument("--topic", default=TOPIC)
    p.add_argument("--results-root", default=str(ROOT / "lab" / "results"))
    p.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    p.add_argument("--k", type=int, default=10, help="menu length, 0 for all admissible commands")
    p.add_argument("--fallbacks-before-handover", type=int, default=2)
    p.add_argument("--llm-rounds", type=int, default=10)
    p.add_argument("--menu-run", type=int, default=3)
    p.add_argument("--trace-window", type=int, default=5)
    p.add_argument("--reward", default="R2", choices=sorted(REWARDS))
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=0, help="at most 4")
    p.add_argument("--tracking", default="foreign", choices=["shipped", "foreign", "always"])
    p.add_argument("--aggregate", nargs="+", metavar="RESULT", help="result files of several seeds, instead of playing")
    p.add_argument("--out", help="with --aggregate, where the verdict goes")
    return p


def main(argv=None) -> Path:
    args = build_parser().parse_args(argv)
    if args.aggregate:
        if not args.out:
            raise SystemExit("--aggregate needs --out")
        out = Path(args.out)
        write_json(out, aggregate(args.aggregate))
        return out
    needed = ("games", "run") if args.expert_only else ("sys1", "menu", "games", "run")
    missing = [name for name in needed if not getattr(args, name)]
    if missing:
        raise SystemExit(f"missing --{', --'.join(missing)}")

    seed_everything(args.seed)
    device = resolve_device(args.device)
    games: list[GameSpec] = load_game_list(args.games)
    reward = REWARDS[args.reward]
    cfg = HarnessConfig(k=args.k or None, fallbacks_before_handover=args.fallbacks_before_handover, llm_rounds=args.llm_rounds, menu_run=args.menu_run,
                        trace_window=args.trace_window, w=reward.w, c=reward.c, beta=reward.beta, max_steps=args.max_steps, match=reward.match)
    if args.expert_only:
        sys1, result = None, expert_through_harness(games, cfg, args.seed, batch_size=args.batch_size, num_workers=args.workers, tracking=args.tracking)
    else:
        sys1 = load_sys1(args.sys1, device, args.seed)
        menu = load_menu(args.menu)
        result = evaluate(sys1, menu, games, cfg, args.seed, batch_size=args.batch_size, num_workers=args.workers, tracking=args.tracking)

    payload = {
        "topic": args.topic,
        "run": args.run,
        "config": {**dataclasses.asdict(cfg), "reward": args.reward, "batch_size": args.batch_size, "workers": args.workers, "tracking": args.tracking, "expert_only": args.expert_only,
                   "sys1": args.sys1, "menu": args.menu, "games": args.games, "n_games": len(games), "games_sha256": file_sha256(args.games), "menu_sha256": file_sha256(args.menu) if args.menu else None,
                   "argv": sys.argv if argv is None else list(argv)},
        "seed": args.seed,
        "deterministic_algorithms": False,
        **provenance(device),
        "precision": "no model" if sys1 is None else getattr(sys1, "precision", str(next(sys1.parameters()).dtype)),
        "sys1_temperature": None if sys1 is None else float(sys1.temperature),
        "peak_memory_bytes": peak_memory_bytes(),
        "acceptance_file_sha256": file_sha256(str(ACCEPTANCE)),
        **result,
    }
    out = Path(args.results_root) / args.topic / f"{args.run}.json"
    write_json(out, payload)
    print(f"wrote {out}")
    for gate in payload.get("gates", []):
        print(f"  {gate['gate']}: {gate['value']} needs {gate['direction']} {gate['threshold']}: {gate['passed']}")
    if args.expert_only:
        print(f"  expert success {payload['aggregates']['expert']['success_pct']:.1f} on {len(games)} games, tracking {args.tracking}")
    print(f"  wall {payload['timing']['wall_seconds']:.1f} s, one batched env step {payload['timing']['batched_env_step_seconds_mean']:.3f} s on {payload['timing']['batch_size']} games")
    return out


if __name__ == "__main__":
    main()
