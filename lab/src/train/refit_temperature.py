"""Score dev rows with a trained Sys1, refit its temperature on them and write the value into the checkpoint's head.pt.

    python -m train.refit_temperature --run RUN_DIR --dev dev.jsonl --device cuda --seed 0 --out refit.json
    python -m train.refit_temperature --run RUN_DIR --dev dev.jsonl --device cuda --seed 0 --out speed.json --measure-only --limit 200

The fit is the one Kev's released temperatures use (kev/metrics.py:272 TEMPERATURE_FIT: minimum mean NLL over a 121-point
log grid on 0.25 to 4, on raw logits). Kev's `scripts/calibrate_checkpoint.py` is not in the installed package and refuses
rows it cannot place in its own suites, so this file calls the same functions and writes the same two fields of head.pt.
The out-of-fold estimate holds whole games out, so the gap between in-sample and out-of-fold ECE shows how much the one fitted
number flatters rows it was fitted on. A temperature never changes which option wins.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

LAB_SRC = Path(__file__).resolve().parents[1]
if str(LAB_SRC) not in sys.path:
    sys.path.insert(0, str(LAB_SRC))

import numpy as np  # noqa: E402
import torch  # noqa: E402

DEV_SPLITS = ("dev", "dev_heldout_rooms")
QUESTION_ID = "command"


def read_rows(path, limit=None) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            split = row.get("_meta", {}).get("split")
            if split not in DEV_SPLITS:
                raise AssertionError(f"temperature refit reads dev rows only, got split {split!r}")
            rows.append(row)
            if limit and len(rows) >= limit:
                break
    return rows


def score_rows(sys1, rows, batch: int = 16) -> tuple[list[dict], float]:
    """-> (rows in the shape `kev.metrics` reads, at T = 1, seconds spent scoring)."""
    if sys1.training:
        raise RuntimeError("score dev rows in eval mode")
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["state"]))
    out: list = [None] * len(rows)
    spent = 0.0
    for start in range(0, len(order), batch):
        part = order[start:start + batch]
        prompts = [rows[i]["state"] for i in part]
        options = [list(rows[i]["questions"][QUESTION_ID]["criteria"]) for i in part]
        t = time.perf_counter()
        logits = [z.detach().cpu() for z in sys1.score(prompts, options)]
        spent += time.perf_counter() - t
        for i, opts, z in zip(part, options, logits):
            question = rows[i]["questions"][QUESTION_ID]
            meta = rows[i]["_meta"]
            out[i] = {"p": torch.softmax(z, -1).tolist(), "logits": z.tolist(), "label": opts.index(question["label"]), "type": "choice", "keys": opts,
                      "task": meta.get("task_type") or "alfworld", "source": meta.get("source", "alfworld_expert"), "group": meta["game_id"], "variant": "clean",
                      "inference_temperature": 1.0}
    return out, spent


def per_game(rows: list[dict], temperature: float) -> dict:
    from kev.metrics import nll_at_temperature
    games = defaultdict(lambda: {"rows": 0, "matches": 0, "nll": 0.0})
    for row in rows:
        game = games[row["group"]]
        game["rows"] += 1
        game["matches"] += int(np.argmax(row["p"]) == row["label"])
        game["nll"] += nll_at_temperature(row, temperature)
    return {name: {"rows": g["rows"], "step_match": g["matches"] / g["rows"], "nll": g["nll"] / g["rows"]} for name, g in games.items()}


def fit(rows: list[dict], seed: int, folds: int = 5, samples: int = 200) -> dict:
    from kev.metrics import TEMPERATURE_FIT, TEMPERATURE_FIT_METHOD, cross_validated_temperature, fit_temperature, metrics
    temperature = fit_temperature(rows, **TEMPERATURE_FIT)
    grid_low, grid_high = 0.25, 4.0
    report = {"temperature": temperature, "method": TEMPERATURE_FIT_METHOD, "n": len(rows), "at_grid_edge": bool(np.isclose(temperature, grid_low) or np.isclose(temperature, grid_high)),
              "games": len({row["group"] for row in rows}), "raw": metrics(rows), "fitted": metrics(rows, temperature)}
    if report["games"] >= folds:
        report["cross_validation"] = cross_validated_temperature(rows, folds=folds, seed=seed, samples=samples, **TEMPERATURE_FIT)
    return report


def main(argv=None) -> None:
    from kev.checkpoint import read_meta, write_meta
    from sys1.kev_adapter import KevSys1
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--device", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--measure-only", action="store_true", help="time the scoring and write no temperature")
    a = ap.parse_args(argv)
    torch.manual_seed(a.seed)
    rows = read_rows(a.dev, a.limit or None)
    t = time.time()
    sys1 = KevSys1.load(a.run, a.device)
    load_seconds = time.time() - t
    if a.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    scored, spent = score_rows(sys1, rows, a.batch)
    report = {"rows": len(rows), "batch": a.batch, "load_seconds": round(load_seconds, 1), "score_seconds": round(spent, 2), "infer_rows_per_second": round(len(rows) / spent, 3), "precision": sys1.precision}
    if a.device == "cuda":
        report["cuda_max_allocated_bytes_scoring"] = torch.cuda.max_memory_allocated()
    if not a.measure_only:
        fitted = fit(scored, a.seed)
        meta = read_meta(a.run)
        meta.temperature = fitted["temperature"]
        meta.extra["temperature_fit"] = {k: v for k, v in fitted.items() if k in ("method", "n", "games", "at_grid_edge")} | {"rows": a.dev, "precision": sys1.precision}
        write_meta(a.run, meta)
        report |= {"temperature_fit": fitted, "per_game": per_game(scored, fitted["temperature"]), "aggregates": {"teacher_state_step_match": float(np.mean([np.argmax(r["p"]) == r["label"] for r in scored]))}}
    Path(a.out).write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("per_game",)}, indent=1, default=float)[:3000], flush=True)


if __name__ == "__main__":
    main()
