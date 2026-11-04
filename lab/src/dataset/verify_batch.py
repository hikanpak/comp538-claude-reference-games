"""Check that the real export goes through `kev.train`'s loader and the real Kev-0.8B model. Nothing trains: there is no optimizer, and the forward pass runs without gradients.

    nice -n 10 python -m dataset.verify_batch --export lab/data/sys1-kev/train.jsonl.gz --out lab/results/sys1-training-set/verify-kev-batch.json

Two stages. `kev.train.training_requests` reads the whole export and applies Kev's filter, which must drop no row at the export's `max_state_needed`. Then Kev's own batch functions
(`microbatch_plan`, `encode_batch`, `batch_loss`) run one batch through `DecisionModel` warm-started from Kev-0.8B, on CPU, under a watchdog at 8 GB.
"""
from __future__ import annotations

import argparse
import contextlib
import gzip
import json
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

import torch

from sys1.base import FALLBACK_TEXT
from sys1.check_real_weights import Watchdog, environment

from .provenance import git_state, hardware, sha256_file

BASE = "Qwen/Qwen3.5-0.8B-Base"
BASE_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
INIT_FROM = "jaredpalmer/kev-0.8b@bf75a6a8848ea6960ff2ed108d9ed44c2941174f"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--export", default="lab/data/sys1-kev/train.jsonl.gz")
    ap.add_argument("--out", default="lab/results/sys1-training-set/verify-kev-batch.json")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-gb", type=float, default=8.0)
    ap.add_argument("--loader-only", action="store_true", help="stop after Kev's loader, before any weights load, and write no result")
    a = ap.parse_args(argv)
    dog = Watchdog(a.max_gb, 20)
    dog.start()
    export = Path(a.export)
    report = json.loads((export.parent / (export.name.removesuffix(".gz").removesuffix(".jsonl") + ".report.json")).read_text())
    max_state = report["tokens"]["max_state_needed"]
    import kev.train as kt
    from kev.checkpoint import Checkpoint, Meta
    from kev.device import select
    from kev.model import DecisionModel, load_tokenizer

    with tempfile.TemporaryDirectory(dir=export.parent) as scratch:
        plain = Path(scratch) / "train.jsonl"
        with gzip.open(export, "rb") as src, open(plain, "wb") as dst:
            shutil.copyfileobj(src, dst)
        lines = sum(1 for _ in open(plain, encoding="utf-8"))
        sys.argv = ["kev.train", "--data", str(plain), "--base", BASE, "--base_revision", BASE_REVISION, "--init_from", INIT_FROM, "--epochs", "1", "--lr", "2e-5", "--batch", str(a.batch), "--accum", "1",
                    "--device", "cpu", "--max_state", str(max_state), "--seed", str(a.seed), "--out", str(Path(scratch) / "never-written"), "--p_none", "0", "--p_none_distract", "0", "--p_distract", "0", "--p_none_pair", "0"]
        args = kt.parse_args()
        t0 = time.time()
        tok = load_tokenizer(args.base, revision=args.base_revision)
        requests = kt.training_requests(args, tok, None, [])
        loader_seconds = time.time() - t0
        result = {"topic": "sys1-training-set", "check": "one real batch through kev.train's loader and the model, no training", "export": str(export), "export_sha256": sha256_file(export), "rows_in_file": lines, "rows_after_kev_filter": len(requests), "rows_dropped_by_kev": lines - len(requests),
                  "max_state": max_state, "loader_seconds": round(loader_seconds, 1)}
        print(json.dumps(result), flush=True)
        if a.loader_only:
            return

        rng = random.Random(a.seed)
        rng.shuffle(requests)
        requests = requests[: max(16, a.batch)]
        dev = select(args.device, dtype=torch.float32)
        torch.manual_seed(a.seed)
        t0 = time.time()
        model = DecisionModel(args.base, tok, dev, lora=args.lora, revision=args.base_revision, head_dim=args.head_dim, lora_targets=args.lora_targets, option_isolation=bool(args.option_isolation),
                              special_embeddings=bool(args.special_embeddings), dtype=torch.float32, direct_load=False)
        meta = Meta(base=args.base, base_revision=args.base_revision, lora=args.lora, head_dim=args.head_dim, option_isolation=bool(args.option_isolation), special_embeddings=bool(args.special_embeddings), weights_dtype=args.weights_dtype, holdout=[], weights="lora")
        init = Checkpoint(args.init_from).warm_start(model, meta)
        model.lm.config.use_cache = False
        load_seconds = time.time() - t0
        model.train()
        chunk, _, _ = kt.microbatch_plan(requests, args, 1, 0)[0]
        t0 = time.time()
        batch = kt.encode_batch(model, tok, args, chunk, 0, None)
        with torch.no_grad():
            logits = model.forward_batch([v.enc for v in batch], args.shared_prefix)
            loss, terms = kt.batch_loss(model, args, batch, dev, {}, None, contextlib.nullcontext())
        forward_seconds = time.time() - t0
        rows = []
        for v, per_question in zip(batch, logits):
            q = v.rec["questions"][0]
            z = per_question[0]
            if len(z) != len(q["options"]) or q["options"].count(FALLBACK_TEXT) != 1:
                raise AssertionError(f"{v.request_id}: {len(z)} logits for {len(q['options'])} options")
            rows.append({"id": v.request_id, "tokens": v.tokens, "options": len(q["options"]), "label": q["options"][q["label"]], "family": next(r["_meta"].get("family") for r in chunk if r["_meta"]["id"] == v.request_id), "top_pick_of_kev_08b": q["options"][int(z.argmax())]})
        result |= {"batch": a.batch, "loss_mean_per_row": terms["ce"] / len(batch), "loss_finite": bool(torch.isfinite(loss)), "tensors_warm_started": init["tensors"], "rows": rows, "load_seconds": round(load_seconds, 1), "forward_seconds": round(forward_seconds, 1),
                   "peak_footprint_gb": round(dog.peak / 1024**3, 2), "lowest_system_free_pct": dog.lowest_free_pct, "seed": a.seed, "precision": "fp32 on CPU", "environment": environment(), "hardware": hardware(), **git_state(),
                   "scope": "This shows that the path runs on this Mac. It says nothing about Sys1, and its speed is no L4 number. Kev-0.8B here is as released, before any fine-tuning."}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("rows_in_file", "rows_dropped_by_kev", "max_state", "loss_mean_per_row", "peak_footprint_gb", "forward_seconds", "tensors_warm_started")}), flush=True)


if __name__ == "__main__":
    main()
