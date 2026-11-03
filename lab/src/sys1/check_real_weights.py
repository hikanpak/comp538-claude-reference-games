"""Check KevSys1 on the real Kev-0.8B weights and measure rows per second and peak memory on this machine.

    nice -n 10 .venv/bin/python lab/src/sys1/check_real_weights.py --device mps --stage train --out X.json
    nice -n 10 .venv/bin/python lab/src/sys1/check_real_weights.py --device mps --stage infer --out Y.json

Each stage runs in its own process so its peak belongs to that stage. A watchdog thread samples the process's physical
footprint (which counts Metal buffers, where RSS does not) and the system's free memory share, and it kills the process
above --max-gb or below --min-free-pct. Prompts are written here to the shape of an ALFWorld state. They come from no game
file, so a speed read off them is a Mac number on invented text, and it is not an L4 number.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from sys1.base import FALLBACK_TEXT, QUESTION_TEXT  # noqa: E402
from sys1.kev_adapter import KEV_08B, KevSys1  # noqa: E402

GB = 1024**3
PLACES = ["cabinet", "countertop", "drawer", "fridge", "garbagecan", "microwave", "shelf", "sinkbasin", "stoveburner", "toaster"]
THINGS = ["apple", "bowl", "bread", "butterknife", "egg", "fork", "kettle", "knife", "lettuce", "mug", "pan", "plate", "potato", "soapbottle", "spoon", "tomato"]


def footprint_bytes() -> int:
    """ri_phys_footprint from proc_pid_rusage (rusage_info_v0: a 16-byte uuid, then u64 fields, footprint the eighth)."""
    buf = (ctypes.c_uint64 * 40)()
    ctypes.CDLL("libproc.dylib").proc_pid_rusage(os.getpid(), 2, ctypes.byref(buf))
    return int(buf[2 + 7])


class Watchdog(threading.Thread):
    def __init__(self, max_gb: float, min_free_pct: int):
        super().__init__(daemon=True)
        self.max_bytes, self.min_free_pct, self.peak = max_gb * GB, min_free_pct, 0
        self.lowest_free_pct, self.stage_peak = 100, 0

    def reset_stage(self):
        self.stage_peak = 0

    def run(self):
        ticks = 0
        while True:
            now = footprint_bytes()
            self.peak, self.stage_peak = max(self.peak, now), max(self.stage_peak, now)
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss   # bytes on macOS; counts mapped weight files, which footprint does not
            if max(now, rss) > self.max_bytes:
                print(f"WATCHDOG footprint {now / GB:.2f} GB, RSS high-water {rss / GB:.2f} GB, limit {self.max_bytes / GB:.1f} GB, exiting", flush=True)
                os._exit(3)
            if ticks % 10 == 0:
                free = int(subprocess.run(["sysctl", "-n", "kern.memorystatus_level"], capture_output=True, text=True).stdout.strip() or 100)
                self.lowest_free_pct = min(self.lowest_free_pct, free)
                if free < self.min_free_pct:
                    print(f"WATCHDOG system free {free}% under {self.min_free_pct}%, exiting", flush=True)
                    os._exit(3)
            ticks += 1
            time.sleep(0.2)


def sample_prompt(i: int, steps: int) -> tuple[str, list[str], int]:
    """A state of the shape render_prompt is specified to build, its menu with the fallback last, and the label's index."""
    place = lambda j: f"{PLACES[j % len(PLACES)]} {j % 6 + 1}"   # noqa: E731
    thing = THINGS[i % len(THINGS)]
    first = "You are in the middle of a room. Looking quickly around you, you see " + ", ".join(f"a {place(i + j)}" for j in range(14)) + "."
    goal = f"Your task is to: put a hot {thing} in {place(i + 3)}."
    trace = []
    for s in range(steps):
        trace.append(f"> go to {place(i + s)}\nYou arrive at {place(i + s)}. On the {place(i + s)}, you see a {THINGS[(i + s) % len(THINGS)]} {s + 1}, and a {THINGS[(i + s + 5) % len(THINGS)]} {s + 2}.")
    state = "\n".join([goal, first, *trace[-5:], "Control: Sys1, fallbacks in a row 0, LLM rounds left 0."])
    k = 3 + (i * 4) % 9   # 3 to 11 commands, so the option count differs between rows
    options = [f"go to {place(i + j)}" for j in range(k)] + [FALLBACK_TEXT]
    return state, options, i % len(options)


TRACE_STEPS = [5]


def batch(start: int, n: int):
    rows = [sample_prompt(start + i, TRACE_STEPS[0]) for i in range(n)]
    return [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows]


def device_sync(device: str):
    if device == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()   # kev.train does this per step on MPS (kev/train.py:638)


def environment() -> dict:
    import kev  # noqa: F401
    import peft
    import transformers
    versions = {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__}
    try:
        from importlib.metadata import version
        versions["kev"] = version("kev")
    except Exception as error:
        versions["kev"] = f"unknown: {error}"
    return versions


def stage_train(a, dog) -> dict:
    t0 = time.perf_counter()
    sys1 = KevSys1.load(a.checkpoint, a.device, trainable=True)
    load_seconds = time.perf_counter() - t0
    load_peak = dog.stage_peak
    sys1.train()
    assert sys1.temperature == 1.0 and sys1.model.head.temperature == 1.0
    named = dict(sys1.model.named_parameters())
    trainable = {k: p for k, p in named.items() if p.requires_grad}
    frozen_with_grad_required = [k for k in trainable if "lora_" not in k and not k.startswith("head.")]
    assert not frozen_with_grad_required, frozen_with_grad_required[:5]

    def one_step(prompts, options, labels):
        logits = sys1.score(prompts, options)
        assert [len(z) for z in logits] == [len(o) for o in options], "one logit per option"
        loss = sum(torch.nn.functional.cross_entropy(z[None], torch.tensor([y], device=a.device)) for z, y in zip(logits, labels)) / len(labels)
        loss.backward()
        return loss, logits

    # first step: gradients are inspected, nothing is applied, and a second batch must take other option counts
    prompts, options, labels = batch(0, a.train_batch)
    dog.reset_stage()
    sys1.model.zero_grad(set_to_none=True)
    loss, logits = one_step(prompts, options, labels)
    grads = {k: (p.grad.detach().abs().sum().item() if p.grad is not None else None) for k, p in trainable.items()}
    lora = {k: v for k, v in grads.items() if "lora_" in k}
    head = {k: v for k, v in grads.items() if k.startswith("head.")}
    lora_nonzero, head_nonzero = sum(1 for v in lora.values() if v), sum(1 for v in head.values() if v)
    frozen_got_grad = [k for k, p in named.items() if not p.requires_grad and p.grad is not None]
    assert not frozen_got_grad, frozen_got_grad[:5]
    grad_norm_lora = float(torch.sqrt(sum(p.grad.detach().pow(2).sum() for k, p in trainable.items() if "lora_" in k and p.grad is not None)).cpu())
    grad_norm_head = float(torch.sqrt(sum(p.grad.detach().pow(2).sum() for k, p in trainable.items() if k.startswith("head.") and p.grad is not None)).cpu())
    zero_lora_names = sorted(k for k, v in lora.items() if not v)
    sys1.model.zero_grad(set_to_none=True)
    device_sync(a.device)

    second_prompts, second_options, second_labels = batch(a.train_batch, a.train_batch)
    k_first, k_second = [len(o) for o in options], [len(o) for o in second_options]
    assert k_first != k_second, "the check needs differing option counts"
    optimizer = torch.optim.AdamW(sys1.trainable_parameters(), lr=2e-5)
    seconds, tokens = [], []
    for step in range(a.train_steps):
        p, o, y = batch(step * a.train_batch, a.train_batch)
        t = time.perf_counter()
        loss_value, _ = one_step(p, o, y)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        device_sync(a.device)
        seconds.append(time.perf_counter() - t)
        tokens.append(sum(len(sys1._encode(x, z)["ids"]) for x, z in zip(p, o)))
        print(f"train step {step}: {seconds[-1]:.2f} s, loss {loss_value.item():.3f}, tokens {tokens[-1]}", flush=True)
    warm = seconds[1:] or seconds
    return {
        "load_seconds": round(load_seconds, 1), "load_peak_footprint_gb": round(load_peak / GB, 2),
        "warm_start_tensors": sys1.provenance.get("tensors"), "resolved": sys1.provenance.get("resolved"),
        "trainable_parameters_millions": round(sum(p.numel() for p in trainable.values()) / 1e6, 2),
        "trainable_tensors": {"lora": len(lora), "head": len(head)},
        "nonzero_grad_tensors": {"lora": lora_nonzero, "head": head_nonzero},
        "zero_grad_lora_tensors": zero_lora_names[:12], "zero_grad_lora_tensor_count": len(zero_lora_names),
        "grad_norm": {"lora": round(grad_norm_lora, 6), "head": round(grad_norm_head, 6)},
        "first_step_loss": round(loss.item(), 4), "option_counts_first_batch": k_first, "option_counts_second_batch": k_second,
        "logit_shapes_first_batch": [len(z) for z in logits],
        "train_batch": a.train_batch, "train_steps": a.train_steps, "step_seconds": [round(s, 2) for s in seconds],
        "tokens_per_batch": tokens, "train_rows_per_second_after_first_step": round(a.train_batch * len(warm) / sum(warm), 3),
        "train_stage_peak_footprint_gb": round(dog.stage_peak / GB, 2), "precision": sys1.precision,
    }


def stage_infer(a, dog) -> dict:
    t0 = time.perf_counter()
    sys1 = KevSys1.load(a.checkpoint, a.device)
    load_seconds = time.perf_counter() - t0
    load_peak = dog.stage_peak
    assert not sys1.training and sys1.temperature > 1.0 and sys1.model.head.temperature == 1.0
    assert not sys1.trainable_parameters()

    prompts, options, _ = batch(0, a.infer_batch)
    z = sys1.score(prompts, options)
    p = sys1.probs(prompts, options)
    assert [len(x) for x in p] == [len(o) for o in options]
    for pi, zi in zip(p, z):
        assert abs(float(pi.sum()) - 1) < 1e-4
        # the temperature is applied once: probs equal softmax(logits / T)
        assert torch.allclose(pi, torch.softmax(zi / sys1.temperature, -1), atol=1e-4)
    single = [sys1.score([q], [o])[0] for q, o in zip(prompts, options)]
    batched_vs_single = max(float((a_ - b_).abs().max().cpu()) for a_, b_ in zip(z, single))

    dog.reset_stage()
    seconds, rows = [], []
    for run in range(a.infer_runs):
        q, o, _ = batch(1000 + run * a.infer_batch, a.infer_batch)
        t = time.perf_counter()
        out = sys1.probs(q, o)
        device_sync(a.device)
        _ = [x.cpu() for x in out]
        seconds.append(time.perf_counter() - t)
        rows.append(len(q))
        print(f"infer run {run}: {seconds[-1]:.2f} s for {len(q)} rows", flush=True)
    warm_s, warm_r = seconds[1:] or seconds, rows[1:] or rows
    state_tokens = sum(len(sys1._encode(x, y)["ids"]) for x, y in zip(q, o)) / len(q)
    return {
        "load_seconds": round(load_seconds, 1), "load_peak_footprint_gb": round(load_peak / GB, 2),
        "temperature_read_from_checkpoint": sys1.temperature, "head_temperature_attribute": sys1.model.head.temperature,
        "max_abs_logit_difference_batched_vs_single": batched_vs_single,
        "infer_batch": a.infer_batch, "infer_runs": a.infer_runs, "run_seconds": [round(s, 2) for s in seconds], "mean_tokens_per_row": round(state_tokens, 1),
        "infer_rows_per_second_after_first_run": round(sum(warm_r) / sum(warm_s), 3),
        "infer_stage_peak_footprint_gb": round(dog.stage_peak / GB, 2), "precision": sys1.precision,
    }


def stage_roundtrip(a, dog) -> dict:
    """Train path saves, eval path loads: the logits of a saved adapter must match the live model's."""
    out = Path(a.roundtrip_dir)
    sys1 = KevSys1.load(a.checkpoint, a.device, trainable=True)
    sys1.temperature = 1.7
    sys1.save(out)
    prompts, options, _ = batch(0, 3)
    sys1.eval()
    before = [z.detach().cpu() for z in sys1.score(prompts, options)]
    assert sys1.temperature == 1.7
    del sys1
    again = KevSys1.load(out, a.device)
    after = [z.detach().cpu() for z in again.score(prompts, options)]
    return {"max_abs_logit_difference_unmerged_vs_saved_merged": max(float((x - y).abs().max()) for x, y in zip(before, after)),
            "temperature_after_roundtrip": again.temperature, "question": QUESTION_TEXT, "precision": again.precision}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    ap.add_argument("--stage", required=True, choices=["train", "infer", "roundtrip"])
    ap.add_argument("--checkpoint", default=KEV_08B)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--max-gb", type=float, default=8.0)
    ap.add_argument("--min-free-pct", type=int, default=20)
    ap.add_argument("--train-batch", type=int, default=4)
    ap.add_argument("--train-steps", type=int, default=4)
    ap.add_argument("--infer-batch", type=int, default=8)
    ap.add_argument("--infer-runs", type=int, default=5)
    ap.add_argument("--roundtrip-dir", default="")
    ap.add_argument("--trace-steps", type=int, default=5, help="trace entries in the invented state, 5 is the plan's start value")
    a = ap.parse_args()
    TRACE_STEPS[0] = a.trace_steps
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    dog = Watchdog(a.max_gb, a.min_free_pct)
    dog.start()
    t0 = time.time()
    result = {"train": stage_train, "infer": stage_infer, "roundtrip": stage_roundtrip}[a.stage](a, dog)
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=Path(__file__).parent).stdout.strip()
    result.update({"stage": a.stage, "device": a.device, "seed": a.seed, "trace_steps": a.trace_steps, "threads": a.threads, "wall_seconds": round(time.time() - t0, 1), "checkpoint": a.checkpoint,
                   "git_commit": git, "git_dirty_note": "written before the parent session commits lab/src/sys1", "environment": environment(),
                   "hardware": {"machine": platform.machine(), "platform": platform.platform(),
                                "memory_gb": int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout) // GB,
                                "cpu": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()},
                   "scope": "Mac measurement on invented ALFWorld-shaped text. Not an L4 number.",
                   "peak_footprint_gb": round(dog.peak / GB, 2), "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / GB, 2),
                   "lowest_system_free_pct_seen": dog.lowest_free_pct})
    Path(a.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in result if k not in ("environment", "hardware")}, indent=1), flush=True)


if __name__ == "__main__":
    main()
