"""`kev.train`, unchanged, with a partial checkpoint written every few minutes.

    PARTIAL_TAR=/content/out/partial.tar PARTIAL_EVERY_MIN=10 python -m train.kev_train_runner <kev.train arguments>

A LoRA run of `kev.train` writes its checkpoint only at the end: resume points and snapshots are for `--full_ft 1`
(kev/train.py:472-477) and the save is at kev/train.py:663. A VM that dies at hour six would then leave nothing for the Mac
to pull. This runner registers one global optimizer post-step hook and captures the model and the Meta that `kev.train` builds.
Between optimizer steps the hook writes the adapter and head into a tar, replaced atomically, so a pull never sees half a
file. The training path itself is not touched. A partial checkpoint has no optimizer state, so it can start a new
`--init_from` run and cannot continue this one's schedule. Its temperature is 1.0, and it is never a result.
"""
from __future__ import annotations

import dataclasses
import json
import os
import random
import resource
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim.optimizer import register_optimizer_step_post_hook


def save_partial(model, meta, tar_path, step: int, seconds: float) -> None:
    from kev.checkpoint import write_meta
    tar_path = Path(tar_path)
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tar_path.with_name(tar_path.name + ".tmp")
    with tempfile.TemporaryDirectory(dir=tar_path.parent) as d:
        model.lm.save_pretrained(d)
        head = {k: v.detach().cpu() for k, v in model.head.state_dict().items()}
        write_meta(d, dataclasses.replace(meta, head=head, temperature=1.0, extra={"partial": {"optimizer_steps": step, "seconds": round(seconds, 1)}}))
        with tarfile.open(tmp, "w") as tar:
            tar.add(d, arcname="checkpoint")
    os.replace(tmp, tar_path)


def install_partial_checkpoints(kev_train, tar_path, every_seconds: float):
    """-> the hook's handle, which a test removes. The hook is global, and a training process has one run."""
    captured = {}
    real_model, real_meta = kev_train.DecisionModel, kev_train.Meta

    class CapturingModel(real_model):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            captured["model"] = self

    def capturing_meta(*args, **kwargs):
        captured["meta"] = real_meta(*args, **kwargs)
        return captured["meta"]

    kev_train.DecisionModel, kev_train.Meta = CapturingModel, capturing_meta
    state = {"last": time.time(), "start": time.time(), "steps": 0}

    def after_step(optimizer, args, kwargs):
        state["steps"] += 1
        now = time.time()
        if now - state["last"] < every_seconds or "model" not in captured or "meta" not in captured:
            return
        save_partial(captured["model"], captured["meta"], tar_path, state["steps"], now - state["start"])
        print(f"partial checkpoint at optimizer step {state['steps']}, {time.time() - now:.1f} s to write", flush=True)
        state["last"] = time.time()

    return register_optimizer_step_post_hook(after_step)


def peak_memory() -> dict:
    out = {"ru_maxrss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)}
    if torch.cuda.is_available():
        out |= {"cuda_max_allocated_bytes": torch.cuda.max_memory_allocated(), "cuda_max_reserved_bytes": torch.cuda.max_memory_reserved(), "cuda_total_bytes": torch.cuda.get_device_properties(0).total_memory}
    return out


def main() -> None:
    from kev import train as kev_train
    seed = int(sys.argv[sys.argv.index("--seed") + 1])
    random.seed(seed)
    np.random.seed(seed)   # kev.train seeds torch and its own random.Random (kev/train.py:531)
    tar_path, every_minutes = os.environ.get("PARTIAL_TAR", ""), float(os.environ.get("PARTIAL_EVERY_MIN", "10"))
    if tar_path and every_minutes > 0:
        install_partial_checkpoints(kev_train, tar_path, every_minutes * 60)
    kev_train.main()
    out = Path(sys.argv[sys.argv.index("--out") + 1])
    (out / "runner_metrics.json").write_text(json.dumps({"peak_memory": peak_memory(), "numpy_seed": seed, "deterministic_algorithms": False}, indent=2) + "\n")


if __name__ == "__main__":
    main()
