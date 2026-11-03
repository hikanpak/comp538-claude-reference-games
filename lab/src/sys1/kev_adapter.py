"""KevSys1: `kev.model.DecisionModel` behind the Sys1 interface, batched over prompts and per-prompt option lists.

Two load paths, because they hold different objects.

`trainable=True` builds the LoRA on the frozen base and warm-starts the adapter and pointer head from the checkpoint,
which is what `kev.train --init_from` does (kev/train.py:543-556). Its fitted temperature starts at 1.0, as a run of
`kev.train` does (kev/train.py:550 builds a Meta whose temperature is the default), so the released T = 2.35 is not carried
onto weights it was not fitted on.

`trainable=False` is Kev's exact evaluation path: the adapter merged into the base (kev/checkpoint.py:312), no parameter
requires a gradient, and the temperature the checkpoint carries is read into `temperature`.

`PointerHead.forward` divides logits by its own `temperature` in eval mode only (kev/model.py:218). This class keeps that
attribute at 1.0 for good and applies the temperature once, in `Sys1.probs`, so `score` is raw logits in every mode and a
fitted value is never applied twice.
"""
from __future__ import annotations

import contextlib
import dataclasses
from pathlib import Path

import torch
from kev.checkpoint import Checkpoint, LoadOptions, Meta, write_meta
from kev.model import SERVE_MAX_BRANCH, SERVE_MAX_STATE, DecisionModel, load_tokenizer

from .base import QUESTION_TEXT, Sys1, check_options

KEV_08B = "jaredpalmer/kev-0.8b"


def to_record(prompt: str, options: list[str]) -> dict:
    """The internal record Kev encodes for one Sys1 question. A test asserts this equals what `kev.data.materialize` makes
    of the row `export_kev` writes, so training and inference see the same text."""
    check_options(options)
    return {"state": prompt, "questions": [{"instr": QUESTION_TEXT, "options": list(options), "label": 0}]}


class KevSys1(Sys1):
    def __init__(self, model: DecisionModel, tok, meta: Meta, *, temperature: float = 1.0, trainable: bool = False, autocast_bf16: bool = False, provenance: dict | None = None):
        super().__init__(temperature)
        self.model, self.tok, self.meta, self.provenance = model, tok, meta, provenance or {}
        self.trainable, self.autocast_bf16 = trainable, autocast_bf16
        if autocast_bf16 and not str(model.device).startswith("cuda"):
            raise ValueError("bf16 autocast is CUDA only, as in kev.train (kev/train.py:443)")
        self.model.head.temperature = 1.0
        self.train(False)

    @property
    def device(self):
        return self.model.device

    @property
    def precision(self) -> str:
        return f"{self.model.dtype} weights" + (", bf16 autocast" if self.autocast_bf16 else "")

    def train(self, mode: bool = True):
        if mode and not getattr(self, "trainable", False):
            raise RuntimeError("this KevSys1 was loaded with the adapter merged: load it with trainable=True to train")
        return super().train(mode)

    def _encode(self, prompt: str, options: list[str]) -> dict:
        # Strict: Kev's non-strict encoder keeps the FIRST max_state tokens (kev/model.py:104), which would cut the last
        # trace entries, the part of the state Sys1 needs most. An overflow raises kev.model.ContextOverflow.
        return self.model.encode(self.tok, to_record(prompt, options), max_state=SERVE_MAX_STATE, max_branch=SERVE_MAX_BRANCH, strict=True)

    def score(self, prompts, options):
        if len(prompts) != len(options):
            raise ValueError(f"{len(prompts)} prompts and {len(options)} option lists")
        if not prompts:
            return []
        encs = [self._encode(p, o) for p, o in zip(prompts, options)]
        order = sorted(range(len(encs)), key=lambda i: len(encs[i]["ids"]))   # neighbours in length pad little
        autocast = torch.autocast("cuda", dtype=torch.bfloat16) if self.autocast_bf16 else contextlib.nullcontext()
        with torch.set_grad_enabled(self.training), autocast:
            logits = self.model.forward_batch([encs[i] for i in order])
        out: list = [None] * len(encs)
        for i, per_question in zip(order, logits):
            out[i] = per_question[0].float()
        return out

    def trainable_parameters(self):
        return self.model.trainable_parameters() if self.trainable else []

    def save(self, path) -> None:
        """A checkpoint `kev.checkpoint.Checkpoint` reads: adapter, tokenizer, and head.pt carrying the fitted temperature."""
        if not self.trainable:
            raise RuntimeError("a merged model has no adapter to save: load it with trainable=True")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.lm.save_pretrained(path)
        head = {k: v.detach().cpu() for k, v in self.model.head.state_dict().items()}
        write_meta(path, dataclasses.replace(self.meta, head=head, temperature=self._fitted_temperature))
        self.tok.save_pretrained(path)

    @classmethod
    def load(cls, path, device, trainable: bool = False, dtype: torch.dtype | None = None, autocast_bf16: bool = False) -> "KevSys1":
        checkpoint = Checkpoint(path)
        meta = checkpoint.meta
        if trainable:
            tok = load_tokenizer(meta.base, revision=meta.base_revision)
            model = DecisionModel(meta.base, tok, device, lora=meta.lora, revision=meta.base_revision, head_dim=meta.head_dim, option_isolation=meta.option_isolation,
                                  special_embeddings=meta.special_embeddings, dtype=dtype or torch.float32)
            ours = dataclasses.replace(meta, head=None, temperature=1.0, extra={})
            provenance = checkpoint.warm_start(model, ours)   # compares the architecture fields, then loads adapter and head
            model.lm.config.use_cache = False   # as kev.train does (kev/train.py:548)
            return cls(model, tok, ours, temperature=1.0, trainable=True, autocast_bf16=autocast_bf16, provenance=provenance)
        tok, model = checkpoint.load(device, LoadOptions(dtype=dtype, temperature=1.0, backend="torch"))
        for p in model.parameters():   # Kev's merged model leaves every parameter requiring a gradient
            p.requires_grad_(False)
        return cls(model, tok, meta, temperature=meta.temperature, autocast_bf16=autocast_bf16, provenance={"init_from": str(path), "resolved": checkpoint.path})
