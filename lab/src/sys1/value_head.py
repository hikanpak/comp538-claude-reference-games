"""The value head PPO needs, on the `<decide>` hidden state, and one forward pass that gives Sys1's logits and that hidden state together.

Kev has no value head (review finding 2 in `lib/issues/2026-11-03-claude-execution-critique.md`). `DecisionModel.forward_batch` passes each
question's `<decide>` hidden state to `PointerHead.forward` as its first argument (kev/model.py:216, called at kev/model.py:327 and kev/model.py:382), so a
forward pre-hook on the head reads it with Kev's forward path untouched. `KevSys1.score` sorts rows by length, and this file keeps the same order
and restores the caller's, so a value lines up with its prompt. `dropout_off` exists because `kev.train` trains with LoRA dropout 0.05
(kev/model.py:274), and a PPO ratio read in training mode would differ from 1 before any update if dropout stayed on.
"""
from __future__ import annotations

import contextlib

import torch
import torch.nn as nn

from .base import check_options
from .stub import StubSys1


class ValueHead(nn.Module):
    """`<decide>` hidden state to one number. The layers are Claude's choice: decider's `docs/RL.md` lists no value head, so nothing is copied from it."""

    def __init__(self, dim: int, hidden: int = 256, seed: int = 0):
        super().__init__()
        self.dim, self.hidden = dim, hidden
        with torch.random.fork_rng(devices=[]):   # the head's start depends on `seed` alone, and the caller's random stream does not move
            torch.manual_seed(seed)
            self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        with torch.no_grad():   # values start near 0, so the first advantages are close to the returns
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.zero_()

    def forward(self, decide: torch.Tensor) -> torch.Tensor:
        """[N, d] -> [N]."""
        return self.net(decide).squeeze(-1)


def decide_dim(sys1) -> int:
    """The size of the `<decide>` hidden state of this Sys1."""
    if isinstance(sys1, StubSys1):
        return sys1.dim
    return sys1.model.lm.config.hidden_size


def _stub_forward(sys1: StubSys1, prompts, options):
    logits, decide = [], []
    with torch.set_grad_enabled(sys1.training):
        for prompt, opts in zip(prompts, options):
            check_options(opts)
            q = sys1.bilinear @ sys1._encode(prompt)   # what StubSys1.score reads each option against, its stand-in for `<decide>`
            logits.append(torch.stack([sys1._encode(o) @ q for o in opts]))
            decide.append(q)
    return logits, torch.stack(decide)


def _kev_forward(sys1, prompts, options):
    encs = [sys1._encode(p, o) for p, o in zip(prompts, options)]
    order = sorted(range(len(encs)), key=lambda i: len(encs[i]["ids"]))   # as KevSys1.score
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if sys1.autocast_bf16 else contextlib.nullcontext()
    seen = []
    handle = sys1.model.head.register_forward_pre_hook(lambda _module, args: seen.append(args[0]))
    try:
        with torch.set_grad_enabled(sys1.training), autocast:
            per_record = sys1.model.forward_batch([encs[i] for i in order])
    finally:
        handle.remove()
    if len(seen) != len(encs):
        raise RuntimeError(f"the pointer head ran {len(seen)} times for {len(encs)} rows, and Sys1 asks one question per row")
    logits: list = [None] * len(encs)
    decide: list = [None] * len(encs)
    for i, questions, hidden in zip(order, per_record, seen):
        logits[i] = questions[0].float()
        decide[i] = hidden.float()
    return logits, torch.stack(decide)


def score_and_decide(sys1, prompts: list[str], options: list[list[str]]) -> tuple[list[torch.Tensor], torch.Tensor]:
    """-> (one logit tensor per prompt at temperature 1, as `sys1.score` gives, and the `<decide>` hidden states as [N, d]), from one forward pass.
    Gradients flow when `sys1` is in training mode, as in `score`."""
    if len(prompts) != len(options):
        raise ValueError(f"{len(prompts)} prompts and {len(options)} option lists")
    if not prompts:
        raise ValueError("no prompts")
    if isinstance(sys1, StubSys1):
        return _stub_forward(sys1, prompts, options)
    return _kev_forward(sys1, prompts, options)


def dropout_off(module: nn.Module) -> None:
    """Call after `module.train()`, which turns every child's dropout back on."""
    for child in module.modules():
        if isinstance(child, nn.Dropout):
            child.eval()
