"""StubSys1: a very small CPU model behind the Sys1 interface, for tests of everything that calls Sys1.

It hashes words into a bag-of-words embedding and scores each option against the prompt with a bilinear form. No file
is read and no weights are downloaded. Equal inputs give equal outputs under a given seed.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn as nn

from .base import Sys1, check_options

BUCKETS = 512


def _bucket(word: str) -> int:
    return int.from_bytes(hashlib.sha256(word.encode()).digest()[:4], "big") % BUCKETS


class StubSys1(Sys1):
    def __init__(self, dim: int = 16, seed: int = 0, device: str = "cpu"):
        super().__init__()
        self.dim, self.seed, self.device = dim, seed, device
        generator = torch.Generator().manual_seed(seed)
        self.embed = nn.Parameter(torch.randn(BUCKETS, dim, generator=generator) / dim**0.5)
        self.bilinear = nn.Parameter(torch.eye(dim))
        self.to(device)

    def _encode(self, text: str) -> torch.Tensor:
        words = text.lower().split() or [""]
        ids = torch.tensor([_bucket(w) for w in words], device=self.embed.device)
        return self.embed[ids].mean(0)

    def score(self, prompts, options):
        if len(prompts) != len(options):
            raise ValueError(f"{len(prompts)} prompts and {len(options)} option lists")
        out = []
        with torch.set_grad_enabled(self.training):
            for prompt, opts in zip(prompts, options):
                check_options(opts)
                q = self.bilinear @ self._encode(prompt)
                out.append(torch.stack([self._encode(o) @ q for o in opts]))
        return out

    def save(self, path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({"state": self.state_dict(), "dim": self.dim, "seed": self.seed, "temperature": self._fitted_temperature}, path / "stub.pt")

    @classmethod
    def load(cls, path, device="cpu") -> "StubSys1":
        saved = torch.load(Path(path) / "stub.pt", map_location=device)
        model = cls(dim=saved["dim"], seed=saved["seed"], device=device)
        model.load_state_dict(saved["state"])
        model.temperature = saved["temperature"]
        model.eval()
        return model
