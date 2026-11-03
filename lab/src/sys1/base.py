"""The Sys1 interface of `lab/INTERFACES.md`, and the option and question texts both training paths share.

Sys1 picks one option from a short list and gives a probability for each option. `score` returns raw logits at
temperature 1 in every mode. `probs` divides by `temperature`, which reads 1.0 while the model is in training mode and
the fitted value otherwise, so a fitted value can never reach a training step.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# The last option of every list. Wording it as Kev's own none-of-the-above family (kev/data.py NONE_OPTIONS) means Kev-0.8B
# starts from something it has read, and the second half names what the option does in the harness.
FALLBACK_TEXT = "None of these commands: the LLM takes this step"

# The one question Sys1 is asked at every step. `export_kev` writes it into training records and `kev_adapter` asks it
# at inference, and a test asserts the two encode to the same token ids.
QUESTION_TEXT = "Which command should run next?"


def check_options(options: list[str]) -> None:
    """Refuse an option list the pointer head would score wrongly: empty, repeated, or without the fallback last."""
    if len(options) < 2:
        raise ValueError(f"an option list needs at least one command and the fallback, got {options!r}")
    if options[-1] != FALLBACK_TEXT:
        raise ValueError(f"options must end with FALLBACK_TEXT, got {options[-1]!r}")
    if len(set(options)) != len(options):
        raise ValueError(f"repeated option in {options!r}")


class Sys1(nn.Module):
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self._fitted_temperature = float(temperature)

    @property
    def temperature(self) -> float:
        return 1.0 if self.training else self._fitted_temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        if not value > 0:
            raise ValueError(f"temperature must be positive, got {value}")
        self._fitted_temperature = float(value)

    def score(self, prompts: list[str], options: list[list[str]]) -> list[torch.Tensor]:
        """One logit tensor per prompt, one logit per option, at temperature 1, with gradients when training."""
        raise NotImplementedError

    def probs(self, prompts: list[str], options: list[list[str]]) -> list[torch.Tensor]:
        return [torch.softmax(z / self.temperature, -1) for z in self.score(prompts, options)]

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def save(self, path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path, device) -> "Sys1":
        raise NotImplementedError
