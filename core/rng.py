"""Seeded sampling. The generator lives on CPU; samples are moved to the env device."""

from __future__ import annotations

import torch
from torch import Tensor


class Rng:
    def __init__(self, seed: int, device: str | torch.device) -> None:
        self.gen = torch.Generator().manual_seed(seed)
        self.device = device

    def seed(self, seed: int) -> None:
        self.gen.manual_seed(seed)

    def rand(self, *shape: int) -> Tensor:
        return torch.rand(shape, generator=self.gen).to(self.device)

    def uniform(self, lo: float, hi: float, *shape: int) -> Tensor:
        return lo + (hi - lo) * self.rand(*shape)

    def randn(self, *shape: int) -> Tensor:
        return torch.randn(shape, generator=self.gen).to(self.device)

    def poisson(self, rate: float | Tensor, *shape: int) -> Tensor:
        """Poisson counts of ``shape`` with a scalar ``rate``, or one per element of a ``rate`` tensor."""
        rates = rate.float().cpu().expand(shape).contiguous() if isinstance(rate, Tensor) else torch.full(shape, rate)
        return torch.poisson(rates, generator=self.gen).long().to(self.device)
