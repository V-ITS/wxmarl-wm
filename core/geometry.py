"""Angle wrapping, bearings, ego-frame transforms and pairwise distances."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def wrap(angle: Tensor) -> Tensor:
    """Wrap angles to (-pi, pi]."""
    return angle - 2 * math.pi * torch.ceil((angle - math.pi) / (2 * math.pi))


def bearing(src: Tensor, dst: Tensor) -> Tensor:
    """Bearing from ``src`` to ``dst`` ([..., 2]); 0 = east, CCW positive."""
    d = dst - src
    return torch.atan2(d[..., 1], d[..., 0])


def to_ego(vec: Tensor, psi: Tensor) -> Tensor:
    """Rotate world-frame vectors [..., 2] by ``-psi`` so the heading points along +x."""
    c, s = torch.cos(psi), torch.sin(psi)
    return torch.stack((vec[..., 0] * c + vec[..., 1] * s, -vec[..., 0] * s + vec[..., 1] * c), dim=-1)


def pairwise_dist(a: Tensor, b: Tensor) -> Tensor:
    """Euclidean distances between [B, N, 2] and [B, M, 2] -> [B, N, M]."""
    return (a.unsqueeze(2) - b.unsqueeze(1)).norm(dim=-1)
