"""Building blocks: SIGReg (from LeWM, Maes et al.), token MLPs and a transformer stack.

``SIGReg`` is the Sketched Isotropic Gaussian Regularizer of LeJEPA / LeWM, copied from the official
``le-wm`` repository (MIT) without change. The transformer uses PyTorch's own encoder layer.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularizer (single-GPU!)"""

    def __init__(self, knots: int = 17, num_proj: int = 1024) -> None:
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: Tensor) -> Tensor:
        """proj: (T, B, D)"""
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


def mlp(i: int, o: int, h: int = 128) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


def transformer(dim: int, layers: int, heads: int) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout=0.0, batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)


def causal_frame_mask(frames: int, tokens: int, device=None) -> Tensor:
    """bool (L, L), True = blocked: a token attends to tokens of its own and earlier frames only."""
    f = torch.arange(frames * tokens, device=device) // tokens
    return f[None, :] > f[:, None]
