import torch
from torch import nn

from .ops import attention


class MultiheadAttention(nn.Module):
    """Self- or cross-attention with input and output projections, taking molflash terms.

    ``x`` is ``[B, N, E]``; terms are passed to ``forward`` as to :func:`molflash.attention`.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        kdim: int | None = None,
        vdim: int | None = None,
        head_dim: int | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim or embed_dim // num_heads
        inner = self.num_heads * self.head_dim
        self.q_proj = nn.Linear(embed_dim, inner, bias=bias)
        self.k_proj = nn.Linear(kdim or embed_dim, inner, bias=bias)
        self.v_proj = nn.Linear(vdim or embed_dim, inner, bias=bias)
        self.out_proj = nn.Linear(inner, embed_dim, bias=bias)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        return x.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None, **terms
    ) -> torch.Tensor:
        context = x if context is None else context
        q = self._heads(self.q_proj(x))
        k = self._heads(self.k_proj(context))
        v = self._heads(self.v_proj(context))
        o = attention(q, k, v, **terms)
        b, h, n, d = o.shape
        return self.out_proj(o.transpose(1, 2).reshape(b, n, h * d))
