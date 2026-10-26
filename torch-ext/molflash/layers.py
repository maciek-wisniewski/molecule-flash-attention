"""Layers for the Hugging Face ``kernels`` hub: stateless modules whose ``forward`` can replace
an attention forward through ``kernelize``."""

import torch
from torch import nn

from .ops import attention, attention_varlen


class StructuredAttention(nn.Module):
    """``forward(q, k, v, **terms)``, the same as :func:`molflash.attention`."""

    can_torch_compile: bool = True
    has_backward: bool = True

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **terms) -> torch.Tensor:
        return attention(q, k, v, **terms)


class StructuredAttentionVarlen(nn.Module):
    """``forward(q, k, v, cu_seqlens_q, cu_seqlens_k, **terms)``, as
    :func:`molflash.attention_varlen`."""

    can_torch_compile: bool = True
    has_backward: bool = True

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        **terms,
    ) -> torch.Tensor:
        return attention_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, **terms)


__all__ = ["StructuredAttention", "StructuredAttentionVarlen"]
