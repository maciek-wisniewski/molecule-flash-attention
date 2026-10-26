"""Converting between padded batches and the packed layout of :func:`molflash.attention_varlen`."""

import torch

from .primitives import EdgeBias, EdgeMask, NodeBias, _PairTerm


def cu_seqlens(lengths: torch.Tensor) -> torch.Tensor:
    """``[S]`` sequence lengths -> ``[S + 1]`` int32 offsets."""
    out = torch.zeros(lengths.numel() + 1, dtype=torch.int32, device=lengths.device)
    out[1:] = torch.cumsum(lengths, 0)
    return out


def pack(x: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
    """``[B, N, ...]`` with a ``[B, N]`` validity mask -> ``[T, ...]`` of the valid tokens.

    Valid tokens must form a prefix of each row, as padding does.
    """
    return x[key_mask.bool()]


def unpack(x: torch.Tensor, cu: torch.Tensor, n: int | None = None) -> torch.Tensor:
    """``[T, ...]`` -> ``[B, N, ...]`` padded with zeros."""
    lengths = (cu[1:] - cu[:-1]).tolist()
    n = max(lengths) if n is None else n
    out = x.new_zeros((len(lengths), n, *x.shape[1:]))
    starts = cu[:-1].tolist()
    for i, (s, ln) in enumerate(zip(starts, lengths, strict=True)):
        out[i, :ln] = x[s : s + ln]
    return out


def slice_terms(terms: list, qs: int, qe: int, ks: int, ke: int) -> list:
    """The terms of one packed sequence as a padded batch of one, for ``molflash.reference``."""
    out = []
    for t in terms:
        if isinstance(t, EdgeBias):
            out.append(EdgeBias(t.index[qs:qe].unsqueeze(0), t.values[qs:qe].unsqueeze(0)))
        elif isinstance(t, EdgeMask):
            out.append(EdgeMask(t.index[qs:qe].unsqueeze(0)))
        elif isinstance(t, NodeBias):
            out.append(
                NodeBias(
                    t.u[qs:qe].transpose(0, 1).unsqueeze(0), t.w[ks:ke].transpose(0, 1).unsqueeze(0)
                )
            )
        elif isinstance(t, _PairTerm):
            clone = object.__new__(type(t))
            clone.__dict__.update(t.__dict__)
            clone.q_features = [f[qs:qe].unsqueeze(0) for f in t.q_features]
            clone.k_features = [f[ks:ke].unsqueeze(0) for f in t.k_features]
            if t.per_sample:
                raise ValueError("per_sample tables are not supported under varlen")
            out.append(clone)
        else:
            raise TypeError(f"unknown term {t!r}")
    return out
