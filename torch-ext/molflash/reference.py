"""Dense PyTorch implementation of :func:`molflash.attention`, for tests and small CPU runs.

It builds every ``[B, H, Nq, Nk]`` tensor the kernel avoids, so its memory grows with ``N^2``. It
accepts the same term objects and follows the same rules, and the test suite compares the kernel
against it on every run. Do not train with it.
"""

import torch

from .primitives import EdgeBias, EdgeMask, Kind, NodeBias, Role, _PairTerm

_FILL = -1.0e30


def _channels(features: list[torch.Tensor]) -> list[torch.Tensor]:
    """Split ``[B, N]`` / ``[B, N, C]`` tensors into ``[B, N]`` channels, in order."""
    out = []
    for f in features:
        if f.dim() == 2:
            out.append(f)
        else:
            out.extend(f.unbind(-1))
    return out


def pair_outputs(term: _PairTerm, nq: int, nk: int):
    """What ``fn`` returns for every pair: ``(code [B,Nq,Nk] | None, t [B,Nq,Nk] | None)``, or the
    keep mask for a ``PairMask``."""
    q_ch = [c.unsqueeze(-1) for c in _channels(term.q_features)]
    k_ch = [c.unsqueeze(-2) for c in _channels(term.k_features)]
    if term.torch_fn is not None:
        out = term.torch_fn(*q_ch, *k_ch)
    else:
        from .kernels import materialize_pair_fn

        out = materialize_pair_fn(term, nq, nk)
    if term.kind is Kind.CODE_VALUE:
        code, t = out
        return code, t
    if term.kind is Kind.VALUE:
        return None, out
    return out, None


def _interp(t: torch.Tensor, lo: float, hi: float, g: int):
    """Left grid index and fraction of ``t`` on ``g`` points spread over ``[lo, hi]``, clamped.

    The index is detached and the fraction is not, so autograd gives ``dvalue/dt`` the slope of
    the segment, and zero outside ``[lo, hi]``, which is what the kernel computes.
    """
    t = torch.where(torch.isnan(t), torch.full_like(t, lo), t)
    u = ((t - lo) * ((g - 1) / (hi - lo))).clamp(0.0, g - 1)
    i0 = u.detach().floor().clamp(max=g - 2)
    frac = u - i0
    i0 = i0.long()
    return i0, frac


def _gather(table: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """``table [B, H, M]`` at ``idx [B, H, Nq, Nk]`` -> ``[B, H, Nq, Nk]``."""
    b, h, nq, nk = idx.shape
    flat = idx.reshape(b, h, nq * nk)
    return torch.gather(table, 2, flat).reshape(b, h, nq, nk)


def pair_dense(term: _PairTerm, b: int, h: int, nq: int, nk: int) -> torch.Tensor:
    """The term's value for every pair, ``[B, H, Nq, Nk]`` float32 (or bool for a mask)."""
    code, t = pair_outputs(term, nq, nk)
    if term.kind is Kind.KEEP:
        return code.bool().expand(b, nq, nk).unsqueeze(1)

    table = term.table.float()
    if not term.per_sample:
        table = table.unsqueeze(0)
    table = table.expand(b, h, *table.shape[2:])
    neutral = term.neutral

    if term.kind is Kind.CODE:
        k = table.shape[-1]
        code = code.long().expand(b, nq, nk)
        valid = (code >= 0) & (code < k)
        idx = code.clamp(0, k - 1).unsqueeze(1).expand(b, h, nq, nk)
        val = _gather(table, idx)
        return torch.where(valid.unsqueeze(1), val, torch.full_like(val, neutral))

    t = t.float().expand(b, nq, nk).unsqueeze(1).expand(b, h, nq, nk)
    valid = torch.ones((b, 1, nq, nk), dtype=torch.bool, device=t.device)
    if term.kind is Kind.CODE_VALUE:
        k = term.num_codes
        code = code.long().expand(b, nq, nk)
        valid = ((code >= 0) & (code < k)).unsqueeze(1)
        code = code.clamp(0, k - 1).unsqueeze(1).expand(b, h, nq, nk)
        if term.affine is not None:
            a, s = (x.float() for x in term.affine)
            a = a.unsqueeze(0).expand(h, -1) if a.dim() == 1 else a
            s = s.unsqueeze(0).expand(h, -1) if s.dim() == 1 else s
            a = _gather(a.unsqueeze(0).expand(b, h, k), code)
            s = _gather(s.unsqueeze(0).expand(b, h, k), code)
            t = a * t + s

    lo, hi = term.grid
    g = table.shape[-1]
    i0, frac = _interp(t, lo, hi, g)
    if term.code_indexed_table:
        table = table.reshape(b, h, -1)
        base = code * g
    else:
        base = 0
    v0 = _gather(table, base + i0)
    v1 = _gather(table, base + i0 + 1)
    val = v0 + (v1 - v0) * frac
    return torch.where(valid, val, torch.full_like(val, neutral))


def edge_dense(term: EdgeBias | EdgeMask, b: int, h: int, nq: int, nk: int) -> torch.Tensor:
    idx = term.index.long()
    valid = (idx >= 0) & (idx < nk)
    safe = idx.clamp(0, nk - 1)
    if isinstance(term, EdgeMask):
        hit = (
            torch.zeros((b, nq, nk), dtype=torch.int32, device=idx.device).scatter_add(
                2, safe, valid.int()
            )
            > 0
        )
        return hit.unsqueeze(1)
    vals = term.values.float() * valid.unsqueeze(-1)
    k = idx.shape[-1]
    dense = torch.zeros((b, nq, nk, h), dtype=torch.float32, device=vals.device)
    dense = dense.scatter_add(2, safe.unsqueeze(-1).expand(b, nq, k, h), vals)
    return dense.permute(0, 3, 1, 2)


def node_dense(term: NodeBias) -> torch.Tensor:
    return term.u.float() @ term.w.float().transpose(-1, -2)


def dense_term(term, b: int, h: int, nq: int, nk: int) -> torch.Tensor:
    if isinstance(term, (EdgeBias, EdgeMask)):
        return edge_dense(term, b, h, nq, nk)
    if isinstance(term, NodeBias):
        return node_dense(term)
    return pair_dense(term, b, h, nq, nk)


def _listify(x) -> list:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    bias=None,
    mask=None,
    multiplier=None,
    gate=None,
    key_mask: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    return_max_score: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Same arguments and result as :func:`molflash.attention`, computed densely in float32."""
    b, h, nq, d = q.shape
    nk = k.shape[2]
    scale = d**-0.5 if softmax_scale is None else float(softmax_scale)

    s = (q.float() @ k.float().transpose(-1, -2)) * scale
    for term in _listify(multiplier):
        s = s * dense_term(term, b, h, nq, nk)
    for term in _listify(bias):
        s = s + dense_term(term, b, h, nq, nk)

    keep = torch.ones((b, 1, nq, nk), dtype=torch.bool, device=q.device)
    if key_mask is not None:
        keep = keep & key_mask.bool()[:, None, None, :]
    for term in _listify(mask):
        keep = keep & dense_term(term, b, h, nq, nk)

    s = torch.where(keep, s, torch.full_like(s, _FILL))
    # Before the softmax, and 0.0 for a row with nothing left, which is what the
    # kernel's `m_safe` substitutes for -inf.
    alive = keep.any(dim=-1, keepdim=True)
    max_score = torch.where(alive, s.max(dim=-1, keepdim=True).values, torch.zeros_like(s[..., :1])).squeeze(-1)
    max_score = max_score.expand(b, h, nq).contiguous()
    p = torch.softmax(s, dim=-1)
    p = torch.where(keep, p, torch.zeros_like(p))
    for term in _listify(gate):
        p = p * dense_term(term, b, h, nq, nk)
    out = (p @ v.float()).to(q.dtype)
    return (out, max_score) if return_max_score else out


def attention_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    bias=None,
    mask=None,
    multiplier=None,
    gate=None,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Same as :func:`molflash.attention_varlen`: each sequence is run through :func:`attention`."""
    from .varlen import slice_terms

    out = []
    cq = cu_seqlens_q.tolist()
    ck = cu_seqlens_k.tolist()
    for i in range(len(cq) - 1):
        qs, qe, ks, ke = cq[i], cq[i + 1], ck[i], ck[i + 1]
        terms = {
            name: slice_terms(_listify(group), qs, qe, ks, ke)
            for name, group in dict(bias=bias, mask=mask, multiplier=multiplier, gate=gate).items()
        }
        o = attention(
            q[qs:qe].transpose(0, 1).unsqueeze(0),
            k[ks:ke].transpose(0, 1).unsqueeze(0),
            v[ks:ke].transpose(0, 1).unsqueeze(0),
            softmax_scale=softmax_scale,
            **terms,
        )
        out.append(o.squeeze(0).transpose(0, 1))
    return torch.cat(out, dim=0)


__all__ = ["attention", "attention_varlen", "dense_term", "Role"]
