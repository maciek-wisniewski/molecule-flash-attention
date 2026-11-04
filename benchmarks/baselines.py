"""The same call computed four other ways, for comparison."""

import molflash as mf
import torch
import torch.nn.functional as F
from molflash import reference
from molflash.primitives import EdgeBias, EdgeMask, Kind, NodeBias, _PairTerm
from molflash.reference import _channels


def molflash(q, k, v, kw):
    return mf.attention(q, k, v, **kw)


def eager(q, k, v, kw):
    """Plain PyTorch: every [B, H, N, N] tensor built explicitly."""
    return reference.attention(q, k, v, **kw)


def sdpa_bias(q, k, v, kw):
    """What most models do: build the dense bias, hand it to SDPA as a float mask."""
    b, h, nq, _ = q.shape
    nk = k.shape[2]
    if kw.get("gate") or kw.get("multiplier"):
        raise NotImplementedError("SDPA takes an additive mask only")
    bias = torch.zeros((b, h, nq, nk), device=q.device, dtype=torch.float32)
    for t in _as_list(kw.get("bias")):
        bias = bias + reference.dense_term(t, b, h, nq, nk)
    keep = torch.ones((b, 1, nq, nk), dtype=torch.bool, device=q.device)
    for t in _as_list(kw.get("mask")):
        keep = keep & reference.dense_term(t, b, h, nq, nk)
    if kw.get("key_mask") is not None:
        keep = keep & kw["key_mask"][:, None, None, :]
    bias = bias.masked_fill(~keep, float("-inf")).to(q.dtype)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=bias)


_BUILD = None


def sdpa_compiled_bias(q, k, v, kw):
    """SDPA with the dense bias built by a torch.compile'd builder, which fuses the elementwise
    work of the terms; the bias is still stored and its gradient still written."""
    global _BUILD
    if kw.get("key_mask") is not None:
        raise NotImplementedError("no key mask")
    _GIVEN.clear()  # a cached bias from sdpa_given_bias must not count in this peak
    if _BUILD is None:
        _BUILD = torch.compile(_dense_bias, dynamic=False)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=_BUILD(q, k, kw, q.dtype))


_GIVEN = {}


def sdpa_given_bias(q, k, v, kw):
    """SDPA with a dense bias that already exists as a leaf tensor: the attention part of a
    dense-bias pipeline alone, without building the bias or reducing its gradient onto the
    parameters. A lower bound for any method that stores the bias."""
    if kw.get("key_mask") is not None:
        raise NotImplementedError("no key mask")
    key = (id(kw), q.shape, q.dtype)
    if key not in _GIVEN:
        _GIVEN.clear()
        with torch.no_grad():
            bias = _dense_bias(q, k, kw, q.dtype)
        _GIVEN[key] = bias.requires_grad_(True)
    _GIVEN[key].grad = None  # written, not accumulated, in every call
    return F.scaled_dot_product_attention(q, k, v, attn_mask=_GIVEN[key])


def sdpa_plain(q, k, v, kw):
    """No bias at all: the speed of attention without structure, not a comparable result."""
    return F.scaled_dot_product_attention(q, k, v)


def _as_list(x):
    return [] if x is None else list(x) if isinstance(x, (list, tuple)) else [x]


# ----- FlexAttention ---------------------------------------------------------------------------


def _pair_value(t, b, h, qi, kj):
    """One pair's value of a Pair term inside score_mod, using its torch_fn on scalars."""
    qc = [c[b, qi] for c in _channels(t.q_features)]
    kc = [c[b, kj] for c in _channels(t.k_features)]
    out = t.torch_fn(*qc, *kc)
    table = t.table
    tb = table[b] if t.per_sample else table
    hh = h if tb.shape[0] > 1 else 0
    if t.kind is Kind.CODE:
        k = tb.shape[-1]
        c = out.long()
        ok = (c >= 0) & (c < k)
        return torch.where(ok, tb[hh, c.clamp(0, k - 1)], t.neutral)
    if t.kind is Kind.CODE_VALUE:
        c, x = out
        k = t.num_codes
        ok = (c >= 0) & (c < k)
        c = c.long().clamp(0, k - 1)
        if t.affine is not None:
            a, s = t.affine
            a = a[hh, c] if a.dim() == 2 else a[c]
            s = s[hh, c] if s.dim() == 2 else s[c]
            x = a * x + s
    else:
        x, ok, c = out, None, None
    lo, hi = t.grid
    g = tb.shape[-1]
    u = ((x - lo) * ((g - 1) / (hi - lo))).clamp(0, g - 1)
    i0 = u.floor().clamp(max=g - 2)
    fr = u - i0
    i0 = i0.long()
    # Flex cannot differentiate through two reads of one captured tensor, so the left values
    # and the slopes are separate tensors, both derived from the table by autograd.
    left, step = t._flex_split
    if t.code_indexed_table:
        val = left[hh, c, i0] + step[hh, c, i0] * fr
    else:
        val = left[hh, i0] + step[hh, i0] * fr
    return val if ok is None else torch.where(ok, val, t.neutral)


def _prepare_flex(kw):
    """Splits every tensor that score_mod would read more than once."""
    for t in _as_list(kw.get("bias")) + _as_list(kw.get("multiplier")):
        if isinstance(t, EdgeBias):
            t._flex_values = t.values.unbind(-2)  # one tensor per slot
        elif isinstance(t, NodeBias):
            t._flex_u = t.u.unbind(-1)
            t._flex_w = t.w.unbind(-1)
        elif isinstance(t, _PairTerm) and t.grid is not None:
            tb = t.table
            t._flex_split = (tb[..., :-1], tb[..., 1:] - tb[..., :-1])


def flex_mods(kw):
    biases = _as_list(kw.get("bias"))
    mults = _as_list(kw.get("multiplier"))
    masks = _as_list(kw.get("mask"))
    key_mask = kw.get("key_mask")
    if kw.get("gate"):
        raise NotImplementedError("FlexAttention cannot scale probabilities after the softmax")
    for t in biases + mults + masks:
        if isinstance(t, _PairTerm) and t.torch_fn is None:
            raise NotImplementedError("Flex baseline needs torch_fn")

    def score_mod(score, b, h, qi, kj):
        for t in mults:
            score = score * _pair_value(t, b, h, qi, kj)
        for t in biases:
            if isinstance(t, EdgeBias):
                for s, vals in enumerate(t._flex_values):
                    score = score + torch.where(t.index[b, qi, s] == kj, vals[b, qi, h], 0.0)
            elif isinstance(t, NodeBias):
                for u, w in zip(t._flex_u, t._flex_w, strict=True):
                    score = score + u[b, h, qi] * w[b, h, kj]
            else:
                score = score + _pair_value(t, b, h, qi, kj)
        return score

    mask_mod = None
    if masks or key_mask is not None:

        def mask_mod(b, h, qi, kj):
            keep = qi >= 0
            if key_mask is not None:
                keep = keep & key_mask[b, kj]
            for t in masks:
                if isinstance(t, EdgeMask):
                    hit = qi < 0
                    for s in range(t.index.shape[-1]):
                        hit = hit | (t.index[b, qi, s] == kj)
                    keep = keep & hit
                else:
                    qc = [c[b, qi] for c in _channels(t.q_features)]
                    kc = [c[b, kj] for c in _channels(t.k_features)]
                    keep = keep & t.torch_fn(*qc, *kc).bool()
            return keep

    return score_mod, mask_mod


_FLEX = None


def flex(q, k, v, kw):
    """torch.compile'd flex_attention with the terms translated to score_mod and mask_mod."""
    global _FLEX
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    if _FLEX is None:
        _FLEX = torch.compile(flex_attention)
    _prepare_flex(kw)
    score_mod, mask_mod = flex_mods(kw)
    block_mask = None
    if mask_mod is not None:
        from torch.nn.attention.flex_attention import noop_mask

        b, _, nq, _ = q.shape
        # Rebuilt in every call, as in training where coordinates change between steps;
        # compiled, so the mask is never materialised at N x N.
        block_mask = create_block_mask(
            mask_mod, b, None, nq, k.shape[2], device=q.device, _compile=True
        )
        # A mask_mod holding captured tensors breaks the compiled backward in this PyTorch
        # version, so the mask is applied in score_mod (-inf) and the block mask is kept only
        # to skip empty blocks.
        block_mask.mask_mod = noop_mask
        inner = score_mod

        def score_mod(score, b, h, qi, kj):
            return torch.where(mask_mod(b, h, qi, kj), inner(score, b, h, qi, kj), -float("inf"))

    return _FLEX(q, k, v, score_mod=score_mod, block_mask=block_mask)


def _dense_bias(q, k, kw, dtype):
    """The sum of all biases with masked pairs at -inf, as a dense [B, H, Nq, Nk] tensor."""
    b, h, nq, _ = q.shape
    nk = k.shape[2]
    if kw.get("gate") or kw.get("multiplier"):
        raise NotImplementedError("only additive biases and masks")
    bias = torch.zeros((b, h, nq, nk), device=q.device, dtype=torch.float32)
    for t in _as_list(kw.get("bias")):
        bias = bias + reference.dense_term(t, b, h, nq, nk)
    keep = torch.ones((b, 1, nq, nk), dtype=torch.bool, device=q.device)
    for t in _as_list(kw.get("mask")):
        keep = keep & reference.dense_term(t, b, h, nq, nk)
    return bias.masked_fill(~keep, float("-inf")).to(dtype)


def flashbias(q, k, v, kw):
    """FlashBias (Wu et al., NeurIPS 2025): exact only when the whole bias is low rank, i.e. a
    single NodeBias. Set MOLFLASH_FLASHBIAS to the directory holding flash_bias_triton.py."""
    import os
    import sys

    terms = _as_list(kw.get("bias"))
    if (
        kw.get("mask")
        or kw.get("gate")
        or kw.get("multiplier")
        or len(terms) != 1
        or not isinstance(terms[0], NodeBias)
    ):
        raise NotImplementedError("FlashBias is exact only for a low-rank bias")
    path = os.environ.get("MOLFLASH_FLASHBIAS")
    if path and path not in sys.path:
        sys.path.insert(0, path)
    from flash_bias_triton import flash_bias_func

    u, w = (x.transpose(1, 2).to(q.dtype) for x in (terms[0].u, terms[0].w))
    # All eight arguments positionally: its backward indexes needs_input_grad[5] (the mask).
    o = flash_bias_func(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), u, w, None, False, None
    )
    return o.transpose(1, 2)


_CUEQ = {}


def cuequivariance(q, k, v, kw):
    """cuEquivariance attention_pair_bias (NVIDIA, v0.13.0) with the bias precomputed densely
    (is_cached_z_proj=True). The op projects its single representation to q, k, v and a gate
    itself, so this times a full AF3-style attention block; q, k and v here only set sizes."""
    import cuequivariance_torch as cuet

    b, h, n, d = q.shape
    e = h * d
    key = (e, q.device, q.dtype)
    if key not in _CUEQ:

        def w(*shape):
            return (torch.randn(*shape, device=q.device) * e**-0.5).to(q.dtype).requires_grad_(True)

        _CUEQ[key] = dict(
            w_ln_a=torch.ones(e, device=q.device, dtype=q.dtype),
            w_proj_q=w(e, e),
            w_proj_k=w(e, e),
            w_proj_v=w(e, e),
            w_proj_g=w(e, e),
            w_proj_o=w(e, e),
        )
    single = q.transpose(1, 2).reshape(b, n, e)
    bias = _dense_bias(q, k, kw, q.dtype)
    out = cuet.attention_pair_bias(
        single, bias, None, h, b_ln_a=None, b_proj_q=None, is_cached_z_proj=True, **_CUEQ[key]
    )
    out = out[0] if isinstance(out, tuple) else out
    return out.view(b, n, h, d).transpose(1, 2)


def xformers(q, k, v, kw):
    """xformers memory_efficient_attention (cutlass) with the dense bias as a tensor."""
    import xformers.ops as xops

    bias = _dense_bias(q, k, kw, q.dtype)
    o = xops.memory_efficient_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_bias=bias
    )
    return o.transpose(1, 2)


def ds4sci(q, k, v, kw):
    """DeepSpeed DS4Sci_EvoformerAttention (CUTLASS), sequence axis of size 1, dense pair bias."""
    from deepspeed.ops.deepspeed4science import DS4Sci_EvoformerAttention

    b, h, n, _ = q.shape
    # The kernel turns -inf in the bias into NaN; a large finite value masks the same pairs.
    bias = _dense_bias(q, k, kw, q.dtype).clamp(min=-3.0e4).unsqueeze(1)  # [B, 1, H, N, N]
    qq, kk, vv = (x.transpose(1, 2).unsqueeze(1) for x in (q, k, v))  # [B, 1, N, H, D]
    o = DS4Sci_EvoformerAttention(qq, kk, vv, [None, bias])
    return o.squeeze(1).transpose(1, 2)


def turbopairformer(q, k, v, kw):
    """TurboPairFormer triangle-attention kernel (Hopper only) with a dense pair bias."""
    from turbopairformer import turbo_attention

    bias = _dense_bias(q, k, kw, torch.bfloat16).unsqueeze(0)  # [B, 1, H, N, N]
    qq, kk, vv = (x.unsqueeze(1) for x in (q, k, v))  # [B, 1, H, N, D]
    o = turbo_attention(qq, kk, vv, pair_bias=bias)  # [B, 1, N, H, D]
    return o.squeeze(1).transpose(1, 2)


METHODS = {
    "molflash": molflash,
    "flex": flex,
    "sdpa+bias": sdpa_bias,
    "sdpa+compiled bias": sdpa_compiled_bias,
    "sdpa (bias given)": sdpa_given_bias,
    "flashbias": flashbias,
    "cuequivariance": cuequivariance,
    "xformers": xformers,
    "ds4sci": ds4sci,
    "turbopairformer": turbopairformer,
    "eager": eager,
    "sdpa (no bias)": sdpa_plain,
}
