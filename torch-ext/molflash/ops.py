import torch

from . import checks, kernels
from ._ops import add_op_namespace_prefix
from .plan import _listify, build_plan, collect_terms, lookup, register
from .validation import check_call


def _grads_for_tensors(st, slots, pair_slots, tensors, grads) -> list[torch.Tensor]:
    out = [torch.empty(0, device=t.device) for t in tensors]

    def put(slot, g):
        out[slot] = g.view(tensors[slot].shape).to(tensors[slot].dtype)

    if st.edge_slots:
        put(slots["edge_values"], grads["edge_values"])
    if st.node_rank:
        put(slots["node_u"], grads["node_u"])
        put(slots["node_w"], grads["node_w"])
    for p, spec in enumerate(st.pairs):
        ps = pair_slots[p]
        if spec.has_table:
            put(ps["table"], grads[("table", p)])
        if spec.has_affine:
            put(ps["a"], grads[("a", p)])
            put(ps["b"], grads[("b", p)])
        for f, need in enumerate(spec.q_feat_grad):
            if need:
                put(ps["q"][f], grads[("q", p, f)])
        for f, need in enumerate(spec.k_feat_grad):
            if need:
                put(ps["k"][f], grads[("k", p, f)])
    return out


@torch.library.custom_op(add_op_namespace_prefix("fwd"), mutates_args=())
def _fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensors: list[torch.Tensor],
    key: int,
    scale: float,
    cu_q: torch.Tensor | None,
    cu_k: torch.Tensor | None,
    max_q: int,
    max_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    st, slots, pair_slots = lookup(key)
    return kernels.forward(st, q, k, v, tensors, slots, pair_slots, scale, cu_q, cu_k, max_q, max_k)


@_fwd.register_fake
def _(q, k, v, tensors, key, scale, cu_q, cu_k, max_q, max_k):
    st = lookup(key)[0]
    if st.varlen:
        o = q.new_empty((q.shape[0], q.shape[1], v.shape[-1]))
        lse = q.new_empty((1, q.shape[1], q.shape[0]), dtype=torch.float32)
    else:
        o = q.new_empty((*q.shape[:3], v.shape[-1]))
        lse = q.new_empty(q.shape[:3], dtype=torch.float32)
    return o, lse


@torch.library.custom_op(add_op_namespace_prefix("bwd"), mutates_args=())
def _bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    do: torch.Tensor,
    tensors: list[torch.Tensor],
    key: int,
    scale: float,
    cu_q: torch.Tensor | None,
    cu_k: torch.Tensor | None,
    max_q: int,
    max_k: int,
) -> list[torch.Tensor]:
    st, slots, pair_slots = lookup(key)
    dq, dk, dv, grads = kernels.backward(
        st, q, k, v, o, lse, do, tensors, slots, pair_slots, scale, cu_q, cu_k, max_q, max_k
    )
    return [dq, dk, dv, *_grads_for_tensors(st, slots, pair_slots, tensors, grads)]


@_bwd.register_fake
def _(q, k, v, o, lse, do, tensors, key, scale, cu_q, cu_k, max_q, max_k):
    st, slots, pair_slots = lookup(key)
    grads = [torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)]
    has_grad = set()
    if st.edge_slots:
        has_grad.add(slots["edge_values"])
    if st.node_rank:
        has_grad |= {slots["node_u"], slots["node_w"]}
    for p, spec in enumerate(st.pairs):
        ps = pair_slots[p]
        if spec.has_table:
            has_grad.add(ps["table"])
        if spec.has_affine:
            has_grad |= {ps["a"], ps["b"]}
        has_grad |= {ps["q"][f] for f, n in enumerate(spec.q_feat_grad) if n}
        has_grad |= {ps["k"][f] for f, n in enumerate(spec.k_feat_grad) if n}
    for i, t in enumerate(tensors):
        grads.append(torch.empty_like(t) if i in has_grad else t.new_empty(0, dtype=torch.float32))
    return grads


def _setup_context(ctx, inputs, output):
    q, k, v, tensors, key, scale, cu_q, cu_k, max_q, max_k = inputs
    o, lse = output
    ctx.save_for_backward(q, k, v, o, lse, *tensors)
    ctx.meta = (key, scale, cu_q, cu_k, max_q, max_k)
    ctx.mark_non_differentiable(lse)


_UNSUPPORTED = "is not implemented yet; see docs/guide.md#not-supported-yet"


def _backward(ctx, do, _dlse):
    if torch.is_grad_enabled():
        raise NotImplementedError(
            f"molflash: the gradient of the gradient (create_graph=True) {_UNSUPPORTED}"
        )
    key, scale, cu_q, cu_k, max_q, max_k = ctx.meta
    q, k, v, o, lse, *tensors = ctx.saved_tensors
    grads = _bwd(q, k, v, o, lse, do, tensors, key, scale, cu_q, cu_k, max_q, max_k)
    dq, dk, dv, *rest = grads
    rest = [g if g.shape == t.shape else None for g, t in zip(rest, tensors, strict=True)]
    return dq, dk, dv, rest, None, None, None, None, None, None


torch.library.register_autograd(
    add_op_namespace_prefix("fwd"), _backward, setup_context=_setup_context
)


def _reject_transforms(tensors) -> None:
    """Without these checks torch.func.jvp and forward-mode AD return a zero tangent."""
    if torch.compiler.is_compiling():
        return
    from torch._C._functorch import is_functorch_wrapped_tensor

    for t in tensors:
        if t is None:
            continue
        if is_functorch_wrapped_tensor(t):
            raise NotImplementedError(
                f"molflash: torch.func transforms (jvp, vmap, ...) {_UNSUPPORTED}"
            )
        if torch.autograd.forward_ad.unpack_dual(t).tangent is not None:
            raise NotImplementedError(f"molflash: forward-mode AD {_UNSUPPORTED}")


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
) -> torch.Tensor:
    """Attention with structural terms computed in the kernel.

    Args:
        q, k: ``[B, H, Nq, D]`` and ``[B, H, Nk, D]``.
        v: ``[B, H, Nk, Dv]``.
        bias: ``EdgeBias``, ``NodeBias`` or ``PairBias`` terms, added to the scores.
        mask: ``EdgeMask`` or ``PairMask`` terms; a pair is kept if every mask keeps it.
        multiplier: ``PairScale`` terms, multiplying ``q.k * softmax_scale``.
        gate: ``PairGate`` terms, multiplying the probabilities after the softmax.
        key_mask: ``[B, Nk]`` bool, ``False`` for padding.
        softmax_scale: defaults to ``D ** -0.5``.

    Returns ``[B, H, Nq, Dv]``. Queries with no key left return zeros.
    """
    terms = collect_terms(bias, mask, multiplier, gate)
    check_call(q, k, v, terms, key_mask=key_mask, varlen=False)
    scale = q.shape[-1] ** -0.5 if softmax_scale is None else float(softmax_scale)
    checking = checks.enabled() and not torch.compiler.is_compiling()
    if checking:
        checks.check_values(q, k, v, terms, key_mask)
    plan = build_plan(q, k, v, terms, key_mask, varlen=False)
    _reject_transforms([q, k, v, *plan.tensors])
    key = register(plan.desc)
    o, _ = _fwd(q, k, v, plan.tensors, key, scale, None, None, 0, 0)
    if checking:
        groups = {
            n: _listify(g)
            for n, g in dict(bias=bias, mask=mask, multiplier=multiplier, gate=gate).items()
        }
        checks.compare_rows(o, q, k, v, groups, key_mask, scale)
    return o


def attention_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    *,
    bias=None,
    mask=None,
    multiplier=None,
    gate=None,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """:func:`attention` on sequences packed without padding.

    ``q`` is ``[Tq, H, D]``, ``k`` and ``v`` are ``[Tk, H, D]``; sequence ``i`` occupies rows
    ``cu_seqlens[i]:cu_seqlens[i + 1]``. Term tensors are packed the same way, and
    ``EdgeBias.index`` counts positions from the start of the sequence. Omitting ``max_seqlen_*``
    costs a device synchronisation. Returns ``[Tq, H, Dv]``.
    """
    terms = collect_terms(bias, mask, multiplier, gate)
    check_call(q, k, v, terms, key_mask=None, varlen=True, cu_q=cu_seqlens_q, cu_k=cu_seqlens_k)
    cu_q = cu_seqlens_q.to(torch.int32).contiguous()
    cu_k = cu_seqlens_k.to(torch.int32).contiguous()
    if max_seqlen_q is None:
        max_seqlen_q = int((cu_q[1:] - cu_q[:-1]).max())
    if max_seqlen_k is None:
        max_seqlen_k = int((cu_k[1:] - cu_k[:-1]).max())
    scale = q.shape[-1] ** -0.5 if softmax_scale is None else float(softmax_scale)
    if checks.enabled() and not torch.compiler.is_compiling():
        checks.check_values(q, k, v, terms, None, cu_q, cu_k)
    plan = build_plan(q, k, v, terms, None, varlen=True)
    _reject_transforms([q, k, v, *plan.tensors])
    key = register(plan.desc)
    o, _ = _fwd(q, k, v, plan.tensors, key, scale, cu_q, cu_k, int(max_seqlen_q), int(max_seqlen_k))
    return o
