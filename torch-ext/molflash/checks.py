"""Value checks, off by default. Enable with ``MOLFLASH_DEBUG=1`` or ``with molflash.debug():``.

They synchronise with the device and recompute a few rows with the reference, so they belong in
a first run of a new model, not in training.
"""

import contextlib
import os
import threading

import torch

from .primitives import EdgeBias, EdgeMask, NodeBias, _PairTerm

_STATE = threading.local()


def enabled() -> bool:
    flag = getattr(_STATE, "on", None)
    if flag is not None:
        return flag
    return os.environ.get("MOLFLASH_DEBUG", "") not in ("", "0")


@contextlib.contextmanager
def debug(on: bool = True):
    prev = getattr(_STATE, "on", None)
    _STATE.on = on
    try:
        yield
    finally:
        _STATE.on = prev


class MolflashDebugError(ValueError):
    pass


def _fail(msg):
    raise MolflashDebugError(msg)


def _finite(name, t):
    if t.is_floating_point() and not torch.isfinite(t).all():
        bad = (~torch.isfinite(t)).nonzero()[0].tolist()
        _fail(f"{name} has a non-finite value at index {bad}")


def check_values(q, k, v, terms, key_mask, cu_q=None, cu_k=None) -> None:
    for name, t in (("q", q), ("k", k), ("v", v)):
        _finite(name, t)
    varlen = cu_q is not None
    if varlen:
        for name, cu, total in (
            ("cu_seqlens_q", cu_q, q.shape[0]),
            ("cu_seqlens_k", cu_k, k.shape[0]),
        ):
            if cu[0] != 0 or (cu[1:] < cu[:-1]).any() or cu[-1] != total:
                _fail(
                    f"{name} must start at 0, never decrease and end at {total}, got "
                    f"{cu.tolist()[:8]}{'...' if cu.numel() > 8 else ''}"
                )
        lk = (cu_k[1:] - cu_k[:-1]).long()
        seq_of_q = torch.repeat_interleave(
            torch.arange(lk.numel(), device=q.device), (cu_q[1:] - cu_q[:-1]).long()
        )
        limit = lk[seq_of_q]
    else:
        limit = None
    nk = k.shape[0] if varlen else k.shape[2]
    for term in terms:
        name = type(term).__name__
        if isinstance(term, (EdgeBias, EdgeMask)):
            idx = term.index.long()
            hi = limit[:, None].expand_as(idx) if varlen else torch.full_like(idx, nk)
            bad = (idx < -1) | (idx >= hi)
            if bad.any():
                pos = bad.nonzero()[0].tolist()
                hint = (
                    " Under varlen the index counts from the start of the sequence."
                    if varlen
                    else ""
                )
                _fail(
                    f"{name}.index holds {idx[tuple(pos)].item()} at {pos}, outside "
                    f"[-1, {hi[tuple(pos)].item()}).{hint}"
                )
            if isinstance(term, EdgeBias):
                _finite(f"{name}.values", term.values)
        elif isinstance(term, NodeBias):
            _finite("NodeBias.u", term.u)
            _finite("NodeBias.w", term.w)
        elif isinstance(term, _PairTerm):
            for i, f in enumerate(term.q_features + term.k_features):
                _finite(f"{name} feature {i}", f)
            if term.table is not None:
                _finite(f"{name}.table", term.table)
    if key_mask is not None and not key_mask.any(dim=1).all():
        empty = (~key_mask.any(dim=1)).nonzero()[:, 0].tolist()
        _fail(f"key_mask removes every key of samples {empty}; their outputs would be zero")


def _rows_of(terms, rows):
    out = []
    for t in terms:
        if isinstance(t, EdgeBias):
            out.append(EdgeBias(t.index[:, rows], t.values[:, rows]))
        elif isinstance(t, EdgeMask):
            out.append(EdgeMask(t.index[:, rows]))
        elif isinstance(t, NodeBias):
            out.append(NodeBias(t.u[:, :, rows], t.w))
        else:
            clone = object.__new__(type(t))
            clone.__dict__.update(t.__dict__)
            clone.q_features = [f[:, rows] for f in t.q_features]
            out.append(clone)
    return out


def compare_rows(o, q, k, v, groups, key_mask, scale, rows: int = 8) -> None:
    """Recomputes a few query rows with the reference and raises if they disagree."""
    from . import reference

    n = q.shape[2]
    idx = torch.randperm(n, device=q.device)[: min(rows, n)].sort().values
    sub = {name: _rows_of(ts, idx) for name, ts in groups.items()}
    with torch.no_grad():
        ref = reference.attention(q[:, :, idx], k, v, key_mask=key_mask, softmax_scale=scale, **sub)
    got = o.detach()[:, :, idx].float()
    err = (got - ref.float()).abs().max().item()
    tol = 1e-3 if q.dtype == torch.float32 else 5e-2
    if err > tol * max(ref.float().abs().max().item(), 1.0):
        _fail(
            f"kernel and reference disagree by {err:.3g} on rows {idx.tolist()}. Check that "
            "every pair function returns the same value as its torch_fn; otherwise please "
            "report this with the inputs."
        )
