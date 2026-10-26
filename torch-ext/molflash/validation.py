"""Checks that run on every call. They read shapes, dtypes and devices only, never values, so
they cost no device synchronisation. Value checks live in :mod:`molflash.debug`."""

import torch

from .primitives import EdgeBias, EdgeMask, Kind, NodeBias, _PairTerm

MAX_HEAD_DIM = 256
_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _fail(msg: str):
    raise ValueError(msg)


def _shape(t: torch.Tensor) -> tuple:
    return tuple(t.shape)


def check_call(q, k, v, terms, *, key_mask, varlen: bool, cu_q=None, cu_k=None) -> None:
    ndim = 3 if varlen else 4
    layout = "[T, H, D]" if varlen else "[B, H, N, D]"
    for name, t in (("q", q), ("k", k), ("v", v)):
        if not isinstance(t, torch.Tensor):
            _fail(f"{name} must be a tensor, got {type(t).__name__}")
        if t.dim() != ndim:
            _fail(f"{name} must be {layout}, got shape {_shape(t)}")
        if t.dtype not in _DTYPES:
            _fail(f"{name} must be float16, bfloat16 or float32, got {t.dtype}")
        if t.dtype != q.dtype:
            _fail(f"q, k and v must share a dtype, got q {q.dtype} and {name} {t.dtype}")
        if t.device != q.device:
            _fail(f"q, k and v must be on one device, got q on {q.device} and {name} on {t.device}")
    if q.device.type != "cuda" and not _interpreting():
        _fail(
            f"molflash.attention runs on CUDA, got tensors on {q.device}. "
            "molflash.reference.attention takes the same arguments and runs on CPU."
        )
    h_axis, n_axis = (1, 0) if varlen else (1, 2)
    if not (q.shape[h_axis] == k.shape[h_axis] == v.shape[h_axis]):
        _fail(
            f"q, k and v must have the same number of heads, got {q.shape[h_axis]}, "
            f"{k.shape[h_axis]}, {v.shape[h_axis]}"
        )
    if k.shape[n_axis] != v.shape[n_axis]:
        _fail(f"k and v must have the same length, got {k.shape[n_axis]} and {v.shape[n_axis]}")
    if q.shape[-1] != k.shape[-1]:
        _fail(f"q and k must have the same head dim, got {q.shape[-1]} and {k.shape[-1]}")
    if not varlen and q.shape[0] != k.shape[0]:
        _fail(f"q and k must have the same batch size, got {q.shape[0]} and {k.shape[0]}")
    for name, d in (("q/k", q.shape[-1]), ("v", v.shape[-1])):
        if not 1 <= d <= MAX_HEAD_DIM:
            _fail(f"{name} head dim must be between 1 and {MAX_HEAD_DIM}, got {d}")

    heads = q.shape[h_axis]
    if varlen:
        b = None
        nq, nk = q.shape[0], k.shape[0]
        for name, cu in (("cu_seqlens_q", cu_q), ("cu_seqlens_k", cu_k)):
            if cu.dim() != 1 or cu.numel() < 2:
                _fail(f"{name} must be [S + 1] with S >= 1, got shape {_shape(cu)}")
            if cu.dtype not in (torch.int32, torch.int64):
                _fail(f"{name} must be an integer tensor, got {cu.dtype}")
            if cu.device != q.device:
                _fail(f"{name} must be on {q.device}, got {cu.device}")
        if cu_q.numel() != cu_k.numel():
            _fail(
                f"cu_seqlens_q and cu_seqlens_k must describe the same number of sequences, "
                f"got {cu_q.numel() - 1} and {cu_k.numel() - 1}"
            )
    else:
        b, _, nq, _ = q.shape
        nk = k.shape[2]
        if key_mask is not None:
            if _shape(key_mask) != (b, nk):
                _fail(f"key_mask must be [B, Nk] = {(b, nk)}, got {_shape(key_mask)}")
            if key_mask.device != q.device:
                _fail(f"key_mask must be on {q.device}, got {key_mask.device}")

    for term in terms:
        _check_term(term, b, heads, nq, nk, varlen, q.device)


def _interpreting() -> bool:
    import os

    return os.environ.get("TRITON_INTERPRET") == "1"


def _tok(b, n, varlen):
    return (n,) if varlen else (b, n)


def _check_term(term, b, h, nq, nk, varlen, device) -> None:
    name = type(term).__name__

    def same_device(label, t):
        if t.device != device:
            _fail(f"{name}.{label} must be on {device}, got {t.device}")

    if isinstance(term, (EdgeBias, EdgeMask)):
        idx = term.index
        same_device("index", idx)
        want = _tok(b, nq, varlen)
        if idx.dim() != len(want) + 1 or _shape(idx)[:-1] != want:
            _fail(f"{name}.index must be {_desc(want, 'K')}, got {_shape(idx)}")
        if idx.is_floating_point():
            _fail(f"{name}.index must be an integer tensor, got {idx.dtype}")
        if isinstance(term, EdgeBias):
            vals = term.values
            same_device("values", vals)
            want_v = (*_shape(idx), h)
            if _shape(vals) != want_v:
                _fail(
                    f"EdgeBias.values must be {want_v} (index shape plus heads), got {_shape(vals)}"
                )
            if not vals.is_floating_point():
                _fail(f"EdgeBias.values must be floating point, got {vals.dtype}")
        return

    if isinstance(term, NodeBias):
        same_device("u", term.u)
        same_device("w", term.w)
        if varlen:
            wu, ww = (nq, h), (nk, h)
            gu, gw = _shape(term.u)[:2], _shape(term.w)[:2]
        else:
            wu, ww = (b, h, nq), (b, h, nk)
            gu, gw = _shape(term.u)[:3], _shape(term.w)[:3]
        if term.u.dim() != len(wu) + 1 or gu != wu:
            _fail(f"NodeBias.u must be {_desc(wu, 'R')}, got {_shape(term.u)}")
        if term.w.dim() != len(ww) + 1 or gw != ww:
            _fail(f"NodeBias.w must be {_desc(ww, 'R')}, got {_shape(term.w)}")
        if term.u.shape[-1] != term.w.shape[-1]:
            _fail(
                f"NodeBias.u and .w must have the same rank, got {term.u.shape[-1]} and "
                f"{term.w.shape[-1]}"
            )
        return

    if isinstance(term, _PairTerm):
        if not term.q_features:
            _fail(f"{name} needs at least one feature tensor")
        for side, feats, n in (
            ("q_features", term.q_features, nq),
            ("k_features", term.k_features, nk),
        ):
            want = _tok(b, n, varlen)
            for i, f in enumerate(feats):
                same_device(f"{side}[{i}]", f)
                if _shape(f)[: len(want)] != want or f.dim() > len(want) + 1:
                    _fail(
                        f"{name}.{side}[{i}] must be {_desc(want)} or {_desc(want, 'C')}, "
                        f"got {_shape(f)}"
                    )
        if term.kind is Kind.KEEP:
            return
        t = term.table
        same_device("table", t)
        if not t.is_floating_point():
            _fail(f"{name}.table must be floating point, got {t.dtype}")
        lead = (b, h) if term.per_sample else (h,)
        got = _shape(t)[: len(lead)]
        if got != lead and got != (*lead[:-1], 1):
            what = "[B, H, ...]" if term.per_sample else "[H, ...]"
            _fail(
                f"{name}.table must be {what} with H = {h} (or 1 to share it across heads)"
                + (f" and B = {b}" if term.per_sample else "")
                + f", got {_shape(t)}"
            )
        if term.grid is not None:
            lo, hi = term.grid
            if not hi > lo:
                _fail(f"{name}.grid must satisfy lo < hi, got {term.grid}")
            if t.shape[-1] < 2:
                _fail(
                    f"{name}: an interpolated table needs at least 2 grid points, got {t.shape[-1]}"
                )
        if term.affine is not None:
            a, s = term.affine
            k = term.num_codes
            for label, x in (("scale", a), ("shift", s)):
                same_device(f"affine {label}", x)
                if _shape(x) not in ((k,), (h, k)):
                    _fail(
                        f"{name}.affine {label} must be [K] or [H, K] with K = {k}, got {_shape(x)}"
                    )
        if varlen and term.per_sample:
            _fail(f"{name}: per_sample tables are not supported under varlen")
        return

    _fail(f"unknown term type {type(term).__name__}")


def _desc(dims: tuple, extra: str | None = None) -> str:
    names = list(dims) + ([extra] if extra else [])
    return "[" + ", ".join(str(x) for x in names) + "]"
