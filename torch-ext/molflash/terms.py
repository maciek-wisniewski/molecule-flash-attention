"""Ready-made terms built from the primitives. Each returns an ordinary EdgeBias, NodeBias or
Pair* object; see docs/recipes.md for writing your own."""

import math

import torch

from .primitives import (
    _PAIR_FNS,
    EdgeBias,
    NodeBias,
    PairBias,
    PairFunction,
    PairGate,
    PairMask,
)

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = tl = None


@torch.compiler.assume_constant_result
def _register(source: str, name: str) -> str:
    # torch.compile runs this at trace time instead of tracing into exec.
    key = f"molflash.terms:{source}"
    if key not in _PAIR_FNS:
        from .kernels import _compile

        # Direct insert: pair_function refuses to run while torch.compile traces.
        _PAIR_FNS[key] = _compile(source, name, {"triton": triton, "tl": tl})
    return key


def _jit(source: str, name: str) -> PairFunction:
    """A pair function with constants baked into its source, one per source text so equal
    arguments share a compiled kernel."""
    return PairFunction(_register(source, name), name)


def _f(x: float) -> str:
    return repr(float(x))


# ----- distances -----------------------------------------------------------------------------

_DIST = """
@triton.jit
def {name}(xi, yi, zi, xj, yj, zj):
    dx = xi - xj
    dy = yi - yj
    dz = zi - zj
    return {body}
"""

_DIST_GRAD = """
@triton.jit
def distance_grad(xi, yi, zi, xj, yj, zj):
    dx = xi - xj
    dy = yi - yj
    dz = zi - zj
    d = tl.sqrt(dx * dx + dy * dy + dz * dz)
    inv = tl.where(d > 0.0, 1.0 / tl.where(d > 0.0, d, 1.0), 0.0)
    return dx * inv, dy * inv, dz * inv, -dx * inv, -dy * inv, -dz * inv
"""


def _sq(xi, yi, zi, xj, yj, zj):
    return (xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2


def _distance_torch(xi, yi, zi, xj, yj, zj):
    d2 = _sq(xi, yi, zi, xj, yj, zj)
    # sqrt has an infinite derivative at 0; the kernel's grad_fn returns 0 there.
    return torch.sqrt(torch.where(d2 > 0, d2, torch.ones_like(d2))) * (d2 > 0)


def _distance_fn():
    return _jit(
        _DIST.format(name="distance", body="tl.sqrt(dx * dx + dy * dy + dz * dz)"), "distance"
    )


def _features(pos, q_pos, k_pos):
    if pos is not None:
        return dict(features=pos)
    return dict(q_features=q_pos, k_features=k_pos)


def distance_buckets(table, pos=None, *, max_distance: float, q_pos=None, k_pos=None) -> PairBias:
    """``table [H, K]`` indexed by ``floor(d / max_distance * (K - 1))``, the last bucket holding
    every distance from ``max_distance`` on."""
    k = table.shape[-1]
    width = max_distance / (k - 1)
    src = _DIST.format(
        name="distance_bucket",
        # Correctly rounded sqrt and division, so bucket edges agree with torch bit for bit.
        body="tl.minimum(tl.floor(tl.div_rn(tl.sqrt_rn(dx * dx + dy * dy + dz * dz), "
        f"{_f(width)})), {_f(k - 1)}).to(tl.int32)",
    )

    def torch_fn(*ch):
        return (torch.sqrt(_sq(*ch)) / width).floor().clamp(max=k - 1).long()

    return PairBias(
        _jit(src, "distance_bucket"), table=table, torch_fn=torch_fn, **_features(pos, q_pos, k_pos)
    )


def tabulate(fn, lo: float, hi: float, points: int, device=None) -> torch.Tensor:
    """Evaluates ``fn: [G] -> [G, H]`` (e.g. a small ``nn.Module``) on ``points`` evenly spaced
    values and returns the ``[H, G]`` table of an interpolated term. Autograd flows through.
    ``device`` defaults to that of ``fn``'s parameters."""
    if device is None:
        params = getattr(fn, "parameters", None)
        device = next(params()).device if params is not None else None
    x = torch.linspace(lo, hi, points, device=device)
    return fn(x).transpose(0, 1)


def distance_function(
    table,
    pos=None,
    *,
    grid: tuple[float, float],
    q_pos=None,
    k_pos=None,
) -> PairBias:
    """``table [H, G]`` interpolated at the distance over ``grid``; see :func:`tabulate`.
    Coordinates that require grad receive it."""
    return PairBias(
        _distance_fn(),
        table=table,
        grid=grid,
        grad_fn=_jit(_DIST_GRAD, "distance_grad"),
        torch_fn=_distance_torch,
        **_features(pos, q_pos, k_pos),
    )


_TYPED = """
@triton.jit
def typed_distance(xi, yi, zi, ti, xj, yj, zj, tj):
    dx = xi - xj
    dy = yi - yj
    dz = zi - zj
    return (ti * {n} + tj).to(tl.int32), tl.sqrt_rn(dx * dx + dy * dy + dz * dz)
"""


def typed_distance_function(
    table,
    pos,
    types,
    num_types: int,
    *,
    grid,
    scale=None,
    shift=None,
) -> PairBias:
    """A function of distance that depends on the pair of token types ``(t_i, t_j)``.

    Either ``table [H, T*T, G]`` holds one curve per type pair, or ``table [H, G]`` is shared and
    ``scale, shift [T*T]`` map the distance to ``scale[c] * d + shift[c]`` first (the
    Gaussian-basis encodings of Transformer-M and Uni-Mol have this form).
    """
    types = types.unsqueeze(-1).to(pos.dtype)
    feats = torch.cat([pos, types], dim=-1)
    affine = None if scale is None else (scale, shift)

    def torch_fn(xi, yi, zi, ti, xj, yj, zj, tj):
        return (ti * num_types + tj).long(), torch.sqrt(_sq(xi, yi, zi, xj, yj, zj))

    return PairBias(
        _jit(_TYPED.format(n=int(num_types)), "typed_distance"),
        feats,
        table,
        grid=grid,
        affine=affine,
        torch_fn=torch_fn,
    )


# ----- positions in a sequence ---------------------------------------------------------------

_RELPOS = """
@triton.jit
def relative_position(i, j):
    return tl.minimum(tl.maximum(j - i, -{m}), {m}) + {m}
"""


def relative_position(table, index, *, max_offset: int) -> PairBias:
    """``table [H, 2m + 1]`` indexed by ``clip(index_j - index_i, -m, m) + m``. ``index`` is
    ``[B, N]`` int (residue number, token position)."""
    m = int(max_offset)

    def torch_fn(i, j):
        return (j - i).clamp(-m, m) + m

    return PairBias(_jit(_RELPOS.format(m=m), "relative_position"), index, table, torch_fn=torch_fn)


def t5_buckets(num_buckets: int, max_distance: int, bidirectional: bool = True) -> torch.Tensor:
    """The T5 bucket of every offset ``j - i`` in ``[-max_distance, max_distance]``."""
    rel = torch.arange(-max_distance, max_distance + 1)
    ret = torch.zeros_like(rel)
    n = num_buckets
    if bidirectional:
        n //= 2
        ret += (rel > 0).long() * n
        rel = rel.abs()
    else:
        rel = -torch.clamp(rel, max=0)
    exact = n // 2
    large = (
        exact
        + (
            torch.log(rel.float().clamp(min=1) / exact)
            / math.log(max_distance / exact)
            * (n - exact)
        ).long()
    )
    large = torch.minimum(large, torch.full_like(large, n - 1))
    return ret + torch.where(rel < exact, rel, large)


def t5_relative_position(
    table,
    index,
    *,
    max_distance: int = 128,
    bidirectional: bool = True,
) -> PairBias:
    """T5's bucketed relative position bias. ``table [H, num_buckets]`` is the model's own
    parameter; offsets are clipped at ``max_distance``, where the buckets already saturate."""
    buckets = t5_buckets(table.shape[-1], max_distance, bidirectional).to(table.device)
    return relative_position(table[:, buckets], index, max_offset=max_distance)


_SAME = """
@triton.jit
def same_label(a, b):
    return (a == b).to(tl.int32)
"""


def same_label(table, label) -> PairBias:
    """``table [H, 2]``: entry 1 when two tokens share a label (chain, residue, molecule)."""
    return PairBias(_jit(_SAME, "same_label"), label, table, torch_fn=lambda a, b: (a == b).long())


# ----- graphs --------------------------------------------------------------------------------


def bonds(table, index, bond_type) -> EdgeBias:
    """``table [num_types, H]`` read at the type of every listed bond.

    ``index [B, N, K]`` lists bonded neighbours (``-1`` = none) and ``bond_type [B, N, K]`` their
    types; list both directions of every bond.
    """
    types = bond_type.long().clamp(min=0)
    return EdgeBias(index, table[types])


def edge_features(index, features, proj) -> EdgeBias:
    """Any per-edge embedding ``features [B, N, K, C]`` projected to heads by ``proj`` (e.g.
    ``nn.Linear(C, H)``)."""
    return EdgeBias(index, proj(features))


def node_features(u, w=None) -> NodeBias:
    """``u [B, N, H, R]`` (and ``w``, defaulting to ``u``) from any per-token embedding, e.g.
    ``Linear(C, H * R)`` reshaped. Returns the transposed layout ``NodeBias`` expects."""
    w = u if w is None else w
    return NodeBias(u.transpose(1, 2), w.transpose(1, 2))


# ----- masks and gates -----------------------------------------------------------------------


def distance_cutoff(pos=None, *, cutoff: float, q_pos=None, k_pos=None) -> PairMask:
    """Keeps pairs closer than ``cutoff``."""
    c2 = float(cutoff) ** 2
    src = _DIST.format(name="within", body=f"(dx * dx + dy * dy + dz * dz) < {_f(c2)}")
    return PairMask(
        _jit(src, "within"), torch_fn=lambda *ch: _sq(*ch) < c2, **_features(pos, q_pos, k_pos)
    )


def same_label_mask(label) -> PairMask:
    """Keeps pairs with equal labels; with a molecule index, several molecules share one
    padded row without attending to each other."""
    return PairMask(_jit(_SAME, "same_label"), label, torch_fn=lambda a, b: a == b)


def cosine_cutoff(
    pos=None,
    *,
    cutoff: float,
    points: int = 256,
    q_pos=None,
    k_pos=None,
) -> PairGate:
    """Gate ``0.5 * (cos(pi * d / cutoff) + 1)`` for ``d < cutoff`` and 0 beyond, interpolated on
    ``points`` grid points."""
    ref = pos if pos is not None else q_pos
    x = torch.linspace(0.0, cutoff, points, device=ref.device)
    curve = 0.5 * (torch.cos(math.pi * x / cutoff) + 1.0)
    table = torch.cat([curve, curve.new_zeros(1)]).unsqueeze(0)
    hi = cutoff * points / (points - 1)
    return PairGate(
        _distance_fn(),
        table=table,
        grid=(0.0, hi),
        torch_fn=_distance_torch,
        **_features(pos, q_pos, k_pos),
    )
