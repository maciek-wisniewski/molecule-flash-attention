"""EdgeBias, NodeBias and PairBias, and the mask, multiplier and gate roles built on them."""

import dataclasses
import enum
from collections.abc import Callable, Sequence

import torch

Features = torch.Tensor | Sequence[torch.Tensor] | None

_PAIR_FNS: dict = {}


@dataclasses.dataclass(frozen=True)
class PairFunction:
    """A pair function by key, so torch.compile can treat it as a constant."""

    key: str
    name: str

    @property
    def jit(self):
        return _PAIR_FNS[self.key]

    def __call__(self, *args, **kwargs):  # lets one pair function call another
        return self.jit(*args, **kwargs)


def pair_function(fn: Callable, key: str | None = None) -> PairFunction:
    """Decorator: ``triton.jit`` plus a handle usable under ``torch.compile``."""
    import triton
    from triton.runtime.interpreter import InterpretedFunction
    from triton.runtime.jit import JITFunction

    if isinstance(fn, PairFunction):
        return fn
    if torch.compiler.is_compiling():
        raise TypeError(
            "under torch.compile, pair functions must be decorated with @molflash.pair_function "
            "where they are defined, outside the compiled code"
        )
    jit = fn if isinstance(fn, (JITFunction, InterpretedFunction)) else triton.jit(fn)
    raw = jit.fn
    name = getattr(raw, "__qualname__", getattr(raw, "__name__", "pair_function"))
    key = key or f"{getattr(raw, '__module__', '')}.{name}#{id(jit)}"
    _PAIR_FNS[key] = jit
    return PairFunction(key, name)


class Role(enum.Enum):
    BIAS = "bias"
    MASK = "mask"
    MULTIPLIER = "multiplier"
    GATE = "gate"


class Kind(enum.Enum):
    """What a pair function returns."""

    CODE = "code"  # an integer index into the table
    VALUE = "value"  # a float, interpolated on the table's grid
    CODE_VALUE = "code_value"  # (code, float)
    KEEP = "keep"  # a boolean, for masks


def _as_list(features: Features) -> list[torch.Tensor]:
    if features is None:
        return []
    if isinstance(features, torch.Tensor):
        return [features]
    return list(features)


class EdgeBias:
    """``values[i, s]`` for the pair ``(i, index[i, s])``, 0 for unlisted pairs.

    ``index [B, Nq, K]`` holds key positions (``-1`` empty), ``values [B, Nq, K, H]``. Repeated keys
    add up; positions outside ``[0, Nk)`` are ignored.
    """

    role = Role.BIAS

    def __init__(self, index: torch.Tensor, values: torch.Tensor):
        self.index = index
        self.values = values

    def __repr__(self) -> str:
        return f"EdgeBias(index={tuple(self.index.shape)}, values={tuple(self.values.shape)})"


class EdgeMask:
    """Keeps only the pairs listed in ``index`` (layout of :class:`EdgeBias`)."""

    role = Role.MASK

    def __init__(self, index: torch.Tensor):
        self.index = index

    def __repr__(self) -> str:
        return f"EdgeMask(index={tuple(self.index.shape)})"


class NodeBias:
    """``u[i] · w[j]`` per head, with ``u [B, H, Nq, R]`` and ``w [B, H, Nk, R]``."""

    role = Role.BIAS

    def __init__(self, u: torch.Tensor, w: torch.Tensor):
        self.u = u
        self.w = w

    def __repr__(self) -> str:
        return f"NodeBias(u={tuple(self.u.shape)}, w={tuple(self.w.shape)})"


class _PairTerm:
    role: Role

    def __init__(
        self,
        fn: Callable,
        features: Features = None,
        table: torch.Tensor | None = None,
        *,
        q_features: Features = None,
        k_features: Features = None,
        grid: tuple[float, float] | None = None,
        affine: tuple[torch.Tensor, torch.Tensor] | None = None,
        grad_fn: Callable | None = None,
        torch_fn: Callable | None = None,
        per_sample: bool = False,
    ):
        if features is not None and (q_features is not None or k_features is not None):
            raise ValueError("pass either features= or q_features=/k_features=, not both")
        if features is not None:
            q_features = k_features = features
        elif (q_features is None) != (k_features is None):
            raise ValueError("cross-attention needs both q_features= and k_features=")
        self.fn = pair_function(fn)
        self.q_features = _as_list(q_features)
        self.k_features = _as_list(k_features)
        self.table = table
        self.grid = None if grid is None else (float(grid[0]), float(grid[1]))
        self.affine = affine
        self.grad_fn = None if grad_fn is None else pair_function(grad_fn)
        self.torch_fn = torch_fn
        self.per_sample = per_sample
        self.kind = self._infer_kind()

    def _infer_kind(self) -> Kind:
        if self.role is Role.MASK:
            return Kind.KEEP
        if self.table is None:
            raise ValueError(f"{type(self).__name__} needs a table")
        rank = self.table.dim() - (1 if self.per_sample else 0)
        if rank not in (2, 3) or (rank == 3 and self.grid is None):
            want = "[H, K] or [H, G] or [H, K, G]" if self.grid is not None else "[H, K]"
            batch = " with a leading batch axis (per_sample=True)" if self.per_sample else ""
            raise ValueError(f"table must be {want}{batch}, got {tuple(self.table.shape)}")
        if self.grid is None:
            if self.affine is not None:
                raise ValueError("affine= transforms the interpolation coordinate; it needs grid=")
            return Kind.CODE
        if self.affine is not None or rank == 3:
            return Kind.CODE_VALUE
        return Kind.VALUE

    @property
    def code_indexed_table(self) -> bool:
        return self.grid is not None and self.table.dim() - (1 if self.per_sample else 0) == 3

    @property
    def num_codes(self) -> int:
        if self.kind is Kind.CODE:
            return self.table.shape[-1]
        if self.kind is Kind.CODE_VALUE:
            if self.code_indexed_table:
                return self.table.shape[-2]
            return self.affine[0].shape[-1]
        return 0

    @property
    def grid_size(self) -> int:
        return self.table.shape[-1] if self.grid is not None else 0

    @property
    def neutral(self) -> float:
        return 0.0 if self.role is Role.BIAS else 1.0

    def __repr__(self) -> str:
        name = self.fn.name
        shape = None if self.table is None else tuple(self.table.shape)
        return f"{type(self).__name__}(fn={name}, kind={self.kind.value}, table={shape})"


class PairBias(_PairTerm):
    """A bias for every pair: ``fn(*q_channels, *k_channels)`` indexes ``table``.

    ``fn`` is a ``@molflash.pair_function``. It returns a code ``c`` (table ``[H, K]``), a float
    ``t`` interpolated on ``grid`` (table ``[H, G]``), or ``(c, t)`` (table ``[H, K, G]``, or
    ``[H, G]`` with ``affine=(a, b)`` mapping ``t`` to ``a[c] * t + b[c]``). See docs/guide.md.

    ``grad_fn`` returns ``dt/d(channel)`` for every channel, for gradients to float features.
    ``torch_fn`` is ``fn`` in torch ops, used by ``molflash.reference`` and debug mode.
    """

    role = Role.BIAS


class PairScale(_PairTerm):
    """Multiplies ``q·k * softmax_scale``; arguments as :class:`PairBias`."""

    role = Role.MULTIPLIER


class PairGate(_PairTerm):
    """Multiplies the probabilities after the softmax; arguments as :class:`PairBias`."""

    role = Role.GATE


class PairMask(_PairTerm):
    """Keeps a pair when ``fn`` is true. Tiles with no kept pair are skipped."""

    role = Role.MASK

    def __init__(
        self,
        fn: Callable,
        features: Features = None,
        *,
        q_features: Features = None,
        k_features: Features = None,
        torch_fn: Callable | None = None,
    ):
        super().__init__(
            fn, features, None, q_features=q_features, k_features=k_features, torch_fn=torch_fn
        )


Term = EdgeBias | EdgeMask | NodeBias | PairBias | PairScale | PairGate | PairMask
