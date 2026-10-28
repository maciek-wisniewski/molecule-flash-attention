"""Turns the terms of one call into a hashable kernel structure and a flat list of tensors.

The structure decides which kernel is generated; the tensors are what it reads. Two calls with
the same structure share a compiled kernel whatever their sizes.
"""

import dataclasses

import torch

from .primitives import EdgeBias, EdgeMask, Kind, NodeBias, PairFunction, Role, _PairTerm


@dataclasses.dataclass(frozen=True)
class PairSpec:
    role: Role
    kind: Kind
    fn: object
    grad_fn: object | None
    q_channels: tuple[int, ...]  # channels per query-side feature tensor
    k_channels: tuple[int, ...]
    q_float: tuple[bool, ...]
    k_float: tuple[bool, ...]
    num_codes: int
    grid_size: int
    lo: float
    hi: float
    per_sample: bool
    code_indexed: bool
    has_affine: bool
    affine_per_head: bool
    q_feat_grad: tuple[bool, ...]  # which query-side tensors receive a gradient
    k_feat_grad: tuple[bool, ...]
    neutral: float
    table_grad: bool = True
    affine_grad: bool = False

    @property
    def has_table(self) -> bool:
        return self.kind is not Kind.KEEP

    @property
    def has_grid(self) -> bool:
        return self.kind in (Kind.VALUE, Kind.CODE_VALUE)

    @property
    def has_code(self) -> bool:
        return self.kind in (Kind.CODE, Kind.CODE_VALUE)

    @property
    def needs_slope(self) -> bool:
        return self.affine_grad or any(self.q_feat_grad) or any(self.k_feat_grad)


@dataclasses.dataclass(frozen=True)
class Structure:
    varlen: bool
    heads: int
    dqk: int
    dv: int
    dtype: torch.dtype
    has_key_mask: bool
    edge_slots: int  # EdgeBias slots after concatenation, 0 when absent
    edge_mask_slots: int
    node_rank: int  # NodeBias rank after concatenation, 0 when absent
    pairs: tuple[PairSpec, ...]
    # Per pair term, how the query-tile kernel accumulates table gradients; see kernels.GradMode.
    grad_modes: tuple[int, ...] = ()

    @property
    def has_mask_terms(self) -> bool:
        return self.edge_mask_slots > 0 or any(p.role is Role.MASK for p in self.pairs)

    @property
    def has_bias(self) -> bool:
        return (
            self.edge_slots > 0
            or self.node_rank > 0
            or any(p.role is not Role.MASK for p in self.pairs)
        )


@dataclasses.dataclass
class Plan:
    # Structure fields, tensor slots and per-pair slots as nested tuples of constants, so that
    # torch.compile can hand them to ``register`` at trace time.
    desc: tuple
    tensors: list[torch.Tensor]


_REGISTRY: dict[int, tuple] = {}
_KEYS: dict[tuple, int] = {}


@torch.compiler.assume_constant_result
def register(desc: tuple) -> int:
    """An integer naming a structure and its tensor layout inside a ``torch.library`` op, whose
    arguments cannot carry Python objects such as Triton functions. Runs eagerly under
    torch.compile, so the op finds the entry when it is traced."""
    st_fields, slot_items, pair_items = desc
    *head, pairs = st_fields
    specs = []
    for role, kind, fn, grad_fn, *rest in pairs:
        specs.append(
            PairSpec(
                role,
                kind,
                PairFunction(fn, fn),
                None if grad_fn is None else PairFunction(grad_fn, grad_fn),
                *rest,
            )
        )
    st = Structure(*head, tuple(specs))
    key = _KEYS.get((st, slot_items, pair_items))
    if key is None:
        key = len(_REGISTRY)
        _KEYS[(st, slot_items, pair_items)] = key
        slots = dict(slot_items)
        pair_slots = [
            {name: (list(v) if isinstance(v, tuple) else v) for name, v in ps} for ps in pair_items
        ]
        _REGISTRY[key] = (st, slots, pair_slots)
    return key


def lookup(key: int) -> tuple:
    """``(structure, slots, pair_slots)``."""
    return _REGISTRY[key]


def _listify(x) -> list:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def collect_terms(bias, mask, multiplier, gate) -> list:
    """Every term with the role its argument gives it, checked against what the term allows."""
    out = []
    for arg, role in (
        (bias, Role.BIAS),
        (mask, Role.MASK),
        (multiplier, Role.MULTIPLIER),
        (gate, Role.GATE),
    ):
        for term in _listify(arg):
            if term.role is not role:
                raise TypeError(
                    f"{type(term).__name__} is a {term.role.value} term and was passed as "
                    f"{role.value}=; pass it as {term.role.value}= instead"
                )
            out.append(term)
    return out


def token_major(t: torch.Tensor, token_dim: int) -> torch.Tensor:
    """The same tensor laid out with the token axis innermost, keeping its shape.

    A feature channel and a slot's index are each read one ``[BM]`` vector at a time inside the
    tile loop. Over ``[B, N, K]`` that vector strides by ``K``, so every one of its ``BM``
    elements lands in a different sector and the load costs ``BM`` transactions; over the same
    tensor with ``N`` innermost it is one. The kernels read the strides by name, so nothing
    there changes, and a tensor already in this layout is returned untouched.
    """
    dim = token_dim % t.dim()
    order = [d for d in range(t.dim()) if d != dim] + [dim]
    inverse = [order.index(d) for d in range(t.dim())]
    return t.permute(order).contiguous().permute(inverse)


def _feature_layout(features: list[torch.Tensor], varlen: bool):
    """``[B, N, C]`` views (``[1, T, C]`` under varlen), channel counts and float flags."""
    views, chans, floats = [], [], []
    token_dims = 1 if varlen else 2
    for f in features:
        if f.dim() == token_dims:
            f = f.unsqueeze(-1)
        is_float = f.is_floating_point()
        f = f.float() if is_float else f.to(torch.int32)
        # A channel is read one [BM] vector at a time, so the token axis has to be the
        # innermost one. `xyz[..., 0]` arrives with stride 3 and would cost BM transactions.
        f = token_major(f, -2)
        if varlen:
            f = f.unsqueeze(0)
        views.append(f)
        chans.append(int(f.shape[-1]))
        floats.append(is_float)
    return views, tuple(chans), tuple(floats)


def build_plan(q, k, v, terms: list, key_mask, varlen: bool) -> Plan:
    heads = q.shape[1] if not varlen else q.shape[1]
    tensors: list[torch.Tensor] = []
    slots: dict[str, int | None] = {}

    def put(name, t):
        slots[name] = None if t is None else len(tensors)
        if t is not None:
            tensors.append(t)

    put("key_mask", None if key_mask is None else key_mask.to(torch.int8).contiguous())

    edges = [t for t in terms if isinstance(t, EdgeBias)]
    emasks = [t for t in terms if isinstance(t, EdgeMask)]
    nodes = [t for t in terms if isinstance(t, NodeBias)]
    pairs = [t for t in terms if isinstance(t, _PairTerm)]
    if len(emasks) > 1:
        raise ValueError("pass at most one EdgeMask; list every allowed key in its slots")

    def token_view(t):
        return t.unsqueeze(0) if varlen else t

    if edges:
        index = torch.cat([e.index for e in edges], dim=-1).to(torch.int32)
        values = torch.cat([e.values for e in edges], dim=-2).float()
        put("edge_index", token_view(token_major(index, -2)))
        put("edge_values", token_view(token_major(values, -3)))
        edge_slots = int(index.shape[-1])
    else:
        put("edge_index", None)
        put("edge_values", None)
        edge_slots = 0

    if emasks:
        index = emasks[0].index.to(torch.int32)
        put("emask_index", token_view(token_major(index, -2)))
        emask_slots = int(index.shape[-1])
    else:
        put("emask_index", None)
        emask_slots = 0

    if nodes:
        u = torch.cat([n.u for n in nodes], dim=-1).float()
        w = torch.cat([n.w for n in nodes], dim=-1).float()
        if varlen:  # [T, H, R] -> [1, H, T, R]
            u = u.permute(1, 0, 2).unsqueeze(0)
            w = w.permute(1, 0, 2).unsqueeze(0)
        put("node_u", u)
        put("node_w", w)
        node_rank = int(u.shape[-1])
    else:
        put("node_u", None)
        put("node_w", None)
        node_rank = 0

    specs, pair_slots = [], []
    for term in pairs:
        qv, qc, qf = _feature_layout(term.q_features, varlen)
        kv, kc, kf = _feature_layout(term.k_features, varlen)
        if qc != kc or qf != kf:
            raise ValueError(
                f"{term!r}: query-side and key-side features must have the same channels and "
                f"dtypes, got {qc}/{kc} channels"
            )
        ps: dict[str, object] = {}
        ps["q"] = [len(tensors) + i for i in range(len(qv))]
        tensors.extend(qv)
        ps["k"] = [len(tensors) + i for i in range(len(kv))]
        tensors.extend(kv)
        if term.table is not None:
            ps["table"] = len(tensors)
            tensors.append(term.table.float())
        has_affine = term.affine is not None
        if has_affine:
            a, b = term.affine
            ps["a"] = len(tensors)
            tensors.append(a.float())
            ps["b"] = len(tensors)
            tensors.append(b.float())
        grad_ok = term.grad_fn is not None

        def wants(feats, floats, grad_ok=grad_ok, term=term):
            out = []
            for f, is_float in zip(feats, floats, strict=True):
                needs = is_float and f.requires_grad
                if needs and not grad_ok:
                    raise ValueError(
                        f"{term!r}: a feature requires grad but the term has no grad_fn, so the "
                        "kernel cannot differentiate through it. Pass grad_fn=, or detach the "
                        "feature if its gradient is not needed."
                    )
                out.append(needs)
            return tuple(out)

        q_grad = wants(term.q_features, qf)
        k_grad = wants(term.k_features, kf)
        specs.append(
            (
                term.role,
                term.kind,
                term.fn.key,
                term.grad_fn.key if (any(q_grad) or any(k_grad)) else None,
                qc,
                kc,
                qf,
                kf,
                int(term.num_codes),
                int(term.grid_size),
                term.grid[0] if term.grid else 0.0,
                term.grid[1] if term.grid else 1.0,
                term.per_sample,
                term.code_indexed_table,
                has_affine,
                has_affine and term.affine[0].dim() == 2,
                q_grad,
                k_grad,
                term.neutral,
                term.table is not None and term.table.requires_grad,
                has_affine and (term.affine[0].requires_grad or term.affine[1].requires_grad),
            )
        )
        pair_slots.append(
            tuple((name, tuple(x) if isinstance(x, list) else x) for name, x in ps.items())
        )

    # Sizes that shape the generated kernel are specialised; sequence lengths stay dynamic.
    st_fields = (
        varlen,
        int(heads),
        int(q.shape[-1]),
        int(v.shape[-1]),
        q.dtype,
        key_mask is not None,
        edge_slots,
        emask_slots,
        node_rank,
        tuple(specs),
    )
    desc = (st_fields, tuple(slots.items()), tuple(pair_slots))
    return Plan(desc, tensors)
