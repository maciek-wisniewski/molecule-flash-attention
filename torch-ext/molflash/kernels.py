"""Generated Triton kernels: forward, backward over key tiles (dk, dv, key-side features) and
backward over query tiles (dq and every other gradient, so the key-tile kernel has no atomics)."""

import dataclasses
import functools
import linecache
import os

import torch

from .plan import PairSpec, Structure
from .primitives import Kind, Role
from .sampling import BackwardSampling, sample_weights, weighted_gradient

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover
    triton = None
    tl = None
    HAS_TRITON = False

_FILL = "-1.0e30"


def _next_pow2(x: int, floor: int = 16) -> int:
    p = floor
    while p < x:
        p *= 2
    return p


class GradMode:
    """Table-gradient accumulation: registers, a private slice per program, or the table."""

    REDUCE = 0
    PRIVATE = 1
    SHARED = 2


# Measured on A100 at N=8192 (fwd+bwd, ms): registers win up to 32 entries (interpolated
# table of 16: 22 vs 55 private), private slices from 64 on (128 entries: 37 vs 132).
_REDUCE_MAX_CODES = 32
_REDUCE_MAX_GRID = 32
_PRIVATE_BUDGET = 256 * 2**20


def _entries(spec) -> int:
    if spec.kind is Kind.CODE:
        return spec.num_codes
    return (spec.num_codes if spec.code_indexed else 1) * spec.grid_size


def choose_grad_modes(st, batch: int, max_q: int) -> tuple[int, ...]:
    bm = _blocks(st, None, backward=True)[0]
    programs = triton.cdiv(max(max_q, 1), bm) * batch * st.heads if HAS_TRITON else 0
    modes = []
    for spec in st.pairs:
        if not spec.has_table or not (spec.table_grad or spec.affine_grad):
            modes.append(-1)
        elif (
            spec.kind is Kind.CODE
            and spec.num_codes <= _REDUCE_MAX_CODES
            or (
                spec.kind is not Kind.CODE
                and not spec.code_indexed
                and not spec.affine_grad
                and spec.grid_size <= _REDUCE_MAX_GRID
            )
        ):
            modes.append(GradMode.REDUCE)
        elif programs * (_entries(spec) + 2 * spec.num_codes) * 4 <= _PRIVATE_BUDGET:
            modes.append(GradMode.PRIVATE)
        else:
            modes.append(GradMode.SHARED)
    return tuple(modes)


class _Lines:
    def __init__(self):
        self.lines: list[str] = []
        self.depth = 0

    def __call__(self, *lines: str) -> None:
        for line in lines:
            self.lines.append("    " * self.depth + line)

    def indent(self):
        lines = self

        class _Ctx:
            def __enter__(self):
                lines.depth += 1

            def __exit__(self, *exc):
                lines.depth -= 1

        return _Ctx()

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


class _Orient:
    """Which axis of the score tile holds queries. ``qk``: rows are queries (forward, dq);
    ``kq``: rows are keys (dk/dv), so the tile is the transpose of the forward's."""

    def __init__(self, name: str):
        self.name = name
        self.tile = "(BM, BN)" if name == "qk" else "(BN, BM)"
        self.q_axis = 1 if name == "qk" else 0  # reduce over keys -> per query
        self.k_axis = 0 if name == "qk" else 1  # reduce over queries -> per key

    def q(self, expr: str) -> str:
        return f"({expr})[:, None]" if self.name == "qk" else f"({expr})[None, :]"

    def k(self, expr: str) -> str:
        return f"({expr})[None, :]" if self.name == "qk" else f"({expr})[:, None]"


def _tensor_params(name: str, ndim: int) -> list[str]:
    return [name] + [f"{name}_s{i}" for i in range(ndim)]


class _Source:
    """Builds the parameter list shared by the three kernels and the per-term code."""

    def __init__(self, st: Structure):
        self.st = st
        self.params: list[str] = []
        self.globals: dict[str, object] = {"triton": triton, "tl": tl}
        for p, spec in enumerate(st.pairs):
            self.globals[f"FN{p}"] = spec.fn.jit
            if spec.grad_fn is not None:
                self.globals[f"GFN{p}"] = spec.grad_fn.jit

    def mode(self, p: int) -> int:
        return self.st.grad_modes[p] if self.st.grad_modes else GradMode.SHARED

    # -- parameters --------------------------------------------------------------------------

    def term_params(self, grads: bool) -> list[str]:
        """Pointers and strides of every term tensor, plus gradient buffers when ``grads``."""
        st = self.st
        out: list[str] = []
        if st.has_key_mask:
            out += _tensor_params("KM", 2)
        if st.varlen:
            out += ["CUQ", "CUK"]
        if st.edge_slots:
            out += _tensor_params("EI", 3) + _tensor_params("EV", 4)
            if grads:
                out += _tensor_params("DEV", 4)
        if st.edge_mask_slots:
            out += _tensor_params("MI", 3)
        if st.node_rank:
            out += _tensor_params("NU", 4) + _tensor_params("NW", 4)
            if grads:
                out += _tensor_params("DNU", 4) + _tensor_params("DNW", 4)
        for p, spec in enumerate(st.pairs):
            for f in range(len(spec.q_channels)):
                out += _tensor_params(f"PQ{p}_{f}", 3)
            for f in range(len(spec.k_channels)):
                out += _tensor_params(f"PK{p}_{f}", 3)
            if spec.has_table:
                out += _tensor_params(f"T{p}", 4)
                if grads:
                    out += _tensor_params(f"DT{p}", 4)
                    if self.mode(p) == GradMode.PRIVATE:
                        out += [f"PRIV{p}"] + (
                            [f"PRIVA{p}", f"PRIVB{p}"] if spec.has_affine else []
                        )
            if spec.has_affine:
                out += _tensor_params(f"A{p}", 2) + _tensor_params(f"B{p}", 2)
                if grads:
                    out += _tensor_params(f"DA{p}", 2) + _tensor_params(f"DB{p}", 2)
            if grads:
                for f, need in enumerate(spec.q_feat_grad):
                    if need:
                        out += _tensor_params(f"DPQ{p}_{f}", 3)
                for f, need in enumerate(spec.k_feat_grad):
                    if need:
                        out += _tensor_params(f"DPK{p}_{f}", 3)
        return out

    @staticmethod
    def constexprs() -> list[str]:
        return [
            "BM: tl.constexpr",
            "BN: tl.constexpr",
            "BDQK: tl.constexpr",
            "BDV: tl.constexpr",
            "BR: tl.constexpr",
            "DQK: tl.constexpr",
            "DV: tl.constexpr",
            "R: tl.constexpr",
            "PREC: tl.constexpr",
            "SEM: tl.constexpr",
        ]

    # -- shared prologue ---------------------------------------------------------------------

    def prologue(self, L: _Lines) -> None:
        L("off_bh = tl.program_id(1)", "off_b = off_bh // H", "off_h = off_bh % H")
        if self.st.varlen:
            L(
                "q_start = tl.load(CUQ + off_b)",
                "q_len = tl.load(CUQ + off_b + 1) - q_start",
                "k_start = tl.load(CUK + off_b)",
                "k_len = tl.load(CUK + off_b + 1) - k_start",
            )
        else:
            L("q_start = 0", "q_len = NQ", "k_start = 0", "k_len = NK")
        L("offs_dq = tl.arange(0, BDQK)", "offs_dv = tl.arange(0, BDV)")
        if self.st.node_rank:
            L("offs_r = tl.arange(0, BR)")

    # -- per-term code -----------------------------------------------------------------------

    def _side_loads(
        self, L: _Lines | None, o: _Orient, p: int, spec: PairSpec, side: str
    ) -> list[str]:
        """The names of one side's channel values, emitting their loads when ``L`` is given."""
        chans = spec.q_channels if side == "q" else spec.k_channels
        floats = spec.q_float if side == "q" else spec.k_float
        tok, valid = ("q_tok", "m_valid") if side == "q" else ("k_tok", "n_valid")
        lay = o.q if side == "q" else o.k
        names, i = [], 0
        for f, (c_n, is_f) in enumerate(zip(chans, floats, strict=True)):
            other = "0.0" if is_f else "0"
            for c in range(c_n):
                n = f"P{side.upper()}{p}_{f}"
                if L is not None:
                    L(
                        f"x{side}{p}_{i} = "
                        + lay(
                            f"tl.load({n} + off_b * {n}_s0 + {tok} * {n}_s1 + {c} * {n}_s2, "
                            f"mask={valid}, other={other})"
                        )
                    )
                names.append(f"x{side}{p}_{i}")
                i += 1
        return names

    def hoist(self, L: _Lines, o: _Orient) -> None:
        """The tile loop walks one axis, so every feature on the other axis is loop invariant.

        Emitting those loads once before the loop rather than letting the compiler lift them is
        what makes the saving certain: the backward kernels write through pointers the compiler
        cannot prove disjoint from these, so it keeps the loads inside.
        """
        side = "q" if o.name == "qk" else "k"
        for p, spec in enumerate(self.st.pairs):
            self._side_loads(L, o, p, spec, side)
        if self.st.edge_slots and o.name == "qk":
            self._edge_loads(L)
        if self.st.edge_mask_slots and o.name == "qk":
            self._emask_loads(L)

    def _channel_loads(self, L: _Lines, o: _Orient, p: int, spec: PairSpec) -> tuple[list, list]:
        """The loop-variant side's loads; the other side came from :meth:`hoist`."""
        inv = "q" if o.name == "qk" else "k"
        qs = self._side_loads(None if inv == "q" else L, o, p, spec, "q")
        ks = self._side_loads(None if inv == "k" else L, o, p, spec, "k")
        return qs, ks

    def _edge_loads(self, L: _Lines) -> None:
        """One ``ei``/``ev`` pair per slot, unrolled here rather than by ``tl.static_range``."""
        for es in range(self.st.edge_slots):
            L(
                f"ei{es} = tl.load(EI + off_b * EI_s0 + q_tok * EI_s1 + {es} * EI_s2, "
                "mask=m_valid, other=-1)",
                f"ev{es} = tl.load(EV + off_b * EV_s0 + q_tok * EV_s1 + {es} * EV_s2 "
                "+ off_h * EV_s3, mask=m_valid, other=0.0).to(tl.float32)",
            )

    def edge_hit(self, L: _Lines) -> None:
        """``ehit_any``: does any listed pair of these query rows fall in this key tile?

        A query row lists ``K`` key positions, so a score tile away from them holds no edge at
        all, and bonds and hops are sequence-local, which makes that nearly every tile. The
        test runs on the ``[BM]`` slot vectors rather than on ``[BM, BN]`` tiles, because the
        tile's keys are exactly ``n_lo .. n_lo + BN``: ``K`` range checks on a vector decide
        it, and the comparisons the term needs stay on the path that is taken.
        """
        in_tile = " | ".join(
            f"((ei{es} >= n_lo) & (ei{es} < n_lo + BN))" for es in range(self.st.edge_slots)
        )
        L(f"ehit_any = tl.max(({in_tile}).to(tl.int32)) > 0")

    def edge_tile(self, o: _Orient, es: int) -> str:
        """The ``[BM, BN]`` membership tile of one slot."""
        return f"({o.q(f'ei{es}')} == {o.k('offs_n')})"

    def _emask_loads(self, L: _Lines) -> None:
        for es in range(self.st.edge_mask_slots):
            L(
                f"mi{es} = tl.load(MI + off_b * MI_s0 + q_tok * MI_s1 + {es} * MI_s2, "
                "mask=m_valid, other=-1)"
            )

    def masks(self, L: _Lines, o: _Orient) -> None:
        """Updates ``keep`` with the key mask and every mask term."""
        st = self.st
        if st.has_key_mask:
            L(
                "km = tl.load(KM + off_b * KM_s0 + offs_n * KM_s1, mask=n_valid, other=0)",
                f"keep = keep & {o.k('km != 0')}",
            )
        if st.edge_mask_slots:
            if o.name != "qk":
                self._emask_loads(L)
            hits = " | ".join(
                f"({o.q(f'mi{es}')} == {o.k('offs_n')})" for es in range(st.edge_mask_slots)
            )
            L(f"keep = keep & ({hits})")
        for p, spec in enumerate(st.pairs):
            if spec.role is not Role.MASK:
                continue
            qs, ks = self._channel_loads(L, o, p, spec)
            L(
                f"keep = keep & (tl.broadcast_to(FN{p}({', '.join(qs + ks)}), {o.tile}).to(tl.int32) != 0)"
            )

    def values(self, L: _Lines, o: _Orient, need_slope: bool) -> None:
        """Evaluates every non-mask pair term and applies it to ``s``, multipliers first so the
        biases land on the scaled score. Gates are combined into ``gate``.

        A bias is folded into ``s`` the moment it is read, which ends the life of its code and
        value tiles there rather than at the end of the term list. That is safe in the backward
        too: a bias term's upstream gradient is ``dS`` alone, and only a multiplier or a gate
        divides its role's product by itself and so needs the other terms' values to survive.
        """
        st = self.st
        mults, gates = [], []
        for role in (Role.MULTIPLIER, Role.BIAS, Role.GATE):
            for p, spec in enumerate(st.pairs):
                if spec.role is not role:
                    continue
                self._pair_value(L, o, p, spec, need_slope and spec.needs_slope)
                if role is Role.MULTIPLIER:
                    mults.append(p)
                elif role is Role.GATE:
                    gates.append(p)
                else:
                    L(f"s = s + val{p}")
            if role is Role.MULTIPLIER and mults:
                L("mult = " + " * ".join(f"val{p}" for p in mults), "s = s * mult")
        if st.edge_slots:
            if o.name != "qk":
                self._edge_loads(L)
            self.edge_hit(L)
            L("if ehit_any:")
            with L.indent():
                # One sum, not one `s = s + ...` per slot: the adds chain through `s` and a
                # hop term brings 40 of them to a score tile.
                adds = " + ".join(
                    f"tl.where({self.edge_tile(o, es)}, {o.q(f'ev{es}')}, 0.0)"
                    for es in range(st.edge_slots)
                )
                L(f"s = s + ({adds})")
        if st.node_rank:
            if o.name == "qk":
                L(
                    "nu = tl.load(NU + off_b * NU_s0 + off_h * NU_s1 + q_tok[:, None] * NU_s2 "
                    "+ offs_r[None, :] * NU_s3, mask=m_valid[:, None] & (offs_r[None, :] < R), "
                    "other=0.0)",
                    "nw = tl.load(NW + off_b * NW_s0 + off_h * NW_s1 + k_tok[:, None] * NW_s2 "
                    "+ offs_r[None, :] * NW_s3, mask=n_valid[:, None] & (offs_r[None, :] < R), "
                    "other=0.0)",
                    "s = s + tl.dot(nu.to(q.dtype), tl.trans(nw.to(q.dtype)), input_precision=PREC)",
                )
            else:
                L(
                    "nu = tl.load(NU + off_b * NU_s0 + off_h * NU_s1 + q_tok[:, None] * NU_s2 "
                    "+ offs_r[None, :] * NU_s3, mask=m_valid[:, None] & (offs_r[None, :] < R), "
                    "other=0.0)",
                    "s = s + tl.dot(nw.to(q.dtype), tl.trans(nu.to(q.dtype)), input_precision=PREC)",
                )
        if gates:
            L("gate = " + " * ".join(f"val{p}" for p in gates))

    def _pair_value(self, L: _Lines, o: _Orient, p: int, spec: PairSpec, slope: bool) -> None:
        qs, ks = self._channel_loads(L, o, p, spec)
        call = f"FN{p}({', '.join(qs + ks)})"
        neutral = repr(float(spec.neutral))
        T = f"T{p}"
        # The batch and head offsets are scalars, so 64 bits there cost nothing; the code and
        # grid offsets are one per pair, and keeping them 32 bits halves both the arithmetic
        # and the registers a live code tile occupies. `_table_args` enforces the bound.
        L(f"tb{p} = {T} + off_b.to(tl.int64) * {T}_s0 + off_h.to(tl.int64) * {T}_s1")
        if spec.kind is Kind.CODE:
            L(
                f"c{p} = tl.broadcast_to({call}, {o.tile}).to(tl.int32)",
                f"ok{p} = (c{p} >= 0) & (c{p} < {spec.num_codes}) & valid",
                f"cs{p} = tl.where(ok{p}, c{p}, 0)",
                f"val{p} = tl.load(tb{p} + cs{p} * {T}_s2, mask=ok{p}, other={neutral})",
            )
            return
        if spec.kind is Kind.CODE_VALUE:
            L(
                f"c{p}, t{p} = {call}",
                f"c{p} = tl.broadcast_to(c{p}, {o.tile}).to(tl.int32)",
                f"ok{p} = (c{p} >= 0) & (c{p} < {spec.num_codes}) & valid",
                f"cs{p} = tl.where(ok{p}, c{p}, 0)",
                f"t{p} = tl.broadcast_to(t{p}, {o.tile}).to(tl.float32)",
            )
            if spec.code_indexed:
                L(f"tb{p} = tb{p} + cs{p} * {T}_s2")
        else:
            L(
                f"t{p} = tl.broadcast_to({call}, {o.tile}).to(tl.float32)",
                f"ok{p} = valid",
            )
        if spec.has_affine:
            A, B = f"A{p}", f"B{p}"
            L(
                f"a{p} = tl.load({A} + off_h * {A}_s0 + cs{p} * {A}_s1, mask=ok{p}, other=1.0)",
                f"bb{p} = tl.load({B} + off_h * {B}_s0 + cs{p} * {B}_s1, mask=ok{p}, other=0.0)",
                f"traw{p} = t{p}",
                f"t{p} = a{p} * t{p} + bb{p}",
            )
        g = spec.grid_size
        inv = (g - 1) / (spec.hi - spec.lo)
        lo = repr(float(spec.lo))
        L(
            f"t{p} = tl.where(t{p} == t{p}, t{p}, {lo})",
            f"u{p} = (t{p} - {lo}) * {inv!r}",
            f"inr{p} = (u{p} >= 0.0) & (u{p} <= {float(g - 1)!r})",
            f"u{p} = tl.minimum(tl.maximum(u{p}, 0.0), {float(g - 1)!r})",
            f"i0f{p} = tl.minimum(tl.floor(u{p}), {float(g - 2)!r})",
            f"fr{p} = u{p} - i0f{p}",
            f"i0{p} = i0f{p}.to(tl.int32)",
            f"v0{p} = tl.load(tb{p} + i0{p} * {T}_s3, mask=ok{p}, other=0.0)",
            f"v1{p} = tl.load(tb{p} + (i0{p} + 1) * {T}_s3, mask=ok{p}, other=0.0)",
            f"val{p} = tl.where(ok{p}, v0{p} + (v1{p} - v0{p}) * fr{p}, {neutral})",
        )
        if slope:
            L(f"sl{p} = tl.where(ok{p} & inr{p}, (v1{p} - v0{p}) * {inv!r}, 0.0)")

    def upstream(self, L: _Lines, p: int, spec: PairSpec) -> None:
        """``dval{p}``: the loss gradient with respect to the term's value at each pair."""
        others = [q for q, sp in enumerate(self.st.pairs) if sp.role is spec.role and q != p]
        prod = "".join(f" * val{q}" for q in others)
        if spec.role is Role.BIAS:
            L(f"dval{p} = dS")
        elif spec.role is Role.MULTIPLIER:
            L(f"dval{p} = dS * qk{prod}")
        else:
            L(f"dval{p} = dG{prod}")

    def table_grad_init(self, L: _Lines) -> None:
        for p, spec in enumerate(self.st.pairs):
            if spec.has_table and self.mode(p) == GradMode.REDUCE:
                L(f"racc{p} = tl.zeros([{_next_pow2(_entries(spec), 2)}], dtype=tl.float32)")
            if spec.has_table and self.mode(p) == GradMode.PRIVATE:
                L(
                    f"pid{p} = (off_bh * MT + start_m).to(tl.int64)",
                    f"pb{p} = PRIV{p} + pid{p} * {_entries(spec)}",
                )
                if spec.has_affine:
                    L(
                        f"pa{p} = PRIVA{p} + pid{p} * {spec.num_codes}",
                        f"pbb{p} = PRIVB{p} + pid{p} * {spec.num_codes}",
                    )

    def table_grad_flush(self, L: _Lines) -> None:
        for p, spec in enumerate(self.st.pairs):
            if spec.has_table and self.mode(p) == GradMode.REDUCE:
                D = f"DT{p}"
                n = _entries(spec)
                stride = f"{D}_s2" if spec.kind is Kind.CODE else f"{D}_s3"
                L(
                    f"offs_c{p} = tl.arange(0, {_next_pow2(n, 2)})",
                    f"tl.atomic_add({D} + off_b * {D}_s0 + off_h * {D}_s1 + offs_c{p} * {stride}, "
                    f"racc{p}, mask=offs_c{p} < {n}, sem=SEM)",
                )

    def table_grads(self, L: _Lines, p: int, spec: PairSpec) -> None:
        """Query-tile kernel only: table and affine gradients."""
        mode = self.mode(p)
        m = f"ok{p} & keep"
        if mode == -1:
            return
        if mode == GradMode.REDUCE:
            # One register accumulator per table entry, flushed once per program.
            n = _entries(spec)
            sel = f"tl.arange(0, {_next_pow2(n, 2)}) == c"
            if spec.kind is Kind.CODE:
                L(f"dvm{p} = tl.where({m}, dval{p}, 0.0)", f"for c in tl.static_range({n}):")
                with L.indent():
                    L(f"racc{p} += tl.where({sel}, tl.sum(tl.where(cs{p} == c, dvm{p}, 0.0)), 0.0)")
            else:
                L(
                    f"lo{p} = tl.where({m}, dval{p} * (1.0 - fr{p}), 0.0)",
                    f"hi{p} = tl.where({m}, dval{p} * fr{p}, 0.0)",
                    f"for c in tl.static_range({n}):",
                )
                with L.indent():
                    L(
                        f"racc{p} += tl.where({sel}, tl.sum(tl.where(i0{p} == c, lo{p}, 0.0)) "
                        f"+ tl.sum(tl.where(i0{p} == c - 1, hi{p}, 0.0)), 0.0)"
                    )
            return
        if mode == GradMode.PRIVATE:
            g = spec.grid_size if spec.kind is not Kind.CODE else 1
            base, s_code, s_grid = f"pb{p}", f"{g}", "1"
            a_ptr = f"pa{p} + cs{p}"
            b_ptr = f"pbb{p} + cs{p}"
        else:
            D = f"DT{p}"
            L(f"gb{p} = {D} + off_b.to(tl.int64) * {D}_s0 + off_h.to(tl.int64) * {D}_s1")
            base, s_code, s_grid = f"gb{p}", f"{D}_s2", f"{D}_s3"
            a_ptr = f"DA{p} + off_h * DA{p}_s0 + cs{p} * DA{p}_s1"
            b_ptr = f"DB{p} + off_h * DB{p}_s0 + cs{p} * DB{p}_s1"
        if spec.kind is Kind.CODE:
            if spec.table_grad:
                L(
                    f"hot{p} = tl.max(tl.where({m}, cs{p}, -1))",
                    f"hot_grad{p} = tl.sum(tl.where({m} & (cs{p} == hot{p}), dval{p}, 0.0))",
                    f"tl.atomic_add({base} + tl.maximum(hot{p}, 0) * {s_code}, hot_grad{p}, "
                    f"mask=hot{p} >= 0, sem=SEM)",
                    f"tl.atomic_add({base} + cs{p} * {s_code}, dval{p}, "
                    f"mask=({m}) & (cs{p} != hot{p}), sem=SEM)",
                )
            return
        row = f"{base} + cs{p} * {s_code}" if spec.code_indexed else base
        if spec.table_grad:
            L(
                f"tl.atomic_add({row} + i0{p} * {s_grid}, dval{p} * (1.0 - fr{p}), mask={m}, sem=SEM)",
                f"tl.atomic_add({row} + (i0{p} + 1) * {s_grid}, dval{p} * fr{p}, mask={m}, sem=SEM)",
            )
        if spec.affine_grad:
            L(
                f"dt{p} = dval{p} * sl{p}",
                f"tl.atomic_add({a_ptr}, dt{p} * traw{p}, mask={m}, sem=SEM)",
                f"tl.atomic_add({b_ptr}, dt{p}, mask={m}, sem=SEM)",
            )

    def feature_grads(self, L: _Lines, o: _Orient, p: int, spec: PairSpec, side: str) -> None:
        """Accumulates ``g{side}{p}_{i}`` for the channels of tensors that need a gradient."""
        need = spec.q_feat_grad if side == "q" else spec.k_feat_grad
        if not any(need):
            return
        chans = spec.q_channels if side == "q" else spec.k_channels
        nq = sum(spec.q_channels)
        nk = sum(spec.k_channels)
        dt = f"dval{p} * sl{p}" + (f" * a{p}" if spec.has_affine else "")
        qs = [f"xq{p}_{i}" for i in range(nq)]
        ks = [f"xk{p}_{i}" for i in range(nk)]
        outs = [f"dq{p}_{i}" for i in range(nq)] + [f"dk{p}_{i}" for i in range(nk)]
        L(
            f"dtr{p} = tl.where(ok{p} & keep, {dt}, 0.0)",
            f"{', '.join(outs)}, = GFN{p}({', '.join(qs + ks)})",
        )
        axis = o.q_axis if side == "q" else o.k_axis
        i = 0
        for f, c_n in enumerate(chans):
            for _ in range(c_n):
                if need[f]:
                    L(
                        f"g{side}{p}_{i} += tl.sum(tl.where(ok{p} & keep, dtr{p} * "
                        f"tl.broadcast_to(d{side}{p}_{i}, {o.tile}), 0.0), axis={axis})"
                    )
                i += 1

    def feature_grad_init(self, L: _Lines, side: str, length: str) -> None:
        for p, spec in enumerate(self.st.pairs):
            need = spec.q_feat_grad if side == "q" else spec.k_feat_grad
            chans = spec.q_channels if side == "q" else spec.k_channels
            i = 0
            for f, c_n in enumerate(chans):
                for _ in range(c_n):
                    if need[f]:
                        L(f"g{side}{p}_{i} = tl.zeros([{length}], dtype=tl.float32)")
                    i += 1

    def feature_grad_store(self, L: _Lines, side: str) -> None:
        tok, valid = ("q_tok", "m_valid") if side == "q" else ("k_tok", "n_valid")
        for p, spec in enumerate(self.st.pairs):
            need = spec.q_feat_grad if side == "q" else spec.k_feat_grad
            chans = spec.q_channels if side == "q" else spec.k_channels
            i = 0
            for f, c_n in enumerate(chans):
                for c in range(c_n):
                    if need[f]:
                        n = f"DP{side.upper()}{p}_{f}"
                        # Every head adds into the same per-token gradient.
                        L(
                            f"tl.atomic_add({n} + off_b * {n}_s0 + {tok} * {n}_s1 + {c} * {n}_s2, "
                            f"g{side}{p}_{i}, mask={valid}, sem=SEM)"
                        )
                    i += 1


def _signature(name: str, params: list[str], src: _Source) -> str:
    return f"def {name}(\n    " + ",\n    ".join(params + src.constexprs()) + ",\n):"


def _gen_fwd(src: _Source) -> str:
    st = src.st
    o = _Orient("qk")
    params = (
        _tensor_params("Q", 4)
        + _tensor_params("K", 4)
        + _tensor_params("V", 4)
        + _tensor_params("O", 4)
        + _tensor_params("LSE", 3)
        + _tensor_params("MAXS", 3)
        + src.term_params(grads=False)
        + ["H", "NQ", "NK", "SCALE"]
    )
    L = _Lines()
    L("@triton.jit", _signature("_fwd", params, src))
    with L.indent():
        src.prologue(L)
        L(
            "start_m = tl.program_id(0)",
            "offs_m = start_m * BM + tl.arange(0, BM)",
            "m_valid = offs_m < q_len",
            "q_tok = q_start + offs_m",
            "q = tl.load(Q + off_b * Q_s0 + off_h * Q_s1 + q_tok[:, None] * Q_s2 "
            "+ offs_dq[None, :] * Q_s3, mask=m_valid[:, None] & (offs_dq[None, :] < DQK), other=0.0)",
            "m_i = tl.full([BM], float('-inf'), dtype=tl.float32)",
            "l_i = tl.zeros([BM], dtype=tl.float32)",
            "acc = tl.zeros([BM, BDV], dtype=tl.float32)",
        )
        src.hoist(L, o)
        L("for start_n in range(0, k_len, BN):")
        with L.indent():
            L(
                "offs_n = start_n + tl.arange(0, BN)",
                "n_lo = start_n",
                "n_valid = offs_n < k_len",
                "k_tok = k_start + offs_n",
                "valid = m_valid[:, None] & n_valid[None, :]",
                "keep = valid",
            )
            src.masks(L, o)
            skip = st.has_mask_terms
            if skip:
                L("if tl.max(keep.to(tl.int32)) > 0:")
                L.depth += 1
            L(
                "k = tl.load(K + off_b * K_s0 + off_h * K_s1 + k_tok[:, None] * K_s2 "
                "+ offs_dq[None, :] * K_s3, mask=n_valid[:, None] & (offs_dq[None, :] < DQK), "
                "other=0.0)",
                "v = tl.load(V + off_b * V_s0 + off_h * V_s1 + k_tok[:, None] * V_s2 "
                "+ offs_dv[None, :] * V_s3, mask=n_valid[:, None] & (offs_dv[None, :] < DV), "
                "other=0.0)",
                "qk = tl.dot(q, tl.trans(k), input_precision=PREC) * SCALE",
                "s = qk",
            )
            src.values(L, o, need_slope=False)
            L(
                f"s = tl.where(keep, s, {_FILL})",
                "m_new = tl.maximum(m_i, tl.max(s, 1))",
                "alpha = tl.exp(m_i - m_new)",
                "p = tl.where(keep, tl.exp(s - m_new[:, None]), 0.0)",
                "l_i = l_i * alpha + tl.sum(p, 1)",
            )
            pv = "p * gate" if any(sp.role is Role.GATE for sp in st.pairs) else "p"
            L(
                f"acc = acc * alpha[:, None] + tl.dot(({pv}).to(v.dtype), v, input_precision=PREC)",
                "m_i = m_new",
            )
            if skip:
                L.depth -= 1
        L(
            "empty = l_i == 0.0",
            "l_safe = tl.where(empty, 1.0, l_i)",
            "m_safe = tl.where(empty, 0.0, m_i)",
            "out = acc / l_safe[:, None]",
            "tl.store(O + off_b * O_s0 + off_h * O_s1 + q_tok[:, None] * O_s2 + offs_dv[None, :] "
            "* O_s3, out.to(O.dtype.element_ty), mask=m_valid[:, None] & (offs_dv[None, :] < DV))",
            "tl.store(LSE + off_b * LSE_s0 + off_h * LSE_s1 + q_tok * LSE_s2, "
            "m_safe + tl.log(l_safe), mask=m_valid)",
            # The largest score this row saw, before the softmax. `m_i` carries it
            # already (the online softmax subtracts it for stability), so this is a
            # store and no arithmetic: one float per query row, never per pair. A row
            # with no key left keeps the 0.0 that `m_safe` substitutes for -inf.
            "tl.store(MAXS + off_b * MAXS_s0 + off_h * MAXS_s1 + q_tok * MAXS_s2, "
            "m_safe, mask=m_valid)",
        )
    return L.text()


def _bwd_common_params(src: _Source) -> list[str]:
    return (
        _tensor_params("Q", 4)
        + _tensor_params("K", 4)
        + _tensor_params("V", 4)
        + _tensor_params("DO", 4)
        + _tensor_params("LSE", 3)
        + _tensor_params("DELTA", 3)
        + _tensor_params("SAMPLE", 3)
    )


def _gen_dkdv(src: _Source) -> str:
    st = src.st
    o = _Orient("kq")
    params = (
        _bwd_common_params(src)
        + _tensor_params("DK", 4)
        + _tensor_params("DV_", 4)
        + src.term_params(grads=True)
        + ["H", "NQ", "NK", "SCALE", "SAMPLED: tl.constexpr", "SAMPLE_BLOCK: tl.constexpr"]
    )
    has_gate = any(sp.role is Role.GATE for sp in st.pairs)
    has_mult = any(sp.role is Role.MULTIPLIER for sp in st.pairs)
    L = _Lines()
    L("@triton.jit", _signature("_bwd_dkdv", params, src))
    with L.indent():
        src.prologue(L)
        L(
            "start_n = tl.program_id(0)",
            "offs_n = start_n * BN + tl.arange(0, BN)",
            "n_lo = start_n * BN",
            "n_valid = offs_n < k_len",
            "k_tok = k_start + offs_n",
            "k = tl.load(K + off_b * K_s0 + off_h * K_s1 + k_tok[:, None] * K_s2 "
            "+ offs_dq[None, :] * K_s3, mask=n_valid[:, None] & (offs_dq[None, :] < DQK), other=0.0)",
            "v = tl.load(V + off_b * V_s0 + off_h * V_s1 + k_tok[:, None] * V_s2 "
            "+ offs_dv[None, :] * V_s3, mask=n_valid[:, None] & (offs_dv[None, :] < DV), other=0.0)",
            "dk = tl.zeros([BN, BDQK], dtype=tl.float32)",
            "dv = tl.zeros([BN, BDV], dtype=tl.float32)",
        )
        if st.node_rank:
            L(
                "nw = tl.load(NW + off_b * NW_s0 + off_h * NW_s1 + k_tok[:, None] * NW_s2 "
                "+ offs_r[None, :] * NW_s3, mask=n_valid[:, None] & (offs_r[None, :] < R), other=0.0)",
                "dnw = tl.zeros([BN, BR], dtype=tl.float32)",
            )
        src.feature_grad_init(L, "k", "BN")
        src.hoist(L, o)
        L("for start_m in range(0, q_len, BM):")
        with L.indent():
            L("sample_weight = 1.0", "if SAMPLED:")
            with L.indent():
                L(
                    "sample_weight = tl.load(SAMPLE + off_b * SAMPLE_s0 + off_h * SAMPLE_s1 "
                    "+ (start_m // SAMPLE_BLOCK) * SAMPLE_s2)"
                )
            L("if sample_weight != 0.0:")
            L.depth += 1
            L(
                "offs_m = start_m + tl.arange(0, BM)",
                "m_valid = offs_m < q_len",
                "q_tok = q_start + offs_m",
                "valid = n_valid[:, None] & m_valid[None, :]",
                "keep = valid",
            )
            src.masks(L, o)
            skip = st.has_mask_terms
            if skip:
                L("if tl.max(keep.to(tl.int32)) > 0:")
                L.depth += 1
            L(
                "q = tl.load(Q + off_b * Q_s0 + off_h * Q_s1 + q_tok[:, None] * Q_s2 "
                "+ offs_dq[None, :] * Q_s3, mask=m_valid[:, None] & (offs_dq[None, :] < DQK), "
                "other=0.0)",
                "do = tl.load(DO + off_b * DO_s0 + off_h * DO_s1 + q_tok[:, None] * DO_s2 "
                "+ offs_dv[None, :] * DO_s3, mask=m_valid[:, None] & (offs_dv[None, :] < DV), "
                "other=0.0)",
                "lse = tl.load(LSE + off_b * LSE_s0 + off_h * LSE_s1 + q_tok * LSE_s2, "
                "mask=m_valid, other=0.0)",
                "delta = tl.load(DELTA + off_b * DELTA_s0 + off_h * DELTA_s1 + q_tok * DELTA_s2, "
                "mask=m_valid, other=0.0)",
                "qk = tl.dot(k, tl.trans(q), input_precision=PREC) * SCALE",
                "s = qk",
            )
            src.values(L, o, need_slope=True)
            L(
                "p = tl.where(keep, tl.exp(s - lse[None, :]), 0.0)",
                "gv = tl.dot(v, tl.trans(do), input_precision=PREC)",
            )
            if has_gate:
                L("dG = p * gv", "dS = p * (gate * gv - delta[None, :])", "pv = p * gate")
            else:
                L("dS = p * (gv - delta[None, :])", "pv = p")
            L("dv += tl.dot(pv.to(do.dtype), do, input_precision=PREC)")
            dsq = "dS * mult" if has_mult else "dS"
            L(f"dk += tl.dot(({dsq}).to(q.dtype), q, input_precision=PREC)")
            if st.node_rank:
                L("dnw += tl.dot(dS.to(q.dtype), nu.to(q.dtype), input_precision=PREC)")
            for p, spec in enumerate(st.pairs):
                if spec.role is Role.MASK or not any(spec.k_feat_grad):
                    continue
                src.upstream(L, p, spec)
                src.feature_grads(L, o, p, spec, "k")
            if skip:
                L.depth -= 1
            L.depth -= 1
        L(
            "kv_mask = n_valid[:, None]",
            "tl.store(DK + off_b * DK_s0 + off_h * DK_s1 + k_tok[:, None] * DK_s2 + offs_dq[None, :] "
            "* DK_s3, (dk * SCALE).to(DK.dtype.element_ty), mask=kv_mask & (offs_dq[None, :] < DQK))",
            "tl.store(DV_ + off_b * DV__s0 + off_h * DV__s1 + k_tok[:, None] * DV__s2 + offs_dv[None, :] "
            "* DV__s3, dv.to(DV_.dtype.element_ty), mask=kv_mask & (offs_dv[None, :] < DV))",
        )
        if st.node_rank:
            L(
                "tl.store(DNW + off_b * DNW_s0 + off_h * DNW_s1 + k_tok[:, None] * DNW_s2 + offs_r[None, :] "
                "* DNW_s3, dnw, mask=kv_mask & (offs_r[None, :] < R))"
            )
        src.feature_grad_store(L, "k")
    return L.text()


def _gen_dq(src: _Source) -> str:
    st = src.st
    o = _Orient("qk")
    params = (
        _bwd_common_params(src)
        + _tensor_params("DQ", 4)
        + src.term_params(grads=True)
        + ["H", "NQ", "NK", "SCALE", "MT", "SAMPLED: tl.constexpr", "SAMPLE_BLOCK: tl.constexpr"]
    )
    has_gate = any(sp.role is Role.GATE for sp in st.pairs)
    has_mult = any(sp.role is Role.MULTIPLIER for sp in st.pairs)
    L = _Lines()
    L("@triton.jit", _signature("_bwd_dq", params, src))
    with L.indent():
        src.prologue(L)
        L(
            "start_m = tl.program_id(0)",
            "offs_m = start_m * BM + tl.arange(0, BM)",
            "m_valid = offs_m < q_len",
            "q_tok = q_start + offs_m",
            "q = tl.load(Q + off_b * Q_s0 + off_h * Q_s1 + q_tok[:, None] * Q_s2 "
            "+ offs_dq[None, :] * Q_s3, mask=m_valid[:, None] & (offs_dq[None, :] < DQK), other=0.0)",
            "do = tl.load(DO + off_b * DO_s0 + off_h * DO_s1 + q_tok[:, None] * DO_s2 "
            "+ offs_dv[None, :] * DO_s3, mask=m_valid[:, None] & (offs_dv[None, :] < DV), other=0.0)",
            "lse = tl.load(LSE + off_b * LSE_s0 + off_h * LSE_s1 + q_tok * LSE_s2, "
            "mask=m_valid, other=0.0)",
            "delta = tl.load(DELTA + off_b * DELTA_s0 + off_h * DELTA_s1 + q_tok * DELTA_s2, "
            "mask=m_valid, other=0.0)",
            "dq = tl.zeros([BM, BDQK], dtype=tl.float32)",
        )
        if st.node_rank:
            L("dnu = tl.zeros([BM, BR], dtype=tl.float32)")
        if st.edge_slots:
            L("dev = tl.zeros([BM, KESP], dtype=tl.float32)")
        src.feature_grad_init(L, "q", "BM")
        src.table_grad_init(L)
        src.hoist(L, o)
        L("sample_weight = 1.0", "if SAMPLED:")
        with L.indent():
            L(
                "sample_weight = tl.load(SAMPLE + off_b * SAMPLE_s0 + off_h * SAMPLE_s1 "
                "+ (start_m * BM // SAMPLE_BLOCK) * SAMPLE_s2, mask=start_m * BM < q_len, other=0.0)"
            )
        L("if sample_weight != 0.0:")
        L.depth += 1
        L("for start_n in range(0, k_len, BN):")
        with L.indent():
            L(
                "offs_n = start_n + tl.arange(0, BN)",
                "n_lo = start_n",
                "n_valid = offs_n < k_len",
                "k_tok = k_start + offs_n",
                "valid = m_valid[:, None] & n_valid[None, :]",
                "keep = valid",
            )
            src.masks(L, o)
            skip = st.has_mask_terms
            if skip:
                L("if tl.max(keep.to(tl.int32)) > 0:")
                L.depth += 1
            L(
                "k = tl.load(K + off_b * K_s0 + off_h * K_s1 + k_tok[:, None] * K_s2 "
                "+ offs_dq[None, :] * K_s3, mask=n_valid[:, None] & (offs_dq[None, :] < DQK), "
                "other=0.0)",
                "v = tl.load(V + off_b * V_s0 + off_h * V_s1 + k_tok[:, None] * V_s2 "
                "+ offs_dv[None, :] * V_s3, mask=n_valid[:, None] & (offs_dv[None, :] < DV), "
                "other=0.0)",
                "qk = tl.dot(q, tl.trans(k), input_precision=PREC) * SCALE",
                "s = qk",
            )
            src.values(L, o, need_slope=True)
            L(
                "p = tl.where(keep, tl.exp(s - lse[:, None]), 0.0)",
                "gv = tl.dot(do, tl.trans(v), input_precision=PREC)",
            )
            if has_gate:
                L("dG = p * gv", "dS = p * (gate * gv - delta[:, None])")
            else:
                L("dS = p * (gv - delta[:, None])")
            dsq = "dS * mult" if has_mult else "dS"
            L(f"dq += tl.dot(({dsq}).to(k.dtype), k, input_precision=PREC)")
            if st.node_rank:
                L("dnu += tl.dot(dS.to(k.dtype), nw.to(k.dtype), input_precision=PREC)")
            if st.edge_slots:
                L("if ehit_any:")
                with L.indent():
                    for es in range(st.edge_slots):
                        L(
                            f"hit{es} = tl.sum(tl.where({src.edge_tile(o, es)}, dS, 0.0), axis=1)",
                            f"dev += tl.where(tl.arange(0, KESP)[None, :] == {es}, "
                            f"hit{es}[:, None], 0.0)",
                        )
            for p, spec in enumerate(st.pairs):
                if spec.role is Role.MASK:
                    continue
                src.upstream(L, p, spec)
                src.table_grads(L, p, spec)
                src.feature_grads(L, o, p, spec, "q")
            if skip:
                L.depth -= 1
        L.depth -= 1
        L(
            "tl.store(DQ + off_b * DQ_s0 + off_h * DQ_s1 + q_tok[:, None] * DQ_s2 + offs_dq[None, :] "
            "* DQ_s3, (dq * SCALE).to(DQ.dtype.element_ty), mask=m_valid[:, None] "
            "& (offs_dq[None, :] < DQK))",
        )
        if st.node_rank:
            L(
                "tl.store(DNU + off_b * DNU_s0 + off_h * DNU_s1 + q_tok[:, None] * DNU_s2 + offs_r[None, :] "
                "* DNU_s3, dnu, mask=m_valid[:, None] & (offs_r[None, :] < R))"
            )
        if st.edge_slots:
            L(
                "offs_es = tl.arange(0, KESP)",
                "tl.store(DEV + off_b * DEV_s0 + q_tok[:, None] * DEV_s1 + offs_es[None, :] * DEV_s2 "
                "+ off_h * DEV_s3, dev, mask=m_valid[:, None] & (offs_es[None, :] < KES))",
            )
        src.feature_grad_store(L, "q")
        src.table_grad_flush(L)
    return L.text()


def _extra_constexprs(st: Structure) -> list[str]:
    out = []
    if st.edge_slots:
        out += ["KES: tl.constexpr", "KESP: tl.constexpr"]
    return out


_GEN_COUNTER = [0]


def _compile(source: str, name: str, glb: dict):
    _GEN_COUNTER[0] += 1
    filename = f"<molflash-generated-{_GEN_COUNTER[0]}-{name}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    # Triton reads __module__ off every function it compiles.
    glb.setdefault("__name__", "molflash._generated")
    exec(compile(source, filename, "exec"), glb)
    return glb[name]


@functools.cache
def generated(st: Structure):
    """``(fwd, dkdv, dq, sources)`` for ``st``, compiled once per structure."""
    if not HAS_TRITON:
        raise RuntimeError("molflash kernels need Triton; use molflash.reference on CPU")
    src = _Source(st)
    extra = _extra_constexprs(st)
    original = src.constexprs

    def constexprs():
        return original() + extra

    src.constexprs = constexprs
    sources = {"fwd": _gen_fwd(src), "dkdv": _gen_dkdv(src), "dq": _gen_dq(src)}
    glb = dict(src.globals)
    fwd = _compile(sources["fwd"], "_fwd", glb)
    dkdv = _compile(sources["dkdv"], "_bwd_dkdv", glb)
    dq = _compile(sources["dq"], "_bwd_dq", glb)
    if os.environ.get("MOLFLASH_DUMP_KERNELS"):
        print("\n\n".join(sources.values()))
    return fwd, dkdv, dq, sources


# -----------------------------------------------------------------------------------------------
# launching
# -----------------------------------------------------------------------------------------------


def _interpret() -> bool:
    return os.environ.get("TRITON_INTERPRET") == "1"


def _blocks(st: Structure, device, backward: bool = False) -> tuple[int, int, int, int]:
    """``(BM, BN, num_warps, num_stages)``, measured on A100 at N=8192, bf16, head dim 64.

    Terms keep several ``[BM, BN]`` tiles live (codes, interpolation weights, slopes), so
    registers rather than arithmetic set the speed: with terms the forward is fastest at 32x64
    and the backward at 32x32; without terms 64x64 is fastest both ways.
    """
    if _interpret():
        return 16, 16, 4, 1
    big = max(st.dqk, st.dv)
    terms = st.has_bias or st.has_mask_terms
    if st.dtype == torch.float32 or big > 64:
        return 32, 32, 4, 1
    if terms:
        return (32, 32, 4, 1) if backward else (32, 64, 4, 1)
    return 64, 64, 4, 1


def _strides(t: torch.Tensor, ndim: int) -> list[int]:
    s = list(t.stride())
    assert len(s) == ndim, (t.shape, ndim)
    return s


def _table4(spec: PairSpec, table: torch.Tensor) -> torch.Tensor:
    """Every table as ``[Bt, H, K', G']``; the batch stride is zeroed for a shared table."""
    t = table if spec.per_sample else table.unsqueeze(0)
    if spec.kind is Kind.CODE:
        t = t.unsqueeze(-1)
    elif not spec.code_indexed:
        t = t.unsqueeze(-2)
    return t


def _t_args(t: torch.Tensor, ndim: int, zero_batch: bool = False) -> list:
    s = _strides(t, ndim)
    if zero_batch:
        s[0] = 0
    return [t, *s]


#: The kernel indexes a table's code and grid axes in 32 bits, which is what keeps a live code
#: tile at one register per pair. 2^31 float entries is an 8 GB slice, unreachable on any GPU
#: that also holds the activations, so this is a guard and not a limit anyone meets.
_MAX_TABLE_SLICE = 2**31


def _table_args(t4: torch.Tensor, spec: PairSpec) -> list:
    """A table with one head row is shared by all heads: its head stride is zeroed."""
    if t4.shape[2] * t4.shape[3] >= _MAX_TABLE_SLICE:
        raise ValueError(
            f"a table slice of {t4.shape[2]} x {t4.shape[3]} exceeds the kernel's 32-bit "
            "index range; split the term"
        )
    args = _t_args(t4, 4, not spec.per_sample)
    if t4.shape[1] == 1:
        args[2] = 0
    return args


def _affine2(t: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if t.dim() == 1:
        return t.unsqueeze(0), True
    return t, False


class Launch:
    """Binds a plan's tensors to kernel arguments for one call."""

    def __init__(
        self,
        st: Structure,
        tensors: list[torch.Tensor],
        slots: dict,
        pair_slots: list,
        batch: int,
        max_q: int,
        max_k: int,
    ):
        self.st, self.tensors, self.slots, self.pair_slots = st, tensors, slots, pair_slots
        self.batch, self.max_q, self.max_k = batch, max_q, max_k

    def _get(self, name):
        i = self.slots.get(name)
        return None if i is None else self.tensors[i]

    def term_args(self, grads: dict | None) -> list:
        st = self.st
        out: list = []
        vb = st.varlen
        if st.has_key_mask:
            out += _t_args(self._get("key_mask"), 2)
        if st.varlen:
            out += [self.cu_q, self.cu_k]
        if st.edge_slots:
            out += _t_args(self._get("edge_index"), 3, vb) + _t_args(
                self._get("edge_values"), 4, vb
            )
            if grads is not None:
                out += _t_args(grads["edge_values"], 4, vb)
        if st.edge_mask_slots:
            out += _t_args(self._get("emask_index"), 3, vb)
        if st.node_rank:
            out += _t_args(self._get("node_u"), 4, vb) + _t_args(self._get("node_w"), 4, vb)
            if grads is not None:
                out += _t_args(grads["node_u"], 4, vb) + _t_args(grads["node_w"], 4, vb)
        for p, spec in enumerate(st.pairs):
            ps = self.pair_slots[p]
            for i in ps["q"]:
                out += _t_args(self.tensors[i], 3, vb)
            for i in ps["k"]:
                out += _t_args(self.tensors[i], 3, vb)
            if spec.has_table:
                t4 = _table4(spec, self.tensors[ps["table"]])
                out += _table_args(t4, spec)
                if grads is not None:
                    out += _table_args(grads[("table", p)], spec)
                    if st.grad_modes and st.grad_modes[p] == GradMode.PRIVATE:
                        out.append(grads[("priv", p)])
                        if spec.has_affine:
                            out += [grads[("priva", p)], grads[("privb", p)]]
            if spec.has_affine:
                for key in ("a", "b"):
                    a2, shared = _affine2(self.tensors[ps[key]])
                    out += _t_args(a2, 2, shared)
                if grads is not None:
                    for key in ("a", "b"):
                        g2, shared = _affine2(grads[(key, p)])
                        out += _t_args(g2, 2, shared)
            if grads is not None:
                for f, need in enumerate(spec.q_feat_grad):
                    if need:
                        out += _t_args(grads[("q", p, f)], 3, vb)
                for f, need in enumerate(spec.k_feat_grad):
                    if need:
                        out += _t_args(grads[("k", p, f)], 3, vb)
        return out

    def constexpr(self, bm: int, bn: int) -> dict:
        st = self.st
        prec = "ieee" if st.dtype == torch.float32 else "tf32"
        kw = dict(
            BM=bm,
            BN=bn,
            BDQK=_next_pow2(st.dqk),
            BDV=_next_pow2(st.dv),
            BR=_next_pow2(max(st.node_rank, 1)),
            DQK=st.dqk,
            DV=st.dv,
            R=max(st.node_rank, 1),
            PREC=prec,
            SEM="relaxed",
        )
        if st.edge_slots:
            kw.update(KES=st.edge_slots, KESP=_next_pow2(st.edge_slots, 1))
        return kw


def _qkv4(x: torch.Tensor, varlen: bool) -> torch.Tensor:
    """``[B, H, N, D]`` as is; varlen ``[T, H, D]`` viewed as ``[1, H, T, D]``."""
    return x.permute(1, 0, 2).unsqueeze(0) if varlen else x


def forward(
    st: Structure,
    q,
    k,
    v,
    tensors,
    slots,
    pair_slots,
    scale: float,
    cu_q=None,
    cu_k=None,
    max_q: int = 0,
    max_k: int = 0,
):
    if q.device.type == "cpu" and q.dtype == torch.bfloat16 and _interpret():
        raise ValueError(
            "bfloat16 dot products are unreliable in the Triton CPU interpreter. "
            "Use float32 or molflash.reference.attention or attention_varlen on CPU."
        )
    fwd, _, _, _ = generated(st)
    varlen = st.varlen
    q4, k4, v4 = (_qkv4(x, varlen) for x in (q, k, v))
    if varlen:
        batch = cu_q.numel() - 1
        nq, nk = max_q, max_k
        total = q.shape[0]
        o = torch.empty((total, st.heads, st.dv), device=q.device, dtype=q.dtype)
        lse = torch.empty((1, st.heads, total), device=q.device, dtype=torch.float32)
        mx = torch.empty((1, st.heads, total), device=q.device, dtype=torch.float32)
    else:
        batch, _, nq, _ = q.shape
        nk = k.shape[2]
        o = torch.empty((*q.shape[:3], st.dv), device=q.device, dtype=q.dtype)
        lse = torch.empty((batch, st.heads, nq), device=q.device, dtype=torch.float32)
        mx = torch.empty((batch, st.heads, nq), device=q.device, dtype=torch.float32)
    o4 = _qkv4(o, varlen)
    launch = Launch(st, tensors, slots, pair_slots, batch, nq, nk)
    launch.cu_q, launch.cu_k = cu_q, cu_k
    bm, bn, warps, stages = _blocks(st, q.device)
    zb = varlen
    args = (
        _t_args(q4, 4, zb)
        + _t_args(k4, 4, zb)
        + _t_args(v4, 4, zb)
        + _t_args(o4, 4, zb)
        + _t_args(lse, 3, zb)
        + _t_args(mx, 3, zb)
        + launch.term_args(None)
        + [st.heads, nq, nk, scale]
    )
    grid = (triton.cdiv(max(nq, 1), bm), batch * st.heads)
    fwd[grid](*args, **launch.constexpr(bm, bn), num_warps=warps, num_stages=stages)
    return o, lse, mx


if HAS_TRITON:

    @triton.jit
    def _delta_kernel(O, DO, DELTA, ROWS: tl.constexpr, D: tl.constexpr,
                      BM: tl.constexpr, BD: tl.constexpr):
        rows = tl.program_id(0) * BM + tl.arange(0, BM)
        channels = tl.arange(0, BD)
        offsets = rows[:, None] * D + channels[None, :]
        valid = (rows[:, None] < ROWS) & (channels[None, :] < D)
        o = tl.load(O + offsets, mask=valid, other=0.0).to(tl.float32)
        do = tl.load(DO + offsets, mask=valid, other=0.0).to(tl.float32)
        value = tl.sum(o * do, axis=1)
        tl.store(DELTA + rows, value, mask=rows < ROWS)


def _delta(o: torch.Tensor, do: torch.Tensor) -> torch.Tensor:
    if not HAS_TRITON:
        return (do.float() * o.float()).sum(-1)
    delta = torch.empty(o.shape[:-1], device=o.device, dtype=torch.float32)
    rows = delta.numel()
    if rows:
        _delta_kernel[(triton.cdiv(rows, 32),)](
            o, do, delta, rows, o.shape[-1], 32, _next_pow2(o.shape[-1]), num_warps=4,
        )
    return delta


def backward(
    st: Structure,
    q,
    k,
    v,
    o,
    lse,
    do,
    tensors,
    slots,
    pair_slots,
    scale: float,
    cu_q=None,
    cu_k=None,
    max_q: int = 0,
    max_k: int = 0,
    *,
    backward_sampling: BackwardSampling | None = None,
):
    """Gradients for ``q, k, v`` and a dict of gradients for the term tensors."""
    varlen = st.varlen
    do = do.contiguous()
    weights = None
    if backward_sampling is not None and backward_sampling.budget:
        backward_sampling.validate_tile(_blocks(st, q.device, backward=True)[0])
        weights = sample_weights(do, backward_sampling, cu_q if varlen else None, max_q)
        do = weighted_gradient(
            do, weights, backward_sampling.block_size, cu_q if varlen else None, max_q
        )
    delta = _delta(o, do)
    if varlen:
        delta = delta.transpose(0, 1).unsqueeze(0).contiguous()  # [1, H, T]
        batch = cu_q.numel() - 1
        nq, nk = max_q, max_k
    else:
        delta = delta.contiguous()
        batch, _, nq, _ = q.shape
        nk = k.shape[2]
    q4, k4, v4, do4 = (_qkv4(x, varlen) for x in (q, k, v, do))
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    dq4, dk4, dv4 = (_qkv4(x, varlen) for x in (dq, dk, dv))

    def zeros_like32(t):
        return torch.zeros(t.shape, device=t.device, dtype=torch.float32)

    def zeros_strided32(t):
        """A buffer whose strides match ``t``, so the kernel writes it the way it reads it."""
        out = torch.empty_strided(t.shape, t.stride(), device=t.device, dtype=torch.float32)
        return out.zero_()

    grads: dict = {}
    if st.edge_slots:
        grads["edge_values"] = zeros_strided32(tensors[slots["edge_values"]])
    if st.node_rank:
        grads["node_u"] = zeros_like32(tensors[slots["node_u"]])
        grads["node_w"] = zeros_like32(tensors[slots["node_w"]])
    for p, spec in enumerate(st.pairs):
        ps = pair_slots[p]
        if spec.has_table:
            grads[("table", p)] = zeros_like32(_table4(spec, tensors[ps["table"]]))
        if spec.has_affine:
            grads[("a", p)] = zeros_like32(tensors[ps["a"]])
            grads[("b", p)] = zeros_like32(tensors[ps["b"]])
        for f, need in enumerate(spec.q_feat_grad):
            if need:
                grads[("q", p, f)] = zeros_like32(tensors[ps["q"][f]])
        for f, need in enumerate(spec.k_feat_grad):
            if need:
                grads[("k", p, f)] = zeros_like32(tensors[ps["k"][f]])

    # The accumulation mode depends on sizes, so it is chosen here rather than at registration.
    st = dataclasses.replace(st, grad_modes=choose_grad_modes(st, batch, nq))
    _, dkdv_k, dq_k, _ = generated(st)
    bm, bn, warps, stages = _blocks(st, q.device, backward=True)
    mt = triton.cdiv(max(nq, 1), bm)
    programs = mt * batch * st.heads
    for p, spec in enumerate(st.pairs):
        if st.grad_modes and st.grad_modes[p] == GradMode.PRIVATE:
            grads[("priv", p)] = torch.zeros(
                (programs, _entries(spec)), device=q.device, dtype=torch.float32
            )
            if spec.has_affine:
                for key in ("priva", "privb"):
                    grads[(key, p)] = torch.zeros(
                        (programs, spec.num_codes), device=q.device, dtype=torch.float32
                    )

    launch = Launch(st, tensors, slots, pair_slots, batch, nq, nk)
    launch.cu_q, launch.cu_k = cu_q, cu_k
    zb = varlen
    common = (
        _t_args(q4, 4, zb)
        + _t_args(k4, 4, zb)
        + _t_args(v4, 4, zb)
        + _t_args(do4, 4, zb)
        + _t_args(lse, 3, zb)
        + _t_args(delta, 3, zb)
        + _t_args(delta if weights is None else weights, 3)
    )
    term = launch.term_args(grads)
    tail = [st.heads, nq, nk, scale]
    ce = launch.constexpr(bm, bn)
    ce.update(
        SAMPLED=weights is not None,
        SAMPLE_BLOCK=1 if weights is None else backward_sampling.block_size,
    )
    dkdv_k[(triton.cdiv(max(nk, 1), bn), batch * st.heads)](
        *common,
        *_t_args(dk4, 4, zb),
        *_t_args(dv4, 4, zb),
        *term,
        *tail,
        **ce,
        num_warps=warps,
        num_stages=stages,
    )
    dq_k[(mt, batch * st.heads)](
        *common,
        *_t_args(dq4, 4, zb),
        *term,
        *tail,
        mt,
        **ce,
        num_warps=warps,
        num_stages=stages,
    )
    for p, spec in enumerate(st.pairs):
        if ("priv", p) in grads:
            _reduce_private(spec, grads, p, batch, st.heads, mt)
    return dq, dk, dv, grads


def _reduce_private(spec, grads, p, batch, heads, mt):
    """Sums the per-program slices into the table (and affine) gradients."""
    table = grads[("table", p)]  # [Bt, Ht, K', G']
    priv = grads.pop(("priv", p)).view(batch, heads, mt, *table.shape[2:]).sum(2)
    if not spec.per_sample:
        priv = priv.sum(0, keepdim=True)
    if table.shape[1] == 1:
        priv = priv.sum(1, keepdim=True)
    table += priv
    if spec.has_affine:
        for key, name in (("priva", "a"), ("privb", "b")):
            sl = grads.pop((key, p)).view(batch, heads, mt, -1).sum((0, 2))  # [H, K]
            target = grads[(name, p)]
            target += sl if target.dim() == 2 else sl.sum(0)


# -----------------------------------------------------------------------------------------------
# evaluating a pair function densely (reference path for terms without torch_fn)
# -----------------------------------------------------------------------------------------------


@functools.cache
def _materialize_kernel(fn, kind: Kind, q_channels: tuple, k_channels: tuple, q_float: tuple):
    L = _Lines()
    params = []
    for f in range(len(q_channels)):
        params += _tensor_params(f"PQ{f}", 3)
    for f in range(len(k_channels)):
        params += _tensor_params(f"PK{f}", 3)
    params += _tensor_params("C", 3) + _tensor_params("TT", 3) + ["NQ", "NK"]
    params += ["BM: tl.constexpr", "BN: tl.constexpr"]
    L("@triton.jit", "def _mat(\n    " + ",\n    ".join(params) + ",\n):")
    with L.indent():
        L(
            "off_b = tl.program_id(2)",
            "offs_m = tl.program_id(0) * BM + tl.arange(0, BM)",
            "offs_n = tl.program_id(1) * BN + tl.arange(0, BN)",
            "m_valid = offs_m < NQ",
            "n_valid = offs_n < NK",
        )
        names = []
        for side, chans, toks, valid, exp in (
            ("Q", q_channels, "offs_m", "m_valid", "[:, None]"),
            ("K", k_channels, "offs_n", "n_valid", "[None, :]"),
        ):
            i = 0
            for f, c_n in enumerate(chans):
                other = "0.0" if q_float[f] else "0"
                for c in range(c_n):
                    n = f"P{side}{f}"
                    L(
                        f"x{side}{i} = tl.load({n} + off_b * {n}_s0 + {toks} * {n}_s1 + {c} * {n}_s2, "
                        f"mask={valid}, other={other}){exp}"
                    )
                    names.append(f"x{side}{i}")
                    i += 1
        call = f"FN({', '.join(names)})"
        L(
            "mask = m_valid[:, None] & n_valid[None, :]",
            "ptr = off_b * C_s0 + offs_m[:, None] * C_s1 + offs_n[None, :] * C_s2",
            "tptr = off_b * TT_s0 + offs_m[:, None] * TT_s1 + offs_n[None, :] * TT_s2",
        )
        if kind is Kind.CODE_VALUE:
            L(
                f"c, t = {call}",
                "tl.store(C + ptr, tl.broadcast_to(c, (BM, BN)).to(tl.int32), mask=mask)",
                "tl.store(TT + tptr, tl.broadcast_to(t, (BM, BN)).to(tl.float32), mask=mask)",
            )
        elif kind is Kind.VALUE:
            L(
                f"t = {call}",
                "tl.store(TT + tptr, tl.broadcast_to(t, (BM, BN)).to(tl.float32), mask=mask)",
            )
        else:
            L(
                f"c = {call}",
                "tl.store(C + ptr, tl.broadcast_to(c, (BM, BN)).to(tl.int32), mask=mask)",
            )
    return _compile(L.text(), "_mat", {"triton": triton, "tl": tl, "FN": fn.jit})


def materialize_pair_fn(term, nq: int, nk: int):
    """``fn`` evaluated on every pair: ``(code, t)``, ``t`` or ``code``/``keep`` as ``[B, Nq, Nk]``."""
    from .plan import _feature_layout

    qv, qc, qf = _feature_layout(term.q_features, False)
    kv, kc, _ = _feature_layout(term.k_features, False)
    b = qv[0].shape[0]
    dev = qv[0].device
    code = torch.zeros((b, nq, nk), dtype=torch.int32, device=dev)
    t = torch.zeros((b, nq, nk), dtype=torch.float32, device=dev)
    kern = _materialize_kernel(term.fn, term.kind, qc, kc, qf)
    args = []
    for x in qv + kv:
        args += _t_args(x, 3)
    args += _t_args(code, 3) + _t_args(t, 3) + [nq, nk]
    grid = (triton.cdiv(nq, 16), triton.cdiv(nk, 16), b)
    kern[grid](*args, BM=16, BN=16)
    if term.kind is Kind.CODE_VALUE:
        return code, t
    if term.kind is Kind.VALUE:
        return t
    return code
