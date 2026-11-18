"""Optional query-block importance sampling for the attention backward."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None


# Every cutoff on this integer grid is represented exactly by float32.
_PROBABILITY_GRID = 2**24


@dataclass(frozen=True)
class BackwardSampling:
    budget: float
    block_size: int = 128
    seed: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.budget, bool) or not isinstance(self.budget, (int, float)):
            raise TypeError("backward sampling budget must be a real number")
        if not math.isfinite(self.budget) or not 0 <= self.budget < 1:
            raise ValueError("backward sampling budget must be finite and in [0, 1)")
        if isinstance(self.block_size, bool) or not isinstance(self.block_size, int):
            raise TypeError("backward sampling block_size must be an integer")
        if self.block_size <= 0 or self.block_size & (self.block_size - 1):
            raise ValueError("backward sampling block_size must be a positive power of two")
        if self.seed is not None:
            if isinstance(self.seed, bool) or not isinstance(self.seed, int):
                raise TypeError("backward sampling seed must be an integer or None")
            if not 0 <= self.seed < 2**63:
                raise ValueError("backward sampling seed must be a nonnegative 63-bit integer")

    def validate_tile(self, query_tile: int) -> None:
        if self.budget and self.block_size % query_tile:
            raise ValueError(
                "backward sampling block_size must be a multiple of the backward query tile "
                f"({query_tile})"
            )


def block_probabilities(scores: torch.Tensor, budget: float) -> torch.Tensor:
    """Keep probabilities from per-block L2 norms; all decisions are per last dimension."""
    if scores.shape[-1] == 0:
        return torch.empty_like(scores, dtype=torch.float32)
    if budget == 0:
        return torch.ones_like(scores, dtype=torch.float32)
    scores64 = scores.double()
    finite = torch.isfinite(scores64).all(dim=-1, keepdim=True)
    positive = scores64 > 0
    safe = torch.where(finite, scores64, torch.ones_like(scores64))
    maximum = safe.amax(dim=-1, keepdim=True)
    normal = safe / maximum.clamp(min=torch.finfo(torch.float64).tiny)
    energy = normal.square().sort(dim=-1).values
    discarded = (energy.cumsum(dim=-1) <= budget * energy.sum(dim=-1, keepdim=True)).sum(
        dim=-1, keepdim=True
    )
    count = scores.shape[-1]
    target = count - discarded
    ordered = normal.sort(dim=-1, descending=True).values
    suffix = ordered.flip(-1).cumsum(-1).flip(-1)
    saturated = torch.arange(count, device=scores.device, dtype=torch.float64)
    candidates = (target - saturated).clamp(min=0) / suffix.clamp(
        min=torch.finfo(torch.float64).tiny
    )
    multiplier = candidates.amax(dim=-1, keepdim=True)
    cutoffs = ((normal * multiplier).clamp(max=1) * _PROBABILITY_GRID).ceil()
    cutoffs = torch.where(positive, cutoffs.clamp(min=1), 0.0)
    probabilities = (cutoffs / _PROBABILITY_GRID).float()
    return torch.where(finite, probabilities, torch.ones_like(probabilities))


def _layout(
    do: torch.Tensor, cu_q: torch.Tensor | None, max_q: int
) -> tuple[torch.Tensor, int, int, int]:
    if cu_q is None:
        batch, heads, length, _ = do.shape
        return do, batch, heads, length
    if max_q < 0:
        raise ValueError("max_q must be nonnegative")
    return do.permute(1, 0, 2).unsqueeze(0), cu_q.numel() - 1, do.shape[1], max_q


if triton is not None:

    @triton.jit
    def _energies(
        DO, DS0, DS1, DS2, DS3, CUQ, OUT,
        HEADS: tl.constexpr, BLOCKS: tl.constexpr, LENGTH: tl.constexpr,
        DIM: tl.constexpr, PACKED: tl.constexpr, GROUP: tl.constexpr,
        ROWS: tl.constexpr, CHANNELS: tl.constexpr,
    ):
        group = tl.program_id(0)
        bh = tl.program_id(1)
        batch, head = bh // HEADS, bh % HEADS
        if PACKED:
            start = tl.load(CUQ + batch)
            length = tl.load(CUQ + batch + 1) - start
        else:
            start, length = 0, LENGTH
        channels = tl.arange(0, CHANNELS)
        total = tl.full((), 0.0, tl.float64)
        for chunk in range(0, GROUP, ROWS):
            rows = group * GROUP + chunk + tl.arange(0, ROWS)
            values = tl.load(
                DO + batch * DS0 + head * DS1 + (start + rows[:, None]) * DS2
                + channels[None, :] * DS3,
                mask=(rows[:, None] < length) & (rows[:, None] < (group + 1) * GROUP)
                & (channels[None, :] < DIM), other=0.0,
            ).to(tl.float64)
            total += tl.sum(tl.sum(values * values, axis=1), axis=0)
        tl.store(OUT + bh * BLOCKS + group, total)

    @triton.jit
    def _weight_gradient(
        DO, DS0, DS1, DS2, DS3, CUQ, WEIGHTS, OUT, OS0, OS1, OS2, OS3,
        HEADS: tl.constexpr, BLOCKS: tl.constexpr, LENGTH: tl.constexpr,
        DIM: tl.constexpr, PACKED: tl.constexpr, GROUP: tl.constexpr,
        ROWS: tl.constexpr, CHANNELS: tl.constexpr,
    ):
        rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
        bh = tl.program_id(1)
        batch, head = bh // HEADS, bh % HEADS
        if PACKED:
            start = tl.load(CUQ + batch)
            length = tl.load(CUQ + batch + 1) - start
        else:
            start, length = 0, LENGTH
        channels = tl.arange(0, CHANNELS)
        weight = tl.load(
            WEIGHTS + bh * BLOCKS + rows // GROUP, mask=rows < length, other=0.0
        )
        valid = (rows[:, None] < length) & (channels[None, :] < DIM)
        values = tl.load(
            DO + batch * DS0 + head * DS1 + (start + rows[:, None]) * DS2
            + channels[None, :] * DS3, mask=valid, other=0.0,
        ).to(tl.float32)
        tl.store(
            OUT + batch * OS0 + head * OS1 + (start + rows[:, None]) * OS2
            + channels[None, :] * OS3, values * weight[:, None], mask=valid,
        )


def sample_weights(
    do: torch.Tensor,
    sampling: BackwardSampling,
    cu_q: torch.Tensor | None = None,
    max_q: int = 0,
) -> torch.Tensor:
    view, batch, heads, length = _layout(do, cu_q, max_q)
    blocks = (length + sampling.block_size - 1) // sampling.block_size
    if sampling.budget == 0:
        return torch.ones(batch, heads, blocks, device=do.device, dtype=torch.float32)
    if blocks == 0 or batch == 0 or heads == 0:
        return torch.empty(batch, heads, blocks, device=do.device, dtype=torch.float32)
    if do.is_cuda:
        energies = torch.empty(batch, heads, blocks, device=do.device, dtype=torch.float64)
        strides = list(view.stride())
        if cu_q is not None:
            strides[0] = 0
        _energies[(blocks, batch * heads)](
            view, *strides, cu_q if cu_q is not None else view, energies,
            HEADS=heads, BLOCKS=blocks, LENGTH=length, DIM=do.shape[-1],
            PACKED=cu_q is not None, GROUP=sampling.block_size, ROWS=32,
            CHANNELS=triton.next_power_of_2(do.shape[-1]), num_warps=4,
        )
        scores = energies.sqrt()
    else:
        sequences = [view[b] for b in range(batch)] if cu_q is None else [
            do[int(cu_q[b]):int(cu_q[b + 1])].transpose(0, 1) for b in range(batch)
        ]
        scores = torch.stack([
            torch.nn.functional.pad(
                sequence.double(), (0, 0, 0, blocks * sampling.block_size - sequence.shape[1])
            ).reshape(heads, blocks, -1).norm(dim=-1)
            for sequence in sequences
        ])
    probabilities = block_probabilities(scores, sampling.budget)
    generator = None
    if sampling.seed is not None:
        generator = torch.Generator(device=do.device).manual_seed(sampling.seed)
    draws = torch.randint(
        _PROBABILITY_GRID, probabilities.shape, device=do.device, generator=generator
    )
    cutoffs = (probabilities * _PROBABILITY_GRID).to(torch.int64)
    inverse = probabilities.clamp(min=1 / _PROBABILITY_GRID).reciprocal()
    return torch.where(draws < cutoffs, inverse, 0.0)


def weighted_gradient(
    do: torch.Tensor,
    weights: torch.Tensor,
    block_size: int,
    cu_q: torch.Tensor | None = None,
    max_q: int = 0,
) -> torch.Tensor:
    view, batch, heads, length = _layout(do, cu_q, max_q)
    if length == 0:
        return torch.empty_like(do)
    blocks = weights.shape[-1]
    if do.is_cuda:
        output = torch.empty_like(do)
        out = output if cu_q is None else output.permute(1, 0, 2).unsqueeze(0)
        source_stride, target_stride = list(view.stride()), list(out.stride())
        if cu_q is not None:
            source_stride[0] = target_stride[0] = 0
        _weight_gradient[(triton.cdiv(length, 32), batch * heads)](
            view, *source_stride, cu_q if cu_q is not None else view,
            weights, out, *target_stride, HEADS=heads, BLOCKS=blocks, LENGTH=length,
            DIM=do.shape[-1], PACKED=cu_q is not None, GROUP=block_size, ROWS=32,
            CHANNELS=triton.next_power_of_2(do.shape[-1]), num_warps=4,
        )
        return output
    if cu_q is None:
        row_weights = weights.repeat_interleave(block_size, dim=-1)[..., :length]
        return (do * row_weights[..., None]).to(do.dtype)
    output = torch.empty_like(do)
    for b in range(batch):
        start, end = int(cu_q[b]), int(cu_q[b + 1])
        row_weights = weights[b].repeat_interleave(block_size, dim=-1)[..., :end - start]
        output[start:end] = (do[start:end] * row_weights.T[..., None]).to(do.dtype)
    return output
