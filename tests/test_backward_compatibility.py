from collections.abc import Callable

import molflash as mf
import pytest
import torch
from _util import CUDA, DTYPES, TOL, check, label_distance, label_distance_torch, qkv
from molflash.varlen import cu_seqlens
from torch.utils.checkpoint import checkpoint

LABEL_DISTANCE = mf.pair_function(label_distance)


def _strided(shape: tuple[int, ...], device: str, dtype: torch.dtype) -> torch.Tensor:
    return torch.randn(*shape[:-1], shape[-1] * 2, device=device, dtype=dtype)[
        ..., ::2
    ].requires_grad_(True)


def _problem(
    dimension: int, device: str, dtype: torch.dtype,
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(273)
    rows, heads, queries, keys = (2, 4, 65, 73) if CUDA else (2, 2, 9, 13)
    q = _strided((rows, heads, queries, dimension), device, dtype)
    k = _strided((rows, heads, keys, dimension), device, dtype)
    v = _strided((rows, heads, keys, dimension), device, dtype)
    table = (0.1 * torch.randn(257, heads, device=device)).T.requires_grad_(True)
    scale = torch.ones(heads, device=device, requires_grad=True)
    gate = torch.ones(heads, device=device, requires_grad=True)
    query_labels = torch.randint(0, 257, (rows, queries), device=device)
    key_labels = torch.randint(0, 257, (rows, keys), device=device)
    query_labels[:, 0], key_labels[:, 0] = 256, 0
    mask = torch.ones(rows, keys, dtype=torch.bool, device=device)
    mask[0, -7:], mask[1] = False, False
    return [q, k, v, table, scale, gate], query_labels, key_labels, mask


def _chain(
    query_labels: torch.Tensor, key_labels: torch.Tensor, mask: torch.Tensor,
    *, reference: bool = False, recompute: bool = False,
) -> Callable[..., torch.Tensor]:
    def invoke(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, table: torch.Tensor,
        scale: torch.Tensor, gate: torch.Tensor,
    ) -> torch.Tensor:
        query = q * scale.to(q.dtype)[None, :, None, None]
        operation = mf.reference.attention if reference else mf.attention
        if reference:
            query, k, v = query.float(), k.float(), v.float()
        result = operation(
            query, k, v,
            bias=mf.PairBias(
                LABEL_DISTANCE, q_features=query_labels, k_features=key_labels,
                table=table * gate[:, None], torch_fn=label_distance_torch,
            ),
            key_mask=mask,
        )
        return result.to(q.dtype)

    def run(*inputs: torch.Tensor) -> torch.Tensor:
        first = checkpoint(invoke, *inputs, use_reentrant=False) if recompute else invoke(*inputs)
        args = (first, *inputs[1:])
        return checkpoint(invoke, *args, use_reentrant=False) if recompute else invoke(*args)

    return run


def _relative_close(actual: torch.Tensor, expected: torch.Tensor, tolerance: float) -> None:
    actual, expected = actual.float(), expected.float()
    assert torch.isfinite(actual).all()
    relative = (actual - expected).abs().max() / expected.abs().max().clamp(min=0.05)
    assert relative.item() <= tolerance


def test_shared_table_and_outer_scalars_accumulate_through_nonreentrant_checkpoint(
    device: str,
) -> None:
    dimension, dtype = (128, torch.bfloat16) if CUDA else (16, torch.float32)
    inputs, query_labels, key_labels, mask = _problem(dimension, device, dtype)
    upstream = torch.randn_like(inputs[0])
    results = []
    for reference, recompute in ((True, False), (False, False), (False, True)):
        operation = _chain(query_labels, key_labels, mask, reference=reference, recompute=recompute)
        output = operation(*inputs)
        gradients = torch.autograd.grad(output, inputs, upstream)
        results.append((output, gradients))
        assert torch.count_nonzero(output[1]) == 0
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert all(gradient.abs().sum() > 0 for gradient in gradients)
    expected_output, expected_gradients = results[0]
    for output, gradients in results[1:]:
        _relative_close(output, expected_output, TOL[dtype])
        for actual, expected in zip(gradients, expected_gradients, strict=True):
            _relative_close(actual, expected, TOL[dtype])
    torch.testing.assert_close(results[1][0], results[2][0], atol=0, rtol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("dimension", [128, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_strided_cross_attention_large_table_at_supported_head_widths(
    dimension: int, dtype: torch.dtype, device: str,
) -> None:
    inputs, query_labels, key_labels, mask = _problem(dimension, device, dtype)
    q, k, v, table, _, _ = inputs
    check(
        q, k, v, [table], key_mask=mask,
        bias=mf.PairBias(
            LABEL_DISTANCE, q_features=query_labels, k_features=key_labels,
            table=table, torch_fn=label_distance_torch,
        ),
    )


@pytest.mark.gpu
def test_compiled_head128_chain_preserves_shared_table_and_scalar_gradients() -> None:
    inputs, query_labels, key_labels, mask = _problem(128, "cuda", torch.bfloat16)
    operation = _chain(query_labels, key_labels, mask)
    compiled = torch.compile(operation, fullgraph=True)
    expected = operation(*inputs)
    actual = compiled(*inputs)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    upstream = torch.randn_like(expected)
    actual_gradients = torch.autograd.grad(actual, inputs, upstream)
    expected_gradients = torch.autograd.grad(expected, inputs, upstream)
    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        _relative_close(actual, expected, TOL[torch.bfloat16])


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", ["strided", "shared_head"])
def test_edge_slots_keep_duplicate_and_masked_contributions(
    dtype: torch.dtype, layout: str, device: str,
) -> None:
    torch.manual_seed(581)
    rows, heads, queries, keys = (2, 4, 35, 67) if CUDA else (2, 2, 5, 21)
    q, k, v = qkv(rows, heads, queries, keys, 16, dtype=dtype, device=device)
    index = torch.tensor([0, 0, 1, 32, 63, 64, -1, -9, keys + 8], device=device)
    index = index.expand(rows, queries, -1).clone()
    index[:, :, 2] = torch.arange(queries, device=device)
    shape = (rows, queries, index.shape[-1], 1 if layout == "shared_head" else heads)
    base = _strided(shape, device, torch.float32)
    values = base.expand(rows, queries, index.shape[-1], heads)
    query_labels = torch.arange(queries, device=device).expand(rows, -1) % 3
    key_labels = torch.arange(keys, device=device).expand(rows, -1) % 3
    mask = torch.ones(rows, keys, dtype=torch.bool, device=device)
    mask[0, -3:], mask[1] = False, False
    check(
        q, k, v, [base], key_mask=mask,
        bias=[
            mf.EdgeBias(index, values),
            mf.EdgeBias(index[..., :3].flip(-1), values[..., :3, :] * 0.3),
        ],
        mask=mf.PairMask(
            LABEL_DISTANCE, q_features=query_labels, k_features=key_labels,
            torch_fn=label_distance_torch,
        ),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_varlen_edge_slots_use_sequence_local_indices_and_packed_offsets(
    dtype: torch.dtype, device: str,
) -> None:
    torch.manual_seed(582)
    query_lengths = [0, 3, 35, 1] if CUDA else [0, 3, 5, 1]
    key_lengths = [0, 67, 2, 0] if CUDA else [0, 21, 2, 0]
    heads = 2
    q, k, v = qkv(
        1, heads, sum(query_lengths), sum(key_lengths), 16,
        dtype=dtype, device=device, varlen=True,
    )
    cu_q = cu_seqlens(torch.tensor(query_lengths, device=device))
    cu_k = cu_seqlens(torch.tensor(key_lengths, device=device))
    indices = []
    for count, keys in zip(query_lengths, key_lengths, strict=True):
        row = torch.tensor([0, 0, keys - 1, 32, -1, keys + 3], device=device)
        indices.append(row.expand(count, -1))
    index = torch.cat(indices)
    base = _strided((sum(query_lengths), index.shape[-1], 1), device, torch.float32)
    values = base.expand(sum(query_lengths), index.shape[-1], heads)
    check(
        q, k, v, [base], varlen=(cu_q, cu_k),
        bias=[mf.EdgeBias(index, values), mf.EdgeBias(index[:, :2], values[:, :2] * 0.25)],
    )


def test_dense_reference_edge_mask_allows_zero_keys() -> None:
    q, k, v = qkv(1, 2, 3, 0, 16)
    index = torch.tensor([[[0, -1], [0, 0], [13, -2]]])
    output = mf.reference.attention(q, k, v, mask=mf.EdgeMask(index))
    assert torch.count_nonzero(output) == 0
    gradients = torch.autograd.grad(output.sum(), (q, k, v))
    assert all(torch.count_nonzero(gradient) == 0 for gradient in gradients)
