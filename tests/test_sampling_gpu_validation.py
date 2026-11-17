from __future__ import annotations

import molflash as mf
import pytest
import torch
from _util import TOL, label_distance, label_distance_torch
from torch.utils.checkpoint import checkpoint

sampling = pytest.importorskip("molflash.sampling")
pytestmark = pytest.mark.gpu
LABEL_DISTANCE = mf.pair_function(label_distance)


def verify_sampled_all_input_gradients(
    dtype: torch.dtype, recompute: bool, device: str,
) -> None:
    torch.manual_seed(631)
    rows, heads, queries, keys, dimension, block_size = (
        (2, 2, 129, 137, 32, 32) if device == "cuda" else (2, 2, 49, 53, 16, 16)
    )
    q = torch.randn(rows, heads, queries, dimension, device=device, dtype=dtype).requires_grad_()
    k = torch.randn(rows, heads, keys, dimension, device=device, dtype=dtype).requires_grad_()
    v = torch.randn_like(k).requires_grad_()
    position_q = torch.randn(rows, queries, 3, device=device).requires_grad_()
    position_k = torch.randn(rows, keys, 3, device=device).requires_grad_()
    tables = [
        (0.1 * torch.randn(257, heads, device=device)).T.requires_grad_(),
        (0.1 * torch.randn(7, heads, device=device)).T.requires_grad_(),
        (0.1 * torch.randn(5, heads, device=device)).requires_grad_(),
        (0.1 * torch.randn(3, heads, device=device)).requires_grad_(),
        (0.1 * torch.randn(3, heads, device=device)).T.requires_grad_(),
    ]
    logit_scale = torch.ones(heads, device=device, requires_grad=True)
    bias_gate = torch.ones(heads, device=device, requires_grad=True)
    node_q = torch.randn(rows, heads, queries, 3, device=device).requires_grad_()
    node_k = torch.randn(rows, heads, keys, 3, device=device).requires_grad_()
    inputs = [q, k, v, *tables, logit_scale, bias_gate, position_q, position_k, node_q, node_k]
    query_labels = torch.randint(0, 7, (rows, queries), device=device)
    key_labels = torch.randint(0, 7, (rows, keys), device=device)
    edge_index = torch.randint(-1, keys, (rows, queries, 5), device=device)
    edge_type = torch.randint(0, 5, edge_index.shape, device=device)
    hop_type = torch.randint(0, 3, edge_index.shape, device=device)
    key_mask = torch.ones(rows, keys, dtype=torch.bool, device=device)
    key_mask[0, -9:], key_mask[1] = False, False
    config = mf.BackwardSampling(budget=0.2, block_size=block_size, seed=632)

    def invoke(*args: torch.Tensor, reference: bool = False, sampled: bool = False) -> torch.Tensor:
        (query, key, value, distance, relation, bond, hop, pocket, scale, gate, pq, pk, u, w) = args
        query = query * scale.to(query.dtype)[None, :, None, None]
        operation = mf.reference.attention if reference else mf.attention
        if reference:
            query, key, value = query.float(), key.float(), value.float()
        terms = [
            mf.NodeBias(u, w),
            mf.terms.distance_function(
                distance * gate[:, None], q_pos=pq, k_pos=pk, grid=(0.0, 8.0),
            ),
            mf.PairBias(
                LABEL_DISTANCE, q_features=query_labels, k_features=key_labels,
                table=relation * gate[:, None], torch_fn=label_distance_torch,
            ),
            mf.EdgeBias(edge_index, bond[edge_type] * gate),
            mf.EdgeBias(edge_index.flip(-1), hop[hop_type] * gate),
            mf.PairBias(
                LABEL_DISTANCE, q_features=query_labels % 3, k_features=key_labels % 3,
                table=pocket * gate[:, None], torch_fn=label_distance_torch,
            ),
        ]
        kwargs = {"backward_sampling": config} if sampled else {}
        return operation(
            query, key, value, bias=terms, key_mask=key_mask,
            multiplier=mf.PairScale(
                LABEL_DISTANCE, q_features=query_labels % 3, k_features=key_labels % 3,
                table=1.0 + 0.1 * pocket, torch_fn=label_distance_torch,
            ),
            gate=mf.PairGate(
                LABEL_DISTANCE, q_features=query_labels, k_features=key_labels,
                table=1.0 + 0.1 * relation, torch_fn=label_distance_torch,
            ),
            **kwargs,
        )

    def sampled(*args: torch.Tensor) -> torch.Tensor:
        return invoke(*args, sampled=True)

    output = checkpoint(sampled, *inputs, use_reentrant=False) if recompute else sampled(*inputs)
    exact = invoke(*inputs)
    torch.testing.assert_close(output, exact, atol=0, rtol=0)
    upstream = torch.randn_like(output)
    for block, amplitude in enumerate((4.0, 2.0, 1.0, 0.25, 0.0)):
        upstream[:, :, block_size * block:block_size * (block + 1)] *= amplitude
    weights = sampling.sample_weights(upstream, config)
    assert torch.count_nonzero(weights == 0) > 0
    expected_upstream = sampling.weighted_gradient(upstream, weights, config.block_size)
    actual = torch.autograd.grad(output, inputs, upstream, retain_graph=True)
    dense = invoke(*inputs, reference=True)
    expected = torch.autograd.grad(dense, inputs, expected_upstream.float())
    for gradient, target in zip(actual, expected, strict=True):
        gradient, target = gradient.float(), target.float()
        assert torch.isfinite(gradient).all()
        error = (gradient - target).abs().max() / target.abs().max().clamp(min=0.05)
        assert error.item() <= TOL[dtype]
    assert torch.count_nonzero(actual[0][1]) == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("recompute", [False, True])
def test_sampled_backward_matches_dense_weighted_output_gradient_for_every_input(
    dtype: torch.dtype, recompute: bool,
) -> None:
    verify_sampled_all_input_gradients(dtype, recompute, "cuda")


def test_compiled_positive_budget_matches_eager_large_code_table_gradients() -> None:
    torch.manual_seed(640)
    rows, heads, length, dimension = 1, 2, 129, 32
    inputs = [
        torch.randn(rows, heads, length, dimension, device="cuda", dtype=torch.bfloat16)
        .requires_grad_()
        for _ in range(3)
    ]
    table = (0.1 * torch.randn(257, heads, device="cuda")).T.requires_grad_()
    inputs.append(table)
    labels = torch.arange(length, device="cuda").expand(rows, -1) * 2
    config = mf.BackwardSampling(budget=0.2, block_size=32, seed=641)

    def operation(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias_table: torch.Tensor,
    ) -> torch.Tensor:
        return mf.attention(
            q, k, v,
            bias=mf.PairBias(
                LABEL_DISTANCE, labels, bias_table,
                torch_fn=label_distance_torch,
            ),
            backward_sampling=config,
        )

    compiled = torch.compile(operation, fullgraph=True)
    expected = operation(*inputs)
    actual = compiled(*inputs)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    upstream = torch.randn_like(actual)
    upstream[:, :, 96:] *= 0.01
    actual_gradients = torch.autograd.grad(actual, inputs, upstream)
    expected_gradients = torch.autograd.grad(expected, inputs, upstream)
    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        actual, expected = actual.float(), expected.float()
        assert torch.isfinite(actual).all()
        error = (actual - expected).abs().max() / expected.abs().max().clamp(min=0.05)
        assert error.item() <= TOL[torch.bfloat16]
