import molflash as mf
import pytest
import torch
from _util import label_distance, label_distance_torch, qkv
from molflash.sampling import sample_weights, weighted_gradient

pytestmark = pytest.mark.gpu
LABEL_DISTANCE = mf.pair_function(label_distance)


def test_sampled_backward_compiles_as_a_full_graph():
    torch.manual_seed(67)
    q, k, v = qkv(1, 2, 259, 259, 128, dtype=torch.bfloat16, device="cuda")
    labels = torch.arange(259, device="cuda")[None] % 5
    table = torch.randn(2, 5, device="cuda", requires_grad=True)
    sampling = mf.BackwardSampling(0.3, block_size=128, seed=8)

    def operation(q, k, v, table):
        bias = mf.PairBias(LABEL_DISTANCE, labels, table, torch_fn=label_distance_torch)
        return mf.attention(q, k, v, bias=bias, backward_sampling=sampling)

    compiled = torch.compile(operation, fullgraph=True)
    output = operation(q, k, v, table)
    actual = compiled(q, k, v, table)
    torch.testing.assert_close(actual, output, atol=0, rtol=0)
    do = torch.randn_like(output)
    expected = torch.autograd.grad(output, [q, k, v, table], do)
    got = torch.autograd.grad(actual, [q, k, v, table], do)
    for a, b in zip(got, expected, strict=True):
        torch.testing.assert_close(a, b, atol=1e-3, rtol=1e-2)


def test_cuda_rng_is_untouched_until_a_positive_budget_backward():
    torch.manual_seed(17)
    inputs = qkv(1, 2, 259, 259, 32, device="cuda")
    state = torch.cuda.get_rng_state().clone()
    zero = mf.attention(*inputs, backward_sampling=mf.BackwardSampling(0.0))
    torch.autograd.grad(zero, inputs, torch.ones_like(zero))
    assert torch.equal(torch.cuda.get_rng_state(), state)
    out = mf.attention(*inputs, backward_sampling=mf.BackwardSampling(0.2))
    assert torch.equal(torch.cuda.get_rng_state(), state)
    torch.autograd.grad(out, inputs, torch.ones_like(out))
    assert not torch.equal(torch.cuda.get_rng_state(), state)


def test_packed_sampling_handles_empty_and_partial_sequences():
    torch.manual_seed(78)
    q, k, v = qkv(1, 2, 261, 130, 32, device="cuda", varlen=True)
    cu_q = torch.tensor([0, 73, 73, 261], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, 0, 29, 130], dtype=torch.int32, device="cuda")
    qlabels = torch.arange(261, device="cuda") % 5
    klabels = torch.arange(130, device="cuda") % 5
    table = torch.randn(2, 5, device="cuda", requires_grad=True)
    bias = mf.PairBias(
        LABEL_DISTANCE, table=table, q_features=qlabels, k_features=klabels,
        torch_fn=label_distance_torch,
    )
    sampling = mf.BackwardSampling(0.3, block_size=128, seed=17)
    output = mf.attention_varlen(
        q, k, v, cu_q, cu_k, 188, 101, bias=bias, backward_sampling=sampling
    )
    reference = mf.reference.attention_varlen(q, k, v, cu_q, cu_k, bias=bias)
    do = torch.randn_like(output)
    weights = sample_weights(do, sampling, cu_q, 188)
    weighted = weighted_gradient(do, weights, sampling.block_size, cu_q, 188)
    got = torch.autograd.grad(output, [q, k, v, table], do, retain_graph=True)
    expected = torch.autograd.grad(reference, [q, k, v, table], weighted)
    torch.testing.assert_close(output, reference, atol=2e-4, rtol=2e-4)
    for a, b in zip(got, expected, strict=True):
        torch.testing.assert_close(a, b, atol=2e-4, rtol=2e-4)
