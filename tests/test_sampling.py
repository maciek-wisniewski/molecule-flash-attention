import dataclasses

import molflash as mf
import pytest
import torch
import triton
from _util import label_distance, label_distance_torch, qkv
from molflash.sampling import (
    _energies,
    _weight_gradient,
    block_probabilities,
    sample_weights,
    weighted_gradient,
)


@pytest.mark.parametrize("budget", [-1, 1, float("nan"), float("inf")])
def test_sampling_budget_requires_a_finite_fraction(budget):
    with pytest.raises(ValueError, match="budget"):
        mf.BackwardSampling(budget)


@pytest.mark.parametrize("block", [0, -16, 3, 48])
def test_sampling_block_is_a_positive_power_of_two(block):
    with pytest.raises(ValueError, match="block_size"):
        mf.BackwardSampling(0.1, block_size=block)


def test_sampling_configuration_is_immutable_and_seed_is_validated():
    config = mf.BackwardSampling(0.1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.budget = 0.2
    with pytest.raises(ValueError, match="seed"):
        mf.BackwardSampling(0.1, seed=-1)
    with pytest.raises(ValueError, match="seed"):
        mf.BackwardSampling(0.1, seed=2**63)
    with pytest.raises(TypeError, match="budget"):
        mf.BackwardSampling(True)
    with pytest.raises(TypeError, match="block_size"):
        mf.BackwardSampling(0.1, block_size=True)
    with pytest.raises(TypeError, match="seed"):
        mf.BackwardSampling(0.1, seed=True)


def test_probability_waterfilling_has_the_requested_expected_count():
    scores = torch.tensor([[1., 1., 1., 1.], [100., 1., 1., 1.]])
    uniform = block_probabilities(scores[:1], 0.5)
    saturated = block_probabilities(scores[1:], 0.0002)
    torch.testing.assert_close(uniform, torch.full_like(uniform, 0.5))
    torch.testing.assert_close(saturated, torch.tensor([[1., 1 / 3, 1 / 3, 1 / 3]]))
    torch.testing.assert_close(saturated.sum(-1), torch.tensor([2.]))


def test_zero_scores_have_zero_probability_and_tiny_positive_scores_survive():
    scores = torch.tensor([[0., 0., 1., 2.], [0., 0., 0., 0.]])
    torch.testing.assert_close(
        block_probabilities(scores, 0.01), torch.tensor([[0., 0., 1., 1.], [0., 0., 0., 0.]])
    )
    tiny = block_probabilities(torch.tensor([[1e-38, 1., 2., 3.]]), 0.1)
    assert 0 < tiny[0, 0] < 1
    assert torch.isfinite(tiny.reciprocal()).all()


@pytest.mark.parametrize("magnitude", [1e-30, 1e30])
def test_finite_extreme_gradients_keep_the_sampling_distribution(magnitude):
    torch.manual_seed(51)
    do = torch.randn(2, 3, 37, 8)
    config = mf.BackwardSampling(0.4, block_size=16, seed=7)
    expected = sample_weights(do, config)
    actual = sample_weights(do * magnitude, config)
    torch.testing.assert_close(actual, expected)
    scaled = weighted_gradient(do * magnitude, actual, config.block_size)
    assert torch.isfinite(actual).all()
    assert torch.isfinite(scaled).all()
    assert torch.count_nonzero(actual) > 0


def test_sampling_weights_respect_varlen_boundaries_and_empty_sequences():
    gradient = torch.arange(30, dtype=torch.float32).reshape(5, 2, 3)
    cu_q = torch.tensor([0, 2, 2, 5])
    config = mf.BackwardSampling(0.5, block_size=2, seed=4)
    weights = sample_weights(gradient, config, cu_q, 3)
    assert weights.shape == (3, 2, 2)
    assert torch.count_nonzero(weights[1]) == 0
    scaled = weighted_gradient(gradient, weights, 2, cu_q, 3)
    for b in range(3):
        start, end = int(cu_q[b]), int(cu_q[b + 1])
        rows = torch.arange(end - start) // 2
        expected = gradient[start:end] * weights[b, :, rows].T[..., None]
        torch.testing.assert_close(scaled[start:end], expected)


def test_default_and_zero_budget_keep_values_gradients_and_rng(device):
    torch.manual_seed(3)
    inputs = qkv(1, 2, 19, 19, 8, device=device)
    gradient = torch.randn_like(inputs[0])
    exact = mf.attention(*inputs)
    want = torch.autograd.grad(exact, inputs, gradient)
    before = torch.random.get_rng_state().clone()
    zero = mf.attention(*inputs, backward_sampling=mf.BackwardSampling(0, block_size=1))
    got = torch.autograd.grad(zero, inputs, gradient)
    torch.testing.assert_close(zero, exact, atol=0, rtol=0)
    for actual, expected in zip(got, want, strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert torch.equal(torch.random.get_rng_state(), before)


def test_sampled_gradient_matches_dense_oracle_with_the_same_weighted_do(device):
    torch.manual_seed(71)
    inputs = qkv(1, 2, 37, 37, 8, device=device)
    labels = torch.arange(37, device=device)[None] % 4
    table = torch.randn(2, 4, device=device, requires_grad=True)
    term = mf.PairBias(label_distance, labels, table, torch_fn=label_distance_torch)
    config = mf.BackwardSampling(0.4, block_size=128 if device == "cuda" else 16, seed=6)
    out = mf.attention(*inputs, bias=term, backward_sampling=config)
    reference = mf.reference.attention(*inputs, bias=term)
    do = torch.randn_like(out)
    do[..., 16:, :] *= 0.01
    weights = sample_weights(do, config)
    weighted = weighted_gradient(do, weights, config.block_size)
    got = torch.autograd.grad(out, [*inputs, table], do, retain_graph=True)
    want = torch.autograd.grad(reference, [*inputs, table], weighted)
    torch.testing.assert_close(out, reference, atol=2e-4, rtol=2e-4)
    for actual, expected in zip(got, want, strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)


def test_sampling_groups_must_cover_whole_backward_query_tiles(device):
    inputs = qkv(1, 2, 19, 19, 8, device=device)
    with pytest.raises(ValueError, match="query tile"):
        mf.attention(*inputs, backward_sampling=mf.BackwardSampling(0.1, block_size=1))


def test_sampling_uses_the_actual_finite_grid_probability(monkeypatch):
    grid = 2**24
    scores = torch.tensor([[1e-30, 1.0, 2.0, 3.0]])
    probabilities = block_probabilities(scores, 0.1)
    assert probabilities[0, 0] == 1 / grid
    assert torch.equal(probabilities * grid, (probabilities * grid).round())
    assert 2 <= probabilities.sum() <= 2 + scores.numel() / grid
    do = scores.reshape(1, 1, 4, 1)
    config = mf.BackwardSampling(0.1, block_size=1, seed=8)

    def draw(value):
        def integer_draws(high, size, *, device, generator):
            assert high == grid
            return torch.full(size, value, device=device, dtype=torch.int64)
        return integer_draws

    monkeypatch.setattr(torch, "randint", draw(0))
    kept = sample_weights(do, config)
    assert kept[0, 0, 0] == grid
    assert torch.isfinite(kept).all()
    monkeypatch.setattr(torch, "randint", draw(1))
    discarded = sample_weights(do, config)
    assert discarded[0, 0, 0] == 0


def test_sampler_kernels_match_cpu_helpers_in_the_interpreter(device):
    if device != "cpu":
        pytest.skip("this checks the CPU interpreter against independent torch arithmetic")
    torch.manual_seed(4)
    do = torch.randn(1, 2, 37, 8)
    block_size, groups = 16, 3
    energy = torch.empty(1, 2, groups, dtype=torch.float64)
    _energies[(groups, 2)](
        do, *do.stride(), do, energy, HEADS=2, BLOCKS=groups, LENGTH=37,
        DIM=8, PACKED=False, GROUP=block_size, ROWS=32,
        CHANNELS=triton.next_power_of_2(8), num_warps=4,
    )
    padded = torch.nn.functional.pad(do.double(), (0, 0, 0, 11))
    expected = padded.reshape(1, 2, groups, block_size, 8).square().sum((-1, -2))
    torch.testing.assert_close(energy, expected)
    weights = torch.tensor([[[2.0, 0.0, 1.0], [0.0, 1.5, 3.0]]])
    scaled = torch.empty_like(do)
    _weight_gradient[(triton.cdiv(37, 32), 2)](
        do, *do.stride(), do, weights, scaled, *scaled.stride(),
        HEADS=2, BLOCKS=groups, LENGTH=37, DIM=8, PACKED=False,
        GROUP=block_size, ROWS=32, CHANNELS=triton.next_power_of_2(8), num_warps=4,
    )
    torch.testing.assert_close(scaled, weighted_gradient(do, weights, block_size))


@pytest.mark.parametrize(
    "sampling",
    [None, mf.BackwardSampling(0), mf.BackwardSampling(0.3, block_size=16, seed=27)],
    ids=["default", "zero", "sampled"],
)
def test_cpu_fullgraph_plumbing_preserves_gradients_and_rng(device, sampling):
    if device != "cpu":
        pytest.skip("this checks custom-op plumbing, not CUDA code generation")
    torch.manual_seed(72)
    inputs = qkv(1, 2, 37, 37, 8, device=device)
    do = torch.randn_like(inputs[0])

    def operation(q, k, v):
        return mf.attention(q, k, v, backward_sampling=sampling)

    compiled = torch.compile(operation, backend="eager", fullgraph=True)
    compiled(*inputs)
    state = torch.get_rng_state().clone()
    expected = operation(*inputs)
    actual = compiled(*inputs)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    want = torch.autograd.grad(expected, inputs, do)
    got = torch.autograd.grad(actual, inputs, do)
    for a, b in zip(got, want, strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert torch.equal(torch.get_rng_state(), state)


def test_sampled_packed_attention_matches_the_weighted_dense_oracle(device):
    torch.manual_seed(78)
    q, k, v = qkv(1, 2, 54, 52, 8, device=device, varlen=True)
    cu_q = torch.tensor([0, 0, 19, 19, 54], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, 7, 7, 24, 52], dtype=torch.int32, device=device)
    qlabels = torch.arange(54, device=device) % 5
    klabels = torch.arange(52, device=device) % 5
    table = torch.randn(2, 5, device=device, requires_grad=True)
    bias = mf.PairBias(
        label_distance, table=table, q_features=qlabels, k_features=klabels,
        torch_fn=label_distance_torch,
    )
    sampling = mf.BackwardSampling(0.3, block_size=128 if device == "cuda" else 16, seed=17)
    output = mf.attention_varlen(
        q, k, v, cu_q, cu_k, 35, 28, bias=bias, backward_sampling=sampling
    )
    reference = mf.reference.attention_varlen(q, k, v, cu_q, cu_k, bias=bias)
    do = torch.randn_like(output)
    weights = sample_weights(do, sampling, cu_q, 35)
    weighted = weighted_gradient(do, weights, sampling.block_size, cu_q, 35)
    got = torch.autograd.grad(output, [q, k, v, table], do, retain_graph=True)
    expected = torch.autograd.grad(reference, [q, k, v, table], weighted)
    torch.testing.assert_close(output, reference, atol=2e-4, rtol=2e-4)
    for a, b in zip(got, expected, strict=True):
        torch.testing.assert_close(a, b, atol=2e-4, rtol=2e-4)
