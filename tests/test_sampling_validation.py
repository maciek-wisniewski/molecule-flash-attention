from __future__ import annotations

import molflash as mf
import pytest
import torch

sampling = pytest.importorskip("molflash.sampling")


def test_water_filling_probabilities_match_an_independent_small_solution() -> None:
    scores = torch.tensor([[4.0, 2.0, 1.0, 0.5]])
    expected = torch.tensor([[1.0, 4.0 / 7.0, 2.0 / 7.0, 1.0 / 7.0]])
    actual = sampling.block_probabilities(scores, budget=0.1)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual.sum(-1), torch.tensor([2.0]))
    for scale in (1e-10, 1e10):
        scaled = sampling.block_probabilities(scores * scale, budget=0.1)
        torch.testing.assert_close(scaled, expected, atol=1e-6, rtol=1e-6)
    exact = sampling.block_probabilities(scores, budget=0.0)
    torch.testing.assert_close(exact, torch.ones_like(exact), atol=0, rtol=0)


def test_sampling_weights_have_the_expected_mean_with_variance_aware_bounds() -> None:
    trials, block_size = 8192, 4
    scores = torch.tensor([4.0, 2.0, 1.0, 0.5])
    output_gradient = (scores / block_size**0.5).repeat_interleave(block_size)
    output_gradient = output_gradient.reshape(1, 1, -1, 1).expand(trials, 1, -1, -1)
    config = mf.BackwardSampling(budget=0.1, block_size=block_size, seed=619)
    weights = sampling.sample_weights(output_gradient, config)[:, 0]
    probabilities = torch.tensor([1.0, 4.0 / 7.0, 2.0 / 7.0, 1.0 / 7.0])
    assert weights.shape == (trials, 4)
    assert torch.isfinite(weights).all()
    assert ((weights == 0) | torch.isclose(weights, probabilities.reciprocal())).all()
    standard_error = ((probabilities.reciprocal() - 1.0) / trials).sqrt()
    assert ((weights.mean(0) - 1.0).abs() <= 6 * standard_error + 1e-6).all()
    contributions = torch.tensor([4.0, -2.0, 3.0, 1.0])
    estimates = weights @ contributions
    expected = contributions.sum()
    error = ((probabilities.reciprocal() - 1.0) * contributions.square()).sum().sqrt()
    error = error / trials**0.5
    assert (estimates.mean() - expected).abs() <= 6 * error + 1e-6


def test_explicit_seed_is_reproducible_without_mutating_the_default_rng() -> None:
    output_gradient = torch.arange(1, 65, dtype=torch.float32).reshape(1, 2, 16, 2)
    config = mf.BackwardSampling(budget=0.2, block_size=4, seed=620)
    state = torch.get_rng_state().clone()
    first = sampling.sample_weights(output_gradient, config)
    second = sampling.sample_weights(output_gradient, config)
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    torch.testing.assert_close(torch.get_rng_state(), state, atol=0, rtol=0)


def test_zero_output_gradient_produces_finite_zero_weights() -> None:
    output_gradient = torch.zeros(2, 3, 17, 4)
    config = mf.BackwardSampling(budget=0.2, block_size=8, seed=621)
    weights = sampling.sample_weights(output_gradient, config)
    assert weights.shape == (2, 3, 3)
    assert torch.count_nonzero(weights) == 0
    weighted = sampling.weighted_gradient(output_gradient, weights, config.block_size)
    assert weighted.shape == output_gradient.shape
    assert torch.count_nonzero(weighted) == 0


def test_varlen_weights_restart_at_sequence_boundaries_and_ignore_empty_segments() -> None:
    lengths = [0, 3, 0, 6, 1]
    cu_q = torch.tensor([0, 0, 3, 3, 9, 10], dtype=torch.int32)
    output_gradient = torch.arange(1, 41, dtype=torch.float32).reshape(10, 2, 2)
    config = mf.BackwardSampling(budget=0.2, block_size=4, seed=622)
    weights = sampling.sample_weights(output_gradient, config, cu_q=cu_q, max_q=max(lengths))
    assert weights.shape == (len(lengths), 2, 2)
    assert torch.count_nonzero(weights[[0, 2]]) == 0
    assert torch.count_nonzero(weights[[1, 4], :, 1]) == 0
    weighted = sampling.weighted_gradient(
        output_gradient, weights, config.block_size, cu_q=cu_q, max_q=max(lengths),
    )
    expected = torch.empty_like(output_gradient)
    for row, length in enumerate(lengths):
        start = int(cu_q[row])
        for local in range(length):
            expected[start + local] = (
                output_gradient[start + local] * weights[row, :, local // config.block_size, None]
            )
    torch.testing.assert_close(weighted, expected, atol=0, rtol=0)


def test_all_input_sampled_gradients_with_checkpoint_on_cpu() -> None:
    from test_sampling_gpu_validation import verify_sampled_all_input_gradients

    verify_sampled_all_input_gradients(torch.float32, True, "cpu")
