import molflash as mf
import pytest
import torch
from molflash import kernels, ops
from molflash.plan import Structure, lookup


@pytest.mark.parametrize("packed", [False, True], ids=["padded", "packed"])
@pytest.mark.parametrize(
    "dtype,dimension,query_tile",
    [(torch.float32, 64, 32), (torch.float16, 64, 64), (torch.bfloat16, 128, 32)],
)
def test_sampling_validates_registered_metadata_without_the_interpreter_shortcut(
    monkeypatch, device, packed, dtype, dimension, query_tile
):
    monkeypatch.setattr(kernels, "_interpret", lambda: False)
    calls = []

    def forward(q, k, v, tensors, key, scale, cu_q, cu_k, max_q, max_k,
                sampling_budget, sampling_block, sampling_seed):
        structure = lookup(key)[0]
        assert isinstance(structure, Structure)
        assert structure.varlen is packed
        assert structure.dtype == dtype
        assert structure.dqk == dimension
        assert sampling_block == query_tile
        calls.append(key)
        output = q.new_zeros((*q.shape[:-1], v.shape[-1]))
        shape = (1, q.shape[1], q.shape[0]) if packed else q.shape[:-1]
        scores = q.new_zeros(shape, dtype=torch.float32)
        return output, scores, scores

    monkeypatch.setattr(ops, "_fwd", forward)
    shape = (5, 2, dimension) if packed else (1, 2, 5, dimension)
    inputs = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(3)]

    def run(block_size):
        sampling = mf.BackwardSampling(0.1, block_size=block_size)
        if packed:
            cumulative = torch.tensor([0, 2, 5], dtype=torch.int32, device=device)
            return mf.attention_varlen(
                *inputs, cumulative, cumulative, 3, 3, backward_sampling=sampling
            )
        return mf.attention(*inputs, backward_sampling=sampling)

    assert run(query_tile).shape == inputs[0].shape
    assert len(calls) == 1
    with pytest.raises(ValueError, match=rf"query tile \({query_tile}\)"):
        run(query_tile // 2)
    assert len(calls) == 1


def test_sampled_fullgraph_uses_the_normal_selector_with_cpu_kernel_interpretation(
    monkeypatch, device
):
    if device != "cpu":
        pytest.skip("this exercises metadata and interpreter math without CUDA compilation")
    monkeypatch.setattr(kernels, "_interpret", lambda: False)
    sampling = mf.BackwardSampling(0.3, block_size=32, seed=19)
    inputs = [torch.randn(1, 2, 9, 8, requires_grad=True) for _ in range(3)]

    def operation(q, k, v):
        return mf.attention(q, k, v, backward_sampling=sampling)

    compiled = torch.compile(operation, backend="eager", fullgraph=True)
    expected = operation(*inputs)
    actual = compiled(*inputs)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    gradient = torch.ones_like(expected)
    want = torch.autograd.grad(expected, inputs, gradient)
    got = torch.autograd.grad(actual, inputs, gradient)
    for a, b in zip(got, want, strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
