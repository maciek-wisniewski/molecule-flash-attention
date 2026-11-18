import molflash as mf
import pytest
import torch


@pytest.mark.parametrize("sampling", [None, mf.BackwardSampling(0.2)])
def test_empty_batch_preserves_empty_gradients(sampling, device):
    tensors = [torch.empty(0, 2, 7, 16, device=device, requires_grad=True) for _ in range(3)]
    output = mf.attention(*tensors, backward_sampling=sampling)
    output.sum().backward()
    assert output.shape == tensors[0].shape
    for tensor in tensors:
        assert tensor.grad is not None
        assert tensor.grad.shape == tensor.shape
