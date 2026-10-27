"""The largest pre-softmax score of each query row.

A score tile never leaves the kernel, so this is the only way to see one. The online softmax
already tracks the row maximum for stability, so the kernel stores what it had rather than
computing anything new -- and these tests are what say the two agree.
"""

import molflash as mf
import pytest
import torch
from _util import CUDA, DTYPES, qkv
from molflash import terms

N = 203 if CUDA else 37
B, H = 2, 3


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _both(q, k, v, **kwargs):
    got, got_max = mf.attention(q, k, v, return_max_score=True, **kwargs)
    want, want_max = mf.reference.attention(
        q.detach().float(), k.detach().float(), v.detach().float(),
        return_max_score=True, **kwargs,
    )
    return got, got_max, want, want_max


def test_the_shape_is_one_float_per_query_row(device):
    """`[B, H, Nq]`, not `[B, H, Nq, Nk]`. That is the point: it costs a row, not a pair."""
    q, k, v = qkv(B, H, N, N, 32, 32, torch.float32, device)
    _, max_score = mf.attention(q, k, v, return_max_score=True)
    assert max_score.shape == (B, H, N)
    assert max_score.dtype == torch.float32


@pytest.mark.parametrize("dtype", DTYPES)
def test_it_matches_the_dense_reference(device, dtype):
    q, k, v = qkv(B, H, N, N, 32, 32, dtype, device)
    _, got, _, want = _both(q, k, v)
    torch.testing.assert_close(got, want, atol=2e-2, rtol=2e-2)


def test_a_bias_moves_it_and_the_two_still_agree(device):
    """The bias is the half of the logit that QK-norm does not bound, which is the reason to
    want this number at all."""
    q, k, v = qkv(B, H, N, N, 32, 32, torch.float32, device)
    pos = torch.randn(B, N, 3, device=device) * 4
    table = torch.full((H, 64), 7.5, device=device)
    bias = terms.distance_buckets(table, pos, max_distance=20.0)

    _, plain = mf.attention(q, k, v, return_max_score=True)
    _, biased, _, want = _both(q, k, v, bias=bias)

    torch.testing.assert_close(biased, want, atol=2e-2, rtol=2e-2)
    # Every pair takes the same 7.5, so every row's maximum moves by it.
    torch.testing.assert_close(biased, plain + 7.5, atol=3e-2, rtol=0)


def test_returning_it_does_not_change_the_attention_output(device):
    """A store is all that is added, so the result has to be bit-identical."""
    q, k, v = qkv(B, H, N, N, 32, 32, torch.float32, device)
    plain = mf.attention(q, k, v)
    out, _ = mf.attention(q, k, v, return_max_score=True)
    torch.testing.assert_close(out, plain, atol=0, rtol=0)


def test_it_is_not_differentiable_and_does_not_disturb_the_gradient(device):
    """It is a diagnostic. Asking for it must not change what the backward computes."""
    q, k, v = qkv(B, H, N, N, 32, 32, torch.float32, device)
    out, max_score = mf.attention(q, k, v, return_max_score=True)
    assert not max_score.requires_grad
    g = torch.randn_like(out)
    dq_gated = torch.autograd.grad(out, q, g, retain_graph=False)[0]

    q2, k2, v2 = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    out2 = mf.attention(q2, k2, v2)
    dq_plain = torch.autograd.grad(out2, q2, g)[0]
    torch.testing.assert_close(dq_gated, dq_plain, atol=0, rtol=0)


def test_a_row_with_no_key_left_reads_zero(device):
    """`m_safe` substitutes 0.0 for the -inf an empty row would otherwise carry, and the
    dense path has to say the same thing rather than -inf or NaN."""
    q, k, v = qkv(B, H, N, N, 32, 32, torch.float32, device)
    key_mask = torch.ones(B, N, dtype=torch.bool, device=device)
    key_mask[1, :] = False

    _, got, _, want = _both(q, k, v, key_mask=key_mask)
    assert torch.isfinite(got).all()
    torch.testing.assert_close(got[1], torch.zeros_like(got[1]), atol=0, rtol=0)
    torch.testing.assert_close(got, want, atol=2e-2, rtol=2e-2)
