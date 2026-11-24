import molflash as mf
import pytest
import torch
from _util import qkv, signed_offset, signed_offset_torch
from torch.autograd import forward_ad as ad


@pytest.mark.parametrize("loss", ["primal", "tangent", "joint"])
def test_qkv_jvp_learned_joint_bias(device, loss):
    torch.manual_seed(73)
    b, h, nq, nk = 2, 2, 35, 23
    q, k, v = qkv(b, h, nq, nk, 16, 8, device=device)
    dq, dk, dv = (torch.randn_like(x).requires_grad_() for x in (q, k, v))
    w = torch.randn((h, 7, 4), device=device, requires_grad=True)
    qi = torch.arange(nq, device=device).expand(b, -1)
    ki = torch.arange(nk, device=device).expand(b, -1)
    targets = (qi + 2) % nk
    rel = signed_offset_torch(qi, targets)
    edge_values = (w[:, rel, 2] - w[:, rel, 0]).permute(1, 2, 0).unsqueeze(2)
    edge = mf.EdgeBias(targets.unsqueeze(-1), edge_values)
    pair = mf.PairBias(signed_offset, q_features=qi, k_features=ki, table=w[..., 0])
    keep = ki < nk - 3
    keep[1] = False
    with ad.dual_level():
        inputs = [ad.make_dual(x, d) for x, d in zip((q, k, v), (dq, dk, dv), strict=True)]
        out, tangent = ad.unpack_dual(mf.attention(*inputs, bias=[pair, edge], key_mask=keep))

    relation = signed_offset_torch(qi.unsqueeze(-1), ki.unsqueeze(-2))
    bonds = (targets.unsqueeze(-1) == ki.unsqueeze(-2)).long() * 2
    bias = w[:, relation, bonds].transpose(0, 1)
    def oracle(q, k, v):
        scores = (q @ k.transpose(-1, -2)) / 4 + bias
        masked = scores.masked_fill(~keep[:, None, None, :], -1e30)
        prob = masked.softmax(-1) * keep[:, None, None, :]
        return prob @ v

    ref, dref = torch.autograd.functional.jvp(oracle, (q, k, v), (dq, dk, dv), create_graph=True)
    g, dg = torch.randn_like(out), torch.randn_like(out)
    variables = (q, k, v, w) if loss == "primal" else (q, k, v, w, dq, dk, dv)
    def grads(x, dx):
        objective = (x * g).sum() if loss == "primal" else (dx * dg).sum()
        if loss == "joint":
            objective = objective + (x * g).sum()
        return torch.autograd.grad(objective, variables)
    for got, want in zip(
        (out, tangent, *grads(out, tangent)), (ref, dref, *grads(ref, dref)), strict=True
    ):
        torch.testing.assert_close(got, want, rtol=5e-4, atol=2e-4)


def test_plain_inputs_inside_dual_level_and_zero_direction(device):
    q, k, v = qkv(1, 1, 7, 7, 16, device=device)
    labels = torch.arange(7, device=device)[None]
    table = torch.randn(1, 7, device=device, requires_grad=True)
    bias = mf.PairBias(signed_offset, labels, table)
    regular = mf.attention(q, k, v, bias=bias)
    with ad.dual_level():
        plain, tangent = ad.unpack_dual(mf.attention(q, k, v, bias=bias))
        assert tangent is None
        q_dual = ad.make_dual(q, torch.zeros_like(q))
        _, zero = ad.unpack_dual(mf.attention(q_dual, k, v, bias=bias))
        assert zero is not None and not zero.count_nonzero()
    torch.testing.assert_close(plain, regular)
    for got, want in zip(
        torch.autograd.grad(plain.sum(), (q, k, v, table)),
        torch.autograd.grad(regular.sum(), (q, k, v, table)),
        strict=True,
    ):
        torch.testing.assert_close(got, want)
