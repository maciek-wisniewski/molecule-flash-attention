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
    rows = torch.arange(nq, device=device).expand(b, -1)
    qi = rows % 7
    ki = torch.arange(nk, device=device).expand(b, -1)
    targets = torch.full_like(qi, 3)
    bond_type = (rows // 7) % 3 + 1
    rel = signed_offset_torch(qi, targets)
    edge_values = (w[:, rel, bond_type] - w[:, rel, 0]).permute(1, 2, 0).unsqueeze(2)
    edge = mf.EdgeBias(targets.unsqueeze(-1), edge_values)
    pair = mf.PairBias(signed_offset, q_features=qi, k_features=ki, table=w[..., 0])
    keep = ki < nk - 3
    keep[1] = False
    with ad.dual_level():
        inputs = [ad.make_dual(x, d) for x, d in zip((q, k, v), (dq, dk, dv), strict=True)]
        out, tangent = ad.unpack_dual(mf.attention(*inputs, bias=[pair, edge], key_mask=keep))

    relation = signed_offset_torch(qi.unsqueeze(-1), ki.unsqueeze(-2))
    bonds = (targets.unsqueeze(-1) == ki.unsqueeze(-2)).long() * bond_type.unsqueeze(-1)
    active_codes = (4 * relation + bonds)[keep.unsqueeze(1).expand_as(bonds)]
    assert active_codes.unique().numel() == 7 * 4
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


@pytest.mark.parametrize(("nq", "nk"), [(0, 3), (3, 0), (0, 0)])
def test_empty_axes_keep_zero_tangent_gradients(device, nq, nk):
    q, k, v = qkv(1, 2, nq, nk, 16, 8, device=device)
    directions = [torch.randn_like(x).requires_grad_() for x in (q, k, v)]
    values = torch.randn(1, nq, 1, 2, device=device, requires_grad=True)
    edge = mf.EdgeBias(torch.full((1, nq, 1), -1, device=device), values)
    table = torch.randn(2, 7, device=device, requires_grad=True)
    pair = mf.PairBias(signed_offset, q_features=torch.arange(nq, device=device)[None],
                       k_features=torch.arange(nk, device=device)[None], table=table,
                       torch_fn=signed_offset_torch)
    with ad.dual_level():
        duals = [ad.make_dual(x, d) for x, d in zip((q, k, v), directions, strict=True)]
        out, tangent = ad.unpack_dual(mf.attention(*duals, bias=[pair, edge]))
    ref = mf.reference.attention(q, k, v, bias=[pair, edge])
    assert tangent is not None and tangent.shape == out.shape == (1, 2, nq, 8)
    torch.testing.assert_close(out, ref)
    assert tangent.count_nonzero() == 0
    for grad in torch.autograd.grad(tangent.sum(), (q, k, v, *directions, table, values)):
        assert grad.count_nonzero() == 0


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int32, torch.float32])
def test_legacy_numeric_key_mask_matches_boolean_jvp(device, dtype):
    q, k, v = qkv(1, 1, 3, 4, 16, device=device)
    keep = torch.tensor([[1, 0, 1, 0]], dtype=torch.bool, device=device)
    direction = torch.randn_like(q)
    with ad.dual_level():
        dual = ad.make_dual(q, direction)
        got = ad.unpack_dual(mf.attention(dual, k, v, key_mask=keep.to(dtype)))
        want = ad.unpack_dual(mf.attention(dual, k, v, key_mask=keep))
    for x, y in zip(got, want, strict=True):
        torch.testing.assert_close(x, y)
