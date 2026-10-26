import molflash as mf
import pytest
import torch
from _util import CUDA, DTYPES, check, label_distance, label_distance_torch, qkv
from molflash import terms
from molflash.varlen import cu_seqlens, pack, unpack

H = 2
LQ = [5, 70, 33, 1] if CUDA else [5, 21, 9, 1]
LK = [7, 40, 90, 3] if CUDA else [7, 17, 30, 3]


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _lengths(device):
    return cu_seqlens(torch.tensor(LQ, device=device)), cu_seqlens(torch.tensor(LK, device=device))


@pytest.mark.parametrize("dtype", DTYPES)
def test_plain(device, dtype):
    cq, ck = _lengths(device)
    q, k, v = qkv(1, H, sum(LQ), sum(LK), 16, 24, dtype, device, varlen=True)
    check(q, k, v, varlen=(cq, ck))


@pytest.mark.parametrize("dtype", DTYPES)
def test_all_primitives(device, dtype):
    cq, ck = _lengths(device)
    tq, tk = sum(LQ), sum(LK)
    q, k, v = qkv(1, H, tq, tk, 16, dtype=dtype, device=device, varlen=True)
    index = torch.cat([torch.randint(-1, n, (m, 3)) for m, n in zip(LQ, LK, strict=True)]).to(
        device
    )
    values = torch.randn(tq, 3, H, device=device, requires_grad=True)
    u = torch.randn(tq, H, 4, device=device, requires_grad=True)
    w = torch.randn(tk, H, 4, device=device, requires_grad=True)
    pq = (torch.randn(tq, 3, device=device) * 3).requires_grad_(True)
    pk = (torch.randn(tk, 3, device=device) * 3).requires_grad_(True)
    table = torch.randn(H, 16, device=device, requires_grad=True)
    lq = torch.randint(0, 5, (tq,), device=device)
    lk = torch.randint(0, 5, (tk,), device=device)
    gate_table = torch.rand(H, 5, device=device, requires_grad=True)
    check(
        q,
        k,
        v,
        [values, u, w, pq, pk, table, gate_table],
        varlen=(cq, ck),
        bias=[
            mf.EdgeBias(index, values),
            mf.NodeBias(u, w),
            terms.distance_function(table, q_pos=pq, k_pos=pk, grid=(0.0, 10.0)),
        ],
        gate=mf.PairGate(
            label_distance,
            q_features=lq,
            k_features=lk,
            table=gate_table,
            torch_fn=label_distance_torch,
        ),
        mask=terms.distance_cutoff(q_pos=pq.detach(), k_pos=pk.detach(), cutoff=7.0),
    )


def test_matches_padded(device):
    lengths = torch.tensor(LQ, device=device)
    n = max(LQ)
    key_mask = torch.arange(n, device=device)[None, :] < lengths[:, None]
    q, k, v = (torch.randn(len(LQ), H, n, 16, device=device) for _ in range(3))
    pos = torch.randn(len(LQ), n, 3, device=device)
    table = torch.randn(H, 16, device=device)
    padded = mf.attention(
        q, k, v, key_mask=key_mask, bias=terms.distance_function(table, pos, grid=(0.0, 5.0))
    )
    cu = cu_seqlens(lengths)
    packed = mf.attention_varlen(
        *(pack(x.transpose(1, 2), key_mask) for x in (q, k, v)),
        cu,
        cu,
        bias=terms.distance_function(table, pack(pos, key_mask), grid=(0.0, 5.0)),
    )
    back = unpack(packed, cu, n).transpose(1, 2)
    torch.testing.assert_close(
        back[key_mask[:, None, :, None].expand_as(back)],
        padded[key_mask[:, None, :, None].expand_as(padded)],
        atol=1e-5,
        rtol=1e-5,
    )
