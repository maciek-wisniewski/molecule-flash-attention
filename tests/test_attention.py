import molflash as mf
import pytest
import torch
from _util import (
    CUDA,
    DTYPES,
    check,
    label_distance,
    label_distance_torch,
    qkv,
    signed_offset,
    signed_offset_torch,
    type_pair_distance,
    type_pair_distance_torch,
)
from molflash import terms

N = 203 if CUDA else 37
B, H = 2, 3


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _pos(b, n, device):
    return torch.randn(b, n, 3, device=device) * 4


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("d,dv", [(16, 16), (8, 8), (33, 20), (64, 64), (128, 32)])
def test_plain(device, dtype, d, dv):
    check(*qkv(B, H, N, N, d, dv, dtype, device))


@pytest.mark.parametrize("dtype", DTYPES)
def test_cross_attention_lengths(device, dtype):
    check(*qkv(B, H, N, N // 2 + 3, 32, 16, dtype, device))


@pytest.mark.parametrize("dtype", DTYPES)
def test_softmax_scale(device, dtype):
    check(*qkv(B, H, N, N, 16, dtype=dtype, device=device), softmax_scale=0.3)


@pytest.mark.parametrize("dtype", DTYPES)
def test_key_mask_and_empty_rows(device, dtype):
    q, k, v = qkv(B, H, N, N, 16, dtype=dtype, device=device)
    key_mask = torch.rand(B, N, device=device) > 0.4
    key_mask[1] = False  # every query of sample 1 has nothing to attend to
    check(q, k, v, key_mask=key_mask)
    assert mf.attention(q, k, v, key_mask=key_mask)[1].abs().max() == 0


@pytest.mark.parametrize("dtype", DTYPES)
def test_edge_bias(device, dtype):
    q, k, v = qkv(B, H, N, N, 16, dtype=dtype, device=device)
    index = torch.randint(-1, N, (B, N, 5), device=device)
    index[0, 0, 1] = index[0, 0, 0]  # a key listed twice adds both values
    index[1, 2, 0] = N + 7  # out of range: ignored
    values = torch.randn(B, N, 5, H, device=device, requires_grad=True)
    check(q, k, v, [values], bias=mf.EdgeBias(index, values))


@pytest.mark.parametrize("dtype", DTYPES)
def test_edge_mask(device, dtype):
    q, k, v = qkv(B, H, N, N, 16, dtype=dtype, device=device)
    index = torch.randint(-1, N, (B, N, 4), device=device)
    index[0, 3] = -1  # a query with no neighbours outputs zeros
    check(q, k, v, mask=mf.EdgeMask(index))
    assert mf.attention(q, k, v, mask=mf.EdgeMask(index))[0, :, 3].abs().max() == 0


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("rank", [1, 7, 40])
def test_node_bias(device, dtype, rank):
    q, k, v = qkv(B, H, N, N, 16, dtype=dtype, device=device)
    u = torch.randn(B, H, N, rank, device=device, requires_grad=True)
    w = torch.randn(B, H, N, rank, device=device, requires_grad=True)
    check(q, k, v, [u, w], bias=mf.NodeBias(u, w))


@pytest.mark.parametrize("dtype", DTYPES)
def test_node_bias_terms_concatenate(device, dtype):
    q, k, v = qkv(B, H, N, N, 16, dtype=dtype, device=device)
    u1, w1, u2, w2 = (
        torch.randn(B, H, N, r, device=device, requires_grad=True) for r in (3, 3, 5, 5)
    )
    check(q, k, v, [u1, w1, u2, w2], bias=[mf.NodeBias(u1, w1), mf.NodeBias(u2, w2)])


@pytest.mark.parametrize("dtype", DTYPES)
def test_pair_code(device, dtype):
    q, k, v = qkv(B, H, N, N, 16, dtype=dtype, device=device)
    label = torch.randint(0, 6, (B, N), device=device)
    table = torch.randn(H, 4, device=device, requires_grad=True)  # codes 4 and 5 out of range
    term = mf.PairBias(label_distance, label, table, torch_fn=label_distance_torch)
    check(q, k, v, [table], bias=term)


@pytest.mark.parametrize("dtype", DTYPES)
def test_pair_code_without_torch_fn(device, dtype):
    q, k, v = qkv(B, H, N, N, 16, dtype=dtype, device=device)
    label = torch.randint(0, 6, (B, N), device=device)
    table = torch.randn(H, 6, device=device, requires_grad=True)
    check(q, k, v, [table], bias=mf.PairBias(label_distance, label, table))


@pytest.mark.parametrize("dtype", DTYPES)
def test_pair_asymmetric(device, dtype):
    q, k, v = qkv(B, H, N, N, 16, dtype=dtype, device=device)
    index = torch.arange(N, device=device).expand(B, N)
    table = torch.randn(H, 7, device=device, requires_grad=True)
    check(
        q,
        k,
        v,
        [table],
        bias=mf.PairBias(signed_offset, index, table, torch_fn=signed_offset_torch),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_pair_table_per_sample_and_shared_heads(device, dtype):
    q, k, v = qkv(B, H, N, N, 16, dtype=dtype, device=device)
    label = torch.randint(0, 6, (B, N), device=device)
    per_sample = torch.randn(B, H, 6, device=device, requires_grad=True)
    shared = torch.randn(1, 6, device=device, requires_grad=True)
    check(
        q,
        k,
        v,
        [per_sample, shared],
        bias=[
            mf.PairBias(
                label_distance, label, per_sample, per_sample=True, torch_fn=label_distance_torch
            ),
            mf.PairBias(label_distance, label, shared, torch_fn=label_distance_torch),
        ],
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_pair_value_interpolated(device, dtype):
    q, k, v = qkv(B, H, N, N, 16, dtype=dtype, device=device)
    pos = _pos(B, N, device)
    table = torch.randn(H, 24, device=device, requires_grad=True)
    check(q, k, v, [table], bias=terms.distance_function(table, pos, grid=(0.5, 9.0)))


@pytest.mark.parametrize("dtype", DTYPES)
def test_pair_value_coordinate_gradient(device, dtype):
    q, k, v = qkv(B, H, N, N, 16, dtype=dtype, device=device)
    pos = _pos(B, N, device).requires_grad_(True)
    table = torch.randn(H, 24, device=device, requires_grad=True)
    check(q, k, v, [table, pos], bias=terms.distance_function(table, pos, grid=(0.0, 12.0)))


@pytest.mark.parametrize("dtype", DTYPES)
def test_coordinate_gradient_cross_attention(device, dtype):
    q, k, v = qkv(B, H, N, N - 5, 16, dtype=dtype, device=device)
    pq = _pos(B, N, device).requires_grad_(True)
    pk = _pos(B, N - 5, device).requires_grad_(True)
    table = torch.randn(H, 24, device=device, requires_grad=True)
    check(
        q,
        k,
        v,
        [table, pq, pk],
        bias=terms.distance_function(table, q_pos=pq, k_pos=pk, grid=(0.0, 12.0)),
    )


def test_feature_gradient_needs_grad_fn(device):
    q, k, v = qkv(B, H, N, N, 16, device=device)
    pos = _pos(B, N, device).requires_grad_(True)
    term = mf.PairBias(
        type_pair_distance,
        [pos, pos[..., 0]],
        torch.randn(H, 16, 8, device=device),
        grid=(0.0, 10.0),
    )
    with pytest.raises(ValueError, match="no grad_fn"):
        mf.attention(q, k, v, bias=term)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("per_head", [False, True])
def test_pair_code_value_affine(device, dtype, per_head):
    q, k, v = qkv(B, H, N, N, 16, dtype=dtype, device=device)
    pos = _pos(B, N, device)
    types = torch.randint(0, 4, (B, N), device=device).float()
    shape = (H, 16) if per_head else (16,)
    a = (torch.rand(shape, device=device) + 0.5).requires_grad_(True)
    b = torch.randn(shape, device=device).requires_grad_(True)
    table = torch.randn(H, 40, device=device, requires_grad=True)
    term = mf.PairBias(
        type_pair_distance,
        [pos, types],
        table,
        grid=(-3.0, 14.0),
        affine=(a, b),
        torch_fn=type_pair_distance_torch,
    )
    check(q, k, v, [table, a, b], bias=term)


@pytest.mark.parametrize("dtype", DTYPES)
def test_pair_code_indexed_curves(device, dtype):
    q, k, v = qkv(B, H, N, N, 16, dtype=dtype, device=device)
    pos = _pos(B, N, device)
    types = torch.randint(0, 4, (B, N), device=device).float()
    table = torch.randn(H, 16, 12, device=device, requires_grad=True)
    term = mf.PairBias(
        type_pair_distance, [pos, types], table, grid=(0.0, 10.0), torch_fn=type_pair_distance_torch
    )
    check(q, k, v, [table], bias=term)


@pytest.mark.parametrize("dtype", DTYPES)
def test_pair_mask_with_skipped_tiles(device, dtype):
    q, k, v = qkv(B, H, N, N, 16, dtype=dtype, device=device)
    # Sorted coordinates along x make whole tiles fall outside the cutoff.
    pos = _pos(B, N, device)
    pos[..., 0] = torch.linspace(0, 200, N, device=device)
    check(q, k, v, mask=terms.distance_cutoff(pos, cutoff=15.0))


@pytest.mark.parametrize("dtype", DTYPES)
def test_multipliers(device, dtype):
    q, k, v = qkv(B, H, N, N, 16, dtype=dtype, device=device)
    label = torch.randint(0, 6, (B, N), device=device)
    t1 = torch.randn(H, 6, device=device, requires_grad=True)
    t2 = torch.rand(H, 6, device=device, requires_grad=True)
    check(
        q,
        k,
        v,
        [t1, t2],
        multiplier=[
            mf.PairScale(label_distance, label, t1, torch_fn=label_distance_torch),
            mf.PairScale(label_distance, label, t2, torch_fn=label_distance_torch),
        ],
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_gates(device, dtype):
    q, k, v = qkv(B, H, N, N, 16, dtype=dtype, device=device)
    pos = _pos(B, N, device)
    label = torch.randint(0, 6, (B, N), device=device)
    t1 = torch.rand(H, 6, device=device, requires_grad=True)
    t2 = torch.rand(H, 32, device=device, requires_grad=True)
    check(
        q,
        k,
        v,
        [t1, t2],
        gate=[
            mf.PairGate(label_distance, label, t1, torch_fn=label_distance_torch),
            mf.PairGate(
                terms._distance_fn(), pos, t2, grid=(0.0, 10.0), torch_fn=terms._distance_torch
            ),
        ],
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_everything_together(device, dtype):
    q, k, v = qkv(B, H, N, N, 32, 16, dtype=dtype, device=device)
    pos = _pos(B, N, device).requires_grad_(True)
    label = torch.randint(0, 6, (B, N), device=device)
    index = torch.randint(-1, N, (B, N, 3), device=device)
    values = torch.randn(B, N, 3, H, device=device, requires_grad=True)
    u = torch.randn(B, H, N, 4, device=device, requires_grad=True)
    w = torch.randn(B, H, N, 4, device=device, requires_grad=True)
    dist_table = torch.randn(H, 20, device=device, requires_grad=True)
    lab_table = torch.randn(H, 6, device=device, requires_grad=True)
    scale_table = torch.rand(H, 6, device=device, requires_grad=True)
    key_mask = torch.rand(B, N, device=device) > 0.2
    check(
        q,
        k,
        v,
        [values, u, w, dist_table, lab_table, scale_table, pos],
        bias=[
            mf.EdgeBias(index, values),
            mf.NodeBias(u, w),
            terms.distance_function(dist_table, pos, grid=(0.0, 15.0)),
            mf.PairBias(label_distance, label, lab_table, torch_fn=label_distance_torch),
        ],
        multiplier=mf.PairScale(label_distance, label, scale_table, torch_fn=label_distance_torch),
        mask=terms.distance_cutoff(pos.detach(), cutoff=12.0),
        gate=terms.cosine_cutoff(pos.detach(), cutoff=12.0),
        key_mask=key_mask,
    )


@pytest.mark.parametrize("mode", ["reduce", "private", "shared"])
def test_table_gradient_modes(device, monkeypatch, mode):
    from molflash import kernels

    monkeypatch.setattr(kernels, "_REDUCE_MAX_CODES", 16 if mode == "reduce" else 0)
    monkeypatch.setattr(kernels, "_PRIVATE_BUDGET", 0 if mode == "shared" else 2**30)
    q, k, v = qkv(B, H, N, N, 16, device=device)
    pos = _pos(B, N, device)
    label = torch.randint(0, 6, (B, N), device=device)
    types = torch.randint(0, 4, (B, N), device=device).float()
    lab = torch.randn(H, 6, device=device, requires_grad=True)
    shared = torch.randn(1, 6, device=device, requires_grad=True)
    per_sample = torch.randn(B, H, 6, device=device, requires_grad=True)
    curve = torch.randn(H, 30, device=device, requires_grad=True)
    a = (torch.rand(16, device=device) + 0.5).requires_grad_(True)
    b = torch.randn(H, 16, device=device, requires_grad=True)
    check(
        q,
        k,
        v,
        [lab, shared, per_sample, curve, a, b],
        bias=[
            mf.PairBias(label_distance, label, lab, torch_fn=label_distance_torch),
            mf.PairBias(label_distance, label, shared, torch_fn=label_distance_torch),
            mf.PairBias(
                label_distance, label, per_sample, per_sample=True, torch_fn=label_distance_torch
            ),
            mf.PairBias(
                type_pair_distance,
                [pos, types],
                curve,
                grid=(-2.0, 12.0),
                affine=(a, b),
                torch_fn=type_pair_distance_torch,
            ),
        ],
    )


def test_interpolated_table_register_reduction(device, monkeypatch):
    from molflash import kernels

    monkeypatch.setattr(kernels, "_REDUCE_MAX_GRID", 64)
    q, k, v = qkv(B, H, N, N, 16, device=device)
    pos = _pos(B, N, device)
    shared = torch.randn(1, 24, device=device, requires_grad=True)
    per_head = torch.randn(H, 24, device=device, requires_grad=True)
    check(
        q,
        k,
        v,
        [shared, per_head],
        bias=[
            terms.distance_function(shared, pos, grid=(0.0, 12.0)),
            terms.distance_function(per_head, pos, grid=(1.0, 9.0)),
        ],
    )
