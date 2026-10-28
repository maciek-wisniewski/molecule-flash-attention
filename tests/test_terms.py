import math

import pytest
import torch
from _util import CUDA, check, qkv
from molflash import reference, terms

N = 160 if CUDA else 33
B, H = 2, 2


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _hf_t5_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
    # Transcribed from transformers' T5Attention._relative_position_bucket.
    relative_buckets = 0
    if bidirectional:
        num_buckets //= 2
        relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
        relative_position = torch.abs(relative_position)
    else:
        relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact
    large = max_exact + (
        torch.log(relative_position.float() / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).to(torch.long)
    large = torch.min(large, torch.full_like(large, num_buckets - 1))
    return relative_buckets + torch.where(is_small, relative_position, large)


@pytest.mark.parametrize("num_buckets,max_distance", [(32, 128), (64, 128), (32, 64)])
@pytest.mark.parametrize("bidirectional", [True, False])
def test_t5_buckets_match_transformers(num_buckets, max_distance, bidirectional):
    rel = torch.arange(-3 * max_distance, 3 * max_distance + 1)
    want = _hf_t5_bucket(rel, bidirectional, num_buckets, max_distance)
    table = terms.t5_buckets(num_buckets, max_distance, bidirectional)
    got = table[rel.clamp(-max_distance, max_distance) + max_distance]
    assert torch.equal(got, want)


def test_t5_relative_position(device):
    q, k, v = qkv(B, H, N, N, 16, device=device)
    index = torch.arange(N, device=device).expand(B, N)
    table = torch.randn(H, 32, device=device, requires_grad=True)
    check(q, k, v, [table], bias=terms.t5_relative_position(table, index, max_distance=20))


def test_relative_position_and_same_label(device):
    q, k, v = qkv(B, H, N, N, 16, device=device)
    index = torch.randint(0, 50, (B, N), device=device)
    chain = torch.randint(0, 3, (B, N), device=device)
    t1 = torch.randn(H, 9, device=device, requires_grad=True)
    t2 = torch.randn(H, 2, device=device, requires_grad=True)
    check(
        q,
        k,
        v,
        [t1, t2],
        bias=[terms.relative_position(t1, index, max_offset=4), terms.same_label(t2, chain)],
    )


def test_distance_buckets_agree_exactly(device):
    pos = torch.randn(B, N, 3, device=device) * 5
    table = torch.arange(32, dtype=torch.float32, device=device).expand(H, 32).contiguous()
    term = terms.distance_buckets(table, pos, max_distance=12.0)
    from molflash.kernels import materialize_pair_fn
    from molflash.reference import pair_outputs

    kernel_codes = materialize_pair_fn(term, N, N)
    torch_codes, _ = pair_outputs(term, N, N)
    assert torch.equal(kernel_codes.long(), torch_codes.long())


def test_distance_buckets(device):
    q, k, v = qkv(B, H, N, N, 16, device=device)
    pos = torch.randn(B, N, 3, device=device) * 5
    table = torch.randn(H, 32, device=device, requires_grad=True)
    check(q, k, v, [table], bias=terms.distance_buckets(table, pos, max_distance=12.0))


def test_tabulated_module(device):
    q, k, v = qkv(B, H, N, N, 16, device=device)
    pos = torch.randn(B, N, 3, device=device) * 5
    # No bias in the last layer: a constant added to every table entry cancels in the softmax,
    # so its exact gradient is zero and both implementations return rounding noise.
    mlp = torch.nn.Sequential(
        torch.nn.Linear(1, 8), torch.nn.GELU(), torch.nn.Linear(8, H, bias=False)
    ).to(device)
    table = terms.tabulate(lambda x: mlp(x[:, None]), 0.0, 15.0, 64, device=device)
    # A 64-entry table is accumulated with atomics, whose order is not fixed, and the MLP's
    # gradient reaches each of its parameters through all 64 entries. On a GPU that lands at
    # 2.0e-4 against the 2e-4 the suite uses for float32 elsewhere, so whether it passes is
    # luck: it fails in a full run and passes ten times out of ten on its own.
    check(
        q,
        k,
        v,
        list(mlp.parameters()),
        tol=5e-4,
        bias=terms.distance_function(table, pos, grid=(0.0, 15.0)),
    )


def test_typed_distance_affine(device):
    q, k, v = qkv(B, H, N, N, 16, device=device)
    pos = torch.randn(B, N, 3, device=device) * 5
    types = torch.randint(0, 5, (B, N), device=device)
    scale = (torch.rand(25, device=device) + 0.5).requires_grad_(True)
    shift = torch.randn(25, device=device, requires_grad=True)
    table = torch.randn(H, 64, device=device, requires_grad=True)
    term = terms.typed_distance_function(
        table, pos, types, 5, grid=(-5.0, 20.0), scale=scale, shift=shift
    )
    check(q, k, v, [table, scale, shift], bias=term)


def test_bonds_and_edge_features(device):
    q, k, v = qkv(B, H, N, N, 16, device=device)
    index = torch.randint(-1, N, (B, N, 4), device=device)
    bond_type = torch.randint(0, 5, (B, N, 4), device=device)
    bond_table = torch.randn(5, H, device=device, requires_grad=True)
    feats = torch.randn(B, N, 4, 6, device=device)
    proj = torch.nn.Linear(6, H).to(device)
    check(
        q,
        k,
        v,
        [bond_table, proj.weight],
        bias=[
            terms.bonds(bond_table, index, bond_type),
            terms.edge_features(index, feats, proj),
        ],
    )


def test_node_features(device):
    q, k, v = qkv(B, H, N, N, 16, device=device)
    emb = torch.randn(B, N, 10, device=device)
    proj = torch.nn.Linear(10, H * 3).to(device)
    u = proj(emb).view(B, N, H, 3)
    check(q, k, v, [proj.weight], bias=terms.node_features(u))


def test_cosine_cutoff_values():
    pos = torch.zeros(1, 2, 3)
    pos[0, 1, 0] = 2.5
    gate = terms.cosine_cutoff(pos, cutoff=5.0, points=1001)
    g = reference.dense_term(gate, 1, 1, 2, 2)
    assert torch.allclose(g[0, 0, 0, 1], torch.tensor(0.5), atol=1e-5)
    pos[0, 1, 0] = 6.0
    assert reference.dense_term(gate, 1, 1, 2, 2)[0, 0, 0, 1] == 0


def test_same_label_mask_separates_molecules(device):
    q, k, v = qkv(1, H, N, N, 16, device=device)
    mol = (torch.arange(N, device=device) >= N // 2).long()[None]
    check(q, k, v, mask=terms.same_label_mask(mol))
