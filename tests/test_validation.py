import molflash as mf
import pytest
import torch
from _util import label_distance, label_distance_torch, qkv
from molflash import terms

B, H, N = 2, 2, 20
LABEL_DISTANCE = mf.pair_function(label_distance)


@pytest.fixture
def inputs(device):
    torch.manual_seed(0)
    return qkv(B, H, N, N, 16, device=device)


def _raises(match, *args, **kw):
    with pytest.raises((ValueError, TypeError), match=match):
        mf.attention(*args, **kw)


def test_layout_errors(inputs, device):
    q, k, v = inputs
    _raises(r"q must be \[B, H, N, D\]", q[0], k, v)
    _raises("share a dtype", q, k.half(), v)
    _raises("float16, bfloat16 or float32", q.double(), k, v)
    _raises("same number of heads", q, k[:, :1], v[:, :1])
    _raises("same head dim", q, k[..., :8], v)
    _raises("same length", q, k, v[:, :, :5])
    _raises(
        "key_mask must be", q, k, v, key_mask=torch.ones(B, N + 1, dtype=torch.bool, device=device)
    )


def test_role_errors(inputs, device):
    pos = torch.randn(B, N, 3, device=device)
    _raises(
        "is a mask term and was passed as bias=",
        *inputs,
        bias=terms.distance_cutoff(pos, cutoff=3.0),
    )
    _raises("pass it as gate=", *inputs, bias=terms.cosine_cutoff(pos, cutoff=3.0))


def test_term_shape_errors(inputs, device):
    idx = torch.zeros(B, N, 3, dtype=torch.long, device=device)
    _raises(
        r"EdgeBias.values must be",
        *inputs,
        bias=mf.EdgeBias(idx, torch.zeros(B, N, 3, H + 1, device=device)),
    )
    _raises(
        r"EdgeBias.index must be",
        *inputs,
        bias=mf.EdgeBias(idx[:, :5], torch.zeros(B, 5, 3, H, device=device)),
    )
    _raises("integer tensor", *inputs, mask=mf.EdgeMask(idx.float()))
    u = torch.zeros(B, H, N, 3, device=device)
    _raises("same rank", *inputs, bias=mf.NodeBias(u, u[..., :2]))
    label = torch.zeros(B, N, dtype=torch.long, device=device)
    _raises(
        r"table must be \[H, \.\.\.\] with H = 2",
        *inputs,
        bias=mf.PairBias(label_distance, label, torch.zeros(3, 4, device=device)),
    )
    _raises(
        r"features\[0\] must be",
        *inputs,
        bias=mf.PairBias(label_distance, label[:, :4], torch.zeros(H, 4, device=device)),
    )


def test_term_construction_errors(device):
    label = torch.zeros(B, N, dtype=torch.long, device=device)
    with pytest.raises(ValueError, match="needs grid="):
        mf.PairBias(label_distance, label, torch.zeros(H, 4), affine=(torch.ones(4), torch.ones(4)))
    with pytest.raises(ValueError, match="not both"):
        mf.PairBias(label_distance, label, torch.zeros(H, 4), q_features=label)
    with pytest.raises(ValueError, match="needs both"):
        mf.PairBias(label_distance, table=torch.zeros(H, 4), q_features=label)
    with pytest.raises(ValueError, match="table must be"):
        mf.PairBias(label_distance, label, torch.zeros(H, 4, 5))


def test_grid_errors(inputs, device):
    pos = torch.randn(B, N, 3, device=device)
    _raises(
        "lo < hi",
        *inputs,
        bias=terms.distance_function(torch.zeros(H, 8, device=device), pos, grid=(5.0, 1.0)),
    )
    _raises(
        "at least 2 grid points",
        *inputs,
        bias=terms.distance_function(torch.zeros(H, 1, device=device), pos, grid=(0.0, 1.0)),
    )


def test_debug_catches_bad_index(inputs, device):
    idx = torch.full((B, N, 2), -1, dtype=torch.long, device=device)
    idx[1, 4, 1] = N + 3
    vals = torch.zeros(B, N, 2, H, device=device)
    with mf.debug(), pytest.raises(mf.MolflashDebugError, match=r"holds 23 at \[1, 4, 1\]"):
        mf.attention(*inputs, bias=mf.EdgeBias(idx, vals))
    # Outside debug mode the slot is ignored rather than read out of bounds.
    mf.attention(*inputs, bias=mf.EdgeBias(idx, vals))


def test_debug_catches_non_finite(inputs, device):
    pos = torch.randn(B, N, 3, device=device)
    pos[0, 3, 1] = float("nan")
    with mf.debug(), pytest.raises(mf.MolflashDebugError, match="non-finite"):
        mf.attention(
            *inputs,
            bias=terms.distance_function(torch.zeros(H, 8, device=device), pos, grid=(0.0, 5.0)),
        )


def test_debug_catches_mismatched_torch_fn(inputs, device):
    label = torch.randint(0, 4, (B, N), device=device)
    table = torch.randn(H, 4, device=device)
    wrong = mf.PairBias(label_distance, label, table, torch_fn=lambda a, b: (a + b).clamp(max=3))
    with mf.debug(), pytest.raises(mf.MolflashDebugError, match="disagree"):
        mf.attention(*inputs, bias=wrong)
    with mf.debug():
        mf.attention(
            *inputs, bias=mf.PairBias(label_distance, label, table, torch_fn=label_distance_torch)
        )


def test_debug_varlen_index_is_local(device):
    from molflash.varlen import cu_seqlens

    cu = cu_seqlens(torch.tensor([4, 6], device=device))
    q, k, v = qkv(1, H, 10, 10, 16, device=device, varlen=True)
    idx = torch.zeros(10, 1, dtype=torch.long, device=device)
    idx[1, 0] = 5  # the first sequence has only 4 keys
    with mf.debug(), pytest.raises(mf.MolflashDebugError, match="start of the sequence"):
        mf.attention_varlen(
            q, k, v, cu, cu, bias=mf.EdgeBias(idx, torch.zeros(10, 1, H, device=device))
        )


def test_compiles_without_graph_breaks(inputs, device):
    q, k, v = inputs
    pos = torch.randn(B, N, 3, device=device)
    label = torch.randint(0, 4, (B, N), device=device)
    table = torch.randn(H, 16, device=device)

    def f(q, k, v, pos, table):
        return mf.attention(
            q,
            k,
            v,
            bias=[
                terms.distance_function(table, pos, grid=(0.0, 10.0)),
                terms.relative_position(torch.ones(H, 9, device=device), label, max_offset=4),
                mf.PairBias(LABEL_DISTANCE, label, torch.ones(H, 4, device=device)),
            ],
            mask=terms.distance_cutoff(pos, cutoff=5.0),
        )

    got = torch.compile(f, backend="eager", fullgraph=True)(q, k, v, pos, table)
    torch.testing.assert_close(got, f(q, k, v, pos, table))


def test_compile_rejects_undecorated_pair_function(inputs, device):
    label = torch.randint(0, 4, (B, N), device=device)

    def f(q, k, v):
        return mf.attention(
            q, k, v, bias=mf.PairBias(label_distance, label, torch.ones(H, 4, device=device))
        )

    with pytest.raises(Exception, match="decorated with @molflash.pair_function"):
        torch.compile(f, backend="eager", fullgraph=True)(*inputs)


def test_multihead_attention_module(device):
    torch.manual_seed(0)
    layer = mf.nn.MultiheadAttention(32, 4).to(device)
    x = torch.randn(B, N, 32, device=device)
    pos = torch.randn(B, N, 3, device=device)
    table = torch.zeros(4, 8, device=device)
    out = layer(x, bias=terms.distance_function(table, pos, grid=(0.0, 5.0)))
    assert out.shape == x.shape
    out.sum().backward()
    assert layer.q_proj.weight.grad is not None


def test_unsupported_autograd_modes_raise(inputs, device):
    q, k, v = inputs
    with pytest.raises(NotImplementedError, match="torch.func"):
        torch.func.jvp(lambda q: mf.attention(q, k, v), (q.detach(),), (torch.ones_like(q),))
    with pytest.raises(NotImplementedError, match="torch.func"):
        torch.vmap(lambda q: mf.attention(q, k, v))(q.detach()[None])
    import torch.autograd.forward_ad as fwAD

    with fwAD.dual_level(), pytest.raises(NotImplementedError, match="forward-mode"):
        mf.attention(fwAD.make_dual(q.detach(), torch.ones_like(q)), k, v)
    o = mf.attention(q, k, v)
    with pytest.raises(NotImplementedError, match="create_graph"):
        torch.autograd.grad(o.sum(), q, create_graph=True)
