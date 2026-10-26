import molflash as mf
import pytest
import torch
from _util import check, label_distance, label_distance_torch, qkv
from molflash import terms
from molflash.varlen import cu_seqlens

pytestmark = pytest.mark.gpu
H = 4
LABEL_DISTANCE = mf.pair_function(label_distance)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _model_terms(pos, label, index, values, dist_table, lab_table):
    return dict(
        bias=[
            terms.distance_function(dist_table, pos, grid=(0.0, 15.0)),
            mf.PairBias(LABEL_DISTANCE, label, lab_table, torch_fn=label_distance_torch),
            mf.EdgeBias(index, values),
        ],
        mask=terms.distance_cutoff(pos.detach(), cutoff=12.0),
    )


def test_torch_compile_fullgraph():
    b, n = 2, 300
    q, k, v = qkv(b, H, n, n, 64, dtype=torch.bfloat16, device="cuda")
    pos = torch.randn(b, n, 3, device="cuda") * 5
    label = torch.randint(0, 6, (b, n), device="cuda")
    index = torch.randint(-1, n, (b, n, 4), device="cuda")
    values = torch.randn(b, n, 4, H, device="cuda", requires_grad=True)
    dist_table = torch.randn(H, 32, device="cuda", requires_grad=True)
    lab_table = torch.randn(H, 6, device="cuda", requires_grad=True)

    def f(q, k, v, values, dist_table, lab_table):
        return mf.attention(
            q, k, v, **_model_terms(pos, label, index, values, dist_table, lab_table)
        )

    compiled = torch.compile(f, fullgraph=True)
    args = (q, k, v, values, dist_table, lab_table)
    want = f(*args)
    got = compiled(*args)
    torch.testing.assert_close(got, want, atol=0, rtol=0)
    g = torch.randn_like(want)
    gw = torch.autograd.grad(want, args, g)
    gc = torch.autograd.grad(compiled(*args), args, g)
    for a, b_ in zip(gc, gw, strict=True):
        torch.testing.assert_close(a, b_, atol=1e-3, rtol=1e-2)


def test_compile_does_not_recompile_for_new_lengths():
    def f(q, k, v, pos, table):
        return mf.attention(q, k, v, bias=terms.distance_function(table, pos, grid=(0.0, 10.0)))

    compiles = []

    def backend(gm, inputs):
        compiles.append(gm)
        return torch._inductor.compile(gm, inputs)

    # Default automatic dynamic shapes: the sequence length becomes symbolic after it changes.
    compiled = torch.compile(f, backend=backend, fullgraph=True)
    table = torch.randn(H, 16, device="cuda")
    for n in (64, 100, 257, 300):
        q, k, v = qkv(1, H, n, n, 32, dtype=torch.float16, device="cuda")
        pos = torch.randn(1, n, 3, device="cuda")
        torch.testing.assert_close(compiled(q, k, v, pos, table), f(q, k, v, pos, table))
    assert len(compiles) == 2


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_long_sequence(dtype):
    n = 4096
    q, k, v = qkv(1, H, n, n, 64, dtype=dtype, device="cuda")
    pos = torch.randn(1, n, 3, device="cuda") * 20
    table = torch.randn(H, 64, device="cuda", requires_grad=True)
    check(
        q,
        k,
        v,
        [table],
        bias=terms.distance_function(table, pos, grid=(0.0, 30.0)),
        mask=terms.distance_cutoff(pos, cutoff=25.0),
    )


def _finite_run(**kw):
    q, k, v = kw.pop("qkv")
    o = mf.attention(q, k, v, **kw)
    o.float().sum().backward()
    torch.cuda.synchronize()
    assert torch.isfinite(o).all()
    for x in (q, k, v):
        assert torch.isfinite(x.grad).all()


def test_fuzz_bad_inputs_do_not_crash():
    for trial in range(20):
        b = int(torch.randint(1, 3, ()))
        nq = int(torch.randint(1, 300, ()))
        nk = int(torch.randint(1, 300, ()))
        d = [8, 16, 40, 64][trial % 4]
        qkv_ = qkv(b, H, nq, nk, d, dtype=torch.bfloat16, device="cuda")
        pos_q = torch.randn(b, nq, 3, device="cuda") * 10
        pos_k = torch.randn(b, nk, 3, device="cuda") * 10
        pos_q[torch.rand(b, nq, device="cuda") < 0.05] = float("nan")
        index = torch.randint(-1000, 1000, (b, nq, 3), device="cuda")
        label_q = torch.randint(-50, 50, (b, nq), device="cuda")
        label_k = torch.randint(-50, 50, (b, nk), device="cuda")
        key_mask = torch.rand(b, nk, device="cuda") < 0.5
        _finite_run(
            qkv=qkv_,
            bias=[
                mf.EdgeBias(index, torch.randn(b, nq, 3, H, device="cuda")),
                mf.PairBias(
                    label_distance,
                    q_features=label_q,
                    k_features=label_k,
                    table=torch.randn(H, 7, device="cuda"),
                ),
                terms.distance_function(
                    torch.randn(H, 8, device="cuda"), q_pos=pos_q, k_pos=pos_k, grid=(0.0, 5.0)
                ),
            ],
            mask=mf.EdgeMask(index),
            key_mask=key_mask,
        )


def test_varlen_with_empty_sequences():
    lengths = torch.tensor([0, 5, 0, 130, 1], device="cuda")
    cu = cu_seqlens(lengths)
    t = int(lengths.sum())
    q, k, v = qkv(1, H, t, t, 32, dtype=torch.bfloat16, device="cuda", varlen=True)
    o = mf.attention_varlen(q, k, v, cu, cu)
    o.float().sum().backward()
    assert torch.isfinite(o).all() and torch.isfinite(q.grad).all()
    check(*qkv(1, H, t, t, 32, device="cuda", varlen=True), varlen=(cu, cu))
