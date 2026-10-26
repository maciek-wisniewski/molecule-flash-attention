import molflash as mf
import torch
import triton
import triton.language as tl

CUDA = torch.cuda.is_available()
DTYPES = [torch.float32, torch.float16, torch.bfloat16] if CUDA else [torch.float32]
# Relative to the largest reference value; low precision compares against fp32 math on the
# same rounded inputs.
TOL = {torch.float32: 2e-4, torch.float16: 1e-2, torch.bfloat16: 4e-2}


def qkv(b, h, nq, nk, d, dv=None, dtype=torch.float32, device="cpu", varlen=False):
    dv = d if dv is None else dv
    shape = (lambda n, e: (n, h, e)) if varlen else (lambda n, e: (b, h, n, e))
    q = torch.randn(shape(nq, d), device=device).to(dtype)
    k = torch.randn(shape(nk, d), device=device).to(dtype)
    v = torch.randn(shape(nk, dv), device=device).to(dtype)
    return [x.requires_grad_(True) for x in (q, k, v)]


def check(q, k, v, params=(), tol=None, varlen=None, **kw):
    """Runs the kernel and the reference on the same terms and compares the output and the
    gradient of every input in ``q, k, v, *params``."""
    params = list(params)
    tol = TOL[q.dtype] if tol is None else tol
    qr, kr, vr = (x.detach().float().requires_grad_(True) for x in (q, k, v))
    if varlen is None:
        o = mf.attention(q, k, v, **kw)
        r = mf.reference.attention(qr, kr, vr, **kw)
    else:
        cq, ck = varlen
        o = mf.attention_varlen(q, k, v, cq, ck, **kw)
        r = mf.reference.attention_varlen(qr, kr, vr, cq, ck, **kw)
    g = torch.randn_like(r)
    # Terms may hold tensors computed from params, shared by both graphs.
    got = torch.autograd.grad(
        o, [q, k, v] + params, g.to(o.dtype), allow_unused=True, retain_graph=True
    )
    want = torch.autograd.grad(r.float(), [qr, kr, vr] + params, g, allow_unused=True)
    names = ["out", "dq", "dk", "dv"] + [f"param{i}" for i in range(len(params))]
    pairs = [(o, r)] + list(zip(got, want, strict=True))
    for name, (a, b) in zip(names, pairs, strict=True):
        assert (a is None) == (b is None), f"{name}: gradient present in only one implementation"
        if a is None:
            continue
        a, b = a.float(), b.float()
        assert torch.isfinite(a).all(), f"{name}: non-finite values"
        # Some exact gradients are zero (a constant added to a table cancels in the softmax).
        scale = b.abs().max().clamp(min=0.05)
        err = ((a - b).abs().max() / scale).item()
        assert err <= tol, f"{name}: relative error {err:.2e} > {tol:.0e}"


@triton.jit
def label_distance(a, b):
    return tl.abs(a - b)


def label_distance_torch(a, b):
    return (a - b).abs()


@triton.jit
def type_pair_distance(xi, yi, zi, ti, xj, yj, zj, tj):
    dx = xi - xj
    dy = yi - yj
    dz = zi - zj
    return (ti * 4 + tj).to(tl.int32), tl.sqrt_rn(dx * dx + dy * dy + dz * dz)


def type_pair_distance_torch(xi, yi, zi, ti, xj, yj, zj, tj):
    return (ti * 4 + tj).long(), torch.sqrt((xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2)


@triton.jit
def signed_offset(i, j):
    return tl.minimum(tl.maximum(j - i, -3), 3) + 3


def signed_offset_torch(i, j):
    return (j - i).clamp(-3, 3) + 3
