# Recipes

Each section solves one common case. The code runs as written, top to bottom, and the test suite executes it, so it stays in step with the package. All sections share this setup:

```python
import torch
import torch.nn as nn
import triton
import triton.language as tl

import molflash as mf
from molflash import terms

device = "cuda"
B, H, N, D = 2, 4, 64, 32
q, k, v = (torch.randn(B, H, N, D, device=device, requires_grad=True) for _ in range(3))
pos = torch.randn(B, N, 3, device=device) * 4      # coordinates in Å
```

## Contents

1. [Bonds](#bonds)
2. [Edge embeddings from a graph network](#edge-embeddings-from-a-graph-network)
3. [Per-token embeddings from another model](#per-token-embeddings-from-another-model)
4. [Distance: buckets or a learned curve](#distance-buckets-or-a-learned-curve)
5. [Distance that depends on atom types](#distance-that-depends-on-atom-types)
6. [Sequence position](#sequence-position)
7. [Cutoff mask and smooth gate](#cutoff-mask-and-smooth-gate)
8. [Several terms at once](#several-terms-at-once)
9. [Your own term without Triton](#your-own-term-without-triton)
10. [Your own pair function in Triton](#your-own-pair-function-in-triton)
11. [Replacing attention in an existing model](#replacing-attention-in-an-existing-model)
12. [Molecules of different sizes in one batch](#molecules-of-different-sizes-in-one-batch)
13. [Gradients with respect to coordinates](#gradients-with-respect-to-coordinates)
14. [torch.compile](#torchcompile)
15. [When something looks wrong](#when-something-looks-wrong)

## Bonds

List each atom's bonded neighbours in `index` (`-1` for unused slots) and their bond types. Write every bond in both directions.

```python
K = 4
index = torch.full((B, N, K), -1, device=device)
bond_type = torch.zeros((B, N, K), dtype=torch.long, device=device)
index[:, 1:, 0] = torch.arange(N - 1, device=device)       # i bonded to i - 1
index[:, :-1, 1] = torch.arange(1, N, device=device)       # and to i + 1
bond_type[:, :, :2] = 1                                    # single bonds

bond_table = nn.Parameter(torch.zeros(5, H, device=device))  # [bond types, heads]
out = mf.attention(q, k, v, bias=terms.bonds(bond_table, index, bond_type))
```

`terms.bonds` is two lines: it looks up `bond_table[bond_type]` and wraps the result in an `EdgeBias`. Pairs that are not bonded get 0, so the table needs no "no bond" entry.

## Edge embeddings from a graph network

Any per-edge embedding works. Project it to one value per head and pass it with the same `index`:

```python
edge_emb = torch.randn(B, N, K, 16, device=device)         # e.g. output of a GNN layer
edge_proj = nn.Linear(16, H).to(device)
out = mf.attention(q, k, v, bias=terms.edge_features(index, edge_emb, edge_proj))
out.sum().backward()                                        # reaches edge_proj and the GNN
```

## Per-token embeddings from another model

A bias that factorises into per-token vectors, `bias_ij = u_i · w_j`, costs O(N) memory. With embeddings from a protein language model:

```python
plm = torch.randn(B, N, 128, device=device)                # e.g. ESM per-residue embeddings
rank = 8
to_u = nn.Linear(128, H * rank).to(device)
to_w = nn.Linear(128, H * rank).to(device)
u = to_u(plm).view(B, N, H, rank)
w = to_w(plm).view(B, N, H, rank)
out = mf.attention(q, k, v, bias=terms.node_features(u, w))
```

`terms.node_features` only transposes to the `[B, H, N, R]` layout of `NodeBias`. Pass `u` alone for a symmetric bias.

## Distance: buckets or a learned curve

Buckets: a table entry per distance range, the last bucket covering everything beyond `max_distance`.

```python
buckets = nn.Parameter(torch.zeros(H, 32, device=device))
out = mf.attention(q, k, v, bias=terms.distance_buckets(buckets, pos, max_distance=16.0))
```

A smooth curve: the table holds values at evenly spaced distances and the kernel interpolates between them. To learn the curve as a small network, evaluate the network on the grid with `terms.tabulate`. The kernel then reads the resulting table, and the gradient flows back into the network's weights.

```python
curve_net = nn.Sequential(nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, H)).to(device)
table = terms.tabulate(lambda d: curve_net(d[:, None]), 0.0, 20.0, points=128,
                        device=device)                                     # [H, 128]
out = mf.attention(q, k, v, bias=terms.distance_function(table, pos, grid=(0.0, 20.0)))
```

The grid spacing sets the approximation error: 128 points over 20 Å is a spacing of 0.16 Å. Build the table inside `forward`, once per step, and every layer and head reads it at no extra memory cost.

## Distance that depends on atom types

Transformer-M and Uni-Mol encode distance as `f(a[c] · d + b[c])`, where `c` indexes the pair of atom types and `f` is a Gaussian basis followed by a small network. The curve `f` is shared, so it is one table, and `(a, b)` is an affine map per type pair:

```python
num_types = 10
types = torch.randint(0, num_types, (B, N), device=device)
a = nn.Parameter(torch.ones(num_types**2, device=device))
b = nn.Parameter(torch.zeros(num_types**2, device=device))
f = nn.Sequential(nn.Linear(1, 64), nn.GELU(), nn.Linear(64, H)).to(device)
f_table = terms.tabulate(lambda s: f(s[:, None]), -5.0, 25.0, points=512, device=device)
out = mf.attention(q, k, v, bias=terms.typed_distance_function(
    f_table, pos, types, num_types, grid=(-5.0, 25.0), scale=a, shift=b))
```

Gradients reach `a`, `b` and the weights of `f`. Without `scale` and `shift`, pass a `[H, num_types**2, G]` table to give every type pair its own curve.

## Sequence position

Clipped offsets, as in AlphaFold's relative position encoding:

```python
residue = torch.arange(N, device=device).expand(B, N)
relpos = nn.Parameter(torch.zeros(H, 65, device=device))     # offsets -32..32
out = mf.attention(q, k, v, bias=terms.relative_position(relpos, residue, max_offset=32))
```

The offset is `j - i`, signed, so the bias is not symmetric. T5 and the protein models derived from it (ProtT5, ProstT5, Ankh) use logarithmic buckets. The model's own `[H, num_buckets]` table plugs in directly:

```python
t5_table = nn.Parameter(torch.zeros(H, 32, device=device))
out = mf.attention(q, k, v, bias=terms.t5_relative_position(t5_table, residue, max_distance=128),
                   softmax_scale=1.0)                          # T5 does not scale q·k
```

## Cutoff mask and smooth gate

A mask removes pairs. When the tokens are ordered so that neighbours in sequence are close in space, whole tiles fall outside the cutoff and the kernel skips them:

```python
out = mf.attention(q, k, v, mask=terms.distance_cutoff(pos, cutoff=10.0))
```

A gate scales probabilities after the softmax. With a cosine envelope, contributions fade to zero at the cutoff instead of stopping abruptly:

```python
out = mf.attention(q, k, v, mask=terms.distance_cutoff(pos, cutoff=10.0),
                   gate=terms.cosine_cutoff(pos, cutoff=10.0))
```

## Several terms at once

Lists combine: biases add, masks intersect, multipliers and gates multiply. Padding goes in `key_mask`.

```python
chain = (torch.arange(N, device=device) >= N // 2).long().expand(B, N)
same_chain = nn.Parameter(torch.zeros(H, 2, device=device))
key_mask = torch.ones(B, N, dtype=torch.bool, device=device)
key_mask[1, 50:] = False                                    # sample 1 has 50 tokens

out = mf.attention(
    q, k, v,
    bias=[terms.bonds(bond_table, index, bond_type),
          terms.distance_function(table, pos, grid=(0.0, 20.0)),
          terms.relative_position(relpos, residue, max_offset=32),
          terms.same_label(same_chain, chain)],
    mask=terms.distance_cutoff(pos, cutoff=12.0),
    key_mask=key_mask,
)
```

## Your own term without Triton

Most new terms are a different way of filling the inputs of `EdgeBias` or `NodeBias`, which needs only PyTorch. Here every atom gets a bias towards its three nearest neighbours, scaled by a learned per-head factor:

```python
def nearest_neighbours(pos, scale, k=3):
    d = torch.cdist(pos, pos)
    d.diagonal(dim1=1, dim2=2).fill_(float("inf"))
    dist, idx = d.topk(k, largest=False)                    # [B, N, k]
    values = -dist[..., None] * scale                       # [B, N, k, H]
    return mf.EdgeBias(idx, values)

nn_scale = nn.Parameter(torch.ones(H, device=device))
out = mf.attention(q, k, v, bias=nearest_neighbours(pos, nn_scale))
```

`torch.cdist` here is O(N²) on its own, which is fine for small molecules. For large systems, build the neighbour list with a cell list or a library such as `torch_cluster.radius_graph`.

## Your own pair function in Triton

When the value of a pair follows a rule over all pairs, write the rule as a Triton function and decorate it with `@mf.pair_function`. Its arguments are the feature channels of the query token, then those of the key token. It returns an integer code into your table. Writing the same rule in `torch` as `torch_fn` lets the term run in `molflash.reference` and lets debug mode check one against the other.

Here a code distinguishes pairs in the same residue, adjacent residues, the same chain, and different chains:

```python
@mf.pair_function
def residue_relation(chain_i, res_i, chain_j, res_j):
    gap = tl.abs(res_j - res_i)
    code = tl.where(gap == 0, 0, tl.where(gap == 1, 1, 2))
    return tl.where(chain_i == chain_j, code, 3)

def residue_relation_torch(chain_i, res_i, chain_j, res_j):
    gap = (res_j - res_i).abs()
    code = torch.where(gap == 0, 0, torch.where(gap == 1, 1, 2))
    return torch.where(chain_i == chain_j, code, 3)

relation_table = nn.Parameter(torch.zeros(H, 4, device=device))
relation = mf.PairBias(residue_relation, [chain, residue // 4], relation_table,
                       torch_fn=residue_relation_torch)
out = mf.attention(q, k, v, bias=relation)
```

Before training with a new function, compare the kernel with the reference on small inputs. This is the test to copy into your project:

```python
def check_against_reference(term, q, k, v, params, tol=1e-4):
    qr, kr, vr = (x.detach().float().requires_grad_(True) for x in (q, k, v))
    got = mf.attention(q, k, v, bias=term)
    want = mf.reference.attention(qr, kr, vr, bias=term)
    g = torch.randn_like(want)
    dgot = torch.autograd.grad(got, [q, k, v, *params], g, retain_graph=True)
    dwant = torch.autograd.grad(want, [qr, kr, vr, *params], g)
    for a, b in zip([got, *dgot], [want, *dwant]):
        assert (a - b).abs().max() <= tol * b.abs().max().clamp(min=1.0)

check_against_reference(relation, q, k, v, [relation_table])
```

A function can also return a float, interpolated on the table's grid (pass `grid=`), or a `(code, float)` pair; see [guide.md](guide.md#pair-functions). If its features should receive gradients, add `grad_fn=`, a second pair function returning the derivative of the float with respect to every channel. `terms.distance_function` in `torch-ext/molflash/terms.py` is a complete example.

## Replacing attention in an existing model

Find the line that computes the scores, for example

```python
def old_attention(q, k, v, dense_bias):
    scores = q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5 + dense_bias   # [B, H, N, N]
    return scores.softmax(-1) @ v
```

and identify which inputs `dense_bias` is built from. If it comes from distances through a learned function, replace the dense tensor with the corresponding term:

```python
def new_attention(q, k, v, pos, table):
    return mf.attention(q, k, v, bias=terms.distance_function(table, pos, grid=(0.0, 20.0)))
```

Check that the two agree on the model's real inputs before retraining anything. With the original weights, the outputs should match up to floating-point rounding:

```python
dense = mf.reference.dense_term(terms.distance_function(table, pos, grid=(0.0, 20.0)), B, H, N, N)
torch.testing.assert_close(new_attention(q, k, v, pos, table),
                           old_attention(q, k, v, dense), atol=1e-4, rtol=1e-4)
```

The repository also builds as a Hugging Face kernel (`build.toml`). A build published on the Hub loads with `kernels.get_kernel` and exposes the same functions as `import molflash`.

## Molecules of different sizes in one batch

Pack the molecules one after another and describe the boundaries with `cu_seqlens`. Edge indices count from the start of each molecule.

```python
from molflash.varlen import cu_seqlens

sizes = torch.tensor([12, 40, 7], device=device)
cu = cu_seqlens(sizes)
T = int(sizes.sum())
qp, kp, vp = (torch.randn(T, H, D, device=device) for _ in range(3))
pos_p = torch.randn(T, 3, device=device) * 4
out = mf.attention_varlen(qp, kp, vp, cu, cu, max_seqlen_q=40, max_seqlen_k=40,
                          bias=terms.distance_function(table, pos_p, grid=(0.0, 20.0)))
```

Passing `max_seqlen_*` avoids a device synchronisation. From a padded batch, `molflash.varlen.pack(x, key_mask)` and `unpack(x, cu)` convert both ways.

If you prefer to keep a padded layout, several small molecules can share one row: a mask with a molecule label keeps them apart.

```python
molecule = (torch.arange(N, device=device) // 16).expand(B, N)   # four molecules per row
out = mf.attention(q, k, v, mask=terms.same_label_mask(molecule))
```

## Gradients with respect to coordinates

`distance_function` returns `∂L/∂pos` when `pos` requires grad, for example to predict forces as the negative gradient of an energy:

```python
pos_g = pos.clone().requires_grad_(True)
energy = mf.attention(q, k, v, bias=terms.distance_function(table, pos_g, grid=(0.0, 20.0))).sum()
forces = -torch.autograd.grad(energy, pos_g)[0]               # [B, N, 3]
```

Training on forces needs the gradient of this gradient, which is not implemented yet.

## torch.compile

`molflash.attention` is a registered custom operator, so compiled models keep it as one node:

```python
@torch.compile(fullgraph=True)
def layer(q, k, v, pos, table):
    return mf.attention(q, k, v, bias=terms.distance_function(table, pos, grid=(0.0, 20.0)))

out = layer(q, k, v, pos, table)
```

## When something looks wrong

Run one step in debug mode. It checks index ranges and finite values, and recomputes a few rows with the reference:

```python
with mf.debug():
    out = mf.attention(q, k, v, bias=relation)
```

To read the kernel generated for a call, set `MOLFLASH_DUMP_KERNELS=1`. To isolate a kernel problem from a model problem, swap `mf.attention` for `mf.reference.attention`, which takes the same arguments and is easy to step through.
