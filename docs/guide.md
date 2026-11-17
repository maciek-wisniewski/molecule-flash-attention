# Guide

This page explains what `molflash.attention` computes, how structural information gets into it, and where gradients go. [recipes.md](recipes.md) has worked examples for common cases.

## What the kernel computes

For one head and one query `i`, with keys `j`:

```
s_ij = (q_i · k_j · softmax_scale) · M_ij + B_ij
s_ij = -inf            where any mask removes (i, j)
p_ij = softmax_j(s_ij)
o_i  = Σ_j p_ij · G_ij · v_j
```

`B` is the sum of the bias terms, `M` the product of the multipliers and `G` the product of the gates. Each of the four is optional, and the kernel is generated without the code for the ones you do not pass. A query with no key left after masking returns zeros.

None of `B`, `M`, `G` or the mask is ever stored as an `N × N` tensor. The kernel evaluates them for one tile of pairs at a time from inputs whose size grows with `N`, not `N²`. Memory therefore stays linear in sequence length, as in FlashAttention.

```python
import torch
import molflash as mf

device = "cuda"
B, H, N, D = 2, 4, 300, 32
q, k, v = (torch.randn(B, H, N, D, device=device) for _ in range(3))
pos = torch.randn(B, N, 3, device=device) * 5            # coordinates, Å
table = torch.randn(H, 64, device=device, requires_grad=True)

out = mf.attention(q, k, v, bias=mf.terms.distance_function(table, pos, grid=(0.0, 20.0)))
```

### Seeing the scores

A score tile never leaves the kernel. That is what keeps memory linear, and it means you cannot inspect one afterwards. The one number you can have is the largest pre-softmax score of each query row, because the online softmax already tracks it for stability:

```python
out, max_score = mf.attention(
    q, k, v,
    bias=mf.terms.distance_function(table, pos, grid=(0.0, 20.0)),
    return_max_score=True,
)
assert max_score.shape == (B, H, N)      # one float per query row, float32
```

It costs one store per query row and no arithmetic. That is `B·H·Nq` floats against the `B·H·Nq·Dv` the output already writes, so at 24,576 tokens, 4 heads and a head dimension of 128 it is 0.8% of what the kernel stores anyway.

It is a diagnostic rather than a term: it carries no gradient, and asking for it changes neither the output nor the backward. Reach for it when a bias can grow without bound while `q·k` cannot (a learned table has no ceiling the way a normalised dot product does), or to drive a clipping rule from the real maximum instead of from a bound. A row whose keys are all masked away reads `0.0`, the value the kernel substitutes for an empty row's `-inf`.

### Backward preprocessing

Backward computes `delta = sum(dO * O, dim=-1)` in fp32 before the query- and key-tile
kernels. A row-reduction kernel loads the original tensors and reduces in registers,
without materializing full fp32 casts or a full fp32 elementwise product. This does not
change the attention definition, required gradients or asymptotic memory bound. Reduction
order can differ from PyTorch's row sum; numerical agreement is tested within tolerance.

For discrete table gradients that use atomic histograms, the largest valid code in each
tile is accumulated in fp32 before one atomic update. Other codes retain their ordinary
histogram updates; no contribution is discarded. Uniform-code tiles are fully reduced,
while mixed tiles can also avoid repeated updates to that selected bin. This requires no
molecule-specific code values, dense pair cache or lower-precision accumulation.

### Optional sampled backward

`attention` and `attention_varlen` accept `backward_sampling=BackwardSampling(...)`.
The default `None`, and a sampling budget of zero, use the original exact backward,
consume no sampling random numbers and allocate no sampling buffers. Forward values and
maximum-score diagnostics are unchanged for every sampling setting.

`BackwardSampling(budget, block_size=128, seed=None)` is a frozen configuration. `budget`
must be finite in `[0, 1)`; `block_size` must be a positive power of two and a multiple
of the selected backward query tile. It defines query-row groups independently for each
batch element and head. Packed sequences have separate groups, including their partial
last group; groups never cross sequence boundaries.

At backward time, each group's output-gradient norm supplies its score. The budget
determines a desired kept-group count from the cumulative squared scores. Positive
scores receive probabilities `min(1, lambda * score)` whose sum matches that count;
kept groups carry inverse-probability weights. Zero-score groups need no work. One
weighted output gradient drives Q/K/V and every learned term gradient, including gates,
multipliers, affine parameters and feature gradients. Both gradient kernels skip
discarded query tiles before evaluating pairs. Additional storage is linear: group
scores/weights and a weighted output-gradient tensor, never a dense pair weight matrix.

This is the sampled-gradient method of
[LazyFlash v1](https://arxiv.org/html/2610.31911v1), adapted to structural terms. The
gradient estimator equals the dense gradient in expectation, subject to ordinary floating
point rounding; individual backward calls differ. This does not imply identical optimizer
updates, lower variance or unchanged training quality.

`seed=None` consumes fresh numbers from the device's default PyTorch generator during
backward. Set the ordinary training seed for reproducibility. An explicit nonnegative
63-bit seed repeats draws for the same group layout and is intended for controlled
comparisons; callers must vary it to obtain independent repeated samples. Reusing a fixed
seed throughout training does not establish conditional unbiasedness. Forward and activation
checkpoint recomputation draw no sampling randomness. Higher-order gradients remain
unsupported. `molflash.reference` remains the exact dense oracle and accepts no sampling
configuration; compare a sampled backward against that oracle with the same weighted
output gradient.

## Three primitives

Every term is built from one of three primitives. They differ in which pairs they reach and where the value of a pair comes from.

| primitive | you pass | value for pair (i, j) | memory |
|---|---|---|---|
| `EdgeBias(index, values)` | `index [B, N, K]` key positions (`-1` = empty slot), `values [B, N, K, H]` | `values[i, s]` if `index[i, s] == j`, else 0 | O(N·K·H) |
| `NodeBias(u, w)` | `u [B, H, Nq, R]`, `w [B, H, Nk, R]` | `u[i] · w[j]` | O(N·H·R) |
| `PairBias(fn, features, table)` | per-token features and a `@triton.jit` function of a pair | `table[h, fn(features_i, features_j)]` | O(N + table) |

All tensors stay with the caller. A term holds no parameters of its own, so a learned table, the output of an `nn.Linear` or an embedding from another model are passed in as they are, and autograd carries the gradient back to whatever produced them.

### EdgeBias or PairBias?

Both add a number to the score of a pair. The difference is how many pairs they reach and how many distinct numbers they can produce.

```
EdgeBias: a few listed pairs per query,     PairBias: every pair, value chosen
each with its own value                     from a small table by a rule

      keys →                                      keys →
q  . 3 . . . . . 1                          q  a b c c d d d d
u  . . 7 . . 2 . .                          u  b a b c c d d d
e  4 . . . . . . .                          e  c b a b c c d d
r  . . . . 5 . 6 .                          r  c c b a b c c d
i  . . . . . . . .                          i  d c c b a b c c
e  . 2 . . . . . 9                          e  d d c c b a b c
s  . . . . 8 . . .                          s  d d d c c b a b
↓  . . 1 . . . . .                          ↓  d d d d c c b a
```

| | EdgeBias | PairBias |
|---|---|---|
| pairs reached | the `K` listed per query; all others get 0 | all `Nq × Nk` |
| distinct values | one per listed pair | at most the table size |
| the information sits in | `values`, computed however you like | the rule `fn` and the table |
| typical use | bonds, contacts, graph edges with features | distance, sequence offset, same chain |

A bond with its own embedding cannot be a table entry, and a distance applies to every pair, so neither primitive replaces the other. When a model has both, pass both.

`NodeBias` covers the remaining case: a bias that factorises into per-token vectors, such as projected embeddings from a protein language model. It is the same computation as appending `u` to `q` and `w` to `k`, without the scale.

## Pair functions

A pair function is a Triton function decorated with `@mf.pair_function`, which applies `triton.jit` and returns a handle that `torch.compile` can treat as a constant. A plain `@triton.jit` function also works outside `torch.compile`. Each feature tensor `[B, N]` contributes one argument per side and `[B, N, C]` contributes `C`. Query-side arguments come first, then key-side ones, and they arrive as a column and a row vector, so elementwise code produces the whole tile:

```python
import triton.language as tl

@mf.pair_function
def chain_offset(chain_i, res_i, chain_j, res_j):
    same = chain_i == chain_j
    return tl.where(same, tl.minimum(tl.abs(res_j - res_i), 32), 33)
```

With `features=[chain, residue]`, both `[B, N]` int, this returns a code in `[0, 33]` for every pair.

What `fn` returns decides how the table is read:

| `fn` returns | `table` | pair value |
|---|---|---|
| int code `c` | `[H, K]` | `table[h, c]` |
| float `t` | `[H, G]`, `grid=(lo, hi)` | `table[h]` linearly interpolated at `t` |
| `(c, t)` | `[H, K, G]`, `grid` | curve `c` of the table, interpolated at `t` |
| `(c, t)` | `[H, G]`, `grid`, `affine=(a, b)` | `table[h]` interpolated at `a[c] · t + b[c]` |

The `G` entries of an interpolated table sit evenly on `[lo, hi]`, and `t` outside that range is clamped. Several rules apply to every kind of table:

- With `per_sample=True` the table has a leading batch axis and each sample reads its own.
- A table with one row instead of `H` is shared by all heads.
- A code outside `[0, K)` contributes nothing: 0 for a bias, 1 for a multiplier or gate.
- The function may treat the query and key sides differently. `fn(i, j)` and `fn(j, i)` need not agree.

For cross-attention, pass `q_features=` and `k_features=` instead of `features=`. They need the same number of channels and the same dtypes.

`torch_fn=` takes the same function written with `torch` operations. `molflash.reference` uses it, which lets the term run on CPU and lets debug mode check the Triton version against it. Terms in `molflash.terms` all provide one.

## Masks, multipliers and gates

The roles are separate classes that take the same arguments:

| role | argument of `attention` | classes | identity |
|---|---|---|---|
| bias | `bias=` | `EdgeBias`, `NodeBias`, `PairBias` | 0 |
| mask | `mask=` | `EdgeMask`, `PairMask` | keep |
| multiplier | `multiplier=` | `PairScale` | 1 |
| gate | `gate=` | `PairGate` | 1 |

Each argument takes one term or a list. Biases add, multipliers and gates multiply, masks combine with AND. Passing a term under the wrong argument raises an error.

`PairMask(fn, features)` keeps a pair when `fn` returns true and takes no table. Before computing a tile the kernel evaluates its masks, and if they remove every pair it skips the tile's matrix products. A distance cutoff on coordinates ordered along the chain therefore saves time as well as excluding pairs. `EdgeMask(index)` keeps only the listed pairs. `key_mask=[B, Nk]` removes padding keys.

A gate multiplies probabilities after the softmax without renormalising. This is how the smooth cutoffs of some equivariant models work. FlexAttention can express a multiplier through `score_mod`, but not a gate.

## Gradients

| input | gradient |
|---|---|
| `q`, `k`, `v` | yes |
| `EdgeBias.values`, `NodeBias.u`, `NodeBias.w` | yes |
| `table`, `affine` | yes |
| float features of a pair term | only with `grad_fn=`; otherwise passing a feature that requires grad raises |
| integer features, `index`, masks | none, they are discrete |

`grad_fn` has the signature of `fn` and returns the derivative of `t` with respect to every channel, query side first. `mf.terms.distance_function` provides it, so coordinates that require grad receive `∂L/∂x`, for example to predict forces as `−∂E/∂x`.

Table gradients are accumulated with atomics in one of three ways, chosen per call by table size: in registers for tables of up to 32 entries, in a private buffer per program for larger ones, or directly in the table when the private buffers would exceed 256 MB. Table gradients are only computed for tables that require grad, so a frozen table costs about half as much in the backward pass. The result is the same up to the order of floating-point additions, which makes table gradients not bitwise reproducible from run to run.

The backward is differentiable once. Gradients of gradients (`create_graph=True`) and forward-mode AD are not implemented; see [Not supported yet](#not-supported-yet).

## Batching: padding and varlen

`mf.attention` takes padded batches `[B, H, N, D]` with `key_mask` marking real tokens. When sizes vary a lot, padding wastes work: a batch of one large and 31 small molecules computes 32 large attention maps. `mf.attention_varlen` takes the sequences concatenated instead, in the layout of `flash_attn_varlen_func`:

```python
from molflash.varlen import cu_seqlens

lengths = torch.tensor([5, 120, 33], device=device)
cu = cu_seqlens(lengths)                    # [0, 5, 125, 158]
T = int(lengths.sum())
qp, kp, vp = (torch.randn(T, H, D, device=device) for _ in range(3))
pos_p = torch.randn(T, 3, device=device)
out = mf.attention_varlen(qp, kp, vp, cu, cu,
                          bias=mf.terms.distance_function(table, pos_p, grid=(0.0, 20.0)))
```

Per-token term tensors are packed the same way (`[T, ...]`). `EdgeBias.index` counts positions from the start of the query's own sequence, as edge indices do in a PyG batch after subtracting the offsets. `NodeBias` takes `[T, H, R]`. `per_sample` tables are not available under varlen.

Queries and keys may have different lengths in both modes. `q` and `k` may have a different head dimension from `v`.

## Checks and debug mode

Every call checks shapes, dtypes and devices of all inputs and names the offending argument in the error. These checks read metadata only and cost nothing.

Values are not checked by default, because that needs a device synchronisation. Inside the kernel, unsafe values have a defined effect: an edge index outside `[−1, Nk)` is ignored, an out-of-range code takes the role's identity, and a NaN interpolation coordinate reads the start of the grid. To find such inputs, turn on debug mode:

```python
with mf.debug():
    out = mf.attention(q, k, v, bias=mf.terms.distance_function(table, pos, grid=(0.0, 20.0)))
```

or set `MOLFLASH_DEBUG=1`. Debug mode raises `molflash.MolflashDebugError` on out-of-range indices, non-finite inputs, broken `cu_seqlens` and samples with no valid key. It also recomputes eight random query rows with `molflash.reference` and raises if they disagree, which catches a `torch_fn` that does not match its Triton function.

## Supported configurations

| | |
|---|---|
| GPUs | NVIDIA, tested on A100 (sm80), H100 and H200 (sm90) |
| dtypes | float16, bfloat16, float32 for `q`, `k`, `v`; terms are computed in float32 |
| head dim | 1 to 256, `q·k` and `v` independently |
| PyTorch | 2.14 (tested), with the Triton it ships |
| `torch.compile` | supported with `fullgraph=True`; sequence lengths may be dynamic, head dims and table sizes are specialised |
| CPU | `molflash.reference`, same arguments, dense |

## Not supported yet

- Forward-mode AD (`torch.func.jvp`, `torch.autograd.forward_ad`), `torch.vmap`, and gradients of gradients (`create_graph=True`). Each raises `NotImplementedError`. Without that check PyTorch would return a zero tangent for forward mode.
- Pair representations updated layer by layer through non-linear operations, as in AlphaFold's Pairformer or Uni-Mol2. Such a model holds an `N × N × c` tensor anyway, so the kernel cannot remove that memory.
- Pair values that cannot be computed from per-token inputs, such as shortest-path distances on a graph. Precompute them densely and use PyTorch SDPA.
- AMD and Intel GPUs. Triton supports them, but the kernel is not tested there.
