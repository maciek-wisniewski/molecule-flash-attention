# molecule-flash-attention

Flash attention for molecules, proteins and graphs, with the structural bias computed inside the kernel.

Transformers over atoms or residues add structure to attention as a bias on the scores: distances, bonds, sequence offsets, atom types. That bias is normally a dense `[B, H, N, N]` tensor, which FlashAttention cannot take without materialising it, so these models fall back to quadratic memory. At 20,000 atoms and 8 heads a float32 bias alone is 12.8 GB.

`molflash` computes the bias tile by tile from per-token inputs (coordinates, edge lists, embeddings, labels), so nothing of size `N × N` is stored. Biases, masks, multipliers and gates are built from three primitives, and every learned table receives its gradient.

## Install

```
pip install git+https://github.com/maciek-wisniewski/molecule-flash-attention
```

Requires PyTorch 2.14 or newer and an NVIDIA GPU. The package is pure Python and Triton, with nothing to compile.

It is also published as a Hugging Face kernel:

```
from kernels import get_kernel
mf = get_kernel("maciek-wisniewski/molflash", version=1, trust_remote_code=True)
```

## Example

```python
import torch
import molflash as mf

device = "cuda"
B, H, N, D = 2, 8, 1000, 64
q, k, v = (torch.randn(B, H, N, D, device=device, dtype=torch.bfloat16) for _ in range(3))
pos = torch.randn(B, N, 3, device=device) * 10                 # atom coordinates
bond_index = torch.randint(-1, N, (B, N, 4), device=device)    # bonded neighbours, -1 = none
bond_type = torch.randint(0, 4, (B, N, 4), device=device)

distance = torch.nn.Parameter(torch.zeros(H, 64, device=device))   # a learned curve per head
bonds = torch.nn.Parameter(torch.zeros(4, H, device=device))       # a value per bond type

out = mf.attention(
    q, k, v,
    bias=[mf.terms.distance_function(distance, pos, grid=(0.0, 20.0)),
          mf.terms.bonds(bonds, bond_index, bond_type)],
    mask=mf.terms.distance_cutoff(pos, cutoff=15.0),
)
```

## What it supports

| primitive | covers |
|---|---|
| `EdgeBias` | values on listed pairs: bonds, contacts, edge embeddings from a GNN |
| `NodeBias` | `u_i · w_j` from per-token vectors: projected embeddings of another model |
| `PairBias` | a `@triton.jit` rule over all pairs reading a learned table: distance buckets or curves, sequence offsets, atom-type pairs, chain identity |

The same primitives act as masks (with skipping of empty tiles), multipliers on the logits and gates on the probabilities. Padded and packed variable-length batches, cross-attention, any head dimension up to 256, float16, bfloat16 and float32, and `torch.compile` are supported. `molflash.terms` has ready-made terms for the cases above, and [docs/recipes.md](docs/recipes.md#your-own-pair-function-in-triton) shows how to write your own.

## Speed and memory

One attention layer, forward and backward, 8 heads of 64, bfloat16, on an A100 40 GB. "protein" adds relative position, a learned distance curve and a 15 Å mask; "molecule" adds bonds and a learned distance curve. Full tables, more baselines and the method are in [docs/benchmarks.md](docs/benchmarks.md).

| scenario | N | molflash | FlexAttention | SDPA + dense bias |
|---|---|---|---|---|
| protein | 4,096 | 4.6 ms, 0.07 GB | 167 ms, 0.23 GB | 743 ms, 9.7 GB |
| protein | 32,768 | 96 ms, 0.41 GB | 3,579 ms, 12.2 GB | out of memory |
| molecule | 4,096 | 14.5 ms, 0.07 GB | 116 ms, 0.07 GB | 485 ms, 9.4 GB |
| molecule | 32,768 | 898 ms, 0.42 GB | 18,718 ms, 0.33 GB | out of memory |

## Documentation

- [docs/guide.md](docs/guide.md): what the kernel computes, the three primitives, gradients, batching, debug mode, limits
- [docs/recipes.md](docs/recipes.md): bonds, embeddings, distances, sequence position, masks, custom terms, porting a model
- [docs/benchmarks.md](docs/benchmarks.md): speed, memory and accuracy against PyTorch, SDPA and FlexAttention

## Tests

```
pytest tests
```

Every term and every gradient is compared with `molflash.reference`, a dense PyTorch implementation of the same function. Without a GPU the kernels run in Triton's interpreter, so the suite also runs on CPU, slowly and on small inputs.

## Citing

If you use this package, please cite:

```bibtex
@software{wisniewski2026molflash,
  author = {Wiśniewski, Maciej},
  title  = {molecule-flash-attention: Dynamic Molecular Graph Flash Attention},
  year   = {2026},
  url    = {https://github.com/maciek-wisniewski/molecule-flash-attention}
}
```

## License

Apache 2.0. The tiled attention loop follows the OpenAI Triton fused-attention tutorial (MIT); see [NOTICE](NOTICE).
