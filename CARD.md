---
library_name: kernels
{% if license %}license: {{ license }}
{% endif %}tags:
- attention
- triton
- chemistry
- biology
---

Flash attention with structural biases computed inside the kernel: distances, bonds, sequence offsets, atom-type pairs and per-token embeddings enter the attention scores without a dense `[B, H, N, N]` bias tensor. Built for [`kernels`](https://github.com/huggingface/kernels) from [molecule-flash-attention](https://github.com/maciek-wisniewski/molecule-flash-attention).

## How to use

```python
# pip install -U kernels
import torch
from kernels import get_kernel

mf = get_kernel("{{ repo_id }}", version={{ version }}, trust_remote_code=True)

q, k, v = (torch.randn(1, 8, 4096, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3))
pos = torch.randn(1, 4096, 3, device="cuda") * 10
table = torch.zeros(8, 64, device="cuda", requires_grad=True)

out = mf.attention(q, k, v,
                   bias=mf.terms.distance_function(table, pos, grid=(0.0, 20.0)),
                   mask=mf.terms.distance_cutoff(pos, cutoff=15.0))
```

The [guide](https://github.com/maciek-wisniewski/molecule-flash-attention/blob/main/docs/guide.md) explains the three primitives (`EdgeBias`, `NodeBias`, `PairBias`) and the [recipes](https://github.com/maciek-wisniewski/molecule-flash-attention/blob/main/docs/recipes.md) cover bonds, embeddings, distances, sequence position, masks and custom terms.

## Available functions
{% if functions %}
{% for func in functions %}
- `{{ func }}`
{% endfor %}
{% endif %}
{% if layers %}

## Available layers
{% for layer in layers %}
- `{{ layer }}`
{% endfor %}
{% endif %}

## Benchmarks
{% if has_benchmark %}

Run `kernels benchmark {{ repo_id }} --version {{ version }}`. Comparisons with PyTorch, SDPA and FlexAttention are in [docs/benchmarks.md](https://github.com/maciek-wisniewski/molecule-flash-attention/blob/main/docs/benchmarks.md).
{% endif %}

## Source code

https://github.com/maciek-wisniewski/molecule-flash-attention, Apache 2.0.
