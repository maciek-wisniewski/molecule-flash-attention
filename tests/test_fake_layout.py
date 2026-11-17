from __future__ import annotations

import molflash as mf
import pytest
import torch
from _util import (
    label_distance,
    label_distance_torch,
    qkv,
    type_pair_distance,
    type_pair_distance_torch,
)
from molflash import ops
from molflash.plan import build_plan, collect_terms, lookup, register


@pytest.mark.parametrize(
    "role", ["table", "features", "affine", "varlen_node", "edge", "edge_half", "edge_bfloat16"]
)
def test_backward_fake_metadata_matches_real_term_gradient_layout(role: str, device: str) -> None:
    torch.manual_seed(817)
    packed = role == "varlen_node"
    q, k, v = qkv(1, 2, 7, 9, 8, device=device, varlen=packed)
    if role == "table":
        labels_q = torch.arange(7, device=device)[None]
        labels_k = torch.arange(9, device=device)[None]
        table = torch.randn(257, 2, device=device).T.requires_grad_()
        bias = mf.PairBias(
            label_distance,
            q_features=labels_q,
            k_features=labels_k,
            table=table,
            torch_fn=label_distance_torch,
        )
    elif role == "features":
        pq = torch.randn(1, 7, 3, device=device).requires_grad_()
        pk = torch.randn(1, 9, 3, device=device).requires_grad_()
        table = torch.randn(2, 8, device=device, requires_grad=True)
        bias = mf.terms.distance_function(table, q_pos=pq, k_pos=pk, grid=(0.0, 8.0))
    elif role == "affine":
        pq = torch.randn(1, 7, 3, device=device)
        pk = torch.randn(1, 9, 3, device=device)
        lq = torch.randint(0, 4, (1, 7), device=device)
        lk = torch.randint(0, 4, (1, 9), device=device)
        a = (0.1 + torch.rand(16, 2, device=device)).T.requires_grad_()
        b = torch.randn(16, 2, device=device).T.requires_grad_()
        table = torch.randn(2, 8, device=device, requires_grad=True)
        bias = mf.PairBias(
            type_pair_distance,
            q_features=[pq, lq],
            k_features=[pk, lk],
            table=table,
            grid=(0.0, 8.0),
            affine=(a, b),
            torch_fn=type_pair_distance_torch,
        )
    elif packed:
        u = torch.randn(7, 2, 3, device=device, requires_grad=True)
        w = torch.randn(9, 2, 3, device=device, requires_grad=True)
        bias = mf.NodeBias(u, w)
    else:
        index = torch.randint(-1, 9, (1, 7, 3), device=device)
        dtype = {"edge_half": torch.float16, "edge_bfloat16": torch.bfloat16}.get(
            role, torch.float32
        )
        values = torch.randn(1, 7, 3, 4, device=device, dtype=dtype)[..., ::2].requires_grad_()
        bias = mf.EdgeBias(index, values)
    terms = collect_terms(bias, None, None, None)
    plan = build_plan(q, k, v, terms, None, varlen=packed)
    if role in {"edge_half", "edge_bfloat16"}:
        _, slots, _ = lookup(register(plan.desc))
        slot = slots["edge_values"]
        packed_values = plan.tensors[slot]
        raw = torch.empty_strided(
            packed_values.shape,
            tuple(2 * stride for stride in packed_values.stride()),
            dtype=dtype,
            device=device,
        )
        raw.copy_(packed_values)
        plan.tensors[slot] = raw
    values = dict(
        q=q,
        k=k,
        v=v,
        tensors=plan.tensors,
        key=register(plan.desc),
        scale=8**-0.5,
        cu_q=torch.tensor([0, 7], dtype=torch.int32, device=device) if packed else None,
        cu_k=torch.tensor([0, 9], dtype=torch.int32, device=device) if packed else None,
        max_q=7 if packed else 0,
        max_k=9 if packed else 0,
        sampling_budget=0.0,
        sampling_block=128,
        sampling_seed=-1,
    )
    forward_args = tuple(values[arg.name] for arg in ops._fwd._opoverload._schema.arguments)
    output, lse, _ = ops._fwd(*forward_args)
    values.update(o=output, lse=lse, do=torch.randn_like(output))
    backward_args = tuple(values[arg.name] for arg in ops._bwd._opoverload._schema.arguments)
    torch.library.opcheck(ops._bwd, backward_args, test_utils=("test_faketensor",))
