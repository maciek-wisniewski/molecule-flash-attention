"""Query-blocked forward-mode derivative for padded, structurally biased attention."""

from functools import partial

import torch
from torch.utils.checkpoint import checkpoint

from .primitives import EdgeBias, Kind, PairBias
from .reference import dense_term


def check_terms(terms):
    if not all(isinstance(t, EdgeBias) or isinstance(t, PairBias) and t.kind is Kind.CODE
               for t in terms):
        raise NotImplementedError(
            "molflash: forward-mode AD supports EdgeBias and categorical PairBias"
        )


def _tangent_block(q, k, v, dq, dk, dv, *, start, terms, key_mask, scale):
    b, h, rows, _ = q.shape
    nk = k.shape[2]
    region = slice(start, start + rows)
    with torch.autocast(q.device.type, enabled=False):
        q, k, v = q.float(), k.float().transpose(-1, -2), v.float()
        scores = (q @ k) * scale
        for term in terms:
            if isinstance(term, EdgeBias):
                block = EdgeBias(term.index[:, region], term.values[:, region])
            else:
                # Evaluate the registered Triton function, as in the primal kernel.
                block = PairBias(term.fn, table=term.table, per_sample=term.per_sample,
                                 q_features=[f[:, region] for f in term.q_features],
                                 k_features=term.k_features)
            scores = scores + dense_term(block, b, h, rows, nk)
        if key_mask is not None:
            keep = key_mask[:, None, None, :]
            scores = scores.masked_fill(~keep, -torch.inf)
            scores = torch.where(keep.any(-1, keepdim=True), scores, 0.)
        p = torch.softmax(scores, dim=-1)
        if key_mask is not None:
            p = p.masked_fill(~keep, 0.)
        ds = ((dq.float() @ k) + (q @ dk.float().transpose(-1, -2))) * scale
        dp = p * (ds - (p * ds).sum(dim=-1, keepdim=True))
        return (dp @ v + p @ dv.float()).to(dv.dtype)


def attention_tangent(inputs, *, terms, key_mask, scale):
    """Recompute query blocks in reverse AD so probabilities never persist across blocks."""
    primals = [value.primal for value in inputs]
    tangents = [value.tangent if value.tangent is not None else torch.zeros_like(value.primal)
                for value in inputs]
    q, k, v = primals
    dq, dk, dv = tangents
    outputs = []
    for start in range(0, q.shape[2], 32):
        block = partial(_tangent_block, start=start, terms=terms, key_mask=key_mask, scale=scale)
        args = (q[:, :, start:start + 32], k, v, dq[:, :, start:start + 32], dk, dv)
        if torch.is_grad_enabled():
            out = checkpoint(block, *args, use_reentrant=False, preserve_rng_state=False)
        else:
            out = block(*args)
        outputs.append(out)
    return torch.cat(outputs, dim=2) if outputs else v[:, :, :0]
