"""Three synthetic workloads with the shape of real ones. Coordinates fill a sphere at the heavy
atom density of a protein (one atom per 20 A^3) and are ordered along a space-filling curve, so
sequence neighbours are spatial neighbours, as in a chain."""

import math

import torch
from molflash import terms


def coordinates(b, n, device):
    radius = (3 * n * 20.0 / (4 * math.pi)) ** (1 / 3)
    pts = torch.randn(b, n, 3, device=device)
    pts = (
        pts
        / pts.norm(dim=-1, keepdim=True)
        * radius
        * torch.rand(b, n, 1, device=device) ** (1 / 3)
    )
    cell = ((pts + radius) / 4.0).long().clamp(min=0)
    code = torch.zeros(b, n, dtype=torch.long, device=device)
    for bit in range(10):  # Morton order on 4 A cells
        for axis in range(3):
            code |= ((cell[..., axis] >> bit) & 1) << (3 * bit + axis)
    order = code.argsort(dim=1)
    return torch.gather(pts, 1, order[..., None].expand(-1, -1, 3))


def _chain_bonds(b, n, device, k=4):
    """Each atom bonded to its sequence neighbours (both directions), random bond types."""
    i = torch.arange(n, device=device)
    offsets = torch.tensor([-2, -1, 1, 2], device=device)[:k]
    index = (i[:, None] + offsets[None]).expand(b, n, k).clone()
    index[(index < 0) | (index >= n)] = -1
    types = torch.randint(0, 5, (b, n, k), device=device)
    return index, types


class Scenario:
    name = ""

    def __init__(self, b, n, heads, device):
        self.b, self.n, self.h, self.device = b, n, heads, device
        self.params = {}

    def param(self, name, *shape, scale=1.0):
        t = (torch.randn(*shape, device=self.device) * scale).requires_grad_(True)
        self.params[name] = t
        return t

    def kwargs(self):
        raise NotImplementedError


class SmallMolecule(Scenario):
    """Bonds plus a learned function of distance (Transformer-M / Uni-Mol style)."""

    name = "molecule"

    def __init__(self, b, n, heads, device):
        super().__init__(b, n, heads, device)
        self.pos = coordinates(b, n, device)
        self.index, self.types = _chain_bonds(b, n, device)
        self.param("bond", 5, heads)
        self.param("dist", heads, 128, scale=0.1)

    def kwargs(self):
        p = self.params
        return dict(
            bias=[
                terms.bonds(p["bond"], self.index, self.types),
                terms.distance_function(p["dist"], self.pos, grid=(0.0, 20.0)),
            ]
        )


class Protein(Scenario):
    """Relative sequence position, distance, and a 15 A neighbourhood mask."""

    name = "protein"

    def __init__(self, b, n, heads, device):
        super().__init__(b, n, heads, device)
        self.pos = coordinates(b, n, device)
        self.res = (torch.arange(n, device=device) // 8).expand(b, n).contiguous()
        self.param("relpos", heads, 65, scale=0.1)
        self.param("dist", heads, 64, scale=0.1)

    def kwargs(self):
        p = self.params
        return dict(
            bias=[
                terms.relative_position(p["relpos"], self.res, max_offset=32),
                terms.distance_function(p["dist"], self.pos, grid=(0.0, 16.0)),
            ],
            mask=terms.distance_cutoff(self.pos, cutoff=15.0),
        )


class Complex(Scenario):
    """Every primitive at once: bonds, distance, relative position, chain identity, a rank-8
    bias from per-atom embeddings, and a distance mask."""

    name = "complex"

    def __init__(self, b, n, heads, device):
        super().__init__(b, n, heads, device)
        self.pos = coordinates(b, n, device)
        self.index, self.types = _chain_bonds(b, n, device)
        self.res = (torch.arange(n, device=device) // 8).expand(b, n).contiguous()
        self.chain = (torch.arange(n, device=device) * 4 // n).expand(b, n).contiguous()
        self.emb = torch.randn(b, n, 32, device=device)
        self.param("bond", 5, heads)
        self.param("dist", heads, 64, scale=0.1)
        self.param("relpos", heads, 65, scale=0.1)
        self.param("chain", heads, 2)
        self.param("proj", 32, heads * 8, scale=0.05)

    def kwargs(self):
        p = self.params
        u = (self.emb @ p["proj"]).view(self.b, self.n, self.h, 8)
        return dict(
            bias=[
                terms.bonds(p["bond"], self.index, self.types),
                terms.distance_function(p["dist"], self.pos, grid=(0.0, 16.0)),
                terms.relative_position(p["relpos"], self.res, max_offset=32),
                terms.same_label(p["chain"], self.chain),
                terms.node_features(u),
            ],
            mask=terms.distance_cutoff(self.pos, cutoff=15.0),
        )


class LowRank(Scenario):
    """A rank-16 bias from per-atom embeddings alone: the case FlashBias computes exactly."""

    name = "lowrank"

    def __init__(self, b, n, heads, device):
        super().__init__(b, n, heads, device)
        self.emb = torch.randn(b, n, 32, device=device)
        self.param("proj", 32, heads * 16, scale=0.05)

    def kwargs(self):
        u = (self.emb @ self.params["proj"]).view(self.b, self.n, self.h, 16)
        return dict(bias=terms.node_features(u))


SCENARIOS = {s.name: s for s in (SmallMolecule, Protein, Complex, LowRank)}
