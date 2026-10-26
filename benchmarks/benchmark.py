"""Benchmarks in the format of the Hugging Face ``kernels`` CLI: ``kernels benchmark <repo>``.

The fuller comparison against other attention implementations is run.py.
"""

import torch
from kernels.benchmark import Benchmark


class _Scenario(Benchmark):
    seed = 0
    n = 2048
    heads = 8
    head_dim = 64

    def setup(self):
        mf = self.kernel
        b, n, h, d = 1, self.n, self.heads, self.head_dim
        self.q, self.k, self.v = (
            torch.randn(b, h, n, d, device=self.device, dtype=torch.bfloat16) for _ in range(3)
        )
        self.pos = torch.randn(b, n, 3, device=self.device) * 10
        self.table = torch.randn(h, 64, device=self.device)
        self.res = torch.arange(n, device=self.device)[None]
        self.relpos = torch.randn(h, 65, device=self.device)
        self.kw = self.terms(mf)
        self.out = torch.empty(b, h, n, d, device=self.device, dtype=torch.bfloat16)

    def terms(self, mf):
        raise NotImplementedError

    def _run(self):
        self.out.copy_(self.kernel.attention(self.q, self.k, self.v, **self.kw))

    def _reference(self):
        return self.kernel.reference.attention(self.q, self.k, self.v, **self.kw)


class DistanceBias(_Scenario):
    def terms(self, mf):
        return dict(bias=mf.terms.distance_function(self.table, self.pos, grid=(0.0, 20.0)))

    def benchmark_distance_bias(self):
        self._run()

    def verify_distance_bias(self):
        return self._reference()


class ProteinBias(_Scenario):
    def terms(self, mf):
        return dict(
            bias=[
                mf.terms.distance_function(self.table, self.pos, grid=(0.0, 20.0)),
                mf.terms.relative_position(self.relpos, self.res, max_offset=32),
            ],
            mask=mf.terms.distance_cutoff(self.pos, cutoff=15.0),
        )

    def benchmark_protein_bias(self):
        self._run()

    def verify_protein_bias(self):
        return self._reference()
