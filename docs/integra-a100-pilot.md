---
kind: experiment-report
status: active
decision: defer
tags: [attention, sampled-backward, a100, training-step]
---

# Integra A100 full-model sampled-backward pilot

The optional sampled backward reduced measured optimizer-step time on PLINDER 6×4096
by 37.75% and PLINDER 1×24576 by 68.56%, with no peak-memory reduction. ChEMBL3D
384×64 retained every sampling group and was 0.67% slower. These are two timed steps per mode from a fresh model, not evidence
of faster convergence or improved time to a validation target. Exact split backward
remains the default; sampled backward is an explicit opt-in experiment.

## Scope and validity

The 2026-11-18 pilot used Integra source
`b57864ad1872678adbc037fde99be97e7be144d0` and GitHub-installed molflash
`1c4ee7be04c839f4d1701222737a0344b7c43b0f`. The model had 235,651,442 parameters:
202,004,176 in the Trunk, 33,302,112 in the Denoiser and 345,154 in the distogram head.
All 32 Trunk blocks were active: eight biased-attention blocks and 24 linear-attention
blocks. The Denoiser had four biased-attention blocks. Model width was 512 and FFN hidden width 3072, with four heads
of dimension 128, four Denoising Copies (K=4), one recycle and accumulation A=8.

Each case ran on one NVIDIA A100-SXM4-40GB, BF16 mixed precision, PyTorch 2.14.0+cu130,
Triton 3.8.0 and Lightning 2.6.6, with FSDP2 world size one. This does not demonstrate
parameter sharding, multi-GPU memory savings or distributed scaling. Trunk biased blocks
had activation checkpointing off, linear blocks used full checkpointing, and Denoiser
checkpointing covered the FFN.

A fresh model received three common warm optimizer steps. Exact and sampled branches
restored the same common checkpoint and RNG state, each ran one untimed branch-warm
optimizer step, then two timed optimizer steps and a separate gradient-energy census.
Each timed optimizer step accumulated eight microbatches. The backward sums exclude
branch-warm and census work. This is not a trained-checkpoint benchmark.

The original YAML AdaGC warmup of 100 steps was preserved. Source b57864a fixes the
zero-history AdaGC bootstrap while preserving positive-history arithmetic and warmup
semantics. GPU gate `1944803`
passed 20/20 tests with no failures or skips. Proof job `1944804` observed 32/32 positive
Denoiser-gradient calls with that warmup. Each exact/sampled case repeated the positive
32/32 guard before timing; the guard and census consumed no sampling RNG. Earlier
job `1944750` is excluded as invalid timing evidence: the zero-history AdaGC trap
froze Denoiser gradients, and that pilot independently overrode the recipe's warmup
from 100 to 1. The bootstrap fix and the discarded recipe override are distinct changes.

Sampled mode used `BackwardSampling(budget=0.1, block_size=128)`. The target retained
count comes from counting the smallest squared block scores whose cumulative energy is
at most `budget * total_energy`; groups are then sampled probabilistically. Budget 0.1
does not mean keeping 10% of groups or guaranteeing 90% retained energy. For fixed inputs
and weights its forward is unchanged; its stochastic backward changes the training-gradient
estimator, and branches can diverge after updates. Exact mode used `None`.

## Measured optimizer steps

Both raw timed observations are shown, in seconds. Peak allocated memory is the maximum
across the timed microbatches and is identical between modes in these cases.

| case / job | exact step seconds | sampled step seconds | exact backward seconds | sampled backward seconds | peak allocated, both modes |
|---|---|---|---|---|---|
| ChEMBL3D 384×64 / 1944807 | 12.468757, 12.299102 | 12.585794, 12.347271 | 5.334517, 5.332642 | 5.386151, 5.387632 | 28.403515 GiB |
| PLINDER 6×4096 / 1944808 | 27.764101, 27.723339 | 17.236730, 17.303757 | 18.984206, 18.978176 | 8.493289, 8.456610 | 32.042175 GiB |
| PLINDER 1×24576 / 1944809 | 117.900454, 117.883398 | 36.951479, 37.179954 | 89.365255, 89.339819 | 8.789405, 9.037011 | 28.877878 GiB |

The ratio of mean sampled/exact step time was 1.006670 for ChEMBL3D, 0.622492 for
PLINDER 6×4096 and 0.314404 for PLINDER 1×24576. The corresponding backward-time
ratios were 1.009996, 0.446492 and 0.099753. No confidence interval is reported and no
training-quality conclusion is drawn from two observations.

The 24,576-token case contained 24,574 unmasked tokens and 24,564 atoms from System
`6kqh__1__1.C_1.D_1.F__1.I_1.N`. It was not a padding-only case. The recorded Trunk
output was `[1, 24576, 512]`; the Denoiser inputs were `[4, 24576, 512]` and
`[4, 24576, 3]`.

## Actual retained groups and energy

The census sums actual retained groups and their unweighted squared-gradient energy,
then divides by the corresponding totals within each stack's biased-attention calls.
It is separate from timing; the 24 linear Trunk blocks are not sampled.
ChEMBL3D and PLINDER 6×4096 had no zero-energy groups. PLINDER 1×24576 had 896
zero-energy Denoiser groups out of 98,304 (0.91146%) in both modes, no zero-energy
Trunk groups, and no zero-gradient head. All three cases passed the positive-call guard;
whole-call gradient freezing does not explain the low retained-group fractions.

| case | Trunk groups kept | Trunk energy retained | Denoiser groups kept | Denoiser energy retained |
|---|---:|---:|---:|---:|
| ChEMBL3D 384×64 | 100% | 100% | 100% | 100% |
| PLINDER 6×4096 | 34.42% | 88.06% | 10.13% | 64.76% |
| PLINDER 1×24576 | 0.9603% | 36.56% | 1.3814% | 41.69% |

A 64-atom ChEMBL3D row has one group at block size 128, so this configuration cannot
skip a smaller within-row group. Its measured overhead supplied no training-step benefit.
PLINDER 6×4096 had 32 groups per row and skipped actual work in both stacks. This
fresh model's nonzero Denoiser gradients were very small; its energy distribution must
not be assumed to match a mature checkpoint.

## Evidence and decision

The producing Integra freeze is
`slurm/cleanup/molflash-v2-validation/lazy-pilot-b57864a-original-adagc-a100/`.
`pilot.json` records resolved policy and source hashes; `lazy_pilot_case.py` runs each
case, `summarize_lazy_pilot.py` selects the timed observations, and each
`results/<job>/summary.json` retains both raw observations. Exact/sampled provenance
files retain GitHub `direct_url`, GPU UUID, common-checkpoint SHA256 and RNG SHA256;
branch-guard files retain the positive-gradient check. Aggregate job `1944810` completed
with status `passed`; its `aggregate/summary.json` links all three case results. No local
or editable molflash package was used.

| case job | node | GPU UUID |
|---|---|---|
| 1944807 | dgx-1 | 6c061105-0292-4e94-75fa-992ea861e9ed |
| 1944808 | dgx-1 | 605b42ff-f57e-3f51-756d-2dfa54a5f22d |
| 1944809 | dgx-3 | 780e004c-bedd-c8b3-667d-4d382837bc14 |

The recommendation is to evaluate validation quality and longer checkpoint-based runs
before changing a default. All eleven evaluated exact A100 execution candidates were
slower on their preregistered weighted PLINDER-24576 predictor (jobs 1944720, 1944735
and 1944748). Those standalone results do not establish a sampled-backward convergence
benefit. Exact split backward remains the production baseline.
No reduction in peak memory, FSDP2 sharding gain, multi-GPU scaling gain or improvement
in time to a quality target is claimed by this pilot.
