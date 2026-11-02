# 910B QSA restricted production routing

Status: implementation and local host checks complete; device validation pending.
The NVIDIA implementation and its draft PR are unchanged.

The production Triton kernel AST is unchanged. Only host dispatch changes:

| Policy | Gate | Otherwise |
| --- | --- | --- |
| Score tile128 | 910B, contiguous BF16 Q/K, H4/D128, page192, contiguous int32 table, width4096..25600 | tile32 |
| Chunk cap32 | Score tile128 gate, group TopK512, contiguous int32 request offsets, width4096..16384 | cap16 |

The existing logits budget still bounds the actual chunk, with a minimum of one
row even for a zero budget. Request-owner mapping, scoring arithmetic, output
contract, and TopK implementation are unchanged. The width intervals are a
restricted dispatch choice based on the sampled widths, not proof that every
width in those intervals has a measured performance benefit.

Use `run_qsa_lightning_910b.sh --suite benchmark --chunk32-production`.
It verifies the registered public entry and compares old tile32/cap16 policy
against actual production dispatch in the same process. The old control patches
only the two policy helpers; it does not change TopK or replace the public host.

The 16 cases include row counts31/32/33/63/64/65, ragged and empty requests,
zero/1MiB budgets, short/medium/long diagnostics, width4096/8192/16384 timings,
and H1/page256 fallback checks. Six cases have timing, ten only correctness.
CPU FP32 score checks and selected-score/range/uniqueness checks run outside
timing. TopK data, native adapter totals and actual launch routing are reported.
This is exploratory paired timing, not NVIDIA kernel-mode acceptance.

Local checks: original kernel AST fingerprint, 45 actual-host allocation/chunk
simulations, width/layout/dtype/device policy guards, formatting and syntax.
These checks do not establish NPU compilation, correctness or performance.

## Kernel-mode first pass

`--suite benchmark --chunk32-production --mode kernel` uses the same
`triton.backends.ascend.testing.do_bench_npu` API as the repository Ascend
benchmark base. Each stage is passed as one callable, including multi-kernel
pipelines. The installed timer source/hash and explicit warmup/active counts
are saved. A one-call/three-call scoring check must scale approximately3x;
otherwise results are rejected. The timer's millisecond result is converted to
microseconds. Values are device kernel duration sums per invocation, excluding
host gaps, not the previous synchronized wall/Event metrics.

First-pass cases: short16_g1024, chunk_width4096, medium32_g8192,
chunk_width16384, long32_g25600, chunk_ragged. Shared domain: BF16,H4,D128,
page192,groupK512,ratio4,contiguous Q/K/table. These are constructed diagnostic
inputs; they do not establish the deployed model's actual input distribution.
The ragged case has request lengths[17,0,16], with33 packed rows. Native
adapter/public and native fused/prepared ratios retain distinct boundaries;
there is no fabricated score-only native baseline. Use one round first, with
kernel warmup2 and active10. The CPU reference and index checks remain enabled
outside the timing call. The NVIDIA-only FP8 pytest benchmark is unchanged.
