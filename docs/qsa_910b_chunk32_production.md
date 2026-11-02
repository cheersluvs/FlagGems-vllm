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

## Upstream 12-case kernel matrix

Use `--suite benchmark --mode kernel --upstream12`. The four(requests, queries
per request) layouts are(1,8192),(1,16384),(2,8192),(4,4096), crossed with
original-token prefixes8192/51200/102400. The common manifest is
`qsa_prefill_upstream_workloads.py`, citing vLLM PR54513. Page196 is rebuilt as
page192; BF16,H4,D128,compression4,groupK512 are explicit common-domain choices.

Production dispatch is unchanged: six cases tile128/cap32, one tile128/cap16,
five tile32/cap16. No forced fast path or smaller-query substitution is used.
The largest benchmark-only full-score staging is1856MiB, for isolated TopK.
Production logits remain chunked under the512MiB budget. CPU FP32 score/TopK
checks cover request and chunk-boundary representative rows; range and
uniqueness checks cover every output row. This is explicitly sampled numerical
validation, not exhaustive FP32 comparison of every query row.

Initial preparation may compile256..1024 chunk-offset specializations per case
because production OFFSET remains constexpr. This cost is outside measured
kernel time but can dominate a cold-cache run. Use one round, kernel warmup1,
active3 for the first complete matrix and allow an extended worker timeout.
Timer-source and 3-call normalization checks remain required. No old-policy
control is timed in this matrix; production stages and both native boundaries
are reported, including TopK. There is no standalone native score baseline.

## Audited image Triton plus Torch TopK baseline

Add `--image-triton-baseline` to the upstream12 kernel command. This replaces
the timed CANN baseline, not production. The source snapshot is byte-identical
to the2026-10-28 audited image file:

- image: quay.io/ascend/vllm-ascend:qwen3.8-next-a3
- image index: sha256:2cbb274af050cb5be7db7318e1c461fe22ec0643c54f1ed32aa7266513a7d421
- original path: /vllm-workspace/vllm-ascend/vllm_ascend/ops/triton/qwen4_exp/qsa.py
- snapshot SHA256:329ddd6d5f57f38cdbdebf0b7459cf6b69760016215827d6614fbb31d24990bb

The original copyright and source remain in the archived text file. Only
_qsa_mqa_paged_kernel, _validate_mqa and qsa_mqa_paged are loaded. Function ASTs
are unchanged and hashed; imports bind to the same installed Torch/FlagTree as
production, avoiding unrelated image-vLLM imports. No compiler/kernel patches.

Keep the image's default sqrt(D) divisor, full cache-capacity score width,
max128 rows /128MiB logits chunk policy, and torch.topk default sorted=True,
followed by int32 conversion and output copy. The only selector adaptation is
to stop before token expansion and write group indices to caller output slices,
rather than reuse the temporary block buffer consumed by expansion. Frozen
scores for isolated TopK are generated by the image scoring function itself.
Both baseline and production use the same actual paged BF16 inputs and causal
query positions. The extra full-score staging can bring combined benchmark
score buffers to about3.7GiB in the largest case; production allocations are
unchanged.

Four ratios are separate: image score/production score, image Torch TopK/
production persistent_topk, image prebuilt-metadata group selector/production
prepared pipeline, and common-ABI image adapter/production public entry. The
last adapter is benchmark-owned metadata reconstruction; it is not claimed to
be image source. Both selectors stop at compressed group indices, no token
expansion. Original per-path chunking and padding differences remain measured.

Correctness: initial sampled CPU FP32 scores for both paths, all-row output
range/uniqueness checks and sampled selected-score multiset comparisons;
post-timing score checks inspect fresh results outside timing. A source or
compile failure is reported, not silently replaced by another baseline.
The current script supports this baseline with upstream12 only.
