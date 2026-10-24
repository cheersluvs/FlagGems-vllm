<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 FlagOS Contributors -->
# BF16 query-major producer plus aligned bulk/tail stores

Pre-code boards,2026-10-24. e166f04d compiled and passed six259-row boundary
checks without spills; first actual timing was rejected by an unattributed GPU
PID. Its IR only transposed layout axes and PTX retained scalar global stores.
Thus changing mathematical dot orientation alone did not establish a better
physical writeback. Keep that private layout frozen and change only stores in
a separate kernel. Earlier BF16 vector-store-only regressions still apply as
negative evidence; a new codegen result is not a performance result.

|Project truth|Frozen decision|
|---|---|
|Editable|New private layout+store kernel,driver,runner,CPU tests and this board|
|Read-only|Production0d7e80e2/YAML/TopK, prior layout kernel/probe, captured78c3 fixtures/configs, timers/guards/tolerances|
|Change vs prior candidate|Same query-major dot/reduction; separate stores for full four-column groups and0–3-column tail. Expose per-row rounded limit divisibility; no output transpose buffer|
|Gate|Only original rectangular BF16/H4/D128/R32-or64/KT16 and contiguous FP32 logits with width multiple4 and base16-byte aligned. Other metadata/routes/buffers use original JIT/config/binary|
|Build/codegen|Actual boundary JIT and correctness first; export PTX/TTGIR/resources before rejection. Require active v4 global store sites and zero spills before any timing|
|Default set|n1/q8192/p102400 BF16 R32, n4/q4096/p102400 BF16 R64/grouped, unchanged FP8 long100k; all original tile/warp/stage/head configs|
|Validation|259-row whole/split schedules, padded Q/cache/table, query base97, zero-length request, varied visible values incl every modulo4 tail; extra misaligned-base and nonmultiple4-width fixtures must fall back to original binary|
|Benchmark|Kernel score excludingTopK is NV metric; TopK/pipeline and four A/A controls retained. Two alternating rounds,200ms warmup/100ms repeat,1800s worker|
|GPU guard|Exclusive idle admission,80C initial ceiling/120s cooldown, identified-self PID checks before/after timing. No bypass/kill/ownership guess|
|Autotune|Private fixed-choice exemption, no production tuner/config changes|
|Aggregation|Current production/candidate ratios, active/control means separately; no historical cross-directory full-binary match requirement|
|Acceptance|Compile-first bounded hypothesis probe only; no integration or full30 claim|

|Contract|Preserved|
|---|---|
|API/reference|Existing QSA prefill/native pinned vLLM0.31; no public torch equivalent|
|Math|Exact same private query-major full-D FP32 dots, ReLU per head, same full-head or two-pair head reduction|
|Store partition|limit=min(visible,width),full=floor(limit/4)*4; bulk columns<full and tail full<=column<limit are disjoint and complete|
|Alignment proof|Per-row base is16-byte aligned from base/width gate. Bulk endpoints are multiples4; no hint about arbitrary visible being divisible4 or constant across rows|
|Bounds|Same valid rows/chunk offsets/invalid-page -inf. No write after visible/width; scalar tail protects arbitrary visibility and page validity|
|Torch/autograd|Metadata-only gate and reference/test use in tools; no production compute fallback. Forward inference only|
|Limits|No new supported dtype/layout/API, approximation,split-K,stage/tile/head/page changes. Misaligned/FP8/compact stay original|

|Path|Treatment|
|---|---|
|Aligned BF16|Prior query-major producer plus vectorizable bulk/scalar-tail store within same scorer|
|Misaligned/nonmultiple4 width|Original production JIT/config; exact PTX/cubin equality against same-worker baseline|
|FP8/compact|Original production JIT/config/binary|
|Codegen rejection|Save resources and PTX/TTGIR, reject absent v4 or nonzero/missing spill info before timer construction|
|Next gate|Useful paired score gain beyond A/A and no material regression, then larger validation; v4 itself never justifies promotion|

Local validation:28 current-baseline CPU checks passed (new9 plus prior layout,
stage-depth and production-entry checks). Actual new kernel code runs in the
masked NumPy pointer model across eight offset/tail combinations, both head
modes,259 padded query rows,512 output columns,page196,negative/out-of-range
pages,all four visibility residues and output canaries. Every intended output
is written exactly once by the disjoint bulk/tail stores. Divisibility hints
are checked against every modeled value. AST proves no producer/reduction
change from the previous private layout; signature/request/key mapping still
matches production. Metadata alignment/fallback, PTX-site parsing ignoring
comments, evidence export before spill/v4 rejection, unchanged configs/binaries,
CLI child handoff and aggregation checked. Changed-file style,shell syntax and
CLI help pass. These CPU checks do not demonstrate vector code generation or
GPU performance: actual v4, correctness and paired timings remain pending.
Production/YAML and earlier candidates are untouched; no integration claimed.
