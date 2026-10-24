<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 FlagOS Contributors -->
# BF16 dot, head reduction and store layout probe

Pre-code boards,2026-10-24. Current production profiling64f15a08 has ten actual
scorers with matched independent preparation/capture binaries. Both BF16 paths
have approximately4 L1 store-sector bytes/useful FP32 output byte, while FP8
is approximately1 after integrated vector stores. BF16 grouped R64 has207
registers,104KiB shared,eligible0.40–0.41,barrier16.93–17.71% and shared-load
conflicts/wavefront0.136–0.145. These counters motivate a dataflow hypothesis;
they do not prove a specific stalled source instruction or predict speedup.
Prior BF16 store-only vectorization regressed, so do not simply repeat it.

|Project truth|Frozen decision|
|---|---|
|Editable|Only new private layout kernel,driver,runner,CPU checks and this board|
|Read-only|Production0d7e80e2/YAML/TopK/native baseline, captured78c3 six-case inputs/configs, timers/guards/tolerances|
|Hypothesis|Produce dot results as [query×head,key], reduce to [query,key], store in that orientation; reduce output-layout conversion/transactions without increasing tile size|
|Baseline|Current canonical production JIT, actual captured configs, same compiler/toolchain and score buffers|
|Active probe|Default3 cases: n1/q8192/p102400 BF16 R32/W8, n4/q4096/p102400 BF16 R64/grouped/W4, unchanged FP8 long100k. Two alternating rounds|
|Config scope|All TILE_R/BLOCK_N/K_TILES/warps/stages/HEAD_GROUP unchanged; production route unchanged. No search|
|Build/resource gate|Compile/execute small boundary fixtures first; export actual registers/shared/spills/PTX/TTGIR before resource rejection. Any active candidate spill or missing resource info stops before timing. No resource improvement is assumed|
|Validation|259-row all-row boundary/canary fixtures, whole/split cross-request chunks, padded strides/nonzero query base/zero-length request; real cases sampled before/after score and selection timings|
|Benchmark|Kernel mode score excludingTopK for NV metric, TopK and sequential pipeline still reported, same timers and four A/A controls, active/control means separately|
|Timeout/thermal|1800s per worker,200ms warmup/100ms repeat,80C initial ceiling/120s cooldown|
|Profiler|No broad new NCU in first probe; compare compiler artifacts/resources and timings. Only positive stable timing justifies focused counters/full regression|
|Autotune|Private fixed-choice exemption; original captured configurations, no production tuning changes|
|Acceptance|Three-case hypothesis test only; no production integration or full30/native acceptance claim|

|Contract|Preserved|
|---|---|
|API/reference|Existing qsa_select_paged_prefill forward contract/native pinned vLLM0.31; no public torch equivalent|
|Input/output|Same BF16/H4/D128 Q/cache/table strides, page lookup, per-request lengths, chunk offsets, visible lengths, FP32 logits, invalid-key -inf and untouched masked columns|
|Math|Full-D FP32 dots, ReLU for each head then head sum, same two head pairs where HEAD_GROUP=true. Only mathematical matrix orientation changes|
|Allocation|No extra global output transpose or staging tensor. Query orientation and key transpose remain Triton tensors within one scorer|
|Unsupported candidate|FP8, compact, non-H4/D128 and other tiles use original JIT/config/binary. No torch fallback|
|Limits|Inference-only; no new dtype/layout support, approximation, split-K, page dedup, eviction policy, stage1 or larger tile|

|Path|Treatment|
|---|---|
|BF16 R32/full-head|dot[query×4,key], reshape[query,4,key], per-headReLU and sum axis1, store[query,key]|
|BF16 R64/grouped|Two full-D dots[query×2,key], per-pairReLU/sum axis1 then pair addition, store[query,key]|
|FP8/compact|Exact original JIT with PTX/cubin equality proof against same-worker baseline|
|Resource rejection|Save per-invocation resource/artifact evidence and fail probe before timer creation; keep production unchanged|
|Next gate|Useful score benefit beyond A/A with no material regression/spill, then expand correctness/performance. Transaction improvements alone do not justify promotion|

Local validation:25 current-baseline CPU checks pass (new7 plus tile/stage and
production-entry checks). Actual private kernel code executes in a NumPy
masked-pointer model for R32/full-head and R64/grouped across eight offset/tail
combinations,259-row padded queries,512 output columns,page196 and valid,
negative and out-of-range physical pages. Full-D per-head math, -inf output,
query/key addressing and canaries checked. Signature/request mapping and key
load prefix match production AST; configs/grid/head flags/fallback binaries,
resource rejection after evidence export, default child handoff and group means
checked. Style, shell syntax and CLI help pass. NumPy checks do not validate
FlagTree code generation; GPU compilation/correctness/performance remain pending.
Production/YAML are unchanged; no speedup or integration claimed.
