<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 FlagOS Contributors -->
# BF16 larger query tile: bounded K-reuse probe

Pre-code boards, 2026-10-24. Production remains 0d7e80e2. The 227ce1c1
query-half replacement lost against head groups (active-case score mean0.9600).
Do not revive page-table dedup or substitute query halves for head pairing.
Historical hardware captures in qsa_prefill_profiling_workflow.md show tensor
utilization/issue/occupancy tradeoffs; they do not establish a current-production
DRAM bottleneck. New hypothesis: increasing rows per CTA amortizes the same
key-tile loads over more query rows. Measure whether that reuse outweighs extra
live queries/accumulators, fewer CTAs, padding and resource pressure.

|Project truth|Frozen decision|
|---|---|
|Editable|Only new private kernel/driver/runner/CPU tests and this document|
|Read-only|Production, YAML, TopK, captured78c3ca4f six-case manifest, timer, GPU guards and reference tolerances|
|Baseline|Current head-group production, captured tiles, stage2 and original warps|
|Candidate|BF16/H4/D128 rectangular R32→64 or R64→128; preserve HEAD_GROUP, BLOCK_N, K_TILES, warps and stages. Grid recomputed using candidate TILE_R through existing host dispatch|
|Default active set|n1/q8192/p102400, n4/q4096/p102400 and unchanged FP8 long100k. Six-case manifest remains available by explicit --cases|
|Build|GPU JIT only; private kernel is exact rectangular production body except grouped-head eligibility admits R128 and a BF16/H4/D128/R64-or128 assertion|
|Validation|259-row all-row boundary/canary checks, whole and split chunk schedules, real-case sampled score/selection before/after timings, exact fallback PTX/cubin|
|Benchmark|Kernel score/TopK/pipeline, production/candidate ratio, microseconds, two alternating rounds, four A/A controls; active/control means separately|
|Resources|Export PTX/TTGIR, registers/shared/spills. Compilation/runtime failure stops run and uploads diagnostics; spill/slowdown blocks promotion. No automatic broad sweep|
|Autotune|Private fixed-choice exemption: one tile doubling per captured config; no production tuning edits|
|Timeout/thermal|1800s per worker,200ms warmup/100ms repetition,80C initial ceiling,120s cooldown|
|Profiler|Prior archived counters motivate reuse/pressure tradeoff; no new NCU required until positive timing signal. Resource counts alone do not prove causality|

|Contract|Preserved|
|---|---|
|API/reference|Existing qsa_select_paged_prefill contract/native pinned vLLM0.31 reference; no corresponding torch public API|
|Input/output|Same Q/cache/table strides, dtype, page mapping, query offsets/lengths, FP32 score output and masking|
|Math|Full-D FP32 dot, per-head ReLU then same head reduction. No approximation, split-K, page lookup change or policy change|
|Grouping|Existing head pairing retained when baseline HEAD_GROUP=true; false stays false. This is not query-half decomposition|
|Unsupported|Other metadata/routes use exact original JIT/config/binary; private entry rejects unsupported inputs|
|Torch/autograd|Reference use only; inference forward; production source unchanged, no torch compute fallback|
|Limitations|Large cases sampled; three-case probe is not full30 regression or native-vLLM acceptance|

|Path|Treatment|
|---|---|
|BF16 rectangular R32|Private full-head R64, original warps/stages/key tiling|
|BF16 rectangular R64/grouped|Private grouped R128, original warps/stages/key tiling|
|FP8/compact|Original JIT/config, binary identity proof|
|Boundary|259 rows, page196, padded strides, nonzero query base, zero-length request and variable visible lengths; schedules [259] and [127,128,4] check query-tile tails and request/chunk boundaries|
|Promotion gate|Useful score benefit beyond A/A, no material active regression/spills, then wider correctness/performance. No production integration in this round|

Local validation:26 current-baseline CPU checks passed (new6 plus query-retest,
stage-depth and head-group production checks). The actual private kernel body
runs in a NumPy masked-pointer model for eight R64/full-head and R128/grouped
layout/offset combinations, including full128-row tiles and tails. The complete
body is AST-identical to production after normalizing only its grouped-tile
guard and removing two private-domain assertions. Tests also exercise actual
FixedEntry grid recomputation, preservation of all non-tile configs/head modes,
original-binary rejection, default three-case child handoff and active/control
aggregation. Shell syntax and CLI help pass. GPU results remain pending.

Historical head-group regression tests pin the pre-integration production hash
and reject current production (six errors in a broader34-test invocation).
They are intentionally not repinned; current-baseline26 checks above pass.
