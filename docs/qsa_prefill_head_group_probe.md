<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 FlagOS Contributors -->
# Production head-group probe

Pre-code boards,2026-10-23. Read workflow.md and prior profiling/store/query
probe evidence; optimization.md/deep_opt.md are absent. Native-comparison report
e1c188ecf888d876674023447d99ee8249df8dea passed45 representative checks and30
measurements. Six pipeline cases remain below0.98 despite score>=1.0499.
Their score fractions are42–57%. Recorded n4 BF16 uses232 registers/thread and
106496 shared bytes; FP8 long uses158 registers and57344 shared bytes, no spills.
Older NCU memory evidence demonstrates costly output transactions and scoreboard
stalls, but is not a dynamic-counter measurement of this integrated production.

|Project truth|Frozen choice|
|---|---|
|Current op/reference|Existing QSA prefill; production SHA8e4ad799, native vLLM7a877ae|
|Editable|New private head-group kernel, fixed-choice probe/manifest/tests/runner and this document|
|Read-only|Production kernels/configs/exports, frozen30 workload table, native source, timer/guard/tolerances|
|Build|Triton JIT on H800; Mac AST/CPU orchestration/lint checks|
|Validation|Exact captured inputs/compiler/configs; record baseline binaries and freeze within-run identities, small all-row canaries/visible/chunk tests; real-case score/TopK/pipeline pre/post sampled checks|
|Benchmark|Six selected cases; kernel mode;2 alternating rounds,200ms warmup/100ms rep; score,unchanged TopK,pipeline plus four A/A controls|
|Active set|Five BF16 long-prefix cases and FP8 long100k; H4,D128,original512MiB chunk geometry|
|Aggregation|Per-case incremental production/candidate latency and arithmetic means; not native acceptance|
|Timeout/profiler|1800s per worker;80C/120s cooldown; export actual PTX/TTGIR and registers/shared/spills, no repeated NCU by default|
|Autotune|Exempt private one-hypothesis fixed-choice probe; reuse exact production choices from e1c188, no retuning or candidate expansion|
|Hypothesis|Two full-row head pairs reduce live dot output footprint; less register pressure may help scheduling. Existing FP8 query subdivision is replaced by head subdivision for this isolated alternative|
|Failure criterion|Reject if benefit fits A/A noise, introduces spills/resource failure or numerical/selection mismatch|

|Contract|Preserved|
|---|---|
|API/dtypes|Existing public BF16/FP8 QSA inference contract; no torch API equivalent|
|Input/layout|Actual H4,D128 paged input tensors, strides, visible counts, offsets and budget unchanged|
|Math|Full-D FP32 dot per head; ReLU before head sum; no split-D accumulation or scaling. Pairwise head aggregation may change FP32 summation order; original tolerances and TopK membership checks stay enforced|
|Output|FP32 logits, exact visible masks and invalid-page -inf, unchanged int32 selection and unspecified ties|
|Unsupported/autograd|Private kernel static-rejects H!=4; no production dispatch changes; inference only|
|Torch|Only existing test/reference/fixture checks; production untouched|

|Path|Implementation/validation|Risk|
|---|---|---|
|BF16 rectangular|Two head-pair dot/reduce operations reuse one key tile, original scalar stores; keep R/N/K/warps/stages|Extra dot/reduction scheduling can increase registers or traffic|
|FP8 rectangular|Two head pairs across all query rows, original FP8 bulk/tail stores and alignment gate|Different subdivision may lose the previous query-mode benefit|
|Small boundaries|Padded strides,nonzero query base,zero/residue/page-tail visibility,cross-request chunks and canaries; both BF16 warp/tile choices plus FP8|FP32 reassociation/incorrect query-head mapping|
|Compact/other heads|Not part of probe; original production read-only|No generalization claimed|
|TopK|Exact same persistent_topk/common buffers and callable for both probe sides|Its native slowdown is not changed by this probe|
|Integration|Only after benefits exceed paired A/A controls and regression30|Six-case probe does not establish production acceptance|

The manifest is pinned to e1c188 and six exact fixtures/inputs/configs. Frozen
FP8 `QUERY_PARTS=2` is retained as a baseline configuration tag; this candidate
replaces row subdivision with head pairing, so the argument is intentionally
unused. Records name the algorithm explicitly. No benefit is inferred by
multiplying this incremental probe by earlier native speedups.

Local validation:35 related CPU checks passed, including execution of the
actual query/head-index and ReLU/reduction AST on padded NumPy buffers over
R4/32/64 and D8/24/128, unchanged key/request/BF16 store AST contracts, independent
fixed entries for all three production routes, all stage timers and four A/A
controls, and stopping on injected numerical failure. Python compilation,
CLI help, shell syntax and pre-commit passed. These checks do not compile Triton
on H800. The new candidate's compilation, numerical behavior, registers and
performance remain unverified until the returned GPU report.

The runner reuses the SHA-verified assembler from the completed production
report and the existing isolated compatibility-library bootstrap; no new tool
installation is required. Only three small17-row boundary cases are checked
once, followed by the six selected real fixtures. Per-case resources and all
stage/A/A tables print to the terminal and are saved/uploaded with source,
PTX/TTGIR, input/compiler identities and pre/post guard results. A failed upload
prints a push-only retry command; it does not require repeating the measurements.
